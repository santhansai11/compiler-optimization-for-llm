"""Code generation of fused GPU kernels (Apple Metal).

For fused IR nodes whose PyTorch lowering needs several kernels, the compiler
GENERATES one Metal kernel (source text specialised to the row length, epsilon
and which optional inputs exist), compiles it with ``torch.mps.compile_shader``
and caches it per specialisation:

  FusedAddLayerNorm  -> 1 kernel (sum of inputs + layer norm)   [torch: add + layer_norm]
  FusedAddRMSNorm    -> 1 kernel (sum of inputs + RMS norm)     [torch: add + rms_norm]
  FusedScaleSoftmax  -> 1 kernel (scale + softmax)              [torch: mul + softmax]

One threadgroup handles one row; reductions are done in threadgroup memory.
Only available when ``torch.mps.compile_shader`` exists (Apple GPUs); on other
devices the executor falls back to PyTorch primitives. CUDA code generation
(Triton / ``torch.compile``) is not implemented here.
"""

import torch

GROUP = 256
_CACHE = {}

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def available(device):
    return device.type == "mps" and hasattr(torch.mps, "compile_shader")


def _reduce_sum(var):
    """Threadgroup tree reduction of per-thread partial ``var`` into sh[0]."""
    return f"""
    sh[t] = {var};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = {GROUP // 2}; s > 0; s >>= 1) {{
        if (t < s) sh[t] += sh[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
"""


def _add_norm_source(name, d, eps, n_inputs, has_w, has_b, rms):
    """Row kernel: out = norm(sum(inputs)) [* w] [+ b]."""
    params = ", ".join(f"device const float* x{i} [[buffer({i})]]" for i in range(n_inputs))
    nxt = n_inputs
    if has_w:
        params += f", device const float* w [[buffer({nxt})]]"
        nxt += 1
    if has_b:
        params += f", device const float* bias [[buffer({nxt})]]"
        nxt += 1
    params += f", device float* out [[buffer({nxt})]]"
    total = " + ".join(f"x{i}[base + i]" for i in range(n_inputs))
    mean_block = "" if rms else f"""
    float s = 0.0f;
    for (uint i = t; i < {d}; i += {GROUP}) s += ({total});
    {_reduce_sum('s')}
    float mean = sh[0] / {d}.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    center = "0.0f" if rms else "mean"
    var_block = f"""
    float v = 0.0f;
    for (uint i = t; i < {d}; i += {GROUP}) {{ float z = ({total}) - {center}; v += z * z; }}
    {_reduce_sum('v')}
    float inv = rsqrt(sh[0] / {d}.0f + {eps}f);
"""
    w_expr = " * w[i]" if has_w else ""
    b_expr = " + bias[i]" if has_b else ""
    return _HEADER + f"""
kernel void {name}({params}, uint row [[threadgroup_position_in_grid]],
                   uint t [[thread_position_in_threadgroup]]) {{
    threadgroup float sh[{GROUP}];
    uint base = row * {d};
{mean_block}{var_block}
    for (uint i = t; i < {d}; i += {GROUP}) {{
        float z = (({total}) - {center}) * inv;
        out[base + i] = z{w_expr}{b_expr};
    }}
}}
"""


def _scale_softmax_source(name, d, scale):
    return _HEADER + f"""
kernel void {name}(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                   uint row [[threadgroup_position_in_grid]],
                   uint t [[thread_position_in_threadgroup]]) {{
    threadgroup float sh[{GROUP}];
    uint base = row * {d};
    float m = -INFINITY;
    for (uint i = t; i < {d}; i += {GROUP}) m = max(m, x[base + i] * {scale!r}f);
    sh[t] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = {GROUP // 2}; s > 0; s >>= 1) {{
        if (t < s) sh[t] = max(sh[t], sh[t + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float mx = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = t; i < {d}; i += {GROUP}) sum += exp(x[base + i] * {scale!r}f - mx);
    {_reduce_sum('sum')}
    float denom = sh[0];
    for (uint i = t; i < {d}; i += {GROUP})
        out[base + i] = exp(x[base + i] * {scale!r}f - mx) / denom;
}}
"""


def _kernel(key, source, name):
    if key not in _CACHE:
        _CACHE[key] = getattr(torch.mps.compile_shader(source), name)
    return _CACHE[key]


def fused_add_norm(inputs, weight, bias, eps, rms):
    """sum(inputs) -> (RMS|Layer)Norm in ONE generated kernel."""
    x0 = inputs[0].contiguous()
    d = x0.shape[-1]
    rows = x0.numel() // d
    ins = [t.contiguous().expand_as(x0).contiguous() if t.shape != x0.shape else t.contiguous()
           for t in inputs]
    out = torch.empty_like(x0)
    name = f"add_{'rms' if rms else 'ln'}_norm"
    key = (name, d, round(float(eps), 12), len(ins), weight is not None, bias is not None)
    kernel = _kernel(key, _add_norm_source(name, d, float(eps), len(ins),
                                           weight is not None, bias is not None, rms), name)
    args = list(ins)
    if weight is not None:
        args.append(weight.contiguous())
    if bias is not None:
        args.append(bias.contiguous())
    kernel(*args, out, threads=(rows * GROUP, 1, 1), group_size=(GROUP, 1, 1))
    return out


def fused_scale_softmax(x, scale):
    x = x.contiguous()
    d = x.shape[-1]
    rows = x.numel() // d
    out = torch.empty_like(x)
    key = ("scale_softmax", d, float(scale))
    kernel = _kernel(key, _scale_softmax_source("scale_softmax", d, float(scale)), "scale_softmax")
    kernel(x, out, threads=(rows * GROUP, 1, 1), group_size=(GROUP, 1, 1))
    return out
