"""PyTorch (GPU) executor for the computation-graph IR.

Runs the same graphs as ``utils.executor`` (identical derived weights, same
operand-order rules) on a real device: CUDA, Apple MPS or CPU. This is what
turns the *modeled* GPU numbers into *measured* ones.

How fused IR nodes are realised (honest mapping):
  * FusedAttention        -> ``F.scaled_dot_product_attention`` (a real fused
                             attention kernel; no materialised score matrix)
  * GEMM / Linear{GELU,Relu,SiLU}
                          -> ``F.linear`` (bias folded into the GEMM) then
                             the activation: saves the separate bias-add pass
  * FusedAddLayerNorm/RMSNorm, FusedScaleSoftmax
                          -> PyTorch primitives (same kernel count as the
                             unfused pair; the graph is smaller but the GPU
                             work is not)
So only some fusions reduce real GPU work; the benchmark reports what it
actually measures.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.executor import DEFAULT_CONFIG, _weight

_TENSOR_CACHE = {}


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        import platform
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            chip = platform.processor()
        return f"{chip} GPU (Metal/MPS)"
    return "CPU"


def _t(array, device):
    """Cached host->device tensor (weights are constants of the model)."""
    key = (id(array), str(device))
    hit = _TENSOR_CACHE.get(key)
    if hit is None:
        hit = (torch.from_numpy(np.ascontiguousarray(
            np.asarray(array), dtype=np.float32)).to(device),
            array)  # keep the key object alive so id() stays valid
        _TENSOR_CACHE[key] = hit
    return hit[0]


def _derived_weight(name, dim, device):
    return _t(_weight(name + "::w", dim), device)


_ATTN_CHOICE = {}


def _time_fn(fn, device, reps=15):
    for _ in range(5):
        fn()
    best = float("inf")
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best


class _Runner:
    """``attention``: how FusedAttention is lowered -
    'sdpa' (always the fused kernel), 'composed' (matmul/softmax/matmul) or
    'auto' (benchmark both once per shape and keep the faster: kernel
    autotuning, as real compilers do)."""

    def __init__(self, graph, device, feed, attention="sdpa"):
        self.attention = attention
        self.device = device
        self.feed = torch.from_numpy(feed).to(device)
        order = list(graph.topological_order())
        nodes = graph.graph.nodes
        self.plan = [(n, nodes[n], list(graph.predecessors(n))) for n in order]
        self.uses0 = {n: graph.graph.out_degree(n) for n in order}
        self.keep = {n for n in order if nodes[n].get("is_output")}
        self.sinks = [n for n in order if graph.graph.out_degree(n) == 0]
        self.graph = graph
        self.launches = 0

    # -- one pass over the graph -----------------------------------------
    def run(self, track=False):
        cache, uses = {}, dict(self.uses0)
        alive = peak = 0
        self.launches = 0
        with torch.no_grad():
            for node, data, preds in self.plan:
                out = self._compute(node, data, [cache[p] for p in preds])
                cache[node] = out
                if track:
                    alive += out.numel() * out.element_size()
                    peak = max(peak, alive)
                for p in preds:
                    uses[p] -= 1
                    if uses[p] == 0 and p not in self.keep:
                        if track:
                            alive -= cache[p].numel() * cache[p].element_size()
                        del cache[p]
        return {n: cache[n] for n in self.sinks if n in cache}, peak

    # -- operators --------------------------------------------------------
    def _matmul(self, ins, base, weight, transpose_rhs):
        dev = self.device
        if len(ins) == 1:
            x = ins[0]
            W = (_t(weight, dev) if weight is not None
                 else _derived_weight(base, x.shape[-1], dev))
            self.launches += 1
            return x @ W
        a, b = ins[0], ins[1]
        self.launches += 1
        if transpose_rhs:
            return a @ b.transpose(-1, -2)
        if a.ndim >= 2 and b.ndim >= 2:
            if a.shape[-1] == b.shape[-2]:
                return a @ b
            if b.shape[-1] == a.shape[-2]:
                return b @ a
            if a.ndim == 3 and b.ndim == 3 and a.shape[-1] == b.shape[-1]:
                return a @ b.transpose(1, 2)
        out = a
        for v in ins[1:]:
            out = out @ v
        return out

    def _linear(self, data, ins, base):
        """x @ W (+ bias) as ONE GEMM kernel (bias folded in)."""
        dev = self.device
        x = ins[0]
        weight = data.get("weight")
        W = (_t(weight, dev) if weight is not None
             else _derived_weight(base, x.shape[-1], dev))
        bias = data.get("bias")
        self.launches += 1
        return F.linear(x, W.t(), None if bias is None else _t(bias, dev))

    def _compute(self, name, data, ins):
        dev = self.device
        op = data.get("op_type", "Operation")
        base = (data.get("fused_from") or [name])[0]
        L = 1  # default: one kernel
        if op == "Input":
            return self.feed
        if op == "Constant":
            value = data.get("value")
            return (torch.tensor(1.0, device=dev) if value is None
                    else _t(value, dev))
        if op == "Identity":
            return ins[0]
        if op == "Scale":
            self.launches += L
            return ins[0] * float(data.get("factor", 1.0))
        if op in ("Add", "Mul"):
            out = ins[0]
            for v in ins[1:]:
                out = out + v if op == "Add" else out * v
                self.launches += L
            return out
        if op == "Sub":
            self.launches += L
            return ins[0] - ins[1]
        if op == "Relu":
            self.launches += L
            return torch.relu(ins[0])
        if op == "GELU":
            self.launches += L
            return F.gelu(ins[0], approximate="tanh")
        if op == "SiLU":
            self.launches += L
            return F.silu(ins[0])
        if op == "Softmax":
            self.launches += L
            return torch.softmax(ins[0], dim=-1)
        if op == "LayerNorm":
            return self._layer_norm(ins[0], data)
        if op == "RMSNorm":
            return self._rms_norm(ins[0], data)
        if op == "MatMul":
            return self._matmul(ins, base, data.get("weight"),
                                bool(data.get("transpose_rhs", False)))
        if op == "GEMM":
            return self._linear(data, ins, base)
        if op in ("LinearGELU", "LinearRelu", "LinearSiLU"):
            y = self._linear(data, ins, base)
            self.launches += L
            return {"LinearGELU": lambda v: F.gelu(v, approximate="tanh"),
                    "LinearRelu": torch.relu, "LinearSiLU": F.silu}[op](y)
        if op in ("FusedAddLayerNorm", "FusedAddRMSNorm"):
            total = ins[0]
            for v in ins[1:]:
                total = total + v
                self.launches += L
            return (self._layer_norm if op == "FusedAddLayerNorm"
                    else self._rms_norm)(total, data)
        if op == "FusedScaleSoftmax":
            self.launches += 2 * L
            return torch.softmax(ins[0] * float(data.get("factor", 1.0)), -1)
        if op == "FusedAttention":
            return self._fused_attention(data, ins)
        raise NotImplementedError(f"torch executor cannot run op {op!r}")

    def _layer_norm(self, x, data):
        dev = self.device
        w, b = data.get("weight"), data.get("bias")
        self.launches += 1
        return F.layer_norm(x, (x.shape[-1],),
                            None if w is None else _t(w, dev),
                            None if b is None else _t(b, dev),
                            float(data.get("eps", 1e-5)))

    def _rms_norm(self, x, data):
        w = data.get("weight")
        self.launches += 4  # square, mean, rsqrt, scale (no fused primitive)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                              + float(data.get("eps", 1e-6)))
        return out if w is None else out * _t(w, self.device)

    def _fused_attention(self, data, ins):
        if data.get("q_transform") is not None or data.get("mask_kind"):
            raise NotImplementedError("real-model attention not supported on GPU path")
        x = ins[0]
        names = data.get("fused_from") or ["Q", "K", "V"]
        dim = x.shape[-1]
        q, k, v = (x @ _derived_weight(names[i], dim, self.device) for i in range(3))
        factor = data.get("factor")
        factor = float(factor) if factor is not None else 1.0 / float(np.sqrt(dim))

        def sdpa():
            return F.scaled_dot_product_attention(q, k, v, scale=factor)

        def composed():
            return torch.softmax((q @ k.transpose(-1, -2)) * factor, -1) @ v

        mode = self.attention
        if mode == "auto":
            key = (tuple(q.shape), str(self.device))
            if key not in _ATTN_CHOICE:
                _ATTN_CHOICE[key] = ("sdpa" if _time_fn(sdpa, self.device)
                                     <= _time_fn(composed, self.device) else "composed")
            mode = _ATTN_CHOICE[key]
        if mode == "sdpa":
            self.launches += 3 + 1  # three projections + ONE fused attention kernel
            return sdpa()
        self.launches += 3 + 4  # projections + matmul, scale, softmax, matmul
        return composed()


def _feed(graph):
    config = dict(DEFAULT_CONFIG)
    config.update(graph.meta.get("config") or {})
    shape = [int(config["batch"])] + list(
        config.get("input_shape") or [int(config["seq"]), int(config["d_model"])])
    return np.random.default_rng(0).standard_normal(shape).astype(np.float32)


def _main(graph, outputs):
    for node in graph.graph.nodes():
        if graph.graph.nodes[node].get("is_output") and node in outputs:
            return outputs[node]
    return outputs[sorted(outputs)[0]]


def gpu_compare(specs, device, repeats=50, warmup=10):
    """Interleaved, synchronised timing of several graphs on ``device``.

    ``specs`` is a list of ``(graph, attention_lowering)``; the first entry
    is treated as the reference. Returns one dict per spec with latency
    (min / median seconds), live tensor peak bytes, dispatched-kernel
    count, the output (numpy) and the CUDA allocator peak when available.
    """
    feed = _feed(specs[0][0])
    runners = [_Runner(g, device, feed, attention=mode) for g, mode in specs]
    for _ in range(warmup):
        for r in runners:
            r.run()
    sync(device)
    n = len(runners)
    times = [[] for _ in runners]
    for i in range(repeats):
        order = list(range(n))
        if i % 2:
            order.reverse()
        for idx in order:
            sync(device)
            t0 = time.perf_counter()
            runners[idx].run()
            sync(device)
            times[idx].append(time.perf_counter() - t0)
    result = []
    for idx, r in enumerate(runners):
        base = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        outs, peak = r.run(track=True)
        sync(device)
        result.append({
            "latency_min_s": float(np.min(times[idx])),
            "latency_median_s": float(np.median(times[idx])),
            "live_peak_bytes": int(peak),
            "allocator_peak_bytes": (torch.cuda.max_memory_allocated() - base
                                     if device.type == "cuda" else None),
            "kernel_dispatches": r.launches,
            "output": _main(r.graph, outs).float().cpu().numpy(),
        })
    return result


def time_graph(graph, device, attention="sdpa", repeats=15, warmup=3):
    """Synchronised min-of-N latency (seconds) of a single graph."""
    runner = _Runner(graph, device, _feed(graph), attention=attention)
    for _ in range(warmup):
        runner.run()
    best = float("inf")
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        runner.run()
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best
