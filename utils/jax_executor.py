"""Interpret a computation-graph IR as a JAX function (so XLA can compile it).

Same weights and operand-order rules as ``utils.executor``. Fused IR nodes
are expressed with their mathematical meaning (XLA decides its own fusion),
so the SAME graph - original or already compiled by our passes - can be run
through ``jax.jit`` / OpenXLA for a like-for-like comparison.
"""

import numpy as np
import jax
import jax.numpy as jnp

from utils.executor import _weight


def _mm(ins, base, weight, transpose_rhs):
    if len(ins) == 1:
        x = ins[0]
        W = jnp.asarray(weight) if weight is not None else jnp.asarray(_weight(base + "::w", x.shape[-1]))
        return x @ W
    a, b = ins[0], ins[1]
    if transpose_rhs:
        return a @ jnp.swapaxes(b, -1, -2)
    if a.ndim >= 2 and b.ndim >= 2:
        if a.shape[-1] == b.shape[-2]:
            return a @ b
        if b.shape[-1] == a.shape[-2]:
            return b @ a
        if a.ndim == 3 and b.ndim == 3 and a.shape[-1] == b.shape[-1]:
            return a @ jnp.transpose(b, (0, 2, 1))
    out = a
    for v in ins[1:]:
        out = out @ v
    return out


def _ln(x, d):
    w, b = d.get("weight"), d.get("bias")
    mu = x.mean(-1, keepdims=True)
    out = (x - mu) / jnp.sqrt(x.var(-1, keepdims=True) + float(d.get("eps", 1e-5)))
    if w is not None:
        out = out * jnp.asarray(w)
    if b is not None:
        out = out + jnp.asarray(b)
    return out


def _rms(x, d):
    out = x / jnp.sqrt((x * x).mean(-1, keepdims=True) + float(d.get("eps", 1e-6)))
    return out if d.get("weight") is None else out * jnp.asarray(d["weight"])


def _gelu(x):
    return jax.nn.gelu(x, approximate=True)


def build(graph):
    """Return ``f(feed) -> main output`` for ``graph`` (traceable by jit)."""
    order = list(graph.topological_order())
    nodes = graph.graph.nodes
    plan = [(n, nodes[n], list(graph.predecessors(n))) for n in order]
    sinks = [n for n in order if graph.graph.out_degree(n) == 0]
    out_node = next((n for n in order if nodes[n].get("is_output")), sorted(sinks)[0])

    def fn(feed):
        v = {}
        for name, d, preds in plan:
            ins = [v[p] for p in preds]
            op = d.get("op_type")
            base = (d.get("fused_from") or [name])[0]
            if op == "Input":
                r = feed
            elif op == "Constant":
                r = jnp.asarray(d.get("value"), dtype=jnp.float32)
            elif op == "Identity":
                r = ins[0]
            elif op == "Transpose":
                dims = d.get("dims")
                r = jnp.transpose(ins[0], None if dims is None else tuple(dims))
            elif op == "Reshape":
                shape = d.get("shape_arg")
                r = ins[0] if shape is None else jnp.reshape(ins[0], tuple(shape))
            elif op == "Broadcast":
                r = jnp.broadcast_to(ins[0], tuple(d["target_shape"]))
            elif op == "Tanh":
                r = jnp.tanh(ins[0])
            elif op == "Pow":
                r = jnp.power(ins[0], float(d.get("exponent", 2.0)))
            elif op == "Scale":
                r = ins[0] * float(d.get("factor", 1.0))
            elif op in ("Add", "Mul"):
                r = ins[0]
                for t in ins[1:]:
                    r = r + t if op == "Add" else r * t
            elif op == "Sub":
                r = ins[0] - ins[1]
            elif op == "Relu":
                r = jnp.maximum(ins[0], 0.0)
            elif op == "GELU":
                r = _gelu(ins[0])
            elif op == "SiLU":
                r = jax.nn.silu(ins[0])
            elif op == "Softmax":
                r = jax.nn.softmax(ins[0], axis=-1)
            elif op == "LayerNorm":
                r = _ln(ins[0], d)
            elif op == "RMSNorm":
                r = _rms(ins[0], d)
            elif op == "MatMul":
                r = _mm(ins, base, d.get("weight"), bool(d.get("transpose_rhs")))
            elif op in ("GEMM", "LinearGELU", "LinearRelu", "LinearSiLU"):
                r = _mm(ins[:1], base, d.get("weight"), False)
                if d.get("bias") is not None:
                    r = r + jnp.asarray(d["bias"])
                r = {"GEMM": lambda t: t, "LinearGELU": _gelu,
                     "LinearRelu": lambda t: jnp.maximum(t, 0.0), "LinearSiLU": jax.nn.silu}[op](r)
            elif op in ("FusedAddLayerNorm", "FusedAddRMSNorm"):
                t = ins[0]
                for u in ins[1:]:
                    t = t + u
                r = _ln(t, d) if op == "FusedAddLayerNorm" else _rms(t, d)
            elif op == "FusedScaleSoftmax":
                r = jax.nn.softmax(ins[0] * float(d.get("factor", 1.0)), axis=-1)
            elif op == "FusedAttention":
                if d.get("q_transform") is not None or d.get("mask_kind"):
                    raise NotImplementedError("real-model attention not supported in JAX path")
                x = ins[0]
                names = d.get("fused_from") or ["Q", "K", "V"]
                dim = x.shape[-1]
                q, k, vv = (x @ jnp.asarray(_weight(names[i] + "::w", dim)) for i in range(3))
                f = d.get("factor")
                f = float(f) if f is not None else 1.0 / float(np.sqrt(dim))
                r = jax.nn.softmax((q @ jnp.swapaxes(k, -1, -2)) * f, axis=-1) @ vv
            else:
                raise NotImplementedError(f"jax executor cannot run {op!r}")
            v[name] = r
        return v[out_node]

    return fn
