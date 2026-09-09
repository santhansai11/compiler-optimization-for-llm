"""NumPy reference executor for the computation-graph IR.

Executes any graph (original or fully fused) op-by-op on small tensors so
the project has *measured* inference latency / throughput / peak-memory
numbers and a real accuracy check (original vs optimized outputs) instead
of only modeled costs. Weights are derived deterministically from operator
names (crc32 seed); fused operators reuse ``fused_from`` provenance so the
optimized graph reproduces the original computation.
"""

import time
import zlib

import numpy as np

DEFAULT_CONFIG = {"batch": 4, "seq": 32, "d_model": 64}

# Weights are constants of the model: generate once, reuse across runs so
# timed executions measure compute rather than RNG noise.
_WEIGHT_CACHE = {}


def _seed(name):
    return int(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)


def _weight(name, dim):
    key = (name, dim)
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(_seed(name))
        cached = (rng.standard_normal((dim, dim)) / np.sqrt(dim)).astype(
            np.float32
        )
        _WEIGHT_CACHE[key] = cached
    return cached


def _softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def _layer_norm(x, eps=1e-5, weight=None, bias=None):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out.astype(np.float32)


_GELU_C = np.float32(np.sqrt(2.0 / np.pi))


def _gelu(x):
    # x * x * x is ~20x faster than x ** 3 (numpy's power ufunc is slow).
    return (
        0.5
        * x
        * (1.0 + np.tanh(_GELU_C * (x * x * x + np.float32(0.044715) * x)))
    ).astype(np.float32)


def execute(graph, repeats=30, warmup=2, feed=None):
    """Run the graph; return outputs, best-case latency and peak memory.

    Latency is the minimum over ``repeats`` timed runs (the timeit-style
    steady-state estimator — least contaminated by OS scheduling noise).
    ``feed`` overrides the synthetic input (e.g. real dataset samples).
    """
    config = dict(DEFAULT_CONFIG)
    config.update(graph.meta.get("config") or {})
    batch = int(config["batch"])
    input_shape = list(config.get("input_shape") or
                       [int(config["seq"]), int(config["d_model"])])
    if feed is None:
        rng = np.random.default_rng(0)
        if config.get("llm"):
            vocab = int(config.get("vocab_size", 50257))
            feed = rng.integers(
                0, vocab, [batch] + input_shape
            ).astype(np.int64)
        else:
            feed = rng.standard_normal([batch] + input_shape).astype(
                np.float32
            )
    else:
        feed = np.asarray(feed, dtype=np.float32)

    def run_once(track_memory):
        cache = {}
        alive = 0
        peak = 0
        uses_left = {
            node: graph.graph.out_degree(node) for node in graph.graph.nodes()
        }
        for node in graph.topological_order():
            data = graph.graph.nodes[node]
            ins = [cache[pred] for pred in graph.predecessors(node)]
            out = _compute(node, data, ins, feed)
            cache[node] = out
            if track_memory:
                alive += int(out.nbytes)
                peak = max(peak, alive)
            for pred in graph.predecessors(node):
                uses_left[pred] -= 1
                if (
                    uses_left[pred] == 0
                    and not graph.graph.nodes[pred].get("is_output")
                ):
                    if track_memory:
                        alive -= int(cache[pred].nbytes)
                    del cache[pred]
        outputs = {
            node: cache[node]
            for node in graph.graph.nodes()
            if node in cache and graph.graph.out_degree(node) == 0
        }
        return outputs, peak

    for _ in range(warmup):
        run_once(False)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        run_once(False)
        timings.append(time.perf_counter() - started)
    outputs, peak = run_once(True)

    main_output = None
    for node, data in graph.graph.nodes(data=True):
        if data.get("is_output") and node in outputs:
            main_output = outputs[node]
            break
    if main_output is None and outputs:
        main_output = outputs[sorted(outputs)[0]]

    return {
        "outputs": outputs,
        "output": main_output,
        "latency_s": float(np.min(timings)),
        "peak_bytes": int(peak),
    }


def _compute(name, data, ins, feed):
    op = data.get("op_type", "Operation")
    base = (data.get("fused_from") or [name])[0]
    weight = data.get("weight")

    if op == "Input":
        return feed
    if op == "Constant":
        value = data.get("value")
        if value is None:
            return np.float32(1.0)
        return np.asarray(value, dtype=np.float32)
    if op == "Identity":
        return ins[0]
    if op == "Scale":
        factor = np.float32(float(data.get("factor", 1.0)))
        return (ins[0] * factor).astype(np.float32)
    if op == "Relu":
        return np.maximum(ins[0], np.float32(0.0)).astype(np.float32)
    if op == "Add":
        out = ins[0]
        for value in ins[1:]:
            out = out + value
        return np.asarray(out, dtype=np.float32)
    if op == "Mul":
        out = ins[0]
        for value in ins[1:]:
            out = out * value
        return np.asarray(out, dtype=np.float32)
    if op == "Sub":
        return np.asarray(ins[0] - ins[1], dtype=np.float32)
    if op == "Softmax":
        return _softmax(ins[0])
    if op == "LayerNorm":
        return _layer_norm(
            ins[0],
            eps=float(data.get("eps", 1e-5)),
            weight=data.get("weight"),
            bias=data.get("bias"),
        )
    if op == "Gather":
        w = data.get("weight")
        if w is None:
            w = _weight(base + "::emb", int(np.max(ins[0])) + 1)
        return w[ins[0].astype(np.int64)].astype(np.float32)
    if op == "Slice":
        axis = int(data.get("axis", -1))
        start = int(data.get("start", 0))
        stop = data.get("stop")
        step = int(data.get("step", 1))
        if stop is None:
            sl = slice(start, None, step)
        else:
            sl = slice(start, int(stop), step)
        out = ins[0]
        if axis < 0:
            axis += out.ndim
        out = out[(slice(None),) * axis + (sl,)]
        if data.get("squeeze"):
            out = np.squeeze(out, axis=axis)
        return np.asarray(out, dtype=np.float32)
    if op == "Reshape":
        shape = data.get("shape_arg")
        if shape is None:
            return ins[0]
        return ins[0].reshape(tuple(int(d) for d in shape))
    if op == "Transpose":
        dims = data.get("dims")
        if dims is None:
            return np.transpose(ins[0]).astype(np.float32)
        return np.transpose(ins[0], tuple(int(d) for d in dims)).astype(
            np.float32
        )
    if op == "Broadcast":
        shape = data.get("target_shape")
        return np.broadcast_to(
            ins[0], tuple(int(d) for d in shape)
        ).astype(np.float32)
    if op == "MaskedFill":
        if data.get("where"):
            mask, x = ins[0], ins[1]
            fill_tensor = ins[2] if len(ins) > 2 else None
        else:
            x, mask = ins[0], ins[1]
            fill_tensor = ins[2] if len(ins) > 2 else None
        fill = data.get("fill_value")
        if fill is None:
            fill = float(np.min(fill_tensor)) if fill_tensor is not None \
                else float("-inf")
        return np.where(mask != 0, x, np.float32(fill)).astype(np.float32)
    if op == "GELU":
        return _gelu(ins[0])
    if op == "Tanh":
        return np.tanh(ins[0]).astype(np.float32)
    if op == "Pow":
        return np.power(
            ins[0], np.float32(float(data.get("exponent", 2.0)))
        ).astype(np.float32)
    if op == "MatMul":
        return _matmul(ins, base, weight)
    if op == "GEMM":
        return (_matmul(ins[:1], base, weight)
                + _bias(data, ins)).astype(np.float32)
    if op == "LinearGELU":
        return _gelu(_matmul(ins[:1], base, weight) + _bias(data, ins))
    if op == "LinearRelu":
        return np.maximum(
            _matmul(ins[:1], base, weight) + _bias(data, ins),
            np.float32(0.0),
        ).astype(np.float32)
    if op == "FusedAddLayerNorm":
        total = ins[0]
        for value in ins[1:]:
            total = total + value
        return _layer_norm(
            total,
            eps=float(data.get("eps", 1e-5)),
            weight=data.get("weight"),
            bias=data.get("bias"),
        )
    if op == "FusedScaleSoftmax":
        factor = np.float32(float(data.get("factor", 1.0)))
        return _softmax(ins[0] * factor)
    if op == "FusedAttention":
        return _fused_attention(data, ins)
    raise NotImplementedError(f"executor cannot run op: {op!r}")


def _bias(data, ins):
    bias = data.get("bias")
    if bias is not None:
        return np.asarray(bias, dtype=np.float32)
    return np.float32(0.0)


def _matmul(ins, base, weight=None):
    """MatMul with orientation resolution.

    Single input -> x @ W, where ``W`` is the node's stored ``weight``
    attribute (trained models) or the deterministic seeded matrix.
    Two inputs -> the unique shape-valid product; equal-shaped 3-D
    operands with no plain interpretation fall back to the Q @ K^T
    convention.
    """
    if len(ins) == 1:
        x = ins[0]
        if weight is None:
            W = _weight(base + "::w", x.shape[-1])
        else:
            W = np.asarray(weight, dtype=np.float32)
        return (x @ W).astype(np.float32)
    a, b = ins[0], ins[1]
    if a.ndim >= 2 and b.ndim >= 2:
        if a.shape[-1] == b.shape[-2]:
            return (a @ b).astype(np.float32)
        if b.shape[-1] == a.shape[-2]:
            return (b @ a).astype(np.float32)
        if a.ndim == 3 and b.ndim == 3 and a.shape[-1] == b.shape[-1]:
            return (a @ b.transpose(0, 2, 1)).astype(np.float32)
    out = a
    for value in ins[1:]:
        out = out @ value
    return np.asarray(out, dtype=np.float32)


def _apply_steps(x, steps):
    """Apply a recorded chain of view transforms (slice/view/perm/mm)."""
    for step in steps or ():
        kind = step[0]
        if kind == "slice":
            _, start, stop, stride, axis, squeeze = step
            ax = int(axis)
            if ax < 0:
                ax += x.ndim
            sl = slice(
                int(start),
                None if stop is None else int(stop),
                int(stride),
            )
            x = x[(slice(None),) * ax + (sl,)]
            if squeeze:
                x = np.squeeze(x, axis=ax)
        elif kind == "view":
            x = x.reshape(tuple(int(d) for d in step[1]))
        elif kind == "broadcast":
            x = np.broadcast_to(
                x, tuple(int(d) for d in step[1])
            ).astype(np.float32)
        elif kind == "perm":
            perm = step[1]
            x = np.transpose(
                x,
                tuple(int(d) for d in perm) if perm is not None else None,
            )
        elif kind == "mm":
            x = (x @ step[1]).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _fused_attention(data, ins):
    factor = data.get("factor")
    x = ins[0]
    q_steps = data.get("q_transform")
    if q_steps is not None:
        # real-graph path: Q/K/V are view chains over the shared input
        dim = x.shape[-1] if factor is None else 0.0
        factor = float(
            factor if factor is not None else 1.0 / np.sqrt(dim)
        )
        q = _apply_steps(x, q_steps)
        k = _apply_steps(x, data.get("k_transform") or [])
        v = _apply_steps(x, data.get("v_transform") or [])
        # k may already be in K^T layout (the transpose can sit inside its
        # view chain); resolve orientation by shape like _matmul does.
        if q.shape[-1] == k.shape[-2]:
            scores = q @ k
        elif q.shape[-1] == k.shape[-1]:
            scores = q @ k.swapaxes(-1, -2)
        else:
            raise ValueError(
                f"fused attention: incompatible q/k shapes "
                f"{q.shape} vs {k.shape}"
            )
        scores = scores * np.float32(factor)
        if data.get("mask_kind") == "add":
            scores = scores + data["mask_array"]
        elif data.get("mask_kind") == "where":
            tail = scores.shape[-2:]
            tril = np.tril(np.ones(tail, dtype=bool))
            scores = np.where(tril, scores,
                              np.float32(data["mask_fill"]))
        probs = _softmax(scores)
        return (probs @ v).astype(np.float32)

    # legacy demo path: derived weights over a 3-D input
    provenance = data.get("fused_from") or ["Q", "K", "V"]
    q_name, k_name, v_name = provenance[0], provenance[1], provenance[2]
    dim = x.shape[-1]
    q = x @ _weight(q_name + "::w", dim)
    k = x @ _weight(k_name + "::w", dim)
    v = x @ _weight(v_name + "::w", dim)
    factor = np.float32(
        float(factor) if factor is not None else 1.0 / np.sqrt(dim)
    )
    scores = (q @ k.transpose(0, 2, 1)) * factor
    return (_softmax(scores) @ v).astype(np.float32)
