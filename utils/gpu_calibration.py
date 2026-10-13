"""GPU-calibrated, shape-aware cost model.

Per-node times are measured on a real device (``profile_nodes``); every node
is mapped to a *class* with a *work* number (FLOPs for GEMMs, elements for
element-wise/normalisation kernels) and a line ``time = a + b * work`` is
fitted per class. Predicting a graph = summing its nodes. The fit is
validated out-of-sample on architectures that were not used for fitting.
"""

import time

import numpy as np
import torch

from utils.torch_executor import _Runner, _feed, sync

ELEMENTWISE = {"Add", "Mul", "Sub", "Scale", "Relu", "GELU", "SiLU", "Identity", "MaskedFill"}
VIEWS = {"Reshape", "Transpose", "Slice", "Broadcast", "Cast", "Gather"}
SKIP = {"Input", "Constant"}


def node_class(op, n_inputs):
    if op in SKIP:
        return None
    if op == "MatMul":
        return "linear" if n_inputs == 1 else "bmm"
    if op == "GEMM":
        return "linear"
    if op in ("LinearGELU", "LinearRelu", "LinearSiLU"):
        return "fused_linear_act"
    if op in ELEMENTWISE:
        return "elementwise"
    if op == "LayerNorm":
        return "layernorm"
    if op == "RMSNorm":
        return "rmsnorm"
    if op == "Softmax":
        return "softmax"
    if op in ("FusedAddLayerNorm", "FusedAddRMSNorm"):
        return "fused_add_norm"
    if op == "FusedScaleSoftmax":
        return "fused_scale_softmax"
    if op == "FusedAttention":
        return "fused_attention"
    if op in VIEWS:
        return "view"
    return "elementwise"


def node_work(op, in_shapes, out_shape):
    """FLOPs for contraction-type ops, element count for everything else."""
    n_out = float(np.prod(out_shape)) if out_shape else 1.0
    if op in ("MatMul", "GEMM", "LinearGELU", "LinearRelu", "LinearSiLU"):
        a = in_shapes[0]
        if len(in_shapes) == 1:
            k = a[-1]
        else:
            b = in_shapes[1]
            k = a[-1] if (a[-1] == b[-2] or a[-1] == b[-1]) else b[-1]
        return 2.0 * n_out * k + (n_out if op != "MatMul" else 0.0)
    if op == "FusedAttention":
        b_, s_, d_ = in_shapes[0][0], in_shapes[0][-2], in_shapes[0][-1]
        return 6.0 * b_ * s_ * d_ * d_ + 4.0 * b_ * s_ * s_ * d_
    return n_out * max(1, len(in_shapes) - 1 if op in ("Add", "Mul") else 1)


def profile_nodes(graph, device, reps=20, trials=5, attention="auto"):
    """Measure every node's kernel time in isolation (batched, sync-corrected)."""
    feed = _feed(graph)
    runner = _Runner(graph, device, feed, attention=attention)
    # cost of an empty synchronise, subtracted from every trial
    sync(device)
    t0 = time.perf_counter()
    for _ in range(50):
        sync(device)
    sync_cost = (time.perf_counter() - t0) / 50
    rows, cache = [], {}
    with torch.no_grad():
        for node, data, preds in runner.plan:
            ins = [cache[p] for p in preds]
            out = runner._compute(node, data, ins)
            cache[node] = out
            op = data.get("op_type", "?")
            cls = node_class(op, len(ins))
            if cls is None:
                continue
            for _ in range(3):
                runner._compute(node, data, ins)
            sync(device)
            t = time.perf_counter()
            runner._compute(node, data, ins)
            sync(device)
            one = max(time.perf_counter() - t, 1e-6)
            n_rep = int(max(3, min(reps, 0.01 / one)))  # keep heavy ops cheap to time
            best = float("inf")
            for _ in range(trials):
                sync(device)
                t = time.perf_counter()
                for _ in range(n_rep):
                    runner._compute(node, data, ins)
                sync(device)
                best = min(best, (time.perf_counter() - t - sync_cost) / n_rep)
            in_shapes = [tuple(i.shape) for i in ins]
            rows.append({"node": node, "op": op, "cls": cls,
                         "work": node_work(op, in_shapes, tuple(out.shape)),
                         "seconds": max(best, 1e-7)})
    return rows


def fit(rows):
    """Least-squares ``seconds = a + b*work`` per class (a, b >= 0)."""
    by = {}
    for r in rows:
        by.setdefault(r["cls"], []).append(r)
    model = {}
    for cls, rs in by.items():
        x = np.array([r["work"] for r in rs])
        y = np.array([r["seconds"] for r in rs])
        a, b = float(np.median(y)), 0.0
        if len(rs) >= 4 and np.ptp(x) > 0:
            A = np.stack([np.ones_like(x), x], 1)
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
            if coef[1] > 0:
                a = max(float(coef[0]), 0.0)
                b = float(coef[1])
                # refit slope if the intercept was clipped
                if coef[0] < 0:
                    b = float((x @ y) / (x @ x))
        model[cls] = {"a_us": a * 1e6, "b_us_per_work": b * 1e6, "samples": len(rs)}
    return model


def predict_rows(model, rows, fallback="elementwise"):
    total = 0.0
    for r in rows:
        c = model.get(r["cls"]) or model.get(fallback) or {"a_us": 0, "b_us_per_work": 0}
        total += c["a_us"] + c["b_us_per_work"] * r["work"]
    return total  # microseconds
