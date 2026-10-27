"""Neuro-symbolic rewrite search.

SYMBOLIC part  - ``search.rewrites``: every legal single rewrite (identity /
                 constant folding, scale merging, CSE, transpose algebra,
                 operator fusion, attention fusion, dead code). Any sequence
                 of them preserves the model's outputs by construction.
NEURAL part    - ``LatencyGNN``: a graph neural network trained on latencies
                 MEASURED on a real GPU, predicting log(latency) of any
                 (partially optimised) graph at a given tensor size.

Search: from the current graph, enumerate all legal rewrites, score each
resulting graph with the network (no GPU needed at search time), apply the
best one if it is predicted faster (or equally fast and smaller), repeat.
The network is what decides e.g. that fusing attention into the fused kernel
is NOT worth it at sizes where that kernel is slower on this GPU.

    python benchmarks/neuro_symbolic_study.py     # data + LOFO + final model
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from search import learned_optimizer as lo
from search.rewrites import RULES, apply_rewrite, enumerate_rewrites

WEIGHTS = Path(__file__).with_name("ns_value.pt")
SIZE_DIM = 5
GLOBAL_DIM = lo.GLOBAL_DIM + SIZE_DIM


# ------------------------------------------------------------------ features
def featurize_sized(graph):
    """Graph features + tensor-size features (batch, seq, d_model, work prior)."""
    from utils.cost_model import get_op_latency

    x, edges, glob = lo.featurize(graph)
    cfg = graph.meta.get("config") or {}
    b, s, d = (int(cfg.get("batch", 4)), int(cfg.get("seq", 32)), int(cfg.get("d_model", 64)))
    prior = sum(get_op_latency(dd.get("op_type")) for _, dd in graph.graph.nodes(data=True))
    size = np.array([math.log(b) / 3, math.log(s) / 6, math.log(d) / 6,
                     math.log(b * s * d) / 14, math.log1p(prior) / 8], dtype=np.float32)
    return x, edges, np.concatenate([glob, size])


class LatencyGNN(nn.Module):
    """Additive learned cost model (as in learned compiler cost models).

    Every operation gets a context-aware predicted time ``softplus(MLP)``
    (own features + tensor-size features + 2 rounds of forward/backward
    message passing); the graph latency is the SUM, so the model generalises
    across architectures that reuse the same operations. Output: log(us).
    """

    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.inp = nn.Linear(lo.NODE_DIM + GLOBAL_DIM, hidden)
        self.fwd = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.bwd = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.node_time = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("mu", torch.zeros(()))     # kept for a uniform API
        self.register_buffer("sigma", torch.ones(()))

    def forward(self, x, src, dst, batch, glob):
        h = torch.relu(self.inp(torch.cat([x, glob[batch]], -1)))
        agg = lo.PlanGNN._mean_agg
        for f, b in zip(self.fwd, self.bwd):
            h = h + torch.relu(f(torch.cat([h, agg(h, src, dst)], -1))) \
                  + torch.relu(b(torch.cat([h, agg(h, dst, src)], -1)))
        t = nn.functional.softplus(self.node_time(h)).squeeze(-1)
        total = torch.zeros(glob.shape[0]).index_add_(0, batch, t)
        return torch.log(total + 1e-6) + self.log_scale

    def predict_log_us(self, feats):
        with torch.no_grad():
            return self(*lo.collate(feats)).numpy()


def fit_value(samples, epochs=400, seed=0, lr=4e-3):
    """samples: [{'features': featurize_sized(...), 'log_us': float}]."""
    torch.manual_seed(seed)
    model = LatencyGNN()
    y = torch.tensor([s["log_us"] for s in samples], dtype=torch.float32)
    with torch.no_grad():
        model.log_scale.fill_(float(y.mean()) - 2.0)
    batch = lo.collate([s["features"] for s in samples])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.smooth_l1_loss(model(*batch), y, beta=0.1)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model, float(loss.item())


# -------------------------------------------------------------------- search
def guided_search(graph, score_fn, max_steps=80, tol=0.003, min_speedup=1.0, stats=None):
    """Greedy neuro-symbolic search.

    ``score_fn(list_of_graphs) -> predicted log-latency`` (lower is better).
    Accept the best candidate if it is predicted at least ``min_speedup``
    times faster, or predicted within ``tol`` (relative, in log units) AND
    strictly smaller. Returns ``(graph, trace)``.

    If ``stats`` is a dict it is filled with decision counts: candidates
    scored, candidates without a usable (finite) estimate, and how the
    candidates left at the end were rejected.
    """
    margin = max(1e-9, math.log(min_speedup))
    if stats is not None:
        stats.update({"candidates_scored": 0, "no_estimate": 0,
                      "final_rejected_predicted_slower": 0,
                      "final_rejected_below_threshold": 0})
    cur = graph
    cur_score = float(score_fn([cur])[0])
    trace = []
    for _ in range(max_steps):
        cands = []
        for rule, match in enumerate_rewrites(cur):
            new = apply_rewrite(cur, rule, match)
            if new is not None:
                cands.append((rule.name, new))
        if not cands:
            break
        scores = np.asarray(score_fn([g for _, g in cands]), dtype=float)
        finite = np.isfinite(scores)
        if stats is not None:
            stats["candidates_scored"] += len(cands)
            stats["no_estimate"] += int((~finite).sum())
        order = [i for i in np.argsort(np.where(finite, scores, np.inf), kind="stable") if finite[i]]
        chosen = None
        for i in order:
            better = scores[i] < cur_score - margin
            tie_smaller = scores[i] <= cur_score + tol and cands[i][1].node_count() < cur.node_count()
            if better or tie_smaller:
                chosen = i
                break
        if chosen is None:
            if stats is not None:
                slower = int(sum(scores[i] >= cur_score for i in order))
                stats["final_rejected_predicted_slower"] = slower
                stats["final_rejected_below_threshold"] = len(order) - slower
            break
        trace.append({"rule": cands[chosen][0], "predicted_log_us": float(scores[chosen]),
                      "nodes": cands[chosen][1].node_count()})
        cur, cur_score = cands[chosen][1], float(scores[chosen])
    return cur, trace


def table_score(graphs):
    """No-learning fallback: sum of calibrated per-op latencies (log)."""
    from utils.cost_model import get_op_latency

    return [math.log(sum(get_op_latency(d.get("op_type")) for _, d in g.graph.nodes(data=True)) + 1.0)
            for g in graphs]


def load_value(path=WEIGHTS):
    model = LatencyGNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def neuro_symbolic_search(graph, max_iterations=80, model=None):
    """Pipeline hook: returns ``(optimised graph, info)``.

    Uses the trained latency GNN when ``search/ns_value.pt`` exists, otherwise
    the calibrated per-op table.
    """
    started = time.perf_counter()
    if model is None and WEIGHTS.exists():
        model = load_value()
    if model is not None:
        score = lambda gs: model.predict_log_us([featurize_sized(g) for g in gs])
        guide = "latency GNN trained on real-GPU measurements"
    else:
        score, guide = table_score, "calibrated per-op table (no trained model found)"
    out, trace = guided_search(graph, score, max_steps=max_iterations)
    info = {"pass": "neuro_symbolic_search", "guidance": guide, "iterations": len(trace),
            "applied": trace, "rules_available": [r.name for r in RULES],
            "nodes_after": out.node_count(), "duration_s": time.perf_counter() - started}
    out.meta["neuro_symbolic_search"] = info
    return out, info
