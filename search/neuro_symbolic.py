"""Neuro-symbolic rewrite search: symbolic rules + hybrid learned guidance.

Each round enumerates every applicable symbolic rewrite, scores each
candidate result with a hybrid objective - the deterministic cost model
(ground truth) blended with the learned GNN surrogate (fast guidance) -
minus a small node-count penalty, and greedily applies the best-scoring
rewrite until no candidate improves the score. This mirrors real learned
compilers: the model proposes and ranks, the cost model validates.
"""

import time

from search.gnn import load_or_train
from search.rewrites import RULES
from utils.cost_model import sequential_latency_us

COST_WEIGHT = 0.7
GNN_WEIGHT = 0.3
NODE_PENALTY = 0.01


def _score_candidate(candidate, base_cost, scorer):
    """Hybrid objective: cost-model ratio + GNN prediction - size penalty."""
    trial_cost = sequential_latency_us(candidate)
    modeled_ratio = base_cost / trial_cost if trial_cost else 1.0
    return (
        COST_WEIGHT * modeled_ratio
        + GNN_WEIGHT * scorer.score_graph(candidate)
        - NODE_PENALTY * candidate.node_count()
    )


def neuro_symbolic_search(graph, max_iterations=25, scorer=None):
    """Greedy rewrite search over symbolic rules with hybrid guidance.

    Returns ``(graph, info)`` with the rewritten graph and a log of the
    applied rules and their scores.
    """
    started = time.perf_counter()
    scorer = scorer or load_or_train()
    current = graph.copy()
    applied = []

    base_cost = sequential_latency_us(graph)
    initial_score = _score_candidate(current, base_cost, scorer)
    best_score = initial_score

    for _ in range(max_iterations):
        candidates = []
        for rule in RULES:
            for match in rule.find(current):
                trial = current.copy()
                try:
                    if not rule.apply(trial, match):
                        continue
                except Exception:
                    continue
                score = _score_candidate(trial, base_cost, scorer)
                candidates.append((score, rule.name, trial))
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item[0], item[1]))
        top_score, top_name, top_graph = candidates[0]
        if top_score <= best_score + 1e-9:
            break
        best_score = top_score
        current = top_graph
        applied.append({
            "rule": top_name,
            "score": float(top_score),
            "nodes": current.node_count(),
        })

    info = {
        "pass": "neuro_symbolic_search",
        "guidance": (
            f"hybrid: {COST_WEIGHT:.0%} cost model + {GNN_WEIGHT:.0%} GNN "
            f"surrogate, node penalty {NODE_PENALTY}"
        ),
        "initial_score": float(initial_score),
        "final_score": float(best_score),
        "iterations": len(applied),
        "applied": applied,
        "rules_available": [rule.name for rule in RULES],
        "nodes_after": current.node_count(),
        "duration_s": time.perf_counter() - started,
    }
    current.meta["neuro_symbolic_search"] = info
    return current, info
