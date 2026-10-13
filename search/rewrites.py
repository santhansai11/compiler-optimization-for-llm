"""Symbolic rewrite rules (one instance at a time) for neuro-symbolic search.

Every rule is a *semantics-preserving* local rewrite with an explicit
matcher, so any sequence of rewrites keeps the model's outputs identical;
the learned component only chooses WHICH legal rewrite to apply next.

    rules:  eliminate-identity | fold-constant | merge-scales | cse |
            double-transpose | fold-transpose | fuse-pair | fuse-attention |
            remove-dead
"""

import numpy as np

from passes.canonicalize import _match_attention_pattern, _replace_attention_pattern
from passes.merge import BIAS_FUSION_PAIR, FUSION_PATTERNS, PROTECTED_OPS, _merge_pair
from passes.normalize import (COMMUTATIVE_OPS, FOLDABLE_OPS, PARAMETERIZED_OPS,
                              _evaluate, _is_last_two_swap, _perm_of,
                              _replace_input, _semantic_sig)


class RewriteRule:
    def __init__(self, name, matcher, applier, description=""):
        self.name, self.matcher, self.applier = name, matcher, applier
        self.description = description

    def find(self, graph):
        return self.matcher(graph)

    def apply(self, graph, match):
        """Apply in place; True on success."""
        return self.applier(graph, match)


# ------------------------------------------------------------------ rules
def _find_identities(graph):
    g = graph.graph
    return [{"node": n} for n, d in g.nodes(data=True)
            if d.get("op_type") == "Identity" and g.in_degree(n) == 1
            and not d.get("is_output")
            and not any(g.has_edge(next(iter(g.predecessors(n))), s)
                        for s in g.successors(n))]


def _apply_identity(graph, m):
    g, n = graph.graph, m["node"]
    src = next(iter(g.predecessors(n)))
    for s in list(g.successors(n)):
        _replace_input(g, s, n, src)
    g.remove_node(n)
    return True


def _find_foldable(graph):
    g = graph.graph
    out = []
    for n, d in g.nodes(data=True):
        if d.get("op_type") in FOLDABLE_OPS:
            preds = list(g.predecessors(n))
            if preds and all(g.nodes[p].get("op_type") == "Constant"
                             and g.nodes[p].get("value") is not None for p in preds):
                out.append({"node": n})
    return out


def _apply_fold(graph, m):
    g, n = graph.graph, m["node"]
    d = g.nodes[n]
    preds = list(g.predecessors(n))
    result = _evaluate(d["op_type"], [g.nodes[p]["value"] for p in preds], d)
    if result is None:
        return False
    d["op_type"], d["value"] = "Constant", result
    for p in preds:
        g.remove_edge(p, n)
    return True


def _find_scale_pairs(graph):
    g = graph.graph
    out = []
    for n, d in g.nodes(data=True):
        if d.get("op_type") == "Scale" and g.in_degree(n) == 1:
            inner = next(iter(g.predecessors(n)))
            idata = g.nodes[inner]
            if idata.get("op_type") == "Scale" and g.out_degree(inner) == 1 \
                    and not idata.get("is_output"):
                out.append({"node": n, "inner": inner})
    return out


def _apply_scale_pair(graph, m):
    g, n, inner = graph.graph, m["node"], m["inner"]
    g.nodes[n]["factor"] = float(g.nodes[inner].get("factor", 1.0)) \
        * float(g.nodes[n].get("factor", 1.0))
    for src in list(g.predecessors(inner)):
        _replace_input(g, n, inner, src)
    g.remove_node(inner)
    return True


def _find_cse(graph):
    g = graph.graph
    seen, out = {}, []
    for n in graph.topological_order():
        d = g.nodes[n]
        op = d.get("op_type")
        if op == "Constant" or op in PARAMETERIZED_OPS or op in PROTECTED_OPS:
            continue
        preds = tuple(sorted(g.predecessors(n))) if op in COMMUTATIVE_OPS \
            else tuple(g.predecessors(n))
        sig = (op, preds, _semantic_sig(d))
        if sig in seen:
            c = seen[sig]
            if not (set(g.successors(n)) & set(g.successors(c))) \
                    and not d.get("is_output"):
                out.append({"node": n, "canonical": c})
        else:
            seen[sig] = n
    return out


def _apply_cse(graph, m):
    g, n, c = graph.graph, m["node"], m["canonical"]
    for s in list(g.successors(n)):
        _replace_input(g, s, n, c)
    g.remove_node(n)
    return True


def _find_double_transpose(graph):
    g = graph.graph
    out = []
    for n, d in g.nodes(data=True):
        if d.get("op_type") != "Transpose" or d.get("is_output") or g.in_degree(n) != 1:
            continue
        inner = next(iter(g.predecessors(n)))
        idata = g.nodes[inner]
        if idata.get("op_type") != "Transpose" or g.in_degree(inner) != 1:
            continue
        a, b = _perm_of(idata), _perm_of(d)
        ok = (a is None and b is None) or (
            a is not None and b is not None and len(a) == len(b)
            and all(a[b[i]] == i for i in range(len(a))))
        src = next(iter(g.predecessors(inner)))
        if ok and not any(g.has_edge(src, s) for s in g.successors(n)):
            out.append({"node": n, "inner": inner, "src": src})
    return out


def _apply_double_transpose(graph, m):
    g, n, inner, src = graph.graph, m["node"], m["inner"], m["src"]
    for s in list(g.successors(n)):
        _replace_input(g, s, n, src)
    g.remove_node(n)
    if g.out_degree(inner) == 0 and not g.nodes[inner].get("is_output"):
        g.remove_node(inner)
    return True


def _find_fold_transpose(graph):
    g = graph.graph
    out = []
    for n, d in g.nodes(data=True):
        if d.get("op_type") != "MatMul" or d.get("transpose_rhs"):
            continue
        preds = list(g.predecessors(n))
        if len(preds) != 2:
            continue
        rhs = preds[1]
        rd = g.nodes[rhs]
        if rd.get("op_type") == "Transpose" and g.out_degree(rhs) == 1 \
                and g.in_degree(rhs) == 1 and not rd.get("is_output") \
                and _is_last_two_swap(_perm_of(rd)):
            src = next(iter(g.predecessors(rhs)))
            if not g.has_edge(src, n):
                out.append({"node": n, "rhs": rhs, "src": src})
    return out


def _apply_fold_transpose(graph, m):
    g, n, rhs, src = graph.graph, m["node"], m["rhs"], m["src"]
    _replace_input(g, n, rhs, src)
    g.remove_node(rhs)
    g.nodes[n]["transpose_rhs"] = True
    return True


def _find_fuse_pairs(graph):
    g = graph.graph
    out = []
    for u, v in g.edges():
        pu, pv = g.nodes[u].get("op_type"), g.nodes[v].get("op_type")
        if (pu, pv) not in FUSION_PATTERNS or pu in PROTECTED_OPS or g.out_degree(u) != 1:
            continue
        extra = [p for p in g.predecessors(v) if p != u]
        if (pu, pv) == BIAS_FUSION_PAIR:
            if len(extra) > 1 or (extra and g.nodes[extra[0]].get("op_type") != "Constant"):
                continue
        elif extra:
            continue
        out.append({"producer": u, "consumer": v})
    return out


def _apply_fuse_pair(graph, m):
    g = graph.graph
    u, v = m["producer"], m["consumer"]
    fused = FUSION_PATTERNS[(g.nodes[u]["op_type"], g.nodes[v]["op_type"])]
    extra = [p for p in g.predecessors(v) if p != u]
    bias = None
    if (g.nodes[u]["op_type"], g.nodes[v]["op_type"]) == BIAS_FUSION_PAIR and extra:
        bias = g.nodes[extra[0]].get("value")
    _merge_pair(g, u, v, fused, bias, extra)
    return True


def _find_attention(graph):
    g = graph.graph
    return [{"softmax": n} for n, d in g.nodes(data=True)
            if d.get("op_type") == "Softmax" and _match_attention_pattern(g, n) is not None]


def _apply_attention(graph, m):
    g = graph.graph
    pattern = _match_attention_pattern(g, m["softmax"])
    if pattern is None:
        return False
    i = 0
    while g.has_node(f"FusedAttention_{i}"):
        i += 1
    _replace_attention_pattern(g, pattern, f"FusedAttention_{i}")
    return True


def _find_dead(graph):
    g = graph.graph
    if not any(d.get("is_output") for _, d in g.nodes(data=True)):
        return []
    return [{"node": n} for n, d in g.nodes(data=True)
            if g.out_degree(n) == 0 and not d.get("is_output")]


def _apply_dead(graph, m):
    graph.graph.remove_node(m["node"])
    return True


RULES = [
    RewriteRule("eliminate-identity", _find_identities, _apply_identity, "Identity(x) => x"),
    RewriteRule("fold-constant", _find_foldable, _apply_fold, "op(Const...) => Const"),
    RewriteRule("merge-scales", _find_scale_pairs, _apply_scale_pair, "Scale(a)->Scale(b) => Scale(ab)"),
    RewriteRule("cse", _find_cse, _apply_cse, "duplicate op => shared op"),
    RewriteRule("double-transpose", _find_double_transpose, _apply_double_transpose, "(A^T)^T => A"),
    RewriteRule("fold-transpose", _find_fold_transpose, _apply_fold_transpose, "A·B^T => GEMM(transpose_rhs)"),
    RewriteRule("fuse-pair", _find_fuse_pairs, _apply_fuse_pair, "producer+consumer => fused op"),
    RewriteRule("fuse-attention", _find_attention, _apply_attention, "Q/K/V·QK^T·scale·softmax·AV => FusedAttention"),
    RewriteRule("remove-dead", _find_dead, _apply_dead, "unused op => removed"),
]
RULE_BY_NAME = {r.name: r for r in RULES}


def enumerate_rewrites(graph):
    """All legal single rewrites: list of (rule, match)."""
    return [(rule, m) for rule in RULES for m in rule.find(graph)]


def apply_rewrite(graph, rule, match):
    """Apply to a COPY; return the new graph or None if it failed/broke the DAG."""
    import networkx as nx

    new = graph.copy()
    try:
        if not rule.apply(new, match):
            return None
    except Exception:
        return None
    return new if nx.is_directed_acyclic_graph(new.graph) else None
