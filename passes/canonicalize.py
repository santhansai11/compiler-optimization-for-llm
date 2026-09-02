"""Attention graph canonicalization pass.

Detects the classic attention sub-graph

    Q/K/V projections -> QK^T -> Scale -> Softmax -> (A)V matmul

and replaces it with a single canonical ``FusedAttention`` operator - the
pattern production runtimes (FlashAttention / SDPA kernels) execute.
"""

import time


FUSED_OP_TYPE = "FusedAttention"
PATTERN_MEMBERS = ("q", "k", "v", "score", "scale", "softmax", "av")


def canonicalize_attention(graph):
    """Rewrite attention clusters into canonical fused operators.

    Returns ``(graph, info)`` where ``info`` reports how many attention
    clusters were detected and how many were canonicalized (the ACR basis).
    """
    started = time.perf_counter()
    optimized = graph.copy()
    g = optimized.graph

    detected = 0
    canonicalized = 0
    details = []
    index = 0
    changed = True

    while changed:
        changed = False
        softmax_nodes = [
            node
            for node, data in g.nodes(data=True)
            if data.get("op_type") == "Softmax"
        ]
        for sm_name in softmax_nodes:
            pattern = _match_attention_pattern(g, sm_name)
            if pattern is None:
                continue
            detected += 1
            fused_name = f"FusedAttention_{index}"
            _replace_attention_pattern(g, pattern, fused_name)
            index += 1
            canonicalized += 1
            details.append(
                "fused ["
                + ", ".join(pattern[member] for member in PATTERN_MEMBERS)
                + f"] -> {fused_name}"
            )
            changed = True
            break

    info = {
        "pass": "attention_canonicalization",
        "detected": detected,
        "canonicalized": canonicalized,
        "attention_canonicalization_rate": (
            canonicalized / detected if detected else 0.0
        ),
        "details": details,
        "nodes_after": optimized.node_count(),
        "duration_s": time.perf_counter() - started,
    }
    optimized.meta["attention_canonicalization"] = info
    return optimized, info


def _match_attention_pattern(g, sm_name):
    """Match a strictly-wired attention cluster around a Softmax node."""
    preds = list(g.predecessors(sm_name))
    if len(preds) != 1:
        return None
    scale = preds[0]
    if g.nodes[scale].get("op_type") != "Scale" or g.in_degree(scale) != 1:
        return None

    score = list(g.predecessors(scale))[0]
    if g.nodes[score].get("op_type") != "MatMul" or g.in_degree(score) != 2:
        return None

    q_node, k_node = list(g.predecessors(score))
    for node in (q_node, k_node):
        if g.nodes[node].get("op_type") != "MatMul":
            return None

    av = None
    for succ in g.successors(sm_name):
        if g.nodes[succ].get("op_type") == "MatMul" and g.in_degree(succ) == 2:
            av = succ
            break
    if av is None:
        return None

    v_node = [p for p in g.predecessors(av) if p != sm_name][0]
    if g.nodes[v_node].get("op_type") != "MatMul":
        return None

    # Strict wiring: every member is used only inside the pattern, so the
    # rewrite preserves the rest of the graph.
    if set(g.successors(q_node)) != {score}:
        return None
    if set(g.successors(k_node)) != {score}:
        return None
    if set(g.successors(v_node)) != {av}:
        return None
    if set(g.successors(score)) != {scale}:
        return None
    if set(g.successors(scale)) != {sm_name}:
        return None
    if set(g.successors(sm_name)) != {av}:
        return None

    return {
        "q": q_node,
        "k": k_node,
        "v": v_node,
        "score": score,
        "scale": scale,
        "softmax": sm_name,
        "av": av,
    }


def _replace_attention_pattern(g, pattern, fused_name):
    """Collapse the 7-node pattern into one fused attention operator."""
    members = [pattern[member] for member in PATTERN_MEMBERS]
    member_set = set(members)

    inputs = set()
    for role in ("q", "k", "v"):
        for pred in g.predecessors(pattern[role]):
            if pred not in member_set:
                inputs.add(pred)

    outputs = []
    for succ in g.successors(pattern["av"]):
        if succ not in member_set and succ not in outputs:
            outputs.append(succ)

    scale_data = g.nodes[pattern["scale"]]
    attrs = {"fused": True, "fused_from": list(members)}
    if "factor" in scale_data:
        attrs["factor"] = scale_data["factor"]

    g.add_node(fused_name, op_type=FUSED_OP_TYPE, **attrs)
    for pred in inputs:
        g.add_edge(pred, fused_name)
    for succ in outputs:
        g.add_edge(fused_name, succ)
    for node in members:
        g.remove_node(node)
    return fused_name
