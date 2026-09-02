"""Semantic operator merging pass (pattern-based kernel fusion)."""

import time
from collections import Counter


FUSION_PATTERNS = {
    ("MatMul", "Add"): "GEMM",
    ("GEMM", "GELU"): "LinearGELU",
    ("MatMul", "GELU"): "LinearGELU",
    ("Add", "LayerNorm"): "FusedAddLayerNorm",
    ("Scale", "Softmax"): "FusedScaleSoftmax",
}

BIAS_FUSION_PAIR = ("MatMul", "Add")
PROTECTED_OPS = {"Input", "Constant"}


def merge_operators(graph):
    """Fuse producer/consumer operator pairs into single fused kernels.

    Supported fusions (classic compiler patterns):
      * MatMul + Add         -> GEMM      (incl. constant-bias folding)
      * GEMM / MatMul + GELU -> LinearGELU
      * Add + LayerNorm      -> FusedAddLayerNorm
      * Scale + Softmax      -> FusedScaleSoftmax

    Returns ``(graph, info)``; ``info['nodes_removed']`` feeds OMR.
    """
    started = time.perf_counter()
    optimized = graph.copy()
    g = optimized.graph

    fused_types = Counter()
    details = []
    nodes_removed = 0
    bias_folds = 0

    changed = True
    while changed:
        changed = False
        for u, v in list(g.edges()):
            producer = g.nodes[u].get("op_type")
            consumer = g.nodes[v].get("op_type")
            fused_type = FUSION_PATTERNS.get((producer, consumer))
            if fused_type is None or producer in PROTECTED_OPS:
                continue
            if g.out_degree(u) != 1:
                continue

            extra_preds = [p for p in g.predecessors(v) if p != u]
            bias_value = None

            if (producer, consumer) == BIAS_FUSION_PAIR:
                # GEMM fusion may absorb a single constant bias input.
                if len(extra_preds) > 1:
                    continue
                if extra_preds:
                    const_node = extra_preds[0]
                    if g.nodes[const_node].get("op_type") != "Constant":
                        continue
                    bias_value = g.nodes[const_node].get("value")
            elif extra_preds:
                continue

            _merge_pair(g, u, v, fused_type, bias_value, extra_preds)
            fused_types[fused_type] += 1
            nodes_removed += 1
            if bias_value is not None:
                bias_folds += 1
            details.append(
                f"{producer}({u}) + {consumer}({v}) -> {fused_type}({v})"
            )
            changed = True
            break

    info = {
        "pass": "semantic_merging",
        "fusions": sum(fused_types.values()),
        "fused_types": dict(fused_types),
        "nodes_removed": nodes_removed,
        "bias_folds": bias_folds,
        "details": details,
        "nodes_after": optimized.node_count(),
        "duration_s": time.perf_counter() - started,
    }
    optimized.meta["semantic_merging"] = info
    return optimized, info


def _merge_pair(g, producer, consumer, fused_type, bias_value=None, extra_preds=()):
    """Fold ``producer`` into ``consumer``; the consumer keeps its name.

    The consumer inherits the producer's inputs and semantic attributes
    (e.g. a Scale ``factor``), accumulates ``fused_from`` provenance (used
    by the executor to derive the same weights as the original ops), and
    optionally absorbs a folded constant bias.
    """
    consumer_data = g.nodes[consumer]
    producer_data = g.nodes[producer]

    provenance = list(producer_data.get("fused_from", [producer]))
    provenance += list(consumer_data.get("fused_from", [consumer]))
    consumer_data["fused_from"] = provenance
    consumer_data["fused"] = True
    consumer_data["op_type"] = fused_type

    # Carry semantic attributes forward (e.g. Scale factor, GEMM bias).
    for key, value in list(producer_data.items()):
        if key in ("op_type", "fused_from", "fused"):
            continue
        consumer_data.setdefault(key, value)

    # Re-point the producer's inputs into the fused consumer.
    for pred in list(g.predecessors(producer)):
        g.add_edge(pred, consumer)
    g.remove_node(producer)

    if bias_value is not None:
        consumer_data["bias"] = bias_value
        for const_node in list(extra_preds):
            if not g.has_node(const_node):
                continue
            if g.has_edge(const_node, consumer):
                g.remove_edge(const_node, consumer)
            if (
                g.out_degree(const_node) == 0
                and not g.nodes[const_node].get("is_output")
            ):
                g.remove_node(const_node)
