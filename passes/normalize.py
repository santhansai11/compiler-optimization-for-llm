"""Graph normalization pass.

Goes beyond basic op canonicalization and runs the full normalizer suite:
constant folding, common-subexpression elimination, dead-code elimination
and shape / dtype inference. Returns ``(graph, info)`` where ``info``
carries per-feature counts for the metrics layer.
"""

import re
import time

from ir.graph import ComputationGraph


OP_TYPE_MAP = {
    "input": "Input",
    "constant": "Constant",
    "matmul": "MatMul",
    "gemm": "GEMM",
    "add": "Add",
    "mul": "Mul",
    "sub": "Sub",
    "scale": "Scale",
    "softmax": "Softmax",
    "layernorm": "LayerNorm",
    "gelu": "GELU",
    "identity": "Identity",
    "lineargelu": "LinearGELU",
    "fusedaddlayernorm": "FusedAddLayerNorm",
    "fusedscalesoftmax": "FusedScaleSoftmax",
    "fusedattention": "FusedAttention",
}

FOLDABLE_OPS = {"Scale", "Add", "Mul", "Sub", "MatMul"}
COMMUTATIVE_OPS = {"Add", "Mul"}
# Ops carrying implicit parameters (e.g. MatMul weight matrices) that the
# (op_type, inputs) signature cannot capture must never be CSE'd - Q/K/V
# projections share inputs but must stay distinct.
PARAMETERIZED_OPS = {"MatMul", "GEMM", "LinearGELU", "FusedAttention"}
ELEMENTWISE_OPS = {
    "Add", "Mul", "Sub", "Scale", "Softmax", "LayerNorm", "GELU",
    "Identity", "FusedAddLayerNorm", "FusedScaleSoftmax",
}


def normalize_graph(graph):
    """Canonicalize, fold, dedupe, prune and annotate the graph.

    Returns ``(graph, info)``. The original graph is left untouched.
    """
    started = time.perf_counter()
    normalized = ComputationGraph(graph.name)
    normalized.meta.update(graph.meta)

    info = {
        "pass": "normalize",
        "renamed": 0,
        "identities_removed": 0,
        "constants_folded": 0,
        "cse_merges": 0,
        "dead_nodes_removed": 0,
        "shapes_inferred": 0,
    }

    # Step 1: canonicalize op types and node names (preserving attrs).
    name_map = {}
    for node, data in graph.graph.nodes(data=True):
        op_type = data.get("op_type", "Operation")
        canonical_type = OP_TYPE_MAP.get(op_type.lower(), op_type)
        canonical_name = re.sub(r"[^a-zA-Z0-9]+", "_", node).strip("_")
        if canonical_name != node:
            info["renamed"] += 1
        name_map[node] = canonical_name
        attrs = {key: value for key, value in data.items() if key != "op_type"}
        normalized.add_operation(canonical_name, canonical_type, **attrs)

    for source, target in graph.graph.edges():
        normalized.add_dependency(name_map[source], name_map[target])

    # Step 2-6: structural cleanups and analysis.
    info["identities_removed"] = _remove_identities(normalized)
    info["constants_folded"] = _fold_constants(normalized)
    info["cse_merges"] = _eliminate_common_subexpressions(normalized)
    info["dead_nodes_removed"] = _eliminate_dead_code(normalized)
    info["shapes_inferred"] = _infer_shapes(normalized)

    info["nodes_after"] = normalized.node_count()
    info["duration_s"] = time.perf_counter() - started
    normalized.meta["normalize"] = info
    return normalized, info


def _remove_identities(graph):
    """Rewire Identity nodes away and remove them."""
    removed = 0
    identity_nodes = [
        node
        for node, data in graph.graph.nodes(data=True)
        if data.get("op_type") == "Identity"
    ]
    for node in identity_nodes:
        preds = list(graph.graph.predecessors(node))
        succs = list(graph.graph.successors(node))
        if not preds:
            continue
        for pred in preds:
            for succ in succs:
                graph.graph.add_edge(pred, succ)
        graph.graph.remove_node(node)
        removed += 1
    return removed


def _evaluate(op_type, values, data):
    """Numerically evaluate an op over constant inputs (NumPy, float32).

    Also used by ``search/rewrites.py`` for its fold-constant rule.
    """
    import numpy as np

    if op_type == "Scale":
        factor = float(data.get("factor", 1.0))
        return np.asarray(values[0] * np.float32(factor), dtype=np.float32)
    if op_type == "Add":
        out = values[0]
        for value in values[1:]:
            out = out + value
        return np.asarray(out, dtype=np.float32)
    if op_type == "Mul":
        out = values[0]
        for value in values[1:]:
            out = out * value
        return np.asarray(out, dtype=np.float32)
    if op_type == "Sub":
        return np.asarray(values[0] - values[1], dtype=np.float32)
    if op_type == "MatMul":
        out = values[0]
        for value in values[1:]:
            out = out @ value
        return np.asarray(out, dtype=np.float32)
    return None


def _fold_constants(graph):
    """Fold ops whose inputs are all constants into Constant nodes.

    Runs to a fixpoint so chains (e.g. Scale(Constant)) collapse fully.
    Input edges of folded nodes are detached so orphaned constants can be
    swept by dead-code elimination.
    """
    folded = 0
    changed = True
    while changed:
        changed = False
        for node, data in list(graph.graph.nodes(data=True)):
            op_type = data.get("op_type")
            if op_type not in FOLDABLE_OPS:
                continue
            preds = list(graph.graph.predecessors(node))
            if not preds:
                continue
            if not all(graph.op_type(p) == "Constant" for p in preds):
                continue
            values = [graph.graph.nodes[p].get("value") for p in preds]
            if any(value is None for value in values):
                continue
            result = _evaluate(op_type, values, data)
            if result is None:
                continue
            data["op_type"] = "Constant"
            data["value"] = result
            for pred in preds:
                if graph.graph.has_edge(pred, node):
                    graph.graph.remove_edge(pred, node)
            folded += 1
            changed = True
    return folded


def _eliminate_common_subexpressions(graph):
    """Merge nodes with identical (op_type, inputs) into one canonical op."""
    merged = 0
    changed = True
    while changed:
        changed = False
        seen = {}
        for node in list(graph.topological_order()):
            data = graph.graph.nodes[node]
            op_type = data.get("op_type")
            if op_type == "Constant" or op_type in PARAMETERIZED_OPS:
                continue
            preds = list(graph.graph.predecessors(node))
            if op_type in COMMUTATIVE_OPS:
                signature = (op_type, tuple(sorted(preds)))
            else:
                signature = (op_type, tuple(preds))
            if signature in seen and graph.graph.has_node(node):
                canonical = seen[signature]
                for succ in list(graph.graph.successors(node)):
                    graph.graph.add_edge(canonical, succ)
                graph.graph.remove_node(node)
                merged += 1
                changed = True
                break
            seen[signature] = node
    return merged


def _eliminate_dead_code(graph):
    """Remove nodes no output depends on (iterative sink pruning).

    Only active when at least one node is marked ``is_output`` so graphs
    without explicit outputs are never damaged.
    """
    if not any(
        data.get("is_output") for _, data in graph.graph.nodes(data=True)
    ):
        return 0
    removed = 0
    changed = True
    while changed:
        changed = False
        for node in list(graph.graph.nodes()):
            if graph.graph.out_degree(node) != 0:
                continue
            if graph.graph.nodes[node].get("is_output"):
                continue
            graph.graph.remove_node(node)
            removed += 1
            changed = True
    return removed


def _infer_shapes(graph):
    """Propagate concrete shapes/dtypes through the DAG.

    MatMul handles the Q@K^T case (equal last dims, differing second-to-last)
    and the batched case; elementwise ops take the broadcast winner (the
    highest-rank input shape).
    """
    config = graph.meta.get("config", {})
    batch = int(config.get("batch", 4))
    seq = int(config.get("seq", 32))
    d_model = int(config.get("d_model", 64))

    inferred = 0
    for node in graph.topological_order():
        data = graph.graph.nodes[node]
        if data.get("shape") is not None:
            inferred += 1
            continue
        op_type = data.get("op_type")
        preds = graph.predecessors(node)
        shapes = [graph.graph.nodes[p].get("shape") for p in preds]
        shape = None

        if op_type == "Input":
            shape = (batch, seq, d_model)
        elif op_type == "Constant":
            value = data.get("value")
            if value is not None:
                shape = tuple(value.shape)
        elif op_type in ("MatMul", "GEMM", "LinearGELU"):
            if shapes and all(s is not None for s in shapes):
                shape = _matmul_output_shape(shapes)
        elif op_type == "FusedAttention":
            if shapes and shapes[0] is not None:
                shape = shapes[0]
        elif op_type in ELEMENTWISE_OPS:
            candidates = [s for s in shapes if s is not None]
            if candidates:
                shape = max(candidates, key=len)

        if shape is not None:
            data["shape"] = tuple(int(dim) for dim in shape)
            data.setdefault("dtype", "float32")
            inferred += 1
    return inferred


def _matmul_output_shape(shapes):
    """Order-independent matmul shape rule.

    Resolves operand orientation by inner-dim matching (a[-1] == b[-2]
    means a @ b; b[-1] == a[-2] means b @ a); equal-shaped 3-D operands
    with no plain interpretation fall back to the Q @ K^T convention.
    """
    first, second = shapes[0], shapes[-1]
    if len(shapes) < 2 or len(first) < 2 or len(second) < 2:
        return first[:-1] + (first[-1],)
    a, b = first, second
    if a[-1] == b[-2]:                       # a @ b
        return a[:-1] + (b[-1],)
    if b[-1] == a[-2]:                       # b @ a
        return b[:-1] + (a[-1],)
    if a[-1] == b[-1] and a[-2] != b[-1]:    # a @ b^T (QK^T pattern)
        return a[:-2] + (a[-2], b[-2])
    return a[:-1] + (b[-1],)
