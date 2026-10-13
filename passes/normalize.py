"""Graph normalization pass.

Goes beyond basic op canonicalization and runs the full normalizer suite:
constant folding, common-subexpression elimination, dead-code elimination
and shape / dtype inference. Returns ``(graph, info)`` where ``info``
carries per-feature counts for the metrics layer.
"""

import re
import time

import numpy as np

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
    "relu": "Relu",
    "identity": "Identity",
    "lineargelu": "LinearGELU",
    "linearrelu": "LinearRelu",
    "fusedaddlayernorm": "FusedAddLayerNorm",
    "fusedscalesoftmax": "FusedScaleSoftmax",
    "fusedattention": "FusedAttention",
    "gather": "Gather",
    "slice": "Slice",
    "reshape": "Reshape",
    "transpose": "Transpose",
    "maskedfill": "MaskedFill",
    "broadcast": "Broadcast",
    "cast": "Cast",
    "rmsnorm": "RMSNorm",
    "silu": "SiLU",
    "linearsilu": "LinearSiLU",
    "fusedaddrmsnorm": "FusedAddRMSNorm",
}

FOLDABLE_OPS = {"Scale", "Add", "Mul", "Sub", "MatMul"}
COMMUTATIVE_OPS = {"Add", "Mul"}
# Ops carrying implicit parameters (e.g. MatMul weight matrices) that the
# (op_type, inputs) signature cannot capture must never be CSE'd - Q/K/V
# projections share inputs but must stay distinct.
PARAMETERIZED_OPS = {"MatMul", "GEMM", "LinearGELU", "LinearRelu",
                     "LinearSiLU", "FusedAttention", "Gather"}

# Semantic attributes that make two otherwise-identical signatures
# distinct (two LayerNorms share inputs but have different gamma/beta;
# two Slices share inputs but cut different ranges).
SEMANTIC_ATTRS = (
    "factor", "start", "stop", "step", "axis", "squeeze", "dims",
    "shape_arg", "fill_value", "where", "eps", "weight", "bias", "value",
)


def _semantic_sig(data):
    """Hash the semantic attrs of a node (empty string when none)."""
    import zlib

    parts = []
    for key in SEMANTIC_ATTRS:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, np.ndarray):
            parts.append(
                (key, value.shape, zlib.crc32(
                    np.ascontiguousarray(value).tobytes()
                ))
            )
        else:
            parts.append((key, repr(value)))
    return repr(parts)
ELEMENTWISE_OPS = {
    "Add", "Mul", "Sub", "Scale", "Softmax", "LayerNorm", "GELU",
    "Relu", "Identity", "FusedAddLayerNorm", "FusedScaleSoftmax",
    "RMSNorm", "SiLU", "FusedAddRMSNorm",
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
        "transposes_removed": 0,
        "transposes_folded": 0,
        "scales_merged": 0,
        "views_removed": 0,
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

    for target in graph.graph.nodes():  # in-edge order = operand order
        for source in graph.graph.predecessors(target):
            normalized.add_dependency(name_map[source], name_map[target])

    # Step 2-6: structural cleanups and analysis.
    info["identities_removed"] = _remove_identities(normalized)
    (info["transposes_removed"],
     info["transposes_folded"]) = _simplify_transposes(normalized)
    info["scales_merged"] = _merge_scale_chains(normalized)
    info["views_removed"] = _eliminate_view_pairs(normalized)
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


def _perm_of(data, ndim=None):
    """Permutation of a Transpose node (``None`` = reverse all axes)."""
    dims = data.get("dims")
    return None if dims is None else tuple(int(d) for d in dims)


def _is_last_two_swap(perm):
    """True for (..., n-1, n-2) permutations, i.e. a matrix transpose."""
    if perm is None or len(perm) < 2:
        return False
    n = len(perm)
    return perm == tuple(range(n - 2)) + (n - 1, n - 2)


def _replace_input(g, node, old, new):
    """Swap one predecessor of ``node`` while keeping operand order.

    MatMul operand roles (lhs/rhs) are encoded by the order of the
    in-edges, so a plain remove+add (which appends) would swap them.
    """
    preds = list(g.predecessors(node))
    attrs = {p: dict(g.edges[p, node]) for p in preds}
    for p in preds:
        g.remove_edge(p, node)
    for p in preds:
        if p == old:
            g.add_edge(new, node)
        elif not g.has_edge(p, node):
            g.add_edge(p, node, **attrs[p])


def _simplify_transposes(graph):
    """Transpose algebra: (A^T)^T -> A and MatMul(A, B^T) -> MatMul^T(A, B).

    * Double transpose: two back-to-back transposes whose permutations
      cancel are removed, deleting two memory-copy kernels.
    * Transpose folding: a matrix transpose feeding the right operand of a
      MatMul is absorbed into the MatMul (``transpose_rhs``) because the
      GEMM kernel can read the operand in transposed layout for free.

    Returns ``(double_transposes_removed, transposes_folded)``.
    """
    g = graph.graph
    removed = folded = 0

    changed = True
    while changed:
        changed = False
        for node in list(g.nodes()):
            data = g.nodes[node]
            if data.get("op_type") != "Transpose" or data.get("is_output"):
                continue
            preds = list(g.predecessors(node))
            if len(preds) != 1:
                continue
            inner = preds[0]
            inner_data = g.nodes[inner]
            if inner_data.get("op_type") != "Transpose":
                continue
            p_in, p_out = _perm_of(inner_data), _perm_of(data)
            if p_in is None and p_out is None:
                cancels = True
            elif p_in is not None and p_out is not None \
                    and len(p_in) == len(p_out):
                cancels = all(p_in[p_out[i]] == i for i in range(len(p_in)))
            else:
                cancels = False
            if not cancels:
                continue
            sources = list(g.predecessors(inner))
            if len(sources) != 1:
                continue
            source = sources[0]
            consumers = list(g.successors(node))
            if any(g.has_edge(source, succ) for succ in consumers):
                continue  # would create a parallel edge
            for succ in consumers:
                _replace_input(g, succ, node, source)
            g.remove_node(node)
            removed += 1
            if g.out_degree(inner) == 0 and not inner_data.get("is_output"):
                g.remove_node(inner)
                removed += 1
            changed = True
            break

    changed = True
    while changed:
        changed = False
        for node in list(g.nodes()):
            data = g.nodes[node]
            if data.get("op_type") != "MatMul" or data.get("transpose_rhs"):
                continue
            preds = list(g.predecessors(node))
            if len(preds) != 2:
                continue
            rhs = preds[1]
            rhs_data = g.nodes[rhs]
            if rhs_data.get("op_type") != "Transpose" \
                    or g.out_degree(rhs) != 1 or rhs_data.get("is_output"):
                continue
            sources = list(g.predecessors(rhs))
            if len(sources) != 1 or not _is_last_two_swap(_perm_of(rhs_data)):
                continue
            if g.has_edge(sources[0], node):
                continue
            _replace_input(g, node, rhs, sources[0])
            g.remove_node(rhs)
            data["transpose_rhs"] = True
            folded += 1
            changed = True
            break
    return removed, folded


def _shape(g, node):
    shape = g.nodes[node].get("shape")
    return tuple(int(d) for d in shape) if shape is not None else None


def _eliminate_view_pairs(graph):
    """View algebra on Reshape nodes (real framework graphs are full of them).

    * ``Reshape(Reshape(x))`` -> one Reshape, or nothing when the result has
      the shape of ``x`` again.
    * ``Reshape -> Linear -> Reshape`` where the first flattens the leading
      dims and the second restores them: both views are dropped because the
      linear layer acts on the last axis only.

    Returns the number of Reshape nodes removed.
    """
    g = graph.graph
    removed = 0
    changed = True
    while changed:
        changed = False
        for node in list(g.nodes()):
            data = g.nodes[node]
            if data.get("op_type") != "Reshape":
                continue
            preds = list(g.predecessors(node))
            if len(preds) != 1:
                continue
            mid = preds[0]
            mid_data = g.nodes[mid]
            out_shape = _shape(g, node)

            # (A) Reshape(Reshape(x))
            if (mid_data.get("op_type") == "Reshape" and g.out_degree(mid) == 1
                    and not mid_data.get("is_output")):
                srcs = list(g.predecessors(mid))
                if len(srcs) != 1:
                    continue
                src = srcs[0]
                if out_shape is not None and out_shape == _shape(g, src) \
                        and not data.get("is_output") \
                        and not any(g.has_edge(src, c) for c in g.successors(node)):
                    for succ in list(g.successors(node)):
                        _replace_input(g, succ, node, src)
                    g.remove_node(node)
                    g.remove_node(mid)
                    removed += 2
                else:
                    _replace_input(g, node, mid, src)
                    g.remove_node(mid)
                    removed += 1
                changed = True
                break

            # (B) Reshape -> Linear -> Reshape
            if (mid_data.get("op_type") in ("GEMM", "MatMul")
                    and mid_data.get("weight") is not None
                    and g.in_degree(mid) == 1 and g.out_degree(mid) == 1
                    and not mid_data.get("is_output")):
                first = list(g.predecessors(mid))[0]
                fd = g.nodes[first]
                if fd.get("op_type") != "Reshape" or g.out_degree(first) != 1 \
                        or g.in_degree(first) != 1 or fd.get("is_output"):
                    continue
                src = list(g.predecessors(first))[0]
                s_in, s_flat = _shape(g, src), _shape(g, first)
                if not (s_in and s_flat and out_shape and len(s_in) >= 3
                        and s_flat == (int(np.prod(s_in[:-1])), s_in[-1])
                        and out_shape[:-1] == s_in[:-1]):
                    continue
                if any(g.has_edge(mid, c) for c in ()):
                    continue
                _replace_input(g, mid, first, src)
                g.remove_node(first)
                for succ in list(g.successors(node)):
                    _replace_input(g, succ, node, mid)
                mid_data["shape"] = out_shape
                if data.get("is_output"):
                    mid_data["is_output"] = True
                g.remove_node(node)
                removed += 2
                changed = True
                break
    return removed


def _merge_scale_chains(graph):
    """Scale(a) -> Scale(b) becomes one Scale(a*b)."""
    g = graph.graph
    merged = 0
    changed = True
    while changed:
        changed = False
        for node in list(g.nodes()):
            data = g.nodes[node]
            if data.get("op_type") != "Scale":
                continue
            preds = list(g.predecessors(node))
            if len(preds) != 1:
                continue
            inner = preds[0]
            inner_data = g.nodes[inner]
            if inner_data.get("op_type") != "Scale" \
                    or g.out_degree(inner) != 1 \
                    or inner_data.get("is_output"):
                continue
            data["factor"] = float(inner_data.get("factor", 1.0)) \
                * float(data.get("factor", 1.0))
            for src in list(g.predecessors(inner)):
                g.add_edge(src, node)
            g.remove_node(inner)
            merged += 1
            changed = True
            break
    return merged


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
                signature = (op_type, tuple(sorted(preds)),
                             _semantic_sig(data))
            else:
                signature = (op_type, tuple(preds), _semantic_sig(data))
            if signature in seen and graph.graph.has_node(node):
                canonical = seen[signature]
                # A DiGraph cannot hold two parallel edges, so merging two
                # operands of the same consumer would silently drop one.
                if (set(graph.graph.successors(node))
                        & set(graph.graph.successors(canonical))):
                    continue
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
            shape = tuple(
                [batch] + list(config.get("input_shape") or [seq, d_model])
            )
        elif op_type == "Constant":
            value = data.get("value")
            if value is not None:
                shape = tuple(value.shape)
        elif op_type == "Transpose":
            if shapes and shapes[0] is not None:
                dims = data.get("dims")
                if dims is None:
                    shape = tuple(reversed(shapes[0]))
                else:
                    shape = tuple(shapes[0][int(d)] for d in dims)
        elif op_type in ("MatMul", "GEMM", "LinearGELU", "LinearRelu",
                         "LinearSiLU"):
            if shapes and all(s is not None for s in shapes):
                shape = _matmul_output_shape(
                    shapes, transpose_rhs=bool(data.get("transpose_rhs", False))
                )
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


def _matmul_output_shape(shapes, transpose_rhs=False):
    """Order-independent matmul shape rule.

    Resolves operand orientation by inner-dim matching (a[-1] == b[-2]
    means a @ b; b[-1] == a[-2] means b @ a); equal-shaped 3-D operands
    with no plain interpretation fall back to the Q @ K^T convention.
    """
    first, second = shapes[0], shapes[-1]
    if len(shapes) < 2 or len(first) < 2 or len(second) < 2:
        return first[:-1] + (first[-1],)
    a, b = first, second
    if transpose_rhs:
        return a[:-1] + (b[-2],)
    if a[-1] == b[-2]:                       # a @ b
        return a[:-1] + (b[-1],)
    if b[-1] == a[-2]:                       # b @ a
        return b[:-1] + (a[-1],)
    if a[-1] == b[-1] and a[-2] != b[-1]:    # a @ b^T (QK^T pattern)
        return a[:-2] + (a[-2], b[-2])
    return a[:-1] + (b[-1],)


# ======================================================================
# PyTorch FX Normalization (Identity rewrites x+0->x, x*1->x, and DCE)
# ======================================================================

def _fx_target_name(node):
    target = node.target
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def _fx_is_constant(value, expected):
    return isinstance(value, (int, float)) and value == expected


def normalize_fx(graph_module):
    """Normalize a PyTorch FX GraphModule by eliminating identities and dead code."""
    graph = graph_module.graph
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        name = _fx_target_name(node)
        if name == "add":
            a, b = node.args
            if _fx_is_constant(a, 0.0):
                node.replace_all_uses_with(b)
                graph.erase_node(node)
            elif _fx_is_constant(b, 0.0):
                node.replace_all_uses_with(a)
                graph.erase_node(node)
        elif name == "mul":
            a, b = node.args
            if _fx_is_constant(a, 1.0):
                node.replace_all_uses_with(b)
                graph.erase_node(node)
            elif _fx_is_constant(b, 1.0):
                node.replace_all_uses_with(a)
                graph.erase_node(node)

    graph.eliminate_dead_code()
    graph.lint()
    graph_module.recompile()
    return graph_module


def normalize(target):
    """Polymorphic normalizer supporting both ComputationGraph and torch.fx GraphModule."""
    if isinstance(target, ComputationGraph):
        return normalize_graph(target)
    return normalize_fx(target)

