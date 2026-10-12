"""Attention graph canonicalization pass.

Detects the classic attention sub-graph

    Q/K/V projections -> QK^T -> Scale -> Softmax -> (A)V matmul

and replaces it with a single canonical ``FusedAttention`` operator - the
pattern production runtimes (FlashAttention / SDPA kernels) execute.
"""

import os
import time

import numpy as np


FUSED_OP_TYPE = "FusedAttention"
PATTERN_MEMBERS = ("q", "k", "v", "score", "scale", "softmax", "av")


def _dbg(msg):
    if os.environ.get("MATCH_DEBUG"):
        import sys

        print("match:", msg, file=sys.stderr)


def _mask_array(g, start):
    """Find the mask Constant behind Slice/Reshape/Transpose/Broadcast ops."""
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        value = g.nodes[node].get("value")
        if isinstance(value, np.ndarray) and value.size > 1:
            return value
        for pred in g.predecessors(node):
            if g.nodes[pred].get("op_type") in (
                "Slice", "Reshape", "Transpose", "Broadcast"
            ):
                stack.append(pred)
    return None


def _materialize_mask(g, mask_node):
    """Replay the view chain from a Constant to ``mask_node`` numerically."""
    import networkx as nx

    # gather the view ops between the mask node and any constant source
    view_ops = ("Slice", "Reshape", "Transpose", "Broadcast")
    between = set()
    stack = [mask_node]
    while stack:
        n = stack.pop()
        for pred in g.predecessors(n):
            if pred in between:
                continue
            value = g.nodes[pred].get("value")
            if isinstance(value, np.ndarray) and value.size > 1:
                continue  # do not walk past a constant source
            if g.nodes[pred].get("op_type") in view_ops:
                between.add(pred)
                stack.append(pred)

    order = list(nx.topological_sort(g))
    target = None
    for node in order:
        if node in between:
            continue
        value = g.nodes[node].get("value")
        if isinstance(value, np.ndarray) and value.size > 1 and set(
            g.successors(node)
        ) & between:
            target = node
            break
    if target is None:
        return None

    arr = np.asarray(g.nodes[target]["value"], dtype=np.float32)
    for node in order:
        if node not in between:
            continue
        data = g.nodes[node]
        op = data.get("op_type")
        try:
            if op == "Slice":
                axis = int(data.get("axis", -1))
                if axis < 0:
                    axis += arr.ndim
                start = int(data.get("start", 0))
                stop = data.get("stop")
                step = int(data.get("step", 1))
                index = [slice(None)] * arr.ndim
                index[axis] = slice(
                    start, None if stop is None else int(stop), step
                )
                arr = arr[tuple(index)]
                if data.get("squeeze"):
                    arr = np.squeeze(arr, axis=axis)
            elif op == "Reshape":
                shape = data.get("shape_arg") or data.get("shape")
                if shape is None:
                    return None
                arr = arr.reshape(tuple(int(d) for d in shape))
            elif op == "Transpose":
                dims = data.get("dims")
                arr = np.transpose(
                    arr,
                    tuple(int(d) for d in dims) if dims is not None else None,
                )
            elif op == "Broadcast":
                arr = np.broadcast_to(
                    arr, tuple(int(d) for d in data.get("target_shape", ()))
                )
            else:
                return None
        except Exception:
            return None
    return np.ascontiguousarray(arr)


def _mask_is_causal(g, mask_node):
    """Return the causal-mask kind ('where' / 'add') or None.

    ``where``  — a boolean/0-1 mask selects the lower triangle (keep) and
                 the fused kernel re-creates it with ``np.tril``.
    ``add``    — an additive 0 / -inf (or 0 / -1e9) mask array; the fused
                 kernel re-applies the stored array.
    """
    data = g.nodes[mask_node]
    op = data.get("op_type")
    consts = [
        g.nodes[p]
        for p in g.predecessors(mask_node)
        if g.nodes[p].get("op_type") == "Constant"
    ]
    arrays = [
        c.get("value")
        for c in consts
        if isinstance(c.get("value"), np.ndarray) and c["value"].size > 1
    ]
    if op in ("Add", "Sub"):
        arr = np.asarray(arrays[0]) if arrays else _materialize_mask(
            g, mask_node
        )
        if arr is None or arr.ndim < 2:
            return None
        m = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
        upper = np.triu(np.ones(m.shape, dtype=bool), 1)
        if np.array_equal(m != 0, upper):
            return "add"
        return None
    # MaskedFill: mask operand is a (possibly sliced/expanded) Constant
    # array; the fill value is an attribute (masked_fill) or a scalar
    # constant input (torch.where).
    arr = arrays[0] if arrays else _materialize_mask(g, mask_node)
    if arr is None or np.asarray(arr).ndim < 2:
        return None
    arr = np.asarray(arr)
    m = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    tril = np.tril(np.ones(m.shape, dtype=bool))
    truthy = m != 0
    if np.array_equal(truthy, tril) or np.array_equal(truthy, ~tril):
        return "where"
    return None



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
    """Match an attention cluster around a Softmax node.

    Handles both the classic demo wiring and the wiring produced by real
    framework exports (GPT-2 style), where a causal-mask node sits between
    the Scale and the Softmax and Q/K/V are view chains (slice/reshape/
    permute, optionally with a real weight MatMul) over one shared input.
    """
    preds = list(g.predecessors(sm_name))
    if len(preds) != 1:
        return None
    mid = preds[0]
    mask_node = None
    scale = mid
    mid_op = g.nodes[mid].get("op_type")

    if mid_op != "Scale":
        if mid_op == "MaskedFill":
            # where(cond, x, fill) / masked_fill(x, mask, value): the
            # scale-path operand is the Scale pred; the others are the
            # mask/fill operands (constants or views over constants).
            scale_preds = [
                p
                for p in g.predecessors(mid)
                if g.nodes[p].get("op_type") == "Scale"
            ]
            if len(scale_preds) != 1:
                return None
            mask_node = mid
            scale = scale_preds[0]
        elif mid_op in ("Add", "Sub"):
            mp = list(g.predecessors(mid))
            consts = [
                p for p in mp if g.nodes[p].get("op_type") == "Constant"
            ]
            if len(mp) != 2 or len(consts) != 1:
                return None
            mask_node = mid
            scale = next(p for p in mp if p != consts[0])
        else:
            return None
        if set(g.successors(mid)) != {sm_name}:
            return None

    if g.nodes[scale].get("op_type") != "Scale" or g.in_degree(scale) != 1:
        return None

    # The score MatMul may sit behind view ops (frameworks decompose a
    # batched matmul into mm + view); walk up over single-use views.
    score_chain = []
    node = list(g.predecessors(scale))[0]
    while (
        g.nodes[node].get("op_type") in ("Reshape", "Transpose", "Broadcast")
        and g.in_degree(node) == 1
        and g.out_degree(node) == 1
    ):
        score_chain.append(node)
        node = list(g.predecessors(node))[0]
    score = node
    if g.nodes[score].get("op_type") != "MatMul" or g.in_degree(score) != 2:
        return None
    chain_from_score = list(reversed(score_chain))
    expected = chain_from_score[0] if chain_from_score else scale
    if set(g.successors(score)) != {expected}:
        return None
    for a, b in zip(chain_from_score, chain_from_score[1:]):
        if set(g.successors(a)) != {b}:
            return None
    if chain_from_score and set(g.successors(chain_from_score[-1])) != {scale}:
        return None

    q_node, k_node = list(g.predecessors(score))

    # Same tolerance on the AV side: walk down over view ops.
    sm_succs = list(g.successors(sm_name))
    if len(sm_succs) != 1:
        return None
    av_chain = []
    node = sm_succs[0]
    while (
        g.nodes[node].get("op_type") in ("Reshape", "Transpose", "Broadcast")
        and g.in_degree(node) == 1
        and g.out_degree(node) == 1
    ):
        av_chain.append(node)
        node = list(g.successors(node))[0]
    av = node
    if g.nodes[av].get("op_type") != "MatMul" or g.in_degree(av) != 2:
        return None
    expected_pred = av_chain[-1] if av_chain else sm_name
    v_node = next(p for p in g.predecessors(av) if p != expected_pred)

    # Strict wiring: every pattern member is used only inside the pattern.
    if set(g.successors(q_node)) != {score}:
        return None
    if set(g.successors(k_node)) != {score}:
        return None
    if set(g.successors(v_node)) != {av}:
        return None
    if set(g.successors(scale)) != {mask_node if mask_node is not None
                                    else sm_name}:
        return None
    for a, b in zip(av_chain, av_chain[1:]):
        if set(g.successors(a)) != {b}:
            return None
    if av_chain and set(g.successors(av_chain[-1])) != {av}:
        return None
    if mask_node is not None:
        kind = _mask_is_causal(g, mask_node)
        if not kind:
            _dbg("fail: mask not causal")
            return None

    pattern = {
        "q": q_node,
        "k": k_node,
        "v": v_node,
        "score": score,
        "scale": scale,
        "softmax": sm_name,
        "av": av,
        "mask": mask_node,
    }

    # Role chains: Q/K/V must reduce to one shared producer for the fused
    # operator to recompute them from a single input.
    members = {pattern[m] for m in PATTERN_MEMBERS if pattern.get(m)}
    inputs = set()
    absorbed = set(score_chain) | set(av_chain)
    for role in ("q", "k", "v"):
        steps, terminal = _walk_role(g, pattern[role])
        if terminal == pattern[role]:
            inputs.update(g.predecessors(pattern[role]))
        else:
            inputs.add(terminal)
        absorbed.update(_chain_nodes(g, pattern[role], terminal))
        pattern[f"{role}_transform"] = steps
    if len(inputs) != 1:
        _dbg(f"fail: inputs={inputs}")
        return None
    # The three roles must be either ALL view-chains (real exports) or ALL
    # plain derived-weight MatMuls (the demo graphs) - never mixed.
    empty = [not pattern[f"{role}_transform"] for role in ("q", "k", "v")]
    if any(empty) != all(empty):
        _dbg("fail: mixed role transforms")
        return None
    pattern["input"] = next(iter(inputs))
    pattern["absorbed"] = absorbed - members
    if os.environ.get("MATCH_DEBUG"):
        import sys

        print("matcher OK:", pattern.get("input"), "absorbed:",
              len(pattern["absorbed"]), file=sys.stderr)
    return pattern


def _walk_role(g, start):
    """Walk view ops / weighted matmuls back from a Q/K/V node.

    Returns ``(steps, terminal)`` where ``steps`` is a list the executor
    replays from the shared input and ``terminal`` is the node the chain
    bottoms out at (or ``start`` itself for the derived-weight demo case).
    """
    steps = []
    node = start
    while True:
        data = g.nodes[node]
        op = data.get("op_type")
        if op in ("Slice", "Reshape", "Transpose", "Broadcast") and (
            g.out_degree(node) == 1 and g.in_degree(node) == 1
        ):
            if op == "Slice":
                steps.append((
                    "slice",
                    data.get("start", 0),
                    data.get("stop"),
                    data.get("step", 1),
                    data.get("axis", -1),
                    bool(data.get("squeeze")),
                ))
            elif op == "Reshape":
                steps.append(("view", data.get("shape_arg")))
            elif op == "Broadcast":
                steps.append(("broadcast", data.get("target_shape")))
            else:
                steps.append(("perm", data.get("dims")))
            node = list(g.predecessors(node))[0]
            continue
        if (
            op in ("MatMul", "GEMM")
            and data.get("weight") is not None
            and g.out_degree(node) == 1
            and g.in_degree(node) == 1
        ):
            steps.append((
                "mm",
                data.get("weight"),
                data.get("bias"),
            ))
            node = list(g.predecessors(node))[0]
            continue
        break
    steps.reverse()
    return steps, node


def _chain_nodes(g, start, terminal):
    """Nodes on the chain from ``start`` back to (excluding) ``terminal``."""
    chain = []
    node = start
    while node != terminal:
        chain.append(node)
        preds = list(g.predecessors(node))
        if not preds:
            break
        node = preds[0]
    return chain



def _replace_attention_pattern(g, pattern, fused_name):
    """Collapse the matched attention cluster into one fused operator."""
    members = [pattern[member] for member in PATTERN_MEMBERS]
    member_set = set(members)
    absorbed = pattern.get("absorbed") or set()
    remove = member_set | absorbed

    scale_data = g.nodes[pattern["scale"]]
    attrs = {
        "fused": True,
        "fused_from": members + sorted(absorbed),
    }
    if "factor" in scale_data:
        attrs["factor"] = scale_data["factor"]

    # Q/K/V view chains (real exports): the fused operator replays them.
    if any(pattern.get(f"{role}_transform") for role in ("q", "k", "v")):
        for role in ("q", "k", "v"):
            attrs[f"{role}_transform"] = pattern.get(f"{role}_transform") or []

    # Causal mask (real exports): either an additive 0/-inf array or a
    # where-style tril selection.
    mask_node = pattern.get("mask")
    if mask_node is not None:
        kind = _mask_is_causal(g, mask_node)
        attrs["mask_kind"] = kind
        if kind == "add":
            for pred in g.predecessors(mask_node):
                value = g.nodes[pred].get("value")
                if isinstance(value, np.ndarray) and value.size > 1:
                    attrs["mask_array"] = value
        else:
            fill = g.nodes[mask_node].get("fill_value")
            if fill is None:
                for pred in g.predecessors(mask_node):
                    value = g.nodes[pred].get("value")
                    if isinstance(value, (int, float, np.number)):
                        fill = float(value)
                    elif isinstance(value, np.ndarray) and value.size == 1:
                        fill = float(np.min(value))
            attrs["mask_fill"] = (
                float(fill) if fill is not None else float("-inf")
            )

    outputs = []
    for succ in g.successors(pattern["av"]):
        if succ not in remove and succ not in outputs:
            outputs.append(succ)

    g.add_node(fused_name, op_type=FUSED_OP_TYPE, **attrs)
    g.add_edge(pattern["input"], fused_name)
    for succ in outputs:
        g.add_edge(fused_name, succ)
    for node in remove:
        if g.has_node(node):
            g.remove_node(node)
    return fused_name
