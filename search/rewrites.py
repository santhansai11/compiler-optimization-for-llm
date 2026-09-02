"""Symbolic rewrite rules used by the neuro-symbolic search.

Each rule pairs a graph matcher with an applier; the neuro-symbolic driver
enumerates matches and uses the learned GNN scorer to choose which rewrite
to apply (symbolic knowledge + learned guidance).
"""


class RewriteRule:
    def __init__(self, name, matcher, applier, description=""):
        self.name = name
        self.matcher = matcher
        self.applier = applier
        self.description = description

    def find(self, graph):
        """Return a list of match dicts (sub-graph handles)."""
        return self.matcher(graph)

    def apply(self, graph, match):
        """Apply the rule in-place for one match; returns True on success."""
        return self.applier(graph, match)


def _find_identities(graph):
    return [
        {"node": node}
        for node, data in graph.graph.nodes(data=True)
        if data.get("op_type") == "Identity"
    ]


def _apply_identity(graph, match):
    node = match["node"]
    for pred in list(graph.predecessors(node)):
        for succ in list(graph.successors(node)):
            graph.graph.add_edge(pred, succ)
    graph.graph.remove_node(node)
    return True


def _find_foldable(graph):
    matches = []
    for node, data in graph.graph.nodes(data=True):
        op_type = data.get("op_type")
        if op_type not in ("Scale", "Add", "Mul"):
            continue
        preds = list(graph.predecessors(node))
        if preds and all(graph.op_type(p) == "Constant" for p in preds):
            matches.append({"node": node})
    return matches


def _apply_fold(graph, match):
    from passes.normalize import _evaluate

    node = match["node"]
    data = graph.graph.nodes[node]
    preds = list(graph.predecessors(node))
    values = [graph.graph.nodes[p].get("value") for p in preds]
    result = _evaluate(data.get("op_type"), values, data)
    if result is None:
        return False
    data["op_type"] = "Constant"
    data["value"] = result
    for pred in preds:
        if graph.graph.has_edge(pred, node):
            graph.graph.remove_edge(pred, node)
    return True


def _find_nested_scales(graph):
    matches = []
    for node, data in graph.graph.nodes(data=True):
        if data.get("op_type") != "Scale":
            continue
        preds = list(graph.predecessors(node))
        if len(preds) == 1 and graph.op_type(preds[0]) == "Scale":
            matches.append({"node": node, "inner": preds[0]})
    return matches


def _apply_nested_scales(graph, match):
    outer = match["node"]
    inner = match["inner"]
    outer_data = graph.graph.nodes[outer]
    inner_data = graph.graph.nodes[inner]
    outer_data["factor"] = float(inner_data.get("factor", 1.0)) * float(
        outer_data.get("factor", 1.0)
    )
    for pred in list(graph.predecessors(inner)):
        graph.graph.add_edge(pred, outer)
    graph.graph.remove_node(inner)
    return True


def _find_common_subexpressions(graph):
    seen = {}
    matches = []
    for node in graph.topological_order():
        data = graph.graph.nodes[node]
        op_type = data.get("op_type")
        if op_type == "Constant":
            continue
        signature = (op_type, tuple(sorted(graph.predecessors(node))))
        if signature in seen:
            matches.append({"node": node, "canonical": seen[signature]})
        else:
            seen[signature] = node
    return matches


def _apply_cse(graph, match):
    node, canonical = match["node"], match["canonical"]
    for succ in list(graph.successors(node)):
        graph.graph.add_edge(canonical, succ)
    graph.graph.remove_node(node)
    return True


def _find_matmul_add(graph):
    matches = []
    for u, v in list(graph.graph.edges()):
        if graph.op_type(u) == "MatMul" and graph.op_type(v) == "Add":
            if graph.graph.out_degree(u) == 1 and graph.graph.in_degree(v) == 1:
                matches.append({"producer": u, "consumer": v})
    return matches


def _apply_matmul_add(graph, match):
    from passes.merge import _merge_pair

    _merge_pair(graph.graph, match["producer"], match["consumer"], "GEMM")
    return True


RULES = [
    RewriteRule(
        "eliminate-identity",
        _find_identities,
        _apply_identity,
        "Identity(x) => x",
    ),
    RewriteRule(
        "fold-constant",
        _find_foldable,
        _apply_fold,
        "op(Const...) => Const",
    ),
    RewriteRule(
        "combine-nested-scales",
        _find_nested_scales,
        _apply_nested_scales,
        "Scale(Scale(x)) => Scale(x)",
    ),
    RewriteRule(
        "common-subexpression",
        _find_common_subexpressions,
        _apply_cse,
        "op(x...) twice => once",
    ),
    RewriteRule(
        "fuse-matmul-add",
        _find_matmul_add,
        _apply_matmul_add,
        "MatMul + Add => GEMM",
    ),
]
