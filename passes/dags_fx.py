"""Dependency-aware list scheduling for PyTorch FX graphs.

The pass preserves FX execution semantics and records a deterministic schedule
in ``node.meta``. Independent nodes are ordered by remaining critical-path
length, with original graph order as a stable tie breaker.
"""

import time

import torch.fx as fx


def schedule_fx(graph_module):
    """Annotate an FX GraphModule with dependency levels and list schedule."""
    started = time.perf_counter()
    nodes = list(graph_module.graph.nodes)
    position = {node: i for i, node in enumerate(nodes)}
    dependencies = {
        node: [arg for arg in node.all_input_nodes if arg in position]
        for node in nodes
    }
    successors = {node: [] for node in nodes}
    for node, deps in dependencies.items():
        for dep in deps:
            successors[dep].append(node)

    level = {}
    for node in nodes:
        level[node] = max((level[parent] + 1 for parent in dependencies[node]),
                          default=0)

    # Unit weights keep the pass backend-agnostic; callers may provide an
    # estimated latency in node.meta["estimated_latency"] (in microseconds).
    latency = {
        node: max(0.0, float(node.meta.get("estimated_latency", 1.0)))
        for node in nodes
    }
    critical = {}
    for node in reversed(nodes):
        critical[node] = latency[node] + max(
            (critical[child] for child in successors[node]), default=0.0
        )

    ready = [node for node in nodes if not dependencies[node]]
    order = []
    remaining = {node: len(dependencies[node]) for node in nodes}
    while ready:
        ready.sort(key=lambda node: (-critical[node], position[node]))
        node = ready.pop(0)
        order.append(node)
        for child in successors[node]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)

    if len(order) != len(nodes):
        raise ValueError("DAGS requires an acyclic FX graph")

    for index, node in enumerate(order):
        node.meta["schedule_order"] = index
        node.meta["schedule_level"] = level[node]
        node.meta["critical_path_priority"] = critical[node]

    level_count = max(level.values(), default=-1) + 1
    levels = [[] for _ in range(level_count)]
    for node in order:
        levels[level[node]].append(node.name)
    info = {
        "pass": "dags_fx",
        "schedule_order": [node.name for node in order],
        "levels": level_count,
        "level_sizes": [len(items) for items in levels],
        "critical_path": _critical_path(nodes, successors, critical),
        "critical_path_cost": max(critical.values(), default=0.0),
        "duration_s": time.perf_counter() - started,
    }
    graph_module.dags_schedule = info
    return graph_module, info


def _critical_path(nodes, successors, critical):
    sinks = [node for node in nodes if not successors[node]]
    if not sinks:
        return []
    node = max(sinks, key=lambda item: (critical[item], -nodes.index(item)))
    path = [node.name]
    while True:
        parents = [candidate for candidate in nodes
                   if node in successors[candidate]]
        if not parents:
            break
        node = max(parents, key=lambda item: (critical[item], -nodes.index(item)))
        path.append(node.name)
    return list(reversed(path))
