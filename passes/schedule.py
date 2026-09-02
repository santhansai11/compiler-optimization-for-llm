"""Dependency-Aware Graph Scheduling (DAGS).

Computes dependency levels, critical-path priorities and a deterministic
list schedule for the data-flow graph; annotates every node with its
``schedule_order`` / ``schedule_level`` and stores makespan statistics in
``graph.meta['schedule']`` for the cost model.
"""

import time

import networkx as nx

from utils.cost_model import get_op_latency


def schedule_graph(graph):
    """List-schedule the DAG by level and critical-path priority."""
    started = time.perf_counter()
    scheduled = graph.copy()
    g = scheduled.graph

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("DAGS requires an acyclic computation graph")

    latency = {
        node: get_op_latency(data.get("op_type"))
        for node, data in g.nodes(data=True)
    }
    order = list(nx.topological_sort(g))

    # Dependency levels: longest path from any source.
    level = {}
    for node in order:
        level[node] = max(
            (level[pred] for pred in g.predecessors(node)), default=-1
        ) + 1

    # Critical-path priority: longest weighted path to any sink.
    critical = {}
    for node in reversed(order):
        critical[node] = latency[node] + max(
            (critical[succ] for succ in g.successors(node)), default=0.0
        )

    schedule_order = sorted(order, key=lambda n: (level[n], -critical[n], n))
    for position, node in enumerate(schedule_order):
        g.nodes[node]["schedule_order"] = position
        g.nodes[node]["schedule_level"] = level[node]

    num_levels = max(level.values()) + 1 if level else 0
    level_sizes = [0] * num_levels
    level_latency = [0.0] * num_levels
    for node in schedule_order:
        level_sizes[level[node]] += 1
        level_latency[level[node]] = max(
            level_latency[level[node]], latency[node]
        )

    info = {
        "pass": "dags",
        "schedule_order": list(schedule_order),
        "levels": num_levels,
        "level_sizes": level_sizes,
        "level_latency_max_us": level_latency,
        "critical_path": _trace_critical_path(g, order, level, critical),
        "critical_path_us": max(critical.values(), default=0.0),
        "makespan_parallel_us": sum(level_latency),
        "sequential_compute_us": sum(latency.values()),
        "avg_parallelism": (len(order) / num_levels) if num_levels else 0.0,
        "duration_s": time.perf_counter() - started,
    }
    scheduled.meta["schedule"] = info
    return scheduled, info


def _trace_critical_path(g, order, level, critical):
    """Walk the critical path from its highest-priority source to a sink."""
    sources = [node for node in order if level[node] == 0]
    if not sources:
        return []
    node = max(sources, key=lambda n: (critical[n], n))
    path = [node]
    while True:
        succs = list(g.successors(node))
        if not succs:
            return path
        node = max(succs, key=lambda n: (critical[n], n))
        path.append(node)
