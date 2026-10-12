"""Graph partitioning and hypergraph partitioning.

Both modes k-way partition the computation graph with community detection
(greedy modularity) over a connection graph, then report cut / balance
statistics. Hypergraph mode additionally builds fan-in / fan-out nets and
counts cut hyperedges and communication volume.
"""

import time
from itertools import combinations

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def partition_graph(graph, num_partitions=2):
    """K-way partition of the data-flow graph. Returns ``(graph, info)``."""
    return _partition(graph, num_partitions, mode="graph")


def partition_hypergraph(graph, num_partitions=2):
    """K-way partition of the hypergraph view. Returns ``(graph, info)``."""
    return _partition(graph, num_partitions, mode="hypergraph")


def partition_latency_aware(graph, num_partitions=2):
    """Latency-aware k-way placement (HEFT-style list scheduling).

    Ops are visited in decreasing upward rank (longest remaining path,
    including transfer cost) and each is placed on the device where it can
    *finish earliest* given device availability and cross-device transfers.
    Unlike community-based cuts this pass optimises makespan directly, so it
    keeps a chain on one device and only spreads genuinely parallel work.
    Also records the visit order as ``schedule_order``.
    """
    from utils.cost_model import COMM_US, LAUNCH_OVERHEAD_US, get_op_latency

    started = time.perf_counter()
    work = graph.copy()
    g = work.graph
    k = max(1, int(num_partitions))
    lat = {n: get_op_latency(d.get("op_type")) + LAUNCH_OVERHEAD_US
           for n, d in g.nodes(data=True)}
    rank = {}
    for n in reversed(work.topological_order()):
        rank[n] = lat[n] + max(
            (COMM_US + rank[s] for s in g.successors(n)), default=0.0)
    order = sorted(g.nodes(), key=lambda n: (-rank[n], n))

    avail = [0.0] * k
    device, finish = {}, {}
    for position, n in enumerate(order):
        best = None
        for d in range(k):
            ready = max((finish[p] + (COMM_US if device[p] != d else 0.0)
                         for p in g.predecessors(n)), default=0.0)
            eft = max(avail[d], ready) + lat[n]
            if best is None or eft < best[0] - 1e-9:
                best = (eft, d)
        finish[n], device[n] = best
        avail[best[1]] = best[0]
        g.nodes[n]["partition"] = best[1]
        g.nodes[n]["schedule_order"] = position

    used = sorted(set(device.values()))
    remap = {d: i for i, d in enumerate(used)}
    for n in g.nodes():
        g.nodes[n]["partition"] = remap[device[n]]
    sizes = [sum(1 for n in g.nodes() if g.nodes[n]["partition"] == i)
             for i in range(len(used))]
    info = {
        "pass": "latency_aware_partitioning",
        "mode": "latency",
        "requested_partitions": k,
        "partitions": len(used),
        "partition_sizes": sizes,
        "balance_ratio": max(sizes) / min(sizes) if sizes and min(sizes) else 1.0,
        "cut_edges": sum(1 for u, v in g.edges()
                         if g.nodes[u]["partition"] != g.nodes[v]["partition"]),
        "estimated_makespan_us": max(finish.values(), default=0.0),
        "duration_s": time.perf_counter() - started,
    }
    work.meta["partition"] = info
    return work, info


def _partition(graph, num_partitions, mode):
    started = time.perf_counter()
    work = graph.copy()
    target = max(1, min(int(num_partitions), max(1, work.node_count())))

    if work.node_count() == 0:
        info = {
            "pass": (
                "hypergraph_partitioning"
                if mode == "hypergraph"
                else "graph_partitioning"
            ),
            "mode": mode,
            "requested_partitions": int(num_partitions),
            "partitions": 0,
            "partition_sizes": [],
            "duration_s": time.perf_counter() - started,
        }
        work.meta["partition"] = info
        return work, info

    connection = _connection_graph(work, mode == "hypergraph")
    communities = greedy_modularity_communities(connection, weight="weight")
    parts = [set(community) for community in communities]
    parts = _adjust_part_count(connection, parts, target)

    covered = set().union(*parts) if parts else set()
    missing = set(connection.nodes()) - covered
    if missing and parts:
        parts[0] = parts[0] | missing
    elif missing:
        parts = [missing]

    parts.sort(key=lambda part: sorted(part)[0])

    assignment = {}
    for part_id, part in enumerate(parts):
        for node in part:
            assignment[node] = part_id
            work.graph.nodes[node]["partition"] = part_id

    hyperedges = _hyperedges(work, mode == "hypergraph")

    if mode == "graph":
        cut_stats = {
            "cut_edges": sum(
                1
                for u, v in work.graph.edges()
                if assignment[u] != assignment[v]
            ),
        }
    else:
        cut_stats = {
            "cut_hyperedges": sum(
                1
                for net in hyperedges
                if len({assignment[n] for n in net}) > 1
            ),
            "communication_volume": sum(
                len({assignment[n] for n in net}) - 1
                for net in hyperedges
            ),
        }

    sizes = [len(part) for part in parts]
    info = {
        "pass": (
            "hypergraph_partitioning"
            if mode == "hypergraph"
            else "graph_partitioning"
        ),
        "mode": mode,
        "requested_partitions": int(num_partitions),
        "partitions": len(parts),
        "partition_sizes": sizes,
        "balance_ratio": (
            max(sizes) / min(sizes)
            if len(sizes) > 1 and min(sizes) > 0
            else 1.0
        ),
        "hyperedges": len(hyperedges),
        **cut_stats,
        "duration_s": time.perf_counter() - started,
    }
    work.meta["partition"] = info
    return work, info


def _connection_graph(graph, include_hyper_nets):
    """2-section connection graph; hypergraph mode adds net cliques."""
    conn = nx.Graph()
    conn.add_nodes_from(graph.graph.nodes())
    for u, v in graph.graph.edges():
        if conn.has_edge(u, v):
            conn[u][v]["weight"] += 1
        else:
            conn.add_edge(u, v, weight=1)
    if include_hyper_nets:
        for net in _hyperedges(graph, True):
            for a, b in combinations(sorted(net), 2):
                if conn.has_edge(a, b):
                    conn[a][b]["weight"] += 1
                else:
                    conn.add_edge(a, b, weight=1)
    return conn


def _hyperedges(graph, include_fanout):
    """Nets: multi-input fan-in cones (+ fan-out sets in hypergraph mode)."""
    nets = []
    for node in graph.graph.nodes():
        preds = list(graph.graph.predecessors(node))
        if len(preds) >= 2:
            nets.append(frozenset(preds))
        if include_fanout:
            succs = list(graph.graph.successors(node))
            if len(succs) >= 2:
                nets.append(frozenset(succs))
    return nets


def _adjust_part_count(conn, parts, target):
    """Merge smallest / split largest communities until exactly k remain."""
    parts = [set(part) for part in parts if part]
    while len(parts) > target:
        parts.sort(key=len)
        merged = parts.pop(0) | parts.pop(0)
        parts.append(merged)
    while len(parts) < target:
        parts.sort(key=len, reverse=True)
        biggest = parts.pop(0)
        if len(biggest) < 2:
            parts.append(biggest)
            break
        half = _bfs_half(conn, biggest)
        parts.append(half)
        parts.append(biggest - half)
    return parts


def _bfs_half(conn, nodes):
    """Deterministic BFS-based bisection of one community."""
    nodes = set(nodes)
    sub = conn.subgraph(nodes)
    start = sorted(nodes)[0]
    reached = set(nx.bfs_tree(sub, start))
    if len(reached) < 2:
        reached = nodes
    half = set(sorted(reached)[: max(1, len(nodes) // 2)])
    return half or {sorted(nodes)[0]}
