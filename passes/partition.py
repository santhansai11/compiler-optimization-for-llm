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
