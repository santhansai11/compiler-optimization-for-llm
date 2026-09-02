"""End-to-end optimization pipeline shared by the UI and the CLI."""

import time

from passes.canonicalize import canonicalize_attention
from passes.merge import merge_operators
from passes.normalize import normalize_graph
from passes.partition import partition_graph, partition_hypergraph
from passes.schedule import schedule_graph


def run_pipeline(graph, normalize=True, canonicalize=True, merge=True,
                 schedule=True, partition=False, hypergraph=False,
                 num_partitions=2, neuro_symbolic=False):
    """Run the selected passes in compiler order.

    Returns ``(optimized_graph, infos, compile_time_s)`` where ``infos``
    carries one info dict per executed pass (incl. durations).
    """
    started = time.perf_counter()
    current = graph.copy()
    infos = {}

    if normalize:
        current, infos["normalize"] = normalize_graph(current)

    if canonicalize:
        current, infos["attention_canonicalization"] = canonicalize_attention(
            current
        )

    if merge:
        current, infos["semantic_merging"] = merge_operators(current)

    if neuro_symbolic:
        try:
            from search.neuro_symbolic import neuro_symbolic_search

            current, infos["neuro_symbolic_search"] = neuro_symbolic_search(
                current
            )
        except ImportError:
            infos["neuro_symbolic_search"] = {
                "pass": "neuro_symbolic_search",
                "status": "search module not available yet",
            }

    if schedule:
        current, infos["dags"] = schedule_graph(current)

    if partition:
        partition_fn = partition_hypergraph if hypergraph else partition_graph
        current, infos["partitioning"] = partition_fn(
            current, num_partitions
        )

    compile_time_s = time.perf_counter() - started

    infos["pipeline"] = {
        "pass": "pipeline",
        "passes_run": [
            key
            for key in (
                "normalize",
                "attention_canonicalization",
                "semantic_merging",
                "neuro_symbolic_search",
                "dags",
                "partitioning",
            )
            if key in infos
        ],
        "compile_time_s": compile_time_s,
        "nodes_after": current.node_count(),
        "duration_s": compile_time_s,
    }
    return current, infos, compile_time_s
