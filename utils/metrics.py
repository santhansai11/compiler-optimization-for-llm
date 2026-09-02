"""Output metrics for the optimization pipeline (the required ten + extras)."""

import numpy as np

from utils.cost_model import (
    kernel_launches,
    live_range_peak_memory_mb,
    naive_peak_memory_mb,
    scheduled_latency_us,
    sequential_latency_us,
)
from utils.executor import execute


def compute_metrics(original_graph, optimized_graph, pass_infos,
                    compile_time_s, batch_size=4):
    """Compute every required output metric; returns a nested dict."""
    nodes_original = original_graph.node_count()
    nodes_optimized = optimized_graph.node_count()
    edges_original = original_graph.edge_count()
    edges_optimized = optimized_graph.edge_count()

    graph_reduction_ratio = (
        (nodes_original - nodes_optimized) / nodes_original
        if nodes_original
        else 0.0
    )

    merge_info = pass_infos.get("semantic_merging", {}) or {}
    attention_info = pass_infos.get("attention_canonicalization", {}) or {}

    operator_merge_ratio = (
        merge_info.get("nodes_removed", 0) / nodes_original
        if nodes_original
        else 0.0
    )
    detected = attention_info.get("detected", 0)
    canonicalized = attention_info.get("canonicalized", 0)
    acr = canonicalized / detected if detected else 0.0

    launches_original = kernel_launches(original_graph)
    launches_optimized = kernel_launches(optimized_graph)
    kernel_launch_reduction = (
        (launches_original - launches_optimized) / launches_original
        if launches_original
        else 0.0
    )

    # ---- Measured (NumPy reference executor) ----
    original_run = execute(original_graph)
    optimized_run = execute(optimized_graph)

    latency_original_ms = original_run["latency_s"] * 1e3
    latency_optimized_ms = optimized_run["latency_s"] * 1e3
    speedup = (
        latency_original_ms / latency_optimized_ms
        if latency_optimized_ms
        else 1.0
    )
    throughput_original = batch_size / original_run["latency_s"]
    throughput_optimized = batch_size / optimized_run["latency_s"]

    peak_original_mb = original_run["peak_bytes"] / (1024**2)
    peak_optimized_mb = optimized_run["peak_bytes"] / (1024**2)

    out_orig = np.asarray(original_run["output"], dtype=np.float64)
    out_opt = np.asarray(optimized_run["output"], dtype=np.float64)
    same_shape = out_orig.shape == out_opt.shape
    max_abs_diff = (
        float(np.max(np.abs(out_orig - out_opt)))
        if same_shape
        else float("inf")
    )
    accuracy_preserved = bool(
        same_shape
        and np.allclose(out_orig, out_opt, rtol=1e-4, atol=1e-5)
    )
    accuracy_preservation_pct = (
        100.0
        if accuracy_preserved
        else max(0.0, 100.0 - min(99.0, max_abs_diff * 1e3))
    )

    # ---- Modeled (GPU-style cost model) ----
    modeled_latency_original_us = sequential_latency_us(original_graph)
    modeled_latency_optimized_us = scheduled_latency_us(optimized_graph)
    modeled_speedup = (
        modeled_latency_original_us / modeled_latency_optimized_us
        if modeled_latency_optimized_us
        else 1.0
    )
    modeled_mem_original = naive_peak_memory_mb(original_graph)
    modeled_mem_optimized = live_range_peak_memory_mb(optimized_graph)

    pass_durations = {
        name: info.get("duration_s", 0.0)
        for name, info in pass_infos.items()
        if isinstance(info, dict)
    }

    return {
        "counts": {
            "nodes_original": nodes_original,
            "nodes_optimized": nodes_optimized,
            "edges_original": edges_original,
            "edges_optimized": edges_optimized,
            "kernel_launches_original": launches_original,
            "kernel_launches_optimized": launches_optimized,
        },
        # Required metric 5
        "graph_reduction_ratio": graph_reduction_ratio,
        # Required metric 7
        "operator_merge_ratio": operator_merge_ratio,
        # Required metric 8
        "attention_canonicalization_rate": acr,
        "attention_subgraphs": {
            "detected": detected,
            "canonicalized": canonicalized,
        },
        # Required metric 9
        "kernel_launch_reduction": kernel_launch_reduction,
        # Required metrics 1 + 2 + 3 (measured)
        "latency_ms": {
            "original": latency_original_ms,
            "optimized": latency_optimized_ms,
        },
        "throughput": {
            "original": throughput_original,
            "optimized": throughput_optimized,
            "batch": batch_size,
        },
        "speedup": speedup,
        # Required metric 4 (measured)
        "peak_memory_mb": {
            "original": peak_original_mb,
            "optimized": peak_optimized_mb,
        },
        # Required metric 6
        "accuracy": {
            "preserved": accuracy_preserved,
            "max_abs_diff": max_abs_diff,
            "preservation_pct": accuracy_preservation_pct,
        },
        # Modeled GPU estimates (cost model) for latency + memory
        "modeled": {
            "latency_original_us": modeled_latency_original_us,
            "latency_optimized_us": modeled_latency_optimized_us,
            "speedup": modeled_speedup,
            "memory_original_mb": modeled_mem_original,
            "memory_optimized_mb": modeled_mem_optimized,
            "memory_reduction_pct": (
                (modeled_mem_original - modeled_mem_optimized)
                / modeled_mem_original
                if modeled_mem_original
                else 0.0
            ),
        },
        # Required metric 10
        "compilation_time_s": compile_time_s,
        "pass_durations_s": pass_durations,
        "pass_infos": pass_infos,
    }


def format_report(metrics):
    """Plain-text metrics report for the CLI and the UI code block."""
    counts = metrics["counts"]
    lines = [
        "=== LLM Compiler Optimizer - Metrics Report ===",
        "",
        f"Graph nodes            : {counts['nodes_original']} -> "
        f"{counts['nodes_optimized']}",
        f"Graph edges            : {counts['edges_original']} -> "
        f"{counts['edges_optimized']}",
        f"Kernel launches        : {counts['kernel_launches_original']} -> "
        f"{counts['kernel_launches_optimized']}",
        "",
        f"GRR  (graph reduction) : "
        f"{metrics['graph_reduction_ratio'] * 100:6.2f} %",
        f"OMR  (operator merge)  : "
        f"{metrics['operator_merge_ratio'] * 100:6.2f} %",
        f"ACR  (attention canon.): "
        f"{metrics['attention_canonicalization_rate'] * 100:6.2f} %",
        f"Kernel launch reduct.  : "
        f"{metrics['kernel_launch_reduction'] * 100:6.2f} %",
        "",
        f"Inference latency      : "
        f"{metrics['latency_ms']['original']:8.3f} ms -> "
        f"{metrics['latency_ms']['optimized']:8.3f} ms  (measured)",
        f"Throughput             : "
        f"{metrics['throughput']['original']:8.1f} -> "
        f"{metrics['throughput']['optimized']:8.1f} samples/s  "
        f"(batch {metrics['throughput']['batch']})",
        f"Speedup                : {metrics['speedup']:6.3f} x  (measured)"
        f" | {metrics['modeled']['speedup']:6.3f} x  (modeled GPU)",
        f"Peak memory (tensors)  : "
        f"{metrics['peak_memory_mb']['original']:8.2f} MB -> "
        f"{metrics['peak_memory_mb']['optimized']:8.2f} MB  (measured)",
        f"Modeled GPU memory     : "
        f"{metrics['modeled']['memory_original_mb']:8.1f} MB -> "
        f"{metrics['modeled']['memory_optimized_mb']:8.1f} MB",
        f"Accuracy preservation  : "
        f"{metrics['accuracy']['preservation_pct']:6.2f} %  "
        f"(max |diff| = {metrics['accuracy']['max_abs_diff']:.3e})",
        "",
        f"Compilation time       : "
        f"{metrics['compilation_time_s'] * 1e3:8.2f} ms",
    ]
    durations = metrics.get("pass_durations_s") or {}
    if durations:
        lines.append("")
        lines.append("Pass durations:")
        for name, duration in sorted(durations.items()):
            lines.append(f"  {name:<28s} {duration * 1e3:8.2f} ms")
    return "\n".join(lines)
