"""Output metrics for the optimization pipeline (the required ten + extras)."""

from utils.cost_model import (
    CALIBRATED,
    deployment_latency_us,
    kernel_launches,
    live_range_peak_memory_mb,
    naive_peak_memory_mb,
    scheduled_latency_us,
    sequential_latency_us,
)
from utils.executor import _make_feed, compare_latency, execute
from utils.benchmarking import paired_summary
from utils.validation import compare_outputs


def compute_metrics(original_graph, optimized_graph, pass_infos,
                    compile_time_s, batch_size=4, feed=None):
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
    attention_info = (
        pass_infos.get("attention_canonicalization")
        or merge_info
        or {}
    )

    operator_merge_ratio = (
        merge_info.get("nodes_removed", 0) / nodes_original
        if nodes_original
        else 0.0
    )
    detected = attention_info.get(
        "detected", attention_info.get("attention_patterns_detected", 0)
    )
    canonicalized = attention_info.get(
        "canonicalized", attention_info.get("attention_patterns_merged", 0)
    )
    acr = canonicalized / detected if detected else None

    launches_original = kernel_launches(original_graph)
    launches_optimized = kernel_launches(optimized_graph)
    kernel_launch_reduction = (
        (launches_original - launches_optimized) / launches_original
        if launches_original
        else 0.0
    )

    # ---- Measured (NumPy reference executor) ----
    shared_feed = _make_feed(original_graph, feed)
    # Always count the batch actually executed (some builders fix their size).
    batch_size = int(shared_feed.shape[0]) if shared_feed.ndim else 1
    original_run = execute(original_graph, repeats=1, warmup=0, feed=shared_feed)
    optimized_run = execute(optimized_graph, repeats=1, warmup=0, feed=shared_feed)
    timing = compare_latency(original_graph, optimized_graph, feed=shared_feed, return_record=True)
    original_run["latency_s"] = timing["median_us"]["original"] / 1e6
    optimized_run["latency_s"] = timing["median_us"]["optimized"] / 1e6
    comparison = paired_summary(timing["samples_us"]["original"], timing["samples_us"]["optimized"])

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

    accuracy = compare_outputs(original_run["outputs"], optimized_run["outputs"])

    # ---- Modeled (per-op GPU cost model; calibrated on a real GPU) ----
    # Primary modeled speedup: one kernel stream per device (what the real
    # GPU measurements show). DAGS overlap is reported separately as an
    # upper bound, because kernels do not actually overlap on one stream.
    modeled_latency_original_us = sequential_latency_us(original_graph)
    modeled_latency_optimized_us = deployment_latency_us(optimized_graph)
    modeled_speedup = (
        modeled_latency_original_us / modeled_latency_optimized_us
        if modeled_latency_optimized_us
        else 1.0
    )
    overlap_latency_us = scheduled_latency_us(optimized_graph)
    parallel_potential = (
        modeled_latency_original_us / overlap_latency_us
        if overlap_latency_us
        else 1.0
    )
    # Compare both graphs with the same liveness policy; retain the naive
    # baseline separately instead of attributing all buffer reuse to rewrites.
    modeled_mem_original = live_range_peak_memory_mb(original_graph)
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
        "timing": timing,
        "paired_comparison": comparison,
        "performance_conclusion": comparison["ci_direction"] if accuracy["preserved"] else "invalid_output",
        # Required metric 4 (measured)
        "peak_memory_mb": {
            "original": peak_original_mb,
            "optimized": peak_optimized_mb,
            "unit": "MiB",
            "scope": "Unique live graph-value storage; excludes kernel temporaries, implicit weights and allocator overhead",
        },
        # Required metric 6
        "accuracy": accuracy,
        # Modeled GPU estimates (cost model) for latency + memory
        "modeled": {
            "latency_original_us": modeled_latency_original_us,
            "latency_optimized_us": modeled_latency_optimized_us,
            "speedup": modeled_speedup,
            "calibrated": CALIBRATED,
            "parallel_overlap_speedup_upper_bound": parallel_potential,
            "memory_original_mb": modeled_mem_original,
            "memory_optimized_mb": modeled_mem_optimized,
            "memory_original_naive_mb": naive_peak_memory_mb(original_graph),
            "memory_method": "Same live-range policy on both graphs; heuristic per-op sizes, not measured GPU allocation",
            "memory_reduction_pct": (
                (modeled_mem_original - modeled_mem_optimized)
                / modeled_mem_original * 100.0
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
    error = metrics["accuracy"]["max_abs_diff"]
    error_text = f"{error:.3e}" if error is not None else "unavailable (invalid output)"
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
        ("ACR  (attention canon.): "
         + (f"{metrics['attention_canonicalization_rate'] * 100:6.2f} %"
            if metrics["attention_canonicalization_rate"] is not None
            else "N/A (pass disabled)")),
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
        f"{metrics['peak_memory_mb']['original']:8.2f} MiB -> "
        f"{metrics['peak_memory_mb']['optimized']:8.2f} MiB  (graph storage only)",
        f"Modeled GPU memory     : "
        f"{metrics['modeled']['memory_original_mb']:8.1f} MB -> "
        f"{metrics['modeled']['memory_optimized_mb']:8.1f} MB",
        f"Elements within tolerance: "
        f"{metrics['accuracy']['preservation_pct']:6.2f} %  "
        f"(max |diff| = {error_text})",
        f"Timing evidence        : {metrics.get('performance_conclusion', 'unavailable')} (paired 95% interval)",
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
