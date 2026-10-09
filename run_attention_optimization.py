"""
End-to-End PyTorch FX Attention Semantic Merging & Workload Benchmark.

1. Symbolically traces AttentionModel using torch.fx.
2. Applies graph normalization (identity elimination, dead code removal).
3. Applies semantic operator merging (pattern matches multi-op attention DAG
   and fuses it into fused_attention).
4. Generates publication-ready NetworkX/Matplotlib DAG visualizations
   (original_graph.png and optimized_graph.png).
5. Runs a comprehensive workload sweep across (batch, seq, d_model),
   measuring latency, speedup, throughput, GRR, and numerical correctness.
"""

import math
import statistics
import time

import torch
import torch.fx as fx

from ir.fx_graph import count_compute_nodes, print_graph
from models.attention import AttentionModel
from passes.normalize import normalize
from passes.semantic_merge import semantic_merge_attention
from passes.dags_fx import schedule_fx
from visualize import draw_graph


def benchmark(model, x, warmup=20, runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            output = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    return output, statistics.median(times)


def run_single_demo():
    torch.manual_seed(42)

    batch = 8
    sequence_length = 32
    d_model = 64

    x = torch.randn(batch, sequence_length, d_model)

    print("=" * 60)
    print("      PYTORCH FX ATTENTION OPTIMIZATION PIPELINE      ")
    print("=" * 60)
    print("Input Tensor Shape:", tuple(x.shape))

    model = AttentionModel(d_model=d_model)
    model.eval()

    # Original Graph
    original = fx.symbolic_trace(model)
    original_nodes = count_compute_nodes(original)
    print_graph(original, "ORIGINAL GRAPH")
    print(f"\nOriginal compute nodes: {original_nodes}")
    print("\n===== ORIGINAL GENERATED CODE =====")
    print(original.code)

    # Save original graph diagram
    draw_graph(original, "Original Attention Graph", "original_graph.png")
    print("[Saved original_graph.png]")

    # Original execution
    original_output, original_time = benchmark(original, x)

    # Optimization: Normalization + Semantic Merging
    optimized = fx.symbolic_trace(model)
    optimized = normalize(optimized)
    merged = semantic_merge_attention(optimized)

    if not merged:
        raise RuntimeError("Attention pattern was not detected.")

    optimized, schedule_info = schedule_fx(optimized)
    print(f"DAGS schedule: {schedule_info['levels']} dependency levels, "
          f"critical path {schedule_info['critical_path_cost']:.0f} ops")

    optimized_nodes = count_compute_nodes(optimized)
    print_graph(optimized, "SEMANTICALLY MERGED GRAPH")
    print(f"\nMerged compute nodes: {optimized_nodes}")
    print("\n===== MERGED GENERATED CODE =====")
    print(optimized.code)

    # Save optimized graph diagram
    draw_graph(optimized, "Semantically Merged Attention Graph", "optimized_graph.png")
    print("[Saved optimized_graph.png]")

    # Optimized execution
    optimized_output, optimized_time = benchmark(optimized, x)

    # Correctness check
    max_error = (original_output - optimized_output).abs().max().item()
    correct = torch.allclose(original_output, optimized_output, rtol=1e-5, atol=1e-6)

    # Metrics
    reduction = ((original_nodes - optimized_nodes) / original_nodes) * 100
    speedup = original_time / optimized_time if optimized_time > 0 else 1.0

    print("\n" + "=" * 60)
    print("                   RESULTS SUMMARY                    ")
    print("=" * 60)
    print(f"Original compute nodes : {original_nodes}")
    print(f"Merged compute nodes   : {optimized_nodes}")
    print(f"Graph reduction ratio  : {reduction:.2f}%")
    print(f"Original latency       : {original_time:.4f} ms")
    print(f"Merged latency         : {optimized_time:.4f} ms")
    print(f"Speedup                : {speedup:.3f}x")
    print(f"Max absolute error     : {max_error:.8e}")
    print(f"Numerical correctness  : {correct}")
    print("Generated files        : original_graph.png, optimized_graph.png")


def run_case(batch, seq, d_model, warmup=20, runs=100):
    torch.manual_seed(42)

    model = AttentionModel(d_model=d_model)
    model.eval()

    x = torch.randn(batch, seq, d_model)

    original = fx.symbolic_trace(model)
    original_output, original_latency = benchmark(
        original, x, warmup=warmup, runs=runs
    )
    original_nodes = count_compute_nodes(original)

    compile_started = time.perf_counter()
    optimized = fx.symbolic_trace(model)
    optimized = normalize(optimized)
    normalized_nodes = count_compute_nodes(optimized)
    if not semantic_merge_attention(optimized):
        raise RuntimeError("Attention pattern not detected")
    compile_time_ms = (time.perf_counter() - compile_started) * 1000

    optimized_output, optimized_latency = benchmark(
        optimized, x, warmup=warmup, runs=runs
    )
    optimized_nodes = count_compute_nodes(optimized)

    max_error = (original_output - optimized_output).abs().max().item()
    correct = torch.allclose(original_output, optimized_output, rtol=1e-5, atol=1e-6)

    grr = ((original_nodes - optimized_nodes) / original_nodes) * 100
    omr = ((normalized_nodes - optimized_nodes) / normalized_nodes * 100
           if normalized_nodes else 0.0)
    speedup = original_latency / optimized_latency if optimized_latency > 0 else 1.0

    tokens = batch * seq
    throughput_orig = tokens / (original_latency / 1000.0) if original_latency > 0 else 0
    throughput_opt = tokens / (optimized_latency / 1000.0) if optimized_latency > 0 else 0

    return {
        "batch": batch,
        "seq": seq,
        "d_model": d_model,
        "original_nodes": original_nodes,
        "optimized_nodes": optimized_nodes,
        "normalized_nodes": normalized_nodes,
        "grr": grr,
        "omr": omr,
        "acr": None,
        "kernel_launches_original_estimate": original_nodes,
        "kernel_launches_optimized_estimate": optimized_nodes,
        "kernel_launch_reduction_estimate": grr,
        "compilation_time_ms": compile_time_ms,
        "original_latency": original_latency,
        "optimized_latency": optimized_latency,
        "speedup": speedup,
        "throughput_original": throughput_orig,
        "throughput_optimized": throughput_opt,
        "max_error": max_error,
        "correct": correct,
        "peak_gpu_memory_mb": None,
    }


def run_benchmark_sweep():
    cases = [
        (1, 32, 64),
        (2, 64, 64),
        (4, 128, 64),
        (8, 256, 64),
        (16, 512, 64),
        (2, 64, 128),
        (4, 128, 128),
        (8, 256, 128),
        (2, 64, 256),
        (4, 128, 256),
        (8, 256, 256),
        (2, 128, 512),
        (4, 256, 512),
    ]

    print("\n" + "=" * 120)
    print("                 MULTI-WORKLOAD LLM BENCHMARK SWEEP (PyTorch FX)                 ")
    print("=" * 120)
    print(
        f"{'BATCH':>5} {'SEQ':>5} {'D_MODEL':>7} | "
        f"{'NODES':>7} | "
        f"{'GRR':>7} | "
        f"{'ORIG ms':>9} | "
        f"{'OPT ms':>9} | "
        f"{'SPEEDUP':>8} | "
        f"{'THRPT ORIG':>11} | "
        f"{'THRPT OPT':>11} | "
        f"{'MAX ERROR':>10} | "
        f"{'CORRECT':>7}"
    )
    print("-" * 120)

    results = []
    for batch, seq, d_model in cases:
        try:
            res = run_case(batch, seq, d_model)
            results.append(res)
            print(
                f"{res['batch']:5d} {res['seq']:5d} {res['d_model']:7d} | "
                f"{res['original_nodes']:2d} -> {res['optimized_nodes']:2d} | "
                f"{res['grr']:6.2f}% | "
                f"{res['original_latency']:9.4f} | "
                f"{res['optimized_latency']:9.4f} | "
                f"{res['speedup']:7.3f}x | "
                f"{res['throughput_original']:11.0f} | "
                f"{res['throughput_optimized']:11.0f} | "
                f"{res['max_error']:10.2e} | "
                f"{str(res['correct']):>7}"
            )
        except Exception as e:
            print(f"{batch:5d} {seq:5d} {d_model:7d} | FAILED: {e}")

    if results:
        print("-" * 120)
        print("===== SUMMARY STATISTICS =====")
        print(f"Average Graph Reduction Ratio : {statistics.mean(r['grr'] for r in results):.2f}%")
        print(f"Median Speedup                : {statistics.median(r['speedup'] for r in results):.3f}x")
        print(f"Max Speedup                   : {max(r['speedup'] for r in results):.3f}x")
        print(f"Min Speedup                   : {min(r['speedup'] for r in results):.3f}x")
        print(f"All Outputs Bit-Exact/Correct : {all(r['correct'] for r in results)}")
        print("=" * 120)


def main():
    run_single_demo()
    run_benchmark_sweep()


if __name__ == "__main__":
    main()
