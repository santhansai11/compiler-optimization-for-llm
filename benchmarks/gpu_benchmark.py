"""Real-GPU benchmark: original vs compiled graphs, measured on this machine.

    python benchmarks/gpu_benchmark.py                 # auto: CUDA > Apple MPS > CPU
    python benchmarks/gpu_benchmark.py --device cuda --out docs/results/gpu_cuda.json

Every row is MEASURED (synchronised wall-clock, interleaved A/B, min of N)
and numerically verified: compiled output vs original on the device, and the
device output vs the NumPy reference executor. Modeled speedups from the
cost table are stored next to them so the two can be compared.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.architectures import (  # noqa: E402
    create_llama_style_graph, create_moe_graph, create_multihead_graph,
    create_postln_transformer_graph, create_preln_transformer_graph,
    create_relu_transformer_graph)
from pipeline import run_pipeline  # noqa: E402
from utils.cost_model import (CALIBRATED, LEGACY_OP_LATENCY_US,  # noqa: E402
                              deployment_latency_us, sequential_latency_us)
from utils.executor import execute  # noqa: E402
from utils.environment import collect as environment  # noqa: E402
from utils.torch_executor import device_name, gpu_compare, pick_device  # noqa: E402

BUILDERS = {
    "Post-LN Transformer": lambda b, s, d: create_postln_transformer_graph(2, b, s, d),
    "Pre-LN Transformer": lambda b, s, d: create_preln_transformer_graph(2, b, s, d),
    "Post-LN + ReLU": lambda b, s, d: create_relu_transformer_graph(2, b, s, d),
    "LLaMA-style": lambda b, s, d: create_llama_style_graph(2, b, s, d),
    "MoE Transformer": lambda b, s, d: create_moe_graph(2, b, s, d, experts=4),
    "Multi-head Transformer": lambda b, s, d: create_multihead_graph(2, b, s, d, heads=4),
}
WORKLOADS = [(4, 64, 64), (8, 256, 256), (8, 512, 512)]  # (batch, seq, d_model)


def run(device, repeats, warmup, workloads, names):
    rows = []
    for name in names:
        for b, s, d in workloads:
            g = BUILDERS[name](b, s, d)
            opt, infos, ctime = run_pipeline(g, partition=False)
            a, o, t, cgn = gpu_compare([(g, "composed"), (opt, "sdpa"), (opt, "auto"),
                                        (opt, "auto", True)],
                                       device, repeats=repeats, warmup=warmup)
            denom = max(1e-6, float(np.abs(a["output"]).max()))
            rel = max(float(np.abs(a["output"] - x["output"]).max()) / denom
                      for x in (o, t, cgn))
            row = {
                "architecture": name, "batch": b, "seq": s, "d_model": d,
                "nodes_original": g.node_count(), "nodes_optimized": opt.node_count(),
                "latency_ms_original": a["latency_min_s"] * 1e3,
                "latency_ms_optimized_sdpa": o["latency_min_s"] * 1e3,
                "latency_ms_optimized_auto": t["latency_min_s"] * 1e3,
                "speedup_sdpa": a["latency_min_s"] / o["latency_min_s"],
                "latency_ms_optimized_codegen": cgn["latency_min_s"] * 1e3,
                "speedup_codegen": a["latency_min_s"] / cgn["latency_min_s"],
                "kernel_dispatches_codegen": cgn["kernel_dispatches"],
                "speedup_auto": a["latency_min_s"] / t["latency_min_s"],
                "tokens_per_s_original": b * s / a["latency_min_s"],
                "tokens_per_s_auto": b * s / t["latency_min_s"],
                "live_peak_mb_original": a["live_peak_bytes"] / 2**20,
                "live_peak_mb_optimized": t["live_peak_bytes"] / 2**20,
                "kernel_dispatches_original": a["kernel_dispatches"],
                "kernel_dispatches_sdpa": o["kernel_dispatches"],
                "kernel_dispatches_auto": t["kernel_dispatches"],
                "dispatch_reduction_sdpa": 1 - o["kernel_dispatches"] / a["kernel_dispatches"],
                "max_rel_error_vs_original": rel,
                "outputs_match": rel < 1e-3,
                "modeled_speedup": sequential_latency_us(g) / deployment_latency_us(opt),
                "modeled_speedup_legacy_table": (
                    sum(LEGACY_OP_LATENCY_US.get(d["op_type"], 150.0) + 12.0
                        for _, d in g.graph.nodes(data=True))
                    / sum(LEGACY_OP_LATENCY_US.get(d["op_type"], 150.0) + 12.0
                          for _, d in opt.graph.nodes(data=True))),
                "cost_model_calibrated": CALIBRATED,
                "compile_ms": ctime * 1e3,
            }
            if a["allocator_peak_bytes"] is not None:
                row["allocator_peak_mb_original"] = a["allocator_peak_bytes"] / 2**20
                row["allocator_peak_mb_optimized"] = t["allocator_peak_bytes"] / 2**20
            if b * s * d <= 4 * 64 * 64:  # cheap: also check against NumPy
                ref = execute(g, repeats=1, warmup=0)["output"]
                row["max_abs_error_vs_numpy"] = float(np.abs(a["output"] - ref).max())
            print(f"{name:24s} B{b} S{s} D{d}: {row['latency_ms_original']:8.3f} ms -> "
                  f"sdpa {row['latency_ms_optimized_sdpa']:8.3f} (x{row['speedup_sdpa']:.2f}) | "
                  f"auto {row['latency_ms_optimized_auto']:8.3f} (x{row['speedup_auto']:.2f}) "
                  f"codegen {row['latency_ms_optimized_codegen']:8.3f} (x{row['speedup_codegen']:.2f}) "
                  f"modeled x{row['modeled_speedup']:.2f}  match={row['outputs_match']}", flush=True)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="2 architectures, 2 sizes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    names = list(BUILDERS)[:2] if args.quick else list(BUILDERS)
    workloads = WORKLOADS[:2] if args.quick else WORKLOADS
    rows = run(device, args.repeats, args.warmup, workloads, names)
    result = {
        "device": device.type, "device_name": device_name(device),
        "torch": torch.__version__, "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "synchronised wall-clock, interleaved original/optimized, min of "
                  f"{args.repeats} after {args.warmup} warmups; float32",
        "kernel_dispatches_note": "counted per torch call issued by the executor (a proxy, "
                                  "not a hardware profiler count)",
        "peak_memory_note": "live_peak = peak bytes of simultaneously live tensors; "
                            "allocator_peak (CUDA only) = torch allocator high-water mark",
        "rows": rows,
        "summary": {
            "all_outputs_match": all(r["outputs_match"] for r in rows),
            "median_speedup_sdpa": float(np.median([r["speedup_sdpa"] for r in rows])),
            "median_speedup_auto": float(np.median([r["speedup_auto"] for r in rows])),
            "median_speedup_codegen": float(np.median([r["speedup_codegen"] for r in rows])),
            "max_speedup_codegen": float(max(r["speedup_codegen"] for r in rows)),
            "min_speedup_codegen": float(min(r["speedup_codegen"] for r in rows)),
            "min_speedup_auto": float(min(r["speedup_auto"] for r in rows)),
            "max_speedup_auto": float(max(r["speedup_auto"] for r in rows)),
            "median_modeled_speedup": float(np.median([r["modeled_speedup"] for r in rows])),
            "median_modeled_speedup_legacy_table": float(np.median([r["modeled_speedup_legacy_table"] for r in rows])),
            "speedup_mae_calibrated": float(np.mean([abs(r["modeled_speedup"] - r["speedup_auto"]) for r in rows])),
            "speedup_mae_legacy_table": float(np.mean([abs(r["modeled_speedup_legacy_table"] - r["speedup_auto"]) for r in rows])),
            "mean_dispatch_reduction_sdpa": float(np.mean([r["dispatch_reduction_sdpa"] for r in rows])),
        },
    }
    out = Path(args.out) if args.out else ROOT / "docs" / "results" / f"gpu_{device.type}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result["environment"] = environment()
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
