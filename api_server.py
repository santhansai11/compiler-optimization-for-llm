"""
Fast, lightweight REST API server for the React Compiler Optimization Frontend.
Exposes endpoints for graph generation, multi-pass optimization,
DAGS scheduling analysis, semantic operator merging, and workload benchmarking.
"""

import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "benchmark_reports"

from ir.graph import ComputationGraph
from models.transformer import create_demo_transformer_graph
from models.architectures import (
    create_postln_transformer_graph,
    create_preln_transformer_graph,
    create_relu_transformer_graph,
    create_llama_style_graph,
    create_moe_graph,
    create_multihead_graph,
    create_mlp_classifier_graph,
)
from models.operations import OPERATION_DEMOS
from models.tiny_lm import WEIGHTS_PATH as LM_WEIGHTS, build_graph as build_tiny_lm
from utils.reports import write_report
from pipeline import run_pipeline
from passes.normalize import normalize as normalize_fx
from passes.semantic_merge import semantic_merge_attention
from ir.fx_graph import count_compute_nodes
from models.attention import AttentionModel
from utils.metrics import compute_metrics
from utils.cost_model import (
    sequential_latency_us,
    scheduled_latency_us,
    naive_peak_memory_mb,
    live_range_peak_memory_mb,
    kernel_launches,
    deployment_latency_us,
)
from utils.executor import execute


# id -> (display name, builder(batch, seq, d_model), description)
MODEL_REGISTRY = {
    "postln_2b": ("Post-LN Transformer (2 blocks)",
                  lambda b, s, d: create_postln_transformer_graph(2, b, s, d),
                  "Reference Vaswani et al. architecture"),
    "postln_4b": ("Post-LN Transformer (4 blocks)",
                  lambda b, s, d: create_postln_transformer_graph(4, b, s, d),
                  "Deeper 4-block transformer DAG"),
    "preln_2b": ("Pre-LN Transformer (GPT-2 style)",
                 lambda b, s, d: create_preln_transformer_graph(2, b, s, d),
                 "LayerNorm inside the residual branches"),
    "relu_2b": ("Post-LN + ReLU FFN",
                lambda b, s, d: create_relu_transformer_graph(2, b, s, d),
                "Exercises the LinearRelu fusion pathway"),
    "llama_2b": ("LLaMA-style (RMSNorm + SwiGLU)",
                 lambda b, s, d: create_llama_style_graph(2, b, s, d),
                 "RMSNorm, gated SwiGLU feed-forward, no biases"),
    "moe_2b": ("MoE Transformer (2 blocks, 4 experts)",
               lambda b, s, d: create_moe_graph(2, b, s, d, experts=4),
               "Parallel expert FFNs (Mixtral-like, dense routing)"),
    "multihead_2b": ("Multi-head Transformer (2 blocks, 4 heads)",
                     lambda b, s, d: create_multihead_graph(2, b, s, d, heads=4),
                     "Explicit parallel attention heads"),
    "mlp_classifier": ("MLP classifier (784-256-64-10)",
                       lambda b, s, d: create_mlp_classifier_graph(batch=256),
                       "Plain feed-forward network (random weights)"),
}
if LM_WEIGHTS.exists():
    MODEL_REGISTRY["tiny_lm"] = (
        "Trained Tiny Shakespeare LM (2 blocks)",
        lambda b, s, d: build_tiny_lm(b, s),
        "Real trained character-level Transformer (112k params)")
for _id, (_name, _builder, _desc) in OPERATION_DEMOS.items():
    MODEL_REGISTRY[_id] = (_name, (lambda f: lambda b, s, d: f(b, s, d))(_builder), _desc)

_LEARNED = None
_RL = None
DEFAULT_PASSES = {"normalize": True, "canonicalize": True, "merge": True,
                  "schedule": True, "partition": False}
STAGE_ORDER = [
    ("normalize", "Graph normalization"),
    ("canonicalize", "Attention canonicalization"),
    ("merge", "Semantic operator merging"),
    ("schedule", "DAGS scheduling"),
    ("partition", "Graph partitioning"),
]


def _passes_from(data):
    flags = dict(DEFAULT_PASSES)
    flags.update({k: bool(v) for k, v in (data.get("passes") or {}).items()
                  if k in flags})
    return flags


def _run(graph, flags, num_partitions=2, hypergraph=False, mode=None):
    """``hypergraph`` is kept for older callers; ``mode`` wins when given."""
    return run_pipeline(
        graph, normalize=flags["normalize"], canonicalize=flags["canonicalize"],
        merge=flags["merge"], schedule=flags["schedule"],
        partition=flags["partition"], hypergraph=hypergraph,
        num_partitions=num_partitions, partition_mode=mode)


def _stage_table(graph, flags, num_partitions, hypergraph, mode=None):
    """Cumulative per-pass table: nodes, launches and modeled latency."""
    rows = [{"stage": "Original graph", "ops": graph.node_count(),
             "launches": kernel_launches(graph),
             "modeled_latency_us": sequential_latency_us(graph)}]
    enabled = {k: False for k in flags}
    for key, label in STAGE_ORDER:
        if not flags[key]:
            continue
        enabled[key] = True
        g, _, _ = _run(graph, enabled, num_partitions, hypergraph, mode)
        rows.append({"stage": f"+ {label}", "ops": g.node_count(),
                     "launches": kernel_launches(g),
                     "modeled_latency_us": deployment_latency_us(g)})
    return rows


def _architecture_row(name, graph, flags, num_partitions, hypergraph, batch):
    opt, infos, ctime = _run(graph, flags, num_partitions, hypergraph)
    m = compute_metrics(graph, opt, infos, ctime, batch_size=batch)
    acr = m["attention_canonicalization_rate"]
    return {
        "name": name,
        "nodes_original": m["counts"]["nodes_original"],
        "nodes_optimized": m["counts"]["nodes_optimized"],
        "grr": m["graph_reduction_ratio"] * 100,
        "omr": m["operator_merge_ratio"] * 100,
        "acr": None if acr is None else acr * 100,
        "klr": m["kernel_launch_reduction"] * 100,
        "latency_original_ms": m["latency_ms"]["original"],
        "latency_optimized_ms": m["latency_ms"]["optimized"],
        "speedup": m["speedup"],
        "modeled_speedup": m["modeled"]["speedup"],
        "throughput_original": m["throughput"]["original"],
        "throughput_optimized": m["throughput"]["optimized"],
        "memory_original_mb": m["peak_memory_mb"]["original"],
        "memory_optimized_mb": m["peak_memory_mb"]["optimized"],
        "gpu_memory_original_mb": m["modeled"]["memory_original_mb"],
        "gpu_memory_optimized_mb": m["modeled"]["memory_optimized_mb"],
        "accuracy_preserved": m["accuracy"]["preserved"],
        "max_abs_diff": m["accuracy"]["max_abs_diff"],
        "compile_ms": m["compilation_time_s"] * 1e3,
    }


def _json_safe(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def graph_to_d3(graph):
    """Convert ComputationGraph into node and edge lists for D3/SVG visualization."""
    nodes = []
    for node_id, data in graph.graph.nodes(data=True):
        nodes.append({
            "id": node_id,
            "label": node_id,
            "op_type": data.get("op_type", "Operation"),
            "shape": str(data.get("shape", "")),
            "is_output": bool(data.get("is_output", False)),
            "schedule_order": data.get("schedule_order"),
            "schedule_level": data.get("schedule_level"),
            "partition": data.get("partition"),
            "fused_from": data.get("fused_from", []),
        })

    edges = []
    for u, v in graph.graph.edges():
        edges.append({"source": u, "target": v})

    return {"nodes": nodes, "edges": edges}


class CompilerAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(_json_safe(data)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/reports/"):
            self.handle_report_download(path.removeprefix("/reports/"))
            return

        if path == "/api/status":
            self._send_json({"status": "ready", "engine": "LLM Compiler Optimizer v2.0", "target": "React Web App"})
        elif path == "/api/gpu_results":
            runs = []
            for f in sorted((PROJECT_ROOT / "docs" / "results").glob("gpu_*.json")):
                runs.append({"file": f.name, **json.loads(f.read_text())})
            self._send_json({"runs": runs})
        elif path == "/api/research_results":
            names = ["real_graph_study", "cost_calibration", "rl_results", "neuro_symbolic_results",
                     "compiler_comparison", "lm_full_eval", "learned_results", "lm_results",
                     "unseen_architecture", "learned_sensitivity"]
            out = {}
            for name in names:
                f = PROJECT_ROOT / "docs" / "results" / f"{name}.json"
                if f.exists():
                    data = json.loads(f.read_text())
                    if name == "neuro_symbolic_results":
                        data.pop("cases", None)  # keep the payload small
                    if name == "cost_calibration":
                        data = {k: data[k] for k in data if k in (
                            "device_name", "validation_held_out_architectures",
                            "in_sample_train_architectures", "train_architectures", "sizes",
                            "timestamp")}
                    out[name] = data
            self._send_json(out)
        elif path == "/api/models":
            models = []
            for model_id, (name, builder, desc) in MODEL_REGISTRY.items():
                models.append({"id": model_id, "name": name, "desc": desc,
                               "ops": builder(4, 32, 64).node_count()})
            self._send_json({"models": models, "passes": [
                {"id": k, "label": label} for k, label in STAGE_ORDER]})
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)
        data = json.loads(body_bytes) if body_bytes else {}

        if path == "/api/optimize":
            self.handle_optimize(data)
        elif path == "/api/dag_analysis":
            self.handle_dag_analysis(data)
        elif path == "/api/merge_analysis":
            self.handle_merge_analysis(data)
        elif path == "/api/benchmark_sweep":
            self.handle_benchmark_sweep(data)
        elif path == "/api/fx_attention":
            self.handle_fx_attention(data)
        elif path == "/api/xla_benchmark":
            self.handle_xla_benchmark(data)
        elif path == "/api/architectures":
            self.handle_architecture_comparison(data)
        elif path == "/api/learned":
            self.handle_learned(data)
        elif path == "/api/neuro_symbolic":
            self.handle_neuro_symbolic(data)
        elif path == "/api/ablation":
            self.handle_ablation(data)
        elif path == "/api/export_report":
            self.handle_export_report(data)
        else:
            self._send_json({"error": "Unknown POST endpoint"}, status=404)

    def _get_model_graph(self, model_id, batch=4, seq=32, d_model=64):
        entry = MODEL_REGISTRY.get(model_id) or MODEL_REGISTRY["postln_2b"]
        return entry[1](batch, seq, d_model)

    def handle_optimize(self, data):
        model_id = data.get("model_id", "postln_2b")
        flags = _passes_from(data)
        num_partitions = max(2, min(8, int(data.get("num_partitions", 2))))
        mode = data.get("partition_mode", "graph")
        mode = mode if mode in ("graph", "hypergraph", "latency") else "graph"
        hypergraph = False
        batch = int(data.get("batch", 4))
        seq = int(data.get("seq", 32))
        d_model = int(data.get("d_model", 64))

        orig_graph = self._get_model_graph(model_id, batch=batch, seq=seq, d_model=d_model)
        opt_graph, infos, compile_time_s = _run(orig_graph, flags, num_partitions, hypergraph, mode)
        metrics = compute_metrics(orig_graph, opt_graph, infos, compile_time_s, batch_size=batch)

        self._send_json({
            "model_id": model_id,
            "model_name": MODEL_REGISTRY.get(model_id, MODEL_REGISTRY["postln_2b"])[0],
            "passes": flags,
            "original_graph": graph_to_d3(orig_graph),
            "optimized_graph": graph_to_d3(opt_graph),
            "metrics": metrics,
            "stages": _stage_table(orig_graph, flags, num_partitions, hypergraph, mode),
            "deployment": {
                "original_us": deployment_latency_us(orig_graph),
                "optimized_us": deployment_latency_us(opt_graph),
                "devices": opt_graph.meta.get("partition", {}).get("partitions", 1),
            },
            "pass_info": infos,
            "schedule_info": opt_graph.meta.get("schedule", {}),
            "partition_info": opt_graph.meta.get("partition", {}),
        })

    def handle_architecture_comparison(self, data):
        """Run the same pass selection on every registered architecture."""
        flags = _passes_from(data)
        batch = int(data.get("batch", 4))
        seq = int(data.get("seq", 32))
        d_model = int(data.get("d_model", 64))
        rows = []
        for model_id, (name, builder, _) in MODEL_REGISTRY.items():
            if model_id.startswith("op_"):
                continue
            rows.append({"id": model_id, **_architecture_row(
                name, builder(batch, seq, d_model), flags, 2, False, batch)})
        self._send_json({"rows": rows, "passes": flags})

    def handle_learned(self, data):
        """Ask the trained GNN for a plan; compare with fixed + oracle."""
        from search import learned_optimizer as lo

        if not lo.WEIGHTS_PATH.exists():
            self._send_json({"error": "Learned optimizer not trained. Run: python -m search.learned_optimizer train"}, status=503)
            return
        global _LEARNED
        if _LEARNED is None:
            _LEARNED = lo.LearnedOptimizer.load()
        model_id = data.get("model_id", "postln_2b")
        graph = self._get_model_graph(model_id, int(data.get("batch", 4)),
                                      int(data.get("seq", 32)), int(data.get("d_model", 64)))
        plan = _LEARNED.plan(graph)
        true = lo.evaluate_actions(graph)
        chosen = lo.ACTIONS.index(tuple(plan["action"]))
        fixed = lo.ACTIONS.index((True, True, True, 1))
        best = max(range(len(true)), key=lambda i: true[i])
        order = sorted(range(len(true)), key=lambda i: -plan["predicted"][i])[:6]
        flags, k = lo.action_flags(plan["action"])
        rl_info = None
        from search import rl_policy as rlp
        if rlp.WEIGHTS.exists():
            global _RL
            if _RL is None:
                _RL = rlp.RLPolicy.load()
            final_g, actions = _RL.plan(graph)
            n_pass = sum(a in rlp.PASSES for a in actions)
            devices = final_g.meta.get("partition", {}).get("partitions", 1) if any(a.startswith("place") for a in actions) else 1
            from utils.cost_model import deployment_latency_us
            base = deployment_latency_us(graph)
            rl_info = {"actions": actions, "devices": devices,
                       "true_reward": rlp._reward(base, final_g, n_pass, devices)}
        results = lo.RESULTS_PATH.read_text() if lo.RESULTS_PATH.exists() else "{}"
        self._send_json({
            "model_id": model_id,
            "chosen": {"label": plan["label"], "flags": flags, "devices": k,
                       "predicted_reward": plan["predicted"][chosen],
                       "true_reward": true[chosen]},
            "fixed_pipeline": {"label": lo.action_label(lo.ACTIONS[fixed]), "true_reward": true[fixed]},
            "oracle": {"label": lo.action_label(lo.ACTIONS[best]), "true_reward": true[best]},
            "top_predicted": [{"label": lo.action_label(lo.ACTIONS[i]),
                               "predicted": plan["predicted"][i], "true": true[i]} for i in order],
            "rl_policy": rl_info,
            "training": json.loads(results),
            "reward_definition": f"deployment speedup − {lo.DEVICE_PENALTY}·(devices−1) − {lo.PASS_PENALTY}·(#passes); modeled, not measured",
        })

    def handle_neuro_symbolic(self, data):
        """Latency-GNN-guided rewrite search on the selected model."""
        from search import neuro_symbolic as nsm

        graph = self._get_model_graph(data.get("model_id", "postln_2b"), int(data.get("batch", 4)),
                                      int(data.get("seq", 32)), int(data.get("d_model", 64)))
        result, info = nsm.neuro_symbolic_search(graph)
        fixed, _, _ = run_pipeline(graph, partition=False)
        metrics = compute_metrics(graph, result, {}, info["duration_s"], batch_size=int(data.get("batch", 4)))
        self._send_json({
            "guidance": info["guidance"], "rewrites_applied": info["applied"],
            "nodes": {"original": graph.node_count(), "neuro_symbolic": result.node_count(),
                      "fixed_pipeline": fixed.node_count()},
            "accuracy_preserved": metrics["accuracy"]["preserved"],
            "max_abs_diff": metrics["accuracy"]["max_abs_diff"],
            "search_seconds": info["duration_s"], "gpu_measurements_used": 0,
            "optimized_graph": graph_to_d3(result),
        })

    def handle_ablation(self, data):
        """Each pass alone, then all together: attributes the gains."""
        model_id = data.get("model_id", "postln_2b")
        batch = int(data.get("batch", 4))
        seq = int(data.get("seq", 32))
        d_model = int(data.get("d_model", 64))
        graph = self._get_model_graph(model_id, batch, seq, d_model)
        none = {k: False for k in DEFAULT_PASSES}
        configs = [("Baseline (no passes)", none)]
        for key, label in STAGE_ORDER:
            configs.append((f"{label} only", {**none, key: True}))
        configs.append(("Normalize + Canonicalize", {**none, "normalize": True, "canonicalize": True}))
        configs.append(("Normalize + Merge", {**none, "normalize": True, "merge": True}))
        configs.append(("All passes", {**DEFAULT_PASSES, "partition": True}))
        rows = [_architecture_row(label, graph, flags, 2, False, batch)
                for label, flags in configs]
        self._send_json({"model_id": model_id, "rows": rows})

    def handle_export_report(self, data):
        """Create a distinct Excel workbook for a completed UI comparison."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        created = datetime.now(timezone.utc)
        run_id = f"{created.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid4().hex[:8]}"
        filename = f"metrics_{run_id}.xlsx"
        report = {**data, "run_id": run_id,
                  "created_utc": created.isoformat(timespec="seconds")}
        try:
            write_report(REPORTS_DIR / filename, _json_safe(report))
        except Exception as exc:  # report generation must not crash the API
            self._send_json({"error": f"Could not create Excel report: {exc}"}, status=500)
            return
        self._send_json({
            "run_id": run_id,
            "filename": filename,
            "download_url": f"/reports/{filename}",
        })

    def handle_report_download(self, filename):
        if not filename or Path(filename).name != filename or not filename.endswith(".xlsx"):
            self._send_json({"error": "Invalid report name"}, status=400)
            return
        report_path = REPORTS_DIR / filename
        if not report_path.is_file():
            self._send_json({"error": "Report not found"}, status=404)
            return
        content = report_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def handle_dag_analysis(self, data):
        """Detailed Dependency-Aware Graph Scheduling (DAGS) analysis."""
        model_id = data.get("model_id", "postln_2b")
        orig_graph = self._get_model_graph(model_id)
        opt_graph, infos, _ = run_pipeline(orig_graph, schedule=True, partition=False)

        schedule_info = opt_graph.meta.get("schedule", {})
        levels_map = {}
        for node_id, d in opt_graph.graph.nodes(data=True):
            lvl = d.get("schedule_level", 0)
            if lvl not in levels_map:
                levels_map[lvl] = []
            levels_map[lvl].append({
                "id": node_id,
                "op_type": d.get("op_type"),
                "order": d.get("schedule_order"),
                "shape": str(d.get("shape", "")),
            })

        levels_list = [{"level": lvl, "nodes": levels_map[lvl]} for lvl in sorted(levels_map.keys())]

        seq_lat = sequential_latency_us(opt_graph)
        par_lat = scheduled_latency_us(opt_graph)
        naive_mem = naive_peak_memory_mb(opt_graph)
        live_mem = live_range_peak_memory_mb(opt_graph)

        self._send_json({
            "levels": levels_list,
            "num_levels": schedule_info.get("levels", len(levels_list)),
            "critical_path": schedule_info.get("critical_path", []),
            "sequential_makespan_us": seq_lat,
            "parallel_makespan_us": par_lat,
            "makespan_speedup": (seq_lat / par_lat) if par_lat > 0 else 1.0,
            "naive_memory_mb": naive_mem,
            "buffer_reuse_memory_mb": live_mem,
            "memory_reduction_pct": ((naive_mem - live_mem) / naive_mem * 100) if naive_mem > 0 else 0,
        })

    def handle_merge_analysis(self, data):
        """Detailed breakdown of Semantic Operator Merging patterns & provenance."""
        model_id = data.get("model_id", "postln_2b")
        orig_graph = self._get_model_graph(model_id)
        opt_graph, infos, _ = run_pipeline(orig_graph, normalize=True, canonicalize=True, merge=True)

        merge_info = infos.get("semantic_merging", {})
        fused_nodes = []
        for node_id, d in opt_graph.graph.nodes(data=True):
            if d.get("fused_from"):
                fused_nodes.append({
                    "fused_node": node_id,
                    "kernel_type": d.get("op_type"),
                    "absorbed_nodes": d.get("fused_from"),
                    "shape": str(d.get("shape", "")),
                    "bias_folded": d.get("bias_value") is not None,
                })

        patterns_supported = [
            {"pattern": "MatMul + Add", "target": "GEMM", "benefit": "Blas GEMM with constant bias absorption, eliminating intermediate tensor."},
            {"pattern": "GEMM + GELU", "target": "LinearGELU", "benefit": "Fused linear projection + activation epilogue; zero HBM write."},
            {"pattern": "Add + LayerNorm", "target": "FusedAddLayerNorm", "benefit": "Fused residual addition and mean/variance reduction."},
            {"pattern": "Q/K/V + QK^T + Scale + Softmax + AV", "target": "FusedAttention", "benefit": "Full FlashAttention sandwich; zero intermediate score tensor traffic."},
            {"pattern": "Scale + Softmax", "target": "FusedScaleSoftmax", "benefit": "Inline scaled softmax reduction kernel."},
        ]

        self._send_json({
            "fused_nodes": fused_nodes,
            "total_fused": len(fused_nodes),
            "nodes_absorbed": merge_info.get("nodes_removed", 0),
            "patterns": patterns_supported,
        })

    def handle_benchmark_sweep(self, data):
        """Run the 13-workload parameter sweep and return latency/throughput/correctness table."""
        from run_attention_optimization import run_case
        cases = data.get("cases") or [
            (1, 32, 64), (2, 64, 64), (4, 128, 64), (8, 256, 64), (16, 512, 64),
            (2, 64, 128), (4, 128, 128), (8, 256, 128),
            (2, 64, 256), (4, 128, 256), (8, 256, 256),
            (2, 128, 512), (4, 256, 512),
        ]
        cases = [(int(case["batch"]), int(case["seq"]), int(case["d_model"]))
                 if isinstance(case, dict) else tuple(map(int, case))
                 for case in cases[:16]]
        warmup = max(1, min(20, int(data.get("warmup", 20))))
        runs = max(3, min(100, int(data.get("runs", 100))))
        results = []
        for b, s, d in cases:
            res = run_case(b, s, d, warmup=warmup, runs=runs)
            results.append({
                "batch": res["batch"],
                "seq": res["seq"],
                "d_model": res["d_model"],
                "original_nodes": res["original_nodes"],
                "optimized_nodes": res["optimized_nodes"],
                "normalized_nodes": res["normalized_nodes"],
                "grr": res["grr"],
                "omr": res["omr"],
                "acr": res["acr"],
                "kernel_launches_original_estimate": res["kernel_launches_original_estimate"],
                "kernel_launches_optimized_estimate": res["kernel_launches_optimized_estimate"],
                "kernel_launch_reduction_estimate": res["kernel_launch_reduction_estimate"],
                "compilation_time_ms": res["compilation_time_ms"],
                "peak_gpu_memory_mb": res["peak_gpu_memory_mb"],
                "original_latency_ms": res["original_latency"],
                "optimized_latency_ms": res["optimized_latency"],
                "speedup": res["speedup"],
                "throughput_orig": res["throughput_original"],
                "throughput_opt": res["throughput_optimized"],
                "max_error": res["max_error"],
                "correct": res["correct"],
            })

        summary = {
            "avg_grr": statistics.mean(r["grr"] for r in results),
            "median_speedup": statistics.median(r["speedup"] for r in results),
            "max_speedup": max(r["speedup"] for r in results),
            "min_speedup": min(r["speedup"] for r in results),
            "all_correct": all(r["correct"] for r in results),
        }
        self._send_json({"results": results, "summary": summary,
                         "warmup_runs": warmup, "timed_runs": runs})

    def handle_fx_attention(self, data):
        """Run PyTorch FX single trace & semantic merge."""
        import torch
        import torch.fx as fx
        from models.attention import AttentionModel
        from passes.semantic_merge import semantic_merge_attention
        from passes.normalize import normalize as normalize_fx
        from ir.fx_graph import count_compute_nodes

        d_model = int(data.get("d_model", 64))
        batch = int(data.get("batch", 8))
        seq = int(data.get("seq", 32))

        model = AttentionModel(d_model=d_model)
        model.eval()

        orig = fx.symbolic_trace(model)
        orig_nodes = count_compute_nodes(orig)
        orig_code = str(orig.code).strip()

        opt = fx.symbolic_trace(model)
        opt = normalize_fx(opt)
        merged = semantic_merge_attention(opt)
        opt_nodes = count_compute_nodes(opt)
        opt_code = str(opt.code).strip()

        self._send_json({
            "original_nodes": orig_nodes,
            "optimized_nodes": opt_nodes,
            "grr": ((orig_nodes - opt_nodes) / orig_nodes * 100) if orig_nodes else 0,
            "original_code": orig_code,
            "optimized_code": opt_code,
            "merged_successfully": merged,
        })

    def handle_xla_benchmark(self, data):
        """Same-workload PyTorch/FX versus XLA benchmark for the dashboard."""
        from xla_compare import compare_attention_xla

        try:
            result = compare_attention_xla(
                batch=int(data.get("batch", 4)),
                seq=int(data.get("seq", 64)),
                d_model=int(data.get("d_model", 64)),
                warmup=int(data.get("warmup", 8)),
                runs=int(data.get("runs", 40)),
            )
        except RuntimeError as exc:
            self._send_json({"error": str(exc), "available": False}, status=503)
            return
        self._send_json(result)


def run_server(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, CompilerAPIHandler)
    print(f"[API Server] Listening on http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
