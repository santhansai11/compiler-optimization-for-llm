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
REPORT_BUILDER = PROJECT_ROOT / "tools" / "export_metrics.mjs"

from ir.graph import ComputationGraph
from models.transformer import create_demo_transformer_graph
from models.architectures import (
    create_postln_transformer_graph,
    create_preln_transformer_graph,
    create_relu_transformer_graph,
    create_mlp_classifier_graph,
)
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
)
from utils.executor import execute


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
        elif path == "/api/models":
            models = [
                {"id": "postln_2b", "name": "Post-LN Transformer (2 blocks)", "ops": 42, "desc": "Reference Vaswani et al. architecture"},
                {"id": "postln_4b", "name": "Post-LN Transformer (4 blocks)", "ops": 78, "desc": "Deeper 4-block transformer DAG"},
                {"id": "preln_2b", "name": "Pre-LN Transformer (GPT/LLaMA)", "ops": 36, "desc": "LayerNorm placed inside residual branches"},
                {"id": "relu_2b", "name": "Post-LN + ReLU FFN", "ops": 42, "desc": "Exercises LinearRelu fusion pathway"},
                {"id": "mlp_classifier", "name": "MLP Classifier (784-256-64-10)", "ops": 13, "desc": "Feedforward classifier trained on MNIST"},
            ]
            self._send_json({"models": models})
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
        elif path == "/api/export_report":
            self.handle_export_report(data)
        else:
            self._send_json({"error": "Unknown POST endpoint"}, status=404)

    def _get_model_graph(self, model_id, batch=4, seq=32, d_model=64):
        if model_id == "postln_4b":
            return create_demo_transformer_graph(num_blocks=4, batch=batch, seq=seq, d_model=d_model)
        elif model_id == "preln_2b":
            return create_preln_transformer_graph(num_blocks=2, batch=batch, seq=seq, d_model=d_model)
        elif model_id == "relu_2b":
            return create_relu_transformer_graph(num_blocks=2, batch=batch, seq=seq, d_model=d_model)
        elif model_id == "mlp_classifier":
            return create_mlp_classifier_graph(batch=256)
        else:
            return create_demo_transformer_graph(num_blocks=2, batch=batch, seq=seq, d_model=d_model)

    def handle_optimize(self, data):
        model_id = data.get("model_id", "postln_2b")
        # Keep this experiment scoped to graph normalization and semantic merging.
        passes = data.get("passes", {"normalize": True, "merge": True})

        batch = int(data.get("batch", 4))
        seq = int(data.get("seq", 32))
        d_model = int(data.get("d_model", 64))

        orig_graph = self._get_model_graph(model_id, batch=batch, seq=seq, d_model=d_model)

        opt_graph, infos, compile_time_s = run_pipeline(
            orig_graph,
            normalize=passes.get("normalize", True),
            canonicalize=False,
            merge=passes.get("merge", True),
            schedule=False,
            partition=False,
        )

        metrics = compute_metrics(orig_graph, opt_graph, infos, compile_time_s, batch_size=batch)

        # Stage progression counts
        stages = [
            {"stage": "Original Graph", "ops": orig_graph.node_count()},
            {"stage": "After Normalization", "ops": infos.get("normalize", {}).get("nodes_after", orig_graph.node_count())},
            {"stage": "After Semantic Operator Merging", "ops": infos.get("semantic_merging", {}).get("nodes_after", opt_graph.node_count())},
        ]

        response = {
            "original_graph": graph_to_d3(orig_graph),
            "optimized_graph": graph_to_d3(opt_graph),
            "metrics": metrics,
            "stages": stages,
            "pass_info": infos,
            "schedule_info": opt_graph.meta.get("schedule", {}),
            "partition_info": opt_graph.meta.get("partition", {}),
        }
        self._send_json(response)

    def handle_export_report(self, data):
        """Create a distinct Excel workbook for a completed UI comparison."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        created = datetime.now(timezone.utc)
        run_id = f"{created.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid4().hex[:8]}"
        filename = f"metrics_{run_id}.xlsx"
        report = {
            **data,
            "run_id": run_id,
            "created_utc": created.isoformat(timespec="seconds"),
        }
        node = shutil.which("node")
        if not node:
            candidates = [
                Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node.exe",
            ]
            node_path = next((candidate for candidate in candidates if candidate.exists()), None)
            node = str(node_path) if node_path else None
        if not node or not REPORT_BUILDER.exists():
            self._send_json({"error": "Excel export runtime is unavailable; install Node.js to generate .xlsx reports."}, status=503)
            return
        try:
            exported = subprocess.run(
                [node, str(REPORT_BUILDER), str(REPORTS_DIR / filename)],
                input=json.dumps(report, default=_json_safe),
                text=True,
                capture_output=True,
                cwd=str(REPORT_BUILDER.parent),
                timeout=45,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._send_json({"error": f"Could not create Excel report: {exc}"}, status=500)
            return
        # artifact-tool writes an inspection sidecar for desktop review; the
        # application only needs the portable workbook in its reports folder.
        sidecar = Path(str(REPORTS_DIR / filename) + ".inspect.ndjson")
        sidecar.unlink(missing_ok=True)
        sheets = ["Summary"]
        if report.get("xla"):
            sheets.append("XLA")
        if report.get("sweep", {}).get("results"):
            sheets.append("Workload sweep")
        if report.get("stages"):
            sheets.append("Graph stages")
        self._send_json({
            "run_id": run_id,
            "filename": filename,
            "download_url": f"/reports/{filename}",
            "sheets": sheets,
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
