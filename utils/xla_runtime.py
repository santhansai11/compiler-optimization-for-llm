"""Explicit XLA CPU compilation, saved optimised HLO, and storage accounting."""

import hashlib
import json
import re
import time
from pathlib import Path

import jax
import numpy as np

from utils import jax_executor
from utils.executor import _make_feed, execute


def serializable(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def graph_record(graph):
    # Incoming edge order is operand order, so do NOT sort predecessors.
    return {"name": graph.name, "config": serializable(graph.meta.get("config", {})),
            "nodes": [{"name": n, "attrs": serializable(d), "inputs": graph.predecessors(n)}
                      for n, d in graph.nodes(data=True)]}


def graph_key(graph):
    return hashlib.sha256(json.dumps(graph_record(graph), sort_keys=True).encode()).hexdigest()


def normalized_hlo(text):
    """Remove provenance only; equality is textual, not a semantic proof."""
    text = re.sub(r",? metadata=\{[^\n]*?\}", "", text)
    text = re.sub(r"(?m)^HloModule [^,\n]+", "HloModule module", text)
    # Recent XLA dumps store source provenance in module-level debug tables,
    # as well as per-instruction metadata. Do not mistake these for changed ops.
    lines, in_debug_table = [], False
    for line in text.splitlines():
        if line.strip() in {"FileNames", "FunctionNames", "FileLocations", "StackFrames"}:
            in_debug_table = True
            continue
        if in_debug_table:
            if not line.strip():
                in_debug_table = False
            continue
        if line.strip():
            lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


class XlaProgram:
    def __init__(self, graph, artifact_dir=None):
        self.key = graph_key(graph)
        self.graph = graph
        self.feed = _make_feed(graph, None)
        self.device = jax.devices("cpu")[0]
        with jax.default_device(self.device):
            self.x = jax.device_put(self.feed, self.device)
            start = time.perf_counter()
            self.lowered = jax.jit(jax_executor.build(graph)).lower(self.x)
            self.compiled = self.lowered.compile()
            self.compile_ms = (time.perf_counter() - start) * 1000
        self.hlo = self.compiled.as_text()
        self.output = np.asarray(self())
        expected = execute(graph, repeats=1, warmup=0, feed=self.feed)["output"]
        if self.output.shape != expected.shape or not np.allclose(self.output, expected, rtol=1e-4, atol=1e-5):
            raise ValueError(f"XLA output differs from NumPy for {graph.name}")
        self.reference_max_abs_error = float(np.max(np.abs(self.output - expected)))
        analysis = self.compiled.memory_analysis()
        self.memory = {k: getattr(analysis, k, None) for k in (
            "temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes")}
        self.memory["definition"] = "XLA compiler allocation analysis in bytes; not measured process/GPU peak memory"
        if artifact_dir:
            folder = Path(artifact_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{self.key}.optimized.hlo").write_text(self.hlo, encoding="utf-8")
            (folder / f"{self.key}.stablehlo.mlir").write_text(str(self.lowered.compiler_ir(dialect="stablehlo")), encoding="utf-8")
            (folder / f"{self.key}.graph.json").write_text(json.dumps(graph_record(graph), indent=2), encoding="utf-8")

    def __call__(self):
        return self.compiled(self.x).block_until_ready()

    def record(self):
        return {"graph_id": self.key, "compile_ms": self.compile_ms,
                "device": str(self.device), "nodes": self.graph.node_count(),
                "memory": self.memory, "numpy_max_abs_error": self.reference_max_abs_error,
                "optimized_hlo_sha256": hashlib.sha256(self.hlo.encode()).hexdigest(),
                "normalized_hlo_sha256": hashlib.sha256(normalized_hlo(self.hlo).encode()).hexdigest()}
