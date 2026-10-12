"""Verify real-LLM graph import end to end: import -> compile -> verify.

Usage:
    python verify_real_model.py tiny-gpt2
    python verify_real_model.py distilgpt2
    python verify_real_model.py gpt2
"""

import json
import sys

from models.import_hf import import_hf_graph
from pipeline import run_pipeline
from utils.metrics import compute_metrics, format_report


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "tiny-gpt2"
    print(f"=== Importing real LLM graph: {model_key} ===")
    graph, info = import_hf_graph(model_key)
    print(f"torch.fx nodes traced : {info['torch.fx nodes']}")
    print(f"IR ops emitted        : {info['ops_emitted']} "
          f"({info['parameters'] / 1e6:.1f}M imported parameters)")
    print(f"op histogram          : {json.dumps(info['op_histogram'])}")
    print(f"outputs               : "
          f"{[n for n, d in graph.nodes(data=True) if d.get('is_output')]}")
    print(f"config                : {graph.meta['config']}")

    print("\n=== Compiling the real graph with the pipeline ===")
    optimized, infos, compile_time = run_pipeline(
        graph, normalize=True, canonicalize=True, merge=True,
        schedule=True, partition=False,
    )
    ns = infos.get("attention_canonicalization", {})
    print(f"attention clusters    : {ns.get('detected', 0)} detected, "
          f"{ns.get('canonicalized', 0)} canonicalized")

    print("\n=== Measuring + verifying (NumPy executor) ===")
    metrics = compute_metrics(graph, optimized, infos, compile_time)
    print(format_report(metrics))


if __name__ == "__main__":
    main()
