"""Comparable CPU benchmark for eager PyTorch, FX semantic merge, and XLA.

All three backends run the same attention equations, weights, and input.
XLA is accessed through JAX because it is a supported Python frontend to the
OpenXLA compiler; the module is imported lazily so the compiler UI can still
start when the optional backend has not been installed.
"""

import re
import statistics
import time

import numpy as np
import torch
import torch.fx as fx

from ir.fx_graph import count_compute_nodes
from models.attention import AttentionModel
from passes.dags_fx import schedule_fx
from passes.normalize import normalize_fx
from passes.semantic_merge import semantic_merge_attention


def _median_latency(call, warmup, runs):
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            output = call()
            if hasattr(output, "block_until_ready"):
                output.block_until_ready()
            samples.append((time.perf_counter() - start) * 1_000)
    return output, statistics.median(samples)


def compare_attention_xla(batch=4, seq=64, d_model=64, warmup=8, runs=40):
    """Benchmark one same-input attention workload across three compilers."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError(
            "XLA comparison needs JAX. Install project dependencies with "
            "'.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
        ) from exc

    torch.manual_seed(42)
    model = AttentionModel(d_model=d_model).cpu().eval()
    x = torch.randn(batch, seq, d_model, dtype=torch.float32)

    original_fx = fx.symbolic_trace(model)
    normalized_fx = fx.symbolic_trace(model)
    normalize_fx(normalized_fx)
    normalized_nodes = count_compute_nodes(normalized_fx)
    optimized_fx = normalized_fx
    merged = semantic_merge_attention(optimized_fx)
    if not merged:
        raise RuntimeError("The FX attention pattern did not match.")
    optimized_fx, dag = schedule_fx(optimized_fx)

    original_out, original_ms = _median_latency(
        lambda: original_fx(x), warmup, runs
    )
    optimized_out, optimized_ms = _median_latency(
        lambda: optimized_fx(x), warmup, runs
    )

    x_np = x.numpy()
    wq, wk, wv = (p.detach().numpy() for p in (model.wq, model.wk, model.wv))
    jx, jq, jk, jv = map(jnp.asarray, (x_np, wq, wk, wv))

    def attention(a, q_weight, k_weight, v_weight):
        q = a @ q_weight
        k = a @ k_weight
        v = a @ v_weight
        scores = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(
            jnp.asarray(a.shape[-1], dtype=a.dtype)
        )
        weights = jax.nn.softmax(scores, axis=-1)
        return weights @ v

    xla_fn = jax.jit(attention)
    compile_started = time.perf_counter()
    lowered = xla_fn.lower(jx, jq, jk, jv)
    compiled = lowered.compile()
    compile_ms = (time.perf_counter() - compile_started) * 1_000
    xla_out, xla_ms = _median_latency(
        lambda: compiled(jx, jq, jk, jv), warmup, runs
    )

    xla_array = np.asarray(xla_out)
    torch_array = original_out.detach().numpy()
    optimized_array = optimized_out.detach().numpy()
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    stablehlo_ops = len(re.findall(r"\bstablehlo\.[A-Za-z0-9_]+", stablehlo))

    def backend(name, latency_ms, output, extra=None):
        payload = {
            "name": name,
            "latency_ms": float(latency_ms),
            "throughput_tokens_s": float(batch * seq / (latency_ms / 1000)),
            "max_abs_error_vs_torch": float(
                np.max(np.abs(torch_array - output))
            ),
            "correct_vs_torch": bool(np.allclose(
                torch_array, output, rtol=1e-4, atol=1e-5
            )),
        }
        if extra:
            payload.update(extra)
        return payload

    backends = [
        backend("PyTorch eager", original_ms, torch_array, {
            "fx_compute_ops": count_compute_nodes(original_fx),
        }),
        backend("PyTorch FX semantic (reference)", optimized_ms, optimized_array, {
            "fx_compute_ops": count_compute_nodes(optimized_fx),
            "fx_normalized_ops": normalized_nodes,
            "graph_reduction_pct": (
                (normalized_nodes
                 - count_compute_nodes(optimized_fx))
                / max(1, normalized_nodes) * 100
            ),
        }),
        backend("XLA (JAX CPU)", xla_ms, xla_array, {
            "compile_ms": float(compile_ms),
            "stablehlo_ops": stablehlo_ops,
            "device": str(jax.devices()[0]),
        }),
    ]
    return {
        "workload": {"batch": batch, "seq": seq, "d_model": d_model,
                     "dtype": "float32", "device": "CPU"},
        "warmup_runs": warmup,
        "timed_runs": runs,
        "backends": backends,
        "dag_schedule": dag,
        "normalization": {
            "fx_nodes_before": count_compute_nodes(original_fx),
            "fx_nodes_after": normalized_nodes,
            "nodes_removed": count_compute_nodes(original_fx) - normalized_nodes,
        },
        "xla_ir": stablehlo,
        "all_correct": all(item["correct_vs_torch"] for item in backends),
        "comparison_note": (
            "Same attention function, weights, input, float32, CPU. "
            "Latencies are per-call medians after warmup. XLA compile time "
            "is reported separately. The FX fusion is a PyTorch reference "
            "function, not a custom GPU kernel. FX node counts and StableHLO operation "
            "counts use different IRs and are not directly equivalent."
        ),
    }
