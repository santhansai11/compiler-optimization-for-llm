"""Neuro-symbolic search study on a real GPU.

1. DATA   - random rewrite trajectories (symbolic rules) from 6 architectures
            x 4 sizes x 3 variants; every intermediate graph is TIMED on the
            real GPU (FusedAttention lowered to the fixed fused kernel).
2. LOFO   - train the latency GNN on 5 architectures, then on the held-out one:
            (a) latency-prediction quality, (b) guided search vs baselines,
            each final graph MEASURED (interleaved) and checked for identical
            outputs.
3. FINAL  - train on everything and save search/ns_value.pt.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from benchmarks.gpu_benchmark import BUILDERS  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from search import learned_optimizer as lo  # noqa: E402
from search import neuro_symbolic as ns  # noqa: E402
from search.rewrites import apply_rewrite, enumerate_rewrites  # noqa: E402
from utils.torch_executor import device_name, gpu_compare, pick_device, time_graph  # noqa: E402

TRAIN_SIZES = [(2, 128, 128), (4, 64, 64), (8, 256, 256), (4, 256, 512)]
EVAL_SIZES = TRAIN_SIZES + [(8, 512, 512)]
CACHE = ROOT / "training" / "artifacts" / "ns_dataset.pkl"
OUT = ROOT / "docs" / "results" / "neuro_symbolic_results.json"


def variants(arch, size, rng):
    """Base graph + 2 copies with injected redundancy (identity / scale pairs)."""
    out = [BUILDERS[arch](*size)]
    for _ in range(2):
        out.append(lo._inject_redundancy(BUILDERS[arch](*size), rng))
    return out


def random_trajectory(graph, rng, max_len):
    graphs, cur = [graph], graph
    for _ in range(int(rng.integers(1, max_len + 1))):
        cands = enumerate_rewrites(cur)
        if not cands:
            break
        rule, match = cands[int(rng.integers(len(cands)))]
        new = apply_rewrite(cur, rule, match)
        if new is None:
            continue
        graphs.append(new)
        cur = new
    return graphs


def build_dataset(device, seed=0):
    if CACHE.exists():
        d = pickle.loads(CACHE.read_bytes())
        if d["device"] == device_name(device):
            return d["samples"]
    rng = np.random.default_rng(seed)
    samples, t0 = [], time.time()
    for arch in BUILDERS:
        for size in TRAIN_SIZES:
            for base in variants(arch, size, rng):
                for _ in range(2):
                    for g in random_trajectory(base, rng, 10):
                        t = time_graph(g, device, "sdpa")
                        samples.append({"arch": arch, "size": size,
                                        "features": ns.featurize_sized(g),
                                        "log_us": float(np.log(t * 1e6)),
                                        "nodes": g.node_count()})
        print(f"  measured {arch:24s} total {len(samples)} graphs ({time.time() - t0:.0f}s)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps({"device": device_name(device), "samples": samples}))
    return samples


def naive_fixpoint(graph):
    """Apply the first legal rewrite (rule order) until none is left."""
    cur = graph
    while True:
        for rule, match in enumerate_rewrites(cur):
            new = apply_rewrite(cur, rule, match)
            if new is not None:
                cur = new
                break
        else:
            return cur


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    device = pick_device()
    print("device:", device_name(device), flush=True)
    samples = build_dataset(device)
    print(len(samples), "measured graphs", flush=True)
    rng = np.random.default_rng(123)
    lofo, cases = {}, []
    for arch in BUILDERS:
        train = [s for s in samples if s["arch"] != arch]
        test = [s for s in samples if s["arch"] == arch]
        model, loss = ns.fit_value(train)
        pred = model.predict_log_us([s["features"] for s in test])
        true = np.array([s["log_us"] for s in test])
        quality = {"graphs": len(test), "mape_pct": float(np.mean(np.abs(np.exp(pred) - np.exp(true)) / np.exp(true)) * 100),
                   "spearman": spearman(pred, true), "train_loss": loss}
        score = lambda gs, m=model: m.predict_log_us([ns.featurize_sized(g) for g in gs])
        rows = []
        for size in EVAL_SIZES:
            for vi, g0 in enumerate(variants(arch, size, rng)[:2]):
                fixed = run_pipeline(g0, partition=False)[0]
                naive = naive_fixpoint(g0)
                guided, trace = ns.guided_search(g0, score)
                res = gpu_compare([(g0, "sdpa"), (fixed, "sdpa"), (naive, "sdpa"), (guided, "sdpa")],
                                  device, repeats=30, warmup=5)
                base = res[0]["output"]
                scale = max(1e-6, float(np.abs(base).max()))
                rel = [float(np.abs(r["output"] - base).max()) / scale for r in res[1:]]
                t0 = res[0]["latency_min_s"]
                row = {"arch": arch, "size": size, "variant": "base" if vi == 0 else "injected",
                       "nodes": [g0.node_count(), fixed.node_count(), naive.node_count(), guided.node_count()],
                       "latency_us": [r["latency_min_s"] * 1e6 for r in res],
                       "speedup_fixed": t0 / res[1]["latency_min_s"],
                       "speedup_naive": t0 / res[2]["latency_min_s"],
                       "speedup_neurosymbolic": t0 / res[3]["latency_min_s"],
                       "outputs_match": all(r < 1e-3 for r in rel),
                       "rewrites_applied": [t["rule"] for t in trace],
                       "attention_fused_by_ns": any(t["rule"] == "fuse-attention" for t in trace),
                       "gpu_measurements_at_search_time": 0}
                rows.append(row)
        cases += rows
        gm = lambda k: float(np.exp(np.mean(np.log([r[k] for r in rows]))))
        lofo[arch] = {"latency_model": quality, "geomean_speedup_fixed": gm("speedup_fixed"),
                      "geomean_speedup_naive": gm("speedup_naive"),
                      "geomean_speedup_neurosymbolic": gm("speedup_neurosymbolic"),
                      "all_outputs_match": all(r["outputs_match"] for r in rows)}
        r = lofo[arch]
        print(f"held-out {arch:24s} latency MAPE {quality['mape_pct']:5.1f}% rho {quality['spearman']:.2f} | "
              f"speedup fixed {r['geomean_speedup_fixed']:.3f} naive {r['geomean_speedup_naive']:.3f} "
              f"NS {r['geomean_speedup_neurosymbolic']:.3f} match={r['all_outputs_match']}", flush=True)
    final, loss = ns.fit_value(samples)
    torch.save(final.state_dict(), ns.WEIGHTS)
    gm = lambda k: float(np.exp(np.mean(np.log([r[k] for r in cases]))))
    summary = {k: gm(k) for k in ("speedup_fixed", "speedup_naive", "speedup_neurosymbolic")}
    summary["ns_beats_fixed_cases_pct"] = float(np.mean([r["speedup_neurosymbolic"] > r["speedup_fixed"] * 1.005 for r in cases]) * 100)
    summary["ns_worse_than_fixed_cases_pct"] = float(np.mean([r["speedup_neurosymbolic"] < r["speedup_fixed"] * 0.995 for r in cases]) * 100)
    summary["all_outputs_match"] = all(r["outputs_match"] for r in cases)
    summary["mean_latency_mape_pct"] = float(np.mean([v["latency_model"]["mape_pct"] for v in lofo.values()]))
    summary["mean_latency_spearman"] = float(np.mean([v["latency_model"]["spearman"] for v in lofo.values()]))
    out = {"device": device_name(device), "torch": torch.__version__, "train_graphs": len(samples),
           "attention_lowering": "fixed fused kernel (sdpa)", "leave_one_architecture_out": lofo,
           "summary": summary, "cases": cases, "final_train_loss": loss}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
