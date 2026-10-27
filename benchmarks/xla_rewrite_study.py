"""Rewrite guidance evaluated on the target it guides: XLA (JAX, CPU).

1. DATA      random rewrite trajectories from every workload. At each state
             the state and a sample of its legal one-step rewrites are
             XLA-compiled and timed together (interleaved, shuffled rounds),
             so every training label is an XLA latency in microseconds.
2. PAIRS     each (graph, rewrite, graph') step has a measured XLA speedup.
             Leave-one-architecture-out, a latency GNN trained on XLA
             timings predicts it; reported: graph latency error (us), pair
             speedup error and rank correlation, and the accept / reject
             confusion of the pair decision at several thresholds.
3. POLICIES  per workload (model never trained on its architecture):
               none          original graph
               all_safe      every legal rewrite, to a fixpoint
               guided        GNN-guided search (search.neuro_symbolic)
               random_same   as many random legal rewrites as guided applied
               fixed_no_cse  every legal rewrite except CSE
             Each variant is compiled once and timed in ``--rounds`` rounds,
             variant order shuffled every round; raw samples are saved and
             compared pairwise (bootstrap CI, Wilcoxon, Holm).
4. HLO       optimised HLO of every variant is fingerprinted (utils.hlo) to
             show whether a rewrite changes the program XLA runs; HLO text
             and diffs of representative single rewrites are saved next to
             the result file (``<out>_hlo/``).

    python benchmarks/xla_rewrite_study.py                  # full run
    python benchmarks/xla_rewrite_study.py --quick          # smoke test
    python benchmarks/xla_rewrite_study.py --single-thread  # 1 XLA thread
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

if "--single-thread" in sys.argv:  # must be set before jax is imported
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
                               " intra_op_parallelism_threads=1").strip()

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmarks.gpu_benchmark import BUILDERS  # noqa: E402
from search import learned_optimizer as lo  # noqa: E402
from search import neuro_symbolic as ns  # noqa: E402
from search.rewrites import apply_all, apply_random, apply_rewrite, enumerate_rewrites  # noqa: E402
from utils import environment, hlo, jax_executor, stats  # noqa: E402
from utils.torch_executor import _feed  # noqa: E402

OUT = ROOT / "docs" / "results" / "xla_rewrite_study.json"
POLICIES = ["none", "all_safe", "guided", "random_same", "fixed_no_cse"]
THRESHOLDS = [1.0, 1.02, 1.05]
NOISE = 0.01  # |speedup - 1| below this counts as "neutral"
FULL = {"sizes": [(4, 64, 64), (4, 128, 128), (8, 256, 256)], "trajectories": 2, "max_len": 10,
        "candidates": 6, "label_rounds": 9, "rounds": 30, "epochs": 400}
QUICK = {"sizes": [(2, 32, 32)], "trajectories": 1, "max_len": 3,
         "candidates": 2, "label_rounds": 3, "rounds": 5, "epochs": 40}


# ----------------------------------------------------------------- workloads
def workloads(sizes):
    """Every architecture x size, as built and with injected redundancy."""
    out = []
    for arch in BUILDERS:
        for size in sizes:
            for variant in ("base", "injected"):
                wid = f"{arch} | B{size[0]} S{size[1]} D{size[2]} | {variant}"
                g = BUILDERS[arch](*size)
                if variant == "injected":
                    seed = int(hashlib.sha1(wid.encode()).hexdigest()[:8], 16)
                    g = lo._inject_redundancy(g, np.random.default_rng(seed))
                out.append({"id": wid, "arch": arch, "size": list(size), "variant": variant, "graph": g})
    return out


def graph_key(graph):
    """Identity of an IR graph and its tensor sizes (same key => same XLA input)."""
    g = graph.graph
    rows = sorted((n, d.get("op_type"), tuple(g.predecessors(n)), d.get("factor"),
                   bool(d.get("transpose_rhs")), tuple(d.get("fused_from") or ()),
                   d.get("bias") is not None, bool(d.get("is_output")))
                  for n, d in g.nodes(data=True))
    config = sorted((graph.meta.get("config") or {}).items())
    return hashlib.sha1(repr((config, rows)).encode()).hexdigest()[:16]


# -------------------------------------------------------------------- timing
class Compiled:
    def __init__(self, graph):
        x = jnp.asarray(_feed(graph))
        lowered = jax.jit(jax_executor.build(graph)).lower(x)
        t = time.perf_counter()
        self.fn = lowered.compile()
        self.compile_ms = (time.perf_counter() - t) * 1e3
        self.x = x
        # XLA embeds absolute source paths; keep saved HLO machine-independent
        self.hlo_text = self.fn.as_text().replace(str(ROOT) + "/", "")
        self.hlo = hlo.summary(self.hlo_text)
        try:
            self.temp_bytes = int(self.fn.memory_analysis().temp_size_in_bytes)
        except Exception:  # not every backend reports it
            self.temp_bytes = None

    def output(self):
        return np.asarray(self.fn(self.x))

    def call_us(self, inner):
        t = time.perf_counter()
        for _ in range(inner):
            self.fn(self.x).block_until_ready()
        return (time.perf_counter() - t) / inner * 1e6


def calibrate_inner(compiled, target_us):
    for _ in range(3):
        compiled.call_us(1)
    single = float(np.median([compiled.call_us(1) for _ in range(5)]))
    return int(min(1000, max(1, math.ceil(target_us / max(single, 1e-3)))))


def interleaved(compiled, rounds, inner, rng):
    """``rounds`` samples (us) per program, program order shuffled each round."""
    for c in compiled:
        c.call_us(inner)  # warm-up
    samples = [[] for _ in compiled]
    for _ in range(rounds):
        for i in rng.permutation(len(compiled)):
            samples[i].append(compiled[i].call_us(inner))
    return samples


# ---------------------------------------------------------------------- data
def collect(work, cfg, rng):
    """Time trajectory states + sampled one-step rewrites on XLA."""
    graphs, transitions, programs = {}, [], {}
    t0 = time.time()
    for w in work:
        for _ in range(cfg["trajectories"]):
            cur = w["graph"]
            for _ in range(cfg["max_len"]):
                cands = [(r, m) for r, m in enumerate_rewrites(cur)]
                if not cands:
                    break
                picks = rng.permutation(len(cands))[: cfg["candidates"]]
                nexts = []
                for i in picks:
                    new = apply_rewrite(cur, *cands[i])
                    if new is not None:
                        nexts.append((cands[i][0].name, new))
                if not nexts:
                    break
                group = [cur] + [g for _, g in nexts]
                fresh = [g for g in dict((graph_key(g), g) for g in group).values()
                         if graph_key(g) not in graphs]
                if fresh:
                    comp = [Compiled(g) for g in fresh]
                    inner = calibrate_inner(comp[0], 1000)
                    for g, c, s in zip(fresh, comp, interleaved(comp, cfg["label_rounds"], inner, rng)):
                        k = graph_key(g)
                        programs[k] = zlib.compress(c.hlo_text.encode())
                        graphs[k] = {"key": k, "workload": w["id"], "arch": w["arch"],
                                     "nodes": g.node_count(), "samples_us": s,
                                     "median_us": float(np.median(s)), "hlo": c.hlo,
                                     "xla_temp_bytes": c.temp_bytes, "graph": g}
                kb = graph_key(cur)
                for rule, g in nexts:
                    ka = graph_key(g)
                    transitions.append({
                        "workload": w["id"], "arch": w["arch"], "rule": rule, "before": kb, "after": ka,
                        "measured_speedup": graphs[kb]["median_us"] / graphs[ka]["median_us"],
                        "node_delta": g.node_count() - cur.node_count(),
                        "hlo_changed": graphs[kb]["hlo"]["fingerprint"] != graphs[ka]["hlo"]["fingerprint"]})
                cur = nexts[int(rng.integers(len(nexts)))][1]
        print(f"  data {w['id']:48s} graphs {len(graphs):5d} pairs {len(transitions):5d} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return graphs, transitions, programs


# -------------------------------------------------------------- pair model
def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def outcome(speedup):
    return "beneficial" if speedup >= 1 + NOISE else "harmful" if speedup <= 1 - NOISE else "neutral"


def confusion(pred, meas, smaller, threshold, tie_rule=True, tol=0.003):
    """Pair decision against the measured outcome. With ``tie_rule`` this is
    guided search's rule (accept if predicted >= threshold, or predicted
    within ``tol`` and the graph gets smaller); without it, threshold only."""
    table = {f"{d}_{o}": 0 for d in ("accepted", "rejected") for o in ("beneficial", "neutral", "harmful")}
    for p, m, s in zip(pred, meas, smaller):
        accept = p >= threshold or (tie_rule and p >= math.exp(-tol) and s)
        table[f"{'accepted' if accept else 'rejected'}_{outcome(m)}"] += 1
    n = max(1, len(pred))
    table["correct_decisions_pct"] = 100.0 * (table["accepted_beneficial"] + table["rejected_harmful"]
                                              + table["rejected_neutral"] + table["accepted_neutral"]) / n
    table["wrong_acceptances"] = table["accepted_harmful"]
    table["missed_beneficial"] = table["rejected_beneficial"]
    return table


def evaluate_fold(model, graphs, transitions):
    keys = list(graphs)
    pred_log = dict(zip(keys, model.predict_log_us([ns.featurize_sized(graphs[k]["graph"]) for k in keys])))
    true = np.array([graphs[k]["median_us"] for k in keys])
    est = np.exp(np.array([pred_log[k] for k in keys]))
    pred_sp = np.array([math.exp(pred_log[t["before"]] - pred_log[t["after"]]) for t in transitions])
    meas_sp = np.array([t["measured_speedup"] for t in transitions])
    smaller = [t["node_delta"] < 0 for t in transitions]
    near = np.abs(pred_sp - 1.0) <= 0.05
    return {
        "graphs": len(keys), "pairs": len(transitions),
        "latency_mae_us": float(np.mean(np.abs(est - true))),
        "latency_mape_pct": float(np.mean(np.abs(est - true) / true) * 100),
        "latency_spearman": spearman(est, true),
        "pair_speedup_mae": float(np.mean(np.abs(pred_sp - meas_sp))) if len(meas_sp) else float("nan"),
        "pair_spearman": spearman(pred_sp, meas_sp),
        "pair_sign_agreement_pct": float(np.mean((pred_sp > 1) == (meas_sp > 1)) * 100) if len(meas_sp) else float("nan"),
        "pairs_near_threshold": int(near.sum()),
        "pair_spearman_near_1": spearman(pred_sp[near], meas_sp[near]),
        "confusion": {str(th): confusion(pred_sp, meas_sp, smaller, th) for th in THRESHOLDS},
        "confusion_threshold_only": {str(th): confusion(pred_sp, meas_sp, smaller, th, tie_rule=False)
                                     for th in THRESHOLDS},
    }, pred_log


# ------------------------------------------------------------------ policies
def policies(w, model, seed):
    g0 = w["graph"]
    out = {"none": (g0, [])}
    out["all_safe"] = apply_all(g0)
    out["fixed_no_cse"] = apply_all(g0, exclude=("cse",))
    score = lambda gs: model.predict_log_us([ns.featurize_sized(g) for g in gs])
    decisions = {}
    guided, trace = ns.guided_search(g0, score, stats=decisions)
    out["guided"] = (guided, [t["rule"] for t in trace])
    out["random_same"] = apply_random(g0, len(trace), np.random.default_rng(seed))
    return out, decisions


def run_policies(work, models, rounds, rng, hlo_dir):
    rows = []
    for i, w in enumerate(work):
        variants, decisions = policies(w, models[w["arch"]], seed=i)
        comp = {p: Compiled(variants[p][0]) for p in POLICIES}
        ref = comp["none"].output()
        scale = max(1e-6, float(np.abs(ref).max()))
        rel_err = {p: float(np.abs(comp[p].output() - ref).max()) / scale for p in POLICIES}
        inner = calibrate_inner(comp["none"], 2000)
        samples = dict(zip(POLICIES, interleaved([comp[p] for p in POLICIES], rounds, inner, rng)))
        base = samples["none"]
        row = {
            "workload": w["id"], "arch": w["arch"], "size": w["size"], "variant": w["variant"],
            "inner_calls_per_sample": inner, "samples_us": samples,
            "nodes": {p: variants[p][0].node_count() for p in POLICIES},
            "rewrites": {p: variants[p][1] for p in POLICIES},
            "guided_decisions": decisions,
            "max_rel_err_vs_none": rel_err, "outputs_match": all(e < 1e-4 for e in rel_err.values()),
            "hlo": {p: comp[p].hlo for p in POLICIES},
            "hlo_same_as_none": {p: comp[p].hlo["fingerprint"] == comp["none"].hlo["fingerprint"] for p in POLICIES},
            "xla_temp_bytes": {p: comp[p].temp_bytes for p in POLICIES},
            "compile_ms": {p: comp[p].compile_ms for p in POLICIES},
            "vs_none": {p: stats.paired_speedup(base, samples[p], seed=i) for p in POLICIES[1:]},
            "guided_vs_all_safe": stats.paired_speedup(samples["all_safe"], samples["guided"], seed=i),
        }
        rows.append(row)
        slug = "".join(ch if ch.isalnum() else "_" for ch in w["id"]).strip("_")
        d = hlo_dir / "policies" / slug
        d.mkdir(parents=True, exist_ok=True)
        for p in POLICIES:
            with gzip.open(d / f"{p}.hlo.txt.gz", "wt") as fh:
                fh.write(comp[p].hlo_text)
        v = row["vs_none"]
        print(f"  {w['id']:48s} " + " ".join(f"{p} {v[p]['median_speedup']:.3f}{v[p]['verdict'][0]}" for p in v)
              + f"  match={row['outputs_match']}", flush=True)
    return rows


def summarise(rows):
    out = {}
    for p in POLICIES[1:]:
        sp = np.array([r["vs_none"][p]["median_speedup"] for r in rows])
        verdicts = [r["vs_none"][p]["verdict"] for r in rows]
        holm = stats.holm([r["vs_none"][p]["wilcoxon_p"] for r in rows])
        out[p] = {"suite_median_speedup": float(np.median(sp)),
                  "suite_geomean_speedup": float(np.exp(np.mean(np.log(sp)))),
                  "suite_median_ci95": list(stats.bootstrap_ci(sp)),
                  "workloads_faster": verdicts.count("faster"),
                  "workloads_slower": verdicts.count("slower"),
                  "workloads_indistinguishable": verdicts.count("indistinguishable"),
                  "workloads_significant_holm": int(sum(holm)),
                  "workloads_hlo_identical_to_none": int(sum(r["hlo_same_as_none"][p] for r in rows))}
    # guided vs all-safe across workloads (one paired value per workload)
    diff = np.array([math.log(r["guided_vs_all_safe"]["median_speedup"]) for r in rows])
    gv = [r["guided_vs_all_safe"]["verdict"] for r in rows]
    out["guided_vs_all_safe"] = {
        "median_ratio": float(np.exp(np.median(diff))),
        "median_ratio_ci95": [float(np.exp(x)) for x in stats.bootstrap_ci(diff)],
        "wilcoxon_p_across_workloads": stats.wilcoxon_p(diff),
        "workloads_guided_faster": gv.count("faster"), "workloads_guided_slower": gv.count("slower"),
        "workloads_indistinguishable": gv.count("indistinguishable"),
        "workloads_same_hlo": int(sum(r["hlo"]["guided"]["fingerprint"] == r["hlo"]["all_safe"]["fingerprint"]
                                      for r in rows))}
    dec = [r["guided_decisions"] for r in rows]
    out["guided_decisions"] = {k: int(sum(d.get(k, 0) for d in dec)) for k in dec[0]} if dec else {}
    out["guided_rewrites_applied_mean"] = float(np.mean([len(r["rewrites"]["guided"]) for r in rows]))
    out["all_safe_rewrites_applied_mean"] = float(np.mean([len(r["rewrites"]["all_safe"]) for r in rows]))
    out["all_outputs_match"] = all(r["outputs_match"] for r in rows)
    return out


# ----------------------------------------------------------------------- HLO
def rule_hlo_report(transitions, programs, hlo_dir):
    """Per rule: how often it changes the compiled program, and its effect.

    Saves before/after HLO + canonical diff for the most positive, most
    negative and most neutral measured instance of each rule.
    """
    report = {}
    d = hlo_dir / "rules"
    d.mkdir(parents=True, exist_ok=True)
    for rule in sorted({t["rule"] for t in transitions}):
        ts = [t for t in transitions if t["rule"] == rule]
        changed = [t for t in ts if t["hlo_changed"]]
        sp = np.array([t["measured_speedup"] for t in ts])
        report[rule] = {"pairs": len(ts), "hlo_changed": len(changed),
                        "median_speedup_all": float(np.median(sp)),
                        "median_speedup_hlo_changed": float(np.median([t["measured_speedup"] for t in changed])) if changed else None,
                        "median_speedup_hlo_unchanged": float(np.median([t["measured_speedup"] for t in ts if not t["hlo_changed"]])) if len(changed) < len(ts) else None,
                        "examples": {}}
        picks = {"positive": max(ts, key=lambda t: t["measured_speedup"]),
                 "negative": min(ts, key=lambda t: t["measured_speedup"]),
                 "neutral": min(ts, key=lambda t: abs(math.log(t["measured_speedup"])))}
        for label, t in picks.items():
            before, after = (zlib.decompress(programs[t[k]]).decode() for k in ("before", "after"))
            stem = d / f"{rule}_{label}"
            stem.with_suffix(".before.hlo.txt").write_text(before)
            stem.with_suffix(".after.hlo.txt").write_text(after)
            stem.with_suffix(".diff").write_text(hlo.diff(before, after) or "(identical optimised HLO)\n")
            report[rule]["examples"][label] = {"workload": t["workload"], "measured_speedup": t["measured_speedup"],
                                               "hlo_changed": t["hlo_changed"], "files": stem.name + ".*"}
    return report


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--single-thread", action="store_true")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cfg = dict(QUICK if args.quick else FULL)
    if args.rounds:
        cfg["rounds"] = args.rounds
    rng = np.random.default_rng(args.seed)
    print("jax", jax.__version__, jax.default_backend(), jax.devices(), flush=True)

    work = workloads(cfg["sizes"])
    print(f"{len(work)} workloads; collecting XLA-timed rewrite data", flush=True)
    graphs, transitions, programs = collect(work, cfg, rng)

    archs = list(BUILDERS)
    folds, models = {}, {}
    for arch in archs:
        train = [g for g in graphs.values() if g["arch"] != arch]
        test_g = {k: g for k, g in graphs.items() if g["arch"] == arch}
        test_t = [t for t in transitions if t["arch"] == arch]
        model, loss = ns.fit_value([{"features": ns.featurize_sized(g["graph"]),
                                     "log_us": math.log(g["median_us"])} for g in train],
                                   epochs=cfg["epochs"], seed=args.seed)
        quality, _ = evaluate_fold(model, test_g, test_t)
        quality.update({"train_graphs": len(train), "train_loss": loss,
                        "train_workloads": sorted({w["id"] for w in work if w["arch"] != arch}),
                        "test_workloads": sorted({w["id"] for w in work if w["arch"] == arch})})
        folds[arch], models[arch] = quality, model
        c = quality["confusion"]["1.0"]
        print(f"held-out {arch:24s} latency MAE {quality['latency_mae_us']:8.1f} us "
              f"MAPE {quality['latency_mape_pct']:5.1f}% | pair rho {quality['pair_spearman']:.2f} "
              f"wrong-accept {c['wrong_acceptances']} missed {c['missed_beneficial']}", flush=True)

    print(f"timing {len(POLICIES)} policies x {len(work)} workloads x {cfg['rounds']} rounds", flush=True)
    out = Path(args.out)
    hlo_dir = out.with_name(out.stem + "_hlo")
    rows = run_policies(work, models, cfg["rounds"], rng, hlo_dir)
    summary = summarise(rows)
    rules = rule_hlo_report(transitions, programs, hlo_dir)

    pooled = lambda key: float(np.nanmean([f[key] for f in folds.values()]))
    result = {
        "study": "xla_rewrite_study", "config": {**cfg, "noise_band": NOISE, "thresholds": THRESHOLDS,
                                                 "seed": args.seed, "quick": args.quick},
        "environment": environment.collect(),
        "units": {"latency": "microseconds (us)", "speedup": "ratio, >1 = faster",
                  "xla_temp_bytes": "bytes"},
        "workloads": [{k: w[k] for k in ("id", "arch", "size", "variant")} for w in work],
        "pair_model": {"leave_one_architecture_out": folds,
                       "mean": {k: pooled(k) for k in ("latency_mae_us", "latency_mape_pct", "latency_spearman",
                                                       "pair_speedup_mae", "pair_spearman",
                                                       "pair_sign_agreement_pct", "pair_spearman_near_1")}},
        "policies": {"summary": summary, "rows": rows},
        "rules": rules,
        "raw": {"graphs": [{k: v for k, v in g.items() if k != "graph"} for g in graphs.values()],
                "transitions": transitions},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
