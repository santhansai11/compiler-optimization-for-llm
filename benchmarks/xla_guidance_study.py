"""Train and evaluate rewrite guidance on the SAME XLA CPU target.

    python -m benchmarks.xla_guidance_study --out outputs/xla-study
    python -m benchmarks.xla_guidance_study --smoke --out outputs/xla-smoke

Each held-out family is excluded from all labels used to train its model.
Smoke mode validates the machinery; it is not a publication-quality result.
Old MPS weights/results are neither loaded nor overwritten. Artifact directories
must be new. Source snapshots, graph records, raw timings, splits and weights
make every reported number traceable to the run that generated it.
"""

import argparse
import difflib
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from models.architectures import ARCHITECTURES
from models.operations import OPERATION_DEMOS
from search import learned_optimizer as lo
from search import neuro_symbolic as ns
from search.rewrites import apply_rewrite, enumerate_rewrites
from utils.benchmarking import (environment_record, paired_summary, randomized_benchmark,
                                source_snapshot, suite_summary, write_json)
from utils.xla_runtime import XlaProgram, graph_key, normalized_hlo

ROOT = Path(__file__).resolve().parents[1]
BUILDERS = {**{k: v for k, v in ARCHITECTURES.items() if k != "MLP classifier"},
            **{k: v[1] for k, v in OPERATION_DEMOS.items()}}


def features(graph):
    """Structural/size features, with old GPU cost priors explicitly zeroed."""
    x, edges, glob = ns.featurize_sized(graph)
    x[:, len(lo.OPS) + 2] = 0.0
    glob[5:8] = 0.0  # GPU-work and GPU-critical-path prior features
    glob[-1] = 0.0   # GPU per-op latency prior
    return x, edges, glob


def candidates(graph, limit):
    result = []
    seen = set()
    for rule, match in enumerate_rewrites(graph):
        new = apply_rewrite(graph, rule, match)
        if new is not None:
            key = graph_key(new)
            if key not in seen:
                seen.add(key)
                result.append((rule.name, match, new))
    return result[:limit], len(result)


def rule_baseline(graph, mode="all_safe", budget=80, seed=0):
    """Greedy rule-order baseline, or random legal rewrites with matched budget."""
    rng = np.random.default_rng(seed)
    cur, trace = graph, []
    for _ in range(budget):
        options, _ = candidates(cur, 100000)
        if mode == "elementwise_no_cse":
            # This IR has scale-chain algebra, but no general elementwise kernel fuser.
            options = [c for c in options if c[0] in {"merge-scales", "eliminate-identity"}]
        if not options:
            break
        selected = options[int(rng.integers(len(options)))] if mode == "random_budget" else options[0]
        trace.append({"rule": selected[0], "match": selected[1]})
        cur = selected[2]
    return cur, trace


def guided(graph, model, steps, limit, threshold):
    """Decisions use predictions ONLY; timing is performed later for audit."""
    cur, trace, audit, coverage = graph, [], [], []
    for step in range(steps):
        options, available = candidates(cur, limit)
        coverage.append({"step": step, "legal_candidates": available, "scored_candidates": len(options)})
        if not options:
            break
        logs = model.predict_log_us([features(cur)] + [features(c[2]) for c in options])
        finite = np.isfinite(logs[0]) & np.isfinite(logs[1:])
        speedups = np.exp(np.clip(logs[0] - logs[1:], -50, 50))
        eligible = finite & (speedups >= threshold)
        best = int(np.argmax(np.where(eligible, speedups, -np.inf))) if eligible.any() else None
        for i, (rule, match, new) in enumerate(options):
            audit.append({"step": step, "before": cur, "after": new, "rule": rule,
                          "match": match, "predicted_speedup": float(speedups[i]) if finite[i] else None,
                          "predicted_accept": bool(eligible[i]), "selected": i == best,
                          "rejection_reason": "missing_estimate" if not finite[i] else
                          "below_threshold" if not eligible[i] else "not_best" if i != best else None})
        if best is None:
            break
        rule, match, cur = options[best]
        trace.append({"rule": rule, "match": match, "predicted_speedup": float(speedups[best])})
    return cur, trace, audit, coverage


def decision_metrics(rows, threshold):
    valid = [r for r in rows if r["predicted_speedup"] is not None]
    counts = {"correct_acceptances": 0, "wrong_acceptances": 0,
              "missed_beneficial_rewrites": 0, "correct_rejections": 0}
    for r in rows:
        beneficial = r["measured_speedup"] >= threshold
        name = ("correct_acceptances" if beneficial else "wrong_acceptances") if r["predicted_accept"] else (
            "missed_beneficial_rewrites" if beneficial else "correct_rejections")
        counts[name] += 1
    pred = np.array([r["predicted_speedup"] for r in valid])
    true = np.array([r["measured_speedup"] for r in valid])
    rho = spearmanr(pred, true).statistic if len(valid) > 1 and np.ptp(pred) and np.ptp(true) else None
    return {**counts, "candidate_decisions": len(rows),
            "missing_estimates": len(rows) - len(valid),
            "missing_estimate_rejection_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
            "gate_accuracy": (counts["correct_acceptances"] + counts["correct_rejections"]) / len(rows) if rows else None,
            "speedup_mae": float(np.mean(np.abs(pred - true))) if len(valid) else None,
            "speedup_rmse": float(np.sqrt(np.mean((pred - true)**2))) if len(valid) else None,
            "speedup_spearman": float(rho) if rho is not None and np.isfinite(rho) else None,
            "threshold": threshold,
            "definition": "accept gate vs paired measured median speedup >= threshold; not a formal benefit proof"}


def run(args):
    if args.rounds < 30 or args.threshold < 1 or args.epochs < 1 or args.max_candidates < 1 or args.steps < 1:
        raise ValueError("Need >=30 rounds, threshold>=1 and positive epochs/candidates/steps")
    names = args.families or list(BUILDERS)
    if len(set(names)) < 2 or any(n not in BUILDERS for n in names):
        raise ValueError("Choose at least two distinct registered families")
    sizes = [(1, 8, 16)] if args.smoke else [(2, 32, 32), (4, 64, 64)]
    if args.smoke and args.families is None:
        names = ["op_transpose", "op_scale_chain", "op_mlp_block"]
    epochs = min(args.epochs, 25) if args.smoke else args.epochs
    steps = min(args.steps, 2) if args.smoke else args.steps
    limit = min(args.max_candidates, 3) if args.smoke else args.max_candidates
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.torch_threads)
    env = environment_record(ROOT)
    config = {"target": "jax_xla_cpu", "label_unit": "us", "label_estimator": "paired-round median",
              "families": names, "sizes": sizes, "seed": args.seed, "rounds": args.rounds,
              "warmup": 5, "epochs": epochs, "max_steps": steps, "max_candidates": limit,
              "threshold": args.threshold, "smoke_only": args.smoke,
              "model": {"name": "LatencyGNN", "hidden": 64, "message_passing_layers": 2,
                        "directions": ["predecessor mean", "successor mean"],
                        "pooling": "sum of positive softplus per-node time contributions",
                        "output": "log(microseconds)", "loss": "smooth L1 on log_us, beta=0.1",
                        "optimizer": "Adam", "lr": 0.004, "weight_decay": 1e-6,
                        "scheduler": "cosine annealing", "GPU_cost_features": "zeroed",
                        "node_feature_dim": lo.NODE_DIM, "global_feature_dim": ns.GLOBAL_DIM,
                        "operator_vocabulary": lo.OPS, "feature_version": lo.FEATURE_VERSION,
                        "GAE": "not present in this repository or used by this study"},
              "split": "leave one architecture family out; no test-family labels in training",
              "random_baseline": "3 seeds, uniformly selected legal rewrites; same accepted-step budget as guided",
              "fixed_rule": "merge scales and remove identities; never CSE (no general elementwise fuser in this IR)"}
    write_json(out / "config.json", config)
    write_json(out / "environment.json", env)
    source_hash = source_snapshot(ROOT, out)
    programs = {}

    def program(g):
        key = graph_key(g)
        if key not in programs:
            programs[key] = XlaProgram(g, out / "hlo")
        return programs[key]

    rng = np.random.default_rng(args.seed)
    samples, datasets, workloads = [], [], []
    # Random trajectories expose partially rewritten graphs, without labels
    # determining which examples get collected.
    for family in names:
        for size in sizes:
            original = BUILDERS[family](batch=size[0], seq=size[1], d_model=size[2])
            workload_id = f"w{len(workloads):03d}"
            workloads.append((workload_id, family, size, original))
            graphs = {graph_key(original): original}
            cur = original
            for _ in range(steps):
                options, _ = candidates(cur, limit)
                graphs.update({graph_key(c[2]): c[2] for c in options})
                if not options:
                    break
                cur = options[int(rng.integers(len(options)))][2]
            calls = {key: program(g) for key, g in graphs.items()}
            timing, outputs = randomized_benchmark(calls, rounds=args.rounds, seed=args.seed + len(workloads))
            ref = program(original).output
            if not all(np.allclose(np.asarray(v), ref, rtol=1e-4, atol=1e-5) for v in outputs.values()):
                raise ValueError(f"Rewrite changed outputs in {workload_id}")
            for key, g in graphs.items():
                samples.append({"id": f"{workload_id}:{key}", "family": family,
                                "features": features(g), "log_us": float(np.log(timing["median_us"][key]))})
            datasets.append({"workload": workload_id, "family": family, "size": size, "timing": timing})
            write_json(out / "labels.json", datasets)
            print(f"labelled {workload_id} {family}: {len(graphs)} XLA graphs", flush=True)
    rows, audit_rows, folds = [], [], []
    for family_index, family in enumerate(names):
        train = [s for s in samples if s["family"] != family]
        test = [s for s in samples if s["family"] == family]
        seed = args.seed + family_index
        model, loss = ns.fit_value(train, epochs=epochs, seed=seed)
        model_path = f"fold-{family_index:02d}.pt"
        torch.save({"state_dict": model.state_dict(), "config": config, "held_out": family}, out / model_path)
        pred_us = np.exp(model.predict_log_us([s["features"] for s in test]))
        true_us = np.exp([s["log_us"] for s in test])
        folds.append({"held_out": family, "train_ids": [s["id"] for s in train],
                      "test_ids": [s["id"] for s in test], "seed": seed, "weights": model_path,
                      "train_loss_log_us": loss,
                      "mae_us": float(np.mean(np.abs(pred_us-true_us))),
                      "rmse_us": float(np.sqrt(np.mean((pred_us-true_us)**2))),
                      "prediction_records": [{"id": s["id"], "predicted_us": float(p), "measured_us": float(t)}
                                             for s,p,t in zip(test,pred_us,true_us)]})
        for wid, fam, size, original in workloads:
            if fam != family:
                continue
            chosen, trace, audit, coverage = guided(original, model, steps, limit, args.threshold)
            variants = {"original": original, "guided": chosen}
            traces = {"guided": trace}
            for name in ("all_safe", "elementwise_no_cse"):
                variants[name], traces[name] = rule_baseline(original, mode=name)
            for rseed in range(3):
                name = f"random_budget_{rseed}"
                variants[name], traces[name] = rule_baseline(original, "random_budget", len(trace), seed+rseed)
            calls = {name: program(g) for name,g in variants.items()}
            timing, outputs = randomized_benchmark(calls, rounds=args.rounds, seed=seed)
            ref = np.asarray(outputs["original"])
            if not all(np.allclose(np.asarray(v), ref, rtol=1e-4, atol=1e-5) for v in outputs.values()):
                raise ValueError(f"Policy output mismatch: {wid}")
            row = {"workload": wid, "family": family, "size": size, "timing": timing,
                   "outputs_match": True, "traces": traces, "candidate_coverage": coverage,
                   "programs": {k: program(g).record() for k,g in variants.items()},
                   "paired": {k: paired_summary(timing["samples_us"]["original"],v,seed=seed)
                              for k,v in timing["samples_us"].items() if k != "original"}}
            row["guided_vs_all_safe"] = paired_summary(timing["samples_us"]["all_safe"],timing["samples_us"]["guided"],seed=seed)
            rows.append(row)
            for name,g in variants.items():
                diff = "".join(difflib.unified_diff(normalized_hlo(program(original).hlo).splitlines(True),
                    normalized_hlo(program(g).hlo).splitlines(True),fromfile="original",tofile=name))
                (out / "hlo" / f"{wid}-{name}.diff").write_text(diff,encoding="utf-8")
            # Audit every scored decision only AFTER the entire prediction-only search.
            for item in audit:
                before, after = item.pop("before"), item.pop("after")
                timings, _ = randomized_benchmark({"before": program(before), "after": program(after)},
                                                  rounds=args.rounds, seed=seed+item["step"])
                paired = paired_summary(timings["samples_us"]["before"],timings["samples_us"]["after"],seed=seed)
                audit_rows.append({**item, "workload": wid, "before_id": graph_key(before),
                    "after_id": graph_key(after), "measured_speedup": paired["speedup_ratio_of_medians"],
                    "timing": timings, "paired": paired})
            write_json(out / "workloads.json", rows)
            write_json(out / "decisions.json", audit_rows)
            print(f"evaluated {wid}: guided {len(trace)} rewrites; audited {len(audit)} decisions",flush=True)
        write_json(out / "folds.json", folds)
    comparisons = {name: suite_summary(rows,"original",name,args.seed) for name in rows[0]["timing"]["samples_us"] if name != "original"}
    comparisons["guided_vs_all_safe"] = suite_summary(rows,"all_safe","guided",args.seed)
    near = [r for r in audit_rows if abs(r["measured_speedup"]-args.threshold) <= 0.05]
    summary = {"target": "jax_xla_cpu", "units": {"latency": "us", "memory": "bytes", "compile": "ms"},
               "smoke_only": args.smoke, "source_archive_sha256": source_hash,
               "comparisons": comparisons, "decision_quality": decision_metrics(audit_rows,args.threshold),
               "near_threshold_quality": decision_metrics(near,args.threshold),
               "near_threshold_band": [args.threshold-0.05,args.threshold+0.05],
               "limitations": ["Single-process timing intervals; no cross-machine generalisation claim",
                               "Unadjusted exploratory p-values; not confirmatory multiple-comparison evidence",
                               "Candidate/step caps and random matched-budget counts are saved",
                               "Textual HLO equality/difference is not proof of equivalent/different machine code"]}
    write_json(out / "summary.json", summary)
    # Representative cases selected transparently by observed median ratio.
    ordered = sorted(rows,key=lambda r:r["paired"]["guided"]["speedup_ratio_of_medians"])
    neutral = min(rows,key=lambda r:abs(r["paired"]["guided"]["speedup_ratio_of_medians"]-1))
    write_json(out / "representative_cases.json", {"worst_observed": ordered[0]["workload"],
        "closest_to_one": neutral["workload"], "best_observed": ordered[-1]["workload"],
        "note": "Inspect the HLO diff plus paired CI; these labels do not assert statistical significance."})
    print(f"Saved complete experiment: {out}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--families", nargs="+", choices=list(BUILDERS))
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
