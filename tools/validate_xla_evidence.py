"""Validate a saved XLA guidance run without re-running its timing experiment.

    python tools/validate_xla_evidence.py outputs/xla-review
"""

import argparse
import hashlib
import json
import statistics
import zipfile
from pathlib import Path


def read(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def validate(folder):
    folder = Path(folder)
    config = read(folder / "config.json")
    summary = read(folder / "summary.json")
    labels = read(folder / "labels.json")
    rows = read(folder / "workloads.json")
    folds = read(folder / "folds.json")
    decisions = read(folder / "decisions.json")
    failures = []

    def check(ok, message):
        if not ok:
            failures.append(message)

    def timing(t, context):
        n = t["rounds"]
        check(n >= 30 and n == config["rounds"], f"{context}: wrong round count")
        check(t["unit"] == "us", f"{context}: wrong sample unit")
        check(len(t["round_order"]) == n, f"{context}: missing round orders")
        names = set(t["samples_us"])
        for order in t["round_order"]:
            check(len(order) == len(names) and set(order) == names, f"{context}: malformed paired order")
        for name, samples in t["samples_us"].items():
            check(len(samples) == n and all(v > 0 for v in samples), f"{context}/{name}: missing/invalid samples")
            check(statistics.median(samples) == t["median_us"][name], f"{context}/{name}: stale median")

    check(config["target"] == summary["target"] == "jax_xla_cpu", "Target mismatch")
    check(summary["units"] == {"latency":"us","memory":"bytes","compile":"ms"}, "Wrong units")
    check(len(rows) == len(config["families"])*len(config["sizes"]), "Missing workload results")
    check(len(folds) == len(config["families"]), "Missing held-out folds")
    sample_families = {}
    for dataset in labels:
        timing(dataset["timing"], dataset["workload"])
        for key in dataset["timing"]["samples_us"]:
            sample_families[f'{dataset["workload"]}:{key}'] = dataset["family"]
            check((folder / "hlo" / f"{key}.optimized.hlo").is_file(), f"Missing label HLO {key}")
    for fold in folds:
        train, test = set(fold["train_ids"]), set(fold["test_ids"])
        check(not train & test, f'{fold["held_out"]}: train/test overlap')
        check(train | test == set(sample_families), f'{fold["held_out"]}: lost sample IDs')
        check(all(sample_families[k] != fold["held_out"] for k in train), "Test-family training leakage")
        check(all(sample_families[k] == fold["held_out"] for k in test), "Incorrect test-family IDs")
        check((folder / fold["weights"]).is_file(), "Missing model checkpoint")
    for row in rows:
        timing(row["timing"], row["workload"])
        check(row["outputs_match"], f'{row["workload"]}: output mismatch')
        for tag, program in row["programs"].items():
            path = folder / "hlo" / f'{program["graph_id"]}.optimized.hlo'
            check(path.is_file(), f"Missing {tag} HLO")
            if path.is_file():
                # Text is written with the platform newline convention.
                digest = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
                check(digest == program["optimized_hlo_sha256"], f"HLO content changed: {path.name}")
    for i, decision in enumerate(decisions):
        timing(decision["timing"], f"decision-{i}")
        pred = decision["predicted_speedup"]
        check(decision["predicted_accept"] == (pred is not None and pred >= config["threshold"]), "Incorrect threshold gate")
        check(decision["measured_speedup"] == decision["timing"]["median_us"]["before"] / decision["timing"]["median_us"]["after"], "Stale decision speedup")
    for tag, stats in summary["comparisons"].items():
        base, variant = ("all_safe","guided") if tag == "guided_vs_all_safe" else ("original",tag)
        slower = sum(r["timing"]["median_us"][variant] > r["timing"]["median_us"][base] for r in rows)
        check(stats["observed_slowdowns"] == slower, f"Stale slowdown count: {tag}")
    archive = folder / "sources.zip"
    check(hashlib.sha256(archive.read_bytes()).hexdigest() == summary["source_archive_sha256"], "Source archive changed")
    with zipfile.ZipFile(archive) as z:
        for name, digest in read(folder / "source_manifest.json").items():
            check(hashlib.sha256(z.read(name)).hexdigest() == digest, f"Source hash mismatch: {name}")
    if failures:
        raise ValueError("\n".join(failures))
    return {"valid": True, "workloads": len(rows), "label_graphs": len(sample_families),
            "held_out_folds": len(folds), "audited_decisions": len(decisions),
            "rounds_per_variant": config["rounds"], "smoke_only": config["smoke_only"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.folder), indent=2))
