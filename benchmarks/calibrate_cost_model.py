"""Calibrate the cost model on a real GPU and validate it out-of-sample.

1. Profile every node of every (architecture x size) graph, original and
   compiled, on the device.
2. Fit ``time = a + b*work`` per op class on TRAIN architectures only.
3. Predict end-to-end latency of the HELD-OUT architectures and compare with
   measured latency (gpu_compare) - and compare predicted vs measured
   speedups against the old hand-written cost table.
4. Refit on everything and write ``docs/results/cost_calibration.json`` (the
   file ``utils.cost_model`` loads for its per-op latencies).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from benchmarks.gpu_benchmark import BUILDERS  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from utils.cost_model import LEGACY_OP_LATENCY_US  # noqa: E402


def legacy_us(graph, scheduled=False):
    """Original hand-written table (+12us launch per node), single stream."""
    return sum(LEGACY_OP_LATENCY_US.get(d["op_type"], 150.0) + 12.0
               for _, d in graph.graph.nodes(data=True))
from utils import environment  # noqa: E402
from utils.gpu_calibration import fit, predict_rows, profile_nodes  # noqa: E402
from utils.torch_executor import device_name, gpu_compare, pick_device  # noqa: E402

SIZES = [(4, 64, 64), (2, 128, 128), (8, 256, 256), (4, 256, 512), (8, 512, 512)]
TRAIN_ARCHS = ["Post-LN Transformer", "LLaMA-style"]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    device = pick_device()
    print("device:", device_name(device), flush=True)
    data = []  # one dict per (arch, size)
    for name, build in BUILDERS.items():
        for b, s, d in SIZES:
            g = build(b, s, d)
            opt, _, _ = run_pipeline(g, partition=False)
            rows_o = profile_nodes(g, device, attention="composed")
            rows_c = profile_nodes(opt, device, attention="auto")
            meas_o, meas_c = gpu_compare([(g, "composed"), (opt, "auto")], device,
                                         repeats=30, warmup=5)
            data.append({
                "arch": name, "size": (b, s, d), "rows_o": rows_o, "rows_c": rows_c,
                "meas_o_us": meas_o["latency_min_s"] * 1e6,
                "meas_c_us": meas_c["latency_min_s"] * 1e6,
                "old_o_us": legacy_us(g),
                "old_c_us": legacy_us(opt),
            })
            print(f"profiled {name:24s} B{b} S{s} D{d}  measured {data[-1]['meas_o_us']:.0f} -> "
                  f"{data[-1]['meas_c_us']:.0f} us", flush=True)

    def predict_all(model, items):
        out = []
        for it in items:
            po = predict_rows(model, it["rows_o"])
            pc = predict_rows(model, it["rows_c"])
            out.append((po, pc))
        return out

    train = [it for it in data if it["arch"] in TRAIN_ARCHS]
    test = [it for it in data if it["arch"] not in TRAIN_ARCHS]
    model_tr = fit([r for it in train for r in it["rows_o"] + it["rows_c"]])
    preds = predict_all(model_tr, test)

    def stats(items, preds):
        meas_lat = np.array([x for it in items for x in (it["meas_o_us"], it["meas_c_us"])])
        pred_lat = np.array([x for p in preds for x in p])
        old_lat = np.array([x for it in items for x in (it["old_o_us"], it["old_c_us"])])
        meas_sp = np.array([it["meas_o_us"] / it["meas_c_us"] for it in items])
        cal_sp = np.array([p[0] / p[1] for p in preds])
        old_sp = np.array([it["old_o_us"] / it["old_c_us"] for it in items])
        return {
            "graphs": len(items),
            "latency_mape_calibrated_pct": float(np.mean(np.abs(pred_lat - meas_lat) / meas_lat) * 100),
            "latency_median_ape_calibrated_pct": float(np.median(np.abs(pred_lat - meas_lat) / meas_lat) * 100),
            "latency_log_corr_calibrated": float(np.corrcoef(np.log(pred_lat), np.log(meas_lat))[0, 1]),
            "latency_log_corr_old_table": float(np.corrcoef(np.log(old_lat), np.log(meas_lat))[0, 1]),
            "speedup_mae_calibrated": float(np.mean(np.abs(cal_sp - meas_sp))),
            "speedup_mae_old_table": float(np.mean(np.abs(old_sp - meas_sp))),
            "speedup_spearman_calibrated": spearman(cal_sp, meas_sp),
            "speedup_spearman_old_table": spearman(old_sp, meas_sp),
            "measured_speedup_median": float(np.median(meas_sp)),
            "calibrated_speedup_median": float(np.median(cal_sp)),
            "old_table_speedup_median": float(np.median(old_sp)),
        }

    validation = stats(test, preds)
    in_sample = stats(train, predict_all(model_tr, train))
    final = fit([r for it in data for r in it["rows_o"] + it["rows_c"]])

    detail = []
    for it, (po, pc) in zip(test, preds):
        detail.append({"arch": it["arch"], "size": it["size"],
                       "measured_us": [it["meas_o_us"], it["meas_c_us"]],
                       "calibrated_us": [po, pc], "old_table_us": [it["old_o_us"], it["old_c_us"]],
                       "measured_speedup": it["meas_o_us"] / it["meas_c_us"],
                       "calibrated_speedup": po / pc,
                       "old_table_speedup": it["old_o_us"] / it["old_c_us"]})

    # per-op reference latencies (at B8 S256 D256) for the table-based cost model
    ref = [it for it in data if it["size"] == (8, 256, 256)]
    per_op = {}
    for it in ref:
        for rows in (it["rows_o"], it["rows_c"]):
            for r in rows:
                c = final.get(r["cls"])
                t = c["a_us"] + c["b_us_per_work"] * r["work"]
                per_op.setdefault(r["op"], []).append(t)
    op_latency_us = {op: float(np.mean(v)) for op, v in per_op.items()}
    launch = float(np.median([r["seconds"] for it in data for r in it["rows_o"]
                              if r["cls"] == "elementwise"]) * 1e6)

    out = {
        "device": device.type, "device_name": device_name(device),
        "torch": torch.__version__, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment.collect(),
        "sizes": SIZES, "train_architectures": TRAIN_ARCHS,
        "validation_held_out_architectures": validation,
        "in_sample_train_architectures": in_sample,
        "validation_detail": detail,
        "class_model_train_only": model_tr,
        "class_model_final": final,
        "op_latency_us": op_latency_us,
        "launch_overhead_us": launch,
        "reference_config": [8, 256, 256],
    }
    path = ROOT / "docs" / "results" / "cost_calibration.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(validation, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
