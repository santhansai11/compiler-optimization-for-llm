"""Like-for-like comparison with other compilers on the SAME graphs (CPU).

For every architecture x size, the same IR graph (same weights, same input)
is executed by:
  eager      - PyTorch eager, original graph
  ours       - PyTorch eager, graph compiled by our passes
  xla        - JAX/OpenXLA ``jit`` of the original graph (XLA does its own fusion)
  ours+xla   - XLA jit of OUR compiled graph
  inductor   - ``torch.compile`` (TorchInductor) of the original graph, if it works here
Reported: steady-state latency (randomized paired rounds, median), one-off compile time,
output agreement, and the post-optimisation op count (XLA: fusions in the
optimised HLO; ours: IR nodes).
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import torch  # noqa: E402

from benchmarks.gpu_benchmark import BUILDERS  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from utils import jax_executor  # noqa: E402
from utils.torch_executor import _Runner, _feed, _main  # noqa: E402
from utils.benchmarking import environment_record, paired_summary, randomized_benchmark, suite_summary, write_json

SIZES = [(4, 64, 64), (8, 256, 256), (4, 256, 512)]
REPS = 40


def jax_system(graph, feed, hlo_path=None):
    f = jax.jit(jax_executor.build(graph))
    x = jax.device_put(feed, jax.devices("cpu")[0])
    t = time.perf_counter()
    compiled = f.lower(x).compile()
    compile_s = time.perf_counter() - t
    hlo = compiled.as_text()
    if hlo_path:
        Path(hlo_path).write_text(hlo, encoding="utf-8")
    out = np.asarray(compiled(x))
    fusions = len(re.findall(r"^\s*(?:ROOT )?\S+ = .*\bfusion\(", hlo, flags=re.M))
    instrs = len(re.findall(r"^\s*(?:ROOT )?%?\S+ = ", hlo, flags=re.M))
    return (lambda: compiled(x).block_until_ready()), out, compile_s, {"hlo_instructions": instrs, "hlo_fusions": fusions}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/results/compiler_comparison.json")
    parser.add_argument("--rounds", type=int, default=REPS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quick", action="store_true", help="One small shape per architecture, for validation")
    parser.add_argument("--skip-inductor", action="store_true")
    args = parser.parse_args()
    if args.rounds < 30:
        parser.error("--rounds must be at least 30")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hlo_dir = args.out.parent / (args.out.stem + "_hlo")
    hlo_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    rows = []
    for name, build in BUILDERS.items():
        for size in ([(1, 8, 16)] if args.quick else SIZES):
            g = build(*size)
            opt, _, ctime = run_pipeline(g, partition=False)
            feed = _feed(g)
            systems, extra = {}, {}
            r_eager, r_ours = _Runner(g, dev, feed, "composed"), _Runner(opt, dev, feed, "composed")
            main_e = lambda: _main(g, r_eager.run()[0])
            main_o = lambda: _main(opt, r_ours.run()[0])
            outs = {"eager": main_e(), "ours": main_o()}
            ref = outs["eager"].numpy()
            systems["eager"], systems["ours"] = main_e, main_o
            comp = {"eager": 0.0, "ours": ctime}
            for tag, graph in (("xla", g), ("ours+xla", opt)):
                hlo_path = hlo_dir / f"case-{len(rows):03d}-{tag}.hlo"
                fn, out, cs, info = jax_system(graph, feed, hlo_path)
                info["optimized_hlo_file"] = str(hlo_path)
                systems[tag], comp[tag], extra[tag] = fn, cs + (ctime if tag != "xla" else 0), info
                outs[tag] = torch.from_numpy(out.copy())
            for tag, fn0 in (() if args.skip_inductor else (("inductor", main_e), ("ours+inductor", main_o))):
                try:  # TorchInductor on CPU (needs a C++ toolchain)
                    torch._dynamo.reset()  # one fresh compile per case (avoid recompile limit)
                    cf = torch.compile(fn0)
                    t = time.perf_counter()
                    oi = cf()
                    comp[tag] = time.perf_counter() - t + (ctime if tag != "inductor" else 0)
                    systems[tag] = cf
                    outs[tag] = oi
                except Exception as exc:  # report honestly
                    extra[tag + "_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            lat = {}
            order = list(systems)
            timing, _ = randomized_benchmark(systems, rounds=args.rounds, seed=args.seed + len(rows))
            for k in order:
                o = outs[k].detach().numpy()
                if not np.allclose(o, ref, rtol=1e-4, atol=1e-5):
                    raise ValueError(f"Output mismatch for {name} {size} {k}")
                lat[k] = {"latency_us": timing["median_us"][k], "compile_ms": comp.get(k, 0.0) * 1e3,
                          "max_rel_err": float(np.abs(o - ref).max() / max(1e-6, np.abs(ref).max()))}
            rows.append({"arch": name, "size": size, "nodes_original": g.node_count(),
                         "nodes_ours": opt.node_count(), "systems": lat, "extra": extra,
                         "timing": timing, "paired": {k: paired_summary(timing["samples_us"]["eager"],timing["samples_us"][k],seed=args.seed)
                                                       for k in order if k != "eager"}})
            print(f"{name:24s} {size}: " + "  ".join(f"{k} {v['latency_us']:.0f}us" for k, v in lat.items()), flush=True)
    summary = {}
    for k in ("ours", "xla", "ours+xla", "inductor", "ours+inductor"):
        sp = [r["systems"]["eager"]["latency_us"] / r["systems"][k]["latency_us"] for r in rows if k in r["systems"]]
        if sp:
            summary[k] = {"geomean_speedup_vs_eager": float(np.exp(np.mean(np.log(sp)))),
                          "cases": len(sp),
                          "max_rel_err": float(max(r["systems"][k]["max_rel_err"] for r in rows if k in r["systems"]))}
    pairs = {}
    for base, plus in (("xla", "ours+xla"), ("inductor", "ours+inductor"), ("eager", "ours")):
        sp = [r["systems"][base]["latency_us"] / r["systems"][plus]["latency_us"]
              for r in rows if base in r["systems"] and plus in r["systems"]]
        pairs[f"{plus}_vs_{base}"] = float(np.exp(np.mean(np.log(sp)))) if sp else None
    summary["effect_of_our_passes_on_top_of"] = pairs
    summary["paired_workload_comparisons"] = {
        f"{plus}_vs_{base}": suite_summary([r for r in rows if plus in r["systems"]],base,plus,args.seed)
        for base,plus in (("xla","ours+xla"),("eager","ours"),("inductor","ours+inductor"))
        if any(plus in r["systems"] for r in rows)}
    out = {"device": "cpu", "jax": jax.__version__, "jax_backend": "cpu",
           "torch": torch.__version__, "rows": rows, "summary": summary,
           "environment": environment_record(ROOT), "timing_unit": "us", "estimator": "median",
           "quick_validation_only": args.quick}
    write_json(args.out, out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
