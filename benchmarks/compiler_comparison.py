"""Like-for-like comparison with other compilers on the SAME graphs (CPU).

For every architecture x size, the same IR graph (same weights, same input)
is executed by:
  eager      - PyTorch eager, original graph
  ours       - PyTorch eager, graph compiled by our passes
  xla        - JAX/OpenXLA ``jit`` of the original graph (XLA does its own fusion)
  ours+xla   - XLA jit of OUR compiled graph
  inductor   - ``torch.compile`` (TorchInductor) of the original graph, if it works here
Reported: steady-state latency (interleaved, min), one-off compile time,
output agreement, and the post-optimisation op count (XLA: fusions in the
optimised HLO; ours: IR nodes).
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import torch  # noqa: E402

from benchmarks.gpu_benchmark import BUILDERS  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from utils import environment, jax_executor  # noqa: E402
from utils.torch_executor import _Runner, _feed, _main  # noqa: E402

SIZES = [(4, 64, 64), (8, 256, 256), (4, 256, 512)]
REPS = 40


def best_time(fn, reps=REPS, warmup=5):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def jax_system(graph, feed):
    f = jax.jit(jax_executor.build(graph))
    x = jax.numpy.asarray(feed)
    t = time.perf_counter()
    compiled = f.lower(x).compile()
    compile_s = time.perf_counter() - t
    hlo = compiled.as_text()
    out = np.asarray(compiled(x))
    fusions = len(re.findall(r"^\s*(?:ROOT )?\S+ = .*\bfusion\(", hlo, flags=re.M))
    instrs = len(re.findall(r"^\s*(?:ROOT )?%?\S+ = ", hlo, flags=re.M))
    return (lambda: compiled(x).block_until_ready()), out, compile_s, {"hlo_instructions": instrs, "hlo_fusions": fusions}


def main():
    dev = torch.device("cpu")
    rows = []
    for name, build in BUILDERS.items():
        for size in SIZES:
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
                fn, out, cs, info = jax_system(graph, feed)
                systems[tag], comp[tag], extra[tag] = fn, cs + (ctime if tag != "xla" else 0), info
                outs[tag] = torch.from_numpy(out)
            x = torch.from_numpy(feed)
            for tag, fn0 in (("inductor", main_e), ("ours+inductor", main_o)):
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
            best = {k: float("inf") for k in order}
            for k in order:
                for _ in range(5):
                    systems[k]()
            for i in range(REPS):  # interleaved
                for k in (order if i % 2 == 0 else order[::-1]):
                    t = time.perf_counter()
                    systems[k]()
                    best[k] = min(best[k], time.perf_counter() - t)
            for k in order:
                o = outs[k].detach().numpy()
                lat[k] = {"latency_us": best[k] * 1e6, "compile_ms": comp.get(k, 0.0) * 1e3,
                          "max_rel_err": float(np.abs(o - ref).max() / max(1e-6, np.abs(ref).max()))}
            rows.append({"arch": name, "size": size, "nodes_original": g.node_count(),
                         "nodes_ours": opt.node_count(), "systems": lat, "extra": extra})
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
    out = {"device": "cpu", "jax": jax.__version__, "jax_backend": jax.default_backend(),
           "torch": torch.__version__, "environment": environment.collect(),
           "rows": rows, "summary": summary}
    (ROOT / "docs" / "results" / "compiler_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
