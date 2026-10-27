"""Frozen-model test on an architecture that was never used for development.

The GPT-J / PaLM-style parallel block (`create_parallel_block_graph`) is not in
`ARCHITECTURES`, not in any training set and not used to pick features or
thresholds. All models below are loaded as-is (no retraining, no tuning).

  (a) plan selector + RL policy on 60 generated parallel-block graphs
      (fresh seed) - modeled reward, vs fixed pipeline and best possible;
  (b) real GPU: original vs fixed pipeline vs neuro-symbolic search vs
      fixed pipeline with generated Metal kernels, 5 sizes, outputs checked.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from models.architectures import create_parallel_block_graph  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from search import learned_optimizer as lo  # noqa: E402
from search import neuro_symbolic as ns  # noqa: E402
from search import rl_policy as rl  # noqa: E402
from utils.environment import collect as environment  # noqa: E402
from utils.torch_executor import device_name, gpu_compare, pick_device  # noqa: E402

SIZES = [(2, 128, 128), (4, 64, 64), (8, 256, 256), (4, 256, 512), (8, 512, 512)]


def modeled(n=60, seed=777):
    rng = np.random.default_rng(seed)
    selector, policy = lo.LearnedOptimizer.load(), rl.QNet()
    policy.load_state_dict(torch.load(rl.WEIGHTS, map_location="cpu", weights_only=True))
    policy.eval()
    fixed_i = lo.ACTIONS.index((True, True, True, 1))
    out = {k: [] for k in ("none", "fixed", "selector", "rl", "oracle")}
    for _ in range(n):
        g = lo._inject_redundancy(create_parallel_block_graph(int(rng.integers(1, 7))), rng)
        true = lo.evaluate_actions(g)
        out["none"].append(1.0)
        out["fixed"].append(true[fixed_i])
        out["selector"].append(true[lo.ACTIONS.index(tuple(selector.plan(g)["action"]))])
        out["oracle"].append(max(true))
        out["rl"].append(rl.rollout(policy, rl.tabulate(g))[0])
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["graphs"] = n
    res["regret_pct_selector"] = float(np.mean((np.array(out["oracle"]) - out["selector"]) / np.array(out["oracle"])) * 100)
    res["regret_pct_rl"] = float(np.mean((np.array(out["oracle"]) - out["rl"]) / np.array(out["oracle"])) * 100)
    return res


def gpu():
    device = pick_device()
    model = ns.load_value()
    score = lambda gs: model.predict_log_us([ns.featurize_sized(g) for g in gs])
    rows = []
    for size in SIZES:
        g0 = create_parallel_block_graph(2, *size)
        fixed = run_pipeline(g0, partition=False)[0]
        guided, trace = ns.guided_search(g0, score)
        res = gpu_compare([(g0, "sdpa"), (fixed, "sdpa"), (guided, "sdpa"), (fixed, "sdpa", True)],
                          device, repeats=30, warmup=5)
        base = res[0]["output"]
        scale = max(1e-6, float(np.abs(base).max()))
        t0 = res[0]["latency_min_s"]
        rows.append({
            "size": size, "nodes": [g0.node_count(), fixed.node_count(), guided.node_count()],
            "speedup_fixed": t0 / res[1]["latency_min_s"],
            "speedup_neurosymbolic": t0 / res[2]["latency_min_s"],
            "speedup_fixed_codegen": t0 / res[3]["latency_min_s"],
            "outputs_match": all(float(np.abs(r["output"] - base).max()) / scale < 1e-3 for r in res[1:]),
            "rewrites": [t["rule"] for t in trace]})
        print(size, {k: round(v, 3) for k, v in rows[-1].items() if k.startswith("speedup")}, rows[-1]["outputs_match"], flush=True)
    gm = lambda k: float(np.exp(np.mean(np.log([r[k] for r in rows]))))
    return {"device": device_name(device), "rows": rows,
            "geomean_speedup_fixed": gm("speedup_fixed"),
            "geomean_speedup_neurosymbolic": gm("speedup_neurosymbolic"),
            "geomean_speedup_fixed_codegen": gm("speedup_fixed_codegen"),
            "all_outputs_match": all(r["outputs_match"] for r in rows)}


def main():
    result = {"architecture": "Parallel-block (GPT-J style)", "modeled": modeled(), "gpu": gpu(),
              "environment": environment()}
    (ROOT / "docs" / "results" / "unseen_architecture.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"modeled": result["modeled"], "gpu": {k: v for k, v in result["gpu"].items() if k != "rows"}}, indent=1))


if __name__ == "__main__":
    main()
