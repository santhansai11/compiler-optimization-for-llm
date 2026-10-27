"""Regenerate README.md from the result files so every number is traceable.

    python tools/make_readme.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / "docs" / "results"


def load(name):
    f = R / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def arch_table():
    from models.architectures import ARCHITECTURES
    from pipeline import run_pipeline
    from utils.metrics import compute_metrics

    rows = ["| Architecture | Nodes | GRR | OMR | ACR | Launch red. | Speedup measured (CPU) | Speedup modeled | Output identical |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, build in ARCHITECTURES.items():
        g = build()
        o, i, t = run_pipeline(g, partition=False)
        m = compute_metrics(g, o, i, t)
        acr = m["attention_canonicalization_rate"]
        rows.append(f"| {name} | {m['counts']['nodes_original']} → {m['counts']['nodes_optimized']} | "
                    f"{m['graph_reduction_ratio'] * 100:.1f}% | {m['operator_merge_ratio'] * 100:.1f}% | "
                    f"{'N/A' if acr is None else str(round(acr * 100)) + '%'} | "
                    f"{m['kernel_launch_reduction'] * 100:.1f}% | {m['speedup']:.2f}× | "
                    f"{m['modeled']['speedup']:.2f}× | {'yes' if m['accuracy']['preserved'] else 'NO'} |")
    return "\n".join(rows)


def xla_study_section(res):
    """§8: rewrite guidance trained and evaluated on XLA (empty until run)."""
    if not res:
        return ("### 8. Rewrite guidance on XLA (`benchmarks/xla_rewrite_study.py`)\n\n"
                "Not run yet: `python benchmarks/xla_rewrite_study.py && python tools/make_xla_tables.py`.\n")
    s, m, env = res["policies"]["summary"], res["pair_model"]["mean"], res["environment"]
    labels = {"all_safe": "All-safe", "guided": "Guided", "random_same": "Random, same count",
              "fixed_no_cse": "Fixed rule (no CSE)"}
    rows = "\n".join(
        f"| {labels[p]} | {v['suite_median_speedup']:.3f}× [{v['suite_median_ci95'][0]:.3f}, {v['suite_median_ci95'][1]:.3f}] | "
        f"{v['workloads_faster']} | {v['workloads_slower']} | {v['workloads_indistinguishable']} | "
        f"{v['workloads_hlo_identical_to_none']} |" for p, v in s.items() if p in labels)
    g = s["guided_vs_all_safe"]
    pk = env["packages"]
    return f"""### 8. Rewrite guidance trained and evaluated on XLA (`benchmarks/xla_rewrite_study.py`)

Labels and evaluation use the same target: every graph is compiled with `jax.jit` (XLA, CPU) and timed. {len(res['workloads'])} workloads (6 architectures × {len(res['config']['sizes'])} sizes × base / injected redundancy); the latency GNN is trained leave-one-architecture-out on {len(res['raw']['graphs'])} XLA-timed graphs. Each policy is timed in {res['config']['rounds']} paired rounds with the variant order shuffled every round. Environment: {env['cpu']}, {env['logical_cores']} cores, {env['os']}, jax {pk['jax']}, jaxlib {pk['jaxlib']}.

Pair model on held-out architectures: graph latency MAE {m['latency_mae_us']:.1f} µs ({m['latency_mape_pct']:.1f}%), pair-speedup MAE {m['pair_speedup_mae']:.3f}, pair rank correlation {m['pair_spearman']:.2f} ({m['pair_spearman_near_1']:.2f} for pairs predicted within ±5% of 1).

| Policy (vs no rewriting) | Suite median speedup [95% CI] | Faster | Slower | Indistinguishable | Same optimised HLO as original |
|---|---|---|---|---|---|
{rows}

Guided vs all-safe: median ratio {g['median_ratio']:.4f} [{g['median_ratio_ci95'][0]:.4f}, {g['median_ratio_ci95'][1]:.4f}], Wilcoxon p = {g['wilcoxon_p_across_workloads']:.3f}; guided faster on {g['workloads_guided_faster']}, slower on {g['workloads_guided_slower']}, indistinguishable on {g['workloads_indistinguishable']} workloads; identical optimised HLO on {g['workloads_same_hlo']}. Guided search scored {s['guided_decisions'].get('candidates_scored')} candidates, {s['guided_decisions'].get('no_estimate')} without a usable estimate. All tables, the accept/reject confusion, per-rule HLO effects and the raw samples: `docs/results/xla_rewrite_study_tables.md`, `docs/results/xla_rewrite_study_samples.csv`, `docs/results/xla_rewrite_study_hlo/`.
"""


def main():
    real = load("real_graph_study")["rows"]
    cal = load("cost_calibration")
    gpu = load("gpu_mps")
    lm = load("lm_results")
    lmf = load("lm_full_eval")
    lop = load("learned_results")
    rl = load("rl_results")
    nsr = load("neuro_symbolic_results")
    cmp_ = load("compiler_comparison")
    v = cal["validation_held_out_architectures"]
    gs = gpu["summary"]
    lm_ = lop["lofo_mean"]
    rlm = rl["lofo_mean"]
    nss = nsr["summary"]
    cs = cmp_["summary"]
    eff = cs["effect_of_our_passes_on_top_of"]
    un = load("unseen_architecture")
    sens = load("learned_sensitivity")
    sens = sens["settings"] if isinstance(sens, dict) else sens
    other = real_study["other_architectures"] if (real_study := load("real_graph_study")) else []
    xla_section = xla_study_section(load("xla_rewrite_study"))

    real_rows = "\n".join(
        f"| {r['model']} | {r['nodes_original']} | −{r['removed_by_normalize']} | −{r['removed_by_attention']} | "
        f"−{r['removed_by_merge']} | **{r['nodes_after_merge']}** ({r['grr'] * 100:.0f}%) | "
        f"{r['attention_canonicalized']}/{r['attention_detected']} | {r['compiled_vs_pytorch_max_diff']:.1e} | {r['same_top1_pct']:.0f}% |"
        for r in real if "error" not in r)
    ns_rows = "\n".join(
        f"| {a} | {x['latency_model']['mape_pct']:.0f}% | {x['latency_model']['spearman']:.2f} | "
        f"{x['geomean_speedup_fixed']:.3f}× | {x['geomean_speedup_naive']:.3f}× | **{x['geomean_speedup_neurosymbolic']:.3f}×** |"
        for a, x in nsr["leave_one_architecture_out"].items())
    cmp_rows = "\n".join(
        f"| {k} | {x['geomean_speedup_vs_eager']:.2f}× |" for k, x in cs.items()
        if k != "effect_of_our_passes_on_top_of")

    sens_rows = "\n".join(
        f"| {r['config']} | {r['comm_us']:g} | {r['device_penalty']:g} | {r['fixed_1dev']:.2f} | **{r['learned']:.2f}** | "
        f"{r['oracle']:.2f} | {r['regret_pct']:.1f}% | {r['best_plan_uses_multiple_devices_pct']:.0f}% |" for r in sens)

    text = f"""# GraphForge — Compiler Optimization for Large Language Models

A compiler-style optimizer for LLM computation graphs, inspired by XLA/OpenXLA. A model becomes a DAG of tensor operations; fixed, semantics-preserving rewrite passes shrink and reorganize it; **learned components (GNN plan selector, RL pass policy, GNN-guided rewrite search) decide what to apply**; everything is checked for identical outputs and measured on CPU, on a real GPU, and against XLA / TorchInductor. A React GUI shows it all.

> This file is generated by `tools/make_readme.py` from `docs/results/*.json`; every number below is produced by a script in this repo.

## Quick start

```bash
./run.sh            # macOS/Linux: API on :8000, GUI on http://127.0.0.1:5173
run-server.bat      # Windows (create .venv and install requirements.txt first)
python -m unittest discover -s tests -t .      # 26 tests
```

Python 3.12 + Node.js. `pip install -r requirements.txt` (torch, jax, scipy, networkx, openpyxl, matplotlib); `pip install -r requirements-real.txt` adds HuggingFace `transformers` for the real-graph study.

## What it does

| Stage | Where | What |
|---|---|---|
| Graph IR | `ir/graph.py` | DAG: nodes = tensor ops, edges = dependencies, operand order encoded by in-edge order |
| Model sources | `models/` | Post-LN, Pre-LN, ReLU, LLaMA-style, MoE, multi-head, MLP, an "operation lab", a **trained** Tiny Shakespeare Transformer, and **any `nn.Module` traced with torch.fx** (`models/import_hf.py`) |
| Normalization | `passes/normalize.py` | identity removal, constant folding, CSE, dead code, transpose algebra, scale merging, **view algebra** (cancels reshape pairs around linear layers) |
| Attention canonicalization | `passes/canonicalize.py` | Q/K/V·QKᵀ·scale·(mask)·softmax·AV → one `FusedAttention` node |
| Semantic operator merging | `passes/merge.py` | MatMul+Add→GEMM, +GELU/ReLU/SiLU→Linear*, Add+LayerNorm/RMSNorm→Fused*, Scale+Softmax |
| DAGS scheduling | `passes/schedule.py` | dependency levels, critical path |
| Partitioning | `passes/partition.py` | graph cut, hypergraph, latency-aware (HEFT-style) placement |
| **Learned plan selector (GNN)** | `search/learned_optimizer.py` | picks passes × devices (32 plans) |
| **RL pass policy (DQN)** | `search/rl_policy.py` | sequential policy over the current graph |
| **Neuro-symbolic search** | `search/neuro_symbolic.py`, `search/rewrites.py` | symbolic rewrite rules + GNN latency model trained on real GPU timings |
| Executors | `utils/executor.py` (NumPy), `utils/torch_executor.py` (CUDA/MPS/CPU), `utils/jax_executor.py` (XLA) | same graph, same weights, three backends |
| Cost model | `utils/cost_model.py`, `utils/gpu_calibration.py` | per-op GPU model **calibrated on a real GPU** |
| GUI / API / reports | `frontend/`, `api_server.py`, `utils/reports.py` | React GUI, REST API, Excel export |

The ten output metrics (latency, throughput, speedup, peak memory, GRR, accuracy preservation, OMR, ACR, kernel-launch reduction, compilation time) are computed by `utils/metrics.py` and shown in the GUI Overview and the Excel export.

## Results

Hardware for all "real GPU" numbers: **{gpu['device_name']}**, torch {gpu['torch']}, float32. No NVIDIA GPU has been used (the Colab notebook is provided but unrun).

### 1. Hand-built architectures (batch 4, seq 32, d_model 64)

{arch_table()}

*GRR = graph reduction ratio, OMR = operator merge ratio, ACR = attention canonicalization rate. Measured speedups are NumPy interleaved timings (±1%); modeled speedups use the calibrated per-op model. Normalization removes nothing on the clean templates (only on the Post-LN ones, which contain planted redundancy) — see §2 for real graphs.*

### 2. Real framework-traced graphs (`benchmarks/real_graph_study.py`)

Traced with `torch.fx.make_fx`; the GPT-2 configs are the real HuggingFace code path with random weights (no download). Outputs are compared with the PyTorch model.

| Model | IR ops | normalize | attention | merge | Final | Attention fused | Max diff vs PyTorch | Same top-1 |
|---|---|---|---|---|---|---|---|---|
{real_rows}

### 3. Trained model (Tiny Shakespeare, `training/train_lm.py`, `benchmarks/lm_full_eval.py`)

112k-parameter, 2-block, character-level Transformer, trained 2,500 steps (~35 s on CPU): validation loss {lm['val_loss']:.3f}. On the **entire validation split** ({lmf['validation_characters']:,} characters): perplexity PyTorch {lmf['pytorch']['perplexity']:.3f} / IR {lmf['ir_original']['perplexity']:.3f} / compiled IR {lmf['ir_compiled']['perplexity']:.3f}; identical top-1 predictions {lmf['compiled_same_top1_as_pytorch_pct']:.1f}%; graph {lmf['graph_nodes'][0]} → {lmf['graph_nodes'][1]} nodes. One attention head only (the IR has no head split/concat).

### 4. Real GPU (`benchmarks/gpu_benchmark.py`) and cost-model calibration (`benchmarks/calibrate_cost_model.py`)

18 runs (6 architectures × 3 sizes), original vs compiled graph, synchronised interleaved timing, all outputs identical:

| | Median | Range |
|---|---|---|
| Real speedup, autotuned attention lowering | **{gs['median_speedup_auto']:.2f}×** | {gs['min_speedup_auto']:.2f}× – {gs['max_speedup_auto']:.2f}× |
| Real speedup, **generated Metal kernels** (fused add+norm, scale+softmax) | **{gs['median_speedup_codegen']:.2f}×** | {gs['min_speedup_codegen']:.2f}× – {gs['max_speedup_codegen']:.2f}× |
| Real speedup, fixed fused-attention kernel | {gs['median_speedup_sdpa']:.2f}× | |
| Modeled speedup, calibrated model | {gs['median_modeled_speedup']:.2f}× | |
| Modeled speedup, old hand-written table | {gs['median_modeled_speedup_legacy_table']:.2f}× | |

The original hand-written cost table over-predicted gains ~1.6×. The per-op model is now **fitted to measured kernel times** (`time = a + b·work` per op class). Fitted on Post-LN + LLaMA only and tested on the other **{v['graphs']} graphs** (4 architectures × 5 sizes):

| Held-out architectures | Old table | Calibrated |
|---|---|---|
| Correlation with measured latency (log) | {v['latency_log_corr_old_table']:.2f} | **{v['latency_log_corr_calibrated']:.3f}** |
| Mean speedup error | {v['speedup_mae_old_table']:.2f} | **{v['speedup_mae_calibrated']:.2f}** |
| Speedup rank correlation | {v['speedup_spearman_old_table']:.2f} | **{v['speedup_spearman_calibrated']:.2f}** |
| Latency error (mean / median) | – | {v['latency_mape_calibrated_pct']:.1f}% / {v['latency_median_ape_calibrated_pct']:.1f}% |

On a real GPU fusion mostly removes launch overhead; only attention and bias-folding reduce actual GPU work. The fixed fused-attention kernel is *slower* than separate matmul+softmax at mid sizes on MPS, which is why an "autotuned" lowering (time both once per shape) exists. The shipped op table is the calibrated one (`docs/results/cost_calibration.json`); `COST_MODEL=legacy` restores the old one.

**Generated fused kernels (`utils/metal_codegen.py`).** For `FusedAddLayerNorm`, `FusedAddRMSNorm` and `FusedScaleSoftmax` the compiler *generates Metal source* specialised to the row length/epsilon, compiles it with `torch.mps.compile_shader` and runs it (verified to ≤2e-6 against PyTorch; 8–30% faster than PyTorch's composed ops in isolation). End to end the gain is small because these ops are a small share of a model's time. Only Apple GPUs; no CUDA code generation. The unfused RMSNorm baseline uses PyTorch's native one-kernel `rms_norm` (an earlier version composed it from four calls, which flattered fusion).

### 5. Learned optimizers (all evaluated on architecture families never seen in training)

* **GNN plan selector** (`search/learned_optimizer.py`): 700 generated graphs × 32 plans (passes × 1–4 devices), reward = deployment speedup − 0.15·(devices−1) − 0.005·(#passes) under the calibrated model. Leave-one-family-out mean reward **{lm_['learned']:.2f}** vs fixed pipeline {lm_['fixed_pipeline_1dev']:.2f}, random {lm_['random_plan']:.2f}, best possible {lm_['oracle']:.2f} (regret {lm_['regret_pct']:.1f}%, within 2% of best on {lm_['within_2pct_of_oracle']:.0f}% of graphs).
* **RL pass policy** (`search/rl_policy.py`): DQN with a GNN Q-network over the *current* graph; actions = apply normalize / canonicalize / merge, place on 2–4 devices, stop; offline training on a tabulated exact simulator. Mean reward **{rlm['rl']:.2f}** (fixed {rlm['fixed_pipeline_1dev']:.2f}, random {rlm['random_policy']:.2f}, best {rlm['oracle']:.2f}; regret {rlm['regret_pct']:.1f}%). It essentially ties the supervised selector.
* **Neuro-symbolic search** (`search/neuro_symbolic.py`, `benchmarks/neuro_symbolic_study.py`): nine symbolic rewrite rules propose every legal rewrite; an additive latency GNN trained on **{nsr['train_graphs']} real-GPU timings** scores them (no GPU at search time). Held-out architectures, final graphs *measured* and checked for identical outputs ({'all match' if nss['all_outputs_match'] else 'MISMATCH'}):

| Held-out architecture | Latency error | Rank corr. | Fixed pipeline | Naive rewrite order | Neuro-symbolic |
|---|---|---|---|---|---|
{ns_rows}

Geomean speedup: fixed **{nss['speedup_fixed']:.3f}×**, naive **{nss['speedup_naive']:.3f}×**, neuro-symbolic **{nss['speedup_neurosymbolic']:.3f}×** (better than fixed in {nss['ns_beats_fixed_cases_pct']:.0f}% of cases, worse in {nss['ns_worse_than_fixed_cases_pct']:.0f}%). It beats the fixed pipeline on several architectures but fails on multi-head attention (a structure unlike any training architecture), so overall it is **on par, not better**. Its value is that correctness is guaranteed by the symbolic rules and no GPU is needed at search time.

**Architectures the importer cannot yet translate:** {', '.join(o['model'] for o in other if not o.get('imported', True))} — they need rotary embeddings, decomposed RMSNorm, `triu`/`cumsum` mask construction and similar ops (see `docs/results/real_graph_study.json`). Real-graph evidence therefore covers GPT-2-style models and our trained Transformer only.

### 6. Comparison with XLA and TorchInductor on the same graphs (CPU, `benchmarks/compiler_comparison.py`)

The same IR graph (same weights/input) run by each system; {len(cmp_['rows'])} architecture×size cases; all outputs agree (max rel. error ≤ 1e-6). "eager" is our PyTorch interpreter.

| System | Geomean speedup vs eager |
|---|---|
{cmp_rows}

Effect of our passes **on top of** each compiler: XLA ×{eff['ours+xla_vs_xla']:.3f}, TorchInductor ×{eff['ours+inductor_vs_inductor']:.3f}. On small CPU workloads the established compilers are *slower* than eager (compile/launch overheads) and our graph-level rewrites add essentially nothing on top of them. Our contribution is therefore the explainable, verifiable pass pipeline and the learned decision-making — not raw speed.

### 7. A never-tuned architecture and sensitivity to my assumptions

* **Unseen architecture** (`benchmarks/unseen_architecture_test.py`): the GPT-J/PaLM-style *parallel block* (attention ‖ FFN) appears in no training set and was never used to choose features or thresholds; models are frozen. Modeled reward on {un['modeled']['graphs']} fresh graphs: fixed {un['modeled']['fixed']:.2f}, plan selector {un['modeled']['selector']:.2f} ({un['modeled']['regret_pct_selector']:.1f}% regret), **RL {un['modeled']['rl']:.2f} ({un['modeled']['regret_pct_rl']:.1f}% regret)**, best {un['modeled']['oracle']:.2f}. Real GPU geomean speedup over 5 sizes: fixed pipeline {un['gpu']['geomean_speedup_fixed']:.3f}×, neuro-symbolic {un['gpu']['geomean_speedup_neurosymbolic']:.3f}×, fixed + generated kernels {un['gpu']['geomean_speedup_fixed_codegen']:.3f}× (outputs identical).
* **Sensitivity** (`benchmarks/learned_sensitivity.py`): the plan selector's advantage over the fixed pipeline holds across the assumed constants, but how often multi-device placement wins depends strongly on them:

| Setting | Transfer µs | Device penalty | Fixed | Learned | Best | Regret | Best plan uses >1 device |
|---|---|---|---|---|---|---|---|
{sens_rows}

{xla_section}
## Honest limitations

* **No NVIDIA GPU measured.** Real-GPU numbers are one Apple M3 Pro (Metal/MPS); `benchmarks/colab_gpu_benchmark.ipynb` is untested.
* **Real speedups are small** (median {gs['median_speedup_auto']:.2f}× on GPU, {gs['median_speedup_codegen']:.2f}× with generated kernels, ~1.00× on CPU). Only three fused ops get generated Metal code; the rest are lowered to PyTorch calls, and only attention (PyTorch's fused kernel) and linear+bias (one GEMM) otherwise reduce real GPU work. No CUDA code generation.
* **Hand-built templates have synthetic weights**; "accuracy preserved" means output equivalence. Only the Tiny Shakespeare LM is trained, and it has one attention head.
* **Learned-optimizer rewards are modeled** (calibrated per-op times; *assumed* 50 µs cross-device transfer and device penalty 0.15 — §7 shows how results move with them). No multi-device run was ever executed. Leave-one-family-out figures are slightly optimistic because features were chosen after seeing them; the frozen unseen-architecture test in §7 is the clean check.
* **The importer handles GPT-2-style graphs only** (see §2); LLaMA/OPT/NeoX fail on unsupported ops.
* Untestable here: the Colab/NVIDIA notebook and `run-server.bat` (Windows).
* The per-op table is shape-independent (calibrated at B8·S256·D256); the shape-aware fit in `utils/gpu_calibration.py` is more accurate but only used for validation.
* Older prototypes (`search/gnn*.py`, `training/train_optimizer.py`) are superseded and kept for history.

## Reproduce

```bash
python benchmarks/real_graph_study.py        # §2   (needs transformers)
python training/train_lm.py                  # §3   (downloads Tiny Shakespeare, 1.1 MB)
python benchmarks/lm_full_eval.py            # §3
python benchmarks/gpu_benchmark.py           # §4   (CUDA > MPS > CPU)
python benchmarks/calibrate_cost_model.py    # §4   (~6 min on GPU)
python -m search.learned_optimizer train     # §5
python -m search.rl_policy train             # §5
python benchmarks/neuro_symbolic_study.py    # §5   (needs a GPU for timings)
python benchmarks/compiler_comparison.py     # §6
python benchmarks/unseen_architecture_test.py && python benchmarks/learned_sensitivity.py
python benchmarks/xla_rewrite_study.py && python tools/make_xla_tables.py   # §8 (CPU)
python tools/make_diagrams.py && python tools/make_readme.py
# or everything, one job at a time:  tools/rerun_all.sh
```
"""
    (ROOT / "README.md").write_text(text)
    print("wrote README.md", len(text.split()), "words")


if __name__ == "__main__":
    main()
