# GraphForge — Compiler Optimization for LLM Computation Graphs

A compiler-style optimizer for LLM computation graphs, inspired by XLA/OpenXLA. A model is represented as a DAG of tensor operations; five passes rewrite it; a reference executor and a GPU cost model measure the result; a React GUI shows everything.

## Quick start

```bash
./run.sh            # macOS/Linux: API on :8000, GUI on http://127.0.0.1:5173
run-server.bat      # Windows (after creating .venv and installing requirements.txt)
python -m unittest discover -s tests -t .   # correctness tests
python tools/make_diagrams.py               # regenerate figures in docs/diagrams
```

Requires Python 3.12 and Node.js. (`requirements.txt` includes torch and jax for the FX/XLA tabs.)

## The passes (the three required algorithms + two extras)

| Pass | File | What it does |
|---|---|---|
| **Graph normalization** | `passes/normalize.py` | Identity removal, constant folding, common-subexpression elimination, dead-code elimination, shape inference, **transpose algebra** ((Aᵀ)ᵀ→A; A·Bᵀ→one transposed GEMM) and Scale-chain merging |
| **Attention graph canonicalization** | `passes/canonicalize.py` | Detects Q/K/V · QKᵀ · scale · (mask) · softmax · AV and replaces it with one `FusedAttention` node |
| **Semantic operator merging** | `passes/merge.py` | MatMul+Add→GEMM, +GELU/ReLU/SiLU→Linear*, Add+LayerNorm/RMSNorm→Fused*, Scale+Softmax→FusedScaleSoftmax |
| **DAGS** (Dependency-Aware Graph Scheduling) | `passes/schedule.py` | Dependency levels, critical path, list schedule |
| **Graph / hypergraph partitioning** | `passes/partition.py` | k-way split for multi-device placement |

`pipeline.py` runs them in order. Every pass keeps the model's outputs identical — this is checked on every run.

## Workloads / architectures

Post-LN Transformer (2 and 4 blocks), Pre-LN (GPT-2 style), Post-LN+ReLU, **LLaMA-style** (RMSNorm + SwiGLU), MLP classifier, and an **operation lab** (`models/operations.py`) of tiny graphs that each isolate one idea: transpose algebra, scale chain + CSE, MLP block, single attention head. Real GPT-2 graphs can be imported with `models/import_hf.py` (`pip install -r requirements-real.txt`, needs internet for weights).

## The ten output metrics

Inference latency, throughput, speedup, peak GPU memory, Graph Reduction Ratio (GRR), model accuracy preservation, Operator Merge Ratio (OMR), Attention Canonicalization Rate (ACR), kernel-launch reduction, compilation time — computed in `utils/metrics.py`, shown in the GUI Overview tab and exported to Excel (`utils/reports.py`).

Sample result (batch 4, seq 32, d_model 64, all passes except partitioning):

| Architecture | Nodes | GRR | OMR | ACR | Launch red. | Speedup meas. | Speedup model | Accuracy |
|---|---|---|---|---|---|---|---|---|
| Post-LN Transformer (2 blocks) | 42 → 14 | 66.7% | 26.2% | 100% | 65.0% | 1.24× | 1.73× | PASS |
| Post-LN Transformer (4 blocks) | 76 → 24 | 68.4% | 27.6% | 100% | 66.7% | 1.26× | 1.76× | PASS |
| Pre-LN Transformer (GPT-2 style) | 36 → 16 | 55.6% | 22.2% | 100% | 55.6% | 1.17× | 1.67× | PASS |
| Post-LN + ReLU FFN | 42 → 14 | 66.7% | 26.2% | 100% | 65.0% | 1.26× | 1.74× | PASS |
| LLaMA-style (RMSNorm + SwiGLU) | 37 → 22 | 40.5% | 8.1% | 100% | 40.5% | 1.15× | 1.71× | PASS |
| MLP classifier (784-256-64-10) | 13 → 5 | 61.5% | 38.5% | N/A | 61.5% | 1.07× | 1.08× | PASS |

### How to read the numbers (honest limitations)

* **Measured** = NumPy reference executor on CPU (noisy; run-to-run variation of ±10% is normal — a no-pass baseline can show 1.0–1.1×).
* **Modeled** = hand-written GPU cost table in `utils/cost_model.py`. **Peak GPU memory is modeled, not measured: no GPU is used.**
* Kernel-launch reduction counts surviving operator nodes; it is a proxy, not a profiler reading.
* The PyTorch FX attention benchmark (`run_attention_optimization.py`) reduces nodes 11→4 but runs the same math, so its speedup is ≈1.0×. The JAX/XLA comparison (`xla_compare.py`) is CPU-only and uses separate equivalent code.
* The model graphs are templates with synthetic weights/inputs; accuracy preservation means output equivalence, not task accuracy.
* The learned optimizer trains on generated graphs with cost-model labels (see above); PassNet-style real-graph data with measured GPU labels is future work. `search/gnn*.py`, `search/rl.py`, `search/neuro_symbolic.py` are older prototypes.

## Learned optimizer (GNN plan selection)

`search/learned_optimizer.py` — the rewrite rules stay symbolic (so outputs are provably preserved), but **which passes to run and across how many devices to place the graph is learned**.

* **Action space:** 8 pass subsets (normalize / canonicalize / merge) × 1–4 devices = 32 plans. Multi-device placement uses the new latency-aware (HEFT-style) partitioner in `passes/partition.py`.
* **Reward (modeled):** deployment speedup − 0.15·(devices − 1) − 0.005·(#passes), from `utils/cost_model.deployment_latency_us` (hand-written op costs + an assumed 150 µs per cross-device transfer). Not measured on a GPU.
* **Data:** 700 generated graphs (7 families × 100: Post-LN, Pre-LN, ReLU, LLaMA, MoE, multi-head, MLP; random depth/width and injected redundant ops). Each graph is labelled by actually running all 32 plans.
* **Model:** directed message-passing GNN (3 layers, 64 hidden) → predicted reward of all 32 plans → pick the argmax.
* **Evaluation:** leave-one-family-out (train on 6 families, test on the unseen 7th).

| Mean over held-out families | Reward |
|---|---|
| No passes | 1.00 |
| Fixed pipeline, 1 device (old default) | 1.48 |
| Fixed pipeline, 4 devices | 1.76 |
| **Learned plan** | **1.81** |
| Oracle (best of 32) | 1.86 |

Mean regret vs oracle 3.3%; within 2% of oracle on 70% of graphs. Large gains on the wide families (MoE 2.05 vs 1.20, multi-head 3.50 vs 1.89 fixed) even when held out; weakest on the MLP family (10.8% regret) and LLaMA (5.9%). Caveats: MoE and multi-head are structurally similar to each other, so each helps the other generalize; rewards come from the cost model, so the model learns that cost model; exact-plan match is only 21% (many plans are near-ties). Retrain: `python -m search.learned_optimizer train` (~1 min). Results: `docs/results/learned_results.json`; GUI tab "Learned optimizer".

## Trained model experiment (Tiny Shakespeare)

`python training/train_lm.py` trains a 112k-parameter, 2-block, pre-LN character-level Transformer on Tiny Shakespeare (1.1 M characters, from the public karpathy/char-rnn repo; downloaded to `training/data/`, not committed), exports the **trained** weights into the IR, compiles it with all passes, and checks the result against PyTorch. Trained weights are committed in `models/tiny_lm_weights.npz`, so the GUI lists "Trained Tiny Shakespeare LM". Results are saved in `docs/results/lm_results.json`.

| Result (2,500 steps, ~35 s on a laptop CPU) | Value |
|---|---|
| Validation loss / perplexity | 1.715 / 5.55 |
| Graph | 48 → 21 nodes (GRR 56.3%), ACR 100% |
| Compiled vs PyTorch logits, max abs diff | 4.4e-5 (float32 rounding) |
| Identical top-1 next-character predictions | 100% |
| Next-char NLL: PyTorch vs compiled | 1.4985 vs 1.4985 |

Here the *model* is trained; the plan-selection above is the separate learned optimizer. The tiny model has a single attention head (the IR has no head-split/concat ops).

## GUI tabs

Overview (10 metrics + charts) · Graph transforms (before/after DAG) · Architectures (side-by-side table, Excel export) · Ablation (each pass alone vs all) · Diagrams (system architecture, block diagram, pass figures) · XLA comparison · Stress sweep. Passes can be toggled; partitioning supports graph or hypergraph mode.

## Layout

`ir/` graph IR · `passes/` the five passes · `models/` architectures, operation lab, HF import · `utils/` executor, cost model, metrics, reports · `api_server.py` REST API · `frontend/` React GUI · `tools/make_diagrams.py` figures · `tests/` · `docs/diagrams/` figures.
