# GraphForge — LLM computation graph optimization

GraphForge is a research prototype for visualizing computation graphs and exploring graph normalization and semantic operator merging. A React dashboard displays hand-built model-shaped DAG examples and runs a separate PyTorch FX attention benchmark with an optional JAX/OpenXLA CPU comparison. Each completed comparison or sweep can be downloaded as a separate Excel workbook.

This repository currently demonstrates synthetic inputs and model-shaped graph templates. The selectable Transformer and MLP DAGs are assembled by project code; they are not captured from a real model file, a text dataset, or the benchmark's input tensors. The separate attention benchmark does trace a small PyTorch attention module with FX. This is **not** an LLM training system, a production compiler, or a trained optimizer based on PassNet. The current GNN training script uses synthetic graphs and modeled labels. Dataset integration and data-backed optimizer training are described below as next work.

## Contents

- [At a glance](#at-a-glance)
- [Active scope](#active-scope)
- [The dataset to use](#the-dataset-to-use)
- [What the current workload computes](#what-the-current-workload-computes)
- [Dashboard controls and graph sources](#dashboard-controls-and-graph-sources)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Algorithms and why these choices](#algorithms-and-why-these-choices)
- [Metrics and their limits](#metrics-and-their-limits)
- [XLA comparison](#xla-comparison)
- [Dashboard and Excel reports](#dashboard-and-excel-reports)
- [Setup and run](#setup-and-run)
- [Training: current state and correct next step](#training-current-state-and-correct-next-step)
- [Repository map](#repository-map)
- [Known limitations](#known-limitations)
- [Sources](#sources)
- [TODO — what is still missing](#todo--what-is-still-missing)

## At a glance

| Question | Current answer |
| --- | --- |
| Where does the graph come from? | The selectable architecture DAGs are hand-built templates. A separate attention benchmark captures a PyTorch function with `torch.fx`. |
| Which optimization passes are active in the dashboard? | Graph normalization and semantic operator merging. DAG construction is the graph representation step. |
| What dataset is recommended for future training/evaluation? | [PassNet](https://huggingface.co/datasets/PassNet/PassNet): a public corpus of graph/model code records and optimization examples. |
| Is PassNet loaded and used to train the current model? | No. The data adapter and evaluation-label pipeline are not integrated yet. |
| What is the current XLA comparison? | When JAX is installed, a real JAX `jit`/OpenXLA CPU compile-and-run of the separate attention function; compile time and steady-state time are separate. |
| Where are run reports? | `outputs/benchmark_reports/metrics_<UTC timestamp>_<id>.xlsx`, one workbook per completed dashboard comparison or sweep. |

## Active scope

The current experiment intentionally concentrates on:

1. **DAG construction:** build nodes and dependency edges from a selected custom IR template, or capture the separate supported PyTorch attention function with FX.
2. **Graph normalization:** clean up the representation while preserving its computation.
3. **Semantic operator merging:** replace recognized multi-operation patterns with a single semantic fused node.

Attention canonicalization as a separate pass, DAGS scheduling, graph partitioning, RL search, and compiler pass generation are not part of the dashboard's active optimization run. Some of those modules remain in the repository from earlier experiments; their presence does not mean the dashboard currently applies them.

Scope detail: the **selected custom IR graph pipeline** runs normalization and semantic merging, with DAG construction as the representation step. The separate attention FX scripts also call `schedule_fx` to calculate dependency levels and attach schedule metadata after merging. That helper does not reorder the FX graph or change the function executed by PyTorch, and it is not used by the selected custom IR pipeline. The scheduling metadata should not be interpreted as a measured scheduling optimization; removing it from the attention benchmark or surfacing it as a separate analysis is on the TODO list.

## The dataset to use

### Recommendation: PassNet

Use the **PassNet** graph corpus and its related PassBench evaluation examples as the leading source for compiler-optimization training data. The Hugging Face dataset card currently reports **2,067,810 records and 967 MB** in Parquet form, with an MIT license. The records contain paths and text content for model files, graph metadata, weights metadata, and input metadata. The PassNet paper describes over 18,000 unique computation graphs mined from 100,000 real-world models; those are unique graphs, while the 2.07M Hugging Face rows are file-level records and should not be reported as 2.07M unique graphs. [Dataset card](https://huggingface.co/datasets/PassNet/PassNet) · [PassNet paper](https://arxiv.org/abs/2605.29357) · [PassNet source repository](https://github.com/PaddlePaddle/PassNet)

PassNet is relevant because it contains real model subgraphs and fusible subgraph examples, which match the graph-rewrite problem more closely than a natural-language prompt corpus or a classification dataset. Its benchmark flow checks output correctness and measures execution speed after a candidate pass. The upstream evaluation setup is CUDA-centered (its README currently specifies Python 3.12+, PyTorch 2.9+, CUDA 12.8, and an NVIDIA GPU); it is not a drop-in PyTorch FX dataset loader for this project. We should use its graph examples as source material, then adapt and re-evaluate compatible cases with our own passes and devices.

PassNet and a text corpus solve different problems. **PassNet is a candidate source of graph structures and compiler-optimization examples.** A dataset such as WikiText-2 can provide real text to tokenize and feed into a language model, but text alone does not provide graph-rewrite labels. A real end-to-end benchmark needs both an actual model with weights and actual tokenized inputs; a graph-optimizer training set additionally needs correctness and measured-performance labels generated by running our transformations.

### Why not the other common choices?

| Option | Why it is not the first choice for this project |
| --- | --- |
| MNIST | Image labels train a classifier. They do not label whether graph normalization or a semantic merge is correct or faster. The existing MNIST MLP experiment is separate from the attention graph work. |
| A text dataset such as C4 or WikiText | It supplies model inputs for language modeling, not computation DAGs or pass-performance labels. It is not an optimizer training dataset by itself. |
| TorchBench | It is a useful suite of real PyTorch workloads and benchmark harnesses, but it is a model/workload suite rather than one ready-made corpus of rewrite examples. It is a strong later validation set. [TorchBench](https://github.com/pytorch/benchmark) |
| Google TPU Graphs | It contains computation graphs and compiler-configuration runtime labels, but targets TPU/XLA layout and tiling choices. That makes it useful for graph cost prediction, less direct for the current PyTorch semantic-rewrite task. [TPU Graphs](https://github.com/google-research-datasets/tpu_graphs) |
| Random synthetic DAGs | Useful for checking that a graph model can train and that code paths run; they do not establish real-workload generalization. |

### What a real data pipeline must do

PassNet should not be treated as a table of ready-to-use labels for *our* passes. The project needs to build those labels:

1. Read the upstream training split and retain graph/model records that can be represented by the operations our IR understands.
2. Convert each supported case into a DAG with operation type, dependency edges, tensor shapes, dtype, constants/attributes, and source-model identity.
3. Run baseline and normalized/merged programs on the same device, inputs, and weights.
4. Reject cases that fail graph capture or numerical equivalence.
5. Record repeatable baseline and optimized latency, peak device memory when available, transformed node/edge counts, and pass time. Keep CPU measurements, GPU measurements, and cost-model estimates in separate columns.
6. Split by **source model**, not by individual graph-file row, so near-duplicate graphs from one model cannot leak between train and test.
7. Train a PyTorch GNN to predict candidate benefit or rank whether a legal normalization/merge should be applied. Keep the optimizer's rewrites symbolic and rule-based; do not have a model invent arbitrary graph code.
8. Evaluate the trained ranker on a held-out model set and confirm that the selected rewrites still pass numerical checks and improve measured runtime.

The PassNet dataset is identified and documented here, but this adapter, GPU-label collection, and PassNet-trained checkpoint are **not implemented yet**. No downloaded dataset or PassNet-trained model is included in this checkout.

## What the current workload computes

The dashboard's attention demo uses a small PyTorch module with input tensor shape `(batch, sequence, d_model)`:

```python
q = x @ Wq
k = x @ Wk
v = x @ Wv
scores = (q @ k.transpose(-2, -1)) / sqrt(d_model)
weights = softmax(scores, dim=-1)
output = weights @ v
```

The actual module and FX merge implementation are in [`models/attention.py`](models/attention.py) and [`passes/semantic_merge.py`](passes/semantic_merge.py). The sample is deliberately small enough to run on a CPU. It gives us graph patterns such as matrix multiplication, transpose, scalar division, softmax, and the final attention/value product. It is not a full LLM with tokenizer, embedding table, KV cache, decoder stack, or a text-generation dataset.

The UI also offers model-shaped IR examples, including Transformer-shaped graphs and an MLP classifier. Those diagrams exercise the repository's custom IR pipeline; they are assembled from fixed operation templates. The per-shape attention sweep is a separate PyTorch FX benchmark path.

## Dashboard controls and graph sources

The controls currently combine two related but separate demonstrations. **Run comparison** applies the custom IR passes to the selected architecture template, then also runs the separate attention benchmark and attempts to save both results in one workbook. The model selector changes the custom IR example. Batch, sequence, and `d_model` are used as shape/configuration inputs; they also parameterize the separate attention benchmark. The selected architecture does **not** change the function used by the XLA comparison.

### Model graph selector

| UI choice | What the template represents | What it does not represent |
| --- | --- | --- |
| Post-LN Transformer (2 blocks) | A two-block post-LayerNorm Transformer-shaped custom IR graph. Each block represents projections, scaled attention, a feed-forward section, residual paths, normalization, and a dropout/identity node. | It is not a loaded Transformer checkpoint and does not consume token IDs. |
| Post-LN Transformer (4 blocks) | The same template repeated for four blocks. Its current builder produces 76 nodes before normalization. | It is not a four-layer pretrained language model. |
| Pre-LN Transformer (GPT/LLaMA) | A two-block pre-LayerNorm-shaped custom IR graph with normalization before attention/feed-forward branches. | It is not an actual GPT/LLaMA implementation, tokenizer, or weight set. |
| Post-LN + ReLU FFN | The post-LN template with ReLU in the feed-forward activation slot, for exercising activation-related rewrite rules. | It is not a trained model or a runtime kernel fusion. |
| MLP Classifier (784-256-64-10) | A fully connected graph with 784 input features, hidden widths 256 and 64, and 10 output values/classes. The 784 dimension is consistent with flattened 28×28 images. | Despite the UI description, the current API creates random graph weights and does not load MNIST examples or a trained MNIST checkpoint. |

The Transformer and MLP graph builders are in [`models/transformer.py`](models/transformer.py) and [`models/architectures.py`](models/architectures.py); model selection and request handling are in [`api_server.py`](api_server.py). These files explicitly add graph operations and dependency edges. This is intentional prototype scaffolding, not automatic model discovery.

### Numeric workload controls

| Control | Meaning | Current use and limits |
| --- | --- | --- |
| Batch | Number of examples/sequences processed together. | Stored in Transformer graph configuration and used as the leading dimension in the attention benchmark input. It normally changes tensor sizes, not the count of operation nodes. The MLP builder fixes its graph batch to 256. |
| Sequence | Number of token positions in each synthetic sequence. | Stored in the Transformer graph configuration and sets the middle dimension of attention input. These positions are not tokenized words. |
| `d_model` | Width of the feature vector at each token position (Transformer hidden size). | Stored in graph metadata and sets the last attention input dimension plus the Q/K/V projection matrix sizes. It normally changes tensor shapes and compute volume, not the operation-node count. |

The attention benchmark input is a random float32 tensor `x` shaped `(batch, sequence, d_model)`. Its three projection matrices are also initialized randomly. With the default values, the input shape is `(4, 64, 64)`. The sequence positions do not contain words, and there is no tokenizer, language corpus, embedding table, or pretrained LLM in this path. The FX benchmark creates and checks actual tensor outputs, but it is a synthetic attention microbenchmark rather than an end-to-end language-model run.

### Why node counts stay fixed—and known count mismatch

A computation graph's node count reflects how many operations its template contains. Batch size and tensor dimensions affect the data flowing through those operations but do not replicate the operation nodes, so the count remains stable when those controls change. Changing the selected architecture or number of Transformer blocks should change the graph. In the current builder, the two-block post-LN example starts at 42 nodes and the four-block example starts at 76 nodes (before normalization). The `/api/models` catalog currently advertises 78 for the four-block example; that catalog value is stale and should be corrected. If the displayed graph does not change after selecting a different architecture and rerunning, check that a new comparison completed and that you are viewing its result.

### What operation is benchmarked and compared?

The separately traced FX/XLA operation is single-head scaled dot-product self-attention:

```python
q = x @ Wq
k = x @ Wk
v = x @ Wv
scores = (q @ k.transpose(-2, -1)) / sqrt(d_model)
weights = softmax(scores, dim=-1)
output = weights @ v
```

The original and semantically rewritten PyTorch functions use the same generated input and weights. The XLA arm computes the equivalent equations in JAX, lowers and compiles them with `jax.jit`/OpenXLA, then executes the compiled function on the JAX CPU device. This XLA result is not the selected custom Transformer DAG compiled by XLA, and it is not a comparison against our own generated GPU kernel. The XLA result is real when the API reports a successful run with backend/device details and StableHLO; if JAX is missing or compilation fails, the UI should report the backend as unavailable instead of treating a previous number as a new measurement.

### Dashboard sections

| Section | What it displays |
| --- | --- |
| Overview | Latest reported counts, selected metrics, and bar-chart summaries. Read each chart's “measured,” “modeled,” or “estimated” label before interpreting a value. |
| Graph transforms | Original and transformed custom IR graphs, stage counts, and normalization/semantic-merge results for the selected architecture template. Nodes/edges are the visible graph structure; they are not a generated machine-code listing. |
| XLA comparison | PyTorch eager, PyTorch FX semantic reference, and JAX/OpenXLA CPU timings for the separate attention workload; also correctness, compile time, and StableHLO operation count when available. |
| Stress sweep | Repeats the attention microbenchmark for a collection of batch/sequence/hidden-size shapes. It is not a dataset evaluation. |

The attention semantic node is a graph-level rewrite description. Unless a backend actually lowers it to a fused kernel, the name “fused” alone does not mean fewer physical GPU launches or faster execution.

## Pipeline walkthrough

### 1. Create a PyTorch workload

Choose the model and tensor dimensions. The weights and random input are initialized deterministically in the sweep so each baseline/optimized pair receives the same values.

### 2. Capture or construct a DAG

For the attention microbenchmark, `torch.fx.symbolic_trace(model)` captures the Python-level forward graph. Each FX operation becomes a node, and each value dependency becomes an edge. The graph is a DAG because forward tensor dependencies are acyclic for this static attention example. This is not how the selectable Transformer and MLP templates are built: those use explicit project code to create custom IR operations and edges.

```python
import torch.fx as fx

original = fx.symbolic_trace(model)
optimized = fx.symbolic_trace(model)
```

The dashboard also has a custom `ComputationGraph` IR with explicit operation attributes and dependency edges. Its architecture examples are created in `models/architectures.py` and `models/transformer.py`.

### 3. Normalize the graph

The normalizer standardizes op names, removes identity nodes, folds supported constant-only operations, merges safe duplicate expressions, removes unreachable/dead operations, and infers shape metadata where available. The pass lives in [`passes/normalize.py`](passes/normalize.py). The transformations must preserve values; the report's output comparison is the guardrail.

### 4. Merge semantic operator patterns

The FX attention rewrite recognizes the score/scale/softmax/value sequence and replaces the recognized subgraph with a `fused_attention` semantic operation. Other IR semantic merges are implemented in `passes/merge.py`. These nodes describe a fused operation at the graph/semantic level. They do not by themselves prove that the runtime emitted a single optimized GPU kernel; the current measured FX path executes a Python/PyTorch reference function.

```python
normalized = normalize(optimized)
merged = semantic_merge_attention(normalized)
if not merged:
    raise RuntimeError("Attention pattern not detected")
```

### 5. Check correctness

The benchmark uses `torch.allclose` with documented tolerances and also reports the maximum absolute output error. A transformation that does not match its baseline must not be presented as a successful optimization.

### 6. Measure and compare

The local FX sweep warms each function, records repeated wall-clock calls, and reports medians, latency, token throughput, graph-size change, pass time, and correctness. The XLA panel compiles an equivalent JAX function and separates compile time from steady-state latency.

### 7. Save the run

After a comparison or sweep completes, the API generates a new timestamped `.xlsx` workbook. It does not overwrite the previous workbook.

## Algorithms and why these choices

### DAG representation

A DAG makes data dependencies explicit. That allows the project to identify patterns spanning multiple operators and visualize exactly which producer/consumer relationships a rewrite changes. Alternatives such as timing only the whole model make it harder to explain or compare transformations.

### Graph normalization

Normalization is applied before pattern matching so redundant or inconsistent graph structure does not unnecessarily block a semantic rewrite. It is deterministic and inspectable. The current pass handles a limited operator and attribute set; it is not a universal algebraic optimizer.

### Semantic operator merging

The merger recognizes known patterns instead of merging by operation names alone. The attention rewrite preserves the meaning of the observed Q/K/V path and emits a semantic fused node. Pattern rules are easier to audit for correctness than a learned system that directly edits arbitrary graphs.

### Why not train a large language model to output the DAG?

The DAG is already determined by the framework program and tensor dependencies. `torch.fx` can capture it without training. A generative language model can emit invalid edges, omit dependencies, or produce a graph inconsistent with tensor shapes. The useful learned subproblem is usually **predicting which legal rewrite is worth trying** from graph structure and measured labels. The current repository's GNN is an early prototype; it is not used to build the attention DAG.

### Other ideas shown in the reference methodology image

The image also mentions attention canonicalization, Dependency-Aware Graph Scheduling, and graph partitioning/scheduling. These are reasonable compiler techniques, but they are deliberately outside the current dashboard run so that comparisons isolate normalization and semantic operator merging. They should be added only with their own correctness and measurement evidence.

## Metrics and their limits

The dashboard and workbooks use these requested measures where the current pipeline can provide a defensible value. The active API has multiple paths: custom architecture-template graph metrics, a separate FX attention microbenchmark, and a separate XLA attention benchmark. A metric belongs to the path identified by its panel/workbook context; do not read all displayed values as measurements of one end-to-end LLM run.

| Metric | Meaning in this project | Status / interpretation |
| --- | --- | --- |
| Inference latency | Time for one invocation of the measured graph/function. | The selected custom-IR overview uses the minimum of 30 NumPy-executor samples after 2 warmups. The FX sweep and XLA panel report medians after their own warmups. All are CPU reference timings; compare only within matching paths/settings. XLA compile time is separate. |
| Throughput | Completed items per second, derived from latency. | Custom-IR overview reports samples/s from batch size. Attention FX/XLA report token positions/s (`batch × sequence / latency`). Neither is server request throughput. |
| Speedup | Baseline latency divided by optimized latency. | Above 1 means faster; below 1 means slower. Small CPU changes are noisy. |
| Peak GPU memory | Device high-water memory usage. | Not measured. The custom IR overview tracks live NumPy array bytes inside its executor, and the cost-model GPU-like memory estimate is separate; neither is actual GPU process/device memory. |
| Graph Reduction Ratio (GRR) | `(original compute nodes − optimized compute nodes) / original compute nodes`. | Structural change; a smaller graph is not automatically faster. In the architecture comparison this counts custom IR nodes; in the FX sweep it counts FX compute nodes. |
| Accuracy preservation | Whether outputs are within tolerance, plus max absolute error. | A pass/fail check, not a task-level language-model accuracy score. |
| Operator Merge Ratio (OMR) | Fraction of original graph nodes removed by the semantic merging pass. | Structural rewrite ratio; not a speedup. |
| Attention Canonicalization Rate (ACR) | Fraction of detected attention patterns merged by the attention rule inside semantic merging. | Reported when a supported attention pattern is detected; otherwise `N/A`. Attention matching is currently inside semantic merging, not a separate active pass. |
| Kernel launch reduction | Difference between original and optimized estimated launches. | Heuristic proxy based on custom IR operations, not an observed GPU launch count. |
| Compilation time | Time spent building/transforming a graph or compiling XLA, depending on the panel. | Custom IR pass time, FX transform time, and backend compile time are distinct. The XLA panel reports JAX/XLA compilation separately from execution. |

The custom cost model in [`utils/cost_model.py`](utils/cost_model.py) assigns hand-written per-op estimates. These values are for illustration and must always be labeled **modeled**. The NumPy executor in `utils/executor.py` reports reference-run measurements, not PyTorch GPU performance. Benchmark conclusions should use the same device, dtype, input shapes, warmup, and measurement methodology across backends.

## XLA comparison

[`xla_compare.py`](xla_compare.py) compares the same seeded attention workload and weights using PyTorch FX and JAX `jit`/OpenXLA on CPU. The report includes:

- median latency and derived token throughput after warmup;
- separate XLA/JAX compilation time;
- output error and tolerance check against the PyTorch reference;
- FX node counts before/after this project's normalization and merge;
- StableHLO operation count, kept separate because StableHLO and FX are different IRs.

This is a small-scale reference comparison. It does not compare identical compiler pipelines, does not establish GPU XLA performance, and does not claim that our semantic fused node is emitted as a fused low-level kernel. A backend can be slower for small CPU shapes, as observed in the earlier demo.

## Dashboard and Excel reports

The site is a React/Vite frontend backed by a small Python HTTP API. The graph panels lay nodes out by topological dependency depth. Each graph has `+`, `−`, and `Fit` controls, and both axes scroll when the graph extends beyond the panel. This avoids squeezing a wide graph into a tiny fixed-size preview.

When a **Run comparison** action finishes, the Excel file includes:

- `Summary`: run ID/time, workload, all compiler metrics and pass metadata;
- `XLA`: backend timing, throughput, compile duration, correctness, and IR counts;
- `Graph stages`: graph node counts before and after the active passes.

When a stress sweep finishes, the workbook includes a `Workload sweep` sheet with one row per input shape, including latency, throughput, GRR/OMR, error, correctness, pass time, and explicit N/A/estimated values for GPU-only or inactive-pass metrics. Each new run creates a fresh filename in `outputs/benchmark_reports/`. The dashboard's **Download XLSX** control downloads the latest workbook. Generated reports are ignored by Git.

The writer uses the bundled `@oai/artifact-tool` spreadsheet runtime through [`tools/export_metrics.mjs`](tools/export_metrics.mjs). In this checkout, `tools/node_modules` is a local junction to the Codex bundled runtime; it is ignored by Git. If using the project outside this Codex desktop environment, configure Node.js and install/provide `@oai/artifact-tool` under `tools/node_modules` before using Excel export. The benchmark itself does not depend on Excel.

## Setup and run

### Requirements

- Python 3.12 (the launcher uses `.venv` and does not install Python automatically)
- Node.js and npm
- A browser
- Optional: JAX for the XLA tab. The current comparison is CPU-only.

### One-time setup on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend install
```

If `py -3.12` reports “No suitable Python runtime found,” install Python 3.12 and the Windows Python launcher, reopen PowerShell, then repeat the setup. `run-server.bat` expects the `.venv` to exist; it intentionally does not install Python or silently select another interpreter.

### Start the dashboard

```powershell
.\run-server.bat
```

Open [http://localhost:5173](http://localhost:5173). The launcher starts the API on `127.0.0.1:8000` and Vite on port 5173. Keep the launcher terminal open while using the site.

### CLI examples

```powershell
.\.venv\Scripts\python.exe run_attention_optimization.py
.\.venv\Scripts\python.exe training\train_optimizer.py --graphs 96 --epochs 160
npm --prefix frontend run build
```

The current `training/train_optimizer.py` command trains the experimental GNN on synthetic random DAGs with **modeled** speedup labels. It is for exercising the training code; it is not a PassNet training command and must not be described as a real-data model.

## Training: current state and correct next step

The current code has two distinct concerns:

- `models/attention.py` plus FX tracing produce the computation DAG from a PyTorch model. This is deterministic model capture, not ML generation.
- `search/gnn_torch.py` defines a small message-passing scorer, but its current `train()` function creates synthetic DAGs and labels them with the project's cost model. It is a prototype scorer and does not drive the dashboard's normalization/merge passes.

For real training, first add a PassNet reader and a graph adapter, then execute this project's *actual* normalization and merge rules against compatible samples and collect correctness/performance labels on the intended hardware. Only after this dataset is available should the GNN be trained to predict optimization benefit. Training a predictor on artificial cost-model scores would merely teach it the same assumptions already encoded by that cost model.

## Repository map

| Path | Purpose |
| --- | --- |
| `frontend/src/App.jsx` | React dashboard, benchmark requests, metric charts, graph visualization, workbook download. |
| `frontend/src/App.css` | Dashboard and scrollable DAG display styles. |
| `api_server.py` | API for model graph optimization, sweeps, XLA, and report downloads. |
| `ir/graph.py`, `ir/fx_graph.py` | Custom computation DAG and FX graph utilities. |
| `models/attention.py` | PyTorch attention workload used for FX/XLA comparisons. |
| `models/architectures.py`, `models/transformer.py` | Model-shaped custom IR example graphs. |
| `passes/normalize.py` | Custom IR normalization. |
| `passes/semantic_merge.py`, `passes/merge.py` | FX attention and custom IR semantic rewrites. |
| `run_attention_optimization.py` | FX reference benchmark and workload sweep. |
| `xla_compare.py` | Same-input PyTorch/JAX XLA comparison. |
| `utils/metrics.py`, `utils/cost_model.py`, `utils/executor.py` | Metric definitions, labeled estimates, and NumPy reference executor. |
| `training/`, `search/` | Earlier MLP/GNN/search experiments; not the active dashboard training loop. |
| `tools/export_metrics.mjs` | Writes a workbook for each report request. |
| `outputs/benchmark_reports/` | Generated run-specific `.xlsx` files (Git-ignored). |

## Known limitations

- PassNet is recommended but not downloaded, adapted, or used for training in this version.
- The dashboard's measured attention runs are CPU reference measurements. GPU memory and launch counts are not measured by the current FX benchmark.
- A semantic fused node is a graph rewrite description, not proof of a generated optimized kernel.
- The XLA comparison is JAX/OpenXLA on CPU; device support and timings vary by installation and hardware.
- The current custom IR executor and per-op GPU cost table are educational estimates, not a production compiler backend.
- Graph normalization and semantic merging support selected ops/patterns. Unsupported ops should pass through unchanged or be rejected; never assume all PyTorch models are transformable.
- A lower node count alone is not a performance result. Use repeated same-device measurements and correctness checks.

## Sources

- PassNet dataset card: https://huggingface.co/datasets/PassNet/PassNet
- PassNet paper: https://arxiv.org/abs/2605.29357
- PassNet implementation and benchmark: https://github.com/PaddlePaddle/PassNet
- TorchBench: https://github.com/pytorch/benchmark
- Google TPU Graphs: https://github.com/google-research-datasets/tpu_graphs
- OpenXLA StableHLO overview: https://openxla.org/stablehlo
- JAX installation support: https://docs.jax.dev/en/latest/installation.html
- JAX JIT/compilation timing: https://docs.jax.dev/en/latest/jit-compilation.html

## TODO — what is still missing

This list separates the current demo from the work needed for a data-backed, input-derived LLM compiler experiment. The order is intentional: make the benchmark truthful and reproducible before using its results to train or claim an optimizer.

### P0 — Make the current demo labels and graph behavior unambiguous

- [ ] Change the `/api/models` catalog's four-block node count from the stale 78 to the 76 nodes actually built, or compute the count from the graph builder instead of storing it manually.
- [ ] Change the MLP selector description so it does not say “trained on MNIST” until real MNIST examples and a trained checkpoint are loaded. Current graph weights are random.
- [ ] Keep the custom-template graph metrics, FX attention measurements, XLA attention measurements, and modeled GPU estimates visibly separated in the UI and workbook.
- [ ] Show actual selected model ID, graph input/configuration metadata, backend device, dtype, and successful/unavailable state with each exported comparison.
- [ ] Remove the hidden `schedule_fx` metadata analysis from the active attention comparison or give it a clearly separate DAGS analysis panel, so the advertised active algorithm scope is exact.

### P1 — Use a real model and real inputs

- [ ] Select a small, licensed pretrained causal language model suitable for CPU development and document its exact checkpoint, tokenizer, license, and dependency versions.
- [ ] Add a small text evaluation corpus such as WikiText-2, tokenize it using the model's tokenizer, and use deterministic held-out token sequences as real inputs. Do not commit model weights or a downloaded dataset unless their licenses and repository size make that appropriate.
- [ ] Trace the selected supported PyTorch model with `torch.fx` and display the captured graph. Keep the hand-built templates only as explicitly labeled examples.
- [ ] Apply normalization and semantic merging to graph patterns actually found in that trace. Preserve unsupported operations or reject unsupported graphs safely.
- [ ] Check the model's output logits before and after each rewrite within numerical tolerances; also report a task-level measure such as held-out next-token loss/perplexity when the complete model path supports it.

### P1 — Make the XLA comparison same-workload

- [ ] Make XLA consume the same selected model subgraph, tensors, weights, dtype, and input token batch as the PyTorch path. The current XLA attention function is a separate equivalent equation, not the selected architecture DAG.
- [ ] Record the JAX/XLA backend and device explicitly. Keep compile time separate from warmed steady-state latency and repeat measurements enough to report a median and spread.
- [ ] Compare the same supported rewrite at the same scope. Do not compare StableHLO operation counts directly with FX/custom-IR node counts; they are different representations.
- [ ] Run GPU comparisons only on available supported hardware and report measured GPU memory and launches only when the backend actually measures them.

### P2 — Build real optimizer training data

- [ ] Add a PassNet reader and license/provenance tracking. Filter to graphs and operations the project can capture and execute.
- [ ] Generate labels by applying this project's legal normalization/merge candidates, checking numerical equivalence, and collecting repeated runtime measurements on the target device.
- [ ] Split train/validation/test data by source model to avoid near-duplicate graphs leaking across splits.
- [ ] Train the PyTorch GNN to rank legal rewrite candidates or predict measured benefit. Keep the rewrite itself rule-based and correctness-checked; do not let the model invent arbitrary graph code.
- [ ] Evaluate on held-out models and publish correctness, latency, structural metrics, and model/data versions. Retire synthetic cost-model training labels from any claim of real-workload performance.

### P2 — Make team setup reproducible

- [ ] Replace machine-specific or Codex-only spreadsheet runtime wiring with a documented, installable dependency so XLSX export works in a fresh teammate checkout.
- [ ] Add a clean setup check for supported Python, PyTorch, JAX CPU, Node.js, and npm versions; document optional CUDA setup separately.
- [ ] Add automated checks for graph construction, supported normalization/merge rules, numerical equivalence, XLA-unavailable behavior, and workbook generation.
- [ ] Define a reproducible benchmark protocol: hardware/software metadata, warmup, repeat count, input seeds, synchronization, and report schema.
