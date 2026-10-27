# GraphForge

GraphForge is a local application for exploring compiler optimizations on
Transformer computation graphs. It provides a React dashboard, a Python API,
and an optional Streamlit interface.

## Run the dashboard

Install Python 3.12 and Node.js 22.12 or newer. For Graphviz-rendered graph
figures, also install the Graphviz desktop package and put `dot` on PATH.

On Windows, run:

```bat
run-server.bat
```

The launcher creates `.venv312` if needed, installs missing Python and frontend
dependencies, and starts the API and dashboard. First-time setup needs internet
access. Open http://127.0.0.1:5173; the API listens on port 8000.

On macOS/Linux, run:

```bash
bash run.sh
```

This uses Python 3.12 by default and creates `.venv`. Set `PYTHON` to another
Python 3.12 executable if necessary.

For manual setup on Windows:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend ci
.\.venv312\Scripts\python.exe api_server.py
# In a second terminal:
npm --prefix frontend run dev -- --host 127.0.0.1
```

The optional interface runs with
`.\.venv312\Scripts\python.exe -m streamlit run app.py`.
Install `requirements-real.txt` for its optional Hugging Face model importer.
The training interface may download datasets on first use.

## Application features

- Transformer templates, operation demonstrations, and a trained tiny language model.
- Graph normalization, attention canonicalization, operator merging, scheduling,
  and graph/hypergraph/latency-aware partitioning.
- Numerical output checks and separate measured and modeled performance metrics.
- Learned plan selection, an RL pass policy, and neural-guided symbolic rewrites.
- Saved GPU/research results, architecture diagrams, stress sweeps, and ablations.
- CPU XLA comparison with randomized paired measurements, confidence intervals,
  raw timing samples, optimized HLO, and Excel export.

The main NumPy dashboard also uses 40 randomized paired rounds and median
latencies. It checks every declared output using
`abs(optimized - original) <= 1e-5 + 1e-4 * abs(original)`, reports MAE/RMSE,
and rejects nonfinite outputs. The displayed agreement percentage counts
elements within tolerance; it is not prediction accuracy on a labelled dataset.
Throughput uses the batch actually executed. Memory is unique live graph-value
storage in MiB, excluding kernel temporaries and allocator overhead.

Cost estimates are models, not measurements of a deployed multi-device system.
Saved research results describe their recorded hardware and are not rerun merely
by opening a dashboard tab. The shipped latency-GNN checkpoint uses historical
GPU training; the separate XLA study trains its own CPU models. The review run
did not establish that learned guidance outperforms the all-safe rewrite baseline.

## Files that must stay

| Path | Purpose |
|---|---|
| `api_server.py`, `pipeline.py`, `xla_compare.py` | API, compiler pipeline, and live XLA comparison |
| `frontend/` | Dashboard source, dependency lockfile, and displayed diagrams |
| `ir/`, `passes/`, `utils/` | Graph representation, compiler passes, execution, metrics, and export |
| `models/` | Model builders, importer, and tiny language-model weights |
| `search/` | Learned optimizers, rewrite rules, and trained checkpoints |
| `training/` | Training functionality used by the Streamlit interface |
| `docs/results/` | Runtime calibration and saved data displayed by dashboard tabs |
| `app.py` | Optional Streamlit interface |
| `requirements*.txt`, launch scripts | Installation and startup |

`docs/results/cost_calibration.json` affects cost estimates. The other result JSON
files populate dashboard tabs. These are application data, not disposable reports.
Similarly, `frontend/public/diagrams/` and model checkpoints are runtime assets.

`benchmarks/`, `tests/`, and the remaining `tools/` contain small reproducibility
and verification scripts. They are retained for publication verification; they
are not required to start the dashboard. Root-level Python demonstrations provide
additional command-line entry points.

## Verify or reproduce

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -t .
npm --prefix frontend run build
.\.venv312\Scripts\python.exe -m benchmarks.xla_guidance_study --out outputs/xla-study
.\.venv312\Scripts\python.exe tools/validate_xla_evidence.py outputs/xla-study
.\.venv312\Scripts\python.exe tools/inspect_xla_hlo.py outputs/xla-study
.\.venv312\Scripts\python.exe -m benchmarks.validate_outputs --out D:/GraphForge-validation
```

The XLA study requires a new output directory. It saves raw timings, held-out
models, source snapshots, and HLO evidence. Generated outputs are ignored by Git.

Excel exports recreate `outputs/benchmark_reports/` on demand. Python caches,
build output, virtual environments, and `node_modules` are local installation
artifacts and should not be included in a publication source archive. Install
dependencies from the manifests on the destination machine.
