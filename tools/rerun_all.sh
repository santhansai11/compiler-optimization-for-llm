#!/usr/bin/env bash
# Regenerate every result file, strictly one job at a time (timings must not overlap).
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
export PYTHONPATH=.
rm -f training/artifacts/ns_dataset.pkl training/artifacts/learned_dataset.pkl training/artifacts/rl_tables.pkl
step() { echo "=== $1 $(date +%H:%M:%S)"; shift; "$@"; }
step real-graphs      $PY benchmarks/real_graph_study.py
step calibrate        $PY benchmarks/calibrate_cost_model.py
step gpu-benchmark    $PY benchmarks/gpu_benchmark.py
step neuro-symbolic   $PY benchmarks/neuro_symbolic_study.py
step compilers        $PY benchmarks/compiler_comparison.py
step lm-eval          $PY benchmarks/lm_full_eval.py
step plan-selector    $PY -m search.learned_optimizer train
step rl-policy        $PY -m search.rl_policy train
step unseen-arch      $PY benchmarks/unseen_architecture_test.py
step sensitivity      $PY benchmarks/learned_sensitivity.py
step figures          $PY tools/make_diagrams.py
echo "=== done $(date +%H:%M:%S)"
