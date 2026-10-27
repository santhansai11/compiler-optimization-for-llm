"""Verify before/after values across the UI registry and independent executors.

python -m benchmarks.validate_outputs --out D:/GraphForge-validation-20261027
The destination must be new. Timing and correctness evidence stays separate
from application sources; run this without other benchmark/training jobs.
"""
import argparse
from pathlib import Path

import numpy as np

from api_server import MODEL_REGISTRY
from models.architectures import create_parallel_block_graph
from pipeline import run_pipeline
from utils.benchmarking import environment_record, source_snapshot, suite_summary, write_json
from utils.executor import _make_feed, execute
from utils.metrics import compute_metrics
from utils.validation import compare_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    registry = dict(MODEL_REGISTRY)
    registry['unseen_parallel'] = ('Unseen parallel block',
        lambda b, s, d: create_parallel_block_graph(batch=b, seq=s, d_model=d), 'held-out architecture')
    import torch
    torch.set_num_threads(1)
    environment = environment_record(Path(__file__).resolve().parents[1])
    environment['source_sha256'] = source_snapshot(Path(__file__).resolve().parents[1], args.out)
    write_json(args.out / 'environment.json', environment)
    cases, checks, failures = [], [], []

    def record(case, backend, original, optimized, **extra):
        result = compare_outputs(original, optimized)
        row = {'case': case, 'backend': backend, **extra, **result}
        row['samples'] = [{'reference': np.asarray(a).reshape(-1)[:8].tolist(),
                           'optimized': np.asarray(b).reshape(-1)[:8].tolist()}
                          for a, b in zip(original.values(), optimized.values())]
        checks.append(row)
        if not result['preserved']:
            failures.append(row)

    for batch, seq, dim in [(1, 8, 16), (2, 16, 32)]:
        for name, (_, build, _) in registry.items():
            g = build(batch, seq, dim)
            optimized, infos, duration = run_pipeline(g)
            key = f'{name}-b{batch}-s{seq}-d{dim}'
            example = _make_feed(g, None)
            for trial in ['seed0', 'seed1', 'seed2', 'zeros', 'ones']:
                rng = np.random.default_rng(int(trial[-1]) if trial.startswith('seed') else 0)
                if trial == 'zeros':
                    feed = np.zeros_like(example)
                elif trial == 'ones':
                    feed = np.ones_like(example)
                elif g.meta.get('config', {}).get('llm'):
                    feed = rng.integers(0, g.meta['config']['vocab_size'], size=example.shape, dtype=np.int64)
                else:
                    feed = rng.standard_normal(example.shape).astype(np.float32)
                before = execute(g, repeats=1, warmup=0, feed=feed)['outputs']
                after = execute(optimized, repeats=1, warmup=0, feed=feed)['outputs']
                record(key, 'numpy', before, after, input_case=trial)
            for mode in ['graph', 'hypergraph', 'latency']:
                placed, _, _ = run_pipeline(g, partition=True, partition_mode=mode)
                record(key, 'numpy_partition',
                       execute(g, repeats=1, warmup=0)['outputs'],
                       execute(placed, repeats=1, warmup=0)['outputs'], partition_mode=mode)

            metrics = compute_metrics(g, optimized, infos, duration)
            cases.append({'case': key, 'requested_shape': [batch, seq, dim],
                          'actual_input_shape': list(example.shape),
                          'nodes_before': g.node_count(), 'nodes_after': optimized.node_count(),
                          'timing': metrics['timing'], 'metrics': metrics})
            print(f'{key}: {g.node_count()} -> {optimized.node_count()} nodes; '
                  f'{metrics["performance_conclusion"]}; outputs {metrics["accuracy"]["preserved"]}', flush=True)

    # Independent implementations: full compiler comparison supports these
    # six Transformer families. Tiny-LM framework tracing is covered by tests.
    import jax
    import jax.numpy as jnp
    from utils.jax_executor import build as build_jax
    from utils.torch_executor import _Runner
    for name in ['postln_2b', 'preln_2b', 'relu_2b', 'llama_2b', 'moe_2b', 'multihead_2b']:
        graph = registry[name][1](1, 8, 16)
        opt, _, _ = run_pipeline(graph)
        feed = _make_feed(graph, None)
        ref = execute(graph, repeats=1, warmup=0, feed=feed)['output']
        for label, candidate in [('original', graph), ('optimized', opt)]:
            with torch.inference_mode():
                outputs, _ = _Runner(candidate, torch.device('cpu'), feed, attention='composed').run()
            output_node = next(n for n, d in candidate.nodes(data=True) if d.get('is_output'))
            record(name, 'torch_cpu_' + label, {'output': ref}, {'output': outputs[output_node].numpy()})
            with jax.default_device(jax.devices('cpu')[0]):
                compiled = jax.jit(build_jax(candidate))
                out = np.asarray(compiled(jnp.asarray(feed)).block_until_ready())
            record(name, 'xla_cpu_' + label, {'output': ref}, {'output': out})

    summary = {'workloads': len(cases), 'numerical_checks': len(checks),
               'failures': len(failures), 'all_outputs_match': not failures,
               'before_after': suite_summary(cases, 'original', 'optimized'),
               'scope': 'NumPy reference timing; CPU Torch/XLA equivalence. No GPU timing or actual multi-device execution.'}
    write_json(args.out / 'workloads.json', cases)
    write_json(args.out / 'output_checks.json', checks)
    write_json(args.out / 'summary.json', summary)
    print(summary, flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} numerical checks failed; inspect output_checks.json')


if __name__ == '__main__':
    main()
