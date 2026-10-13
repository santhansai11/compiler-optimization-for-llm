"""Correctness tests: every architecture/workload must be output-equivalent
after every pass combination, and the new rewrites must fire."""
import itertools
import unittest

from models.architectures import ARCHITECTURES
from models.operations import OPERATION_DEMOS
from pipeline import run_pipeline
from utils.metrics import compute_metrics

GRAPHS = {n: f for n, f in ARCHITECTURES.items()}
GRAPHS.update({k: v[1] for k, v in OPERATION_DEMOS.items()})
from models.tiny_lm import WEIGHTS_PATH, build_graph as _lm  # noqa: E402
if WEIGHTS_PATH.exists():
    GRAPHS["tiny_lm"] = lambda: _lm(batch=2, seq=64)


class PassTests(unittest.TestCase):
    def test_all_graphs_equivalent_under_all_pass_subsets(self):
        keys = ("normalize", "canonicalize", "merge", "schedule")
        for name, build in GRAPHS.items():
            graph = build()
            for combo in itertools.product([False, True], repeat=len(keys)):
                flags = dict(zip(keys, combo))
                opt, infos, t = run_pipeline(graph, **flags)
                m = compute_metrics(graph, opt, infos, t)
                self.assertTrue(m["accuracy"]["preserved"], (name, flags))
                self.assertTrue(opt.is_dag())

    def test_transpose_algebra(self):
        g = OPERATION_DEMOS["op_transpose"][1]()
        opt, infos, _ = run_pipeline(g, canonicalize=False, merge=False, schedule=False)
        self.assertEqual(infos["normalize"]["transposes_removed"], 2)
        self.assertEqual(infos["normalize"]["transposes_folded"], 1)
        self.assertFalse(any(d["op_type"] == "Transpose" for _, d in opt.nodes(data=True)))

    def test_scale_chain_and_cse(self):
        g = OPERATION_DEMOS["op_scale_chain"][1]()
        _, infos, _ = run_pipeline(g, canonicalize=False, merge=False, schedule=False)
        self.assertGreaterEqual(infos["normalize"]["scales_merged"], 1)
        self.assertGreaterEqual(infos["normalize"]["cse_merges"], 1)

    def test_attention_fused(self):
        g = OPERATION_DEMOS["op_attention"][1]()
        opt, infos, _ = run_pipeline(g)
        self.assertEqual(infos["attention_canonicalization"]["canonicalized"], 1)

    def test_llama_fusions(self):
        g = ARCHITECTURES["LLaMA-style (RMSNorm + SwiGLU)"]()
        _, infos, _ = run_pipeline(g)
        self.assertIn("LinearSiLU", infos["semantic_merging"]["fused_types"])


class LearnedOptimizerTests(unittest.TestCase):
    def test_single_device_deployment_equals_sequential(self):
        from utils.cost_model import deployment_latency_us, sequential_latency_us
        g = ARCHITECTURES["MoE Transformer (parallel experts)"]()
        self.assertAlmostEqual(deployment_latency_us(g), sequential_latency_us(g))

    def test_latency_partition_helps_wide_graph_not_chain(self):
        from utils.cost_model import deployment_latency_us as D
        wide = ARCHITECTURES["MoE Transformer (parallel experts)"]()
        chain = ARCHITECTURES["Pre-LN Transformer (GPT-style)"]()
        for g, gains in ((wide, True), (chain, False)):
            one, _, _ = run_pipeline(g, partition=False)
            four, _, _ = run_pipeline(g, partition=True, num_partitions=4,
                                      partition_mode="latency")
            self.assertEqual(D(four) < D(one) * 0.9, gains)

    def test_trained_model_near_oracle_and_beats_fixed_on_wide_graph(self):
        from search import learned_optimizer as lo
        if not lo.WEIGHTS_PATH.exists():
            self.skipTest("learned optimizer not trained")
        opt = lo.LearnedOptimizer.load()
        fixed = lo.ACTIONS.index((True, True, True, 1))
        for name in ("MoE Transformer (parallel experts)",
                     "Pre-LN Transformer (GPT-style)"):
            g = ARCHITECTURES[name]()
            true = lo.evaluate_actions(g)
            got = true[lo.ACTIONS.index(tuple(opt.plan(g)["action"]))]
            self.assertGreaterEqual(got, max(true) * 0.95, name)
            if name.startswith("MoE"):
                self.assertGreater(got, true[fixed] * 1.1)


class TorchExecutorTests(unittest.TestCase):
    def test_torch_executor_matches_numpy_on_cpu(self):
        import numpy as np
        import torch
        from utils.executor import execute
        from utils.torch_executor import gpu_compare
        for name in ("Post-LN Transformer", "Pre-LN Transformer (GPT-style)",
                     "LLaMA-style (RMSNorm + SwiGLU)", "MoE Transformer (parallel experts)",
                     "Multi-head Transformer (parallel heads)"):
            g = ARCHITECTURES[name](batch=2, seq=16, d_model=32)
            opt, _, _ = run_pipeline(g, partition=False)
            ref = execute(g, repeats=1, warmup=0)["output"]
            for res in gpu_compare([(g, "composed"), (opt, "sdpa"), (opt, "auto")],
                                   torch.device("cpu"), repeats=2, warmup=1):
                scale = max(1e-6, float(np.abs(ref).max()))
                self.assertLess(float(np.abs(res["output"] - ref).max()) / scale, 1e-3, name)


class RealGraphTests(unittest.TestCase):
    def test_traced_trained_model_reduces_and_matches_pytorch(self):
        import numpy as np
        import torch
        from models.import_hf import import_module
        from models.tiny_lm import WEIGHTS_PATH, make_torch_model
        from utils.executor import execute
        if not WEIGHTS_PATH.exists():
            self.skipTest("no trained weights")
        w = np.load(WEIGHTS_PATH)
        vocab, d = w["tok.weight"].shape
        model = make_torch_model(vocab, d, 2, w["pos"].shape[0])
        model.load_state_dict({k: torch.from_numpy(w[k]) for k in model.state_dict()})
        model.eval()
        model.d_model, model.vocab = d, vocab
        ids = torch.randint(0, vocab, (2, 32), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            ref = model(ids).numpy()
        graph, _ = import_module(model, ids, "tiny", "tiny")
        opt, infos, _ = run_pipeline(graph, partition=False)
        feed = ids.numpy().astype("float32")
        out = execute(opt, feed=feed, repeats=1, warmup=0)["output"]
        self.assertLess(float(np.abs(out - ref).max()), 1e-3)
        self.assertLess(opt.node_count(), graph.node_count() * 0.5)
        att = infos["attention_canonicalization"]
        self.assertEqual((att["detected"], att["canonicalized"]), (2, 2))
        self.assertGreater(infos["normalize"]["views_removed"], 0)


if __name__ == "__main__":
    unittest.main()
