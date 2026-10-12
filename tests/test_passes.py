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


if __name__ == "__main__":
    unittest.main()
