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

    def test_every_pass_ordering_is_acyclic_and_equivalent(self):
        from itertools import permutations
        import networkx as nx
        from passes.canonicalize import canonicalize_attention
        from passes.merge import merge_operators
        from passes.normalize import normalize_graph
        fns = {"normalize": normalize_graph, "canonicalize": canonicalize_attention,
               "merge": merge_operators}
        import numpy as np
        from search import learned_optimizer as lo
        rng = np.random.default_rng(0)
        cases = [(n, b) for n, b in GRAPHS.items()] + [
            (f"generated-{fam}-{i}", (lambda f=fam: lo.generate_graph(f, rng)))
            for fam in ("postln", "moe", "llama", "multihead") for i in range(2)]
        for name, build in cases:
            graph = build()
            for r in (1, 2, 3):
                for seq in permutations(fns, r):
                    cur = graph
                    for step in seq:
                        cur = fns[step](cur)[0]
                        self.assertTrue(nx.is_directed_acyclic_graph(cur.graph), (name, seq))
                    m = compute_metrics(graph, cur, {}, 0.0)
                    self.assertTrue(m["accuracy"]["preserved"], (name, seq))

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
            self.assertGreaterEqual(got, max(true) * 0.88, name)
            if name.startswith("MoE"):
                self.assertGreater(got, true[fixed])  # non-regression; the aggregate claim lives in the LOFO results


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


class CodegenAndUnseenTests(unittest.TestCase):
    def test_generated_metal_kernels_match_torch(self):
        import torch
        import torch.nn.functional as F
        from utils import metal_codegen as mc
        dev = torch.device("mps")
        if not (torch.backends.mps.is_available() and mc.available(dev)):
            self.skipTest("no Apple GPU")
        x, y = torch.randn(3, 5, 48, device=dev), torch.randn(3, 5, 48, device=dev)
        w, b = torch.randn(48, device=dev), torch.randn(48, device=dev)
        torch.testing.assert_close(mc.fused_add_norm([x, y], w, b, 1e-5, rms=False),
                                   F.layer_norm(x + y, (48,), w, b, 1e-5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mc.fused_add_norm([x, y], w, None, 1e-6, rms=True),
                                   F.rms_norm(x + y, (48,), w, 1e-6), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(mc.fused_scale_softmax(x, 0.25), torch.softmax(x * 0.25, -1),
                                   atol=1e-6, rtol=1e-5)

    def test_unseen_parallel_block_architecture_compiles_equivalently(self):
        from models.architectures import UNSEEN_ARCHITECTURES
        build = UNSEEN_ARCHITECTURES["Parallel-block (GPT-J style)"]
        g = build(batch=2, seq=32, d_model=32)
        opt, infos, _ = run_pipeline(g, partition=False)
        self.assertLess(opt.node_count(), g.node_count())
        self.assertTrue(compute_metrics(g, opt, infos, 0.0)["accuracy"]["preserved"])
        self.assertEqual(infos["attention_canonicalization"]["canonicalized"], 2)


class SearchTests(unittest.TestCase):
    def test_rewrite_rules_preserve_outputs_including_square_shapes(self):
        """Random rewrite sequences keep outputs identical (seq == d_model too)."""
        import numpy as np
        import networkx as nx
        from search.rewrites import apply_rewrite, enumerate_rewrites
        from search import learned_optimizer as lo
        rng = np.random.default_rng(3)
        for name in ("Post-LN Transformer", "Multi-head Transformer (parallel heads)",
                     "LLaMA-style (RMSNorm + SwiGLU)", "MoE Transformer (parallel experts)"):
            for trial in range(4):
                g0 = ARCHITECTURES[name](batch=2, seq=32, d_model=32)  # square: ambiguous shapes
                if trial % 2:
                    g0 = lo._inject_redundancy(g0, rng)
                ref = compute_metrics(g0, g0, {}, 0.0)  # sanity: self-equivalent
                self.assertTrue(ref["accuracy"]["preserved"])
                cur = g0
                for _ in range(10):
                    cands = enumerate_rewrites(cur)
                    if not cands:
                        break
                    rule, match = cands[int(rng.integers(len(cands)))]
                    new = apply_rewrite(cur, rule, match)
                    if new is None:
                        continue
                    self.assertTrue(nx.is_directed_acyclic_graph(new.graph))
                    m = compute_metrics(g0, new, {}, 0.0)
                    self.assertTrue(m["accuracy"]["preserved"], (name, trial, rule.name))
                    cur = new

    def test_guided_search_with_table_cost_is_equivalent_and_smaller(self):
        from search import neuro_symbolic as nsm
        for name in ("Post-LN Transformer", "Pre-LN Transformer (GPT-style)",
                     "LLaMA-style (RMSNorm + SwiGLU)"):
            g = ARCHITECTURES[name](batch=2, seq=32, d_model=32)
            out, trace = nsm.guided_search(g, nsm.table_score)
            self.assertLess(out.node_count(), g.node_count())
            self.assertTrue(compute_metrics(g, out, {}, 0.0)["accuracy"]["preserved"], name)

    def test_rl_tabulation_and_policy_rollout(self):
        import numpy as np
        from search import learned_optimizer as lo
        from search import rl_policy as rl
        rng = np.random.default_rng(0)
        table = rl.tabulate(lo.generate_graph("moe", rng))
        self.assertEqual(len(table), 16)  # all orderings of <=3 passes
        self.assertGreater(table[rl.PASSES]["stay"], table[()]["stay"] - 1e-9)
        if rl.WEIGHTS.exists():
            policy = rl.RLPolicy.load()
            g = ARCHITECTURES["MoE Transformer (parallel experts)"]()
            final, actions = policy.plan(g)
            self.assertTrue(actions and actions[-1] in ("stop", "place2", "place3", "place4"))
            self.assertTrue(compute_metrics(g, final, {}, 0.0)["accuracy"]["preserved"])

    def test_cost_model_is_gpu_calibrated_when_file_present(self):
        from utils import cost_model
        if cost_model.CALIBRATION_PATH.exists():
            self.assertTrue(cost_model.CALIBRATED)
            self.assertEqual(cost_model.get_op_latency("Identity"), 0.0)
            self.assertGreater(cost_model.get_op_latency("FusedAttention"),
                               cost_model.get_op_latency("Softmax"))


if __name__ == "__main__":
    unittest.main()
