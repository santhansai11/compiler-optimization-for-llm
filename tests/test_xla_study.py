"""Tests for the XLA rewrite study helpers: rewrite policies, guided-search
decision counts, paired statistics, HLO fingerprints, environment record."""
import unittest

import numpy as np

from models.architectures import ARCHITECTURES
from search import neuro_symbolic as ns
from search.rewrites import apply_all, apply_random
from utils import environment, hlo, stats
from utils.metrics import compute_metrics


class PolicyTests(unittest.TestCase):
    def test_policies_preserve_outputs(self):
        for name, build in ARCHITECTURES.items():
            g = build()
            for out, _ in (apply_all(g), apply_all(g, exclude=("cse",)),
                           apply_random(g, 5, np.random.default_rng(0))):
                self.assertTrue(compute_metrics(g, out, {}, 0.0)["accuracy"]["preserved"], name)

    def test_fixed_rule_never_applies_cse(self):
        for build in ARCHITECTURES.values():
            self.assertNotIn("cse", apply_all(build(), exclude=("cse",))[1])

    def test_random_applies_requested_count(self):
        g = next(iter(ARCHITECTURES.values()))()
        total = len(apply_all(g)[1])
        self.assertEqual(len(apply_random(g, min(3, total), np.random.default_rng(1))[1]), min(3, total))

    def test_guided_counts_candidates_without_estimate(self):
        g = next(iter(ARCHITECTURES.values()))()
        calls = []

        def score(graphs):  # every other candidate has no estimate
            calls.append(len(graphs))
            return [float("nan") if i % 2 else ns.table_score([x])[0] for i, x in enumerate(graphs)]

        info = {}
        out, trace = ns.guided_search(g, score, stats=info)
        self.assertGreater(info["no_estimate"], 0)
        self.assertEqual(info["candidates_scored"], sum(calls[1:]))
        self.assertTrue(compute_metrics(g, out, {}, 0.0)["accuracy"]["preserved"])

    def test_min_speedup_one_keeps_old_behaviour(self):
        g = next(iter(ARCHITECTURES.values()))()
        a, ta = ns.guided_search(g, ns.table_score)
        b, tb = ns.guided_search(g, ns.table_score, min_speedup=1.0)
        self.assertEqual([t["rule"] for t in ta], [t["rule"] for t in tb])


class GraphKeyTests(unittest.TestCase):
    def test_key_depends_on_tensor_sizes(self):
        from benchmarks.gpu_benchmark import BUILDERS
        from benchmarks.xla_rewrite_study import graph_key
        build = BUILDERS["Post-LN Transformer"]
        self.assertEqual(graph_key(build(2, 32, 32)), graph_key(build(2, 32, 32)))
        self.assertNotEqual(graph_key(build(2, 32, 32)), graph_key(build(4, 64, 64)))


class StatsTests(unittest.TestCase):
    def test_paired_verdicts(self):
        rng = np.random.default_rng(0)
        base = 100 + rng.normal(0, 1, 30)
        self.assertEqual(stats.paired_speedup(base, base * 0.9)["verdict"], "faster")
        self.assertEqual(stats.paired_speedup(base, base * 1.1)["verdict"], "slower")
        self.assertEqual(stats.paired_speedup(base, base * (1 + rng.normal(0, 0.01, 30)))["verdict"],
                         "indistinguishable")

    def test_holm(self):
        self.assertEqual(stats.holm([0.001, 0.04, 0.5]), [True, False, False])
        self.assertEqual(stats.holm([0.01, 0.02]), [True, True])


class HloTests(unittest.TestCase):
    def test_fingerprint_ignores_names_not_program(self):
        import jax
        import jax.numpy as jnp
        x = jnp.ones((4, 8))

        def f(a):
            return jnp.maximum(a * 2.0, 0.0) + 1

        def g(b):
            t = b * 2.0
            return jnp.maximum(t, 0.0) + 1

        def h(a):
            return jnp.maximum(a * 3.0, 0.0) + 1

        text = {n: jax.jit(fn).lower(x).compile().as_text() for n, fn in (("f", f), ("g", g), ("h", h))}
        self.assertEqual(hlo.fingerprint(text["f"]), hlo.fingerprint(text["g"]))
        self.assertNotEqual(hlo.fingerprint(text["f"]), hlo.fingerprint(text["h"]))
        self.assertEqual(hlo.diff(text["f"], text["g"]), "")


class EnvironmentTests(unittest.TestCase):
    def test_collect(self):
        env = environment.collect()
        for key in ("cpu", "logical_cores", "os", "python", "packages", "thread_env", "git"):
            self.assertIn(key, env)
        self.assertIsNotNone(env["packages"]["numpy"])


if __name__ == "__main__":
    unittest.main()
