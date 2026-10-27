"""Paired-timing statistics: bootstrap CIs, Wilcoxon signed-rank, Holm.

All comparisons are PAIRED: sample ``i`` of variant A and sample ``i`` of
variant B were taken in the same timing round, so per-round ratios cancel
slow drift of the machine.
"""

import numpy as np
from scipy import stats as _sp


def bootstrap_ci(values, statistic=np.median, reps=2000, level=0.95, seed=0):
    """Percentile bootstrap CI of ``statistic(values)``."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    boot = statistic(values[idx], axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def wilcoxon_p(log_ratios):
    """Two-sided Wilcoxon signed-rank p-value of ``log_ratios`` vs 0."""
    x = np.asarray(log_ratios, dtype=float)
    if len(x) < 2 or np.allclose(x, 0.0):
        return 1.0
    return float(_sp.wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)


def paired_speedup(base_us, variant_us, seed=0):
    """Speedup of ``variant`` over ``base`` from paired per-round samples.

    Returns the median per-round speedup (base / variant), its 95% bootstrap
    CI, the Wilcoxon p-value and a verdict: ``faster`` / ``slower`` when the
    CI excludes 1, otherwise ``indistinguishable``.
    """
    base_us, variant_us = np.asarray(base_us, float), np.asarray(variant_us, float)
    ratio = base_us / variant_us
    lo, hi = bootstrap_ci(ratio, seed=seed)
    verdict = "faster" if lo > 1.0 else "slower" if hi < 1.0 else "indistinguishable"
    return {"median_speedup": float(np.median(ratio)), "ci95": [lo, hi],
            "wilcoxon_p": wilcoxon_p(np.log(ratio)), "verdict": verdict,
            "rounds": int(len(ratio))}


def holm(pvalues, alpha=0.05):
    """Holm-Bonferroni: list of booleans, True where the test is rejected."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    reject = np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(order):
        if p[i] > alpha / (len(p) - rank):
            break
        reject[i] = True
    return reject.tolist()
