"""Reproducible, paired timing and explicit-unit benchmark records.

Rounds (not calls pooled across workloads) are the resampling unit. Confidence
intervals describe within-process timing variability, not hardware generality.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def paired_summary(reference_us, candidate_us, seed=0, bootstrap=4000):
    """Positive paired differences mean the candidate is faster.

    A slowdown count based on medians is separate from a confidence interval
    that overlaps zero. An inconclusive interval never gets called a tie.
    """
    a, b = np.asarray(reference_us, dtype=float), np.asarray(candidate_us, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Paired timing vectors must be nonempty, 1-D and equal length")
    if not (np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError("Timings must be finite positive microseconds")
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    delta = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(bootstrap, len(a)))
    ci = np.quantile(np.median(delta[indices], axis=1), [0.025, 0.975])
    ratio_ci = np.quantile(np.median((a / b)[indices], axis=1), [0.025, 0.975])
    # All-zero differences need explicit handling in SciPy's signed-rank test.
    test = None if np.all(delta == 0) else wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="auto")
    return {
        "pairs": len(a), "reference_median_us": float(np.median(a)),
        "candidate_median_us": float(np.median(b)),
        "speedup_ratio_of_medians": float(np.median(a) / np.median(b)),
        "median_paired_speedup": float(np.median(a / b)),
        "paired_speedup_ci95": ratio_ci.tolist(),
        "median_paired_difference_us": float(np.median(delta)),
        "paired_difference_ci95_us": ci.tolist(),
        "observed_slowdown": bool(np.median(b) > np.median(a)),
        "ci_direction": "faster" if ci[0] > 0 else "slower" if ci[1] < 0 else "inconclusive",
        "wilcoxon": {"statistic": float(test.statistic) if test else 0.0,
                     "pvalue_two_sided": float(test.pvalue) if test else 1.0,
                     "zero_method": "wilcox", "multiplicity_adjusted": False},
        "bootstrap_resamples": bootstrap, "bootstrap_seed": seed,
    }


def randomized_benchmark(calls, rounds=40, warmup=5, seed=0):
    """Warm all variants, then time each once per independently shuffled round.

    Callbacks must complete their device work before returning, or return an
    array exposing block_until_ready (JAX). CUDA/MPS callbacks must synchronise.
    Raw samples align by round regardless of the execution order in that round.
    """
    if rounds < 30 or warmup < 1 or not calls:
        raise ValueError("Use at least 30 rounds, one warmup and one callable")
    names = list(calls)
    outputs = {}

    def run(name):
        out = calls[name]()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        return out

    for _ in range(warmup):
        for name in names:
            outputs[name] = run(name)
    samples = {name: [] for name in names}
    orders = []
    rng = np.random.default_rng(seed)
    for _ in range(rounds):
        order = [names[int(i)] for i in rng.permutation(len(names))]
        orders.append(order)
        for name in order:
            started = time.perf_counter_ns()
            outputs[name] = run(name)
            samples[name].append((time.perf_counter_ns() - started) / 1000.0)
    return {
        "unit": "us", "estimator": "median", "rounds": rounds,
        "warmup": warmup, "seed": seed, "round_order": orders,
        "samples_us": samples,
        "median_us": {k: float(np.median(v)) for k, v in samples.items()},
        "pairing": "same randomized round; within one process",
    }, outputs


def suite_summary(rows, reference, candidate, seed=0):
    """One pair of medians per workload, avoiding pooled-trial pseudoreplication."""
    a = [r["timing"]["median_us"][reference] for r in rows]
    b = [r["timing"]["median_us"][candidate] for r in rows]
    result = paired_summary(a, b, seed=seed)
    result["pairing"] = "one median per workload; workloads equally weighted"
    result["workloads"] = len(rows)
    result["observed_slowdowns"] = sum(y > x for x, y in zip(a, b))
    result["observed_improvements"] = sum(y < x for x, y in zip(a, b))
    result["exact_median_ties"] = sum(y == x for x, y in zip(a, b))
    result["geomean_speedup"] = float(np.exp(np.mean(np.log(np.asarray(a) / b))))
    return result


def environment_record(root=None):
    """Record the actual runtime; never infer missing hardware details."""
    cpu = platform.processor() or None
    physical = None
    affinity = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        affinity = psutil.Process().cpu_affinity()
    except (ImportError, AttributeError, OSError):
        pass
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        if physical is None:
            try:
                physical = int(subprocess.check_output([
                    "powershell", "-NoProfile", "-NonInteractive", "-Command",
                    "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"
                ], text=True, timeout=10).strip())
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if affinity is None:
            try:
                import ctypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.GetCurrentProcess.restype = ctypes.c_void_p
                kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
                process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
                if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
                    affinity = [i for i in range(process_mask.value.bit_length()) if process_mask.value & (1 << i)]
            except (OSError, AttributeError):
                pass
    elif affinity is None and hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    versions = {}
    for package in ("numpy", "scipy", "torch", "jax", "jaxlib", "networkx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "os": platform.platform(), "machine": platform.machine(),
        "cpu_model": cpu, "logical_cores": os.cpu_count(),
        "physical_cores": physical, "process_cpu_affinity": affinity,
        "versions": versions,
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS")},
        "jax_thread_count": None,
        "jax_thread_note": "No stable runtime query; flags are recorded, null is unknown, not single-threaded.",
    }
    if "torch" in sys.modules:
        import torch
        result["torch_threads"] = torch.get_num_threads()
        result["torch_interop_threads"] = torch.get_num_interop_threads()
    if "jax" in sys.modules:
        import jax
        result["jax_devices"] = [str(d) for d in jax.devices()]
        result["jax_default_backend"] = jax.default_backend()
    if root:
        try:
            result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
            result["git_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            result["git_commit"] = None
    return result


def source_snapshot(root, out):
    """Save exact tracked experiment sources, including current dirty edits."""
    import zipfile
    root, out = Path(root), Path(out)
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    # New implementation files are not necessarily tracked yet.
    for directory in ("benchmarks", "utils", "search", "tests"):
        names.extend(str(p.relative_to(root)).replace("\\", "/") for p in (root / directory).glob("*.py"))
    hashes = {}
    with zipfile.ZipFile(out / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            path = root / name
            if path.is_file() and (path.suffix in {".py", ".md", ".txt", ".sh", ".bat", ".jsx", ".js"}
                                   or name == "docs/results/cost_calibration.json"):
                data = path.read_bytes()
                archive.writestr(name, data)
                hashes[name] = hashlib.sha256(data).hexdigest()
    write_json(out / "source_manifest.json", hashes)
    return hashlib.sha256((out / "sources.zip").read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
