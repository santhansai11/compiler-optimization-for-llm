"""Record the exact environment a result was produced in.

Every benchmark stores ``collect()`` in its result JSON, so each number can
be traced to one machine, OS, thread setting, library set and git commit.
"""

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("jax", "jaxlib", "torch", "numpy", "scipy", "networkx")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS",
               "PYTHONHASHSEED")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                              cwd=ROOT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_model():
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def physical_cores():
    if platform.system() == "Darwin":
        value = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(value) if value.isdigit() else None
    try:
        ids = {line for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith(("physical id", "core id"))}
        return len(ids) // 2 or None
    except OSError:
        return None


def git_state():
    commit = _run(["git", "rev-parse", "HEAD"])
    return {"commit": commit or None,
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or None,
            "dirty": bool(_run(["git", "status", "--porcelain"])) if commit else None,
            "tag": _run(["git", "describe", "--tags", "--exact-match"]) or None}


def collect():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import torch
        torch_threads = torch.get_num_threads()
    except ImportError:
        torch_threads = None
    try:
        import jax
        jax_devices = [str(d) for d in jax.devices()]
    except ImportError:
        jax_devices = None
    return {
        "cpu": cpu_model(),
        "machine": platform.machine(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.mac_ver()[0] or platform.version(),
        "python": platform.python_version(),
        "packages": versions,
        "torch_threads": torch_threads,
        "jax_devices": jax_devices,
        "thread_env": {k: os.environ.get(k) for k in THREAD_VARS},
        "git": git_state(),
    }
