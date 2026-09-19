"""
determinism.py — shared determinism harness for LNN_LowLight experiments.

Usage in any run script (run_experiment_dense.py, run_cfc_mmrnn.py,
run_capacity_*.py, etc.):

    from determinism import set_determinism, capture_env_metadata

    set_determinism(seed)                 # per-seed, before creating any model
    ...
    meta = capture_env_metadata(device)   # once per run, attach to the result dict

Added 2026-08-28 as module 1 of the determinism-harness + cross-environment
protocol plan (see /areas/lnn-lowlight.md in mempalace for the full plan
and rationale).
"""

import os
import random
import subprocess
import socket
import platform

# CUBLAS_WORKSPACE_CONFIG must be set before any CUDA context is created,
# so this has to happen at import time, not inside a function called later.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def set_determinism(seed: int, warn_only: bool = True) -> None:
    """
    Make a single seed's run as numerically reproducible as practical.

    warn_only=True (default): operations without a deterministic
    implementation emit a warning instead of raising. Recommended: run
    once with warn_only=False to discover which ops in this codebase lack
    a deterministic implementation (candidates: scatter/index_add paths
    in the disocclusion/staleness handling) before relying on
    warn_only=True for production runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF32 is a no-op on Pascal (Kaggle P100) and on Apple MPS, but disabling
    # it now costs nothing and matters if the project ever runs on Ampere+.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def get_backend_name(device: "torch.device | None" = None) -> str:
    """
    Name of the compute backend actually in use for this run.

    If `device` is given (the torch.device this run/model is actually
    using — e.g. what run_experiment_dense.py's _select_device() or a
    force_all_cpu override produced), the name is read directly from it,
    which is always correct. Without a device, falls back to guessing
    what a fresh _select_device()-style call would pick right now.
    """
    if device is not None:
        return device.type  # "cpu" / "mps" / "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return "unknown"


def capture_env_metadata(device: "torch.device | None" = None) -> dict:
    """
    Metadata to attach to a run's result dict, so future sessions can tell
    which environment a given significant/not-significant verdict was
    actually produced on. Pass the actual torch.device the run used (not
    left to auto-detect) so this is accurate even when a run deliberately
    overrides the default backend (e.g. a forced-CPU cross-environment
    check run) — this is what the cross-environment robustness-check
    protocol (see /areas/lnn-lowlight.md) relies on.
    """
    return {
        "backend": get_backend_name(device),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(),
    }
