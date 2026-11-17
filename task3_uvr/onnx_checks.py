"""Helpers shared by this repository's environment and UVR's environment.

Only the standard library and numpy are imported here, so that
check_onnxruntime_113 can run in UVR's environment (numpy 1.23,
onnxruntime 1.13.1) without importing torch.
"""

import hashlib
import json
import socket
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

# Conversion and re-export count as "identical" when the max absolute
# difference is below this fraction of the largest absolute reference output.
PARITY_RTOL = 1e-4

# onnxruntime 1.13.1 (UVR's environment) supports at most these versions.
MAX_OPSET = 17
MAX_IR_VERSION = 8

RESULTS_PATH = Path("results") / "task3_hq3_trainable.json"


def uvr_hash(path: Path) -> str:
    """MD5 of the file's last 10,000 * 1024 bytes, or of the whole file if it
    is smaller. This is how UVR identifies a model (UVR.py, get_model_hash)."""
    with open(path, "rb") as f:
        try:
            f.seek(-10000 * 1024, 2)
        except OSError:
            f.seek(0)
        return hashlib.md5(f.read()).hexdigest()


def full_md5(path: Path) -> str:
    """MD5 of the whole file, used to prove a file was not modified."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_outputs(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Max absolute difference, absolute and relative to the largest absolute
    reference value, and whether it is below PARITY_RTOL."""
    max_abs_diff = float(np.max(np.abs(reference - candidate)))
    max_abs_reference = float(np.max(np.abs(reference)))
    relative = max_abs_diff / max_abs_reference if max_abs_reference else max_abs_diff
    return {
        "max_abs_diff": max_abs_diff,
        "max_abs_reference": max_abs_reference,
        "relative_diff": relative,
        "threshold": PARITY_RTOL,
        "identical": relative < PARITY_RTOL,
    }


def package_versions(*names: str) -> dict:
    """Installed version of each package, or None if it is not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def machine_name() -> str:
    return socket.gethostname()


def update_results(path: Path, key: str, value: dict) -> None:
    """Set one top-level key of the JSON results file, keeping the others."""
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text()) if path.exists() else {}
    results[key] = value
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    tmp.replace(path)
