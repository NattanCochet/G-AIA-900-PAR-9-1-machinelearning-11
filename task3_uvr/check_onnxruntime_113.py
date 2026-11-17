"""T3-05, step 5b: check that UVR's onnxruntime loads the re-exported HQ 3.

Run from the repository root in UVR's environment (onnxruntime 1.13.1),
after `task3_uvr.hq3_trainable check`:

    uv run --project <UVR> --no-sync python -m task3_uvr.check_onnxruntime_113 \
        --model <UVR>/models/MDX_Net_Models/UVR-MDX-NET-Inst_HQ_3.onnx

It runs the re-exported model on the saved input and compares the output with
PyTorch's. It also runs the original model and compares it with the newer
onnxruntime's output, to show both versions agree on the reference.
Only numpy, onnxruntime and the standard library are imported here.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from task3_uvr.onnx_checks import (
    RESULTS_PATH,
    compare_outputs,
    machine_name,
    package_versions,
    update_results,
    uvr_hash,
)

DEFAULT_EXPORT = Path("task3_uvr") / "artifacts" / "HQ3_T3-05_step1.onnx"
DEFAULT_IO = Path("task3_uvr") / "artifacts" / "HQ3_T3-05_step1_io.npz"


def run(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="original UVR-MDX-NET-Inst_HQ_3.onnx")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--io", type=Path, default=DEFAULT_IO)
    args = parser.parse_args()

    io = np.load(args.io)
    exported = compare_outputs(io["torch_output_after_step"], run(args.export, io["input"]))
    original = compare_outputs(io["ort_output_original"], run(args.model, io["input"]))
    summary = {
        "machine": machine_name(),
        "versions": {
            "python": sys.version.split()[0],
            **package_versions("onnxruntime", "numpy"),
        },
        "file": args.export.name,
        "uvr_hash": uvr_hash(args.export),
        "loads": True,
        "parity_export_vs_pytorch": exported,
        "parity_original_vs_newer_onnxruntime": original,
        "passed": exported["identical"] and original["identical"],
    }
    update_results(RESULTS_PATH, "onnxruntime_1_13_check", summary)
    print("onnxruntime", ort.__version__)
    print("re-exported vs PyTorch:", exported)
    print("original vs newer onnxruntime:", original)
    print("PASSED" if summary["passed"] else "FAILED", "->", RESULTS_PATH)
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
