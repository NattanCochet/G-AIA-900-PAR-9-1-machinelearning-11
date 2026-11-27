"""T3-08: check that UVR's onnxruntime (1.13.1) loads a model and matches PyTorch.

Run from the repository root in UVR's environment, after
task3_uvr.train_small_nn has exported the model and its <model>_io.npz:

    uv run --project <UVR> --no-sync python -m task3_uvr.check_onnx_io \
        --model task3_uvr/artifacts/small_nn/SmallMaskNet_musdb7_remix.onnx

The .npz holds one input spectrogram and PyTorch's output on it. Only numpy,
onnxruntime and the standard library are imported (same rule as
check_onnxruntime_113, which is specific to HQ 3). Exit code 1 if the
outputs differ by more than PARITY_RTOL of the largest output value.
"""

import argparse
import sys
from pathlib import Path

import onnxruntime as ort

from task3_uvr.onnx_checks import compare_outputs, package_versions, uvr_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--io", type=Path, help="default: <model>_io.npz next to the model"
    )
    args = parser.parse_args(argv)
    import numpy as np

    io = np.load(args.io or args.model.with_name(args.model.stem + "_io.npz"))
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: io["input"]})[0]
    result = compare_outputs(io["output"], output)
    print("python", sys.version.split()[0], package_versions("onnxruntime", "numpy"))
    print("model", args.model.name, "hash", uvr_hash(args.model))
    print("input", session.get_inputs()[0].shape, "output", output.shape)
    print("onnxruntime vs PyTorch:", result)
    print("PASSED" if result["identical"] else "FAILED")
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
