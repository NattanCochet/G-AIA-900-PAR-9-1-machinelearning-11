"""T3-05: check that UVR-MDX-NET-Inst_HQ_3 can be fine-tuned in PyTorch.

Two subcommands, both run from the repository root in this repository's
environment:

    uv run python -m task3_uvr.hq3_trainable check \
        --model <UVR>/models/MDX_Net_Models/UVR-MDX-NET-Inst_HQ_3.onnx

        1. inspects the ONNX graph,
        2. converts it to PyTorch with onnx2pytorch,
        3. compares PyTorch with onnxruntime on a random input,
        4. runs one backward pass and one optimizer step,
        5. re-exports the stepped model to ONNX (opset 17, for UVR's
           onnxruntime 1.13.1) and compares it with PyTorch,
        6. looks up the model's UVR settings from its hash.
    It writes the re-exported model and its input/output to --out-dir, and a
    summary to results/task3_hq3_trainable.json. The original model is only
    read; its MD5 is checked before and after.

    uv run python -m task3_uvr.hq3_trainable install --model <same path>

        copies the re-exported model next to the original under a new name
        and writes its settings JSON, so UVR lists it under MDX-Net. It
        refuses to run unless check_onnxruntime_113 passed on that file.

Between the two, run task3_uvr.check_onnxruntime_113 in UVR's environment.
"""

import argparse
import copy
import json
import resource
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnx2pytorch import ConvertModel

from common import set_seed
from common.timer import PhaseTimer
from task3_uvr.onnx_checks import (
    MAX_IR_VERSION,
    MAX_OPSET,
    RESULTS_PATH,
    compare_outputs,
    full_md5,
    machine_name,
    package_versions,
    update_results,
    uvr_hash,
)

HQ3_HASH = "55657dd70583b0fedfba5f67df11d711"
# Settings of HQ 3 in UVR's model_data.json. dim_t is stored as a power of 2
# (2**8 = 256 frames); the input is [batch, 4, dim_f, 2**dim_t].
HQ3_SETTINGS = {
    "compensate": 1.022,
    "mdx_dim_f_set": 3072,
    "mdx_dim_t_set": 8,
    "mdx_n_fft_scale_set": 6144,
    "primary_stem": "Instrumental",
}
INPUT_SHAPE = (1, 4, HQ3_SETTINGS["mdx_dim_f_set"], 2 ** HQ3_SETTINGS["mdx_dim_t_set"])

EXPORT_NAME = "HQ3_T3-05_step1.onnx"
IO_NAME = "HQ3_T3-05_step1_io.npz"
DEFAULT_OUT_DIR = Path("task3_uvr") / "artifacts"


# --- Step 1: inspect -------------------------------------------------------


def batchnorm_statistics(model: onnx.ModelProto) -> set[str]:
    """Names of the BatchNormalization running mean and variance initializers.
    They are statistics, not weights: they never receive a gradient."""
    return {
        name
        for node in model.graph.node
        if node.op_type == "BatchNormalization"
        for name in node.input[3:5]
    }


def _shape(value_info) -> list:
    return [d.dim_value or d.dim_param for d in value_info.type.tensor_type.shape.dim]


def inspect_onnx(model: onnx.ModelProto) -> dict:
    initializers = model.graph.initializer
    statistics = batchnorm_statistics(model)
    n_values = {t.name: int(np.prod(t.dims)) for t in initializers}
    return {
        "ir_version": model.ir_version,
        "opsets": {o.domain or "ai.onnx": o.version for o in model.opset_import},
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "inputs": {i.name: _shape(i) for i in model.graph.input},
        "outputs": {o.name: _shape(o) for o in model.graph.output},
        "operators": dict(Counter(n.op_type for n in model.graph.node).most_common()),
        "n_initializers": len(initializers),
        "n_values": sum(n_values.values()),
        "n_batchnorm_statistics_values": sum(n_values[n] for n in statistics),
        "n_trainable_values_expected": sum(
            v for n, v in n_values.items() if n not in statistics
        ),
    }


# --- Step 2: convert -------------------------------------------------------


def convert(model: onnx.ModelProto) -> torch.nn.Module:
    """onnx2pytorch conversion. Any error propagates unchanged."""
    return ConvertModel(model)


def parameter_summary(module: torch.nn.Module) -> dict:
    return {
        "n_parameter_tensors": sum(1 for _ in module.parameters()),
        "n_parameter_values": sum(p.numel() for p in module.parameters()),
        "n_buffer_values": sum(b.numel() for b in module.buffers()),
    }


# --- Step 3: compare with onnxruntime ---------------------------------------


def run_onnxruntime(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return session.run(None, {name: x})[0]


@torch.no_grad()
def forward_eval(module: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    module.eval()
    return module(torch.from_numpy(x)).numpy()


@torch.no_grad()
def train_mode_effect(module: torch.nn.Module, x: np.ndarray, eval_output: np.ndarray) -> dict:
    """In train mode, BatchNorm normalizes with the current batch's statistics
    and updates its running averages, so the output changes. Measured on a
    copy, so the original module is untouched."""
    module_copy = copy.deepcopy(module).train()
    train_output = module_copy(torch.from_numpy(x)).numpy()
    return compare_outputs(eval_output, train_output)


# --- Step 4: one training step ----------------------------------------------


def _layer(parameter_name: str) -> str:
    return parameter_name.rsplit(".", 1)[0]


def gradient_step(
    module: torch.nn.Module, x: torch.Tensor, target: torch.Tensor, lr: float
) -> dict:
    """One backward pass and one Adam step, with BatchNorm frozen (eval mode):
    the running statistics stay those of HQ 3, as when fine-tuning with small
    batches. Reports the parameters whose gradient is missing or all zeros,
    grouped by layer, instead of failing."""
    module.eval()
    parameters = dict(module.named_parameters())
    before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)

    loss = torch.nn.functional.l1_loss(module(x), target)
    loss.backward()

    no_gradient = [n for n, p in parameters.items() if p.grad is None]
    zero_gradient = [
        n for n, p in parameters.items() if p.grad is not None and not p.grad.any()
    ]
    optimizer.step()

    changed = [n for n, p in parameters.items() if not torch.equal(before[n], p.detach())]
    max_change = max(
        (float((p.detach() - before[n]).abs().max()) for n, p in parameters.items()),
        default=0.0,
    )
    without_gradient = no_gradient + zero_gradient
    return {
        "loss": "l1 between output and a random target",
        "loss_value": loss.item(),
        "optimizer": "adam",
        "lr": lr,
        "n_parameter_tensors": len(parameters),
        "n_without_gradient": len(no_gradient),
        "n_zero_gradient": len(zero_gradient),
        "layers_without_gradient": sorted({_layer(n) for n in without_gradient}),
        "parameters_without_gradient": sorted(without_gradient),
        "n_values_without_gradient": sum(parameters[n].numel() for n in without_gradient),
        "n_tensors_changed_by_step": len(changed),
        "max_abs_weight_change": max_change,
    }


# --- Step 5: re-export --------------------------------------------------------


def export_onnx(module: torch.nn.Module, x: torch.Tensor, path: Path) -> None:
    """Legacy TorchScript exporter at opset 17: onnxruntime 1.13.1 supports at
    most opset 17 and IR version 8."""
    module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        (x,),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=MAX_OPSET,
        dynamo=False,
    )


def count_changed_initializers(original: onnx.ModelProto, exported: onnx.ModelProto) -> int:
    """Number of exported initializers that are not byte-identical to any
    original one. After a step, it must equal the number of trainable tensors,
    which proves the export holds the updated weights and not the
    `_initializer_*` buffers onnx2pytorch keeps from the original file."""
    original_bytes = {numpy_helper.to_array(t).tobytes() for t in original.graph.initializer}
    return sum(
        numpy_helper.to_array(t).tobytes() not in original_bytes
        for t in exported.graph.initializer
    )


def describe_export(path: Path, original: onnx.ModelProto) -> dict:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    return {
        "n_initializers_changed": count_changed_initializers(original, model),
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "uvr_hash": uvr_hash(path),
        "ir_version": model.ir_version,
        "opset": opset,
        "fits_onnxruntime_1_13": model.ir_version <= MAX_IR_VERSION and opset <= MAX_OPSET,
        "operators": dict(Counter(n.op_type for n in model.graph.node).most_common()),
        "n_values": sum(int(np.prod(t.dims)) for t in model.graph.initializer),
    }


# --- Step 6: UVR settings -----------------------------------------------------


def lookup_settings(model_path: Path) -> dict:
    """Settings UVR uses for this model: model_data/<hash>.json if present,
    otherwise the hash's entry in model_data/model_data.json."""
    model_hash = uvr_hash(model_path)
    model_data = model_path.parent / "model_data"
    own = model_data / f"{model_hash}.json"
    if own.exists():
        return {"uvr_hash": model_hash, "source": own.name, "settings": json.loads(own.read_text())}
    shared = json.loads((model_data / "model_data.json").read_text(encoding="utf-8"))
    return {
        "uvr_hash": model_hash,
        "source": "model_data.json",
        "settings": shared.get(model_hash),
    }


# --- Subcommands ----------------------------------------------------------------


def run_check(model_path: Path, out_dir: Path, seed: int, lr: float) -> bool:
    timer = PhaseTimer()
    original_md5 = full_md5(model_path)
    summary = {
        "machine": machine_name(),
        "versions": {
            "python": sys.version.split()[0],
            **package_versions("torch", "onnx", "onnx2pytorch", "onnxruntime", "numpy"),
        },
        "torch_threads": torch.get_num_threads(),
        "seed": seed,
        "model": {"file": model_path.name, "md5": original_md5},
        "input_shape": list(INPUT_SHAPE),
        "input": "standard normal noise, not a real spectrogram",
    }

    with timer.phase("inspect"):
        onnx_model = onnx.load(str(model_path))
        summary["inspect"] = inspect_onnx(onnx_model)
    print("1. inspect:", json.dumps(summary["inspect"], indent=2))

    with timer.phase("settings"):
        summary["settings"] = lookup_settings(model_path)
    summary["settings"]["matches_expected"] = (
        summary["settings"]["uvr_hash"] == HQ3_HASH
        and summary["settings"]["settings"] == HQ3_SETTINGS
    )
    print("6. settings:", summary["settings"])

    with timer.phase("convert"):
        module = convert(onnx_model)
    summary["converted"] = parameter_summary(module)
    print("2. converted:", summary["converted"])

    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(INPUT_SHAPE, generator=generator)
    target = torch.randn(INPUT_SHAPE, generator=generator)
    x_np = x.numpy()

    with timer.phase("parity_conversion"):
        ort_original = run_onnxruntime(model_path, x_np)
        torch_original = forward_eval(module, x_np)
        summary["parity_conversion"] = compare_outputs(ort_original, torch_original)
    print("3. parity (onnxruntime vs PyTorch):", summary["parity_conversion"])

    with timer.phase("train_mode_effect"):
        summary["train_mode_effect"] = train_mode_effect(module, x_np, torch_original)
    print("   train mode vs eval mode:", summary["train_mode_effect"])

    with timer.phase("gradient_step"):
        summary["gradient_step"] = gradient_step(module, x, target, lr)
    print("4. gradient step:", summary["gradient_step"])

    export_path = out_dir / EXPORT_NAME
    with timer.phase("export"):
        export_onnx(module, x, export_path)
        summary["export"] = describe_export(export_path, onnx_model)
    print("5. export:", summary["export"])

    with timer.phase("parity_export"):
        torch_stepped = forward_eval(module, x_np)
        summary["parity_export"] = compare_outputs(
            torch_stepped, run_onnxruntime(export_path, x_np)
        )
        summary["step_changed_output"] = compare_outputs(torch_original, torch_stepped)
    print("   parity (PyTorch vs re-exported ONNX):", summary["parity_export"])
    print("   output change caused by the step:", summary["step_changed_output"])

    np.savez(
        out_dir / IO_NAME,
        input=x_np,
        torch_output_after_step=torch_stepped,
        ort_output_original=ort_original,
    )

    summary["model"]["unchanged"] = full_md5(model_path) == original_md5
    summary["timings_seconds"] = {k: round(v, 2) for k, v in timer.times.items()}
    summary["peak_memory_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    passed = all(
        [
            summary["settings"]["matches_expected"],
            summary["parity_conversion"]["identical"],
            summary["gradient_step"]["n_values_without_gradient"] == 0,
            summary["gradient_step"]["n_tensors_changed_by_step"]
            == summary["gradient_step"]["n_parameter_tensors"],
            summary["export"]["fits_onnxruntime_1_13"],
            summary["export"]["n_initializers_changed"]
            == summary["gradient_step"]["n_parameter_tensors"],
            summary["parity_export"]["identical"],
            summary["model"]["unchanged"],
        ]
    )
    summary["passed"] = passed
    update_results(RESULTS_PATH, "pytorch_check", summary)
    print("timings (s):", summary["timings_seconds"], "peak memory (MB):", summary["peak_memory_mb"])
    print("PASSED" if passed else "FAILED", "->", RESULTS_PATH)
    return passed


def run_install(model_path: Path, out_dir: Path, name: str) -> bool:
    """Copy the re-exported model next to the original and write its settings,
    once onnxruntime 1.13.1 has accepted it."""
    export_path = out_dir / EXPORT_NAME
    destination = model_path.parent / name
    if destination.resolve() == model_path.resolve() or destination.name == model_path.name:
        raise SystemExit(f"refusing to overwrite the original model {model_path.name}")

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    check = results.get("onnxruntime_1_13_check", {})
    export_hash = uvr_hash(export_path)
    if not (check.get("passed") and check.get("uvr_hash") == export_hash):
        raise SystemExit(
            "run task3_uvr.check_onnxruntime_113 on this file first (it must pass)"
        )

    found = lookup_settings(model_path)
    if found["uvr_hash"] != HQ3_HASH or found["settings"] != HQ3_SETTINGS:
        raise SystemExit(f"unexpected settings for {model_path.name}: {found}")

    original_md5 = full_md5(model_path)
    shutil.copyfile(export_path, destination)
    new_hash = uvr_hash(destination)
    settings_path = model_path.parent / "model_data" / f"{new_hash}.json"
    settings_path.write_text(json.dumps(HQ3_SETTINGS, indent=4))
    unchanged = full_md5(model_path) == original_md5

    install = {
        "file": destination.name,
        "uvr_hash": new_hash,
        "settings_file": f"model_data/{settings_path.name}",
        "settings": HQ3_SETTINGS,
        "original_unchanged": unchanged,
    }
    update_results(RESULTS_PATH, "uvr_install", install)
    print("installed", destination.name)
    print("UVR hash:", new_hash)
    print("settings:", settings_path.name, HQ3_SETTINGS)
    print("original model unchanged:", unchanged)
    return unchanged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["check", "install"])
    parser.add_argument("--model", type=Path, required=True, help="path to UVR-MDX-NET-Inst_HQ_3.onnx")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--name", default=EXPORT_NAME, help="file name in UVR's MDX-Net folder")
    args = parser.parse_args()

    if args.command == "check":
        ok = run_check(args.model, args.out_dir, args.seed, args.lr)
    else:
        ok = run_install(args.model, args.out_dir, args.name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
