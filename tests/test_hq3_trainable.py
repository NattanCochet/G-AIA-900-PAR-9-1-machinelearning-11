"""T3-05 checks on a tiny ONNX model with HQ 3's kinds of layers (Conv,
BatchNorm, MatMul, ConvTranspose), so they run without UVR or the real model."""

import hashlib

import numpy as np
import onnx
import pytest
import torch
from torch import nn

from task3_uvr.hq3_trainable import (
    convert,
    describe_export,
    export_onnx,
    forward_eval,
    gradient_step,
    inspect_onnx,
    run_onnxruntime,
)
from task3_uvr.onnx_checks import MAX_IR_VERSION, MAX_OPSET, compare_outputs, uvr_hash

SHAPE = (1, 4, 8, 16)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 6, kernel_size=3, padding=1)
        # After the ReLU, so the exporter cannot fuse it into the convolution.
        self.norm = nn.BatchNorm2d(6)
        self.tdf = nn.Linear(SHAPE[-1], SHAPE[-1], bias=False)  # exported as MatMul
        self.up = nn.ConvTranspose2d(6, 4, kernel_size=1)

    def forward(self, x):
        return self.up(self.tdf(self.norm(torch.relu(self.conv(x)))))


@pytest.fixture
def tiny_onnx(tmp_path):
    torch.manual_seed(0)
    source = Tiny()
    source.norm.running_mean.uniform_(-1, 1)
    source.norm.running_var.uniform_(0.5, 2)
    source.eval()
    path = tmp_path / "tiny.onnx"
    torch.onnx.export(
        source, (torch.zeros(SHAPE),), str(path),
        input_names=["input"], output_names=["output"],
        opset_version=13, dynamo=False,
    )
    return source, path, onnx.load(str(path))


@pytest.fixture
def x():
    return torch.randn(SHAPE, generator=torch.Generator().manual_seed(1))


def test_inspect_excludes_batchnorm_statistics(tiny_onnx):
    source, _, model = tiny_onnx
    info = inspect_onnx(model)
    assert "BatchNormalization" in info["operators"]
    assert "MatMul" in info["operators"]
    assert info["n_batchnorm_statistics_values"] == 2 * 6
    assert info["n_trainable_values_expected"] == sum(p.numel() for p in source.parameters())


def test_conversion_matches_onnxruntime(tiny_onnx, x):
    _, path, model = tiny_onnx
    module = convert(model)
    parity = compare_outputs(run_onnxruntime(path, x.numpy()), forward_eval(module, x.numpy()))
    assert parity["identical"], parity


def test_gradient_step_reaches_every_parameter(tiny_onnx, x):
    _, _, model = tiny_onnx
    module = convert(model)
    report = gradient_step(module, x, torch.randn(SHAPE), lr=1e-3)
    assert report["n_values_without_gradient"] == 0
    assert report["n_tensors_changed_by_step"] == report["n_parameter_tensors"] > 0


def test_gradient_step_reports_layers_without_gradient(tiny_onnx, x):
    _, _, model = tiny_onnx
    module = convert(model)
    frozen = next(name for name, _ in module.named_parameters())
    module.get_parameter(frozen).requires_grad_(False)
    report = gradient_step(module, x, torch.randn(SHAPE), lr=1e-3)
    assert report["parameters_without_gradient"] == [frozen]
    assert report["layers_without_gradient"] == [frozen.rsplit(".", 1)[0]]
    assert report["n_tensors_changed_by_step"] == report["n_parameter_tensors"] - 1


def test_export_fits_onnxruntime_113_and_holds_updated_weights(tiny_onnx, x, tmp_path):
    _, _, model = tiny_onnx
    module = convert(model)
    report = gradient_step(module, x, torch.randn(SHAPE), lr=1e-3)
    path = tmp_path / "stepped.onnx"
    export_onnx(module, x, path)
    exported = describe_export(path, model)
    assert exported["ir_version"] <= MAX_IR_VERSION
    assert exported["opset"] <= MAX_OPSET
    assert exported["n_initializers_changed"] == report["n_parameter_tensors"]
    parity = compare_outputs(forward_eval(module, x.numpy()), run_onnxruntime(path, x.numpy()))
    assert parity["identical"], parity


def test_uvr_hash_uses_last_10000_kib(tmp_path):
    small = tmp_path / "small.bin"
    small.write_bytes(b"abc")
    assert uvr_hash(small) == hashlib.md5(b"abc").hexdigest()

    data = np.random.default_rng(0).bytes(10000 * 1024 + 100)
    large = tmp_path / "large.bin"
    large.write_bytes(data)
    assert uvr_hash(large) == hashlib.md5(data[100:]).hexdigest()


def test_compare_outputs_threshold_is_relative():
    reference = np.array([10.0, -2.0])
    assert compare_outputs(reference, reference + 9e-4)["identical"]
    assert not compare_outputs(reference, reference + 2e-3)["identical"]
