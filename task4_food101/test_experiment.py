from argparse import Namespace

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from task4_food101.experiment import (
    accuracy,
    compress,
    load_model,
    make_model,
    measure,
    quantize,
    reference_order,
    save_model,
    split_indices,
    train,
)
from task4_food101.report import frontier, report


def test_split_is_deterministic_and_disjoint():
    train, val = split_indices(100, 4)
    assert (train, val) == split_indices(100, 4)
    assert len(train) == 90 and len(val) == 10
    assert set(train).isdisjoint(val)
    assert set(train + val) == set(range(100))


def test_reference_class_order_is_not_assumed():
    assert reference_order({0: "b", 1: "a"}, ["a", "b"]).tolist() == [1, 0]
    with pytest.raises(ValueError):
        reference_order({0: "b", 1: "c"}, ["a", "b"])


def test_top1_and_top5():
    class Fixed(nn.Module):
        def forward(self, x):
            return x

    data = TensorDataset(
        torch.tensor([[5.0, 4, 3, 2, 1, 0], [5.0, 4, 3, 2, 1, 0]]),
        torch.tensor([0, 4]),
        torch.arange(2),
    )
    assert accuracy(Fixed(), DataLoader(data, batch_size=2), "cpu") == (0.5, 1.0)


@pytest.mark.parametrize("method", ["dynamic", "static"])
def test_quantized_checkpoint_roundtrip(tmp_path, method):
    torch.set_num_threads(1)
    model = make_model("resnet18")
    images = torch.randn(1, 3, 224, 224)
    calibration = (
        [(images, torch.tensor([0]), torch.tensor([0]))] if method == "static" else None
    )
    model = quantize(model, method, calibration)
    path = tmp_path / "model.pt"
    save_model(path, model, {"architecture": "resnet18", "method": method, "seed": 0})
    restored, _ = load_model(path)
    with torch.inference_mode():
        torch.testing.assert_close(restored(images), model(images))


def test_frontier_excludes_dominated_points():
    rows = pd.DataFrame(
        {"name": ["a", "b", "c"], "size": [1, 2, 3], "top1": [0.8, 0.7, 0.9]}
    )
    assert frontier(rows, "size")["name"].tolist() == ["a", "c"]


def test_pruning_retrains_and_saves_dense_zero_weights(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    moves = []
    original_to = nn.Module.to

    def track_to(model, *args, **kwargs):
        moves.append((model, args))
        return original_to(model, *args, **kwargs)

    monkeypatch.setattr(nn.Module, "to", track_to)
    data = TensorDataset(
        torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), torch.arange(2)
    )
    monkeypatch.setattr("task4_food101.experiment.dataset_for", lambda *args: data)
    start, output = tmp_path / "start.pt", tmp_path / "pruned.pt"
    save_model(
        start,
        make_model("resnet18"),
        {"architecture": "resnet18", "method": "baseline", "seed": 0},
    )
    train(
        Namespace(
            seed=0,
            data_dir=tmp_path,
            batch_size=2,
            workers=0,
            cache=None,
            alpha=0.0,
            architecture="resnet18",
            start=start,
            prune=0.5,
            learning_rate=1e-4,
            epochs=1,
            output=output,
            temperature=4.0,
            resume=False,
        )
    )
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert any(args == (expected_device,) for _, args in moves)
    model, metadata = load_model(output)
    weights = [
        m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    assert metadata["method"] == "pruned"
    assert sum((w == 0).sum().item() for w in weights) / sum(
        w.numel() for w in weights
    ) == pytest.approx(0.5, abs=1e-5)


def test_qat_checkpoint_roundtrip(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = TensorDataset(
        torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), torch.arange(2)
    )
    monkeypatch.setattr("task4_food101.experiment.dataset_for", lambda *args: data)
    start, output = tmp_path / "start.pt", tmp_path / "qat.pt"
    save_model(
        start,
        make_model("resnet18"),
        {"architecture": "resnet18", "method": "distilled", "seed": 0},
    )
    compress(
        Namespace(
            input=start,
            output=output,
            method="qat",
            seed=0,
            data_dir=tmp_path,
            batch_size=2,
            workers=0,
            epochs=1,
            learning_rate=1e-5,
        )
    )
    model, metadata = load_model(output)
    assert metadata["source_method"] == "distilled"
    with torch.inference_mode():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 101)


def test_training_resumes_with_matching_state(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = TensorDataset(
        torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), torch.arange(2)
    )
    monkeypatch.setattr("task4_food101.experiment.dataset_for", lambda *args: data)
    start = tmp_path / "start.pt"
    save_model(
        start,
        make_model("resnet18"),
        {"architecture": "resnet18", "method": "baseline", "seed": 0},
    )
    args = Namespace(
        seed=0,
        data_dir=tmp_path,
        batch_size=2,
        workers=0,
        cache=None,
        alpha=0.0,
        architecture="resnet18",
        start=start,
        prune=0.5,
        learning_rate=1e-4,
        epochs=2,
        output=tmp_path / "uninterrupted.pt",
        temperature=4.0,
        resume=False,
    )
    train(args)
    args.output = tmp_path / "resumed.pt"
    args.epochs = 1
    train(args)
    args.epochs = 2
    args.resume = True
    train(args)
    first = torch.load(tmp_path / "uninterrupted.pt.resume.pt", weights_only=True)[
        "model"
    ]
    resumed = torch.load(tmp_path / "resumed.pt.resume.pt", weights_only=True)["model"]
    for key in first:
        torch.testing.assert_close(first[key], resumed[key])


def test_measure_and_report_and_reject_mismatched_artifacts(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = TensorDataset(
        torch.randn(2, 3, 32, 32), torch.tensor([0, 1]), torch.arange(2)
    )
    monkeypatch.setattr(
        "task4_food101.experiment.dataset_for", lambda *args, **kwargs: data
    )
    checkpoint = tmp_path / "model.pt"
    save_model(
        checkpoint,
        make_model("resnet18"),
        {"architecture": "resnet18", "method": "baseline", "seed": 0},
    )
    result = tmp_path / "results.csv"
    args = Namespace(
        threads=1,
        data_dir=tmp_path,
        split="validation",
        seed=0,
        reference=False,
        input=checkpoint,
        batch_size=2,
        workers=0,
        latency_images=2,
        warmup=1,
        passes=2,
        name="student",
        results=result,
    )
    measure(args)
    args.split = "test"
    measure(args)
    rows = pd.read_csv(result)
    ref = rows.iloc[1].copy()
    ref["name"] = "swin"
    ref["artifact"] = "reference"
    pd.concat([rows, ref.to_frame().T], ignore_index=True).to_csv(result, index=False)
    report(result, tmp_path, "swin")
    assert (tmp_path / "task4_accuracy_size.png").is_file()
    assert (tmp_path / "task4_accuracy_latency.png").is_file()
    rows = pd.read_csv(result)
    rows.loc[(rows.name == "student") & (rows.split == "test"), "artifact"] = (
        "different"
    )
    rows.to_csv(result, index=False)
    with pytest.raises(ValueError, match="different model artifacts"):
        report(result, tmp_path, "swin")
