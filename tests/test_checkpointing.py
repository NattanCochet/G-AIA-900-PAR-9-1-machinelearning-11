"""Checks that an interrupted training run can resume exactly (SETUP-05)."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import set_seed
from common.checkpointing import load_checkpoint, save_checkpoint

EPOCHS = 6
INTERRUPT_AFTER_EPOCH = 2


def make_loader() -> DataLoader:
    """Small fixed dataset; shuffling makes training depend on the RNG state."""
    generator = torch.Generator().manual_seed(123)
    features = torch.randn(64, 8, generator=generator)
    targets = torch.randn(64, 1, generator=generator)
    return DataLoader(TensorDataset(features, targets), batch_size=8, shuffle=True)


def make_training_objects():
    """Dropout and shuffling draw random numbers; the scheduler has its own state."""
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    return model, optimizer, scheduler


def train_one_epoch(model, optimizer, scheduler, loader) -> None:
    model.train()
    for features, targets in loader:
        optimizer.zero_grad()
        nn.functional.mse_loss(model(features), targets).backward()
        optimizer.step()
    scheduler.step()


def assert_same_weights(first: nn.Module, second: nn.Module) -> None:
    for (name, value), other in zip(
        first.state_dict().items(), second.state_dict().values(), strict=True
    ):
        assert torch.equal(value, other), f"weights differ for {name}"


def test_resumed_training_matches_uninterrupted_run(tmp_path):
    loader = make_loader()
    path = tmp_path / "checkpoint.pt"

    set_seed(0)
    straight = make_training_objects()
    for _ in range(EPOCHS):
        train_one_epoch(*straight, loader)

    set_seed(0)
    model, optimizer, scheduler = make_training_objects()
    for epoch in range(INTERRUPT_AFTER_EPOCH + 1):
        train_one_epoch(model, optimizer, scheduler, loader)
    save_checkpoint(path, model, optimizer, epoch, scheduler=scheduler)

    # Simulate a new session: different RNG state and brand new objects.
    set_seed(999)
    resumed_model, resumed_optimizer, resumed_scheduler = make_training_objects()
    next_epoch, _ = load_checkpoint(
        path, resumed_model, resumed_optimizer, scheduler=resumed_scheduler
    )

    assert next_epoch == INTERRUPT_AFTER_EPOCH + 1
    for _ in range(next_epoch, EPOCHS):
        train_one_epoch(resumed_model, resumed_optimizer, resumed_scheduler, loader)

    assert_same_weights(straight[0], resumed_model)
    assert resumed_scheduler.get_last_lr() == straight[2].get_last_lr()


def test_scaler_state_is_restored(tmp_path):
    class FakeScaler:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def state_dict(self) -> dict[str, float]:
            return {"scale": self.scale}

        def load_state_dict(self, state: dict[str, float]) -> None:
            self.scale = state["scale"]

    model, optimizer, _ = make_training_objects()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 0, scaler=FakeScaler(128.0))

    restored = FakeScaler(1.0)
    load_checkpoint(path, model, optimizer, scaler=restored)

    assert restored.scale == 128.0


def test_failed_save_keeps_previous_checkpoint(tmp_path, monkeypatch):
    model, optimizer, _ = make_training_objects()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated disconnect")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        save_checkpoint(path, model, optimizer, epoch=2)
    monkeypatch.undo()

    next_epoch, _ = load_checkpoint(path, model, optimizer)
    assert next_epoch == 2
    assert [file.name for file in tmp_path.iterdir()] == ["checkpoint.pt"]


def test_checkpoint_without_rng_state_still_loads(tmp_path):
    model, optimizer, _ = make_training_objects()
    path = tmp_path / "old_checkpoint.pt"
    torch.save(
        {
            "epoch": 3,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "extra": {"note": "old format"},
        },
        path,
    )

    next_epoch, extra = load_checkpoint(path, model, optimizer)

    assert next_epoch == 4
    assert extra == {"note": "old format"}


def test_missing_scheduler_state_raises(tmp_path):
    model, optimizer, scheduler = make_training_objects()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 0)

    with pytest.raises(KeyError, match="scheduler"):
        load_checkpoint(path, model, optimizer, scheduler=scheduler)