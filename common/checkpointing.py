"""Save/resume a PyTorch training run, for Colab/Kaggle sessions that can
disconnect mid-training (task 3 UVR and task 4 Food-101 fine-tuning).
Checkpoints should live on persistent storage (e.g. a mounted Google Drive
folder), not Colab's ephemeral local disk.

Checkpoints are written atomically, so a disconnect during a save leaves the
previous checkpoint intact. Only load checkpoints you created yourself:
``torch.load`` with ``weights_only=False`` can execute code stored in the file.
"""

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _get_rng_state() -> dict[str, Any]:
    """Capture every RNG a training run may draw from."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG states captured by ``_get_rng_state``.

    RNG tensors must be CPU byte tensors, even if ``map_location`` moved them.
    A checkpoint can be resumed on a machine with no GPU or a different number
    of GPUs; only the GPUs present in both are restored.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())

    if "cuda" in state and torch.cuda.is_available():
        for index, cuda_state in enumerate(state["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state.cpu(), index)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> None:
    """Atomically save everything needed to resume after ``epoch`` completed.

    scheduler: optional learning-rate scheduler, saved with its state_dict.
    scaler: optional ``torch.amp.GradScaler``, saved with its state_dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra or {},
        "rng_state": _get_rng_state(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()

    # The temporary file must be in the same directory: os.replace is only
    # atomic within a single filesystem.
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Restore the training state in place.

    Returns the next epoch to run and any extra metadata that was saved.
    Checkpoints written before RNG states were saved still load, without
    restoring RNG. Passing a scheduler or scaler the checkpoint does not contain
    raises ``KeyError``, rather than silently resuming with a reset state.
    """
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)

    for name, component in (("scheduler", scheduler), ("scaler", scaler)):
        if component is not None and f"{name}_state" not in checkpoint:
            raise KeyError(f"Checkpoint {path} has no saved {name} state")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if "rng_state" in checkpoint:
        _set_rng_state(checkpoint["rng_state"])

    next_epoch = checkpoint["epoch"] + 1
    return next_epoch, checkpoint["extra"]