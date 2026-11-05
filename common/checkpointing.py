"""Save/resume a PyTorch training run, for Colab/Kaggle sessions that can
disconnect mid-training (task 3 UVR and task 4 Food-101 fine-tuning).
Checkpoints should live on persistent storage (e.g. a mounted Google Drive
folder), not Colab's ephemeral local disk.
"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save everything needed to resume training after ``epoch`` completed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> tuple[int, dict[str, Any]]:
    """Restore model/optimizer state in place and return the next epoch to
    run plus any extra metadata that was saved."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    next_epoch = checkpoint["epoch"] + 1
    return next_epoch, checkpoint["extra"]
