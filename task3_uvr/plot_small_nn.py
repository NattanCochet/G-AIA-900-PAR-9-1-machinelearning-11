"""T3-08: training and validation loss curves of SmallMaskNet runs.

    uv run python -m task3_uvr.plot_small_nn <run>_log.csv [...] \
        --output results/task3_small_nn_losses.png

Each CSV is a log written by task3_uvr.train_small_nn (step, train_loss,
val_loss, train_seconds). Training losses are solid lines, validation losses
dashed, one colour per run.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import new_figure, set_style

# Categorical palette, validated for colour blindness (same as T3-07's figure).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c9a227", "#d6457a")


def plot_losses(
    logs: dict[str, Path],
    output: Path,
    *,
    ylabel: str = "L1 spectrogram loss",
    yunit: str = "STFT amplitude",
    title: str = "SmallMaskNet: training and validation loss",
) -> Path:
    """One figure with the training (solid) and validation (dashed) L1 loss
    of every run, against the training step. Rows without a training loss
    (T3-10's step 0) only get a validation point."""
    set_style()
    fig, ax = new_figure(
        "training step (batches)", ylabel, yunit=yunit, title=title
    )
    for i, (label, path) in enumerate(logs.items()):
        log = pd.read_csv(path)
        color = COLORS[i % len(COLORS)]
        trained = log.dropna(subset=["train_loss"])
        ax.plot(
            trained["step"], trained["train_loss"], color=color, label=f"{label} train"
        )
        ax.plot(
            log["step"],
            log["val_loss"],
            color=color,
            linestyle="--",
            label=f"{label} validation",
        )
    ax.legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results") / "task3_small_nn_losses.png"
    )
    args = parser.parse_args(argv)
    logs = {
        path.stem.removesuffix("_log").removeprefix("SmallMaskNet_"): path
        for path in args.logs
    }
    print("->", plot_losses(logs, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
