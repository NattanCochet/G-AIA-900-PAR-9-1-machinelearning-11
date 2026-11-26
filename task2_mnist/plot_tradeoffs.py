"""Task 2 synthesis figures, built from results/task2_mnist.csv:

1. results/task2_mnist_bs_tradeoff.png, the batch size trade-off: each point is
   a batch size, placed by the number of steps it needs to reach the target
   and by the cost of one step. Dotted lines join the points of equal training
   time (steps x cost of a step): a bigger batch needs fewer steps, but each
   step costs more, and the best batch size is the one on the lowest line.
2. results/task2_mnist_methods_summary.png: learning time and inference time of
   the successive configurations, from the first measured one to the fastest,
   to show which method acts on which of the two times.

    uv run python -m task2_mnist.plot_tradeoffs

The two Colab rows of the summary come from Results/result_task_2_default.md
(section 4): those runs were logged in Colab's copy of the CSV, not versioned.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from common import set_style
from task2_mnist.baseline import FIGURES_DIR
from task2_mnist.plot_sweep import load_results

TRADEOFF_FIGURE = Path(FIGURES_DIR) / "task2_mnist_bs_tradeoff.png"
SUMMARY_FIGURE = Path(FIGURES_DIR) / "task2_mnist_methods_summary.png"

BATCH_SWEEPS = {  # tag -> legend label
    "bs_sweep_lr0.003": "lr fixe = 0,003",
    "bs_sweep_sqrt_lr": "lr = 0,003 × √(batch / 32)",
}
ISO_SECONDS = (0.5, 1, 2, 3, 4)  # lines of equal training time

MACHINE_COLORS = {"T4": "C2", "CPU": "C7", "RTX 3080": "C0"}
# Colab runs (Tesla T4), section 4 of the report, 5 seeds each:
# (label, machine, learning time mean and std in s, inference time in ms)
COLAB_ROWS = [
    ("DataLoader (référence)\nbatch 32, lr 0,001", "T4", 9.3, 0.7, 1000.0),
    ("Données sur le GPU\nbatch 32, lr 0,001", "T4", 4.2, 0.6, 20.0),
]


def french(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def plot_batch_tradeoff(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    curves = []
    for i, (tag, label) in enumerate(BATCH_SWEEPS.items()):
        runs = df[(df["tag"] == tag) & df["reached"]]
        if runs.empty:
            continue
        points = runs.groupby("batch_size").agg(
            steps=("steps", "mean"), ms=("ms_per_step", "mean")
        )
        curves.append(points)
        ax.plot(points["steps"], points["ms"], marker="o", color=f"C{i}", label=label)
        for batch, row in points.iterrows():  # first sweep labelled above, second below
            ax.annotate(f"{batch}", (row["steps"], row["ms"]), textcoords="offset points",
                        xytext=(-7, 4) if i == 0 else (4, -12), ha="right" if i == 0 else "left",
                        fontsize=8, color=f"C{i}")

    all_points = pd.concat(curves)
    x_low, x_high = all_points["steps"].min() * 0.75, all_points["steps"].max() * 1.3
    y_low, y_high = all_points["ms"].min() * 0.75, all_points["ms"].max() * 1.25
    xs = np.geomspace(x_low, x_high, 200)
    for seconds in ISO_SECONDS:
        ax.plot(xs, seconds * 1000 / xs, color="grey", linestyle=":", linewidth=1)
        for y in (y_high * 0.93, y_low * 1.07):  # label where the line is in view
            x = seconds * 1000 / y
            if x_low * 1.05 < x < x_high * 0.95:
                ax.text(x, y, f"{french(seconds, 1).replace(',0', '')} s", fontsize=8,
                        color="grey", ha="center", va="center",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
                break

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.xaxis.set_major_locator(FixedLocator([200, 300, 500, 700, 1000, 1500]))
    ax.yaxis.set_major_locator(FixedLocator([2, 2.5, 3, 3.5, 4, 5]))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.set_xlabel("pas pour atteindre 0,97 (échelle log)")
    ax.set_ylabel("temps par pas (ms, échelle log)")
    ax.set_title("Compromis du batch size : moins de pas, mais des pas plus chers")
    ax.legend(fontsize=8, loc="lower right", title="étiquettes : batch size ; pointillés : "
              "temps\nd'entraînement égal (pas × temps par pas)", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"figure written to {output}")


def local_rows(df: pd.DataFrame) -> list[tuple]:
    """Configurations measured on the local PC, from the CSV."""
    cpu = (
        (df["device"] == "cpu") & df["tag"].isna() & (df["learning_rate"] == 0.001)
        & df["data_on_device"].eq(True) & (df["batch_size"] == 32)
    )
    lr_sweep = df["tag"] == "lr_sweep_gpu"
    selections = [
        ("Données en mémoire (CPU seul)\nbatch 32, lr 0,001", "CPU", cpu),
        ("Données sur le GPU\nbatch 32, lr 0,001", "RTX 3080",
         lr_sweep & (df["learning_rate"] == 0.001)),
        ("Données sur le GPU\nbatch 32, lr 0,003", "RTX 3080",
         lr_sweep & (df["learning_rate"] == 0.003)),
        ("Données sur le GPU\nbatch 1 024, lr 0,003", "RTX 3080",
         (df["tag"] == "bs_sweep_lr0.003") & (df["batch_size"] == 1024)),
    ]
    rows = []
    for label, machine, mask in selections:
        runs = df[mask & df["reached"]]
        if runs.empty:
            print(f"no run for {label!r}: skipped")
            continue
        rows.append((label, machine, runs["duration_seconds"].mean(),
                     runs["duration_seconds"].std(), runs["inference_seconds"].mean() * 1000))
    return rows


def plot_methods_summary(df: pd.DataFrame, output: Path) -> None:
    rows = COLAB_ROWS + local_rows(df)
    labels = [f"{machine} : {label}" for label, machine, *_ in rows]
    colors = [MACHINE_COLORS[machine] for _, machine, *_ in rows]
    y = np.arange(len(rows))

    fig, (ax_learn, ax_infer) = plt.subplots(1, 2, figsize=(12, 0.75 * len(rows) + 1.6),
                                             sharey=True)
    learn = [row[2] for row in rows]
    spread = [row[3] for row in rows]
    ax_learn.barh(y, learn, xerr=spread, color=colors, capsize=3)
    for yi, (_, machine, value, std, _) in zip(y, rows, strict=True):
        digits = 1 if machine == "T4" else 2  # the Colab figures have one decimal
        ax_learn.text(value + std + 0.25, yi, f"{french(value, digits)} s", va="center",
                      fontsize=8)
    ax_learn.set_xlabel("temps d'apprentissage pour atteindre 0,97 (s)")
    ax_learn.set_title("Apprentissage : chargement + setup + entraînement")
    ax_learn.set_xlim(0, max(learn) * 1.25)

    infer = [row[4] for row in rows]
    ax_infer.barh(y, infer, color=colors)
    for yi, value in zip(y, infer, strict=True):
        ax_infer.text(value * 1.15, yi, f"{value:,.0f} ms".replace(",", " "),
                      va="center", fontsize=8)
    ax_infer.set_xscale("log")
    ax_infer.set_xlim(right=max(infer) * 8)
    ax_infer.set_xlabel("temps d'inférence sur les 10 000 images de test (ms, échelle log)")
    ax_infer.set_title("Inférence")

    ax_learn.set_yticks(y, labels, fontsize=8)
    ax_learn.invert_yaxis()
    ax_learn.set_ylabel("configuration (dans l'ordre des essais)")
    for ax in (ax_learn, ax_infer):
        ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.02,
             "Contrôle de la test accuracy tous les 100 pas (T4, CPU) ou tous les 10 pas "
             "(RTX 3080). Moyennes sur 5 graines (T4) ou 3 graines (PC local) ; barres "
             "noires : écart-type.", fontsize=7, color="grey")
    fig.suptitle("Effet de chaque méthode sur le temps d'apprentissage et le temps d'inférence")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"figure written to {output}")


def main() -> None:
    set_style()
    df = load_results()
    plot_batch_tradeoff(df, TRADEOFF_FIGURE)
    plot_methods_summary(df, SUMMARY_FIGURE)


if __name__ == "__main__":
    main()
