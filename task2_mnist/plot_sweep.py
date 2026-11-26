"""Task 2 figure: time to reach the target as a function of one parameter, for
the runs of one or several sweeps (the rows of results/task2_mnist.csv with a
given tag), with the mean and the standard deviation over the seeds.

    uv run python -m task2_mnist.plot_sweep --tag bs_sweep_lr0.003
    uv run python -m task2_mnist.plot_sweep --tag lr_sweep_gpu --param learning_rate
    uv run python -m task2_mnist.plot_sweep --tag bs_sweep_lr0.003 bs_sweep_sqrt_lr \
        --labels "lr fixe" "lr en racine carrée"

Three panels: the time to reach the target (what we optimise), and its two
factors, the number of steps and the cost of one step. Several tags are drawn
on the same axes, to compare two ways of running the same sweep. Averages only
use the runs that reached the target; the printed tables say how many did.
Writes results/task2_mnist_<tag>.png (tags joined by "_vs_").
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import set_style
from task2_mnist.baseline import FIGURES_DIR, RESULTS_CSV

TRAIN_SIZE = 60_000  # MNIST training images, to convert steps into epochs
PARAM_LABELS = {
    "batch_size": "batch size (images par pas)",
    "learning_rate": "learning rate",
}
FIXED_LABELS = {
    "learning_rate": "lr",
    "batch_size": "batch",
    "eval_every": "contrôle tous les",
}
SHIFT = 0.04  # horizontal offset between sweeps (in decades/octaves), for legibility


def load_results() -> pd.DataFrame:
    """Every row of the results CSV, the JSON config spread into columns, plus
    the time per step (ms) and the number of epochs seen. Rows logged before a
    setting existed (tag, git_commit...) have NaN for it."""
    df = pd.read_csv(RESULTS_CSV)
    config = pd.json_normalize(df["config"].map(json.loads))
    df = pd.concat([df.drop(columns="config"), config], axis=1)
    df["reached"] = df["reached"].eq(True)
    df["ms_per_step"] = df["train_seconds"] / df["steps"] * 1000
    df["epochs"] = df["steps"] * df["batch_size"] / TRAIN_SIZE
    return df


def load_sweep(tag: str) -> pd.DataFrame:
    """Rows of the results CSV with this tag (see load_results)."""
    df = load_results()
    sweep = df[df["tag"] == tag].copy()
    if sweep.empty:
        raise SystemExit(f"no run tagged {tag!r} in {RESULTS_CSV}")
    return sweep


def summarise(sweep: pd.DataFrame, param: str) -> pd.DataFrame:
    """One line per value of `param`: mean and standard deviation over the
    seeds. Steps and times only use the runs that reached the target; the cost
    of a step uses every run, since it does not depend on reaching it."""
    values = sorted(sweep[param].unique())
    reached = sweep[sweep["reached"]]
    summary = (
        reached.groupby(param)
        .agg(
            steps=("steps", "mean"),
            steps_std=("steps", "std"),
            epochs=("epochs", "mean"),
            time_s=("duration_seconds", "mean"),
            time_std=("duration_seconds", "std"),
            load_s=("load_seconds", "mean"),
            inference_ms=("inference_seconds", lambda s: s.mean() * 1000),
        )
        .reindex(values)
    )
    summary["ms_per_step"] = sweep.groupby(param)["ms_per_step"].mean().reindex(values)
    counts = sweep.groupby(param)["reached"].agg(["sum", "count"])
    summary.insert(0, "reached", counts["sum"].astype(str) + "/" + counts["count"].astype(str))
    return summary


def fixed_settings(runs: pd.DataFrame, param: str) -> str:
    """The settings shared by every run, for the figure title."""
    parts = []
    for column, label in FIXED_LABELS.items():
        values = runs[column].unique()
        if column != param and len(values) == 1:
            value = f"{values[0]:g}".replace(".", ",")
            parts.append(f"{label} {value}" + (" pas" if column == "eval_every" else ""))
    device = runs["gpu"].iloc[0] if runs["device"].iloc[0] == "cuda" else "CPU"
    return ", ".join([*parts, device])


def plot(
    tags: list[str], param: str, output: Path, labels: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Draw the sweeps of `tags` on the same three panels; return their
    summaries, by tag."""
    labels = labels or (["moyenne ± écart-type"] if len(tags) == 1 else tags)
    if len(labels) != len(tags):
        raise SystemExit("--labels needs one label per tag")
    set_style()
    sweeps = {tag: load_sweep(tag) for tag in tags}
    runs = pd.concat(sweeps.values())
    target = f"{runs['target'].iloc[0]:g}".replace(".", ",")
    base = 2 if param == "batch_size" else 10
    values = sorted(runs[param].unique())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    ax_time, ax_steps, ax_cost = axes
    panels = (
        (ax_time, "duration_seconds", "time_s", "time_std",
         f"temps pour atteindre {target} (s)", "Temps d'apprentissage"),
        (ax_steps, "steps", "steps", "steps_std",
         f"pas pour atteindre {target}", "Nombre de pas"),
        (ax_cost, "ms_per_step", "ms_per_step", None,
         "temps par pas (ms)", "Coût d'un pas"),
    )
    summaries = {}
    for i, (tag, label) in enumerate(zip(tags, labels, strict=True)):
        sweep = sweeps[tag]
        summary = summaries[tag] = summarise(sweep, param)
        reached = sweep[sweep["reached"]]
        missed = sweep[~sweep["reached"]]
        shift = base ** (SHIFT * (i - (len(tags) - 1) / 2))
        for ax, run_column, mean_column, std_column, _, _ in panels:
            ax.scatter(reached[param] * shift, reached[run_column], color=f"C{i}",
                       alpha=0.3, s=14)
            ax.errorbar(summary.index.to_numpy() * shift, summary[mean_column],
                        yerr=summary[std_column] if std_column else None,
                        marker="o", capsize=3, color=f"C{i}", label=label)
        # the cost of a step is known even when the target is not reached
        ax_cost.scatter(missed[param] * shift, missed["ms_per_step"], color=f"C{i}",
                        alpha=0.5, s=18, marker="x")
        for value, counts in sweep.groupby(param)["reached"].agg(["sum", "count"]).iterrows():
            if counts["sum"] < counts["count"]:
                for ax in (ax_time, ax_steps):
                    ax.annotate(
                        f"cible non\natteinte\n({counts['sum']}/{counts['count']})",
                        xy=(value * shift, 0.97), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=7, color=f"C{i}",
                    )
    for ax, _, _, _, ylabel, title in panels:
        ax.set_xscale("log", base=base)
        tick_labels = [f"{v:g}".replace(".", ",") for v in values]
        if param == "batch_size":
            ax.set_xticks(values, tick_labels)
        else:  # close values on a log axis: tilt the labels so they don't overlap
            ax.set_xticks(values, tick_labels, rotation=40, ha="right")
        ax.minorticks_off()
        ax.set_xlabel(PARAM_LABELS.get(param, param))
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
    legend_title = "points pâles : un run (une graine)"
    if not runs["reached"].all():
        legend_title += "\n× : run qui n'atteint pas la cible"
    ax_time.legend(fontsize=8, title=legend_title, title_fontsize=7)
    heading = tags[0] if len(tags) == 1 else f"{len(tags)} balayages"
    fig.suptitle(
        f"{heading} : {len(runs)} runs, cible {target} de test accuracy "
        f"({fixed_settings(runs, param)})"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"figure written to {output}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", nargs="+", required=True, help="tag(s) of the sweeps")
    parser.add_argument("--labels", nargs="+", help="legend label of each tag")
    parser.add_argument("--param", default="batch_size", choices=list(PARAM_LABELS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    name = "_vs_".join(args.tag)
    output = args.output or Path(FIGURES_DIR) / f"task2_mnist_{name}.png"
    summaries = plot(args.tag, args.param, output, args.labels)
    pd.set_option("display.width", 200)
    for tag, summary in summaries.items():
        print(f"\n{tag}:\n{summary.round(2).to_string()}")


if __name__ == "__main__":
    main()
