"""T3-07: reference scores of HQ 3, the benchmark and the mixture, clean vs degraded.

Run from the repository root, after task3_uvr.evaluate on the three datasets:

    uv run python -m task3_uvr.reference_scores

It reads results/task3_sdr_clips.csv only (no audio) and writes:

- results/task3_reference_scores.json: per dataset, model, split and stem,
  the median SDR, HQ 3's drop from musdb7 to each degraded dataset, and each
  model's improvement over the mixture reference of the same dataset;
- results/task3_reference_sdr.png: SDR per test clip as box plots, by model
  and dataset, one panel per stem.

Drops and improvements are medians of per-clip differences (paired by clip),
so they need not equal the difference of the medians. The validation values
on degraded data are the "before" values of T3-10; the test values are for
final reporting only.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import set_style
from task3_uvr.evaluate import CLIPS_CSV, DATASETS, MIXTURE, STEMS

RESULTS_PATH = Path("results") / "task3_reference_scores.json"
FIGURE_PATH = Path("results") / "task3_reference_sdr.png"
HQ3 = "UVR-MDX-NET-Inst_HQ_3"
BENCHMARK = "Benchmark_CentreChannel"
MODELS = (MIXTURE, BENCHMARK, HQ3)
MODEL_LABELS = {MIXTURE: "Mixture (reference)", BENCHMARK: "Benchmark", HQ3: "HQ 3"}
DATASET_LABELS = {
    "musdb7": "clean",
    "musdb7_phone": "phone",
    "musdb7_narrowband": "narrowband",
}
# Categorical slots 1-3 of the reference palette, validated for colour blindness.
DATASET_COLORS = {
    "musdb7": "#2a78d6",
    "musdb7_phone": "#eb6834",
    "musdb7_narrowband": "#1baf7a",
}
SPLITS = ("validation", "test")


def wide(clips: pd.DataFrame) -> pd.DataFrame:
    """SDR indexed by (dataset, split, stem, clip), one column per model."""
    rows = clips[~clips["match_freq_cut"].astype(bool)]
    return rows.pivot_table(
        index=["dataset", "split", "stem", "clip"], columns="model", values="sdr_db"
    )


def medians(clips: pd.DataFrame) -> dict:
    """{dataset: {split: {model: {stem: median SDR}}}} and clip counts."""
    out = {}
    table = wide(clips)
    for (dataset, split, stem), group in table.groupby(level=[0, 1, 2]):
        entry = out.setdefault(dataset, {}).setdefault(split, {"n_clips": len(group)})
        for model in group.columns:
            entry.setdefault(model, {})[stem] = float(group[model].median())
    return out


def hq3_drops(clips: pd.DataFrame) -> dict:
    """{degraded: {split: {stem: median over clips of (clean - degraded) SDR}}}."""
    hq3 = wide(clips)[HQ3].unstack("dataset")
    out = {}
    for degraded in DATASETS[1:]:
        drop = (hq3["musdb7"] - hq3[degraded]).dropna()
        for (split, stem), values in drop.groupby(level=["split", "stem"]):
            out.setdefault(degraded, {}).setdefault(split, {})[stem] = {
                "median_drop_db": float(values.median()),
                "n_clips_dropping": int((values > 0).sum()),
                "n_clips": len(values),
            }
    return out


def improvements(clips: pd.DataFrame) -> dict:
    """{dataset: {split: {model: {stem: median over clips of (model - mixture)}}}}."""
    table = wide(clips)
    out = {}
    for (dataset, split, stem), group in table.groupby(level=[0, 1, 2]):
        for model in (BENCHMARK, HQ3):
            gain = (group[model] - group[MIXTURE]).dropna()
            out.setdefault(dataset, {}).setdefault(split, {}).setdefault(model, {})[
                stem
            ] = {
                "median_gain_db": float(gain.median()),
                "n_clips_above_mixture": int((gain > 0).sum()),
                "n_clips": len(gain),
            }
    return out


def save_figure(clips: pd.DataFrame, split: str = "test") -> Path:
    set_style()
    table = wide(clips).xs(split, level="split")
    fig, axes = plt.subplots(
        1, len(STEMS), figsize=(12, 4.6), sharey=True, constrained_layout=True
    )
    width = 0.24
    for ax, stem in zip(axes, STEMS, strict=True):
        for i, model in enumerate(MODELS):
            for j, dataset in enumerate(DATASETS):
                values = table.xs((dataset, stem), level=["dataset", "stem"])[model]
                box = ax.boxplot(
                    values.dropna(),
                    positions=[i + (j - 1) * (width + 0.03)],
                    widths=width,
                    patch_artist=True,
                    showfliers=True,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.6},
                    medianprops={"color": "black", "linewidth": 1.5},
                )
                for patch in box["boxes"]:
                    patch.set(facecolor=DATASET_COLORS[dataset], alpha=0.85)
                for flier in box["fliers"]:
                    flier.set(
                        markerfacecolor=DATASET_COLORS[dataset], markeredgecolor="none"
                    )
        ax.axhline(0, color="0.6", linewidth=0.8, zorder=0)
        ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        ax.set_xlabel("Model")
        ax.set_title(f"{stem.capitalize()}")
    axes[0].set_ylabel("Global SDR per clip (dB)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DATASET_COLORS[d], alpha=0.85)
        for d in DATASETS
    ]
    n = table.xs((DATASETS[0], STEMS[0]), level=["dataset", "stem"]).shape[0]
    axes[-1].legend(
        handles,
        [DATASET_LABELS[d] for d in DATASETS],
        title="Dataset",
        loc="lower right",
    )
    fig.suptitle(f"SDR per clip, {split} split ({n} clips per box)")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=110)
    plt.close(fig)
    return FIGURE_PATH


def print_tables(summary: dict) -> None:
    print("| Model | Dataset | Split | Clips | Vocals | Accompaniment |")
    print("| --- | --- | --- | --- | --- | --- |")
    for split in SPLITS:
        for model in MODELS:
            for dataset in DATASETS:
                entry = summary["median_sdr_db"][dataset][split]
                v, a = (entry[model][s] for s in STEMS)
                print(
                    f"| {MODEL_LABELS[model]} | {DATASET_LABELS[dataset]} | {split} | "
                    f"{entry['n_clips']} | {v:.2f} | {a:.2f} |"
                )
    print("\nHQ 3 drop from clean (median per-clip, dB):")
    for degraded, by_split in summary["hq3_drop_db"].items():
        for split, by_stem in by_split.items():
            text = ", ".join(
                f"{s} {d['median_drop_db']:.2f} ({d['n_clips_dropping']}/{d['n_clips']})"
                for s, d in by_stem.items()
            )
            print(f"  {DATASET_LABELS[degraded]} {split}: {text}")
    print("\nGain over the mixture reference (median per-clip, dB):")
    for dataset, by_split in summary["gain_over_mixture_db"].items():
        for split, by_model in by_split.items():
            for model, by_stem in by_model.items():
                text = ", ".join(
                    f"{s} {g['median_gain_db']:+.2f} "
                    f"({g['n_clips_above_mixture']}/{g['n_clips']})"
                    for s, g in by_stem.items()
                )
                print(
                    f"  {DATASET_LABELS[dataset]} {split} {MODEL_LABELS[model]}: {text}"
                )


def main(argv: list[str] | None = None) -> int:
    clips = pd.read_csv(CLIPS_CSV)
    missing = sorted(set(DATASETS) - set(clips["dataset"]))
    if missing:
        sys.exit(f"No rows for {missing} in {CLIPS_CSV}: run task3_uvr.evaluate first.")
    summary = {
        "source": str(CLIPS_CSV),
        "note": (
            "Global SDR (dB) against each dataset's own stems. Drops and gains are "
            "medians of per-clip differences. Validation on degraded data = the "
            "'before' values of T3-10; test = final reporting only."
        ),
        "median_sdr_db": medians(clips),
        "hq3_drop_db": hq3_drops(clips),
        "gain_over_mixture_db": improvements(clips),
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print_tables(summary)
    print("->", RESULTS_PATH, save_figure(clips))
    return 0


if __name__ == "__main__":
    sys.exit(main())
