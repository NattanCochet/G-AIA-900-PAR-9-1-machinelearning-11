"""Select and plot Food-101 compression results.

Selection rule (fixed before looking at test scores): among the candidate models,
keep those whose *validation* top-1 is within ``--max-drop`` of the best candidate's
validation top-1, then take the cheapest one (``--cost``, default size on disk).
The test split is only used to report the result, never to choose.

The reference (Swin) is excluded from selection: it was fine-tuned by a third
party on the full Food-101 train split, which contains our validation images, so
its validation score is not comparable with the students'.
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the script only writes files
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from common.plotting import set_style  # noqa: E402

MAX_DROP = 0.01
FIGSIZE = (16, 9)  # 16:9, fills a slide
SAVE_DPI = 200

METHOD_ORDER = ["reference", "baseline", "distilled", "pruned", "dynamic", "static", "qat"]
METHOD_LABELS = {
    "reference": "Reference (Swin)",
    "baseline": "Baseline",
    "distilled": "Distilled",
    "pruned": "Pruned",
    "dynamic": "Dynamic int8",
    "static": "Static int8",
    "qat": "QAT int8",
}
METHOD_COLORS = dict(zip(METHOD_ORDER, plt.get_cmap("tab10").colors))
ARCH_MARKERS = {"mobilenet_v3_large": "o", "efficientnet_b0": "s", "resnet18": "^"}
ARCH_NAMES = {
    "mobilenet_v3_large": "MobileNetV3-L",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet-18",
}
ARCH_SHORT = {"mobilenet_v3_large": "MNv3", "efficientnet_b0": "EffB0", "resnet18": "R18"}

REQUIRED = {
    "name",
    "split",
    "top1",
    "top5",
    "size_mb",
    "latency_ms",
    "threads",
    "device",
    "seed",
    "artifact",
}
COSTS = (
    ("size_mb", "task4_accuracy_size.png", "Model size on disk (MB)", "model size"),
    (
        "latency_ms",
        "task4_accuracy_latency.png",
        "CPU latency, 1 thread (ms per image)",
        "CPU latency",
    ),
    (
        "gzip_mb",
        "task4_accuracy_gzip_size.png",
        "Gzip-compressed model size (MB)",
        "compressed size",
    ),
)


def frontier(frame: pd.DataFrame, cost: str) -> pd.DataFrame:
    ordered = frame.sort_values([cost, "top1"], ascending=[True, False])
    return ordered[ordered["top1"] > ordered["top1"].cummax().shift(fill_value=-1)]


def short_label(row) -> str:
    if row["method"] == "reference":
        return "Swin (reference)"
    architecture = row["architecture"]
    prefix = f"{architecture}_"
    suffix = row["name"][len(prefix) :] if row["name"].startswith(prefix) else row["name"]
    if match := re.fullmatch(r"t([\d.]+)_a([\d.]+)", suffix):
        suffix = f"KD T={match[1]} α={match[2]}"
    elif match := re.fullmatch(r"pruned_([\d.]+)", suffix):
        suffix = f"pruned {float(match[1]) * 100:.0f}%"
    return f"{ARCH_SHORT.get(architecture, architecture)} {suffix}"


def place_labels(fig, ax, points, fontsize=13):
    """Greedy label placement. points: (x, y, text) in priority order.

    Tries positions on rings around each marker and takes the first one that stays
    inside the axes and overlaps neither a marker nor an earlier label. Labels that
    fit nowhere are skipped and returned, so nothing is ever drawn on top of text.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72
    area = ax.get_window_extent(renderer)
    centres = [ax.transData.transform((x, y)) for x, y, _ in points]
    radius = 11 * scale
    taken = [
        Bbox.from_extents(cx - radius, cy - radius, cx + radius, cy + radius)
        for cx, cy in centres
    ]
    angles = [30, -30, 150, -150, 0, 180, 60, -60, 120, -120, 90, -90]
    candidates = [
        (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
        for r in (18, 40, 68, 100)
        for a in angles
    ]
    skipped = []
    for (x, y, text), (cx, cy) in zip(points, centres):
        probe = fig.text(0, 0, text, fontsize=fontsize)
        width, height = probe.get_window_extent(renderer).size
        probe.remove()
        for dx, dy in candidates:
            ha = "left" if dx > 3 else "right" if dx < -3 else "center"
            va = "bottom" if dy > 3 else "top" if dy < -3 else "center"
            x0 = cx + dx * scale - {"left": 0, "right": width, "center": width / 2}[ha]
            y0 = cy + dy * scale - {"bottom": 0, "top": height, "center": height / 2}[va]
            box = Bbox.from_extents(x0 - 4, y0 - 4, x0 + width + 4, y0 + height + 4)
            inside = (
                area.x0 <= box.x0
                and box.x1 <= area.x1
                and area.y0 <= box.y0
                and box.y1 <= area.y1
            )
            if inside and not any(box.overlaps(other) for other in taken):
                far = math.hypot(dx, dy) > 24
                ax.annotate(
                    text,
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=fontsize,
                    zorder=6,
                    annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=(
                        dict(arrowstyle="-", color="0.45", lw=1.0, shrinkA=0, shrinkB=9)
                        if far
                        else None
                    ),
                )
                taken.append(box)
                break
        else:
            skipped.append(text)
    return skipped


def plot_tradeoff(test, cost, filename, xlabel, what, output, selected_name):
    data = test.dropna(subset=[cost, "top1"]).copy()
    if data.empty:
        print(f"Skipping {filename}: no '{cost}' values")
        return
    data["label"] = data.apply(short_label, axis=1)
    pareto = frontier(data, cost)
    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(
            pareto[cost],
            pareto["top1"] * 100,
            color="0.35",
            linestyle="--",
            linewidth=2,
            zorder=2,
        )
        for _, row in data.iterrows():
            ax.scatter(
                row[cost],
                row["top1"] * 100,
                s=230,
                marker="D" if row["method"] == "reference" else ARCH_MARKERS.get(row["architecture"], "X"),
                color=METHOD_COLORS.get(row["method"], "0.5"),
                edgecolor="black",
                linewidth=0.9,
                zorder=3,
            )
        chosen = data[data["name"] == selected_name]
        if not chosen.empty:
            ax.scatter(
                chosen[cost],
                chosen["top1"] * 100,
                s=900,
                facecolors="none",
                edgecolors="black",
                linewidths=2.5,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.10, y=0.12)
        ax.grid(True, which="major", alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test top-1 accuracy (%)")
        ax.set_title(f"Food-101: accuracy vs {what} (test split)")

        methods = [m for m in METHOD_ORDER if m in set(data["method"])]
        handles = [
            Line2D([], [], marker="o", linestyle="", markersize=13, markeredgecolor="black",
                   color=METHOD_COLORS[m], label=METHOD_LABELS[m])
            for m in methods
        ]
        handles += [
            Line2D([], [], marker=marker, linestyle="", markersize=13, markeredgecolor="black",
                   color="0.8", label=ARCH_NAMES[arch])
            for arch, marker in ARCH_MARKERS.items()
            if arch in set(data["architecture"])
        ]
        handles.append(Line2D([], [], color="0.35", linestyle="--", linewidth=2, label="Pareto frontier"))
        if not chosen.empty:
            handles.append(
                Line2D([], [], marker="o", linestyle="", markersize=18, markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=2.5, label="Selected on validation")
            )
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=13, frameon=False)
        fig.tight_layout(rect=(0, 0.11, 1, 1))

        priority = data.assign(
            rank=(
                (data["method"] != "reference").astype(int) * 3
                + (data["name"] != selected_name).astype(int) * 2
                + (~data["name"].isin(pareto["name"])).astype(int)
            )
        ).sort_values(["rank", "top1"], ascending=[True, False])
        skipped = place_labels(
            fig, ax, [(r[cost], r["top1"] * 100, r["label"]) for _, r in priority.iterrows()]
        )
        if skipped:
            print(f"{filename}: no free space for labels {skipped}; see the summary table")
        fig.savefig(output / filename, dpi=SAVE_DPI)
        plt.close(fig)


def summarise(test, validation, reference, selected_name):
    ref = test[test["name"] == reference].iloc[0]
    table = test.merge(
        validation[["name", "top1"]].rename(columns={"top1": "val_top1"}), on="name", how="left"
    ).rename(columns={"top1": "test_top1", "top5": "test_top5"})
    table["drop_vs_reference_pp"] = (ref["top1"] - table["test_top1"]) * 100
    table["size_reduction_x"] = ref["size_mb"] / table["size_mb"]
    table["speedup_x"] = ref["latency_ms"] / table["latency_ms"]
    table["selected"] = table["name"] == selected_name
    if "parameters" in table:
        table["parameters_m"] = table["parameters"] / 1e6
    columns = [
        "name", "method", "architecture", "parameters_m", "size_mb", "gzip_mb", "sparsity",
        "latency_ms", "val_top1", "test_top1", "test_top5", "drop_vs_reference_pp",
        "size_reduction_x", "speedup_x", "selected",
    ]
    return table[[c for c in columns if c in table]].sort_values("size_mb")


def report(path: Path, output: Path, reference: str, max_drop: float = MAX_DROP, cost: str = "size_mb"):
    results = pd.read_csv(path)
    if not REQUIRED.issubset(results.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED - set(results.columns))}")
    if cost not in results.columns:
        raise ValueError(f"Unknown cost column: {cost}")
    if results.duplicated(["name", "split"]).any():
        raise ValueError("Duplicate model/split rows")
    if results[["threads", "device", "seed"]].drop_duplicates().shape[0] != 1:
        raise ValueError("All results must use the same CPU, thread count and split seed")
    test = results[results["split"] == "test"].copy()
    validation = results[results["split"] == "validation"].copy()
    paired = validation.merge(test, on="name", suffixes=("_val", "_test"))
    if (paired["artifact_val"] != paired["artifact_test"]).any():
        raise ValueError("Validation and test results refer to different model artifacts")
    if test.empty:
        raise ValueError("No test measurements")
    if len(test[test["name"] == reference]) != 1:
        raise ValueError("Exactly one test measurement of the reference is required")

    candidates = validation[validation["name"] != reference]
    if candidates.empty:
        raise ValueError("No validation measurements of candidate models")
    best = candidates["top1"].max()
    threshold = best - max_drop
    eligible = candidates[candidates["top1"] >= threshold]
    chosen = eligible.sort_values([cost, "top1", "name"], ascending=[True, False, True]).iloc[0]
    print(
        f"Selection on validation: best candidate top-1={best:.4f}, tolerance={max_drop:.4f}, "
        f"threshold={threshold:.4f}; {len(eligible)} eligible, cheapest by {cost}."
    )
    print(
        f"Selected: {chosen['name']} (validation top-1={chosen['top1']:.4f}, {cost}={chosen[cost]:.2f})"
    )
    selected_test = test[test["name"] == chosen["name"]]
    if selected_test.empty:
        print("The selected model has not been measured on the test split yet.")
    else:
        score = float(selected_test.iloc[0]["top1"])
        ref_score = float(test[test["name"] == reference].iloc[0]["top1"])
        print(
            f"Selected, test top-1={score:.4f} vs reference test top-1={ref_score:.4f} "
            f"(drop {100 * (ref_score - score):.2f} pp)"
        )

    output.mkdir(parents=True, exist_ok=True)
    summary = summarise(test, validation, reference, chosen["name"])
    summary.to_csv(output / "task4_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    set_style()
    for column, filename, xlabel, what in COSTS:
        if column in test.columns:
            plot_tradeoff(test, column, filename, xlabel, what, output, chosen["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results/task4_comparison.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--reference", default="swin")
    parser.add_argument("--max-drop", type=float, default=MAX_DROP,
                        help="allowed validation top-1 loss versus the best candidate (fraction, default 0.01)")
    parser.add_argument("--cost", default="size_mb", choices=("size_mb", "latency_ms", "gzip_mb", "parameters"),
                        help="what to minimise among eligible models")
    args = parser.parse_args()
    if not 0 <= args.max_drop < 1:
        raise ValueError("--max-drop must be in [0, 1)")
    report(args.results, args.output, args.reference, args.max_drop, args.cost)


if __name__ == "__main__":
    main()