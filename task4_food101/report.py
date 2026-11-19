"""Plot Food-101 trade-offs and select using validation, not candidate test scores."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.plotting import set_style

MAX_DROP = 0.02


def frontier(frame: pd.DataFrame, cost: str) -> pd.DataFrame:
    ordered = frame.sort_values([cost, "top1"], ascending=[True, False])
    return ordered[ordered["top1"] > ordered["top1"].cummax().shift(fill_value=-1)]


def report(path: Path, output: Path, reference: str):
    results = pd.read_csv(path)
    required = {
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
    if not required.issubset(results.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(results.columns))}")
    if results.duplicated(["name", "split"]).any():
        raise ValueError("Duplicate model/split rows")
    if results[["threads", "device", "seed"]].drop_duplicates().shape[0] != 1:
        raise ValueError(
            "All results must use the same CPU, thread count and split seed"
        )
    test = results[results["split"] == "test"].copy()
    validation = results[results["split"] == "validation"].copy()
    paired = validation.merge(test, on="name", suffixes=("_val", "_test"))
    if (paired["artifact_val"] != paired["artifact_test"]).any():
        raise ValueError(
            "Validation and test results refer to different model artifacts"
        )
    ref = test[test["name"] == reference]
    if len(ref) != 1:
        raise ValueError("Exactly one test measurement of the reference is required")
    threshold = float(ref.iloc[0]["top1"]) - MAX_DROP
    candidates = validation[validation["name"] != reference]
    candidates = candidates[candidates["top1"] >= threshold]
    if candidates.empty:
        print(
            f"No validation candidate reaches the predeclared threshold {threshold:.4f}"
        )
    else:
        chosen = candidates.sort_values(["size_mb", "name"]).iloc[0]
        print(
            f"Selected from validation: {chosen['name']} ({chosen['size_mb']:.2f} MB); threshold={threshold:.4f}"
        )
        selected_test = test[test["name"] == chosen["name"]]
        if selected_test.empty:
            print("Selected model has not yet been measured on the test set.")
        else:
            score = float(selected_test.iloc[0]["top1"])
            print(
                f"Selected test top-1={score:.4f}; meets threshold: {score >= threshold}"
            )
    if test.empty:
        raise ValueError("No test measurements")
    output.mkdir(parents=True, exist_ok=True)
    set_style()
    for cost, filename, xlabel in (
        ("size_mb", "task4_accuracy_size.png", "Model size (MB)"),
        ("latency_ms", "task4_accuracy_latency.png", "CPU latency (ms/image)"),
    ):
        fig, ax = plt.subplots()
        ax.scatter(test[cost], test["top1"] * 100)
        for _, row in test.iterrows():
            ax.annotate(
                row["name"],
                (row[cost], row["top1"] * 100),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        pareto = frontier(test, cost)
        ax.plot(
            pareto[cost], pareto["top1"] * 100, color="tab:red", label="Pareto frontier"
        )
        ax.axhline(
            threshold * 100, color="grey", linestyle="--", label="Reference - 2 pp"
        )
        if cost == "size_mb":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test top-1 accuracy (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/task4_comparison.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--reference", default="swin")
    args = parser.parse_args()
    report(args.results, args.output, args.reference)


if __name__ == "__main__":
    main()
