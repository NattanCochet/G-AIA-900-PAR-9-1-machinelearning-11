"""T3-10: before/after table of a fine-tuned HQ 3, from the per-clip SDR CSV.

    uv run python -m task3_uvr.finetune_report \
        --model HQ3_ft_musdb7_phone_lr1e-06_s2000 [--split validation]

For every dataset the fine-tuned model was measured on (its training
dataset, then clean musdb7), it pairs each clip and stem with the original
HQ 3 on the same clip, and reports both medians, the median and mean of the
per-clip difference (fine-tuned - original) and the number of clips that
improve. The original's rows must come from the same SDR backend:
fine-tuned rows with variant "onnx" are paired with HQ 3 rows of variant ""
(onnxruntime, as T3-04 and T3-07 measured them), "torch" with "torch".

The table is printed in Markdown and saved to
results/task3_finetune_before_after.csv (rows of the same model and split
are replaced). Validation rows also carry T3-07's median for the original
HQ 3; on the full 14-clip split, the "before" median must match it.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HQ3_NAME = "UVR-MDX-NET-Inst_HQ_3"
CLEAN = "musdb7"
STEMS = ("vocals", "accompaniment")
# SDR backend of the fine-tuned rows -> `variant` of the original HQ 3's rows.
BASELINE_VARIANT = {"onnx": "", "torch": "torch"}
CLIPS_CSV = Path("results") / "task3_sdr_clips.csv"
TABLE_CSV = Path("results") / "task3_finetune_before_after.csv"
# T3-07: median validation SDR (dB) of the original HQ 3, onnxruntime.
T3_07_VALIDATION = {
    "musdb7_phone": {"vocals": 10.47, "accompaniment": 8.82},
    "musdb7_narrowband": {"vocals": 8.60, "accompaniment": 6.08},
    "musdb7": {"vocals": 10.57, "accompaniment": 14.26},
}
N_VALIDATION_CLIPS = 14
T3_07_TOLERANCE_DB = 0.01  # T3-07 rounded to 2 decimals


def read_clip_rows(path: Path) -> pd.DataFrame:
    """Per-clip rows as text, with `dataset` ("musdb7") and `variant` ("")
    added to rows written before those columns existed, and sdr_db as a
    float."""
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "dataset" not in rows.columns:
        rows["dataset"] = CLEAN
    if "variant" not in rows.columns:
        rows["variant"] = ""
    rows["sdr_db"] = rows["sdr_db"].astype(float)
    return rows


def before_after(
    rows: pd.DataFrame, model: str, split: str = "validation"
) -> pd.DataFrame:
    """One row per dataset and stem: paired SDR of the original HQ 3 and of
    `model` on the same clips."""
    rows = rows[(rows["split"] == split) & (rows["match_freq_cut"] == "False")]
    after = rows[rows["model"] == model]
    if after.empty:
        raise ValueError(f"no {split} rows for {model}")
    variants = after["variant"].unique()
    if len(variants) != 1 or variants[0] not in BASELINE_VARIANT:
        raise ValueError(f"{model}: expected one SDR backend, found {list(variants)}")
    base_variant = BASELINE_VARIANT[variants[0]]
    before = rows[(rows["model"] == HQ3_NAME) & (rows["variant"] == base_variant)]

    table = []
    # The training dataset first, clean musdb7 last.
    for dataset in sorted(after["dataset"].unique(), key=lambda d: (d == CLEAN, d)):
        b = before[before["dataset"] == dataset]
        if b.empty:
            raise ValueError(
                f"no {split} rows for {HQ3_NAME} on {dataset} (variant {base_variant!r})"
            )
        paired = after[after["dataset"] == dataset].merge(
            b, on=["clip", "stem"], suffixes=("_after", "_before")
        )
        for stem in STEMS:
            s = paired[paired["stem"] == stem]
            diff = s["sdr_db_after"] - s["sdr_db_before"]
            table.append(
                {
                    "model": model,
                    "split": split,
                    "dataset": dataset,
                    "stem": stem,
                    "n_clips": len(s),
                    "before_median_db": s["sdr_db_before"].median(),
                    "after_median_db": s["sdr_db_after"].median(),
                    "median_difference_db": diff.median(),
                    "mean_difference_db": diff.mean(),
                    "n_improved": int((diff > 0).sum()),
                    "t3_07_before_db": (
                        T3_07_VALIDATION.get(dataset, {}).get(stem, float("nan"))
                        if split == "validation" and base_variant == ""
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(table)


def t3_07_mismatches(table: pd.DataFrame) -> list[str]:
    """Rows on the full validation split whose "before" median is not T3-07's:
    the original HQ 3 was then not measured as in T3-07."""
    full = table[
        (table["n_clips"] == N_VALIDATION_CLIPS) & table["t3_07_before_db"].notna()
    ]
    return [
        f"{r.dataset} {r.stem}: before {r.before_median_db:.2f} dB, "
        f"T3-07 {r.t3_07_before_db:.2f} dB"
        for r in full.itertuples()
        if abs(r.before_median_db - r.t3_07_before_db) > T3_07_TOLERANCE_DB
    ]


def to_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Data | Stem | HQ 3 original | Fine-tuned | Difference, median (mean) | Clips improved |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in table.itertuples():
        data = f"{r.dataset} (clean)" if r.dataset == CLEAN else r.dataset
        lines.append(
            f"| {data} | {r.stem} | {r.before_median_db:.2f} | {r.after_median_db:.2f} | "
            f"{r.median_difference_db:+.2f} ({r.mean_difference_db:+.2f}) | "
            f"{r.n_improved}/{r.n_clips} |"
        )
    return "\n".join(lines)


def save_table(table: pd.DataFrame, path: Path) -> None:
    """Add the table to `path`, replacing earlier rows of the same model and
    split."""
    if path.exists():
        old = pd.read_csv(path)
        keep = ~(old["model"].isin(table["model"]) & old["split"].isin(table["split"]))
        table = pd.concat([old[keep], table], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model", required=True, help="fine-tuned model name (ONNX stem)"
    )
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--clips", type=Path, default=CLIPS_CSV)
    parser.add_argument("--output", type=Path, default=TABLE_CSV)
    args = parser.parse_args(argv)
    table = before_after(read_clip_rows(args.clips), args.model, args.split)
    save_table(table, args.output)
    print(f"{args.model}, {args.split} (global SDR, dB, median over clips):\n")
    print(to_markdown(table))
    print("->", args.output)
    for line in t3_07_mismatches(table):
        print("WARNING, the original HQ 3 differs from T3-07:", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
