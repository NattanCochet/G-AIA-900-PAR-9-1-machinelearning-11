"""T3-04: SDR of MDX-Net ONNX models on MUSDB18-7, outside the UVR GUI.

    uv run python -m task3_uvr.evaluate \
        --models UVR-MDX-NET-Inst_HQ_3 Benchmark_CentreChannel \
        --splits validation test

Models are ONNX paths, or names in <UVR>/models/MDX_Net_Models/. The UVR
folder (only read, for the models and their settings) comes from --uvr-dir or
ML_TEK5_UVR_DIR, the musdb7 folder from --data-dir or ML_TEK5_MUSDB7_DIR.
The validation and train splits are task3_uvr/splits/*.json (clips from
musdb7/train/); the test split is every clip in musdb7/test/.

Each model separates every clip as UVR does (task3_uvr.mdx_inference), and
the "mixture" reference uses the mixture itself as the estimate of both
stems. The global SDR (task3_uvr.sdr) is computed per stem, named "vocals"
and "accompaniment" whatever the model's primary stem.

Outputs:
- results/task3_sdr_clips.csv: one row per model, split, clip and stem.
  Rows of a previous run with the same model, split and settings are
  replaced.
- results/task3_uvr.csv: median and mean SDR per model, split and stem
  (common.log_result).

Use the validation split for model selection; the test split is for final
reporting only.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import log_result
from task3_uvr.check_data import resolve_data_dir
from task3_uvr.compare_gui import read_stereo, resolve_model, resolve_uvr_dir
from task3_uvr.mdx_inference import load_settings, model_data_dir, onnx_model, separate
from task3_uvr.sdr import EPS, sdr

CLIPS_CSV = Path("results") / "task3_sdr_clips.csv"
SUMMARY_CSV = Path("results") / "task3_uvr.csv"
SPLITS_DIR = Path(__file__).parent / "splits"
STEMS = ("vocals", "accompaniment")
MIXTURE = "mixture"
KEY = ["model", "split", "match_freq_cut"]


def split_clips(data_dir: Path, split: str) -> list[Path]:
    """Clip folders of a split: validation/train from the JSON lists, test
    from the official folder."""
    if split == "test":
        return sorted(p for p in (data_dir / "test").iterdir() if p.is_dir())
    names = json.loads((SPLITS_DIR / f"{split}.json").read_text())["clips"]
    return [data_dir / "train" / name for name in names]


def evaluate_split(name: str, estimator, clips: list[Path]) -> tuple[list[dict], float]:
    """Rows for one model on one split, and the total separation time."""
    rows, total = [], 0.0
    for i, clip in enumerate(clips, 1):
        mix = read_stereo(clip / "mixture.wav")
        start = time.perf_counter()
        estimates = estimator(mix)
        seconds = time.perf_counter() - start
        total += seconds
        for stem in STEMS:
            reference = read_stereo(clip / f"{stem}.wav")
            rows.append(
                {
                    "clip": clip.name,
                    "stem": stem,
                    "sdr_db": sdr(reference, estimates[stem]),
                    "reference_energy": float(
                        np.sum(reference.astype(np.float64) ** 2)
                    ),
                    "seconds": seconds,
                }
            )
        print(f"  {name} {i}/{len(clips)} {clip.name}: {seconds:.1f} s", flush=True)
    return rows, total


def save_clip_rows(new: pd.DataFrame, path: Path) -> None:
    if path.exists():
        old = pd.read_csv(path)
        done = new[KEY].drop_duplicates()
        old = old.merge(done, on=KEY, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="*", default=[], help="ONNX paths or names")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--uvr-dir")
    parser.add_argument("--data-dir")
    parser.add_argument(
        "--match-freq-cut",
        action="store_true",
        help="secondary stem as with UVR's Match Freq Cut (not checked against the GUI)",
    )
    parser.add_argument(
        "--no-mixture", action="store_true", help="skip the mixture reference"
    )
    args = parser.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    estimators = {}  # name -> (hash, settings dict, estimator)
    if not args.no_mixture:
        estimators[MIXTURE] = ("", {}, lambda mix: {s: mix for s in STEMS})
    if args.models:
        uvr_dir = resolve_uvr_dir(args.uvr_dir)
    for name in args.models:
        path = resolve_model(name, uvr_dir)
        model_hash, settings = load_settings(path, model_data_dir(uvr_dir))
        model = onnx_model(path, settings)
        estimators[path.stem] = (
            model_hash,
            {
                "n_fft": settings.n_fft,
                "dim_f": settings.dim_f,
                "segment_size": settings.dim_t,
                "overlap": "Default",
                "compensate": settings.compensate,
                "primary_stem": settings.primary_stem,
            },
            lambda mix, m=model, s=settings: separate(
                mix, m, s, match_freq_cut=args.match_freq_cut
            ),
        )

    date = datetime.date.today().isoformat()
    for split in args.splits:
        clips = split_clips(data_dir, split)
        for name, (model_hash, settings, estimator) in estimators.items():
            print(f"{name} on {split} ({len(clips)} clips)")
            rows, total = evaluate_split(name, estimator, clips)
            frame = pd.DataFrame(rows).assign(
                model=name,
                model_hash=model_hash,
                split=split,
                match_freq_cut=args.match_freq_cut,
                date=date,
            )
            frame = frame[
                [
                    "date",
                    "model",
                    "model_hash",
                    "split",
                    "match_freq_cut",
                    "clip",
                    "stem",
                    "sdr_db",
                    "reference_energy",
                    "seconds",
                ]
            ]
            save_clip_rows(frame, CLIPS_CSV)

            config = {
                "model": name,
                "model_hash": model_hash,
                "n_clips": len(clips),
                "sdr": "global (MDX Challenge)",
                "eps": EPS,
                "match_freq_cut": args.match_freq_cut,
                **settings,
            }
            for stem in STEMS:
                values = frame.loc[frame["stem"] == stem, "sdr_db"]
                for stat, value in (
                    ("median", values.median()),
                    ("mean", values.mean()),
                ):
                    log_result(
                        SUMMARY_CSV,
                        task="task3_uvr",
                        seed=0,
                        split=split,
                        metric=f"sdr_{stat}_{stem}",
                        score=float(value),
                        duration_seconds=total,
                        config=config,
                    )
                print(
                    f"{name} {split} {stem}: median {values.median():.2f} dB, "
                    f"mean {values.mean():.2f} dB"
                )
    print("->", CLIPS_CSV, SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
