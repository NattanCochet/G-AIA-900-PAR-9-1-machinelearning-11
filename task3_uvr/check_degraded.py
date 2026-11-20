"""T3-06: check degraded copies of MUSDB18-7 against the clean clips.

Run from the repository root:

    uv run python -m task3_uvr.check_degraded phone narrowband

The musdb7 folder comes from --data-dir or ML_TEK5_MUSDB7_DIR, and each
variant is read from <musdb7 parent>/musdb7_<variant>/. For every clip it
checks:

- the clip names are the same as in musdb7, split by split;
- the three WAV files are 44100 Hz stereo, with the clean clip's length;
- degradation.json is present and holds the settings;
- |mixture - vocals - accompaniment| <= 2 LSB (each file is rounded to
  16 bits on its own, so up to 1.5 LSB is expected);
- no sample reaches 16-bit full scale (that would mean clipping).

It also measures how much quieter each stem is than its clean version. A
summary per variant goes to results/task3_degraded_data.json; the exit code is
1 if any clip has an anomaly. The audio is only read.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from task3_uvr.check_data import (
    FULL_SCALE,
    SPLITS,
    STEMS,
    check_clip,
    list_clips,
    resolve_data_dir,
    to_db,
)
from task3_uvr.degradation import LSB, RESULTS_PATH, VARIANTS, variant_dir
from task3_uvr.onnx_checks import update_results

MAX_ABS_RESIDUAL = 2 * LSB


def energy(path: Path) -> tuple[float, float]:
    """Total energy and peak absolute value of a WAV file."""
    audio, _ = sf.read(str(path), dtype="float64", always_2d=True)
    return float(np.sum(audio**2)), float(np.max(np.abs(audio)))


def check_degraded_clip(
    clean_dir: Path, degraded_dir: Path, settings: list[str]
) -> dict:
    """Check one degraded clip folder against its clean version."""
    report = check_clip(
        degraded_dir,
        max_abs_residual=MAX_ABS_RESIDUAL,
        allowed_extra=("degradation.json",),
    )
    anomalies = report["anomalies"]

    json_path = degraded_dir / "degradation.json"
    if not json_path.is_file():
        anomalies.append("missing degradation.json")
    else:
        try:
            saved = json.loads(json_path.read_text())
            report["settings"] = {k: saved[k] for k in settings}
        except (json.JSONDecodeError, KeyError) as error:
            anomalies.append(f"degradation.json: {error!r}")

    if "frames" not in report:  # a stem is missing: nothing more to measure
        return report
    clean_frames = sf.info(str(clean_dir / "mixture.wav")).frames
    if report["frames"] != clean_frames:
        anomalies.append(f"{report['frames']} frames, clean clip has {clean_frames}")

    report["peak_abs"], report["level_drop_db"] = {}, {}
    for stem in STEMS:
        clean_energy, _ = energy(clean_dir / f"{stem}.wav")
        degraded_energy, peak = energy(degraded_dir / f"{stem}.wav")
        report["peak_abs"][stem] = peak
        if peak >= FULL_SCALE:
            anomalies.append(f"{stem} reaches full scale")
        if clean_energy > 0 and degraded_energy > 0:
            report["level_drop_db"][stem] = to_db(clean_energy / degraded_energy)
    return report


def check_variant(data_dir: Path, variant: str) -> dict:
    """Check every clip of a degraded copy of musdb7."""
    root = variant_dir(data_dir, variant)
    settings = list(VARIANTS[variant])
    summary = {"folder": root.name, "splits": {}, "anomalies": []}
    observed = {k: [] for k in settings}

    for split in SPLITS:
        clean_clips, _ = list_clips(data_dir / split)
        if not (root / split).is_dir():
            summary["anomalies"].append(f"{split}/ is missing")
            continue
        clips, others = list_clips(root / split)
        if others:
            summary["anomalies"].append(f"{split}/: unexpected files {others}")
        missing = sorted(set(clean_clips) - set(clips))
        extra = sorted(set(clips) - set(clean_clips))
        if missing:
            summary["anomalies"].append(f"{split}/: missing clips {missing}")
        if extra:
            summary["anomalies"].append(f"{split}/: clips not in musdb7 {extra}")

        reports = [
            check_degraded_clip(data_dir / split / c, root / split / c, settings)
            for c in sorted(set(clips) & set(clean_clips))
        ]
        for r in reports:
            summary["anomalies"] += [
                f"{split}/{r['clip']}: {a}" for a in r["anomalies"]
            ]
            for k, v in r.get("settings", {}).items():
                observed[k].append(v)
        measured = [r for r in reports if "max_abs_residual" in r]
        worst = max(measured, key=lambda r: r["max_abs_residual"], default=None)
        summary["splits"][split] = {
            "n_clips": len(clips),
            "n_clips_with_anomalies": sum(bool(r["anomalies"]) for r in reports),
            "max_abs_residual": worst and worst["max_abs_residual"],
            "max_abs_residual_lsb": worst and worst["max_abs_residual"] / LSB,
            "max_abs_residual_clip": worst and worst["clip"],
            "max_peak_abs": {
                s: max(
                    (r["peak_abs"][s] for r in reports if "peak_abs" in r), default=None
                )
                for s in STEMS
            },
            "median_level_drop_db": {
                s: float(
                    np.median(
                        [
                            r["level_drop_db"][s]
                            for r in reports
                            if s in r.get("level_drop_db", {})
                        ]
                    )
                )
                for s in STEMS
            },
        }

    summary["ranges"] = {k: list(v) for k, v in VARIANTS[variant].items()}
    summary["observed_ranges"] = {
        k: [min(v), max(v)] if v else None for k, v in observed.items()
    }
    summary["ok"] = not summary["anomalies"]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("variants", nargs="+", choices=sorted(VARIANTS))
    parser.add_argument(
        "--data-dir", help="musdb7 folder (default: $ML_TEK5_MUSDB7_DIR)"
    )
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    all_ok = True
    for variant in args.variants:
        summary = check_variant(data_dir, variant)
        results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
        entry = results.get(variant, {})
        entry["check"] = {
            "date": datetime.date.today().isoformat(),
            "max_abs_residual_allowed": MAX_ABS_RESIDUAL,
            **summary,
        }
        update_results(RESULTS_PATH, variant, entry)

        for split, s in summary["splits"].items():
            drops = ", ".join(
                f"{k} {v:.1f} dB" for k, v in s["median_level_drop_db"].items()
            )
            print(
                f"{variant} {split}: {s['n_clips']} clips, "
                f"{s['n_clips_with_anomalies']} with anomalies, "
                f"max residual {s['max_abs_residual_lsb']:.2f} LSB, "
                f"max peak {max(s['max_peak_abs'].values()):.3f}, "
                f"median level drop: {drops}"
            )
        for a in summary["anomalies"]:
            print("ANOMALY:", a)
        all_ok &= summary["ok"]
    print("OK" if all_ok else "FAILED", "->", RESULTS_PATH)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
