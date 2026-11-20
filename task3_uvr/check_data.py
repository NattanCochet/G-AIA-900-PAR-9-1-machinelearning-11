"""T3-03: check the MUSDB18-7 WAV clips made by the course's 0_download_data.py.

Run from the repository root:

    uv run python -m task3_uvr.check_data <musdb7>

where <musdb7> holds train/ and test/, with one folder per clip containing
mixture.wav, vocals.wav and accompaniment.wav. Without the argument, the
folder is read from the ML_TEK5_MUSDB7_DIR environment variable.

For every clip it checks that the three files exist, are 44100 Hz stereo and
have the same length, reports the duration, and checks that
mixture = vocals + accompaniment up to WAV quantisation. It also lists the
artists found in both the official training and test sets. The audio is only
read. A summary goes to results/task3_data_check.json; the exit code is 1 if
any clip has an anomaly.
"""

import argparse
import datetime
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from task3_uvr.onnx_checks import update_results

ENV_VAR = "ML_TEK5_MUSDB7_DIR"
RESULTS_PATH = Path("results") / "task3_data_check.json"
SPLITS = ("train", "test")
STEMS = ("mixture", "vocals", "accompaniment")
EXPECTED_COUNTS = {"train": 94, "test": 50}  # announced by 0_download_data.py
SAMPLE_RATE = 44100
CHANNELS = 2
# accompaniment.wav is written as mixture - vocals, so the residual
# mixture - vocals - accompaniment is only the rounding of three 16-bit files:
# at most 3 * 0.5 / 32768 = 4.6e-5. Anything above these limits means a
# truncated, clipped or mismatched stem.
MAX_ABS_RESIDUAL = 1e-3
MAX_RELATIVE_RESIDUAL_ENERGY = 1e-6  # -60 dB
# Largest 16-bit sample value: a mixture reaching it is probably clipped.
FULL_SCALE = 32767 / 32768


def resolve_data_dir(arg: str | None) -> Path:
    """Return the data folder from the argument, else from ML_TEK5_MUSDB7_DIR."""
    value = arg or os.environ.get(ENV_VAR)
    if not value:
        sys.exit(f"Give the musdb7 folder as an argument or set {ENV_VAR}.")
    path = Path(value).expanduser()
    if not path.is_dir():
        sys.exit(f"Not a folder: {path}")
    return path


def artist_of(clip_name: str) -> str:
    """Artist of a MUSDB clip folder named '<artist> - <title>'.

    Guests and remixers are dropped, so 'Triviul feat. The Fiend' and
    'The Easton Ellises (Baumi)' count as 'Triviul' and 'The Easton Ellises'.
    """
    artist = clip_name.split(" - ", 1)[0]
    artist = re.split(r"\s+(?:feat\.|ft\.|featuring)\s", artist, maxsplit=1)[0]
    artist = re.sub(r"\s*\([^)]*\)$", "", artist)
    return artist.strip()


def list_clips(split_dir: Path) -> tuple[list[str], list[str]]:
    """Clip folder names in a split, and any other entries found there."""
    clips, others = [], []
    for entry in sorted(split_dir.iterdir()):
        (clips if entry.is_dir() else others).append(entry.name)
    return clips, others


def check_clip(
    clip_dir: Path,
    max_abs_residual: float = MAX_ABS_RESIDUAL,
    allowed_extra: tuple[str, ...] = (),
) -> dict:
    """Check one clip folder; 'anomalies' is empty when everything is fine.

    max_abs_residual: limit on |mixture - vocals - accompaniment|.
    allowed_extra: other file names expected in the folder.
    """
    report = {"clip": clip_dir.name, "anomalies": []}
    anomalies = report["anomalies"]

    missing = [s for s in STEMS if not (clip_dir / f"{s}.wav").is_file()]
    if missing:
        anomalies.append(f"missing {', '.join(missing)}")
        return report
    expected = {f"{s}.wav" for s in STEMS} | set(allowed_extra)
    extra = sorted(p.name for p in clip_dir.iterdir() if p.name not in expected)
    if extra:
        anomalies.append(f"unexpected files {extra}")

    audio, infos = {}, {}
    for stem in STEMS:
        path = clip_dir / f"{stem}.wav"
        infos[stem] = sf.info(str(path))
        audio[stem], _ = sf.read(str(path), dtype="float64", always_2d=True)

    report["subtypes"] = sorted({i.subtype for i in infos.values()})
    rates = {s: i.samplerate for s, i in infos.items()}
    channels = {s: i.channels for s, i in infos.items()}
    frames = {s: i.frames for s, i in infos.items()}
    if any(r != SAMPLE_RATE for r in rates.values()):
        anomalies.append(f"sample rate {rates}")
    if any(c != CHANNELS for c in channels.values()):
        anomalies.append(f"channels {channels}")
    if len(set(frames.values())) != 1:
        anomalies.append(f"lengths differ {frames}")
    report["frames"] = frames["mixture"]
    report["duration_s"] = frames["mixture"] / infos["mixture"].samplerate

    for stem, x in audio.items():
        if not np.all(np.isfinite(x)):
            anomalies.append(f"{stem} has NaN or inf")
    if anomalies:  # the residual needs equal shapes and finite values
        return report

    mix = audio["mixture"]
    residual = mix - audio["vocals"] - audio["accompaniment"]
    mix_energy = float(np.sum(mix**2))
    res_energy = float(np.sum(residual**2))
    report["max_abs_residual"] = float(np.max(np.abs(residual)))
    if mix_energy == 0.0:
        anomalies.append("silent mixture")
        report["relative_residual_energy"] = None
    else:
        ratio = res_energy / mix_energy
        report["relative_residual_energy"] = ratio
        if ratio > MAX_RELATIVE_RESIDUAL_ENERGY:
            anomalies.append(f"residual energy {to_db(ratio):.1f} dB of mixture")
    if report["max_abs_residual"] > max_abs_residual:
        anomalies.append(f"max abs residual {report['max_abs_residual']:.2e}")
    report["peak_abs_mixture"] = float(np.max(np.abs(mix)))
    return report


def to_db(ratio: float) -> float:
    return float(10 * np.log10(ratio)) if ratio > 0 else float("-inf")


def artist_overlap(train_clips: list[str], test_clips: list[str]) -> dict:
    """Artists present in both splits, with their clips in each."""
    train_artists = {artist_of(c) for c in train_clips}
    test_artists = {artist_of(c) for c in test_clips}
    shared = sorted(train_artists & test_artists)
    return {
        "n_train_artists": len(train_artists),
        "n_test_artists": len(test_artists),
        "shared_artists": {
            a: {
                "train": [c for c in train_clips if artist_of(c) == a],
                "test": [c for c in test_clips if artist_of(c) == a],
            }
            for a in shared
        },
    }


def run_check(data_dir: Path) -> dict:
    """Check every clip of both splits and return the summary."""
    summary = {"splits": {}, "anomalies": []}
    clips_by_split = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            summary["anomalies"].append(f"{split}/ is missing")
            clips_by_split[split] = []
            continue
        clips, others = list_clips(split_dir)
        clips_by_split[split] = clips
        if others:
            summary["anomalies"].append(f"{split}/: unexpected files {others}")
        if len(clips) != EXPECTED_COUNTS[split]:
            summary["anomalies"].append(
                f"{split}/: {len(clips)} clips, expected {EXPECTED_COUNTS[split]}"
            )

        reports = [check_clip(split_dir / c) for c in clips]
        for r in reports:
            summary["anomalies"] += [
                f"{split}/{r['clip']}: {a}" for a in r["anomalies"]
            ]
        measured = [r for r in reports if "max_abs_residual" in r]
        durations = [r["duration_s"] for r in reports if "duration_s" in r]
        worst = max(measured, key=lambda r: r["max_abs_residual"], default=None)
        summary["splits"][split] = {
            "n_clips": len(clips),
            "n_artists": len({artist_of(c) for c in clips}),
            "n_clips_with_anomalies": sum(bool(r["anomalies"]) for r in reports),
            "subtypes": sorted({s for r in reports for s in r.get("subtypes", [])}),
            "duration_s": {
                "min": min(durations, default=None),
                "max": max(durations, default=None),
                "total": sum(durations),
            },
            "max_abs_residual": worst and worst["max_abs_residual"],
            "max_abs_residual_clip": worst and worst["clip"],
            "max_relative_residual_energy": max(
                (r["relative_residual_energy"] for r in measured), default=None
            ),
            "max_peak_abs_mixture": max(
                (r["peak_abs_mixture"] for r in measured), default=None
            ),
            "clips_reaching_full_scale": [
                r["clip"] for r in measured if r["peak_abs_mixture"] >= FULL_SCALE
            ],
        }

    shared_names = sorted(set(clips_by_split["train"]) & set(clips_by_split["test"]))
    if shared_names:
        summary["anomalies"].append(f"clips in both train and test: {shared_names}")
    summary["artist_overlap"] = artist_overlap(
        clips_by_split["train"], clips_by_split["test"]
    )
    summary["ok"] = not summary["anomalies"]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "data_dir", nargs="?", help=f"musdb7 folder (default: ${ENV_VAR})"
    )
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    summary = run_check(data_dir)
    summary = {
        "date": datetime.date.today().isoformat(),
        "expected": {
            "counts": EXPECTED_COUNTS,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "max_abs_residual": MAX_ABS_RESIDUAL,
            "max_relative_residual_energy": MAX_RELATIVE_RESIDUAL_ENERGY,
        },
        **summary,
    }
    update_results(RESULTS_PATH, "musdb7", summary)

    for split, s in summary["splits"].items():
        d = s["duration_s"]
        print(
            f"{split}: {s['n_clips']} clips, {s['n_artists']} artists, "
            f"{s['n_clips_with_anomalies']} with anomalies, "
            f"duration {d['min']:.3f}-{d['max']:.3f} s, "
            f"max abs residual {s['max_abs_residual']:.2e}, "
            f"max residual energy {to_db(s['max_relative_residual_energy']):.1f} dB, "
            f"{len(s['clips_reaching_full_scale'])} mixtures reach full scale"
        )
    shared = summary["artist_overlap"]["shared_artists"]
    print(f"Artists in both train and test: {len(shared)} {sorted(shared)}")
    for a in summary["anomalies"]:
        print("ANOMALY:", a)
    print("OK" if summary["ok"] else "FAILED", "->", RESULTS_PATH)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
