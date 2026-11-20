"""T3-06: make degraded copies of MUSDB18-7 ("phone recording" and variants).

Port of the course files 2_3_uvr/uvr_my_models/5_degrade_data.py and
src/degradation.py. Each clip gets one random draw of settings, seeded by
the clip name: a synthetic room reverb, then a 4th-order Butterworth
band-pass. The chain is applied to the vocals and to the accompaniment
separately, and the degraded mixture is their sum, scaled with the same gain
as the stems. Both effects are linear, so mixture = vocals + accompaniment
still holds: the degraded stems are the new labels (the task is "separate a
phone recording", not "restore the clean sound").

Run from the repository root:

    uv run python -m task3_uvr.degradation make narrowband
    uv run python -m task3_uvr.degradation verify phone

The musdb7 folder comes from --data-dir or ML_TEK5_MUSDB7_DIR. A variant is
written to <musdb7 parent>/musdb7_<variant>/{train,test}/<clip>/ with
mixture.wav, vocals.wav, accompaniment.wav and degradation.json. Like the
course script, `make` skips clips that already have degradation.json, so it
never overwrites the course's musdb7_phone. `verify` recomputes a variant in
memory and compares it with the files on disk, sample by sample.
"""

import argparse
import datetime
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from task3_uvr.check_data import SPLITS, resolve_data_dir
from task3_uvr.onnx_checks import machine_name, package_versions, update_results

SR = 44100
SEED = 0
STEMS = ("vocals", "accompaniment")
RESULTS_PATH = Path("results") / "task3_degraded_data.json"
LSB = 1 / 32768
# Mixture peak after the common gain: leaves headroom below 16-bit full scale.
TARGET_PEAK = 0.9

# Ranges of the per-clip random settings, drawn uniformly in this order.
VARIANTS = {
    # The course's "phone recording" (src/degradation.py, PARAM_RANGES).
    "phone": {
        "rt60": (0.3, 1.0),  # reverb decay time in s (small room -> large room)
        "wet": (0.2, 0.5),  # share of reverberated sound
        "low_cut": (100.0, 300.0),  # Hz, phone microphones lose the low bass
        "high_cut": (3000.0, 5000.0),  # Hz, and the high frequencies
    },
    # A phone call: the fixed 300-3400 Hz telephone band, in a larger and
    # more reverberant room than "phone".
    "narrowband": {
        "rt60": (0.8, 1.5),
        "wet": (0.4, 0.7),
        "low_cut": (300.0, 300.0),
        "high_cut": (3400.0, 3400.0),
    },
}


def clip_seed(clip_name: str, variant: str) -> list[int]:
    """Seed of a clip's random generator.

    "phone" keeps the course seed [0, crc32(clip)], so it reproduces the
    course files. Other variants add crc32(variant), so their reverb is an
    independent draw rather than the phone reverb stretched.
    """
    seed = [SEED, zlib.crc32(clip_name.encode())]
    if variant != "phone":
        seed.append(zlib.crc32(variant.encode()))
    return seed


def draw_params(rng: np.random.Generator, ranges: dict) -> dict[str, float]:
    """One uniform draw per setting, in the order of `ranges`."""
    return {
        name: float(rng.uniform(low=lo, high=hi)) for name, (lo, hi) in ranges.items()
    }


def make_impulse_response(rt60: float, rng: np.random.Generator, channels: int = 2):
    """Synthetic room: exponentially decaying noise, independent per channel."""
    length = int(rt60 * 1.5 * SR)
    t = np.arange(length) / SR
    decay = np.exp(-6.91 * t / rt60)  # -60 dB after rt60 seconds
    ir = rng.standard_normal((length, channels)) * decay[:, None]
    ir /= np.sqrt(np.sum(ir**2, axis=0, keepdims=True))  # unit energy per channel
    return ir.astype(np.float32)


def band_pass(params: dict[str, float]) -> np.ndarray:
    """4th-order Butterworth band-pass, as second-order sections."""
    return signal.butter(
        N=4,
        Wn=[params["low_cut"], params["high_cut"]],
        btype="band",
        fs=SR,
        output="sos",
    )


def degrade(audio: np.ndarray, params: dict[str, float], ir: np.ndarray) -> np.ndarray:
    """Reverb, then band-pass. audio: [samples, channels], float32.

    Linear in `audio` for fixed params and ir, and the output keeps the input
    length (the reverb tail past the clip end is dropped).
    """
    n_samples, n_channels = audio.shape
    reverberated = np.stack(
        [
            signal.fftconvolve(audio[:, c], ir[:, c])[:n_samples]
            for c in range(n_channels)
        ],
        axis=1,
    )
    out = (1 - params["wet"]) * audio + params["wet"] * reverberated
    return np.asarray(signal.sosfilt(band_pass(params), out, axis=0), dtype=np.float32)


def read_float32(path: Path) -> np.ndarray:
    return sf.read(str(path), dtype="float32", always_2d=True)[0]


def degrade_clip(clip_dir: Path, variant: str) -> tuple[dict[str, np.ndarray], dict]:
    """Degraded stems and mixture of one clean clip, with their metadata.

    Same steps as the course script: one draw of settings, one impulse
    response shared by both stems, mixture = vocals + accompaniment, then one
    common gain (at most 1) so the loudest of the three peaks at 0.9.
    """
    ranges = VARIANTS[variant]
    seed = clip_seed(clip_dir.name, variant)
    rng = np.random.default_rng(seed)
    params = draw_params(rng, ranges)
    ir = make_impulse_response(params["rt60"], rng)

    stems = {s: degrade(read_float32(clip_dir / f"{s}.wav"), params, ir) for s in STEMS}
    n = min(len(a) for a in stems.values())
    stems = {s: a[:n] for s, a in stems.items()}
    stems["mixture"] = stems["vocals"] + stems["accompaniment"]
    gain = min(1.0, TARGET_PEAK / max(np.abs(a).max() for a in stems.values()))
    stems = {s: a * gain for s, a in stems.items()}

    metadata = {
        **params,
        "variant": variant,
        "ranges": {name: list(r) for name, r in ranges.items()},
        "seed": seed,
        "gain": float(gain),
    }
    return stems, metadata


def variant_dir(data_dir: Path, variant: str) -> Path:
    return data_dir.parent / f"{data_dir.name}_{variant}"


def make_variant(data_dir: Path, variant: str, overwrite: bool = False) -> dict:
    """Write the degraded copy of every clip; returns counts per split."""
    out_root = variant_dir(data_dir, variant)
    counts = {}
    for split in SPLITS:
        written = skipped = 0
        for clip in sorted(p for p in (data_dir / split).iterdir() if p.is_dir()):
            out = out_root / split / clip.name
            if (out / "degradation.json").exists() and not overwrite:
                skipped += 1
                continue
            stems, metadata = degrade_clip(clip, variant)
            out.mkdir(parents=True, exist_ok=True)
            for stem, audio in stems.items():
                sf.write(str(out / f"{stem}.wav"), audio, SR, subtype="PCM_16")
            # Written last: its presence marks a finished clip.
            (out / "degradation.json").write_text(json.dumps(metadata, indent=2) + "\n")
            written += 1
        counts[split] = {"written": written, "skipped": skipped}
        print(f"{out_root.name}/{split}: {written} written, {skipped} already done")
    return counts


def as_pcm16(audio: np.ndarray) -> np.ndarray:
    """The integers soundfile writes for `audio` as 16-bit PCM WAV.

    Encoded in memory with soundfile itself, so the rounding is exactly the
    one `make` uses.
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio, SR, subtype="PCM_16", format="WAV")
    buffer.seek(0)
    return sf.read(buffer, dtype="int16", always_2d=True)[0].astype(np.int32)


def verify_variant(data_dir: Path, variant: str) -> dict:
    """Recompute every clip in memory and compare with the files on disk."""
    out_root = variant_dir(data_dir, variant)
    worst = {"max_diff_lsb": 0, "clip": None, "stem": None}
    n_clips = n_differing_samples = n_samples = 0
    param_errors = []
    for split in SPLITS:
        for clip in sorted(p for p in (data_dir / split).iterdir() if p.is_dir()):
            out = out_root / split / clip.name
            stems, metadata = degrade_clip(clip, variant)
            saved = json.loads((out / "degradation.json").read_text())
            for name in VARIANTS[variant]:
                if saved[name] != metadata[name]:
                    param_errors.append(f"{split}/{clip.name}: {name}")
            for stem, audio in stems.items():
                on_disk = sf.read(
                    str(out / f"{stem}.wav"), dtype="int16", always_2d=True
                )[0]
                diff = np.abs(as_pcm16(audio) - on_disk.astype(np.int32))
                n_differing_samples += int(np.count_nonzero(diff))
                n_samples += diff.size
                if diff.max() > worst["max_diff_lsb"]:
                    worst = {
                        "max_diff_lsb": int(diff.max()),
                        "clip": f"{split}/{clip.name}",
                        "stem": stem,
                    }
            n_clips += 1
    return {
        "compared_with": str(out_root.name),
        "n_clips": n_clips,
        "max_diff_lsb": worst["max_diff_lsb"],
        "worst_clip": worst["clip"],
        "worst_stem": worst["stem"],
        "differing_samples": n_differing_samples,
        "total_samples": n_samples,
        "settings_identical": not param_errors,
        "settings_errors": param_errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("make", "verify"))
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument(
        "--data-dir", help="musdb7 folder (default: $ML_TEK5_MUSDB7_DIR)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="make: redo clips already done"
    )
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    if args.command == "make":
        make_variant(data_dir, args.variant, overwrite=args.overwrite)
        return 0

    report = verify_variant(data_dir, args.variant)
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    entry = results.get(args.variant, {})
    entry["reproduction"] = {
        "date": datetime.date.today().isoformat(),
        "machine": machine_name(),
        "versions": package_versions("numpy", "scipy", "soundfile"),
        **report,
    }
    update_results(RESULTS_PATH, args.variant, entry)
    print(
        f"{args.variant}: {report['n_clips']} clips vs {report['compared_with']}, "
        f"max diff {report['max_diff_lsb']} LSB "
        f"({report['differing_samples']} of {report['total_samples']} samples differ), "
        f"settings identical: {report['settings_identical']}"
    )
    return 0 if report["max_diff_lsb"] <= 1 and report["settings_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
