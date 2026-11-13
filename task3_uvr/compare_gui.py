"""T3-04: compare the pipeline's output with UVR GUI output on one clip.

    uv run python -m task3_uvr.compare_gui \
        --uvr-dir <UVR> --gui-dir <folder with the GUI's WAV outputs>

The defaults match the reference run: UVR-MDX-NET-Inst_HQ_3 on the musdb7
test clip "Al James - Schoolboy Facination" (Segment Size 256, Overlap
Default, CPU, 16-bit PCM, no normalization), saved by the GUI as
1_mixture_UVR-MDX-NET-Inst_HQ_3_(Instrumental|Vocals).wav.

For each stem it reports the max absolute difference and the SDR of the
pipeline output (default settings) against the GUI output, both for the
float output and after writing it as 16-bit PCM like the GUI. For the
secondary stem it also measures the Match Freq Cut variant. The summary goes
to results/task3_gui_compare.json.
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from task3_uvr.check_data import resolve_data_dir
from task3_uvr.mdx_inference import (
    STEM_NAMES,
    load_settings,
    model_data_dir,
    onnx_model,
    separate,
)
from task3_uvr.onnx_checks import machine_name, package_versions, update_results
from task3_uvr.sdr import sdr

RESULTS_PATH = Path("results") / "task3_gui_compare.json"
UVR_ENV_VAR = "ML_TEK5_UVR_DIR"
DEFAULT_CLIP = "test/Al James - Schoolboy Facination"
DEFAULT_MODEL = "UVR-MDX-NET-Inst_HQ_3"
GUI_STEM_LABELS = {"vocals": "Vocals", "accompaniment": "Instrumental"}
LSB = 1 / 32768


def resolve_uvr_dir(arg: str | None) -> Path:
    value = arg or os.environ.get(UVR_ENV_VAR)
    if not value:
        sys.exit(f"Give --uvr-dir or set {UVR_ENV_VAR}.")
    path = Path(value).expanduser()
    if not (path / "models" / "MDX_Net_Models").is_dir():
        sys.exit(f"Not a UVR folder (no models/MDX_Net_Models): {path}")
    return path


def resolve_model(name_or_path: str, uvr_dir: Path) -> Path:
    """An existing path, else <UVR>/models/MDX_Net_Models/<name>[.onnx]."""
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return path
    candidate = uvr_dir / "models" / "MDX_Net_Models" / name_or_path
    for p in (candidate, candidate.with_name(candidate.name + ".onnx")):
        if p.is_file():
            return p
    sys.exit(f"Model not found: {name_or_path}")


def read_stereo(path: Path) -> np.ndarray:
    """[2, N] float32, as UVR's librosa.load(mono=False) gives for 16-bit WAV."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != 44100 or audio.shape[1] != 2:
        raise ValueError(f"{path}: expected 44100 Hz stereo")
    return audio.T


def as_pcm16(wave: np.ndarray) -> np.ndarray:
    """Write [2, N] like UVR does (sf.write, subtype PCM_16) and read it back."""
    buffer = io.BytesIO()
    sf.write(buffer, wave.T, 44100, subtype="PCM_16", format="WAV")
    buffer.seek(0)
    return sf.read(buffer, dtype="float32")[0].T


def difference(reference: np.ndarray, estimate: np.ndarray) -> dict:
    diff = np.abs(reference.astype(np.float64) - estimate)
    return {
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_lsb": float(diff.max() / LSB),
        "samples_differing": int(np.count_nonzero(diff)),
        "sdr_vs_gui_db": sdr(reference, estimate),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--uvr-dir", help=f"UVR clone (default: ${UVR_ENV_VAR})")
    parser.add_argument("--gui-dir", required=True)
    parser.add_argument("--gui-prefix", default="1_mixture")
    parser.add_argument("--data-dir", help="musdb7 folder")
    parser.add_argument("--clip", default=DEFAULT_CLIP, help="<split>/<clip>")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    uvr_dir = resolve_uvr_dir(args.uvr_dir)
    model_path = resolve_model(args.model, uvr_dir)
    clip_dir = resolve_data_dir(args.data_dir) / args.clip
    model_hash, settings = load_settings(model_path, model_data_dir(uvr_dir))
    model = onnx_model(model_path, settings)

    mix = read_stereo(clip_dir / "mixture.wav")
    outputs = {
        cut: separate(mix, model, settings, match_freq_cut=cut) for cut in (True, False)
    }
    primary = STEM_NAMES[settings.primary_stem]

    report = {
        "clip": args.clip,
        "model": model_path.stem,
        "model_hash": model_hash,
        "machine": machine_name(),
        "versions": package_versions("onnxruntime", "torch", "numpy", "soundfile"),
        "stems": {},
    }
    for stem, label in GUI_STEM_LABELS.items():
        gui_path = (
            Path(args.gui_dir) / f"{args.gui_prefix}_{model_path.stem}_({label}).wav"
        )
        gui = read_stereo(gui_path)
        truth = read_stereo(clip_dir / f"{stem}.wav")
        ours = outputs[False][stem]
        entry = {
            "gui_file": gui_path.name,
            "role": "primary" if stem == primary else "secondary",
            "float_vs_gui": difference(gui, ours),
            "pcm16_vs_gui": difference(gui, as_pcm16(ours)),
            "samples_above_full_scale": int(np.count_nonzero(np.abs(ours) > 1.0)),
            "sdr_vs_ground_truth_db": {
                "gui": sdr(truth, gui),
                "pipeline": sdr(truth, ours),
            },
        }
        if stem != primary:
            entry["match_freq_cut_vs_gui"] = difference(gui, outputs[True][stem])
            entry["sdr_vs_ground_truth_db"]["pipeline_match_freq_cut"] = sdr(
                truth, outputs[True][stem]
            )
        report["stems"][stem] = entry
    update_results(RESULTS_PATH, f"{model_path.stem} | {args.clip}", report)

    for stem, e in report["stems"].items():
        f, q = e["float_vs_gui"], e["pcm16_vs_gui"]
        print(
            f"{stem} ({e['role']}): float max diff {f['max_abs_diff']:.2e} "
            f"({f['max_abs_diff_lsb']:.2f} LSB), SDR vs GUI {f['sdr_vs_gui_db']:.1f} dB; "
            f"16-bit max diff {q['max_abs_diff_lsb']:.0f} LSB on "
            f"{q['samples_differing']} samples, SDR vs GUI {q['sdr_vs_gui_db']:.1f} dB; "
            f"{e['samples_above_full_scale']} samples above 1.0"
        )
        if "match_freq_cut_vs_gui" in e:
            p = e["match_freq_cut_vs_gui"]
            print(
                f"  with Match Freq Cut: max diff {p['max_abs_diff']:.2e}, "
                f"SDR vs GUI {p['sdr_vs_gui_db']:.1f} dB"
            )
        print(
            "  SDR vs ground truth:",
            {k: round(v, 3) for k, v in e["sdr_vs_ground_truth_db"].items()},
        )
    print("->", RESULTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
