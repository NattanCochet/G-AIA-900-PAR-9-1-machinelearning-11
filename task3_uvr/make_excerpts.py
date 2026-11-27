"""T3-07: audio excerpts of HQ 3 and the benchmark on clean and degraded clips.

Run from the repository root, after task3_uvr.evaluate on the three datasets:

    uv run python -m task3_uvr.make_excerpts

For each chosen test clip and each dataset (musdb7, musdb7_phone,
musdb7_narrowband), it writes to task3_uvr/artifacts/excerpts/<dataset>/<clip>/
(ignored by git), as 16-bit WAV like UVR:

- mixture.wav, vocals.wav, accompaniment.wav: the dataset's files, so the
  true stems are the degraded ones on degraded data;
- hq3_vocals.wav, hq3_accompaniment.wav: UVR-MDX-NET-Inst_HQ_3;
- benchmark_vocals.wav, benchmark_accompaniment.wav: Benchmark_CentreChannel.

For each clip, excerpts/compare/<clip>.wav plays the clean mixture, then HQ 3's
vocals on clean, phone and narrowband, with 0.5 s of silence between them.
Each part keeps its own level. excerpts/index.md lists the files and their
global SDR against the dataset's true stem.

Default clips (test split), chosen from results/task3_sdr_clips.csv:

- Al James - Schoolboy Facination, the reference clip of T3-04;
- the clip whose HQ 3 vocal SDR drops the most from musdb7 to musdb7_phone;
- the clip with the loudest vocals relative to the accompaniment in musdb7.

The models come from --uvr-dir or ML_TEK5_UVR_DIR, the musdb7 folder from
--data-dir or ML_TEK5_MUSDB7_DIR.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from task3_uvr.check_data import resolve_data_dir
from task3_uvr.compare_gui import read_stereo, resolve_model, resolve_uvr_dir
from task3_uvr.evaluate import CLIPS_CSV, DATASETS, STEMS, dataset_dir
from task3_uvr.mdx_inference import load_settings, model_data_dir, onnx_model, separate
from task3_uvr.sdr import sdr

EXCERPTS_DIR = Path("task3_uvr") / "artifacts" / "excerpts"
SR = 44100
GAP_SECONDS = 0.5
REFERENCE_CLIP = "Al James - Schoolboy Facination"
MODELS = {"hq3": "UVR-MDX-NET-Inst_HQ_3", "benchmark": "Benchmark_CentreChannel"}
HQ3 = MODELS["hq3"]
DATASET_LABELS = {
    "musdb7": "clean",
    "musdb7_phone": "phone",
    "musdb7_narrowband": "narrowband",
}


def largest_drop_clip(
    clips: pd.DataFrame, degraded: str = "musdb7_phone", exclude=()
) -> tuple[str, float]:
    """Test clip whose HQ 3 vocal SDR drops the most from musdb7 to `degraded`,
    and the drop in dB."""
    rows = clips[
        (clips["model"] == HQ3)
        & (clips["split"] == "test")
        & (clips["stem"] == "vocals")
    ]
    sdr_by = rows.pivot_table(index="clip", columns="dataset", values="sdr_db")
    drop = (sdr_by["musdb7"] - sdr_by[degraded]).drop(list(exclude), errors="ignore")
    return drop.idxmax(), float(drop.max())


def loudest_vocals_clip(clips: pd.DataFrame, exclude=()) -> tuple[str, float]:
    """Test clip of musdb7 with the largest vocals / accompaniment energy ratio,
    and that ratio in dB (the mixture reference's vocal SDR)."""
    rows = clips[
        (clips["dataset"] == "musdb7")
        & (clips["model"] == "mixture")
        & (clips["split"] == "test")
    ]
    energy = rows.pivot_table(index="clip", columns="stem", values="reference_energy")
    ratio_db = 10 * np.log10(energy["vocals"] / energy["accompaniment"])
    ratio_db = ratio_db.drop(list(exclude), errors="ignore")
    return ratio_db.idxmax(), float(ratio_db.max())


def default_clips(clips: pd.DataFrame) -> dict[str, str]:
    """{clip: reason it was chosen}."""
    chosen = {REFERENCE_CLIP: "reference clip of T3-04"}
    clip, drop = largest_drop_clip(clips, exclude=chosen)
    chosen[clip] = f"largest HQ 3 vocal SDR drop, clean to phone ({drop:.2f} dB)"
    clip, ratio = loudest_vocals_clip(clips, exclude=chosen)
    chosen[clip] = f"loudest vocals in musdb7 (vocals {ratio:+.2f} dB vs accompaniment)"
    return chosen


def with_gaps(parts: list[np.ndarray], gap_seconds: float = GAP_SECONDS) -> np.ndarray:
    """[N, 2] parts joined with silence between them (none at the ends)."""
    gap = np.zeros((round(gap_seconds * SR), 2), dtype=np.float32)
    joined = []
    for part in parts:
        joined += [np.asarray(part, dtype=np.float32), gap]
    return np.concatenate(joined[:-1])


def write_wav(path: Path, wave: np.ndarray) -> None:
    """[2, N] as 16-bit PCM, like UVR's output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.T, SR, subtype="PCM_16")


def load_models(uvr_dir: Path) -> dict:
    """{prefix: estimator(mix [2, N]) -> {stem: [2, N]}}, as evaluate runs them."""
    estimators = {}
    for prefix, name in MODELS.items():
        path = resolve_model(name, uvr_dir)
        _, settings = load_settings(path, model_data_dir(uvr_dir))
        model = onnx_model(path, settings)
        estimators[prefix] = lambda mix, m=model, s=settings: separate(mix, m, s)
    return estimators


def make_clip(data_dir: Path, clip: str, estimators: dict) -> tuple[list, dict]:
    """Write one clip's files for every dataset; return the index rows and
    HQ 3's vocals per dataset."""
    rows, hq3_vocals = [], {}
    for dataset in DATASETS:
        clip_dir = dataset_dir(data_dir, dataset) / "test" / clip
        out = EXCERPTS_DIR / dataset / clip
        truth = {s: read_stereo(clip_dir / f"{s}.wav") for s in ("mixture", *STEMS)}
        for name, wave in truth.items():
            write_wav(out / f"{name}.wav", wave)
        rows.append((dataset, "mixture.wav", "input", None))
        for stem in STEMS:
            rows.append((dataset, f"{stem}.wav", "true stem", None))
        for prefix, estimator in estimators.items():
            estimates = estimator(truth["mixture"])
            for stem in STEMS:
                write_wav(out / f"{prefix}_{stem}.wav", estimates[stem])
                score = sdr(truth[stem], estimates[stem])
                rows.append((dataset, f"{prefix}_{stem}.wav", MODELS[prefix], score))
            if prefix == "hq3":
                hq3_vocals[dataset] = estimates["vocals"]
        print(f"{dataset}/{clip}", flush=True)
    return rows, hq3_vocals


def write_compare(data_dir: Path, clip: str, hq3_vocals: dict) -> Path:
    mixture = read_stereo(data_dir / "test" / clip / "mixture.wav")
    parts = [mixture.T] + [hq3_vocals[d].T for d in DATASETS]
    path = EXCERPTS_DIR / "compare" / f"{clip}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), with_gaps(parts), SR, subtype="PCM_16")
    return path


def csv_score(clips: pd.DataFrame, dataset: str, model: str, clip: str, stem: str):
    match = clips[
        (clips["dataset"] == dataset)
        & (clips["model"] == model)
        & (clips["split"] == "test")
        & (clips["clip"] == clip)
        & (clips["stem"] == stem)
    ]["sdr_db"]
    return float(match.iloc[0]) if len(match) else None


def write_index(chosen: dict, rows: dict, compare: dict) -> Path:
    lines = [
        "# T3-07 excerpts",
        "",
        "Test clips. Global SDR in dB against the dataset's own true stem (the",
        "degraded stem on phone and narrowband), computed on the float output",
        "before the 16-bit write, as in results/task3_sdr_clips.csv.",
        "",
    ]
    for clip, reason in chosen.items():
        lines += [f"## {clip}", "", f"Chosen as: {reason}.", ""]
        lines += [
            (
                f"- `compare/{compare[clip].name}`: clean mixture, then HQ 3 vocals on "
                "clean, phone and narrowband, 0.5 s apart"
            ),
            "",
            "| Dataset | File | Content | SDR (dB) |",
            "| --- | --- | --- | --- |",
        ]
        for dataset, name, content, score in rows[clip]:
            shown = "—" if score is None else f"{score:.2f}"
            lines.append(
                f"| {dataset} | `{dataset}/{clip}/{name}` | {content} | {shown} |"
            )
        lines.append("")
    path = EXCERPTS_DIR / "index.md"
    path.write_text("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--clips", nargs="+", help="test clip names (default: see above)"
    )
    parser.add_argument("--uvr-dir")
    parser.add_argument(
        "--data-dir", help="musdb7 folder (default: $ML_TEK5_MUSDB7_DIR)"
    )
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    clips_csv = pd.read_csv(CLIPS_CSV)

    if args.clips:
        chosen = {c: "given on the command line" for c in args.clips}
    else:
        chosen = default_clips(clips_csv)
    estimators = load_models(resolve_uvr_dir(args.uvr_dir))

    rows, compare, worst = {}, {}, 0.0
    for clip in chosen:
        rows[clip], hq3_vocals = make_clip(data_dir, clip, estimators)
        compare[clip] = write_compare(data_dir, clip, hq3_vocals)
        for dataset, name, model, score in rows[clip]:
            if score is None:
                continue
            stem = name.split("_", 1)[1].removesuffix(".wav")
            expected = csv_score(clips_csv, dataset, model, clip, stem)
            if expected is not None:
                worst = max(worst, abs(score - expected))
    index = write_index(chosen, rows, compare)
    print(f"Largest SDR difference with {CLIPS_CSV}: {worst:.2e} dB")
    print("->", index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
