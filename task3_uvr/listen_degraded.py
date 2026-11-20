"""T3-06: listening files and spectrogram figure for the degraded datasets.

Run from the repository root, after `task3_uvr.degradation make`:

    uv run python -m task3_uvr.listen_degraded listen
    uv run python -m task3_uvr.listen_degraded figure

`listen` writes one WAV per clip to task3_uvr/artifacts/listen/ (ignored by
git): the clean mixture, then the phone mixture, then the narrowband mixture,
separated by 0.5 s of silence. Each part keeps its own level, so the level
drop is audible too. The default clips are from the test split:

- Al James - Schoolboy Facination, the reference clip of T3-04;
- Triviul feat. The Fiend - Widow, the longest phone reverb (rt60 0.98 s);
- Signe Jakobsen - What Have You Done To Me, the narrowest phone band
  (137-3006 Hz), the closest to narrowband.

`figure` saves results/task3_degradation_spectrograms.png: spectrograms of one
mixture, clean vs phone vs narrowband. The musdb7 folder comes from
--data-dir or ML_TEK5_MUSDB7_DIR.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from common import set_style
from task3_uvr.check_data import resolve_data_dir
from task3_uvr.degradation import SR, variant_dir

LISTEN_DIR = Path("task3_uvr") / "artifacts" / "listen"
FIGURE_PATH = Path("results") / "task3_degradation_spectrograms.png"
VERSIONS = ("clean", "phone", "narrowband")
DEFAULT_CLIPS = (
    "test/Al James - Schoolboy Facination",
    "test/Triviul feat. The Fiend - Widow",
    "test/Signe Jakobsen - What Have You Done To Me",
)
GAP_SECONDS = 0.5
DYNAMIC_RANGE_DB = 90.0


def version_dir(data_dir: Path, version: str) -> Path:
    return data_dir if version == "clean" else variant_dir(data_dir, version)


def read_mixture(data_dir: Path, version: str, clip: str) -> np.ndarray:
    path = version_dir(data_dir, version) / clip / "mixture.wav"
    return sf.read(str(path), dtype="float32", always_2d=True)[0]


def write_listening_files(data_dir: Path, clips: tuple[str, ...]) -> list[Path]:
    LISTEN_DIR.mkdir(parents=True, exist_ok=True)
    gap = np.zeros((int(GAP_SECONDS * SR), 2), dtype=np.float32)
    paths = []
    for clip in clips:
        parts = []
        for version in VERSIONS:
            parts += [read_mixture(data_dir, version, clip), gap]
        path = LISTEN_DIR / f"{Path(clip).name} - {', '.join(VERSIONS)}.wav"
        sf.write(str(path), np.concatenate(parts[:-1]), SR, subtype="PCM_16")
        paths.append(path)
        print(path)
    return paths


def settings_label(data_dir: Path, version: str, clip: str) -> str:
    if version == "clean":
        return "clean"
    s = json.loads(
        (version_dir(data_dir, version) / clip / "degradation.json").read_text()
    )
    return (
        f"{version}: band {s['low_cut']:.0f}-{s['high_cut']:.0f} Hz, "
        f"rt60 {s['rt60']:.2f} s, wet {s['wet']:.2f}"
    )


def save_figure(data_dir: Path, clip: str) -> Path:
    """Spectrograms of one mixture (both channels averaged), on one dB scale."""
    set_style()
    spectra = {}
    for version in VERSIONS:
        mono = read_mixture(data_dir, version, clip).astype(np.float64).mean(axis=1)
        freqs, times, power = signal.spectrogram(
            mono, fs=SR, window="hann", nperseg=4096, noverlap=3072, scaling="spectrum"
        )
        spectra[version] = 10 * np.log10(power + 1e-20)
    reference = spectra["clean"].max()  # 0 dB = loudest bin of the clean mixture

    fig, axes = plt.subplots(
        1, len(VERSIONS), figsize=(13, 4.2), sharey=True, constrained_layout=True
    )
    for ax, version in zip(axes, VERSIONS, strict=True):
        image = ax.pcolormesh(
            times,
            freqs[1:],
            spectra[version][1:] - reference,
            shading="auto",
            cmap="magma",
            vmin=-DYNAMIC_RANGE_DB,
            vmax=0,
            rasterized=True,
        )
        ax.set_yscale("log")
        ax.set_ylim(40, SR / 2)
        ax.grid(False)
        ax.set_title(settings_label(data_dir, version, clip), fontsize=10)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    colorbar = fig.colorbar(image, ax=axes, pad=0.01)
    colorbar.set_label("Power (dB, 0 dB = loudest bin of the clean mixture)")
    fig.suptitle(f"Mixture spectrograms: {clip}")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=110)
    plt.close(fig)
    print(FIGURE_PATH)
    return FIGURE_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("listen", "figure"))
    parser.add_argument(
        "--clips",
        nargs="+",
        default=list(DEFAULT_CLIPS),
        help="clips as <split>/<name> (figure: the first one)",
    )
    parser.add_argument(
        "--data-dir", help="musdb7 folder (default: $ML_TEK5_MUSDB7_DIR)"
    )
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    if args.command == "listen":
        write_listening_files(data_dir, tuple(args.clips))
    else:
        save_figure(data_dir, args.clips[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
