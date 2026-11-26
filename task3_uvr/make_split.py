"""T3-03: set aside a validation split from the MUSDB18-7 training clips.

Run from the repository root, after task3_uvr.check_data:

    uv run python -m task3_uvr.make_split <musdb7>

(or set ML_TEK5_MUSDB7_DIR). Only train/ is split; the official test set is
left as it is. Clips are grouped by artist, so no artist is in both the
training and validation lists. The data check is run again first, and nothing
is written if any clip fails it. The clip names go to
task3_uvr/splits/validation.json and task3_uvr/splits/train.json.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from task3_uvr.check_data import artist_of, list_clips, resolve_data_dir, run_check

SPLITS_DIR = Path(__file__).parent / "splits"
DEFAULT_SEED = 0
DEFAULT_FRACTION = 0.15
METHOD = (
    "Group the official training clips by artist (folder name before ' - ', "
    "without 'feat.' guests or a trailing '(...)'). Sort the artists, shuffle "
    "them with numpy.random.default_rng(seed), then take whole artists in that "
    "order while the validation set stays at or below round(fraction * n) clips."
)


def group_by_artist(clips: list[str]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for clip in sorted(clips):
        groups[artist_of(clip)].append(clip)
    return dict(groups)


def split_by_artist(
    clips: list[str], fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Return sorted (train, validation) clip names, grouped by artist.

    The result depends only on the set of names, the fraction and the seed,
    not on the order of `clips`.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    if len(set(clips)) != len(clips):
        raise ValueError("duplicate clip names")
    groups = group_by_artist(clips)
    artists = sorted(groups)
    target = round(fraction * len(clips))
    rng = np.random.default_rng(seed)

    validation = []
    for i in rng.permutation(len(artists)):
        group = groups[artists[i]]
        if len(validation) + len(group) <= target:
            validation += group
        if len(validation) == target:
            break
    validation = sorted(validation)
    chosen = set(validation)
    train = sorted(c for c in clips if c not in chosen)
    return train, validation


def write_split(
    train: list[str], validation: list[str], fraction: float, seed: int, out_dir: Path
) -> None:
    common = {
        "source": "MUSDB18-7 train/",
        "method": METHOD,
        "seed": seed,
        "target_fraction": fraction,
    }
    files = {
        "validation.json": {
            **common,
            "n_clips": len(validation),
            "artists": sorted({artist_of(c) for c in validation}),
            "clips": validation,
        },
        "train.json": {**common, "n_clips": len(train), "clips": train},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_text(json.dumps(content, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data_dir", nargs="?", help="musdb7 folder")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    summary = run_check(data_dir)
    if not summary["ok"]:
        for a in summary["anomalies"]:
            print("ANOMALY:", a)
        print("Data check failed: no split written. See task3_uvr.check_data.")
        return 1

    clips, _ = list_clips(data_dir / "train")
    train, validation = split_by_artist(clips, args.fraction, args.seed)
    write_split(train, validation, args.fraction, args.seed, SPLITS_DIR)
    n_val_artists = len({artist_of(c) for c in validation})
    print(
        f"{len(clips)} training clips -> train {len(train)}, validation "
        f"{len(validation)} ({len(validation) / len(clips):.1%}, "
        f"{n_val_artists} artists), seed {args.seed} -> {SPLITS_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
