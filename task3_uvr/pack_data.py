"""T3-08, T3-10: pack the task 3 datasets into one zip for Google Drive.

    uv run python -m task3_uvr.pack_data [--data-dir <musdb7>] [--output <zip>]

The archive holds musdb7/, musdb7_phone/ and musdb7_narrowband/ (train and
test, every clip's WAV files and degradation.json) and the split lists:

    musdb7/{train,test}/<clip>/{mixture,vocals,accompaniment}.wav
    musdb7_phone/...  musdb7_narrowband/...
    splits/{train,validation}.json

It is written next to the datasets, outside this repository (default:
<musdb7 parent>/ml_tek5_task3_data.zip), then reopened to check every
member's CRC. The datasets are only read. Upload the zip to the team's Drive
folder ml_tek5_checkpoints/task3/; task3_uvr/colab_train.ipynb unzips it to
/content and sets ML_TEK5_MUSDB7_DIR=/content/data/musdb7.
"""

import argparse
import sys
import zipfile
from pathlib import Path

from task3_uvr.check_data import resolve_data_dir
from task3_uvr.small_nn_data import DATASETS, SPLITS_DIR, dataset_dir

ARCHIVE_NAME = "ml_tek5_task3_data.zip"
EXPECTED_CLIPS = {"train": 94, "test": 50}


def members(musdb7: Path) -> list[tuple[Path, str]]:
    """(file, name in the archive) for every file to pack, sorted."""
    files = []
    for dataset in DATASETS:
        root = dataset_dir(musdb7, dataset)
        for split, expected in EXPECTED_CLIPS.items():
            clips = sorted(p for p in (root / split).iterdir() if p.is_dir())
            if len(clips) != expected:
                sys.exit(f"{root / split}: {len(clips)} clips, expected {expected}")
            for clip in clips:
                for path in sorted(clip.iterdir()):
                    if path.is_file():
                        files.append(
                            (path, f"{dataset}/{split}/{clip.name}/{path.name}")
                        )
    for path in sorted(SPLITS_DIR.glob("*.json")):
        files.append((path, f"splits/{path.name}"))
    return files


def pack(musdb7: Path, output: Path, compress: bool = True) -> dict:
    files = members(musdb7)
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    tmp = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=method, compresslevel=6) as archive:
        for path, name in files:
            archive.write(path, name)
    with zipfile.ZipFile(tmp) as archive:
        bad = archive.testzip()
    if bad is not None:
        tmp.unlink()
        sys.exit(f"CRC error on {bad}")
    tmp.replace(output)
    raw = sum(path.stat().st_size for path, _ in files)
    return {"files": len(files), "raw_bytes": raw, "zip_bytes": output.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir", help="musdb7 folder (default: $ML_TEK5_MUSDB7_DIR)"
    )
    parser.add_argument(
        "--output", type=Path, help=f"default: <musdb7 parent>/{ARCHIVE_NAME}"
    )
    parser.add_argument("--store", action="store_true", help="no compression (faster)")
    args = parser.parse_args(argv)
    musdb7 = resolve_data_dir(args.data_dir)
    output = (args.output or musdb7.parent / ARCHIVE_NAME).resolve()
    if Path.cwd().resolve() in output.parents:
        sys.exit(f"{output} is inside the repository; write it outside")
    info = pack(musdb7, output, compress=not args.store)
    print(
        f"{output}: {info['files']} files, {info['raw_bytes'] / 1e6:.0f} MB of data, "
        f"zip {info['zip_bytes'] / 1e6:.0f} MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
