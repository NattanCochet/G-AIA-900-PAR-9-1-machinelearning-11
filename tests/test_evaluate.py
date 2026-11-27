"""T3-07 dataset option of task3_uvr.evaluate and migration of the per-clip CSV."""

import csv
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from task3_uvr import evaluate
from task3_uvr.sdr import sdr

OLD_HEADER = (
    "date,model,model_hash,split,match_freq_cut,clip,stem,sdr_db,"
    "reference_energy,seconds"
)
# Rows as T3-04 wrote them: empty hash for the mixture, a clip name with a
# comma (quoted), and floats whose text must survive unchanged.
OLD_ROWS = [
    (
        "2026-11-13,mixture,,validation,False,Alexander Ross - Goodbye Bolero,"
        "vocals,-3.9650826098752523,2312.206074461341,2.8969998311367817e-06"
    ),
    (
        "2026-11-13,UVR-MDX-NET-Inst_HQ_3,55657dd70583b0fedfba5f67df11d711,test,False,"
        '"A, B - C",accompaniment,14.655055070159714,1e-05,10.9'
    ),
]


def write_old_csv(path):
    path.write_text("\n".join([OLD_HEADER, *OLD_ROWS]) + "\n")


def rows_without(path, column):
    with path.open(newline="") as f:
        return [
            {k: v for k, v in row.items() if k != column} for row in csv.DictReader(f)
        ]


def clip_frame(dataset, model, split, sdr_db):
    return pd.DataFrame(
        [
            {
                "date": "2026-11-27",
                "dataset": dataset,
                "model": model,
                "model_hash": "",
                "split": split,
                "match_freq_cut": False,
                "clip": "Al James - Schoolboy Facination",
                "stem": stem,
                "sdr_db": sdr_db,
                "reference_energy": 1.0,
                "seconds": 0.1,
            }
            for stem in evaluate.STEMS
        ]
    )[evaluate.COLUMNS]


# --- dataset folder ---


def test_default_dataset_is_the_data_dir(tmp_path):
    (tmp_path / "musdb7").mkdir()
    assert evaluate.dataset_dir(tmp_path / "musdb7", "musdb7") == tmp_path / "musdb7"


@pytest.mark.parametrize("dataset", ["musdb7_phone", "musdb7_narrowband"])
def test_degraded_dataset_is_next_to_musdb7(tmp_path, dataset):
    (tmp_path / "musdb7").mkdir()
    (tmp_path / dataset).mkdir()
    assert evaluate.dataset_dir(tmp_path / "musdb7", dataset) == tmp_path / dataset


def test_missing_dataset_folder_exits(tmp_path):
    (tmp_path / "musdb7").mkdir()
    with pytest.raises(SystemExit):
        evaluate.dataset_dir(tmp_path / "musdb7", "musdb7_phone")


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError):
        evaluate.dataset_dir(tmp_path, "musdb7_raw")


def test_cli_rejects_unknown_dataset():
    with pytest.raises(SystemExit):
        evaluate.main(["--dataset", "musdb7_raw"])


# --- migration ---


def test_migration_adds_musdb7_without_changing_values(tmp_path):
    path = tmp_path / "clips.csv"
    write_old_csv(path)
    before = rows_without(path, "dataset")

    assert evaluate.migrate_clips_csv(path) is True

    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(evaluate.COLUMNS)
    assert rows_without(path, "dataset") == before  # same text, field by field
    assert {r["dataset"] for r in csv.DictReader(path.open())} == {"musdb7"}


def test_migration_runs_once(tmp_path):
    path = tmp_path / "clips.csv"
    write_old_csv(path)
    evaluate.migrate_clips_csv(path)
    text = path.read_text()
    assert evaluate.migrate_clips_csv(path) is False
    assert path.read_text() == text


def test_migration_of_missing_file_does_nothing(tmp_path):
    assert evaluate.migrate_clips_csv(tmp_path / "clips.csv") is False
    assert not (tmp_path / "clips.csv").exists()


# --- saving rows ---


def test_degraded_rows_do_not_replace_clean_rows(tmp_path):
    path = tmp_path / "clips.csv"
    evaluate.save_clip_rows(clip_frame("musdb7", "mixture", "test", 1.0), path)
    evaluate.save_clip_rows(clip_frame("musdb7_phone", "mixture", "test", 2.0), path)
    saved = pd.read_csv(path)
    assert len(saved) == 4
    assert saved.groupby("dataset")["sdr_db"].first().to_dict() == {
        "musdb7": 1.0,
        "musdb7_phone": 2.0,
    }


def test_rerun_replaces_rows_of_the_same_dataset_only(tmp_path):
    path = tmp_path / "clips.csv"
    evaluate.save_clip_rows(clip_frame("musdb7", "mixture", "test", 1.0), path)
    evaluate.save_clip_rows(clip_frame("musdb7_phone", "mixture", "test", 2.0), path)
    evaluate.save_clip_rows(clip_frame("musdb7_phone", "mixture", "test", 3.0), path)
    saved = pd.read_csv(path)
    assert len(saved) == 4
    assert sorted(saved["sdr_db"].unique()) == [1.0, 3.0]


def test_saving_into_an_old_file_keeps_its_rows_as_musdb7(tmp_path):
    path = tmp_path / "clips.csv"
    write_old_csv(path)
    evaluate.save_clip_rows(clip_frame("musdb7_phone", "mixture", "test", 2.0), path)
    saved = pd.read_csv(path)
    assert list(saved.columns) == evaluate.COLUMNS
    assert saved["dataset"].value_counts().to_dict() == {"musdb7": 2, "musdb7_phone": 2}


# --- end to end, mixture reference only (no model needed) ---


def write_clip(folder, vocals, accompaniment):
    folder.mkdir(parents=True)
    for name, audio in (
        ("vocals", vocals),
        ("accompaniment", accompaniment),
        ("mixture", vocals + accompaniment),
    ):
        sf.write(str(folder / f"{name}.wav"), audio, 44100, subtype="PCM_16")


def test_main_reads_the_degraded_copy_and_records_the_dataset(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    clean, phone = tmp_path / "musdb7", tmp_path / "musdb7_phone"
    shape = (4410, 2)
    write_clip(
        clean / "test" / "x",
        0.1 * rng.standard_normal(shape),
        0.2 * rng.standard_normal(shape),
    )
    vocals, accompaniment = (
        0.2 * rng.standard_normal(shape),
        0.05 * rng.standard_normal(shape),
    )
    write_clip(phone / "test" / "x", vocals, accompaniment)

    clips_csv, summary_csv = tmp_path / "clips.csv", tmp_path / "summary.csv"
    write_old_csv(clips_csv)
    monkeypatch.setattr(evaluate, "CLIPS_CSV", clips_csv)
    monkeypatch.setattr(evaluate, "SUMMARY_CSV", summary_csv)

    argv = ["--data-dir", str(clean), "--dataset", "musdb7_phone", "--splits", "test"]
    assert evaluate.main(argv) == 0

    saved = pd.read_csv(clips_csv)
    new = saved[saved["dataset"] == "musdb7_phone"].set_index("stem")
    assert list(new["clip"].unique()) == ["x"]
    # The labels are the degraded stems: the mixture reference is scored
    # against the phone vocals, read back as the 16-bit files.
    v = sf.read(str(phone / "test/x/vocals.wav"), dtype="float32")[0].T
    m = sf.read(str(phone / "test/x/mixture.wav"), dtype="float32")[0].T
    assert new.loc["vocals", "sdr_db"] == pytest.approx(sdr(v, m))
    assert new.loc["vocals", "sdr_db"] > 0  # the phone vocals are the louder stem here
    assert (saved["dataset"] == "musdb7").sum() == 2  # old rows kept

    summary = pd.read_csv(summary_csv)
    assert list(summary.columns) == [
        "date",
        "machine",
        "task",
        "seed",
        "config",
        "split",
        "metric",
        "score",
        "duration_seconds",
    ]
    assert {json.loads(c)["dataset"] for c in summary["config"]} == {"musdb7_phone"}
