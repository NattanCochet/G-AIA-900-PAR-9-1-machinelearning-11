"""T3-10: before/after table from per-clip SDR rows."""

import pandas as pd
import pytest

from task3_uvr.finetune_report import (
    HQ3_NAME,
    before_after,
    read_clip_rows,
    save_table,
    t3_07_mismatches,
    to_markdown,
)

FT = "HQ3_ft_musdb7_phone_lr1e-06_s2000"


def rows(model, variant, dataset, values, split="validation"):
    """values: {clip: (vocals, accompaniment)}."""
    return [
        {
            "dataset": dataset,
            "model": model,
            "variant": variant,
            "split": split,
            "match_freq_cut": "False",
            "clip": clip,
            "stem": stem,
            "sdr_db": sdr,
        }
        for clip, pair in values.items()
        for stem, sdr in zip(("vocals", "accompaniment"), pair, strict=True)
    ]


@pytest.fixture
def clips():
    data = (
        rows(HQ3_NAME, "", "musdb7_phone", {"A": (10, 8), "B": (11, 9), "C": (9, 7)})
        + rows(
            FT, "onnx", "musdb7_phone", {"A": (11, 9), "B": (10.5, 9.5), "C": (10, 8)}
        )
        + rows(HQ3_NAME, "", "musdb7", {"A": (12, 14), "B": (13, 15), "C": (11, 13)})
        + rows(FT, "onnx", "musdb7", {"A": (11, 13), "B": (13, 15), "C": (10, 12)})
        # Same model on another backend or with Match Freq Cut: never paired.
        + rows(
            HQ3_NAME, "torch", "musdb7_phone", {"A": (0, 0), "B": (0, 0), "C": (0, 0)}
        )
        + rows(HQ3_NAME, "", "musdb7_phone", {"A": (0, 0)}, split="test")
    )
    frame = pd.DataFrame(data)
    frame["sdr_db"] = frame["sdr_db"].astype(float)
    return frame


def test_pairs_each_clip_with_the_original(clips):
    table = before_after(clips, FT).set_index(["dataset", "stem"])
    phone = table.loc[("musdb7_phone", "accompaniment")]
    assert phone["n_clips"] == 3
    assert phone["before_median_db"] == 8 and phone["after_median_db"] == 9
    assert phone["median_difference_db"] == 1  # +1, +0.5, +1
    assert phone["mean_difference_db"] == pytest.approx(2.5 / 3)
    assert phone["n_improved"] == 3
    vocals = table.loc[("musdb7_phone", "vocals")]
    assert (
        vocals["median_difference_db"] == 1 and vocals["n_improved"] == 2
    )  # +1, -0.5, +1
    clean = table.loc[("musdb7", "accompaniment")]
    assert clean["median_difference_db"] == -1 and clean["n_improved"] == 0
    # The training dataset first, clean musdb7 last.
    assert (
        list(table.index.get_level_values(0)) == ["musdb7_phone"] * 2 + ["musdb7"] * 2
    )


def test_torch_rows_are_paired_with_the_torch_baseline(clips):
    clips = clips[~((clips["model"] == FT) & (clips["dataset"] == "musdb7"))].copy()
    clips.loc[clips["model"] == FT, "variant"] = "torch"
    table = before_after(clips, FT)
    assert table["dataset"].unique().tolist() == ["musdb7_phone"]
    assert (table["before_median_db"] == 0).all()  # the torch rows, not the onnx ones


def test_missing_baseline_is_an_error(clips):
    clips.loc[clips["model"] == FT, "variant"] = "torch"
    with pytest.raises(ValueError, match="variant 'torch'"):
        before_after(clips, FT)
    with pytest.raises(ValueError, match="no test rows"):
        before_after(clips, FT, "test")


def test_markdown_and_csv(clips, tmp_path):
    table = before_after(clips, FT)
    text = to_markdown(table)
    assert (
        "| musdb7_phone | accompaniment | 8.00 | 9.00 | +1.00 (+0.83) | 3/3 |" in text
    )
    assert "| musdb7 (clean) | vocals |" in text
    path = tmp_path / "table.csv"
    save_table(table, path)
    save_table(table, path)  # replaces, does not duplicate
    assert len(pd.read_csv(path)) == 4


def test_read_clip_rows_adds_missing_columns(tmp_path):
    path = tmp_path / "clips.csv"
    path.write_text(
        "date,model,model_hash,split,match_freq_cut,clip,stem,sdr_db,reference_energy,seconds\n"
        "2026-11-13,mixture,,validation,False,A,vocals,-3.96,2312.2,2.9e-06\n"
    )
    frame = read_clip_rows(path)
    assert frame.loc[0, "dataset"] == "musdb7" and frame.loc[0, "variant"] == ""
    assert frame.loc[0, "sdr_db"] == -3.96


def test_t3_07_check_only_on_the_full_validation_split(clips):
    table = before_after(clips, FT)
    assert (
        table.set_index(["dataset", "stem"]).loc[
            ("musdb7_phone", "accompaniment"), "t3_07_before_db"
        ]
        == 8.82
    )
    assert t3_07_mismatches(table) == []  # 3 clips: not comparable with T3-07
    table["n_clips"] = 14
    mismatches = t3_07_mismatches(table)
    assert len(mismatches) == 4  # the synthetic medians are not T3-07's
    assert mismatches[0] == "musdb7_phone vocals: before 10.00 dB, T3-07 10.47 dB"
    table.loc[0, "before_median_db"] = 10.475  # within the rounding of T3-07
    assert len(t3_07_mismatches(table)) == 3


def test_no_t3_07_value_for_the_torch_backend(clips):
    clips = clips[~((clips["model"] == FT) & (clips["dataset"] == "musdb7"))].copy()
    clips.loc[clips["model"] == FT, "variant"] = "torch"
    assert before_after(clips, FT)["t3_07_before_db"].isna().all()
