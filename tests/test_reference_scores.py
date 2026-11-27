"""T3-07 medians, paired drops and gains of task3_uvr.reference_scores."""

import pandas as pd
import pytest

from task3_uvr import reference_scores as rs

# Per clip: (mixture, benchmark, HQ 3) vocal SDR on each dataset.
SCORES = {
    "musdb7": {"a": (-4, -4, 10), "b": (-2, -3, 12), "c": (-6, -5, 7)},
    "musdb7_phone": {"a": (0, 0, 9), "b": (1, 0, 6), "c": (-1, -2, 8)},
    "musdb7_narrowband": {"a": (2, 1, 5), "b": (3, 2, 4), "c": (1, 1, 6)},
}


@pytest.fixture
def clips():
    rows = []
    for dataset, by_clip in SCORES.items():
        for clip, values in by_clip.items():
            for model, value in zip(rs.MODELS, values, strict=True):
                for stem, sign in (("vocals", 1), ("accompaniment", -1)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "split": "test",
                            "match_freq_cut": False,
                            "clip": clip,
                            "stem": stem,
                            "sdr_db": sign * value,
                        }
                    )
    return pd.DataFrame(rows)


def test_medians(clips):
    m = rs.medians(clips)
    assert m["musdb7"]["test"]["n_clips"] == 3
    assert m["musdb7"]["test"][rs.HQ3]["vocals"] == 10
    assert m["musdb7_phone"]["test"][rs.MIXTURE]["accompaniment"] == 0


def test_hq3_drop_is_paired_by_clip(clips):
    drop = rs.hq3_drops(clips)["musdb7_phone"]["test"]["vocals"]
    # Per-clip drops 1, 6, -1: median 1, two clips drop. The difference of
    # the medians would be 10 - 8 = 2.
    assert drop == {"median_drop_db": 1.0, "n_clips_dropping": 2, "n_clips": 3}


def test_gain_is_over_the_mixture_of_the_same_dataset(clips):
    gains = rs.improvements(clips)["musdb7_narrowband"]["test"]
    # HQ 3 - mixture on narrowband: 3, 1, 5.
    assert gains[rs.HQ3]["vocals"]["median_gain_db"] == 3.0
    assert gains[rs.HQ3]["vocals"]["n_clips_above_mixture"] == 3
    # Benchmark - mixture: -1, -1, 0.
    assert gains[rs.BENCHMARK]["vocals"] == {
        "median_gain_db": -1.0,
        "n_clips_above_mixture": 0,
        "n_clips": 3,
    }
