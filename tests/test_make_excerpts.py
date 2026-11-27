"""T3-07 clip choice and compare-file layout of task3_uvr.make_excerpts."""

import numpy as np
import pandas as pd
import pytest

from task3_uvr import make_excerpts as mx


def row(dataset, model, clip, stem, sdr_db=0.0, energy=1.0, split="test"):
    return {
        "dataset": dataset,
        "model": model,
        "split": split,
        "clip": clip,
        "stem": stem,
        "sdr_db": sdr_db,
        "reference_energy": energy,
    }


@pytest.fixture
def clips():
    rows = []
    # HQ 3 vocal SDR, clean then phone: drops of 1, 5 and 3 dB.
    for clip, clean, phone in (("a", 10, 9), ("b", 12, 7), ("c", 8, 5)):
        rows += [
            row("musdb7", mx.HQ3, clip, "vocals", clean),
            row("musdb7_phone", mx.HQ3, clip, "vocals", phone),
            row("musdb7", mx.HQ3, clip, "accompaniment", 0.0),
            row("musdb7_phone", mx.HQ3, clip, "accompaniment", -50.0),
        ]
    # Vocals / accompaniment energy: c is the loudest (2x), then a (1x).
    for clip, vocals in (("a", 1.0), ("b", 0.5), ("c", 2.0)):
        rows += [
            row("musdb7", "mixture", clip, "vocals", energy=vocals),
            row("musdb7", "mixture", clip, "accompaniment", energy=1.0),
            row("musdb7_phone", "mixture", clip, "vocals", energy=100.0),
            row("musdb7_phone", "mixture", clip, "accompaniment", energy=1.0),
        ]
    # A validation clip with a bigger drop must be ignored.
    rows += [
        row("musdb7", mx.HQ3, "v", "vocals", 20, split="validation"),
        row("musdb7_phone", mx.HQ3, "v", "vocals", 0, split="validation"),
    ]
    return pd.DataFrame(rows)


def test_largest_drop_uses_test_vocals_of_hq3(clips):
    assert mx.largest_drop_clip(clips) == ("b", pytest.approx(5.0))


def test_largest_drop_skips_excluded_clips(clips):
    assert mx.largest_drop_clip(clips, exclude=["b"]) == ("c", pytest.approx(3.0))


def test_loudest_vocals_uses_clean_mixture_energies(clips):
    clip, ratio = mx.loudest_vocals_clip(clips)
    assert clip == "c"
    assert ratio == pytest.approx(10 * np.log10(2))


def test_default_clips_are_distinct_and_start_with_al_james(clips):
    clips = pd.concat(
        [
            clips,
            pd.DataFrame(
                [
                    row("musdb7", "mixture", mx.REFERENCE_CLIP, "vocals", energy=9.0),
                    row("musdb7", "mixture", mx.REFERENCE_CLIP, "accompaniment"),
                ]
            ),
        ]
    )
    chosen = list(mx.default_clips(clips))
    # Al James has the loudest vocals here, so the third pick is the next one.
    assert chosen == [mx.REFERENCE_CLIP, "b", "c"]


def test_with_gaps_puts_silence_between_parts_only():
    parts = [np.ones((10, 2)), 2 * np.ones((20, 2)), 3 * np.ones((5, 2))]
    joined = mx.with_gaps(parts, gap_seconds=4 / mx.SR)
    assert joined.shape == (10 + 4 + 20 + 4 + 5, 2)
    assert np.all(joined[10:14] == 0) and np.all(joined[34:38] == 0)
    assert joined[0, 0] == 1 and joined[14, 0] == 2 and joined[-1, 0] == 3
