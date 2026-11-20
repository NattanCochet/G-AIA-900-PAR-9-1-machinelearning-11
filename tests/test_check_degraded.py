"""Tests of the degraded-dataset check, on a tiny synthetic musdb7."""

import numpy as np
import soundfile as sf

from task3_uvr.check_degraded import check_variant
from task3_uvr.degradation import LSB, SR, make_variant, variant_dir


def fake_musdb7(root):
    """Two 0.5 s clips per split, with mixture = vocals + accompaniment."""
    data_dir = root / "musdb7"
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for i in range(2):
            clip = data_dir / split / f"Artist {split} {i} - Song"
            clip.mkdir(parents=True)
            vocals = 0.1 * rng.standard_normal((SR // 2, 2))
            accompaniment = 0.1 * rng.standard_normal((SR // 2, 2))
            for stem, audio in (
                ("vocals", vocals),
                ("accompaniment", accompaniment),
                ("mixture", vocals + accompaniment),
            ):
                sf.write(str(clip / f"{stem}.wav"), audio, SR, subtype="PCM_16")
    return data_dir


def test_a_fresh_variant_passes(tmp_path):
    data_dir = fake_musdb7(tmp_path)
    make_variant(data_dir, "narrowband")
    summary = check_variant(data_dir, "narrowband")
    assert summary["ok"], summary["anomalies"]
    for split in ("train", "test"):
        assert summary["splits"][split]["n_clips"] == 2
        assert summary["splits"][split]["max_abs_residual"] <= 2 * LSB
    assert summary["observed_ranges"]["low_cut"] == [300.0, 300.0]


def test_broken_clips_are_reported(tmp_path):
    data_dir = fake_musdb7(tmp_path)
    make_variant(data_dir, "phone")
    root = variant_dir(data_dir, "phone")

    # 1. accompaniment no longer sums with vocals to the mixture
    clip = root / "train" / "Artist train 0 - Song"
    audio, _ = sf.read(str(clip / "accompaniment.wav"), always_2d=True)
    sf.write(str(clip / "accompaniment.wav"), audio + 10 * LSB, SR, subtype="PCM_16")
    # 2. degradation.json missing
    (root / "train" / "Artist train 1 - Song" / "degradation.json").unlink()
    # 3. a clip missing from the degraded copy
    for f in (root / "test" / "Artist test 0 - Song").iterdir():
        f.unlink()
    (root / "test" / "Artist test 0 - Song").rmdir()

    summary = check_variant(data_dir, "phone")
    assert not summary["ok"]
    text = "\n".join(summary["anomalies"])
    assert "train/Artist train 0 - Song: max abs residual" in text
    assert "train/Artist train 1 - Song: missing degradation.json" in text
    assert "test/: missing clips ['Artist test 0 - Song']" in text
