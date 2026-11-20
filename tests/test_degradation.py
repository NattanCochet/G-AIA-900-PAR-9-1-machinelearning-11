"""Tests of the T3-06 degradation chain, on synthetic signals only."""

import zlib

import numpy as np
import pytest
import soundfile as sf

from task3_uvr.degradation import (
    SR,
    TARGET_PEAK,
    VARIANTS,
    clip_settings,
    degrade,
    degrade_clip,
)

CLIP_NAMES = [f"Artist {i} - Title {i}" for i in range(50)]


def noise(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal((int(seconds * SR), 2))).astype(np.float32)


def sine(freq: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return np.stack([wave, wave], axis=1).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2)))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_degrade_is_linear(variant):
    _, params, ir = clip_settings("Artist - Song", variant)
    a, b = noise(1.0, seed=1), noise(1.0, seed=2)
    together = degrade(a + b, params, ir)
    apart = degrade(a, params, ir) + degrade(b, params, ir)
    assert together.shape == a.shape
    # float32 rounding only: far below one 16-bit step (3e-5)
    np.testing.assert_allclose(together, apart, atol=1e-6)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_same_clip_name_gives_same_settings(variant):
    seed1, params1, ir1 = clip_settings("Artist - Song", variant)
    seed2, params2, ir2 = clip_settings("Artist - Song", variant)
    assert seed1 == seed2
    assert params1 == params2
    np.testing.assert_array_equal(ir1, ir2)

    _, other, _ = clip_settings("Artist - Other song", variant)
    assert other["rt60"] != params1["rt60"]


def test_phone_keeps_the_course_seed_and_variants_are_independent():
    name = "Artist - Song"
    phone_seed, phone, _ = clip_settings(name, "phone")
    narrow_seed, narrow, _ = clip_settings(name, "narrowband")
    assert phone_seed == [0, zlib.crc32(name.encode())]
    assert narrow_seed[:2] == phone_seed and len(narrow_seed) == 3
    assert phone["rt60"] != narrow["rt60"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_settings_stay_within_their_ranges(variant):
    for name in CLIP_NAMES:
        _, params, ir = clip_settings(name, variant)
        assert list(params) == list(VARIANTS[variant])
        for setting, (lo, hi) in VARIANTS[variant].items():
            assert lo <= params[setting] <= hi
        assert params["low_cut"] < params["high_cut"]
        # 1.5 x rt60 of decaying noise, unit energy per channel
        assert ir.shape == (int(params["rt60"] * 1.5 * SR), 2)
        np.testing.assert_allclose(
            np.sum(ir.astype(np.float64) ** 2, axis=0), 1.0, rtol=1e-5
        )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_energy_outside_the_band_is_attenuated(variant):
    for name in CLIP_NAMES[:10]:
        _, params, ir = clip_settings(name, variant)
        dry = {**params, "wet": 0.0}  # isolate the band-pass from the room
        skip = SR // 5  # filter start-up transient
        levels = {
            f: rms_db(degrade(sine(f), dry, ir)[skip:]) - rms_db(sine(f))
            for f in (40.0, 1000.0, 15000.0)
        }
        # 1 kHz is inside every band; 40 Hz and 15 kHz are more than an
        # octave outside it, where a 4th-order Butterworth edge gives >= 24 dB.
        assert levels[1000.0] > -3.0
        assert levels[40.0] < levels[1000.0] - 24.0
        assert levels[15000.0] < levels[1000.0] - 24.0


def test_narrowband_band_is_fixed():
    for name in CLIP_NAMES:
        _, params, _ = clip_settings(name, "narrowband")
        assert (params["low_cut"], params["high_cut"]) == (300.0, 3400.0)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_degraded_clip_sums_to_its_mixture_with_headroom(tmp_path, variant):
    clip = tmp_path / "Artist - Song"
    clip.mkdir()
    vocals, accompaniment = noise(1.0, seed=3), noise(1.0, seed=4)
    sf.write(str(clip / "vocals.wav"), vocals, SR, subtype="FLOAT")
    sf.write(str(clip / "accompaniment.wav"), accompaniment, SR, subtype="FLOAT")

    stems, metadata = degrade_clip(clip, variant)
    assert {s: a.shape for s, a in stems.items()} == {
        s: vocals.shape for s in ("vocals", "accompaniment", "mixture")
    }
    np.testing.assert_allclose(
        stems["mixture"], stems["vocals"] + stems["accompaniment"], atol=1e-6
    )
    peak = max(float(np.abs(a).max()) for a in stems.values())
    assert peak <= TARGET_PEAK + 1e-6
    assert 0.0 < metadata["gain"] <= 1.0
    assert metadata["variant"] == variant
    assert metadata["ranges"] == {k: list(v) for k, v in VARIANTS[variant].items()}
