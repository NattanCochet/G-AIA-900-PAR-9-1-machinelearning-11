"""T3-04 global SDR on known signals."""

import numpy as np
import pytest

from task3_uvr.sdr import EPS, sdr

RNG = np.random.default_rng(0)
S = RNG.standard_normal((2, 44100))


def test_zero_estimate_is_0_db():
    assert sdr(S, np.zeros_like(S)) == pytest.approx(0.0, abs=1e-9)


def test_half_scale_estimate_is_6_db():
    # error = 0.5 s, so SDR = 10 log10(1 / 0.25)
    assert sdr(S, 0.5 * S) == pytest.approx(10 * np.log10(4), abs=1e-6)


def test_noise_10_db_below_signal():
    noise = RNG.standard_normal(S.shape)
    noise *= np.sqrt(np.sum(S**2) / np.sum(noise**2) / 10)
    assert sdr(S, S + noise) == pytest.approx(10.0, abs=1e-6)


def test_perfect_estimate_is_capped_by_eps():
    assert sdr(S, S) == pytest.approx(10 * np.log10((np.sum(S**2) + EPS) / EPS))


def test_silent_reference_and_estimate_is_0_db():
    silence = np.zeros_like(S)
    assert sdr(silence, silence) == 0.0


def test_sums_over_channels():
    # A perfect left channel and a zero right channel, equal energies: the
    # error is half the signal energy, so 3.01 dB.
    ref = np.stack([S[0], S[0]])
    est = np.stack([S[0], np.zeros_like(S[0])])
    assert sdr(ref, est) == pytest.approx(10 * np.log10(2), abs=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sdr(S, S[:, :-1])
