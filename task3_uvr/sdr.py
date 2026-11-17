"""Global SDR, as in the Music Demixing Challenge (2021).

    SDR = 10 log10((sum s^2 + eps) / (sum (s - s_hat)^2 + eps))

summed over all channels and samples of one stem of one clip. Unlike
museval's median over 1 s frames, it is defined even when the stem is silent
for part of the clip, which matters on 6.8 s MUSDB18-7 excerpts (about 7
frames each).
"""

import numpy as np

EPS = 1e-7  # the MDX Challenge value


def sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = EPS) -> float:
    """Global SDR in dB of `estimate` against `reference` (same shape)."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError(f"shapes differ: {reference.shape} vs {estimate.shape}")
    signal = np.sum(reference**2)
    error = np.sum((reference - estimate) ** 2)
    return float(10 * np.log10((signal + eps) / (error + eps)))
