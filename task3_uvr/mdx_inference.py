"""T3-04: UVR v5.6's MDX-Net inference, outside the GUI.

A port of SeperateMDX in UVR's separate.py and of the STFT class in
lib_v5/tfc_tdf_v3.py, for ONNX MDX-Net models run with onnxruntime (UVR's
path when the segment size equals the model's dim_t). Line references are to
UVR v5.6.

Supported: the GUI defaults used in this project (segment size = dim_t,
overlap "Default", batch size 1, no denoise, no pitch shift, no secondary
model), with "Match Freq Cut" off (default) or on. Other settings raise instead of
silently differing from UVR. `margin` and `chunks` are assigned by UVR but
never read by SeperateMDX.demix, so they have no equivalent here.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from task3_uvr.onnx_checks import uvr_hash

HOP = 1024  # separate.py:173
DEFAULT_OVERLAP = "Default"
MATCH_MIX_OVERLAP = 0.02  # separate.py:510
MATCH_MIX_SEGMENT = 256  # separate.py:509
FREQ_CUT_STEMS = ("Vocals", "Instrumental")  # constants.MDX_NET_FREQ_CUT
STEM_NAMES = {"Vocals": "vocals", "Instrumental": "accompaniment"}

ChunkFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MdxSettings:
    """Settings UVR reads from model_data (UVR.py:515-519)."""

    n_fft: int
    dim_f: int
    dim_t: int  # frames, i.e. 2 ** mdx_dim_t_set
    compensate: float
    primary_stem: str
    hop: int = HOP

    @property
    def trim(self) -> int:
        return self.n_fft // 2

    @classmethod
    def from_model_data(cls, data: dict) -> "MdxSettings":
        if "config_yaml" in data:
            raise NotImplementedError("MDX-C (config_yaml) models are not supported")
        return cls(
            n_fft=data["mdx_n_fft_scale_set"],
            dim_f=data["mdx_dim_f_set"],
            dim_t=2 ** data["mdx_dim_t_set"],
            compensate=data["compensate"],
            primary_stem=data["primary_stem"],
        )


def model_data_dir(uvr_dir: Path) -> Path:
    return uvr_dir / "models" / "MDX_Net_Models" / "model_data"


def load_settings(model_path: Path, hash_dir: Path) -> tuple[str, MdxSettings]:
    """Look up the model's settings from its hash, as UVR.get_model_data does:
    <hash>.json first, then the first model_data.json key containing the hash.
    """
    model_hash = uvr_hash(model_path)
    own = hash_dir / f"{model_hash}.json"
    if own.is_file():
        return model_hash, MdxSettings.from_model_data(json.loads(own.read_text()))
    mapper = json.loads((hash_dir / "model_data.json").read_text())
    for key, data in mapper.items():
        if model_hash in key:
            return model_hash, MdxSettings.from_model_data(data)
    raise KeyError(f"no UVR settings for {model_path.name} (hash {model_hash})")


class STFT:
    """lib_v5/tfc_tdf_v3.py STFT: [B, 2, T] waves <-> [B, 4, dim_f, frames]."""

    def __init__(self, n_fft: int, hop_length: int, dim_f: int):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)
        self.dim_f = dim_f

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # = return_complex=False in UVR's torch
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape(
            [*batch_dims, c * 2, -1, x.shape[-1]]
        )
        return x[..., : self.dim_f, :]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t])
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.0j
        x = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
        )
        return x.reshape([*batch_dims, 2, -1])


def spectral_chunk_fn(
    settings: MdxSettings, model: Callable[[np.ndarray], np.ndarray] | None
) -> ChunkFn:
    """SeperateMDX.run_model (separate.py:587): STFT, zero the 3 lowest bins,
    run the model on the spectrogram (or skip it, for the Match Freq Cut
    mixture), inverse STFT."""
    stft = STFT(settings.n_fft, settings.hop, settings.dim_f)

    def run(wave: np.ndarray) -> np.ndarray:
        spek = stft(torch.tensor(wave, dtype=torch.float32))
        spek[:, :, :3, :] *= 0
        pred = spek.numpy() if model is None else model(spek.numpy())
        return stft.inverse(torch.tensor(pred)).numpy()

    return run


def demix(
    mix: np.ndarray,
    run_chunk: ChunkFn,
    *,
    chunk_size: int,
    trim: int,
    step: int,
) -> np.ndarray:
    """SeperateMDX.demix (separate.py:503-570), without pitch shift, denoise
    and compensate. `mix` is [2, N] float32; `run_chunk` maps [1, 2,
    chunk_size] to a wave of at least the same length. Returns [2, N]."""
    gen_size = chunk_size - 2 * trim
    pad = gen_size + trim - (mix.shape[-1] % gen_size)
    mixture = np.concatenate(
        (
            np.zeros((2, trim), dtype="float32"),
            mix,
            np.zeros((2, pad), dtype="float32"),
        ),
        1,
    )
    result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

    for start in range(0, mixture.shape[-1], step):
        end = min(start + chunk_size, mixture.shape[-1])
        actual = end - start
        window = np.tile(np.hanning(actual)[None, None, :], (1, 2, 1))
        part = mixture[:, start:end]
        if end != start + chunk_size:
            part = np.concatenate(
                (part, np.zeros((2, start + chunk_size - end), dtype="float32")), -1
            )
        out = np.array(run_chunk(part[None].astype(np.float32)), dtype=np.float32)
        out[..., :actual] *= window
        divider[..., start:end] += window
        result[..., start:end] += out[..., : end - start]

    # The first and last samples have zero window weight (0 / 0); they lie in
    # the trimmed padding, as in UVR.
    with np.errstate(invalid="ignore", divide="ignore"):
        waves = result / divider
    return waves[0, :, trim:-trim][:, : mix.shape[-1]]


def default_step(settings: MdxSettings, chunk_size: int, overlap) -> int:
    """separate.py:523. Note UVR uses the model's chunk size for "Default"."""
    model_chunk = settings.hop * (settings.dim_t - 1)
    if overlap == DEFAULT_OVERLAP:
        return model_chunk - settings.n_fft
    return int((1 - overlap) * chunk_size)


def separate(
    mix: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    settings: MdxSettings,
    *,
    match_freq_cut: bool = False,
) -> dict[str, np.ndarray]:
    """Return {'vocals': [2, N], 'accompaniment': [2, N]} as UVR writes them
    before 16-bit conversion (separate.py:460-486).

    Primary stem = model output * compensate. Secondary stem = mixture -
    primary (the GUI reference run, checked by compare_gui). With
    `match_freq_cut` (UVR's "Match Freq Cut") and a Vocals or Instrumental
    model, it is instead the mixture passed through the STFT round trip
    without the model (3 lowest bins zeroed, bins above dim_f dropped) minus
    primary; that path is not checked against GUI output.
    """
    if settings.primary_stem not in STEM_NAMES:
        raise NotImplementedError(f"primary stem {settings.primary_stem!r}")
    mix = np.asarray(mix, dtype=np.float32)
    chunk_size = settings.hop * (settings.dim_t - 1)
    primary = demix(
        mix,
        spectral_chunk_fn(settings, model),
        chunk_size=chunk_size,
        trim=settings.trim,
        step=default_step(settings, chunk_size, DEFAULT_OVERLAP),
    )
    primary = primary * settings.compensate

    if match_freq_cut and settings.primary_stem in FREQ_CUT_STEMS:
        match_chunk = settings.hop * (MATCH_MIX_SEGMENT - 1)
        raw_mix = demix(
            mix,
            spectral_chunk_fn(settings, None),
            chunk_size=match_chunk,
            trim=settings.trim,
            step=default_step(settings, match_chunk, MATCH_MIX_OVERLAP),
        )
    else:
        raw_mix = mix
    secondary = raw_mix - primary

    primary_name = STEM_NAMES[settings.primary_stem]
    secondary_name = "vocals" if primary_name == "accompaniment" else "accompaniment"
    return {primary_name: primary, secondary_name: secondary}


def onnx_model(model_path: Path, settings: MdxSettings) -> Callable:
    """onnxruntime on CPU, as UVR runs it when segment size = dim_t
    (separate.py:450-452)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    if shape[-1] not in (settings.dim_t, None) and isinstance(shape[-1], int):
        raise NotImplementedError(
            f"model expects {shape[-1]} frames, settings say dim_t={settings.dim_t}"
        )
    return lambda spek: session.run(None, {name: spek})[0]
