"""T3-04 MDX-Net inference port, without UVR or real data: chunking and
reassembly with fake models, the STFT round trip, and the settings lookup."""

import json

import numpy as np
import pytest
import torch

from task3_uvr.mdx_inference import (
    STFT,
    MdxSettings,
    default_step,
    demix,
    load_settings,
    separate,
    spectral_chunk_fn,
)
from task3_uvr.onnx_checks import uvr_hash

# A small model: chunks of 1024 * 7 samples instead of 1024 * 255.
SMALL = MdxSettings(
    n_fft=2048, dim_f=1024, dim_t=8, compensate=1.0, primary_stem="Vocals"
)
CHUNK = SMALL.hop * (SMALL.dim_t - 1)
RNG = np.random.default_rng(0)


def identity(wave: np.ndarray) -> np.ndarray:
    return wave.copy()


@pytest.mark.parametrize(
    "length", [1, 1000, CHUNK - SMALL.n_fft, CHUNK, 3 * CHUNK + 17]
)
@pytest.mark.parametrize("overlap", ["Default", 0.02, 0.5])
def test_demix_with_identity_returns_input(length, overlap):
    mix = RNG.uniform(-1, 1, (2, length)).astype(np.float32)
    out = demix(
        mix,
        identity,
        chunk_size=CHUNK,
        trim=SMALL.trim,
        step=default_step(SMALL, CHUNK, overlap),
    )
    assert out.shape == mix.shape
    np.testing.assert_allclose(out, mix, atol=1e-6)


def test_demix_full_size_chunks_on_a_clip_length():
    # HQ 3's real chunking on a MUSDB18-7 clip length, with the identity.
    hq3 = MdxSettings(6144, 3072, 256, 1.022, "Instrumental")
    chunk = hq3.hop * (hq3.dim_t - 1)
    mix = RNG.uniform(-1, 1, (2, 300032)).astype(np.float32)
    calls = []

    def counting_identity(wave):
        calls.append(wave.shape)
        return wave.copy()

    out = demix(
        mix,
        counting_identity,
        chunk_size=chunk,
        trim=hq3.trim,
        step=default_step(hq3, chunk, "Default"),
    )
    np.testing.assert_allclose(out, mix, atol=1e-6)
    assert calls == [(1, 2, chunk)] * 3


def test_stft_round_trip():
    stft = STFT(n_fft=SMALL.n_fft, hop_length=SMALL.hop, dim_f=SMALL.n_fft // 2 + 1)
    wave = torch.tensor(RNG.standard_normal((1, 2, CHUNK)), dtype=torch.float32)
    spec = stft(wave)
    assert spec.shape == (1, 4, SMALL.n_fft // 2 + 1, SMALL.dim_t)
    torch.testing.assert_close(stft.inverse(spec), wave, atol=1e-5, rtol=0)


def test_spectral_identity_keeps_a_band_limited_tone():
    # A model returning its input only drops bins 0-2 and those above dim_f:
    # a tone between them goes through unchanged (away from the edges).
    t = np.arange(CHUNK) / 44100
    tone = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32)
    out = spectral_chunk_fn(SMALL, lambda spec: spec)(tone[None])[0]
    edge = SMALL.n_fft
    np.testing.assert_allclose(out[:, edge:-edge], tone[:, edge:-edge], atol=1e-4)


def test_separate_stems_and_compensate():
    settings = MdxSettings(2048, 1024, 8, compensate=2.0, primary_stem="Instrumental")
    t = np.arange(3 * CHUNK) / 44100
    mix = np.stack([np.sin(2 * np.pi * 440 * t)] * 2).astype(np.float32) * 0.25
    unit = MdxSettings(2048, 1024, 8, compensate=1.0, primary_stem="Instrumental")
    stems = separate(mix, lambda spec: spec, settings)
    assert set(stems) == {"vocals", "accompaniment"}
    # Primary = model output * compensate; secondary = mixture - primary.
    primary_unit = separate(mix, lambda spec: spec, unit)["accompaniment"]
    np.testing.assert_allclose(stems["accompaniment"], 2 * primary_unit, atol=1e-7)
    np.testing.assert_allclose(stems["vocals"], mix - stems["accompaniment"], atol=1e-7)
    # The identity spectral model is close to the identity, but not exactly:
    # zeroing the 3 lowest bins changes the tone's on/off steps at the clip
    # edges, and each chunk's STFT edge error is only partly hidden by the
    # cross-fade (about 1e-3 here). UVR does the same.
    inner = slice(settings.n_fft, -settings.n_fft)
    np.testing.assert_allclose(primary_unit[:, inner], mix[:, inner], atol=2e-3)


def test_match_freq_cut_removes_dc_from_secondary():
    settings = MdxSettings(2048, 1024, 8, compensate=1.0, primary_stem="Vocals")
    mix = np.full((2, 3 * CHUNK), 0.1, dtype=np.float32)  # pure DC
    zero = lambda spec: np.zeros_like(spec)
    plain = separate(mix, zero, settings)["accompaniment"]
    cut = separate(mix, zero, settings, match_freq_cut=True)["accompaniment"]
    np.testing.assert_allclose(plain, mix)
    # Bins 0-2 zeroed: the DC is gone, except where it switches on and off at
    # the clip edges (see above).
    inner = slice(settings.n_fft, -settings.n_fft)
    assert np.abs(cut[:, inner]).max() < 1e-3
    assert np.abs(cut).max() < 0.1


def write_model(tmp_path, content: bytes):
    path = tmp_path / "model.onnx"
    path.write_bytes(content)
    hash_dir = tmp_path / "model_data"
    hash_dir.mkdir()
    return path, hash_dir, uvr_hash(path)


DATA = {
    "compensate": 1.022,
    "mdx_dim_f_set": 3072,
    "mdx_dim_t_set": 8,
    "mdx_n_fft_scale_set": 6144,
    "primary_stem": "Instrumental",
}


def test_settings_from_hash_json_first(tmp_path):
    path, hash_dir, h = write_model(tmp_path, b"fake model")
    (hash_dir / f"{h}.json").write_text(json.dumps(DATA))
    (hash_dir / "model_data.json").write_text(
        json.dumps({h: {**DATA, "compensate": 9}})
    )
    model_hash, settings = load_settings(path, hash_dir)
    assert model_hash == h
    assert settings == MdxSettings(6144, 3072, 256, 1.022, "Instrumental")


def test_settings_fall_back_to_model_data_json(tmp_path):
    path, hash_dir, h = write_model(tmp_path, b"another fake model")
    (hash_dir / "model_data.json").write_text(json.dumps({"other": {}, h: DATA}))
    assert load_settings(path, hash_dir)[1].compensate == 1.022


def test_unknown_model_raises(tmp_path):
    path, hash_dir, _ = write_model(tmp_path, b"unknown")
    (hash_dir / "model_data.json").write_text("{}")
    with pytest.raises(KeyError):
        load_settings(path, hash_dir)


def test_mdx_c_model_is_refused():
    with pytest.raises(NotImplementedError):
        MdxSettings.from_model_data({"config_yaml": "model_2_stem_061321.yaml"})
