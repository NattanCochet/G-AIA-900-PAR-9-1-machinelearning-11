"""T3-08: training and validation data for the small network.

load_clips, random_chunk and make_batch are ported from the course file
ml-tek5-project/2_3_uvr/uvr_my_models/src/data_preparation.py. Two changes:

- every random draw comes from a `random.Random` passed in, not the global
  `random` module, so that its state can be saved in the checkpoint and a
  resumed run draws exactly the batches the uninterrupted run would have;
- clips are listed from task3_uvr/splits/*.json instead of a whole folder, so
  training never sees the validation clips (the course used all 94
  training clips).
"""

import json
import random
import sys
from pathlib import Path

import soundfile as sf
import torch

from task3_uvr.small_nn import N_SAMPLES_IN_CHUNK

SPLITS_DIR = Path(__file__).parent / "splits"
DATASETS = ("musdb7", "musdb7_phone", "musdb7_narrowband")

Clip = tuple[torch.Tensor, torch.Tensor]  # (vocals, accompaniment), [2, samples]


def dataset_dir(musdb7_dir: Path, dataset: str) -> Path:
    """musdb7 itself, or a degraded copy next to it (T3-06): the same rule
    as T3-07's evaluate --dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {DATASETS}")
    path = musdb7_dir if dataset == DATASETS[0] else musdb7_dir.parent / dataset
    if not path.is_dir():
        sys.exit(f"Not a folder: {path}")
    return path


def split_names(split: str, splits_dir: Path | None = None) -> list[str]:
    """Clip names of the training or validation list (both from the official
    training folder)."""
    if split not in ("train", "validation"):
        raise ValueError(f"no clip list for split {split!r}")
    splits_dir = SPLITS_DIR if splits_dir is None else splits_dir
    return json.loads((splits_dir / f"{split}.json").read_text())["clips"]


def load(path: Path) -> torch.Tensor:
    return torch.tensor(sf.read(str(path), dtype="float32")[0].T)  # [2, samples]


def load_clips(data_dir: Path, names: list[str]) -> list[Clip]:
    """(vocals, accompaniment) of the named clips of <data_dir>/train/."""
    clips = []
    for name in names:
        clip = data_dir / "train" / name
        clips.append((load(clip / "vocals.wav"), load(clip / "accompaniment.wav")))
    return clips


def random_chunk(wave: torch.Tensor, rng: random.Random) -> torch.Tensor:
    start = rng.randint(0, wave.shape[1] - N_SAMPLES_IN_CHUNK)
    return wave[:, start : start + N_SAMPLES_IN_CHUNK]


def make_batch(
    clips: list[Clip], batch_size: int, remix: bool, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor]:
    """(mixture, vocals) waveforms, each [batch_size, 2, N_SAMPLES_IN_CHUNK].

    remix=True: the vocals of one clip over the accompaniment of another,
    each from its own random position (data augmentation: many new songs
    from few). remix=False: one clip, one position, its real mixture.
    """
    mixes, vocals = [], []
    for _ in range(batch_size):
        v, a = rng.choice(clips)
        if remix:
            a = rng.choice(clips)[1]
            v, a = random_chunk(v, rng), random_chunk(a, rng)
        else:
            start = rng.randint(0, v.shape[1] - N_SAMPLES_IN_CHUNK)
            v = v[:, start : start + N_SAMPLES_IN_CHUNK]
            a = a[:, start : start + N_SAMPLES_IN_CHUNK]
        vocals.append(v)
        mixes.append(v + a)
    return torch.stack(mixes), torch.stack(vocals)


def validation_chunks(
    clips: list[Clip], chunks_per_clip: int = 2, seed: int = 1234
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed (mixture, vocals) chunks: `chunks_per_clip` positions per clip,
    drawn once with `seed` and never remixed, so every validation loss of
    every run is measured on the same audio."""
    rng = random.Random(seed)
    mixes, vocals = [], []
    for v, a in clips:
        for _ in range(chunks_per_clip):
            start = rng.randint(0, v.shape[1] - N_SAMPLES_IN_CHUNK)
            v_chunk = v[:, start : start + N_SAMPLES_IN_CHUNK]
            vocals.append(v_chunk)
            mixes.append(v_chunk + a[:, start : start + N_SAMPLES_IN_CHUNK])
    return torch.stack(mixes), torch.stack(vocals)
