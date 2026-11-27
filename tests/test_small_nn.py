"""T3-08: SmallMaskNet, its data and its training loop, on synthetic data."""

import json
import random

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest
import soundfile as sf
import torch

from task3_uvr import small_nn_data, train_small_nn
from task3_uvr.small_nn import (
    DIM_F,
    DIM_T,
    N_SAMPLES_IN_CHUNK,
    SmallMaskNet,
    export_onnx,
    model_input,
    n_parameters,
    uvr_settings,
)
from task3_uvr.small_nn_data import make_batch, split_names

ALLOWED_OPS = {
    "Constant",
    "Slice",
    "Mul",
    "Add",
    "Conv",
    "Relu",
    "Concat",
    "Sqrt",
    "Log",
    "Sigmoid",
}
SAMPLES = N_SAMPLES_IN_CHUNK + 4000
TRAIN = ["Artist A - One", "Artist B - Two", "Artist C - Three"]
VALIDATION = ["Artist D - Four", "Artist E - Five"]
UNLISTED = ["Artist F - Six"]


def test_output_shape_and_size():
    model = SmallMaskNet()
    with torch.no_grad():
        out = model(torch.randn(2, 4, DIM_F, DIM_T))
    assert out.shape == (2, 4, DIM_F, DIM_T)
    assert n_parameters(model) == 37_746


def test_mask_stays_in_unit_interval():
    torch.manual_seed(0)
    model = SmallMaskNet(dim_f=64)
    with torch.no_grad():
        for p in model.parameters():  # large weights push the sigmoid to 0 and 1
            p.normal_(0, 5)
        spec = 100 * torch.randn(3, 4, 64, 32)
        mask = model.mask(spec)
        out = model(spec)
    assert mask.shape == (3, 2, 64, 32)
    assert mask.min() >= 0 and mask.max() <= 1
    assert mask.min() < 0.01 and mask.max() > 0.99
    # Each channel's real and imaginary parts are scaled by the same mask.
    for c, m in ((0, 0), (1, 0), (2, 1), (3, 1)):
        assert torch.allclose(out[:, c], spec[:, c] * mask[:, m])


def test_onnx_export_round_trip(tmp_path):
    torch.manual_seed(0)
    model = SmallMaskNet()
    path = tmp_path / "small.onnx"
    export_onnx(model, path)
    graph = onnx.load(str(path))
    assert graph.opset_import[0].version <= 17 and graph.ir_version <= 8
    assert {node.op_type for node in graph.graph.node} <= ALLOWED_OPS

    x = torch.randn(2, 4, DIM_F, DIM_T)
    with torch.no_grad():
        expected = model(x).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"input": x.numpy()})[0]
    assert np.abs(got - expected).max() < 1e-4 * np.abs(expected).max()
    assert model.training is True  # export leaves the model as it found it


def test_uvr_settings():
    assert uvr_settings() == {
        "compensate": 1.0,
        "mdx_dim_f_set": 2048,
        "mdx_dim_t_set": 8,
        "mdx_n_fft_scale_set": 4096,
        "primary_stem": "Vocals",
    }


def test_model_input_zeroes_lowest_bins():
    spec = model_input(torch.randn(1, 2, N_SAMPLES_IN_CHUNK))
    assert spec.shape == (1, 4, DIM_F, DIM_T)
    assert spec[:, :, :3].abs().max() == 0 and spec[:, :, 3].abs().max() > 0


# -- synthetic dataset --------------------------------------------------------


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """musdb7-like folder. Clip k's vocals are the constant k + 1 and its
    accompaniment -(k + 1) / 10, so a chunk tells which clip it came from."""
    musdb7 = tmp_path / "musdb7"
    for k, name in enumerate(TRAIN + VALIDATION + UNLISTED):
        clip = musdb7 / "train" / name
        clip.mkdir(parents=True)
        value = float(k + 1)
        for stem, level in (("vocals", value), ("accompaniment", -value / 10)):
            sf.write(
                clip / f"{stem}.wav",
                np.full((SAMPLES, 2), level, "float32"),
                44100,
                subtype="FLOAT",
            )
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, names in (("train", TRAIN), ("validation", VALIDATION)):
        (splits / f"{split}.json").write_text(json.dumps({"clips": names}))
    monkeypatch.setattr(small_nn_data, "SPLITS_DIR", splits)
    return musdb7


def make_args(musdb7, tmp_path, *extra):
    return train_small_nn.parse_args(
        [
            "--data-dir", str(musdb7),
            "--device", "cpu",
            "--batch-size", "2",
            "--steps", "4",
            "--eval-every", "2",
            "--checkpoint-every", "2",
            "--val-chunks-per-clip", "1",
            "--channels", "4",
            "--checkpoint", str(tmp_path / "run" / "ck.pt"),
            *extra,
        ]
    )  # fmt: skip


def test_real_splits_are_disjoint():
    train, validation = split_names("train"), split_names("validation")
    assert len(train) == 80 and len(validation) == 14
    assert not set(train) & set(validation)


def test_training_reads_only_train_split_clips(dataset, tmp_path):
    trainer = train_small_nn.Trainer(make_args(dataset, tmp_path))
    vocal_levels = sorted(float(v[0, 0]) for v, _ in trainer.train_clips)
    assert vocal_levels == [1.0, 2.0, 3.0]  # TRAIN only
    rng = random.Random(0)
    for remix in (True, False):
        for _ in range(20):
            _, vocals = make_batch(trainer.train_clips, 4, remix, rng)
            assert set(vocals[:, 0, 0].tolist()) <= {1.0, 2.0, 3.0}
    # Validation chunks come from the validation list.
    assert set(trainer.val_vocals[:, 0, 0].tolist()) == {4.0, 5.0}


def test_no_remix_keeps_each_clip_whole(dataset, tmp_path):
    trainer = train_small_nn.Trainer(make_args(dataset, tmp_path))
    mix, vocals = make_batch(trainer.train_clips, 8, False, random.Random(1))
    assert torch.allclose(mix, vocals * 0.9)  # v + a with a = -v / 10
    mix, vocals = make_batch(trainer.train_clips, 64, True, random.Random(1))
    assert not torch.allclose(mix, vocals * 0.9)  # some vocals over another clip


def test_resume_gives_the_same_next_batch_and_weights(dataset, tmp_path):
    straight = train_small_nn.Trainer(make_args(dataset, tmp_path / "a"))
    straight.run(stop_after=2)
    expected_state = straight.rng.getstate()  # draws the batch of step 3
    straight.run()  # steps 3-4

    args = make_args(dataset, tmp_path / "b")
    first = train_small_nn.Trainer(args)
    first.run(stop_after=2)  # saves at step 2
    del first
    args.resume = True
    resumed = train_small_nn.Trainer(args)
    resumed.restore()
    assert resumed.step == 2
    assert resumed.rng.getstate() == expected_state
    copy = random.Random()
    copy.setstate(expected_state)
    expected_batch = make_batch(resumed.train_clips, 2, True, copy)
    peek = random.Random()
    peek.setstate(resumed.rng.getstate())
    for got, want in zip(
        make_batch(resumed.train_clips, 2, True, peek), expected_batch, strict=True
    ):
        assert torch.equal(got, want)

    resumed.run()
    assert resumed.step == 4
    for p, q in zip(
        resumed.model.parameters(), straight.model.parameters(), strict=True
    ):
        assert torch.allclose(p, q, atol=1e-6)
    log_a = pd.read_csv(train_small_nn.log_path(straight.args.checkpoint))
    log_b = pd.read_csv(train_small_nn.log_path(args.checkpoint))
    assert log_b["step"].tolist() == [2, 4]
    assert np.allclose(
        log_a[["train_loss", "val_loss"]], log_b[["train_loss", "val_loss"]], atol=1e-6
    )


def test_resume_refuses_other_settings(dataset, tmp_path):
    args = make_args(dataset, tmp_path)
    train_small_nn.Trainer(args).run(stop_after=2)
    other = make_args(dataset, tmp_path, "--no-remix", "--resume")
    with pytest.raises(SystemExit, match="remix"):
        train_small_nn.Trainer(other).restore()


def test_resume_drops_log_rows_after_the_checkpoint(dataset, tmp_path):
    args = make_args(dataset, tmp_path, "--checkpoint-every", "4")
    trainer = train_small_nn.Trainer(args)
    trainer.run(stop_after=2)  # checkpoint at step 2 (end of session), log row 2
    train_small_nn.append_log(
        train_small_nn.log_path(args.checkpoint),
        {"step": 3, "train_loss": 0, "val_loss": 0, "train_seconds": 0},
    )  # as if the session died after logging step 3
    args.resume = True
    resumed = train_small_nn.Trainer(args)
    resumed.restore()
    assert pd.read_csv(train_small_nn.log_path(args.checkpoint))["step"].tolist() == [2]


def test_save_clip_rows_adds_columns_and_replaces_same_run(tmp_path):
    path = tmp_path / "clips.csv"
    path.write_text(
        "date,model,model_hash,split,match_freq_cut,clip,stem,sdr_db,reference_energy,seconds\n"
        "2026-11-13,mixture,,validation,False,A,vocals,-3.9650826098752523,2312.2,2.9e-06\n"
    )
    row = {
        "date": "2026-11-27",
        "dataset": "musdb7",
        "model": "SmallMaskNet_musdb7_remix",
        "model_hash": "abc",
        "variant": "remix",
        "split": "validation",
        "match_freq_cut": False,
        "clip": "A",
        "stem": "vocals",
        "sdr_db": 1.0,
        "reference_energy": 2312.2,
        "seconds": 0.1,
    }
    train_small_nn.save_clip_rows(pd.DataFrame([row]), path)
    train_small_nn.save_clip_rows(pd.DataFrame([{**row, "sdr_db": 2.0}]), path)
    out = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(out.columns) == train_small_nn.CLIP_COLUMNS
    assert out["model"].tolist() == ["mixture", "SmallMaskNet_musdb7_remix"]
    assert out["dataset"].tolist() == ["musdb7", "musdb7"]
    assert out["variant"].tolist() == ["", "remix"]
    # The old row keeps its exact text; the first new row is replaced.
    assert out["sdr_db"].tolist() == ["-3.9650826098752523", "2.0"]
