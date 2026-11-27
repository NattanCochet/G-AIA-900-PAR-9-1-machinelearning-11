"""T3-10: HQ 3 fine-tuning, on a tiny ONNX U-Net and synthetic audio.

The tiny model has HQ 3's kinds of layers (Conv, BatchNorm, a stride-2 Conv,
ConvTranspose, a multiplicative skip) and its input shape [B, 4, 3072, 256],
so the whole pipeline runs with HQ 3's real STFT settings, without UVR or
the real model.
"""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch import nn

from task3_uvr import evaluate, finetune, small_nn_data
from task3_uvr.finetune import CHUNK, SETTINGS
from task3_uvr.hq3_trainable import HQ3_SETTINGS
from task3_uvr.mdx_inference import onnx_model, spectral_chunk_fn
from task3_uvr.onnx_checks import compare_outputs

SAMPLES = CHUNK + 4000
TRAIN = ["Artist A - One", "Artist B - Two", "Artist C - Three"]
VALIDATION = ["Artist D - Four", "Artist E - Five"]
UNLISTED = ["Artist F - Six"]
PHONE_GAIN = 0.5  # musdb7_phone = 0.5 x musdb7: linear, like T3-06


class TinyUNet(nn.Module):
    """input conv -> BatchNorm -> stride-2 conv -> ConvTranspose ->
    BatchNorm -> x skip -> output conv. BatchNorm follows a ReLU so that the
    exporter cannot fold it into a convolution."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Conv2d(4, 4, 1)
        self.norm1 = nn.BatchNorm2d(4)
        self.down = nn.Conv2d(4, 4, 2, stride=2)
        self.up = nn.ConvTranspose2d(4, 4, 2, stride=2)
        self.norm2 = nn.BatchNorm2d(4)
        self.out = nn.Conv2d(4, 4, 1)

    def forward(self, x):
        h = self.norm1(torch.relu(self.inp(x)))
        u = self.norm2(torch.relu(self.up(torch.relu(self.down(h)))))
        return self.out(u * h)


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    torch.manual_seed(0)
    source = TinyUNet()
    # Trained-looking BatchNorm, as in HQ 3. With the default weight 1 and
    # bias 0, the exporter merges norm2's tensors into norm1's through
    # Identity nodes, and onnx2pytorch then overwrites norm2's parameters
    # with those inputs on every forward.
    with torch.no_grad():
        for norm in (source.norm1, source.norm2):
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
            norm.running_mean.uniform_(-1, 1)
            norm.running_var.uniform_(0.5, 2)
    source.eval()
    path = tmp_path_factory.mktemp("model") / "tiny.onnx"
    torch.onnx.export(
        source, (torch.zeros(1, 4, SETTINGS.dim_f, SETTINGS.dim_t),), str(path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13, dynamo=False,
    )  # fmt: skip
    return path


def level(k: int) -> float:
    return float(k + 1)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """musdb7 and musdb7_phone side by side. Clip k's vocals are noise
    around k + 1 and its accompaniment noise around -(k + 1) / 10, so the
    mean of a chunk tells which clip (and dataset) it came from."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for k, name in enumerate(TRAIN + VALIDATION + UNLISTED):
        stems = {
            "vocals": level(k) + 0.1 * rng.standard_normal((SAMPLES, 2)),
            "accompaniment": -level(k) / 10 + 0.1 * rng.standard_normal((SAMPLES, 2)),
        }
        stems["mixture"] = stems["vocals"] + stems["accompaniment"]
        for dataset, gain in (("musdb7", 1.0), ("musdb7_phone", PHONE_GAIN)):
            clip = root / dataset / "train" / name
            clip.mkdir(parents=True)
            for stem, wave in stems.items():
                sf.write(
                    clip / f"{stem}.wav",
                    (gain * wave).astype("float32"),
                    44100,
                    subtype="FLOAT",
                )
    splits = root / "splits"
    splits.mkdir()
    for split, names in (("train", TRAIN), ("validation", VALIDATION)):
        (splits / f"{split}.json").write_text(json.dumps({"clips": names}))
    return root


@pytest.fixture
def data(data_root, monkeypatch):
    monkeypatch.setattr(small_nn_data, "SPLITS_DIR", data_root / "splits")
    monkeypatch.setattr(evaluate, "SPLITS_DIR", data_root / "splits")
    for var in (finetune.ORT113_ENV_VAR, finetune.UVR_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    return data_root / "musdb7"


def make_args(musdb7, model, tmp_path, *extra):
    return finetune.parse_args(
        [
            "--data-dir", str(musdb7),
            "--model", str(model),
            "--dataset", "musdb7_phone",
            "--device", "cpu",
            "--batch-size", "2",
            "--steps", "4",
            "--warmup-steps", "3",
            "--lr", "1e-3",
            "--eval-every", "2",
            "--checkpoint-every", "2",
            "--val-chunks-per-clip", "1",
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--results-dir", str(tmp_path / "results"),
            "--checkpoint", str(tmp_path / "run" / "ck.pt"),
            *extra,
        ]
    )  # fmt: skip


def parameters(trainer) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in trainer.model.named_parameters()}


# -- model -----------------------------------------------------------------------


def test_batchnorm_stays_in_eval_and_keeps_its_statistics(data, tiny_onnx, tmp_path):
    trainer = finetune.Trainer(make_args(data, tiny_onnx, tmp_path))
    norms = finetune.batchnorm_modules(trainer.model)
    assert len(norms) == 2
    stats = [(n.running_mean.clone(), n.running_var.clone()) for n in norms]
    trainer.run()
    assert trainer.step == 4
    assert finetune.weight_change(trainer)["n_tensors_changed"] > 0
    for norm, (mean, var) in zip(norms, stats, strict=True):
        assert not norm.training
        assert torch.equal(norm.running_mean, mean) and torch.equal(
            norm.running_var, var
        )
    trainer.model.train()  # what the script must never do
    with pytest.raises(RuntimeError, match="train mode"):
        trainer.train_step()


def test_training_input_and_output_match_inference(data, tiny_onnx, tmp_path):
    """The training estimate of one chunk is UVR's primary stem for that
    chunk: mdx_inference's STFT, bins 0-2 zeroed, the model (here the ONNX
    file in onnxruntime), iSTFT, times compensate."""
    trainer = finetune.Trainer(make_args(data, tiny_onnx, tmp_path))
    mix = trainer.val[0][:1]
    with torch.no_grad():
        got = trainer.estimate(mix).numpy()
    run_chunk = spectral_chunk_fn(SETTINGS, onnx_model(tiny_onnx, SETTINGS))
    expected = SETTINGS.compensate * run_chunk(mix.numpy())
    assert got.shape == expected.shape == (1, 2, CHUNK)
    assert compare_outputs(expected, got)["identical"]


def test_lr_zero_exports_the_original_model(data, tiny_onnx, tmp_path):
    trainer = finetune.Trainer(make_args(data, tiny_onnx, tmp_path, "--lr", "0"))
    trainer.run()
    assert finetune.weight_change(trainer)["n_tensors_changed"] == 0
    report = finetune.export_and_check(trainer, tmp_path / "ft.onnx")
    assert report["export_vs_original_hq3"]["identical"]
    assert report["export_vs_original_hq3"]["max_abs_diff"] < 1e-5
    assert report["n_initializers_changed"] == 0
    assert report["parity_pytorch_vs_export"]["identical"]
    assert report["opset"] <= 17 and report["ir_version"] <= 8
    assert report["onnxruntime_1_13"]["ran"] is False  # no Python given


def test_real_lr_changes_every_tensor_and_the_export(data, tiny_onnx, tmp_path):
    trainer = finetune.Trainer(make_args(data, tiny_onnx, tmp_path))
    trainer.run()
    change = finetune.weight_change(trainer)
    assert change["n_tensors_changed"] == change["n_trainable_tensors"] == 12
    report = finetune.export_and_check(trainer, tmp_path / "ft.onnx")
    assert report["n_initializers_changed"] == 12
    assert report["parity_pytorch_vs_export"]["identical"]
    assert not report["export_vs_original_hq3"]["identical"]
    settings = json.loads((tmp_path / report["settings_file"]).read_text())
    assert settings == HQ3_SETTINGS
    assert report["settings_file"].endswith(f"model_data/{report['uvr_hash']}.json")


@pytest.mark.parametrize(
    ("freeze", "frozen"),
    [
        ("none", []),
        ("encoder", ["inp", "norm1", "down"]),
        ("all-but-last-2", ["inp", "norm1", "down", "up"]),
    ],
)
def test_freezing_really_freezes(data, tiny_onnx, tmp_path, freeze, frozen):
    trainer = finetune.Trainer(make_args(data, tiny_onnx, tmp_path, "--freeze", freeze))
    # onnx2pytorch names layers after the ONNX nodes: map them to TinyUNet's.
    names = [name for name, _ in finetune.parameter_layers(trainer.model)]
    tiny = ["inp", "norm1", "down", "up", "norm2", "out"]
    assert len(names) == len(tiny)
    assert [tiny[names.index(n)] for n in trainer.frozen_layers] == frozen
    before = parameters(trainer)
    trainer.run()
    frozen_prefixes = tuple(f"{n}." for n in trainer.frozen_layers)
    n_optimised = sum(len(g["params"]) for g in trainer.optimizer.param_groups)
    assert n_optimised == len(trainer.trainable)
    for name, p in trainer.model.named_parameters():
        if name.startswith(frozen_prefixes):
            assert not p.requires_grad and torch.equal(p, before[name]), name
        else:
            assert p.requires_grad and not torch.equal(p, before[name]), name
    assert finetune.weight_change(trainer)["frozen_tensors_changed"] == []


def test_parse_freeze():
    for value in ("none", "encoder", "all-but-last-1", "all-but-last-12"):
        assert finetune.parse_freeze(value) == value
    for value in ("decoder", "all-but-last-0", "all-but-last-", "all-but-last-x"):
        with pytest.raises(Exception, match="expected none"):
            finetune.parse_freeze(value)


def test_resolve_hq3_rejects_another_model(tiny_onnx):
    with pytest.raises(SystemExit, match="is not HQ 3"):
        finetune.resolve_hq3(str(tiny_onnx))


def test_warmup_is_linear_then_constant():
    factor = finetune.warmup_factor(4)
    assert [factor(k) for k in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    assert finetune.warmup_factor(0)(0) == 1.0


def test_names():
    args = finetune.parse_args(["--model", "x.onnx", "--data-dir", "."])
    assert finetune.run_name(args) == "HQ3_ft_musdb7_phone_lr1e-06"
    assert finetune.export_name(args, 2000) == "HQ3_ft_musdb7_phone_lr1e-06_s2000"
    assert args.checkpoint.name == "HQ3_ft_musdb7_phone_lr1e-06.pt"
    args = finetune.parse_args(
        [
            "--dataset",
            "musdb7_narrowband",
            "--freeze",
            "encoder",
            "--remix",
            "--seed",
            "3",
        ]
    )
    assert (
        finetune.run_name(args)
        == "HQ3_ft_musdb7_narrowband_lr1e-06_encoder_remix_seed3"
    )


# -- data ------------------------------------------------------------------------


def test_training_reads_only_train_split_clips(data, tiny_onnx, tmp_path):
    trainer = finetune.Trainer(make_args(data, tiny_onnx, tmp_path, "--remix"))
    phone_levels = {PHONE_GAIN * level(k) for k in range(len(TRAIN))}
    clip_levels = {round(float(v.mean()), 2) for v, _ in trainer.train_clips}
    assert clip_levels == phone_levels  # TRAIN only, from musdb7_phone
    seen = set()
    for _ in range(20):
        _, vocals = small_nn_data.make_batch(trainer.train_clips, 4, True, trainer.rng)
        seen |= {round(float(v.mean()), 2) for v in vocals}
    assert seen <= phone_levels
    # Validation chunks: the validation clips, on the dataset and on clean musdb7.
    val_levels = {level(k) for k in range(len(TRAIN), len(TRAIN) + len(VALIDATION))}
    mix, accompaniment = trainer.val
    assert {round(float(v.mean()), 2) for v in mix - accompaniment} == {
        PHONE_GAIN * x for x in val_levels
    }
    mix, accompaniment = trainer.clean_val
    assert {round(float(v.mean()), 2) for v in mix - accompaniment} == val_levels


# -- resume ----------------------------------------------------------------------


def test_resume_continues_identically(data, tiny_onnx, tmp_path):
    straight = finetune.Trainer(make_args(data, tiny_onnx, tmp_path / "a"))
    straight.run()

    args = make_args(data, tiny_onnx, tmp_path / "b")
    first = finetune.Trainer(args)
    first.run(stop_after=2)  # saves at step 2, inside the warmup
    del first
    args.resume = True
    resumed = finetune.Trainer(args)
    resumed.restore()
    assert resumed.step == 2
    resumed.run()
    assert resumed.step == 4

    for (name, p), q in zip(
        resumed.model.named_parameters(), straight.model.parameters(), strict=True
    ):
        assert torch.equal(p, q), name
    assert resumed.rng.getstate() == straight.rng.getstate()
    assert resumed.scheduler.get_last_lr() == straight.scheduler.get_last_lr()
    log_a = pd.read_csv(finetune.log_path(straight.args.checkpoint))
    log_b = pd.read_csv(finetune.log_path(args.checkpoint))
    assert log_b["step"].tolist() == [0, 2, 4]
    columns = ["lr", "train_loss", "grad_norm", "val_loss", "clean_val_loss"]
    pd.testing.assert_frame_equal(log_a[columns], log_b[columns])


def test_resume_refuses_other_settings(data, tiny_onnx, tmp_path):
    args = make_args(data, tiny_onnx, tmp_path)
    finetune.Trainer(args).run(stop_after=2)
    other = make_args(data, tiny_onnx, tmp_path, "--freeze", "encoder", "--resume")
    with pytest.raises(SystemExit, match="freeze"):
        finetune.Trainer(other).restore()


# -- end of a run ----------------------------------------------------------------


def test_finish_measures_before_and_after(data, tiny_onnx, tmp_path):
    args = make_args(data, tiny_onnx, tmp_path, "--max-sdr-clips", "2")
    trainer = finetune.train(args)
    summary = finetune.finish(trainer)
    name = "HQ3_ft_musdb7_phone_lr0.001_s4"
    assert summary["model"] == name and summary["passed"]
    assert (tmp_path / "artifacts" / f"{name}.onnx").is_file()

    clips = pd.read_csv(
        tmp_path / "results" / "task3_sdr_clips.csv", keep_default_na=False
    )
    counts = clips.groupby(["model", "variant", "dataset"]).size().to_dict()
    assert counts == {
        (finetune.HQ3_NAME, "", "musdb7"): 4,
        (finetune.HQ3_NAME, "", "musdb7_phone"): 4,
        (name, "onnx", "musdb7"): 4,
        (name, "onnx", "musdb7_phone"): 4,
    }
    assert set(clips["split"]) == {"validation"}  # no test without --report-test
    table = pd.read_csv(tmp_path / "results" / "task3_finetune_before_after.csv")
    assert table[["dataset", "stem"]].values.tolist() == [
        ["musdb7_phone", "vocals"],
        ["musdb7_phone", "accompaniment"],
        ["musdb7", "vocals"],
        ["musdb7", "accompaniment"],
    ]
    assert (table["n_clips"] == 2).all()

    # The original HQ 3 is measured only once.
    before = clips[clips["model"] == finetune.HQ3_NAME]
    finetune.measure_sdr(trainer, tmp_path / "artifacts" / f"{name}.onnx", "onnx")
    again = pd.read_csv(
        tmp_path / "results" / "task3_sdr_clips.csv", keep_default_na=False
    )
    pd.testing.assert_frame_equal(
        again[again["model"] == finetune.HQ3_NAME].reset_index(drop=True),
        before.reset_index(drop=True),
    )


# -- smoke -----------------------------------------------------------------------


def test_smoke_forward_equivalence(data, tiny_onnx, tmp_path):
    report = finetune.smoke_forward_equivalence(make_args(data, tiny_onnx, tmp_path))
    assert report["clip"] == VALIDATION[0]
    assert report["model_output_vs_onnxruntime"]["identical"]
    assert report["estimate_vs_uvr_inference"]["identical"]


def test_smoke_runs_every_phase(data, tiny_onnx, tmp_path, monkeypatch):
    """--smoke end to end on the tiny model; the onnxruntime 1.13.1
    subprocess is replaced by a stub."""
    monkeypatch.setattr(finetune, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(finetune, "ort113_python", lambda _: tmp_path / "python")
    monkeypatch.setattr(
        finetune,
        "check_onnxruntime_113",
        lambda *_: {"ran": True, "onnxruntime": "1.13.1", "passed": True},
    )
    args = finetune.parse_args(
        ["--smoke", "--model", str(tiny_onnx), "--data-dir", str(data), "--lr", "1e-3"]
    )
    assert (args.steps, args.batch_size, args.max_val_clips) == (1, 1, 1)
    threads = torch.get_num_threads()
    try:
        report = finetune.smoke(args)
        again = finetune.smoke(args)
    finally:
        torch.set_num_threads(threads)  # smoke() sets 2 for the whole process
    assert report["passed"], report["checks"]
    assert list(report["phases"]) == [
        "1_lr0_forward",
        "2_train_step",
        "3_resume",
        "4_export_and_sdr",
    ]
    assert all(p["peak_rss_mb"] > 0 for p in report["phases"].values())
    assert report["torch_threads"] == finetune.SMOKE_THREADS
    assert [r["step"] for r in report["train_log"]] == [0, 1]
    assert {r["n_clips"] for r in report["before_after_validation"]} == {1}
    # A second smoke starts from scratch rather than resuming the first.
    assert again["passed"] and [r["step"] for r in again["train_log"]] == [0, 1]
