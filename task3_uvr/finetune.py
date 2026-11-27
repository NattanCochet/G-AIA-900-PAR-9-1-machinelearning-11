"""T3-10: fine-tune UVR-MDX-NET-Inst_HQ_3 on degraded audio.

    uv run python -m task3_uvr.finetune --dataset musdb7_phone \
        --steps 2000 --batch-size 2 --checkpoint <path>.pt [--resume]
    uv run python -m task3_uvr.finetune --smoke      # CPU check, see smoke()

Model: HQ 3's ONNX file (--model, or <ML_TEK5_UVR_DIR>/models/MDX_Net_Models/
UVR-MDX-NET-Inst_HQ_3.onnx), converted to PyTorch with onnx2pytorch (T3-05).
Its hash must be HQ 3's. The converted module stays in eval mode for the
whole run: BatchNorm keeps HQ 3's running statistics (in train mode the
output changes by 2.6 times its largest value). --freeze none | encoder |
all-but-last-N leaves some layers out of the optimiser.

Input and target: chunks of 1024 x 255 samples (one UVR chunk) at random
positions in the 80 clips of task3_uvr/splits/train.json, from --dataset.
The input is the spectrogram UVR gives HQ 3 (mdx_inference.model_input: its
STFT, n_fft 6144, dim_f 3072, 3 lowest bins zeroed). HQ 3's primary stem is
Instrumental, so the target is the dataset's own accompaniment (degraded,
for a degraded dataset).

Loss: L1 in the time domain between compensate x iSTFT(model output) and the
accompaniment, i.e. the primary stem as UVR computes it before overlap-add.
Since UVR's vocals are mixture - primary, the vocal error is the same signal
with the opposite sign: this loss trains both stems. Adam, linear warmup
over --warmup-steps then constant --lr, gradient norm clipped to
--clip-grad-norm.

Every --eval-every steps (and at step 0, before any update) the mean training
loss since the last row and the loss on fixed validation chunks (seed 1234,
--val-chunks-per-clip per clip of the 14 validation clips) go to
<checkpoint>_log.csv, on --dataset and on clean musdb7. Checkpoints are atomic
and hold the model, optimiser, scheduler, RNG states and data generator
(common.checkpointing); --resume continues as in T3-08.

At the end:
- export to <artifacts>/HQ3_ft_<dataset>_<tag>_s<steps>.onnx (opset 17,
  legacy exporter) and its UVR settings (HQ 3's) to <artifacts>/model_data/
  <hash>.json; parity with PyTorch in this onnxruntime, then in UVR's
  onnxruntime 1.13.1 (task3_uvr.check_onnx_io, run with --ort113-python);
- validation SDR on --dataset and on clean musdb7 (to check that it did not
  forget), and of the original HQ 3 on the same clips if the per-clip CSV
  lacks them; test SDR only with --report-test;
- per-clip rows in results/task3_sdr_clips.csv, medians and means in
  results/task3_uvr.csv, the before/after table (task3_uvr.finetune_report)
  and a summary in results/task3_finetune.json.
"""

import argparse
import contextlib
import copy
import csv
import datetime
import gc
import json
import os
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
import torch.nn.functional as F
from onnx2pytorch import ConvertModel
from torch import nn

from common import load_checkpoint, log_result, save_checkpoint, set_seed
from common.code_version import git_commit
from task3_uvr.check_data import resolve_data_dir
from task3_uvr.compare_gui import UVR_ENV_VAR
from task3_uvr.evaluate import STEMS, evaluate_split, split_clips
from task3_uvr.finetune_report import (
    BASELINE_VARIANT,
    HQ3_NAME,
    before_after,
    read_clip_rows,
    save_table,
    t3_07_mismatches,
    to_markdown,
)
from task3_uvr.hq3_trainable import (
    HQ3_HASH,
    HQ3_SETTINGS,
    describe_export,
    export_onnx,
    run_onnxruntime,
)
from task3_uvr.mdx_inference import (
    STFT,
    MdxSettings,
    load_settings,
    model_input,
    onnx_model,
    separate,
    spectral_chunk_fn,
)
from task3_uvr.onnx_checks import compare_outputs, update_results, uvr_hash
from task3_uvr.plot_small_nn import plot_losses
from task3_uvr.sdr import EPS, sdr
from task3_uvr.small_nn import N_SAMPLES_IN_CHUNK
from task3_uvr.small_nn_data import (
    DATASETS,
    dataset_dir,
    load_clips,
    make_batch,
    split_names,
    validation_chunks,
)
from task3_uvr.train_small_nn import (
    CLIP_COLUMNS,
    VALIDATION_SEED,
    log_path,
    resolve_device,
    save_clip_rows,
    truncate_log,
)

SETTINGS = MdxSettings.from_model_data(HQ3_SETTINGS)
CHUNK = SETTINGS.hop * (SETTINGS.dim_t - 1)  # 261,120 samples, 5.9 s
# make_batch and validation_chunks (T3-08) cut chunks of this length.
assert CHUNK == N_SAMPLES_IN_CHUNK

CLEAN = DATASETS[0]
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path("task3_uvr") / "artifacts"
SMOKE_DIR = ARTIFACTS_DIR / "smoke" / "finetune"
ORT113_ENV_VAR = "ML_TEK5_ORT113_PYTHON"
ORT113_VERSION = "1.13.1"
LOG_FIELDS = [
    "step",
    "lr",
    "train_loss",
    "grad_norm",
    "val_loss",
    "val_sdr_accompaniment",
    "clean_val_loss",
    "clean_val_sdr_accompaniment",
    "train_seconds",
]
# A resumed run must train the same model on the same data stream.
RESUME_KEYS = (
    "model_hash",
    "dataset",
    "remix",
    "batch_size",
    "lr",
    "warmup_steps",
    "clip_grad_norm",
    "freeze",
    "seed",
    "max_train_clips",
)
LOSS_NAME = "l1_waveform_loss"


# --- Model --------------------------------------------------------------------


def resolve_hq3(arg: str | None) -> Path:
    """--model, else HQ 3 in <ML_TEK5_UVR_DIR>/models/MDX_Net_Models/. Its
    hash must be HQ 3's: the settings below are HQ 3's."""
    if arg:
        path = Path(arg).expanduser()
    else:
        uvr = os.environ.get(UVR_ENV_VAR)
        if not uvr:
            sys.exit(f"Give --model <HQ 3 ONNX file> or set {UVR_ENV_VAR}.")
        path = Path(uvr).expanduser() / "models" / "MDX_Net_Models" / f"{HQ3_NAME}.onnx"
    if not path.is_file():
        sys.exit(f"Model not found: {path}")
    if uvr_hash(path) != HQ3_HASH:
        sys.exit(f"{path} is not HQ 3 (UVR hash {uvr_hash(path)}, expected {HQ3_HASH})")
    return path


def load_model(onnx_model_proto: onnx.ModelProto) -> nn.Module:
    """onnx2pytorch conversion, in eval mode. Never call .train() on it."""
    return ConvertModel(onnx_model_proto).eval()


def batchnorm_modules(module: nn.Module) -> list[nn.Module]:
    return [
        m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]


def check_batchnorm_eval(module: nn.Module) -> None:
    """Fail rather than train with batch statistics."""
    if any(m.training for m in batchnorm_modules(module)):
        raise RuntimeError("a BatchNorm layer is in train mode")


def parameter_layers(module: nn.Module) -> list[tuple[str, nn.Module]]:
    """Direct children that own parameters, in graph order: onnx2pytorch adds
    one child per ONNX node, in the order of the nodes. For HQ 3: 94 layers
    (40 Conv, 22 MatMul, 27 BatchNorm, 5 ConvTranspose), 166 tensors."""
    return [
        (name, child)
        for name, child in module.named_children()
        if next(child.parameters(), None) is not None
    ]


def parse_freeze(value: str) -> str:
    if value in ("none", "encoder") or re.fullmatch(r"all-but-last-[1-9]\d*", value):
        return value
    raise argparse.ArgumentTypeError(
        f"{value!r}: expected none, encoder or all-but-last-N (N >= 1)"
    )


def frozen_layer_names(module: nn.Module, freeze: str) -> list[str]:
    """Layers --freeze leaves out of training.

    encoder: every layer before the first ConvTranspose, i.e. HQ 3's input
    convolution, its 5 downsampling levels and its bottleneck (each a
    TFC-TDF block). The 5 upsampling levels and the output convolution
    train.
    all-but-last-N: every layer except the last N parameter layers.
    """
    layers = parameter_layers(module)
    if freeze == "none":
        return []
    if freeze == "encoder":
        for i, (_, child) in enumerate(layers):
            if any(isinstance(m, nn.ConvTranspose2d) for m in child.modules()):
                return [name for name, _ in layers[:i]]
        raise ValueError("no ConvTranspose layer: cannot tell the encoder apart")
    n = int(freeze.rsplit("-", 1)[1])
    if n > len(layers):
        raise ValueError(f"{freeze}: the model has only {len(layers)} parameter layers")
    return [name for name, _ in layers[: len(layers) - n]]


def apply_freeze(module: nn.Module, freeze: str) -> list[str]:
    names = frozen_layer_names(module, freeze)
    for name in names:
        for p in module.get_submodule(name).parameters():
            p.requires_grad_(False)
    return names


def estimate(model: nn.Module, stft: STFT, mix: torch.Tensor) -> torch.Tensor:
    """Accompaniment of [B, 2, CHUNK] mixtures, as UVR computes the primary
    stem of one chunk: model on the input spectrogram, iSTFT, times
    compensate."""
    return SETTINGS.compensate * stft.inverse(model(model_input(stft, mix)))


def warmup_factor(warmup_steps: int):
    """LR multiplier of step k (0-based): linear from 1/warmup to 1, then 1."""

    def factor(step: int) -> float:
        return 1.0 if warmup_steps <= 0 else min(1.0, (step + 1) / warmup_steps)

    return factor


# --- Arguments ------------------------------------------------------------------


def default_tag(args: argparse.Namespace) -> str:
    """Run settings that change the model, without the number of steps (a
    run resumed with more steps keeps its checkpoint)."""
    parts = [f"lr{args.lr:g}"]
    if args.freeze != "none":
        parts.append(args.freeze)
    if args.remix:
        parts.append("remix")
    if args.seed:
        parts.append(f"seed{args.seed}")
    return "_".join(parts)


def run_name(args: argparse.Namespace) -> str:
    """Checkpoint stem, e.g. HQ3_ft_musdb7_phone_lr1e-06."""
    return f"HQ3_ft_{args.dataset}_{args.tag}"


def export_name(args: argparse.Namespace, step: int) -> str:
    """ONNX file stem and `model` column, e.g. HQ3_ft_musdb7_phone_lr1e-06_s2000:
    every export gets its own file name, hence its own hash and JSON."""
    return f"{run_name(args)}_s{step}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=DATASETS, default="musdb7_phone")
    parser.add_argument(
        "--model", help=f"HQ 3 ONNX file (default: from ${UVR_ENV_VAR})"
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--freeze",
        type=parse_freeze,
        default="none",
        help="none, encoder or all-but-last-N (default: none)",
    )
    parser.add_argument(
        "--remix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="vocals of one clip over the accompaniment of another (default: off)",
    )
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tag",
        help="name part after the dataset (default: from lr, freeze, remix, seed)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="default: <artifacts-dir>/HQ3_ft_<dataset>_<tag>.pt",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--val-chunks-per-clip", type=int, default=2)
    parser.add_argument(
        "--stop-after",
        type=int,
        help="save and stop after this many steps in this session (no export)",
    )
    parser.add_argument(
        "--data-dir", help="musdb7 folder (default: $ML_TEK5_MUSDB7_DIR)"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument(
        "--sdr", action=argparse.BooleanOptionalAction, default=True,
        help="measure the SDR after training (default: on)",
    )  # fmt: skip
    parser.add_argument(
        "--sdr-backend",
        choices=("onnx", "torch"),
        default="onnx",
        help="onnx: the exported file in onnxruntime on CPU, as UVR runs it; "
        "torch: the PyTorch weights on --device (faster on a GPU)",
    )
    parser.add_argument(
        "--baseline", action=argparse.BooleanOptionalAction, default=True,
        help="SDR of the original HQ 3 on the same clips, if the CSV lacks it",
    )  # fmt: skip
    parser.add_argument(
        "--report-test",
        action="store_true",
        help="also measure the test SDR (final reporting only, never for tuning)",
    )
    parser.add_argument(
        "--ort113-python",
        help=f"Python with onnxruntime {ORT113_VERSION} (default: ${ORT113_ENV_VAR}, "
        f"else <${UVR_ENV_VAR}>/.venv/bin/python)",
    )
    parser.add_argument("--max-train-clips", type=int)
    parser.add_argument("--max-val-clips", type=int)
    parser.add_argument("--max-sdr-clips", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CPU check with little memory: forward equivalence with HQ 3, 1 step "
        "of batch 1, resume, export, SDR on 1 validation clip; outputs in "
        "task3_uvr/artifacts/smoke/finetune/",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    finalize_paths(args)
    return args


def finalize_paths(args: argparse.Namespace) -> None:
    if args.tag is None:
        args.tag = default_tag(args)
    if args.checkpoint is None:
        args.checkpoint = args.artifacts_dir / f"{run_name(args)}.pt"


def apply_smoke(args: argparse.Namespace) -> None:
    """One backward pass of HQ 3 on a full chunk peaks at about 6.7 GB on
    CPU: batch 1, one step, one clip per split."""
    args.steps, args.batch_size, args.device = 1, 1, "cpu"
    args.warmup_steps = 0
    args.max_train_clips, args.max_val_clips, args.max_sdr_clips = 1, 1, 1
    args.val_chunks_per_clip = 1
    args.eval_every, args.checkpoint_every = 1, 1
    args.artifacts_dir = SMOKE_DIR
    args.results_dir = SMOKE_DIR / "results"
    args.report_test = False
    args.sdr_backend = "onnx"


def run_config(args: argparse.Namespace, model_hash: str) -> dict:
    return {
        "model_hash": model_hash,
        "dataset": args.dataset,
        "remix": args.remix,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "clip_grad_norm": args.clip_grad_norm,
        "freeze": args.freeze,
        "seed": args.seed,
        "max_train_clips": args.max_train_clips,
        "loss": "L1 on the waveform: compensate x iSTFT(output) vs accompaniment",
        "optimizer": "adam",
        "batchnorm": "eval (HQ 3's running statistics)",
        "validation_seed": VALIDATION_SEED,
        "val_chunks_per_clip": args.val_chunks_per_clip,
    }


# --- Training -------------------------------------------------------------------


class Trainer:
    """HQ 3 in PyTorch, its data and optimiser; `run` trains from the current
    step, `save` writes the checkpoint."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = resolve_device(args.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # fixed input size
        self.musdb7_dir = resolve_data_dir(args.data_dir)
        self.data_dir = dataset_dir(self.musdb7_dir, args.dataset)
        train_names = split_names("train")[: args.max_train_clips]
        val_names = split_names("validation")[: args.max_val_clips]
        self.train_clips = load_clips(self.data_dir, train_names)
        self.val = self._validation_set(self.data_dir, val_names)
        self.clean_val = (
            None
            if args.dataset == CLEAN
            else self._validation_set(self.musdb7_dir, val_names)
        )
        self.n_train_clips, self.n_val_clips = len(train_names), len(val_names)

        set_seed(args.seed)
        self.model_path = Path(args.model)
        self.model_hash = uvr_hash(self.model_path)
        self.original = onnx.load(str(self.model_path))
        self.model = load_model(self.original).to(self.device)
        self.initial = {
            n: p.detach().cpu().clone() for n, p in self.model.named_parameters()
        }
        self.frozen_layers = apply_freeze(self.model, args.freeze)
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable, lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, warmup_factor(args.warmup_steps)
        )
        self.stft = STFT(SETTINGS.n_fft, SETTINGS.hop, SETTINGS.dim_f)
        self.rng = random.Random(args.seed)  # batches only
        self.step = 0  # completed steps
        self.train_seconds = 0.0
        self.loss_sum, self.grad_sum, self.loss_count = (
            0.0,
            0.0,
            0,
        )  # since the last row
        self.last_lr = 0.0  # learning rate of the last step
        self.config = run_config(args, self.model_hash)

    def _validation_set(
        self, data_dir: Path, names: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed (mixture, accompaniment) chunks, at the same positions on
        every dataset (seed VALIDATION_SEED, clips of equal length)."""
        mix, vocals = validation_chunks(
            load_clips(data_dir, names), self.args.val_chunks_per_clip, VALIDATION_SEED
        )
        return mix, mix - vocals

    def n_parameters(self) -> dict:
        return {
            "n_parameter_tensors": sum(1 for _ in self.model.parameters()),
            "n_trainable_tensors": len(self.trainable),
            "n_trainable_values": sum(p.numel() for p in self.trainable),
            "n_frozen_layers": len(self.frozen_layers),
        }

    # -- checkpoint --------------------------------------------------------
    def save(self) -> None:
        extra = {
            "config": self.config,
            "data_rng": self.rng.getstate(),
            "train_seconds": self.train_seconds,
            "loss_sum": self.loss_sum,
            "grad_sum": self.grad_sum,
            "loss_count": self.loss_count,
        }
        # `epoch` is the index of the last completed step.
        save_checkpoint(
            self.args.checkpoint,
            self.model,
            self.optimizer,
            self.step - 1,
            extra,
            scheduler=self.scheduler,
        )

    def restore(self) -> None:
        # Settings first: another --freeze would make the optimiser state
        # fail to load with a less helpful error. mmap reads only `extra`.
        saved = torch.load(
            self.args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )["extra"]["config"]
        changed = {
            k: (saved.get(k), self.config[k])
            for k in RESUME_KEYS
            if saved.get(k) != self.config[k]
        }
        if changed:
            sys.exit(f"--resume with different settings (saved, given): {changed}")
        self.step, extra = load_checkpoint(
            self.args.checkpoint,
            self.model,
            self.optimizer,
            map_location=self.device,
            scheduler=self.scheduler,
        )
        self.model.eval()
        self.rng.setstate(extra["data_rng"])
        self.train_seconds = extra["train_seconds"]
        self.loss_sum, self.grad_sum = extra["loss_sum"], extra["grad_sum"]
        self.loss_count = extra["loss_count"]
        truncate_log(log_path(self.args.checkpoint), self.step)

    # -- loss --------------------------------------------------------------
    def estimate(self, mix: torch.Tensor) -> torch.Tensor:
        return estimate(self.model, self.stft, mix)

    def loss(self, mix: torch.Tensor, accompaniment: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.estimate(mix), accompaniment)

    @torch.no_grad()
    def validation(
        self, data: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[float, float]:
        """Mean L1 loss and median accompaniment SDR (dB) over the fixed
        chunks. The SDR is per 5.9 s chunk, a proxy for the clip SDR."""
        if data is None:
            return float("nan"), float("nan")
        mix, accompaniment = data
        losses, sdrs = [], []
        for i in range(0, len(mix), self.args.batch_size):
            estimate = self.estimate(
                mix[i : i + self.args.batch_size].to(self.device)
            ).cpu()
            reference = accompaniment[i : i + self.args.batch_size]
            losses.append((estimate - reference).abs().mean(dim=(1, 2)))
            sdrs += [
                sdr(r.numpy(), e.numpy())
                for r, e in zip(reference, estimate, strict=True)
            ]
        return float(torch.cat(losses).mean()), float(np.median(sdrs))

    def log_row(self, lr: float) -> dict:
        val_loss, val_sdr = self.validation(self.val)
        clean_loss, clean_sdr = self.validation(self.clean_val)
        count = self.loss_count
        row = {
            "step": self.step,
            "lr": lr,
            "train_loss": self.loss_sum / count if count else float("nan"),
            "grad_norm": self.grad_sum / count if count else float("nan"),
            "val_loss": val_loss,
            "val_sdr_accompaniment": val_sdr,
            "clean_val_loss": clean_loss,
            "clean_val_sdr_accompaniment": clean_sdr,
            "train_seconds": self.train_seconds,
        }
        append_log(log_path(self.args.checkpoint), row)
        self.loss_sum, self.grad_sum, self.loss_count = 0.0, 0.0, 0
        print(
            f"step {row['step']}/{self.args.steps}  lr {lr:.2e}  "
            f"train {row['train_loss']:.6f}  val {val_loss:.6f} ({val_sdr:.2f} dB)  "
            f"clean val {clean_loss:.6f} ({clean_sdr:.2f} dB)  {self.train_seconds:.0f} s",
            flush=True,
        )
        return row

    # -- training ----------------------------------------------------------
    def train_step(self) -> None:
        args = self.args
        check_batchnorm_eval(self.model)
        start = time.perf_counter()
        mix, vocals = make_batch(
            self.train_clips, args.batch_size, args.remix, self.rng
        )
        accompaniment = mix - vocals
        lr = self.scheduler.get_last_lr()[0]
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(mix.to(self.device), accompaniment.to(self.device))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, args.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.loss_sum += loss.item()
        self.grad_sum += float(grad_norm)
        self.loss_count += 1
        self.step += 1
        self.last_lr = lr
        self.train_seconds += time.perf_counter() - start

    def run(self, stop_after: int | None = None) -> None:
        args = self.args
        end = (
            args.steps
            if stop_after is None
            else min(args.steps, self.step + stop_after)
        )
        if self.step == 0 and not log_path(args.checkpoint).exists():
            self.log_row(lr=0.0)  # the original HQ 3, before any update
        session_start, first = time.perf_counter(), self.step
        while self.step < end:
            self.train_step()
            last = self.step == args.steps
            if self.step % args.eval_every == 0 or last:
                self.log_row(self.last_lr)
            if self.step % args.checkpoint_every == 0 or last or self.step == end:
                self.save()
        if self.step > first:
            seconds = time.perf_counter() - session_start
            print(
                f"session: steps {first + 1}-{self.step} in {seconds:.1f} s "
                f"({seconds / (self.step - first):.1f} s per step with validation and saving)"
                + (
                    f", CUDA peak {cuda_peak_gb():.1f} GB"
                    if self.device.type == "cuda"
                    else ""
                )
            )


def cuda_peak_gb() -> float | None:
    """Peak memory PyTorch allocated on the GPU in this process, in GB."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace, stop_after: int | None = None) -> Trainer:
    """One training session: restore if asked, then train until --steps or
    `stop_after` more steps."""
    trainer = Trainer(args)
    if args.checkpoint.exists():
        if not args.resume:
            sys.exit(
                f"{args.checkpoint} exists: pass --resume, or choose another --checkpoint"
            )
        trainer.restore()
        print(f"resumed {args.checkpoint} at step {trainer.step}")
    else:
        if args.resume:
            print(f"no checkpoint at {args.checkpoint}: starting from step 0")
        log_path(args.checkpoint).unlink(missing_ok=True)
    sizes = trainer.n_parameters()
    print(
        f"{run_name(args)}: {sizes['n_trainable_tensors']}/{sizes['n_parameter_tensors']} "
        f"tensors trained ({sizes['n_trainable_values']:,} values, freeze {args.freeze}), "
        f"{trainer.n_train_clips} training / {trainer.n_val_clips} validation clips, "
        f"device {trainer.device}"
    )
    trainer.run(stop_after)
    return trainer


# --- Export and checks ------------------------------------------------------------


def write_settings(onnx_path: Path) -> Path:
    """<artifacts>/model_data/<hash>.json with HQ 3's settings: the fine-tuned
    model has a new hash, and UVR finds its settings from it."""
    model_data = onnx_path.parent / "model_data"
    model_data.mkdir(parents=True, exist_ok=True)
    path = model_data / f"{uvr_hash(onnx_path)}.json"
    path.write_text(json.dumps(HQ3_SETTINGS, indent=4))
    return path


def weight_change(trainer: Trainer) -> dict:
    """How far the weights moved from HQ 3's, and whether frozen ones did."""
    frozen = {
        f"{layer}.{name}"
        for layer in trainer.frozen_layers
        for name, _ in trainer.model.get_submodule(layer).named_parameters()
    }
    changed, max_abs, max_relative = [], 0.0, 0.0
    for name, p in trainer.model.named_parameters():
        before, after = trainer.initial[name], p.detach().cpu()
        if not torch.equal(before, after):
            changed.append(name)
            diff = after - before
            max_abs = max(max_abs, float(diff.abs().max()))
            max_relative = max(
                max_relative, float(diff.norm() / (before.norm() + 1e-12))
            )
    return {
        **trainer.n_parameters(),
        "n_tensors_changed": len(changed),
        "frozen_tensors_changed": sorted(set(changed) & frozen),
        "max_abs_change": max_abs,
        "max_relative_norm_change": max_relative,
    }


def ort113_python(arg: str | None) -> Path | None:
    value = arg or os.environ.get(ORT113_ENV_VAR)
    if value:
        return Path(value).expanduser()
    uvr = os.environ.get(UVR_ENV_VAR)
    candidate = Path(uvr).expanduser() / ".venv" / "bin" / "python" if uvr else None
    return candidate if candidate and candidate.is_file() else None


def check_onnxruntime_113(onnx_path: Path, io_path: Path, python: Path | None) -> dict:
    """Run task3_uvr.check_onnx_io in a Python that has onnxruntime 1.13.1
    (UVR's), from the repository root. Passes only if it exits 0 and reports
    that exact version."""
    command = [
        str(python) if python else f"<python with onnxruntime {ORT113_VERSION}>",
        "-m", "task3_uvr.check_onnx_io",
        "--model", str(onnx_path.resolve()),
        "--io", str(io_path.resolve()),
    ]  # fmt: skip
    if python is None:
        print(f"onnxruntime {ORT113_VERSION} check NOT RUN (give --ort113-python): run")
        print("   ", " ".join(command))
        return {"ran": False, "passed": None, "command": " ".join(command)}
    proc = subprocess.run(
        command, check=False, cwd=REPO_ROOT, capture_output=True, text=True
    )
    output = proc.stdout + proc.stderr
    print(output.strip())
    found = re.search(r"'onnxruntime': '([^']+)'", output)
    version = found[1] if found else None
    return {
        "ran": True,
        "onnxruntime": version,
        "passed": proc.returncode == 0 and version == ORT113_VERSION,
        "output": output[-1500:],
    }


def export_and_check(trainer: Trainer, onnx_path: Path) -> dict:
    """Export a CPU copy to ONNX, write its settings, and compare on one
    real validation chunk: PyTorch vs the export (this onnxruntime, then
    1.13.1), and the export vs the original HQ 3 (how much the output moved).
    The chunk and PyTorch's output are kept in <model>_io.npz."""
    args = trainer.args
    model = copy.deepcopy(trainer.model).cpu().eval()
    # Without gradients, tracing keeps no autograd graph (several GB for HQ 3).
    model.requires_grad_(False)
    x = model_input(trainer.stft, trainer.val[0][:1])
    export_onnx(model, x, onnx_path)
    with torch.no_grad():
        expected = model(x).numpy()
    io_path = onnx_path.with_name(onnx_path.stem + "_io.npz")
    np.savez_compressed(io_path, input=x.numpy(), output=expected)
    exported = run_onnxruntime(onnx_path, x.numpy())
    report = {
        "file": str(onnx_path),
        "settings_file": str(write_settings(onnx_path)),
        **describe_export(onnx_path, trainer.original),
        "parity_pytorch_vs_export": compare_outputs(expected, exported),
        "export_vs_original_hq3": compare_outputs(
            run_onnxruntime(trainer.model_path, x.numpy()), exported
        ),
        "onnxruntime_1_13": check_onnxruntime_113(
            onnx_path, io_path, ort113_python(args.ort113_python)
        ),
    }
    return report


# --- SDR --------------------------------------------------------------------------


def torch_model_fn(module: nn.Module, device: torch.device):
    @torch.no_grad()
    def run(spek: np.ndarray) -> np.ndarray:
        return module(torch.from_numpy(spek).to(device)).cpu().numpy()

    return run


def eval_datasets(dataset: str) -> list[str]:
    """The training dataset, then clean musdb7 (did it forget?)."""
    return [dataset] if dataset == CLEAN else [dataset, CLEAN]


def has_rows(
    path: Path, dataset: str, model: str, variant: str, split: str, clips
) -> bool:
    if not path.exists():
        return False
    rows = read_clip_rows(path)
    rows = rows[
        (rows["dataset"] == dataset)
        & (rows["model"] == model)
        & (rows["variant"] == variant)
        & (rows["split"] == split)
        & (rows["match_freq_cut"] == "False")
    ]
    return {c.name for c in clips} <= set(rows["clip"])


def sdr_frame(
    name, model_hash, variant, dataset, split, estimator, clips
) -> tuple[pd.DataFrame, float]:
    rows, total = evaluate_split(name, estimator, clips)
    frame = pd.DataFrame(rows).assign(
        date=datetime.date.today().isoformat(),
        dataset=dataset,
        model=name,
        model_hash=model_hash,
        variant=variant,
        split=split,
        match_freq_cut=False,
    )[CLIP_COLUMNS]
    return frame, total


def log_sdr(trainer: Trainer, frame: pd.DataFrame, total: float, config: dict) -> dict:
    args, medians = trainer.args, {}
    split, dataset, name = (
        frame["split"].iloc[0],
        frame["dataset"].iloc[0],
        frame["model"].iloc[0],
    )
    for stem in STEMS:
        values = frame.loc[frame["stem"] == stem, "sdr_db"]
        medians[stem] = float(values.median())
        for stat in ("median", "mean"):
            log_result(
                args.results_dir / "task3_uvr.csv",
                task="task3_uvr",
                seed=args.seed,
                split=split,
                metric=f"sdr_{stat}_{stem}",
                score=float(getattr(values, stat)()),
                duration_seconds=total,
                config={
                    **config,
                    "dataset": dataset,
                    "model": name,
                    "n_clips": len(values),
                },
            )
        print(f"{name} {dataset} {split} {stem}: median {values.median():.2f} dB")
    return medians


def measure_sdr(trainer: Trainer, onnx_path: Path, backend: str) -> dict:
    """SDR of the fine-tuned model (and of the original HQ 3 where the CSV
    lacks it) on validation, and test with --report-test. The settings are
    read back from the export's hash, as UVR would."""
    args = trainer.args
    name = onnx_path.stem
    clips_csv = args.results_dir / "task3_sdr_clips.csv"
    model_hash, settings = load_settings(onnx_path, onnx_path.parent / "model_data")
    if backend == "onnx":
        fine_tuned = onnx_model(onnx_path, settings)
    else:
        fine_tuned = torch_model_fn(trainer.model, trainer.device)
    baseline = None
    base_variant = BASELINE_VARIANT[backend]
    common = {
        "sdr": "global (MDX Challenge)",
        "eps": EPS,
        "match_freq_cut": False,
        "n_fft": settings.n_fft,
        "dim_f": settings.dim_f,
        "segment_size": settings.dim_t,
        "overlap": "Default",
        "compensate": settings.compensate,
        "primary_stem": settings.primary_stem,
        "sdr_backend": backend,
        "code": git_commit(),
    }
    medians = {}
    splits = ["validation", "test"] if args.report_test else ["validation"]
    for dataset in eval_datasets(args.dataset):
        data_dir = dataset_dir(trainer.musdb7_dir, dataset)
        for split in splits:
            clips = split_clips(data_dir, split)[: args.max_sdr_clips]
            if args.baseline and not has_rows(
                clips_csv, dataset, HQ3_NAME, base_variant, split, clips
            ):
                if baseline is None:
                    baseline = (
                        onnx_model(trainer.model_path, SETTINGS)
                        if backend == "onnx"
                        else torch_model_fn(
                            load_model(trainer.original).to(trainer.device),
                            trainer.device,
                        )
                    )
                print(f"{HQ3_NAME} (before) on {dataset} {split} ({len(clips)} clips)")
                frame, total = sdr_frame(
                    HQ3_NAME, trainer.model_hash, base_variant, dataset, split,
                    lambda mix, b=baseline: separate(mix, b, SETTINGS), clips,
                )  # fmt: skip
                save_clip_rows(frame, clips_csv)
                medians[f"{HQ3_NAME}/{dataset}/{split}"] = log_sdr(
                    trainer, frame, total, {**common, "model_hash": trainer.model_hash}
                )
            print(f"{name} on {dataset} {split} ({len(clips)} clips)")
            frame, total = sdr_frame(
                name, model_hash, backend, dataset, split,
                lambda mix: separate(mix, fine_tuned, settings), clips,
            )  # fmt: skip
            save_clip_rows(frame, clips_csv)
            medians[f"{name}/{dataset}/{split}"] = log_sdr(
                trainer, frame, total,
                {**trainer.config, **common, "model_hash": model_hash, "steps": trainer.step},
            )  # fmt: skip
    return medians


def report_before_after(trainer: Trainer, name: str) -> dict:
    """Validation before/after table (and test with --report-test)."""
    args = trainer.args
    rows = read_clip_rows(args.results_dir / "task3_sdr_clips.csv")
    tables = {}
    for split in ["validation", "test"] if args.report_test else ["validation"]:
        table = before_after(rows, name, split)
        save_table(table, args.results_dir / "task3_finetune_before_after.csv")
        print(f"\nBefore/after, {split} (global SDR, dB):\n{to_markdown(table)}\n")
        for line in t3_07_mismatches(table):
            print("WARNING, the original HQ 3 differs from T3-07:", line)
        tables[split] = table.to_dict(orient="records")
    return tables


# --- End of a run -----------------------------------------------------------------


def finish(trainer: Trainer) -> dict:
    args = trainer.args
    name = export_name(args, trainer.step)
    log = pd.read_csv(log_path(args.checkpoint))
    final = log.iloc[-1]
    loss_config = {
        **trainer.config,
        "model": name,
        "steps": trainer.step,
        "device": str(trainer.device),
        "code": git_commit(),
    }
    for split, column, dataset in (
        ("train", "train_loss", args.dataset),
        ("validation", "val_loss", args.dataset),
        ("validation", "clean_val_loss", CLEAN),
    ):
        if pd.notna(final[column]):
            log_result(
                args.results_dir / "task3_uvr.csv",
                task="task3_uvr",
                seed=args.seed,
                split=split,
                metric=LOSS_NAME,
                score=float(final[column]),
                duration_seconds=trainer.train_seconds,
                config={**loss_config, "loss_dataset": dataset},
            )
    figure = args.results_dir / f"task3_finetune_loss_{run_name(args)}.png"
    plot_losses(
        {name: log_path(args.checkpoint)},
        figure,
        ylabel="L1 waveform loss, accompaniment",
        yunit="full scale = 1",
        title="HQ 3 fine-tuning: training and validation loss",
    )

    onnx_path = args.artifacts_dir / f"{name}.onnx"
    export = export_and_check(trainer, onnx_path)
    parity, ort113 = export["parity_pytorch_vs_export"], export["onnxruntime_1_13"]
    print(
        f"exported {onnx_path} (hash {export['uvr_hash']}), settings {export['settings_file']}"
    )
    print(f"onnxruntime vs PyTorch: relative diff {parity['relative_diff']:.1e}")
    print(
        f"export vs original HQ 3: relative diff {export['export_vs_original_hq3']['relative_diff']:.1e}"
    )
    summary = {
        "model": name,
        "config": trainer.config,
        "steps": trainer.step,
        "device": str(trainer.device),
        "train_seconds": trainer.train_seconds,
        "seconds_per_step": trainer.train_seconds / max(trainer.step, 1),
        "cuda_peak_gb": cuda_peak_gb() if trainer.device.type == "cuda" else None,
        "initial_val_loss": float(log.iloc[0]["val_loss"]),
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "final_clean_val_loss": float(final["clean_val_loss"]),
        "weights": weight_change(trainer),
        "export": export,
        "figure": str(figure),
        "code": git_commit(),
    }
    if args.sdr:
        summary["sdr_backend"] = args.sdr_backend
        summary["median_sdr_db"] = measure_sdr(trainer, onnx_path, args.sdr_backend)
        summary["before_after"] = report_before_after(trainer, name)
    summary["passed"] = bool(parity["identical"]) and ort113["passed"] is not False
    update_results(args.results_dir / "task3_finetune.json", name, summary)
    return summary


# --- Smoke test -------------------------------------------------------------------

SMOKE_THREADS = 2
# The smoke's own result files, cleared at its start so that it also measures
# the original HQ 3 (as the first real run does).
SMOKE_RESULT_FILES = (
    "task3_sdr_clips.csv",
    "task3_uvr.csv",
    "task3_finetune_before_after.csv",
)


def reset_peak_rss() -> None:
    """Reset this process's peak RSS (VmHWM). Linux only; elsewhere the
    peaks below are since the start of the process."""
    with contextlib.suppress(OSError):
        Path("/proc/self/clear_refs").write_text("5")


def peak_rss_mb() -> float:
    """Peak RSS since the last reset_peak_rss (VmHWM)."""
    with contextlib.suppress(OSError):
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class MemoryPhases:
    """Seconds and peak RSS of each phase. Each phase starts after a
    gc.collect(); its large objects must be locals of a function that has
    returned, so that they are freed."""

    def __init__(self):
        self.report: dict[str, dict] = {}

    @contextlib.contextmanager
    def phase(self, name: str):
        gc.collect()
        reset_peak_rss()
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds, peak = time.perf_counter() - start, peak_rss_mb()
            self.report[name] = {
                "seconds": round(seconds, 1),
                "peak_rss_mb": round(peak),
            }
            print(f"[{name}] {seconds:.1f} s, peak RSS {peak:.0f} MB", flush=True)


def fingerprint(model: nn.Module) -> dict[str, tuple[float, float]]:
    """Sum and largest absolute value of every parameter tensor: equal
    fingerprints after a resume mean the same weights, without keeping a
    second copy of them."""
    return {
        n: (float(p.detach().double().sum()), float(p.detach().abs().max()))
        for n, p in model.named_parameters()
    }


def smoke_forward_equivalence(run: argparse.Namespace) -> dict:
    """lr 0, without training: the converted HQ 3 under torch.no_grad() on
    one validation chunk. Its raw output must match HQ 3 in onnxruntime, and
    the training estimate (compensate x iSTFT) must match UVR's inference
    path for that chunk (mdx_inference.spectral_chunk_fn)."""
    model_path = Path(run.model)
    model = load_model(onnx.load(str(model_path)))
    check_batchnorm_eval(model)
    stft = STFT(SETTINGS.n_fft, SETTINGS.hop, SETTINGS.dim_f)
    data_dir = dataset_dir(resolve_data_dir(run.data_dir), run.dataset)
    name = split_names("validation")[0]
    mix, _ = validation_chunks(load_clips(data_dir, [name]), 1, VALIDATION_SEED)
    hq3 = onnx_model(model_path, SETTINGS)
    captured = {}

    def hq3_kept(spek: np.ndarray) -> np.ndarray:
        captured["output"] = hq3(spek)
        return captured["output"]

    reference = SETTINGS.compensate * spectral_chunk_fn(SETTINGS, hq3_kept)(mix.numpy())
    with torch.no_grad():
        output = model(model_input(stft, mix))
        got = SETTINGS.compensate * stft.inverse(output)
    return {
        "clip": name,
        "model_output_vs_onnxruntime": compare_outputs(
            captured["output"], output.numpy()
        ),
        "estimate_vs_uvr_inference": compare_outputs(reference, got.numpy()),
    }


def smoke_train_step(run: argparse.Namespace) -> tuple[dict, dict]:
    """One step of batch 1 at --lr, then the checkpoint. Returns the report
    and the weights' fingerprint."""
    trainer = train(run)
    reference = load_model(trainer.original)  # HQ 3's BatchNorm statistics
    stats_unchanged = all(
        torch.equal(a.running_mean.cpu(), b.running_mean)
        and torch.equal(a.running_var.cpu(), b.running_var)
        for a, b in zip(
            batchnorm_modules(trainer.model), batchnorm_modules(reference), strict=True
        )
    )
    log = pd.read_csv(log_path(run.checkpoint))
    report = {
        "steps": trainer.step,
        "seconds_per_step": trainer.train_seconds / max(trainer.step, 1),
        "weights": weight_change(trainer),
        "batchnorm_statistics_unchanged": stats_unchanged,
        "log": log.to_dict(orient="records"),
    }
    return report, fingerprint(trainer.model)


def smoke_resume_and_finish(
    run: argparse.Namespace, phases: MemoryPhases, trained: dict
) -> tuple[dict, dict]:
    """Reload the checkpoint in a new Trainer, then run the end of a real
    run on it: export, parity, onnxruntime 1.13.1, SDR, before/after."""
    with phases.phase("3_resume"):
        trainer = Trainer(run)
        trainer.restore()
        resumed = {
            "step": trainer.step,
            "same_weights": fingerprint(trainer.model) == trained,
        }
    with phases.phase("4_export_and_sdr"):
        summary = finish(trainer)
    return resumed, summary


def smoke(args: argparse.Namespace) -> dict:
    """End-to-end check on CPU that fits in about 7 GB, in four phases with
    gc.collect() and the peak RSS of each:

    1. lr 0: forward equivalence with HQ 3 (smoke_forward_equivalence), no
       optimiser;
    2. one training step of batch 1 at --lr on 1 training clip, validation
       on 1 chunk of 1 clip, checkpoint: every trainable tensor must change,
       BatchNorm statistics must not;
    3. resume from the checkpoint in a new Trainer: same step, same weights;
    4. finish(): export (opset 17), parity with PyTorch, onnxruntime 1.13.1,
       settings JSON, SDR of HQ 3 and of the export on 1 validation clip
       (degraded and clean), before/after table.
    """
    python113 = ort113_python(args.ort113_python)
    if python113 is None:
        sys.exit(f"--smoke checks onnxruntime {ORT113_VERSION}: give --ort113-python "
                 f"or set {UVR_ENV_VAR}")  # fmt: skip
    torch.set_num_threads(SMOKE_THREADS)
    start = time.perf_counter()
    run = copy.copy(args)
    run.resume = False
    for path in (run.checkpoint, log_path(run.checkpoint)):
        path.unlink(missing_ok=True)
    for name in SMOKE_RESULT_FILES:
        (run.results_dir / name).unlink(missing_ok=True)
    phases = MemoryPhases()

    with phases.phase("1_lr0_forward"):
        forward = smoke_forward_equivalence(run)
    with phases.phase("2_train_step"):
        trained, weights_fingerprint = smoke_train_step(run)
    run.resume = True
    resumed, summary = smoke_resume_and_finish(run, phases, weights_fingerprint)
    gc.collect()

    weights, export = trained["weights"], summary["export"]
    steps = [r for r in trained["log"] if r["step"] > 0]
    table = summary.get("before_after", {}).get("validation", [])
    checks = {
        "lr0_model_output_matches_onnxruntime": forward["model_output_vs_onnxruntime"][
            "identical"
        ],
        "lr0_estimate_matches_uvr_inference": forward["estimate_vs_uvr_inference"][
            "identical"
        ],
        "train_loss_and_gradient_finite": bool(steps)
        and all(np.isfinite([r["train_loss"], r["grad_norm"]]).all() for r in steps),
        "train_all_trainable_tensors_changed": weights["n_tensors_changed"]
        == weights["n_trainable_tensors"]
        > 0,
        "train_batchnorm_statistics_unchanged": trained[
            "batchnorm_statistics_unchanged"
        ],
        "resume_same_step_and_weights": resumed["step"] == run.steps
        and resumed["same_weights"],
        "export_holds_new_weights": export["n_initializers_changed"]
        == weights["n_trainable_tensors"],
        "export_fits_onnxruntime_1_13": export["fits_onnxruntime_1_13"],
        "export_parity_pytorch_vs_onnxruntime": export["parity_pytorch_vs_export"][
            "identical"
        ],
        "export_onnxruntime_1_13": export["onnxruntime_1_13"]["passed"] is True,
        "sdr_before_after_table": len(table) == 2 * len(eval_datasets(run.dataset)),
    }
    report = {
        "model": summary["model"],
        "phases": phases.report,
        "ort113_subprocess_peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
        ),
        "torch_threads": torch.get_num_threads(),
        "seconds_per_step": trained["seconds_per_step"],
        "forward": forward,
        "train_log": trained["log"],
        "weights": weights,
        "export_vs_original_hq3": export["export_vs_original_hq3"],
        "before_after_validation": table,
        "checks": checks,
        "passed": all(checks.values()),
        "smoke_seconds": time.perf_counter() - start,
    }
    update_results(args.results_dir / "task3_finetune.json", "smoke", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.model = str(resolve_hq3(args.model))
    if args.smoke:
        report = smoke(args)
        for key, value in report.items():
            print(f"{key}: {value}")
        print("SMOKE PASSED" if report["passed"] else "SMOKE FAILED")
        return 0 if report["passed"] else 1
    trainer = train(args, args.stop_after)
    if trainer.step < args.steps:
        print(f"stopped at step {trainer.step}; rerun with --resume to continue")
        return 0
    summary = finish(trainer)
    for key in (
        "model",
        "steps",
        "train_seconds",
        "final_train_loss",
        "final_val_loss",
    ):
        print(f"{key}: {summary[key]}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
