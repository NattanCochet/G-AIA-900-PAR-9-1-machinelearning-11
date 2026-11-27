"""T3-08: train SmallMaskNet from scratch, export it for UVR and measure its SDR.

    uv run python -m task3_uvr.train_small_nn --dataset musdb7 --remix \
        --steps 3000 --batch-size 8 --checkpoint <path>.pt [--resume]
    uv run python -m task3_uvr.train_small_nn --smoke     # ~2 min check on CPU

Data: the 80 clips of task3_uvr/splits/train.json, from musdb7 or one of its
degraded copies next to it (--dataset). The 14 clips of
task3_uvr/splits/validation.json give the validation loss, on fixed chunks
(seed VALIDATION_SEED), and the validation SDR. The test clips are only read
once, after training, for the final test SDR: they never influence training.

Loss: L1 between the predicted and the true vocal spectrograms (all 4
channels: real and imaginary parts, so phase errors count). It is computed
on the spectrogram, not on the waveform.

Every --eval-every steps, the mean training loss since the last row and the
validation loss go to <checkpoint>_log.csv. Every --checkpoint-every steps
and at the end, the checkpoint is saved atomically with the model, the
optimiser, every RNG state and the data generator (common.checkpointing).
--resume continues from it: the next batches are the ones the uninterrupted
run would have drawn. Without --resume an existing checkpoint is an error;
with --resume and no checkpoint, training starts from step 0.

At the end:
- ONNX export (opset 17) to task3_uvr/artifacts/small_nn/<model>.onnx and its
  UVR settings to artifacts/small_nn/model_data/<hash>.json;
- validation and test SDR with task3_uvr.mdx_inference, the UVR v5.6 port, and
  the model's settings read back from its hash: one row per clip and stem in
  results/task3_sdr_clips.csv (with the dataset, model and variant), medians
  and means in results/task3_uvr.csv;
- the loss curves in results/task3_small_nn_loss_<dataset>_<variant>.png.
"""

import argparse
import csv
import datetime
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import load_checkpoint, log_result, save_checkpoint, set_seed
from common.code_version import git_commit
from task3_uvr.check_data import resolve_data_dir
from task3_uvr.evaluate import STEMS, evaluate_split, split_clips
from task3_uvr.mdx_inference import load_settings, onnx_model, separate
from task3_uvr.onnx_checks import compare_outputs
from task3_uvr.plot_small_nn import plot_losses
from task3_uvr.sdr import EPS
from task3_uvr.small_nn import (
    MODEL_NAME,
    N_ZEROED_BINS,
    SmallMaskNet,
    export_onnx,
    model_input,
    n_parameters,
    stft,
    write_settings,
)
from task3_uvr.small_nn_data import (
    DATASETS,
    dataset_dir,
    load_clips,
    make_batch,
    split_names,
    validation_chunks,
)

ARTIFACTS_DIR = Path("task3_uvr") / "artifacts" / "small_nn"
SMOKE_DIR = Path("task3_uvr") / "artifacts" / "smoke"
VALIDATION_SEED = 1234
LOG_FIELDS = ["step", "train_loss", "val_loss", "train_seconds"]
CLIP_COLUMNS = [
    "date",
    "dataset",
    "model",
    "model_hash",
    "variant",
    "split",
    "match_freq_cut",
    "clip",
    "stem",
    "sdr_db",
    "reference_energy",
    "seconds",
]
CLIP_KEY = ["dataset", "model", "variant", "split", "match_freq_cut"]
# A resumed run must train the same model on the same data stream.
RESUME_KEYS = (
    "dataset",
    "remix",
    "batch_size",
    "lr",
    "seed",
    "channels",
    "max_train_clips",
)


def variant_name(remix: bool) -> str:
    return "remix" if remix else "noremix"


def model_name(dataset: str, remix: bool) -> str:
    """ONNX file stem and `model` column, e.g. SmallMaskNet_musdb7_phone_remix."""
    return f"{MODEL_NAME}_{dataset}_{variant_name(remix)}"


def log_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + "_log.csv")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=DATASETS, default=DATASETS[0])
    parser.add_argument(
        "--remix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="vocals of one clip over the accompaniment of another (default: on)",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="default: task3_uvr/artifacts/small_nn/<model>.pt",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
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
        "--sdr-splits",
        nargs="*",
        default=["validation", "test"],
        choices=["validation", "test"],
        help="splits to measure the SDR on after training (none: skip)",
    )
    parser.add_argument("--max-train-clips", type=int)
    parser.add_argument("--max-val-clips", type=int)
    parser.add_argument("--max-sdr-clips", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 steps, batch 2, 4 training and 2 validation clips, CPU, with a "
        "stop and resume at step 10; outputs in task3_uvr/artifacts/smoke/",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    if args.checkpoint is None:
        args.checkpoint = (
            args.artifacts_dir / f"{model_name(args.dataset, args.remix)}.pt"
        )
    return args


def apply_smoke(args: argparse.Namespace) -> None:
    args.steps, args.batch_size, args.device = 20, 2, "cpu"
    args.max_train_clips, args.max_val_clips, args.max_sdr_clips = 4, 2, 2
    args.val_chunks_per_clip = 1
    args.eval_every, args.checkpoint_every = 5, 10
    args.artifacts_dir = SMOKE_DIR
    args.results_dir = SMOKE_DIR / "results"
    args.checkpoint = SMOKE_DIR / f"{model_name(args.dataset, args.remix)}.pt"


def run_config(args: argparse.Namespace) -> dict:
    return {
        "dataset": args.dataset,
        "remix": args.remix,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "channels": args.channels,
        "max_train_clips": args.max_train_clips,
        "loss": "L1 on the spectrogram (real and imaginary parts)",
        "validation_seed": VALIDATION_SEED,
        "val_chunks_per_clip": args.val_chunks_per_clip,
    }


class Trainer:
    """Everything a training session needs; `run` trains from the current
    step, `save` writes the checkpoint."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = resolve_device(args.device)
        musdb7 = resolve_data_dir(args.data_dir)
        self.data_dir = dataset_dir(musdb7, args.dataset)
        train_names = split_names("train")[: args.max_train_clips]
        val_names = split_names("validation")[: args.max_val_clips]
        self.train_clips = load_clips(self.data_dir, train_names)
        val_clips = load_clips(self.data_dir, val_names)
        self.val_mix, self.val_vocals = validation_chunks(
            val_clips, args.val_chunks_per_clip, VALIDATION_SEED
        )
        self.n_train_clips, self.n_val_clips = len(train_names), len(val_names)

        set_seed(args.seed)
        self.model = SmallMaskNet(channels=args.channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.rng = random.Random(args.seed)  # batches only
        self.step = 0  # completed steps
        self.train_seconds = 0.0
        self.loss_sum, self.loss_count = 0.0, 0  # since the last log row
        self.config = run_config(args)

    # -- checkpoint --------------------------------------------------------
    def save(self) -> None:
        extra = {
            "config": self.config,
            "data_rng": self.rng.getstate(),
            "train_seconds": self.train_seconds,
            "loss_sum": self.loss_sum,
            "loss_count": self.loss_count,
        }
        # `epoch` is the index of the last completed step.
        save_checkpoint(
            self.args.checkpoint, self.model, self.optimizer, self.step - 1, extra
        )

    def restore(self) -> None:
        self.step, extra = load_checkpoint(
            self.args.checkpoint, self.model, self.optimizer, map_location=self.device
        )
        saved = extra["config"]
        changed = {
            k: (saved.get(k), self.config[k])
            for k in RESUME_KEYS
            if saved.get(k) != self.config[k]
        }
        if changed:
            sys.exit(f"--resume with different settings (saved, given): {changed}")
        self.rng.setstate(extra["data_rng"])
        self.train_seconds = extra["train_seconds"]
        self.loss_sum, self.loss_count = extra["loss_sum"], extra["loss_count"]
        truncate_log(log_path(self.args.checkpoint), self.step)

    # -- training ----------------------------------------------------------
    def loss(self, mix: torch.Tensor, vocals: torch.Tensor) -> torch.Tensor:
        """L1 on the spectrogram. The target's 3 lowest bins are zeroed like
        the input's: the model output is 0 there whatever it does."""
        x = model_input(mix.to(self.device))
        target = stft(vocals.to(self.device))
        target[:, :, :N_ZEROED_BINS] = 0
        return F.l1_loss(self.model(x), target)

    @torch.no_grad()
    def validation_loss(self) -> float:
        self.model.eval()
        total, n = 0.0, len(self.val_mix)
        for i in range(0, n, self.args.batch_size):
            mix = self.val_mix[i : i + self.args.batch_size]
            total += self.loss(
                mix, self.val_vocals[i : i + self.args.batch_size]
            ).item() * len(mix)
        self.model.train()
        return total / n

    def run(self, stop_after: int | None = None) -> None:
        args = self.args
        end = (
            args.steps
            if stop_after is None
            else min(args.steps, self.step + stop_after)
        )
        self.model.train()
        while self.step < end:
            start = time.perf_counter()
            mix, vocals = make_batch(
                self.train_clips, args.batch_size, args.remix, self.rng
            )
            loss = self.loss(mix, vocals)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.loss_sum += loss.item()
            self.loss_count += 1
            self.step += 1
            self.train_seconds += time.perf_counter() - start

            last = self.step == args.steps
            if self.step % args.eval_every == 0 or last:
                row = {
                    "step": self.step,
                    "train_loss": self.loss_sum / self.loss_count,
                    "val_loss": self.validation_loss(),
                    "train_seconds": self.train_seconds,
                }
                append_log(log_path(args.checkpoint), row)
                self.loss_sum, self.loss_count = 0.0, 0
                print(
                    f"step {row['step']}/{args.steps}  train {row['train_loss']:.5f}  "
                    f"val {row['val_loss']:.5f}  {self.train_seconds:.0f} s",
                    flush=True,
                )
            if self.step % args.checkpoint_every == 0 or last or self.step == end:
                self.save()


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def truncate_log(path: Path, step: int) -> None:
    """Drop log rows written after the checkpoint (they will be redone)."""
    if not path.exists():
        return
    log = pd.read_csv(path)
    log[log["step"] <= step].to_csv(path, index=False)


def save_clip_rows(new: pd.DataFrame, path: Path) -> None:
    """Add per-clip rows to the shared CSV, replacing those of a previous run
    with the same dataset, model, variant and split. Older rows get
    dataset "musdb7" if the file predates T3-07's dataset column, and an
    empty variant. Old rows are read and written back as text, so they keep
    their exact values."""
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "dataset" not in old.columns:
            old.insert(1, "dataset", DATASETS[0])
        if "variant" not in old.columns:
            old.insert(old.columns.get_loc("model_hash") + 1, "variant", "")
        done = new[CLIP_KEY].astype(str).drop_duplicates()
        old = old.merge(done, on=CLIP_KEY, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


def export_and_check(trainer: Trainer, onnx_path: Path) -> tuple[Path, dict]:
    """Export to ONNX, write the UVR settings, compare onnxruntime with
    PyTorch on one validation chunk and save that input/output pair for the
    onnxruntime 1.13.1 check (task3_uvr.check_onnx_io)."""
    export_onnx(trainer.model, onnx_path)
    settings_path = write_settings(onnx_path, onnx_path.parent / "model_data")
    x = model_input(trainer.val_mix[:1])
    trainer.model.eval()
    with torch.no_grad():
        expected = trainer.model(x.to(trainer.device)).cpu().numpy()
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"input": x.numpy()})[0]
    np.savez_compressed(
        onnx_path.with_name(onnx_path.stem + "_io.npz"),
        input=x.numpy(),
        output=expected,
    )
    return settings_path, compare_outputs(expected, got)


def measure_sdr(trainer: Trainer, onnx_path: Path, name: str) -> dict:
    """Validation (and test) SDR of the exported model, through the UVR port,
    with the settings found from its hash as UVR would."""
    args = trainer.args
    model_hash, settings = load_settings(onnx_path, onnx_path.parent / "model_data")
    model = onnx_model(onnx_path, settings)

    def estimator(mix: np.ndarray) -> dict[str, np.ndarray]:
        return separate(mix, model, settings)

    variant = variant_name(args.remix)
    date = datetime.date.today().isoformat()
    medians = {}
    for split in args.sdr_splits:
        clips = split_clips(trainer.data_dir, split)[: args.max_sdr_clips]
        print(f"{name} on {args.dataset} {split} ({len(clips)} clips)")
        rows, total = evaluate_split(name, estimator, clips)
        frame = pd.DataFrame(rows).assign(
            date=date,
            dataset=args.dataset,
            model=name,
            model_hash=model_hash,
            variant=variant,
            split=split,
            match_freq_cut=False,
        )[CLIP_COLUMNS]
        save_clip_rows(frame, args.results_dir / "task3_sdr_clips.csv")
        config = {
            **trainer.config,
            "model": name,
            "model_hash": model_hash,
            "variant": variant,
            "steps": trainer.step,
            "n_clips": len(clips),
            "sdr": "global (MDX Challenge)",
            "eps": EPS,
            "match_freq_cut": False,
            "n_fft": settings.n_fft,
            "dim_f": settings.dim_f,
            "segment_size": settings.dim_t,
            "overlap": "Default",
            "compensate": settings.compensate,
            "primary_stem": settings.primary_stem,
            "code": git_commit(),
        }
        for stem in STEMS:
            values = frame.loc[frame["stem"] == stem, "sdr_db"]
            medians[f"{split}_{stem}"] = float(values.median())
            for stat in ("median", "mean"):
                log_result(
                    args.results_dir / "task3_uvr.csv",
                    task="task3_uvr",
                    seed=args.seed,
                    split=split,
                    metric=f"sdr_{stat}_{stem}",
                    score=float(getattr(values, stat)()),
                    duration_seconds=total,
                    config=config,
                )
            print(
                f"{name} {args.dataset} {split} {stem}: median {values.median():.2f} dB"
            )
    return medians


def finish(trainer: Trainer) -> dict:
    args = trainer.args
    name = model_name(args.dataset, args.remix)
    log = pd.read_csv(log_path(args.checkpoint))
    final = log.iloc[-1]
    for split, column in (("train", "train_loss"), ("validation", "val_loss")):
        log_result(
            args.results_dir / "task3_uvr.csv",
            task="task3_uvr",
            seed=args.seed,
            split=split,
            metric="l1_spectrogram_loss",
            score=float(final[column]),
            duration_seconds=trainer.train_seconds,
            config={
                **trainer.config,
                "model": name,
                "steps": trainer.step,
                "n_parameters": n_parameters(trainer.model),
                "device": str(trainer.device),
                "code": git_commit(),
            },
        )
    figure = (
        args.results_dir
        / f"task3_small_nn_loss_{args.dataset}_{variant_name(args.remix)}.png"
    )
    plot_losses({name: log_path(args.checkpoint)}, figure)

    onnx_path = args.artifacts_dir / f"{name}.onnx"
    settings_path, parity = export_and_check(trainer, onnx_path)
    print(f"exported {onnx_path}, settings {settings_path}")
    print(
        f"onnxruntime vs PyTorch: max abs diff {parity['max_abs_diff']:.2e} "
        f"({'identical' if parity['identical'] else 'DIFFERENT'})"
    )
    summary = {
        "model": name,
        "n_parameters": n_parameters(trainer.model),
        "steps": trainer.step,
        "train_seconds": trainer.train_seconds,
        "final_train_loss": float(final["train_loss"]),
        "final_val_loss": float(final["val_loss"]),
        "onnx": str(onnx_path),
        "parity": parity,
        "figure": str(figure),
    }
    if args.sdr_splits:
        summary["median_sdr_db"] = measure_sdr(trainer, onnx_path, name)
    return summary


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
    elif args.resume:
        print(f"no checkpoint at {args.checkpoint}: starting from step 0")
        log_path(args.checkpoint).unlink(missing_ok=True)
    else:
        log_path(args.checkpoint).unlink(missing_ok=True)
    print(
        f"{model_name(args.dataset, args.remix)}: {n_parameters(trainer.model):,} parameters, "
        f"{trainer.n_train_clips} training / {trainer.n_val_clips} validation clips, "
        f"device {trainer.device}"
    )
    trainer.run(stop_after)
    return trainer


def smoke(args: argparse.Namespace) -> dict:
    """Train 10 steps and stop, resume from the checkpoint to step 20, then
    export and measure the SDR on 2 validation and 2 test clips."""
    start = time.perf_counter()
    for path in (args.checkpoint, log_path(args.checkpoint)):
        path.unlink(missing_ok=True)
    args.resume = False
    train(args, stop_after=10)
    args.resume = True
    trainer = train(args)
    if trainer.step != args.steps:
        raise RuntimeError(
            f"resume ended at step {trainer.step}, expected {args.steps}"
        )
    summary = finish(trainer)
    summary["smoke_seconds"] = time.perf_counter() - start
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        summary = smoke(args)
    else:
        trainer = train(args, args.stop_after)
        if trainer.step < args.steps:
            print(f"stopped at step {trainer.step}; rerun with --resume to continue")
            return 0
        summary = finish(trainer)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
