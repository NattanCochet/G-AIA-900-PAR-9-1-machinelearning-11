"""Task 2 protocol: time needed to reach a target test accuracy on MNIST.

Trains the course baseline CNN (PyTorch port, see baseline.py) until the test
accuracy reaches TARGET_ACCURACY, and reports the complete learning time (data
loading + model setup + optimization), as asked in the subject. The control
evaluations used to decide when to stop, and the final inference timing, are
timed in their own PhaseTimer phases and excluded from that figure.

The accuracy is checked every `eval_every` optimization steps rather than once
per epoch: an epoch is 1875 steps at batch size 32, which is too coarse to
tell two methods apart if the target is reached early.

By default the images go through a torchvision DataLoader, which converts every
image on the CPU at each step (the bottleneck measured in
Results/result_task_2_default.md). With --data-on-device, MNIST is converted
once and kept as tensors on the device, and the batches are cut directly from
them. The conversion and the transfer count in the "load" phase.

Run from the repository root (-m, so that `import common` resolves):

    uv run python -m task2_mnist.time_to_target
    uv run python -m task2_mnist.time_to_target --batch-size 128 --lr 0.003
    uv run python -m task2_mnist.time_to_target --data-on-device
    uv run python -m task2_mnist.time_to_target --lr 0.003 --tag lr_sweep_gpu
    uv run python -m task2_mnist.time_to_target --data-on-device --profile

Each run appends one row to results/task2_mnist.csv (disable with --no-log),
with its settings, its `tag` (to group or discard the runs of one experiment),
the hardware, and the git commit of the code (`-dirty` if tracked files were
modified). The first run downloads MNIST, which inflates `load_seconds`: run it
once before measuring.

--profile replaces the time-to-target run: it records torch.profiler traces of
a few training steps and of one inference pass (task2_mnist/traces/, to open in
https://ui.perfetto.dev), and prints how long the GPU actually computes during
a step. Nothing is written to the CSV: the profiler slows the CPU side down, so
its timings must not be mixed with the measured runs.
"""

import argparse
import math
import os
import platform
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from torch import nn
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import PhaseTimer, git_commit, log_result, set_seed
from task2_mnist.baseline import DATA_DIR, RESULTS_CSV, CourseBaselineCNN

SEED = 0
TARGET_ACCURACY = 0.97
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Adam's default, as in baseline.py
EVAL_EVERY = 100  # optimization steps between two control evaluations
MAX_EPOCHS = 20  # give up if the target is not reached by then

TRACES_DIR = Path("task2_mnist/traces")  # trace_*.json files are git-ignored
PROFILE_WARMUP = 5  # profiled steps left out of the trace (CUDA, cuDNN warm up)
PROFILE_STEPS = 20  # training steps kept in the trace
WARMUP_SECONDS = 1.0  # training before timing: the GPU clocks up from idle
TIMED_STEPS = 500  # steps timed without the profiler, for the real step cost
INFERENCE_PASSES = 5  # inference passes averaged for the timing

Split = tuple[torch.Tensor, torch.Tensor]  # (images, labels)


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    """CUDA kernels run asynchronously: wait for them before reading a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cpu_model() -> str:
    """Commercial CPU name, e.g. "AMD Ryzen 7 3800XT 8-Core Processor"."""
    cpuinfo = Path("/proc/cpuinfo")  # Linux (Colab, WSL)
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Windows":
        # platform.processor() only gives "AMD64 Family 23 Model 113 ..." here
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def hardware_info(device: torch.device) -> dict:
    """Hardware description stored with each result (the subject asks for it).

    The clock frequency is not collected: note it from lscpu or Task Manager.
    """
    info = {
        "device": device.type,
        "cpu": cpu_model(),
        "cpu_threads": os.cpu_count(),
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Same as baseline.load_data, with a configurable batch size."""
    to_tensor = transforms.ToTensor()  # scales pixels to [0, 1], like /255
    train_set = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=to_tensor
    )
    test_set = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=to_tensor
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, test_loader


def cycle(loader: DataLoader):
    """Iterate over the loader forever, reshuffled at each pass.

    (itertools.cycle would replay the first pass's batches in the same order.)
    """
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        correct += (predictions == labels).sum().item()
    return correct / len(loader.dataset)


def to_float_images(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 images (N, 28, 28) -> float (N, 1, 28, 28) in [0, 1] on `device`,
    the same preprocessing as transforms.ToTensor. The transfer is done on the
    uint8 tensor (4 times smaller) and the conversion on the device."""
    return data.to(device).unsqueeze(1).float().div(255)


def load_data_on_device(device: torch.device) -> tuple[Split, Split]:
    """Whole MNIST as tensors on `device`: (train, test), each (images, labels)."""
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True)
    return (
        (to_float_images(train_set.data, device), train_set.targets.to(device)),
        (to_float_images(test_set.data, device), test_set.targets.to(device)),
    )


def batches_on_device(
    images: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> Iterator[Split]:
    """Same sampling as DataLoader(shuffle=True), repeated forever: one random
    permutation per epoch, cut in batches (the last one may be smaller)."""
    while True:
        permutation = torch.randperm(len(images), device=images.device)
        for start in range(0, len(images), batch_size):
            indices = permutation[start : start + batch_size]
            yield images[indices], labels[indices]


@torch.no_grad()
def evaluate_on_device(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, chunk: int = 2048
) -> float:
    """Test accuracy when the whole test set already lives on the device."""
    model.eval()
    correct = 0
    for start in range(0, len(images), chunk):
        predictions = model(images[start : start + chunk]).argmax(dim=1)
        correct += (predictions == labels[start : start + chunk]).sum()
    return correct.item() / len(images)


def prepare_data(
    batch_size: int, data_on_device: bool, device: torch.device
) -> tuple[Iterator[Split], int, Callable[[nn.Module], float]]:
    """Training batches (forever), number of steps per epoch, and a function
    giving the test accuracy of a model, through the DataLoader or with the
    whole dataset on `device`. Shared by the measured runs and --profile."""
    if data_on_device:
        (x_train, y_train), (x_test, y_test) = load_data_on_device(device)
        synchronize(device)  # the transfer counts as loading time
        return (
            batches_on_device(x_train, y_train, batch_size),
            math.ceil(len(x_train) / batch_size),
            lambda model: evaluate_on_device(model, x_test, y_test),
        )
    train_loader, test_loader = load_data(batch_size)
    return (
        cycle(train_loader),
        len(train_loader),
        lambda model: evaluate(model, test_loader, device),
    )


def make_train_step(
    model: nn.Module, learning_rate: float, device: torch.device
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """One optimization step on a batch (Adam, cross-entropy loss). Shared by
    the measured runs and --profile, so that both run exactly the same step."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def train_step(images: torch.Tensor, labels: torch.Tensor) -> None:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        criterion(model(images), labels).backward()
        optimizer.step()

    return train_step


def train_until_target(
    model: nn.Module,
    batches: Iterator[Split],
    steps_per_epoch: int,
    evaluate_fn: Callable[[], float],
    device: torch.device,
    learning_rate: float,
    target: float,
    eval_every: int,
    max_epochs: int,
    timer: PhaseTimer,
) -> dict:
    """Train by chunks of `eval_every` steps, checking the test accuracy after
    each chunk, until `target` is reached or `max_epochs` epochs are spent.

    `batches` yields (images, labels) forever, `evaluate_fn()` returns the
    current test accuracy of `model`.
    Time spent in a chunk goes to the "train" phase, the check to "eval".
    """
    train_step = make_train_step(model, learning_rate, device)
    max_steps = max_epochs * steps_per_epoch

    steps, accuracy = 0, 0.0
    while steps < max_steps:
        model.train()
        with timer.phase("train"):
            for _ in range(eval_every):
                train_step(*next(batches))
            synchronize(device)
        steps += eval_every

        with timer.phase("eval"):
            accuracy = evaluate_fn()
        if accuracy >= target:
            break

    return {
        "reached": accuracy >= target,
        "steps": steps,
        "test_accuracy": accuracy,
    }


def measure_inference(
    evaluate_fn: Callable[[], float], device: torch.device, timer: PhaseTimer
) -> None:
    """Time one full pass over the test set (data loader included, if any),
    after a warm-up pass."""
    evaluate_fn()
    with timer.phase("inference"):
        evaluate_fn()
        synchronize(device)


def run_experiment(
    *,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    target: float = TARGET_ACCURACY,
    eval_every: int = EVAL_EVERY,
    max_epochs: int = MAX_EPOCHS,
    data_on_device: bool = False,
    device: torch.device | None = None,
    tag: str | None = None,
    log: bool = True,
) -> dict:
    """One full run: seed, load, build the model, train until the target,
    time the inference. Importable by the sweep scripts.

    `tag` is only stored in the CSV, to group the runs of one experiment.
    """
    device = device or get_device("auto")
    code_version = git_commit()  # before training, in case files change meanwhile
    set_seed(seed)

    timer = PhaseTimer()
    with timer.phase("load"):
        batches, steps_per_epoch, test_accuracy = prepare_data(
            batch_size, data_on_device, device
        )
    with timer.phase("setup"):
        model = CourseBaselineCNN().to(device)  # also starts CUDA if used
        synchronize(device)

    def evaluate_fn() -> float:
        return test_accuracy(model)

    outcome = train_until_target(
        model,
        batches,
        steps_per_epoch,
        evaluate_fn,
        device,
        learning_rate,
        target,
        eval_every,
        max_epochs,
        timer,
    )
    measure_inference(evaluate_fn, device, timer)

    # complete learning procedure, without the control evaluations
    learning_seconds = timer.total("load", "setup", "train")
    result = {
        **outcome,
        "learning_seconds": learning_seconds,
        "phase_seconds": dict(timer.times),
    }

    if log:
        log_result(
            RESULTS_CSV,
            task="task2_mnist",
            seed=seed,
            split="test",
            metric="accuracy",
            score=outcome["test_accuracy"],
            duration_seconds=learning_seconds,
            config={
                "tag": tag,
                "git_commit": code_version,
                "protocol": "time_to_target",
                "model": "course_baseline_cnn",
                "optimizer": "adam",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "data_on_device": data_on_device,
                "target": target,
                "reached": outcome["reached"],
                "steps": outcome["steps"],
                "eval_every": eval_every,
                "load_seconds": timer.times["load"],
                "setup_seconds": timer.times["setup"],
                "train_seconds": timer.times["train"],
                "eval_seconds": timer.times["eval"],
                "inference_seconds": timer.times["inference"],
                **hardware_info(device),
            },
        )
    return result


def gpu_time_ms(prof: profile) -> float:
    """Time the GPU spent running kernels and copies in the recorded window, in
    ms. User annotations (ProfilerStep#N) also appear on the GPU timeline but
    are not GPU work: they are left out."""
    total_us = 0.0
    for event in prof.key_averages():
        if event.device_type == DeviceType.CUDA and not getattr(
            event, "is_user_annotation", False
        ):
            total_us += event.self_device_time_total
    return total_us / 1000


def profile_run(
    *,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    data_on_device: bool = False,
    device: torch.device | None = None,
) -> dict:
    """--profile: same data, model and training step as a measured run, but
    instead of training until the target:

    1. times TIMED_STEPS training steps without the profiler (real step cost),
       after WARMUP_SECONDS of training,
    2. records a trace of PROFILE_STEPS training steps, after PROFILE_WARMUP
       steps left out (CUDA and cuDNN are slower on the first steps),
    3. times INFERENCE_PASSES passes over the test set, then records one,
    4. prints the share of each step during which the GPU actually computes.

    The step cost and the GPU time come from different steps (the profiler
    slows the CPU side down), so when the GPU is the limit their ratio can
    read slightly above 100 %.
    The traces go to TRACES_DIR, named after the batch size, the data mode and
    the device, so that two configurations can be compared side by side.
    """
    device = device or get_device("auto")
    on_gpu = device.type == "cuda"
    set_seed(seed)
    batches, _, test_accuracy = prepare_data(batch_size, data_on_device, device)
    model = CourseBaselineCNN().to(device)
    train_step = make_train_step(model, learning_rate, device)

    activities = [ProfilerActivity.CPU]
    if on_gpu:
        activities.append(ProfilerActivity.CUDA)
    data_mode = "on_device" if data_on_device else "dataloader"
    label = f"bs{batch_size}_{data_mode}_{device.type}"
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    train_trace = TRACES_DIR / f"trace_train_{label}.json"
    inference_trace = TRACES_DIR / f"trace_inference_{label}.json"

    # 1. real cost of a training step, without the profiler
    model.train()
    deadline = time.perf_counter() + WARMUP_SECONDS
    while time.perf_counter() < deadline:
        train_step(*next(batches))
    synchronize(device)
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        train_step(*next(batches))
    synchronize(device)
    step_ms = (time.perf_counter() - start) * 1000 / TIMED_STEPS

    # 2. trace of the training steps (wait 1 step, warm up, then record).
    #    repeat=1: a single cycle, so that key_averages() reads the recorded
    #    steps and not a new, empty cycle started right after them.
    total_steps = 1 + PROFILE_WARMUP + PROFILE_STEPS
    with profile(
        activities=activities,
        schedule=schedule(
            wait=1, warmup=PROFILE_WARMUP, active=PROFILE_STEPS, repeat=1
        ),
        on_trace_ready=lambda p: p.export_chrome_trace(str(train_trace)),
    ) as prof:
        for step in range(total_steps):
            train_step(*next(batches))
            if step == total_steps - 1:
                synchronize(device)  # let the last kernels finish in the trace
            prof.step()
    train_gpu_ms = gpu_time_ms(prof) / PROFILE_STEPS
    sort_by = "self_device_time_total" if on_gpu else "self_cpu_time_total"
    train_table = prof.key_averages().table(sort_by=sort_by, row_limit=12)

    # 3. inference: timed passes without the profiler, then one traced pass
    #    (after one pass under the profiler left out, as for the training)
    test_accuracy(model)  # warm-up
    synchronize(device)
    start = time.perf_counter()
    for _ in range(INFERENCE_PASSES):
        test_accuracy(model)
    synchronize(device)
    inference_ms = (time.perf_counter() - start) * 1000 / INFERENCE_PASSES
    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(str(inference_trace)),
    ) as prof:
        for _ in range(2):
            test_accuracy(model)
            synchronize(device)
            prof.step()
    inference_gpu_ms = gpu_time_ms(prof)

    # 4. report
    print(f"\ntraining steps ({PROFILE_STEPS} recorded, sorted by {sort_by}):")
    print(train_table)
    print(f"training step: {step_ms:.2f} ms (mean of {TIMED_STEPS}, no profiler)")
    if on_gpu:
        print(
            f"  GPU busy: {train_gpu_ms:.2f} ms per step "
            f"({train_gpu_ms / step_ms:.0%} of the step)"
        )
    print(
        f"inference (whole test set): {inference_ms:.1f} ms "
        f"(mean of {INFERENCE_PASSES} passes, no profiler)"
    )
    if on_gpu:
        print(
            f"  GPU busy: {inference_gpu_ms:.1f} ms "
            f"({inference_gpu_ms / inference_ms:.0%} of the pass)"
        )
    print(f"traces (open in https://ui.perfetto.dev):\n  {train_trace}")
    print(f"  {inference_trace}")

    return {
        "step_ms": step_ms,
        "train_gpu_ms": train_gpu_ms if on_gpu else None,
        "inference_ms": inference_ms,
        "inference_gpu_ms": inference_gpu_ms if on_gpu else None,
        "traces": (train_trace, inference_trace),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--target", type=float, default=TARGET_ACCURACY)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--data-on-device",
        action="store_true",
        help="keep the whole dataset as tensors on the device (no DataLoader)",
    )
    parser.add_argument(
        "--tag",
        help="label stored in the CSV to group the runs of one experiment, "
        "e.g. lr_sweep_gpu",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="do not append to the results CSV"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record profiler traces of a few training steps and of one "
        "inference pass instead of a time-to-target run (nothing is logged)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"hardware: {hardware_info(device)}")
    print(f"data: {'on device' if args.data_on_device else 'DataLoader'}")

    if args.profile:
        profile_run(
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_on_device=args.data_on_device,
            device=device,
        )
        return

    result = run_experiment(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        target=args.target,
        eval_every=args.eval_every,
        max_epochs=args.max_epochs,
        data_on_device=args.data_on_device,
        device=device,
        tag=args.tag,
        log=not args.no_log,
    )

    status = "reached" if result["reached"] else "NOT reached"
    print(
        f"target {args.target} {status} after {result['steps']} steps "
        f"(test accuracy {result['test_accuracy']:.4f})"
    )
    print(f"learning time (load + setup + train): {result['learning_seconds']:.2f} s")
    for phase, seconds in result["phase_seconds"].items():
        print(f"  {phase}: {seconds:.2f} s")


if __name__ == "__main__":
    main()
