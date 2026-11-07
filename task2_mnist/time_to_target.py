"""Task 2 protocol: time needed to reach a target test accuracy on MNIST.

Trains the course baseline CNN (PyTorch port, see baseline.py) until the test
accuracy reaches TARGET_ACCURACY, and reports the complete learning time (data
loading + model setup + optimization), as asked in the subject. The control
evaluations used to decide when to stop, and the final inference timing, are
timed in their own PhaseTimer phases and excluded from that figure.

The accuracy is checked every `eval_every` optimization steps rather than once
per epoch: an epoch is 1875 steps at batch size 32, which is too coarse to
tell two methods apart if the target is reached early.

Run from the repository root (-m, so that `import common` resolves):

    uv run python -m task2_mnist.time_to_target
    uv run python -m task2_mnist.time_to_target --batch-size 128 --lr 0.003

Each run appends one row to results/task2_mnist.csv (disable with --no-log).
The first run downloads MNIST, which inflates `load_seconds`: run it once
before measuring.
"""

import argparse
import os
import platform
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import PhaseTimer, log_result, set_seed
from task2_mnist.baseline import DATA_DIR, RESULTS_CSV, CourseBaselineCNN

SEED = 0
TARGET_ACCURACY = 0.97
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Adam's default, as in baseline.py
EVAL_EVERY = 100  # optimization steps between two control evaluations
MAX_EPOCHS = 20  # give up if the target is not reached by then


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    """CUDA kernels run asynchronously: wait for them before reading a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")  # Linux (Colab, WSL)
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
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


def train_until_target(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    target: float,
    eval_every: int,
    max_epochs: int,
    timer: PhaseTimer,
) -> dict:
    """Train by chunks of `eval_every` steps, checking the test accuracy after
    each chunk, until `target` is reached or `max_epochs` epochs are spent.

    Time spent in a chunk goes to the "train" phase, the check to "eval".
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    batches = cycle(train_loader)
    max_steps = max_epochs * len(train_loader)

    steps, accuracy = 0, 0.0
    while steps < max_steps:
        model.train()
        with timer.phase("train"):
            for _ in range(eval_every):
                images, labels = next(batches)
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                criterion(model(images), labels).backward()
                optimizer.step()
            synchronize(device)
        steps += eval_every

        with timer.phase("eval"):
            accuracy = evaluate(model, test_loader, device)
        if accuracy >= target:
            break

    return {
        "reached": accuracy >= target,
        "steps": steps,
        "test_accuracy": accuracy,
    }


def measure_inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    timer: PhaseTimer,
) -> None:
    """Time one full pass over the test set (data loader included), after a
    warm-up pass."""
    evaluate(model, test_loader, device)
    with timer.phase("inference"):
        evaluate(model, test_loader, device)
        synchronize(device)


def run_experiment(
    *,
    seed: int = SEED,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    target: float = TARGET_ACCURACY,
    eval_every: int = EVAL_EVERY,
    max_epochs: int = MAX_EPOCHS,
    device: torch.device | None = None,
    log: bool = True,
) -> dict:
    """One full run: seed, load, build the model, train until the target,
    time the inference. Importable by the sweep scripts."""
    device = device or get_device("auto")
    set_seed(seed)

    timer = PhaseTimer()
    with timer.phase("load"):
        train_loader, test_loader = load_data(batch_size)
    with timer.phase("setup"):
        model = CourseBaselineCNN().to(device)  # also starts CUDA if used
        synchronize(device)

    outcome = train_until_target(
        model,
        train_loader,
        test_loader,
        device,
        learning_rate,
        target,
        eval_every,
        max_epochs,
        timer,
    )
    measure_inference(model, test_loader, device, timer)

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
                "protocol": "time_to_target",
                "model": "course_baseline_cnn",
                "optimizer": "adam",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
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
        "--no-log", action="store_true", help="do not append to the results CSV"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"hardware: {hardware_info(device)}")

    result = run_experiment(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        target=args.target,
        eval_every=args.eval_every,
        max_epochs=args.max_epochs,
        device=device,
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
