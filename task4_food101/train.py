"""Fine-tune a Food-101 reference model or distill it into a smaller model."""

import argparse
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Food101

from common import PhaseTimer, log_result, set_seed
from common.checkpointing import load_checkpoint, save_checkpoint

SEED = 0
NUM_CLASSES = 101
TEACHER_MODEL = "resnet50"
STUDENT_MODEL = "mobilenetv3_small_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--data-dir", type=Path, default=Path("task4_food101/data"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to save/resume. Defaults to task4_food101/checkpoints/<mode>.pt",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Fine-tuned reference-model checkpoint (required in student mode)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def make_model(model_name: str, *, pretrained: bool) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)


def make_loaders(
    data_dir: Path, model: nn.Module, batch_size: int, num_workers: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    data_config = resolve_model_data_config(model)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)

    train_data = Food101(
        root=data_dir, split="train", download=True, transform=train_transform
    )
    eval_data = Food101(
        root=data_dir, split="train", download=True, transform=eval_transform
    )
    train_size = int(0.9 * len(train_data))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_data), generator=generator).tolist()
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(eval_data, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def load_teacher(path: Path, device: torch.device) -> nn.Module:
    teacher = make_model(TEACHER_MODEL, pretrained=False)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    extra = checkpoint.get("extra", {})
    if extra.get("model_name") != TEACHER_MODEL or extra.get("mode") != "teacher":
        raise ValueError(f"{path} is not a task 4 {TEACHER_MODEL} teacher checkpoint")
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.to(device).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    teacher: nn.Module | None,
    temperature: float,
    distillation_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(images)
        hard_loss = F.cross_entropy(student_logits, labels)
        if teacher is None:
            loss = hard_loss
        else:
            with torch.no_grad():
                teacher_logits = teacher(images)
            soft_loss = (
                F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                )
                * temperature**2
            )
            hard_weight = 1.0 - distillation_weight
            loss = hard_weight * hard_loss + distillation_weight * soft_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError(
            "epochs and batch-size must be positive; num-workers cannot be negative"
        )
    if not 0.0 <= args.distillation_weight <= 1.0:
        raise ValueError("distillation-weight must be between 0 and 1")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.mode == "student" and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required in student mode")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = TEACHER_MODEL if args.mode == "teacher" else STUDENT_MODEL
    model = make_model(model_name, pretrained=True).to(device)
    teacher = (
        load_teacher(args.teacher_checkpoint, device)
        if args.mode == "student"
        else None
    )
    train_loader, val_loader = make_loaders(
        args.data_dir, model, args.batch_size, args.num_workers, args.seed
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.checkpoint or Path(
        f"task4_food101/checkpoints/{args.mode}.pt"
    )

    start_epoch = 0
    previous_training_seconds = 0.0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot resume; checkpoint not found: {checkpoint_path}"
            )
        start_epoch, extra = load_checkpoint(
            checkpoint_path, model, optimizer, map_location=device
        )
        if extra.get("model_name") != model_name or extra.get("mode") != args.mode:
            raise ValueError(
                f"{checkpoint_path} was created for a different model or mode"
            )
        previous_training_seconds = extra.get("training_seconds", 0.0)

    timer = PhaseTimer()
    for epoch in range(start_epoch, args.epochs):
        with timer.phase("train"):
            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                teacher,
                args.temperature,
                args.distillation_weight,
            )
        with timer.phase("validation"):
            val_accuracy = evaluate(model, val_loader, device)
        total_training_seconds = previous_training_seconds + timer.total("train")
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"train loss={loss:.4f}, validation accuracy={val_accuracy:.4f}"
        )
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            extra={
                "mode": args.mode,
                "model_name": model_name,
                "training_seconds": total_training_seconds,
            },
        )

    training_seconds = previous_training_seconds + timer.total("train")
    test_data = Food101(
        root=args.data_dir,
        split="test",
        download=True,
        transform=create_transform(
            **resolve_model_data_config(model), is_training=False
        ),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_accuracy = evaluate(model, test_loader, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"test accuracy: {test_accuracy:.4f}")
    print(f"trainable model parameters: {parameter_count:,}")
    log_result(
        "results/task4_food101.csv",
        task="task4_food101",
        seed=args.seed,
        split="test",
        metric="accuracy",
        score=test_accuracy,
        duration_seconds=training_seconds,
        config={
            "mode": args.mode,
            "model": model_name,
            "parameters": parameter_count,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature if teacher is not None else None,
            "distillation_weight": (
                args.distillation_weight if teacher is not None else None
            ),
        },
    )


if __name__ == "__main__":
    main()
