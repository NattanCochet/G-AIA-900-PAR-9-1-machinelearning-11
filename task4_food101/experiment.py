"""Food-101 compression experiments. See the task 4 section of the README."""

import argparse
import csv
import hashlib
import io
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.datasets import Food101
from torchvision.models import quantization as quant_models
from transformers import AutoImageProcessor, AutoModelForImageClassification

from common import set_seed

REFERENCE = "skylord/swin-finetuned-food101"
REFERENCE_REVISION = "490c9d67095b36fb709dbd553946e8eb5d97390c"
ARCHITECTURES = {
    "mobilenet_v3_large": (
        quant_models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
    "resnet18": (quant_models.resnet18, models.ResNet18_Weights.DEFAULT),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
RESULT_FIELDS = [
    "name",
    "split",
    "top1",
    "top5",
    "parameters",
    "size_mb",
    "latency_ms",
    "threads",
    "device",
    "seed",
    "method",
    "architecture",
    "artifact",
]


def split_indices(length: int, seed: int) -> tuple[list[int], list[int]]:
    indices = torch.randperm(
        length, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    cut = int(length * 0.9)
    return indices[:cut], indices[cut:]


def make_model(architecture: str, pretrained: bool = False) -> nn.Module:
    constructor, weights = ARCHITECTURES[architecture]
    model = constructor(weights=weights if pretrained else None)
    if architecture == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 101)
    elif architecture == "mobilenet_v3_large":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 101)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 101)
    return model


def reference_order(id2label: dict, classes: list[str]) -> torch.Tensor:
    labels = [
        id2label[i] if i in id2label else id2label[str(i)] for i in range(len(id2label))
    ]
    if len(labels) != len(classes) or set(labels) != set(classes):
        raise ValueError("Reference labels do not match Food-101 classes")
    return torch.tensor([labels.index(label) for label in classes])


class IndexedFood(Dataset):
    def __init__(self, dataset: Food101, indices: list[int], transform=None):
        self.dataset, self.indices, self.transform = dataset, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.indices[index]


def dataset_for(
    data_dir: Path, split: str, seed: int, transform=TRANSFORM
) -> IndexedFood:
    data = Food101(
        root=data_dir, split="test" if split == "test" else "train", download=False
    )
    if split == "test":
        indices = list(range(len(data)))
    else:
        train, validation = split_indices(len(data), seed)
        indices = train if split == "train" else validation
    return IndexedFood(data, indices, transform)


def loader(data, batch_size: int, workers: int, shuffle: bool = False, collate_fn=None):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
    )


def reference(device):
    processor = AutoImageProcessor.from_pretrained(
        REFERENCE, revision=REFERENCE_REVISION
    )
    model = (
        AutoModelForImageClassification.from_pretrained(
            REFERENCE, revision=REFERENCE_REVISION
        )
        .to(device)
        .eval()
    )
    return processor, model


def collate_images(batch):
    images, labels, indices = zip(*batch)
    return list(images), torch.tensor(labels), torch.tensor(indices)


def cache_logits(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset_for(args.data_dir, "train", args.seed, transform=None)
    processor, teacher = reference(device)
    order = reference_order(teacher.config.id2label, data.dataset.classes).to(device)
    logits = torch.zeros((len(data.dataset), 101), dtype=torch.float32)
    batches = loader(data, args.batch_size, args.workers, collate_fn=collate_images)
    with torch.inference_mode():
        for images, _, indices in batches:
            inputs = processor(images=images, return_tensors="pt").to(device)
            logits[indices] = teacher(**inputs).logits[:, order].cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": logits,
            "indices": data.indices,
            "seed": args.seed,
            "classes": data.dataset.classes,
            "reference": REFERENCE,
            "revision": REFERENCE_REVISION,
        },
        args.output,
    )
    print(f"Cached {len(data)} training examples in {args.output}")


def load_cache(path: Path, data: IndexedFood, seed: int) -> torch.Tensor:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if (
        cache["seed"] != seed
        or cache["indices"] != data.indices
        or cache["classes"] != data.dataset.classes
        or cache["reference"] != REFERENCE
        or cache["revision"] != REFERENCE_REVISION
        or cache["logits"].shape != (len(data.dataset), 101)
    ):
        raise ValueError("Logit cache does not match the dataset, split, or reference")
    return cache["logits"]


@torch.inference_mode()
def accuracy(model, batches, device):
    model.eval()
    correct1 = correct5 = total = 0
    for images, labels, _ in batches:
        logits = model(images.to(device))
        top5 = logits.topk(5, dim=1).indices.cpu()
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        total += len(labels)
    if not total:
        raise ValueError("Cannot evaluate an empty dataset")
    return correct1 / total, correct5 / total


def save_model(path: Path, model: nn.Module, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "metadata": metadata}, path)


def pruning_targets(model):
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset_for(args.data_dir, "train", args.seed)
    validation = dataset_for(args.data_dir, "validation", args.seed)
    batches = loader(data, args.batch_size, args.workers, shuffle=True)
    val_batches = loader(validation, args.batch_size, args.workers)
    cache = load_cache(args.cache, data, args.seed) if args.cache else None
    if args.alpha and cache is None:
        raise ValueError("A logit cache is required when alpha is nonzero")
    model = make_model(args.architecture, pretrained=not (args.start or args.resume))
    if args.start:
        model, metadata = load_model(args.start)
        if (
            metadata["architecture"] != args.architecture
            or metadata["method"] not in ("baseline", "distilled", "pruned")
            or metadata["seed"] != args.seed
        ):
            raise ValueError("Starting checkpoint must be a matching float model")
    targets = pruning_targets(model)
    if args.prune:
        prune.global_unstructured(
            targets, pruning_method=prune.L1Unstructured, amount=args.prune
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best = -1.0
    start_epoch = 0
    resume_path = args.output.with_name(args.output.name + ".resume.pt")
    config = {
        "architecture": args.architecture,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "data_dir": str(args.data_dir.resolve()),
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "prune": args.prune,
        "start": file_hash(args.start) if args.start else None,
        "cache": file_hash(args.cache) if args.cache else None,
        "device_type": device.type,
    }
    if args.resume:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        if saved["config"] != config or not args.output.exists():
            raise ValueError("Resume configuration or best checkpoint does not match")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best = saved["epoch"] + 1, saved["best"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    method = "pruned" if args.prune else ("distilled" if args.alpha else "baseline")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for images, labels, indices in batches:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if cache is not None and args.alpha:
                soft = cache[indices].to(device)
                loss = (
                    1 - args.alpha
                ) * loss + args.alpha * args.temperature**2 * F.kl_div(
                    F.log_softmax(logits / args.temperature, dim=1),
                    F.softmax(soft / args.temperature, dim=1),
                    reduction="batchmean",
                )
            loss.backward()
            optimizer.step()
        top1, top5 = accuracy(model, val_batches, device)
        print(
            f"epoch {epoch + 1}: validation top-1={top1:.4f}, top-5={top5:.4f}",
            flush=True,
        )
        if top1 > best:
            best = top1
            if args.prune:
                state = {}
                for key, value in model.state_dict().items():
                    if key.endswith("_mask"):
                        continue
                    if key.endswith("_orig"):
                        prefix = key[:-5]
                        state[prefix] = (
                            value * model.state_dict()[prefix + "_mask"]
                        ).cpu()
                    else:
                        state[key] = value.cpu()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state": state,
                        "metadata": {
                            "architecture": args.architecture,
                            "method": method,
                            "seed": args.seed,
                            "alpha": args.alpha,
                            "temperature": args.temperature,
                            "prune": args.prune,
                            "validation_top1": best,
                        },
                    },
                    args.output,
                )
            else:
                save_model(
                    args.output,
                    model,
                    {
                        "architecture": args.architecture,
                        "method": method,
                        "seed": args.seed,
                        "alpha": args.alpha,
                        "temperature": args.temperature,
                        "prune": args.prune,
                        "validation_top1": best,
                    },
                )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "config": config,
                "rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
            },
            resume_path,
        )


def quantize(model, method: str, calibration=None):
    if method == "dynamic":
        return torch.ao.quantization.quantize_dynamic(
            model.eval(), {nn.Linear}, dtype=torch.qint8
        )
    if not hasattr(model, "fuse_model"):
        raise ValueError(
            "Static quantization and QAT require a quantizable torchvision model"
        )
    torch.backends.quantized.engine = "x86"
    model.eval()
    model.fuse_model()
    model.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    torch.ao.quantization.prepare(model, inplace=True)
    if calibration is not None:
        with torch.inference_mode():
            for images, _, _ in calibration:
                model(images)
    torch.ao.quantization.convert(model, inplace=True)
    return model


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    model = make_model(metadata["architecture"])
    if metadata["method"] in ("dynamic", "static", "qat"):
        model = quantize(
            model, "dynamic" if metadata["method"] == "dynamic" else "static"
        )
    model.load_state_dict(checkpoint["state"])
    return model.eval(), metadata


def compress(args):
    model, metadata = load_model(args.input)
    if metadata["method"] not in ("baseline", "distilled", "pruned"):
        raise ValueError("Compression input must be a float model")
    if metadata["seed"] != args.seed:
        raise ValueError("Compression seed must match the input checkpoint")
    if args.method in ("static", "qat") and metadata["architecture"] != "resnet18":
        raise ValueError("Static quantization and QAT currently support ResNet-18 only")
    torch.backends.quantized.engine = "x86"
    if args.method == "qat":
        set_seed(args.seed)
        model.train()
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        torch.ao.quantization.prepare_qat(model, inplace=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        data = dataset_for(args.data_dir, "train", args.seed)
        validation = dataset_for(args.data_dir, "validation", args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best = -1.0
        import copy

        for epoch in range(args.epochs):
            model.train()
            for images, labels, _ in loader(
                data, args.batch_size, args.workers, shuffle=True
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
            converted = torch.ao.quantization.convert(copy.deepcopy(model).cpu().eval())
            score, _ = accuracy(
                converted, loader(validation, args.batch_size, args.workers), "cpu"
            )
            print(f"QAT epoch {epoch + 1}: validation top-1={score:.4f}", flush=True)
            if score > best:
                best = score
                save_model(
                    args.output,
                    converted,
                    {
                        **metadata,
                        "method": "qat",
                        "source_method": metadata["method"],
                        "source_artifact": file_hash(args.input),
                        "seed": args.seed,
                    },
                )
        return
    calibration = None
    if args.method == "static":
        data = dataset_for(args.data_dir, "train", args.seed)
        calibration = loader(
            Subset(data, range(min(args.calibration_samples, len(data)))),
            args.batch_size,
            args.workers,
        )
    model = quantize(model, args.method, calibration)
    save_model(
        args.output,
        model,
        {
            **metadata,
            "method": args.method,
            "source_method": metadata["method"],
            "source_artifact": file_hash(args.input),
            "seed": args.seed,
        },
    )


def median_latency(model, images: torch.Tensor, warmup: int, passes: int) -> float:
    model.eval()
    with torch.inference_mode():
        for i in range(warmup):
            model(images[i % len(images) : i % len(images) + 1])
        timings = []
        for _ in range(passes):
            start = time.perf_counter()
            for image in images:
                model(image.unsqueeze(0))
            timings.append((time.perf_counter() - start) * 1000 / len(images))
    return statistics.median(timings)


def write_result(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != RESULT_FIELDS:
                raise ValueError(f"Unexpected result columns in {path}")
            rows = [
                r
                for r in reader
                if (r["name"], r["split"]) != (row["name"], row["split"])
            ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows + [row])


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_description() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def measure(args):
    torch.set_num_threads(args.threads)
    data = dataset_for(
        args.data_dir,
        args.split,
        args.seed,
        transform=None if args.reference else TRANSFORM,
    )
    if args.reference:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor, model = reference(device)
        order = reference_order(model.config.id2label, data.dataset.classes).to(device)
        batches = loader(data, args.batch_size, args.workers, collate_fn=collate_images)
        correct1 = correct5 = total = 0
        benchmark = []
        with torch.inference_mode():
            for images, labels, _ in batches:
                inputs = processor(images=images, return_tensors="pt")["pixel_values"]
                if sum(len(x) for x in benchmark) < args.latency_images:
                    benchmark.append(
                        inputs[: args.latency_images - sum(len(x) for x in benchmark)]
                    )
                top5 = (
                    model(pixel_values=inputs.to(device))
                    .logits[:, order]
                    .topk(5, dim=1)
                    .indices.cpu()
                )
                correct1 += (top5[:, 0] == labels).sum().item()
                correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
                total += len(labels)
        if not total:
            raise ValueError("Cannot evaluate an empty dataset")
        top1, top5 = correct1 / total, correct5 / total
        model.cpu()
        size = io.BytesIO()
        torch.save(model.state_dict(), size)
        size_mb = size.tell() / 1_000_000
        parameters = sum(p.numel() for p in model.parameters())
        # Swin accepts its input as the first positional argument.
        latency_model = model
        benchmark_images = torch.cat(benchmark)
        method, architecture = "reference", REFERENCE
        artifact = f"{REFERENCE}@{REFERENCE_REVISION}"
    else:
        if args.input is None:
            raise ValueError("--input is required unless --reference is used")
        model, metadata = load_model(args.input)
        if metadata["seed"] != args.seed:
            raise ValueError("Evaluation seed must match checkpoint seed")
        top1, top5 = accuracy(model, loader(data, args.batch_size, args.workers), "cpu")
        benchmark_images = torch.stack(
            [data[i][0] for i in range(min(args.latency_images, len(data)))]
        )
        latency_model = model
        parameters = sum(
            p.numel() for p in make_model(metadata["architecture"]).parameters()
        )
        size_mb = args.input.stat().st_size / 1_000_000
        method, architecture = metadata["method"], metadata["architecture"]
        artifact = file_hash(args.input)
    latency = median_latency(latency_model, benchmark_images, args.warmup, args.passes)
    row = {
        "name": args.name,
        "split": args.split,
        "top1": top1,
        "top5": top5,
        "parameters": parameters,
        "size_mb": size_mb,
        "latency_ms": latency,
        "threads": args.threads,
        "device": cpu_description(),
        "seed": args.seed,
        "method": method,
        "architecture": architecture,
        "artifact": artifact,
    }
    write_result(args.results, row)
    print(json.dumps(row, indent=2))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    def data_options(p):
        p.add_argument("--data-dir", type=Path, default=Path("task4_food101/data"))
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--workers", type=int, default=2)

    cache = commands.add_parser("cache")
    data_options(cache)
    cache.add_argument("--output", type=Path, required=True)
    cache.set_defaults(func=cache_logits)

    training = commands.add_parser("train")
    data_options(training)
    training.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--start", type=Path)
    training.add_argument("--cache", type=Path)
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--learning-rate", type=float, default=1e-4)
    training.add_argument("--alpha", type=float, default=0.0)
    training.add_argument("--temperature", type=float, default=4.0)
    training.add_argument("--prune", type=float, default=0.0)
    training.add_argument("--resume", action="store_true")
    training.set_defaults(func=train)

    compression = commands.add_parser("compress")
    data_options(compression)
    compression.add_argument("--input", type=Path, required=True)
    compression.add_argument("--output", type=Path, required=True)
    compression.add_argument(
        "--method", choices=("dynamic", "static", "qat"), required=True
    )
    compression.add_argument("--calibration-samples", type=int, default=1024)
    compression.add_argument("--epochs", type=int, default=2)
    compression.add_argument("--learning-rate", type=float, default=1e-5)
    compression.set_defaults(func=compress)

    evaluation = commands.add_parser("measure")
    data_options(evaluation)
    evaluation.add_argument("--name", required=True)
    evaluation.add_argument("--split", choices=("validation", "test"), required=True)
    evaluation.add_argument("--input", type=Path)
    evaluation.add_argument("--reference", action="store_true")
    evaluation.add_argument(
        "--results", type=Path, default=Path("results/task4_comparison.csv")
    )
    evaluation.add_argument("--threads", type=int, default=1)
    evaluation.add_argument("--latency-images", type=int, default=32)
    evaluation.add_argument("--warmup", type=int, default=10)
    evaluation.add_argument("--passes", type=int, default=5)
    evaluation.set_defaults(func=measure)
    return root


def main():
    args = parser().parse_args()
    for key in (
        "batch_size",
        "epochs",
        "calibration_samples",
        "threads",
        "latency_images",
        "passes",
    ):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.workers < 0 or getattr(args, "warmup", 0) < 0:
        raise ValueError("workers and warmup must be non-negative")
    if hasattr(args, "alpha") and (
        not 0 <= args.alpha <= 1 or args.temperature <= 0 or not 0 <= args.prune < 1
    ):
        raise ValueError("Require alpha in [0,1], temperature > 0 and prune in [0,1)")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.command == "train" and args.prune and not args.start:
        raise ValueError("Pruning requires an already-trained --start checkpoint")
    if args.command == "train" and args.output in (args.start, args.cache):
        raise ValueError(
            "The output must not overwrite the starting model or logit cache"
        )
    if args.command == "compress" and args.output == args.input:
        raise ValueError("The output must not overwrite the input checkpoint")
    if args.command == "measure" and args.reference and args.input is not None:
        raise ValueError("--reference and --input are mutually exclusive")
    args.func(args)


if __name__ == "__main__":
    main()
