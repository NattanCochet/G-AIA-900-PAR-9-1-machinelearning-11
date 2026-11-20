"""Task 2 figure: CPU and GPU timelines of a few training steps, read from the
traces written by `time_to_target --profile`, to show how long the GPU waits.

    uv run python -m task2_mnist.plot_profile
    uv run python -m task2_mnist.plot_profile --batch-sizes 32 512 --steps 3

Reads task2_mnist/traces/trace_train_bs{N}_on_device_cuda.json (run
`time_to_target --data-on-device --batch-size N --profile` first) and writes
results/task2_mnist_profile_timeline.png.

The traces are recorded under the profiler, which slows the CPU side down: the
gaps are wider than in a normal run. The figure shows the mechanism; the
occupancy to report is the one printed by --profile (GPU time divided by the
step time measured without the profiler).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from common import set_style
from task2_mnist.baseline import FIGURES_DIR
from task2_mnist.time_to_target import TRACES_DIR

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
OUTPUT = Path(FIGURES_DIR) / "task2_mnist_profile_timeline.png"
CPU_COLOR = "tab:blue"
GPU_COLOR = "tab:orange"

Interval = tuple[float, float]  # (start, end) in microseconds


def load_events(path: Path) -> list[dict]:
    """Complete events ("X") of a Chrome trace: they have a start and a length."""
    with path.open(encoding="utf-8") as f:
        return [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]


def merge(intervals: list[Interval]) -> list[Interval]:
    """Union of possibly overlapping intervals (nested ops, parallel streams)."""
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def clip(intervals: list[Interval], low: float, high: float) -> list[Interval]:
    return [(max(s, low), min(e, high)) for s, e in intervals if e > low and s < high]


def step_starts(events: list[dict], n_steps: int) -> list[float]:
    """Start times of `n_steps` consecutive recorded steps, taken in the middle
    of the trace (the first recorded step carries the profiler's start-up), and
    the end of the last one."""
    steps = sorted(
        (e for e in events if e.get("cat") == "user_annotation"
         and e["name"].startswith("ProfilerStep#")),
        key=lambda e: e["ts"],
    )
    first = (len(steps) - n_steps) // 2
    chosen = steps[first : first + n_steps]
    return [e["ts"] for e in chosen] + [chosen[-1]["ts"] + chosen[-1]["dur"]]


def timelines(path: Path, n_steps: int) -> dict:
    """CPU and GPU busy intervals (ms, from the start of the window) over
    `n_steps` training steps, and the share of the window each one is busy."""
    events = load_events(path)
    boundaries = step_starts(events, n_steps)
    low, high = boundaries[0], boundaries[-1]

    cpu = merge([(e["ts"], e["ts"] + e["dur"]) for e in events
                 if e.get("cat") == "cpu_op"])
    gpu = merge([(e["ts"], e["ts"] + e["dur"]) for e in events
                 if e.get("cat") in GPU_CATEGORIES])
    cpu, gpu = clip(cpu, low, high), clip(gpu, low, high)

    def to_ms(intervals: list[Interval]) -> list[Interval]:
        return [((s - low) / 1000, (e - s) / 1000) for s, e in intervals]

    window = high - low
    return {
        "cpu": to_ms(cpu),
        "gpu": to_ms(gpu),
        "steps_ms": [(b - low) / 1000 for b in boundaries],
        "window_ms": window / 1000,
        "cpu_share": sum(e - s for s, e in cpu) / window,
        "gpu_share": sum(e - s for s, e in gpu) / window,
    }


def plot(batch_sizes: list[int], n_steps: int, output: Path) -> None:
    set_style()
    data = {
        bs: timelines(TRACES_DIR / f"trace_train_bs{bs}_on_device_cuda.json", n_steps)
        for bs in batch_sizes
    }
    fig, axes = plt.subplots(
        len(batch_sizes), 1, sharex=True, squeeze=False,
        figsize=(10, 1.9 * len(batch_sizes) + 1.2),
    )
    for ax, (bs, d) in zip(axes[:, 0], data.items(), strict=True):
        ax.broken_barh(d["cpu"], (0.65, 0.7), color=CPU_COLOR)
        ax.broken_barh(d["gpu"], (-0.35, 0.7), color=GPU_COLOR)
        for i, x in enumerate(d["steps_ms"]):
            ax.axvline(x, color="grey", linestyle="--", linewidth=0.8)
            if i < n_steps:
                ax.text(x + 0.05, 1.45, f"pas {i + 1}", fontsize=8, color="grey")
        ax.set_yticks([0, 1], ["GPU (kernels)", "CPU (opérations\nPyTorch)"])
        ax.set_ylim(-0.6, 1.75)
        ax.set_ylabel("exécution")
        ax.grid(axis="y", visible=False)
        ax.set_title(
            f"batch {bs} : GPU occupé {d['gpu_share']:.0%} de la fenêtre, "
            f"CPU {d['cpu_share']:.0%}",
            loc="left",
        )
    axes[-1, 0].set_xlabel("temps depuis le début de la fenêtre (ms)")
    axes[-1, 0].set_xlim(0, max(d["window_ms"] for d in data.values()))
    fig.suptitle(
        f"{n_steps} pas d'entraînement enregistrés par torch.profiler "
        "(RTX 3080, données sur le GPU)"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"figure written to {output}")
    for bs, d in data.items():
        print(
            f"  batch {bs}: window {d['window_ms']:.1f} ms, "
            f"GPU busy {d['gpu_share']:.0%}, CPU busy {d['cpu_share']:.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 512])
    parser.add_argument("--steps", type=int, default=3, help="steps shown")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    plot(args.batch_sizes, args.steps, args.output)


if __name__ == "__main__":
    main()
