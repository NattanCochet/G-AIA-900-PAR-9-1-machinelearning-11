# ML Tek5 project

Six independent ML tasks (knowledge questions, MNIST, UVR vocal removal,
Food-101 compression, Covertype, Fashion-MNIST clustering).

## Installation

```bash
uv sync
```

This installs every dependency needed across all six tasks (see
`pyproject.toml`) into a single `.venv`, including the CPU-only build of
PyTorch (this machine has no dedicated GPU; tasks 3 and 4 that need a GPU run
remotely on Colab/Kaggle instead).

Run any script or notebook with:

```bash
uv run python -m task2_mnist.baseline   # -m, so `import common` resolves
uv run jupyter lab
```

## Layout

```
common/              # seeds, timer, CSV logger, shared figure style, checkpointing
task1_general/
task2_mnist/
task3_uvr/            # copy of uvr_my_models + training/fine-tuning/eval scripts
task4_food101/
task5_covertype/
task6_fashion/
results/              # CSV results and final figures
slides/
```

Data, trained models and audio files are not versioned (see `.gitignore`);
each task's scripts re-download/regenerate what they need.

## Remote GPU (tasks 3 and 4)

Neither task 2/5/6's laptop nor this sandbox has a GPU PyTorch/TensorFlow can
use (no CUDA; the AMD GPU isn't on ROCm's supported hardware list), so UVR
(task 3) and Food-101 (task 4) training runs on **Google Colab** (Google One
student plan: more compute units / better GPU priority than the free tier).

1. Open [common/colab_remote_gpu_setup.ipynb](common/colab_remote_gpu_setup.ipynb)
   in Colab, with **Runtime > Change runtime type > GPU**.
2. Run it top to bottom: it reports the assigned GPU, mounts Drive (for
   persistent checkpoints — Colab's local disk is wiped between sessions),
   clones this repo, installs the task 3/4 extras on top of Colab's
   preinstalled CUDA-matched PyTorch, and proves a training run can be
   interrupted and resumed without losing progress (`common/checkpointing.py`).
3. In your own task 3/4 training scripts, call
   `common.checkpointing.save_checkpoint`/`load_checkpoint` every epoch, with
   the checkpoint path under the mounted Drive folder, and re-run the
   notebook's setup cells after any disconnect to resume where you left off.

### Task 4: Food-101 reference model and distillation

From the repository root, fine-tune the ResNet-50 reference model first, then
distill it into the smaller MobileNetV3 model:

```bash
uv run python -m task4_food101.train --mode teacher \
  --checkpoint /content/drive/MyDrive/ml_tek5_checkpoints/food101_teacher.pt
uv run python -m task4_food101.train --mode student \
  --teacher-checkpoint /content/drive/MyDrive/ml_tek5_checkpoints/food101_teacher.pt \
  --checkpoint /content/drive/MyDrive/ml_tek5_checkpoints/food101_student.pt
```

Both runs use a seeded 90/10 train/validation split, save a resumable
checkpoint after every epoch, and evaluate once on the held-out Food-101 test
split. Resume an interrupted run by adding `--resume` to the same command.
Accuracy, cumulative training time and parameter count are recorded in
`results/task4_food101.csv`.
