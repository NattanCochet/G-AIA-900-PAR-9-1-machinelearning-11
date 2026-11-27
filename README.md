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
# From the repository root
uv run python -m task2_mnist.baseline
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

The laptop used for tasks 2, 5 and 6 has no GPU that PyTorch/TensorFlow can
use (no CUDA; the AMD GPU isn't on ROCm's supported hardware list), so UVR
(task 3) and Food-101 (task 4) training runs on **Google Colab** (Google One
student plan: more compute units / better GPU priority than the free tier).

1. The repo is private, so the notebook needs a token to clone it. Create a
   GitHub fine-grained personal access token limited to this repository only,
   with read-only **Contents** access, and store it in Colab's **Secrets**
   panel (key icon in the left sidebar) as `GITHUB_TOKEN`, with notebook
   access enabled. Never paste the token into the notebook itself.
2. Open [common/colab_remote_gpu_setup.ipynb](common/colab_remote_gpu_setup.ipynb)
   in Colab, with **Runtime > Change runtime type > GPU**.
3. Run it top to bottom: it reports the assigned GPU, mounts Drive (for
   persistent checkpoints — Colab's local disk is wiped between sessions),
   clones this repo, installs the task 3/4 extras on top of Colab's
   preinstalled CUDA-matched PyTorch, and proves a training run can be
   interrupted and resumed without losing progress (`common/checkpointing.py`).
4. In your own task 3/4 training scripts, call
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

### Task 4: compression comparison

The original runs above are retained as preliminary results, not the Swin
comparison. The separate `task4_food101.experiment` pipeline uses the public
Swin Food-101 model as teacher, maps its output labels to torchvision's class
order, and trains ImageNet-initialized MobileNetV3-large, EfficientNet-B0 and
ResNet-18 students. The Swin revision is pinned in the code.

**Predeclared selection rule:** the selected student must achieve validation
top-1 within 2 percentage points of the independently measured Swin test
top-1; among qualifying students, choose the smallest serialized model. Report
whether that selected model also passes the threshold on test. Tune on the
seeded 90/10 split of the original training set, and only run the test
measurements after the experiment configurations are frozen. The Swin model
card does not specify its training/evaluation splits: its reported score may
have been selected on the official test set, and it may already have seen our
student validation images. Consequently neither the teacher's official test
score nor our student validation split is guaranteed independent of teacher
training/selection; disclose this limitation.

Run training and caching on a GPU host using the setup above; keep `RUNS` on
Drive or another persistent disk. Run **all measurements on the same CPU**,
with the same software, thread count and timing settings. The commands below
use one thread, batch-one model inference, ten warmups and the median of five
passes over 32 images; image preprocessing and loading are excluded from
latency. `size_mb` is decimal MB of the serialized model state/checkpoint,
without optimizer state (the Swin measurement serializes its state dict in
memory). Pruned weights are saved densely, so zeros do not imply smaller
files or faster inference. Parameter count is the original architecture's
parameter count, including zeroed or quantized parameters.

For a guided Colab sweep, run sections 1–3 and then section 7 of
`common/colab_remote_gpu_setup.ipynb`. First push or copy the new task 4 files
into the Colab checkout; the notebook's Git clone cannot see uncommitted local
changes. Section 6 and its existing Drive checkpoints are from the preliminary
ResNet-50 experiment and are not compatible with this Swin-based pipeline.

```bash
# Start in the repository root. Download Food-101 beforehand if not present:
python -c "from torchvision.datasets import Food101; Food101('task4_food101/data', split='train', download=True)"
RUNS=/content/drive/MyDrive/ml_tek5_checkpoints/food101_compression
mkdir -p "$RUNS"

# Teacher logits are cached only for student-training images.
python -m task4_food101.experiment cache --output "$RUNS/swin_logits.pt"

# Hard-label baselines, plus a temperature/alpha sweep for each student.
for model in mobilenet_v3_large efficientnet_b0 resnet18; do
  python -m task4_food101.experiment train --architecture "$model" \
    --output "$RUNS/${model}_baseline.pt"
  for temperature in 2 4 8; do
    for alpha in 0.25 0.5 0.75; do
      python -m task4_food101.experiment train --architecture "$model" \
        --cache "$RUNS/swin_logits.pt" --temperature "$temperature" --alpha "$alpha" \
        --output "$RUNS/${model}_t${temperature}_a${alpha}.pt"
    done
  done
done
```

Train runs save the best validation checkpoint and a separate
`<output>.resume.pt` after each epoch. To resume, rerun the identical command
with `--resume`; `--epochs` is the total desired epoch count. The cached logits
use the teacher's processor on each original image; the student receives a
deterministic resize/center crop. This deliberately avoids random views that
would no longer correspond to cached predictions.

Choose promising configurations using validation only, then, for example:

```bash
# Replace with the validation-selected ResNet-18 distillation run.
BEST="$RUNS/resnet18_t4_a0.5.pt"

# Magnitude pruning + fine-tuning, with multiple sparsity rates.
for amount in 0.3 0.5 0.7; do
  python -m task4_food101.experiment train --architecture resnet18 \
    --start "$BEST" --prune "$amount" --output "$RUNS/resnet18_pruned_${amount}.pt"
done

# ResNet-18 dynamic, calibrated static and QAT, including distillation + quantization.
for method in dynamic static qat; do
  python -m task4_food101.experiment compress --input "$BEST" \
    --method "$method" --output "$RUNS/resnet18_distilled_${method}.pt"
  python -m task4_food101.experiment compress --input "$RUNS/resnet18_baseline.pt" \
    --method "$method" --output "$RUNS/resnet18_baseline_${method}.pt"
done
```

Static calibration uses up to 1,024 training images; QAT fine-tunes the
selected float checkpoint using hard labels and selects by validation.
Static/QAT currently support the quantizable torchvision ResNet-18 on x86;
dynamic quantization affects linear layers only and can also be applied to
the other students. QAT runs are not resumable. PyTorch's eager quantization
APIs currently emit deprecation warnings.

Measure each frozen candidate on validation first, and then on test after
freezing the study. Use a unique name per artifact; the tool records a hash
and refuses mismatched validation/test artifacts when plotting. Example:

```bash
python -m task4_food101.experiment measure --reference --name swin --split test
python -m task4_food101.experiment measure --input "$BEST" --name resnet18_distilled --split validation
python -m task4_food101.experiment measure --input "$BEST" --name resnet18_distilled --split test
# Repeat both measurements for each frozen baseline, sweep and compression artifact.
python -m task4_food101.report
```

The comparison CSV contains top-1, top-5, parameter count, size and CPU
latency; `report` produces accuracy-versus-size (log scale) and
accuracy-versus-latency plots with Pareto frontiers. `task4_food101.csv`
contains the older runs and cannot supply these missing measurements.
