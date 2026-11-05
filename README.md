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
uv run python task2_mnist/some_script.py
uv run jupyter lab
```

## Layout

```
common/              # seeds, timer, CSV logger, shared figure style
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
