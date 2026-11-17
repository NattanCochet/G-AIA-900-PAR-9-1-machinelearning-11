# Task 3 · Ultimate Vocal Remover (UVR)

## Installation

1. Clone UVR at tag v5.6 **outside this repository**:

   ```bash
   git clone --branch v5.6 --depth 1 https://github.com/Anjok07/ultimatevocalremovergui.git
   ```

2. Install the system dependencies. On Fedora, check which ffmpeg package is
   installed (`ffmpeg` from RPM Fusion or `ffmpeg-free` from Fedora):

   ```bash
   rpm -q ffmpeg ffmpeg-free
   ```

   Install one of them if neither is present.

3. Deactivate this repository's virtual environment, otherwise `uv` warns
   about `VIRTUAL_ENV`:

   ```bash
   deactivate
   ```

4. Copy the three files from `task3_uvr/uvr_env/` (`pyproject.toml`,
   `uv.lock`, `.python-version`) to the UVR root, then install and launch from
   there:

   ```bash
   cp task3_uvr/uvr_env/{pyproject.toml,uv.lock,.python-version} /path/to/ultimatevocalremovergui/
   cd /path/to/ultimatevocalremovergui
   uv sync
   uv run UVR.py
   ```

5. If UVR crashes at startup, apply the Wayland fix from
   [Known issues](#known-issues).

6. Download **UVR-MDX-NET Inst HQ 3** from the GUI download center.

7. For the benchmark model, copy the course's `project/uvr_my_models` folder
   to the UVR root, then run `1_create_benchmark_model.py`.

## Setup

- UVR v5.6 is cloned **outside this repository** and is not versioned here.
  It has its own `uv` environment on Python 3.10.
- Models go in `models/MDX_Net_Models/` at the UVR root:

  | Model | Source | Notes |
  | --- | --- | --- |
  | `UVR-MDX-NET-Inst_HQ_3.onnx` | GUI download center | Primary stem is **Instrumental** |
  | `Benchmark_CentreChannel.onnx` | Created by the course's `1_create_benchmark_model.py` | Use **Segment Size 256** |

- HQ 3's primary stem is the instrumental: the GUI shows
  "Instrumental Only / Vocals Only". Fine-tuning must therefore target the
  accompaniment, not the vocals.
- UVR names outputs after the input file, so running a second model
  overwrites the first model's stems. Enable
  **Settings > Additional Settings > Model Test Mode** to keep the model name
  in the output file names.

## Known issues

### Crash at startup on Fedora under Wayland

UVR crashes at startup with:

```
[xcb] Extra reply data still left in queue
```

The cause is the bundled `tkinterdnd2` drag-and-drop library. Plain Tk,
pyglet fonts and `screeninfo` work.

Fix: in `UVR.py`, line 83, set

```python
is_dnd_compatible = False
```

Drag and drop is then disabled; choose files with the **Select Input** and
**Select Output** buttons.

## Verification

Run on 2026-11-12, CPU only (Intel Core i7-1165G7, no CUDA), on the
7-second musdb7 test clip "Al James - Schoolboy Facination":

| Model | Time | Output stems |
| --- | --- | --- |
| UVR-MDX-NET Inst HQ 3 | 10 s | Vocals, Instrumental |
| Benchmark_CentreChannel (Segment Size 256) | < 1 s | Vocals, Instrumental |

## HQ 3 in PyTorch (T3-05)

Checks that HQ 3 can be converted with onnx2pytorch, trained for one step and
re-exported to a file UVR's onnxruntime 1.13.1 loads. The original model is
only read; its MD5 is checked before and after. Run from the repository root
(`<UVR>` is the UVR clone):

```bash
M=<UVR>/models/MDX_Net_Models/UVR-MDX-NET-Inst_HQ_3.onnx

# Steps 1-6, this repository's environment
uv run python -m task3_uvr.hq3_trainable check --model "$M"

# Load the re-exported model in UVR's environment (onnxruntime 1.13.1)
uv run --project <UVR> --no-sync python -m task3_uvr.check_onnxruntime_113 --model "$M"

# Copy it to UVR as HQ3_T3-05_step1.onnx and write its settings JSON
uv run python -m task3_uvr.hq3_trainable install --model "$M"
```

The re-exported model and its input/output go to `task3_uvr/artifacts/`
(not versioned). Results and exact library versions are in
`results/task3_hq3_trainable.json`.

### Results

Run on 2026-11-12, CPU only (Intel Core i7-1165G7, 4 torch threads), torch
2.15.0, onnx 1.23.2, onnx2pytorch 0.6.0, onnxruntime 1.31.1 (and 1.13.1 for
the UVR check). The input is one random-noise spectrogram of shape
`[1, 4, 3072, 256]`, not real audio. "Identical" means the max absolute
difference is below 1e-4 times the largest absolute output (9.12).

| Step | Result |
| --- | --- |
| 1. Graph | Opset 13, IR 6, exported by PyTorch 1.9. 16,684,228 values in 220 initializers, of which 8,352 are BatchNorm running statistics. Operators: Relu 66, Conv 40, BatchNormalization 27, MatMul 22, Add 11, ConvTranspose 5, Mul 5, Transpose 2 |
| 2. Conversion | No error. 166 trainable tensors, 16,675,876 values: every weight except the BatchNorm statistics |
| 3. Parity, onnxruntime vs PyTorch | Max abs diff 8.0e-5 (relative 8.8e-6): identical |
| 4. Gradient step (L1 loss, Adam, lr 1e-5) | All 166 tensors get a non-zero gradient; all 166 change after the step |
| 5. Re-export (opset 17, IR 8) | 166 of 220 initializers differ from the original (the 54 BatchNorm statistics do not). PyTorch vs onnxruntime 1.31: max abs diff 7.0e-5 (relative 7.7e-6). onnxruntime 1.13.1 loads it: max abs diff 7.7e-5 (relative 8.5e-6) |
| 6. UVR settings | Hash `55657dd70583b0fedfba5f67df11d711`: compensate 1.022, dim_f 3072, dim_t 8 (256 frames), n_fft 6144, primary stem Instrumental |

The whole check takes about 40 s, with a peak memory of 6.7 GB (the backward
pass on one full-size spectrogram takes 13 s). The re-exported model is
installed in UVR as `HQ3_T3-05_step1.onnx`, hash
`fac837f566082189e25ab946a13e6d27`, with HQ 3's settings.

### Notes for fine-tuning (T3-10)

- **Keep BatchNorm in eval mode.** In train mode the output on the same input
  changes by 2.6 times the largest output value, because BatchNorm uses the
  batch's statistics instead of HQ 3's running averages.
- **Ignore the `_initializer_*` buffers.** onnx2pytorch keeps a copy of every
  original initializer as a buffer (16.7 M values). The forward pass uses the
  parameters; the export check confirms the updated weights are exported.
- **Export with opset 17 and the legacy exporter** (`dynamo=False`).
  onnxruntime 1.13.1 supports at most opset 17 and IR version 8. The legacy
  exporter is deprecated since PyTorch 2.9, so check it still exists in the
  Colab version, and rerun `check_onnxruntime_113` on every exported model.
- **One step moves the output a lot.** On the noise input, one Adam step at
  lr 1e-5 changed the output by up to 35% of its largest value. Use a small
  learning rate.
- **Give every fine-tuned model a new file name.** Its hash changes, so it
  needs its own `model_data/<hash>.json` with HQ 3's settings, and UVR
  caches hashes by file path until it restarts.
