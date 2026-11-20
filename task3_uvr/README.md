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
