"""T3-08: a small network trained from scratch, in UVR's MDX-Net format.

The STFT, the constants and the ONNX export are ported from the course files
ml-tek5-project/2_3_uvr/uvr_my_models/src/spectrograms.py (stft) and
src/model_saving.py (export_onnx, install_in_uvr settings). They are copied
here, not imported, so that this repository does not depend on the course
folder.

UVR contract (separate.py, SeperateMDX): the model receives the spectrogram of
the mixture, [batch, 4, DIM_F, DIM_T] with the channels L-real, L-imag,
R-real, R-imag, and returns the spectrogram of the primary stem (here the
vocals) with the same shape. UVR computes the other stem as mixture - primary.
"""

import json
from pathlib import Path

import torch
from torch import nn

from task3_uvr.onnx_checks import uvr_hash

# Course spectrograms.py. UVR fixes the hop at 1024 for MDX-Net (separate.py).
HOP = 1024
N_FFT = 4096  # 2049 bins, 10.8 Hz apart; consecutive windows overlap by 75%
DIM_F = 2048  # bins given to the model: the top bin (22050 Hz) is dropped
DIM_T = 256  # frames per chunk = the GUI's Segment Size
N_SAMPLES_IN_CHUNK = HOP * (DIM_T - 1)  # 261120 samples, 5.9 s
# UVR zeroes the 3 lowest bins of the model input (separate.py, run_model).
N_ZEROED_BINS = 3

PRIMARY_STEM = "Vocals"
MODEL_NAME = "SmallMaskNet"


def stft(wave: torch.Tensor, n_fft: int = N_FFT, dim_f: int = DIM_F) -> torch.Tensor:
    """[B, 2, samples] -> [B, 4, dim_f, frames], as UVR's STFT (course stft()).

    The window is built on the wave's device, so the STFT runs on the GPU
    during training.
    """
    batch, channels, n_samples = wave.shape
    window = torch.hann_window(n_fft, periodic=True, device=wave.device)
    spec = torch.stft(
        wave.reshape(batch * channels, n_samples),
        n_fft=n_fft,
        hop_length=HOP,
        window=window,
        center=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec).permute(0, 3, 1, 2)  # [B*2, 2, F, T]
    return spec.reshape(batch, 4, spec.shape[2], spec.shape[3])[:, :, :dim_f]


def model_input(wave: torch.Tensor) -> torch.Tensor:
    """Spectrogram as UVR gives it to the model: 3 lowest bins zeroed."""
    spec = stft(wave)
    spec[:, :, :N_ZEROED_BINS] = 0
    return spec


class SmallMaskNet(nn.Module):
    """Predict a soft mask in [0, 1] on the magnitude of each stereo channel.

    1. Magnitude of each channel, compressed with log1p: [B, 2, F, T].
    2. Four convolutions: 3x3 (+ a learned bias per frequency bin), 3x3 with
       dilation 2, 3x3 with dilation 4, then 1x1. ReLU between them, sigmoid
       at the end. The receptive field is 15 bins x 15 frames, about
       160 Hz x 350 ms.
    3. The mask of each channel multiplies its real and imaginary parts, so
       the output keeps the mixture's phase and is at most the mixture.

    Only Conv, Relu, Sigmoid, Mul, Add, Sqrt, Log, Slice and Concat reach the
    ONNX graph, which onnxruntime 1.13.1 (UVR's) runs.
    """

    def __init__(self, channels: int = 16, dim_f: int = DIM_F):
        super().__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        # Convolutions are the same at every frequency; vocals are not.
        self.freq_bias = nn.Parameter(torch.zeros(1, channels, dim_f, 1))
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(channels, 2, 1)
        self.relu = nn.ReLU()

    def mask(self, spec: torch.Tensor) -> torch.Tensor:
        """[B, 4, F, T] spectrogram -> [B, 2, F, T] mask (left, right)."""
        left = spec[:, 0:1] * spec[:, 0:1] + spec[:, 1:2] * spec[:, 1:2]
        right = spec[:, 2:3] * spec[:, 2:3] + spec[:, 3:4] * spec[:, 3:4]
        magnitude = torch.sqrt(torch.cat([left, right], dim=1) + 1e-8)
        x = torch.log1p(magnitude)
        x = self.relu(self.conv1(x) + self.freq_bias)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return torch.sigmoid(self.conv4(x))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        m = self.mask(spec)
        m4 = torch.cat([m[:, 0:1], m[:, 0:1], m[:, 1:2], m[:, 1:2]], dim=1)
        return spec * m4


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_onnx(model: nn.Module, path: Path, dim_f: int = DIM_F) -> None:
    """Course export_onnx: opset 17 (the most onnxruntime 1.13.1 supports),
    the legacy exporter (dynamo=False), a dynamic batch axis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    device = next(model.parameters()).device
    model = model.cpu().eval()
    try:
        torch.onnx.export(
            model,
            (torch.zeros(1, 4, dim_f, DIM_T),),
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    finally:
        model.to(device).train(was_training)


def uvr_settings() -> dict:
    """The settings UVR reads from model_data/<hash>.json (course
    install_in_uvr): compensate 1.0 (no rescaling), dim_f 2048, dim_t 8
    (2**8 = 256 frames), n_fft 4096, primary stem Vocals."""
    return {
        "compensate": 1.0,
        "mdx_dim_f_set": DIM_F,
        "mdx_dim_t_set": DIM_T.bit_length() - 1,
        "mdx_n_fft_scale_set": N_FFT,
        "primary_stem": PRIMARY_STEM,
    }


def write_settings(onnx_path: Path, model_data_dir: Path) -> Path:
    """Write <hash>.json for the model, as install_in_uvr does in UVR's
    models/MDX_Net_Models/model_data/. Copying the ONNX file and this JSON to
    UVR installs the model (T3-09)."""
    model_data_dir.mkdir(parents=True, exist_ok=True)
    path = model_data_dir / f"{uvr_hash(onnx_path)}.json"
    path.write_text(json.dumps(uvr_settings(), indent=4))
    return path
