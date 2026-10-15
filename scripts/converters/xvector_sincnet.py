"""pyannote XVectorSincNet export for src/models/xvector_sincnet.cpp.

Emits under `<prefix>.`:

    wav_norm.{w,b}            InstanceNorm1d(1) on the waveform
    conv0.w                   sinc band-pass filters [80, 1, 251], precomputed
    conv{1,2}.{w,b}           SincNet convs
    norm{0,1,2}.{w,b}         SincNet InstanceNorms
    tdnn{0..4}.{w,b}          TDNN convs
    tdnn{0..4}.bn.{s,t}       BatchNorm after LeakyReLU as scale/shift
    emb.{w,b}                 Linear(3000, 512)
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

BN_EPS = 1e-5
SAMPLE_RATE = 16000


def sinc_filters(low_hz_: np.ndarray, band_hz_: np.ndarray, kernel_size: int = 251,
                 min_low_hz: float = 50.0, min_band_hz: float = 50.0) -> np.ndarray:
    """ParamSincFB taps (asteroid / pyannote): cos filters then sin filters, [n, 1, k]."""
    import torch
    low_hz_ = torch.as_tensor(np.asarray(low_hz_, np.float32))
    band_hz_ = torch.as_tensor(np.asarray(band_hz_, np.float32))
    half = kernel_size // 2
    window = torch.from_numpy(np.hamming(kernel_size)[:half]).float()
    n_ = 2 * np.pi * (torch.arange(-half, 0.0).view(1, -1) / SAMPLE_RATE)
    low = min_low_hz + torch.abs(low_hz_)
    high = torch.clamp(low + min_band_hz + torch.abs(band_hz_), min_low_hz, SAMPLE_RATE / 2)

    def band_pass(kind: str) -> torch.Tensor:
        band = (high - low)[:, 0]
        ft_low, ft_high = low @ n_, high @ n_
        if kind == "cos":
            left = ((torch.sin(ft_high) - torch.sin(ft_low)) / (n_ / 2)) * window
            center = 2 * band.view(-1, 1)
            right = torch.flip(left, dims=[1])
        else:
            left = ((torch.cos(ft_low) - torch.cos(ft_high)) / (n_ / 2)) * window
            center = torch.zeros_like(band.view(-1, 1))
            right = -torch.flip(left, dims=[1])
        bp = torch.cat([left, center, right], dim=1) / (2 * band[:, None])
        return bp.view(-1, 1, kernel_size)

    return torch.cat([band_pass("cos"), band_pass("sin")], dim=0).numpy().astype(np.float32)


def xvector_tensors(state: Mapping[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    S = {k: np.asarray(v, dtype=np.float32) for k, v in state.items()
         if not k.startswith("loss_func.") and not k.endswith(("num_batches_tracked", "window_", "n_"))}
    out: list[tuple[str, np.ndarray]] = [
        ("wav_norm.w", S["sincnet.wav_norm1d.weight"]),
        ("wav_norm.b", S["sincnet.wav_norm1d.bias"]),
        ("conv0.w", sinc_filters(S["sincnet.conv1d.0.filterbank.low_hz_"], S["sincnet.conv1d.0.filterbank.band_hz_"])),
    ]
    for i in (1, 2):
        out += [(f"conv{i}.w", S[f"sincnet.conv1d.{i}.weight"]), (f"conv{i}.b", S[f"sincnet.conv1d.{i}.bias"])]
    for i in range(3):
        out += [(f"norm{i}.w", S[f"sincnet.norm1d.{i}.weight"]), (f"norm{i}.b", S[f"sincnet.norm1d.{i}.bias"])]
    for i in range(5):
        conv, bn = f"tdnns.{3 * i}", f"tdnns.{3 * i + 2}"
        s = S[bn + ".weight"].astype(np.float64) / np.sqrt(S[bn + ".running_var"].astype(np.float64) + BN_EPS)
        t = S[bn + ".bias"] - S[bn + ".running_mean"] * s
        out += [(f"tdnn{i}.w", S[conv + ".weight"]), (f"tdnn{i}.b", S[conv + ".bias"]),
                (f"tdnn{i}.bn.s", s.astype(np.float32)), (f"tdnn{i}.bn.t", t.astype(np.float32))]
    out += [("emb.w", S["embedding.weight"]), ("emb.b", S["embedding.bias"])]
    return out


def write_xvector(add: Callable[[str, np.ndarray], None], state: Mapping[str, np.ndarray], prefix: str) -> None:
    for name, arr in xvector_tensors(state):
        add(f"{prefix}.{name}", arr)
