"""CAMPPlus (3D-Speaker) speaker-encoder export with BatchNorm folded.

Emits tensors consumed by src/models/campplus.cpp under `<prefix>.`:

    head.conv1.{w,b}, head.l{1,2}.{0,1}.conv{1,2}.{w,b}, head.l{1,2}.0.sc.{w,b}, head.conv2.{w,b}
        conv + following BN folded into weight/bias
    tdnn.{w,b}                         conv + BN folded (ReLU follows)
    b{1..3}.{i}.bn.{s,t}               pre-activation BN as scale/shift
    b{1..3}.{i}.lin1.{w,b}             1x1 conv + following BN folded
    b{1..3}.{i}.local.w                CAM local conv (k=3, dilated)
    b{1..3}.{i}.cl{1,2}.{w,b}          CAM context 1x1 convs
    tr{1..3}.bn.{s,t}, tr{1..3}.w      transition BN + 1x1 conv
    out_bn.{s,t}                       final BN before stats pooling
    dense.{w,b}                        1x1 conv + non-affine BN folded
    fbank_mel                          Kaldi mel banks [80, 257] (16 kHz, FFT 512)
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

BN_EPS = 1e-5
BLOCK_LAYERS = (12, 24, 16)


def _bn(state: Mapping[str, np.ndarray], p: str, affine: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Inference BatchNorm as y = x * s + t."""
    mean = state[p + ".running_mean"].astype(np.float64)
    var = state[p + ".running_var"].astype(np.float64)
    s = 1.0 / np.sqrt(var + BN_EPS)
    t = -mean * s
    if affine:
        g = state[p + ".weight"].astype(np.float64)
        b = state[p + ".bias"].astype(np.float64)
        s, t = s * g, t * g + b
    return s, t


def _fold(w: np.ndarray, s: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """conv(x) (no bias) followed by BN → conv with scaled weight and bias."""
    w = w.astype(np.float64) * s.reshape((-1,) + (1,) * (w.ndim - 1))
    return w.astype(np.float32), t.astype(np.float32)


def kaldi_fbank_mel(n_mels: int = 80) -> np.ndarray:
    """torchaudio Kaldi mel banks for 16 kHz / FFT 512, padded with the zero Nyquist column."""
    import torchaudio.compliance.kaldi as kaldi
    banks, _ = kaldi.get_mel_banks(n_mels, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    banks = banks.numpy().astype(np.float32)
    return np.pad(banks, ((0, 0), (0, 1)))


def campplus_tensors(state: Mapping[str, np.ndarray], src: str) -> list[tuple[str, np.ndarray]]:
    """Folded CAMPPlus tensors from a state dict whose keys start with `src` (e.g. "speaker_encoder.")."""
    S = {k[len(src):]: np.asarray(v) for k, v in state.items() if k.startswith(src)}
    out: list[tuple[str, np.ndarray]] = []

    def conv_bn(dst: str, conv: str, bn: str, affine: bool = True) -> None:
        w, b = _fold(S[conv + ".weight"], *_bn(S, bn, affine))
        out.append((dst + ".w", w))
        out.append((dst + ".b", b))

    def bn_st(dst: str, bn: str) -> None:
        s, t = _bn(S, bn)
        out.append((dst + ".s", s.astype(np.float32)))
        out.append((dst + ".t", t.astype(np.float32)))

    conv_bn("head.conv1", "head.conv1", "head.bn1")
    for li in (1, 2):
        for bi in (0, 1):
            p = f"head.layer{li}.{bi}"
            d = f"head.l{li}.{bi}"
            conv_bn(d + ".conv1", p + ".conv1", p + ".bn1")
            conv_bn(d + ".conv2", p + ".conv2", p + ".bn2")
            if p + ".shortcut.0.weight" in S:
                conv_bn(d + ".sc", p + ".shortcut.0", p + ".shortcut.1")
    conv_bn("head.conv2", "head.conv2", "head.bn2")

    conv_bn("tdnn", "xvector.tdnn.linear", "xvector.tdnn.nonlinear.batchnorm")
    for bi, n_layers in enumerate(BLOCK_LAYERS, start=1):
        for li in range(n_layers):
            p = f"xvector.block{bi}.tdnnd{li + 1}"
            d = f"b{bi}.{li}"
            bn_st(d + ".bn", p + ".nonlinear1.batchnorm")
            conv_bn(d + ".lin1", p + ".linear1", p + ".nonlinear2.batchnorm")
            out.append((d + ".local.w", S[p + ".cam_layer.linear_local.weight"].astype(np.float32)))
            for c in (1, 2):
                out.append((d + f".cl{c}.w", S[p + f".cam_layer.linear{c}.weight"].astype(np.float32)))
                out.append((d + f".cl{c}.b", S[p + f".cam_layer.linear{c}.bias"].astype(np.float32)))
        p = f"xvector.transit{bi}"
        bn_st(f"tr{bi}.bn", p + ".nonlinear.batchnorm")
        out.append((f"tr{bi}.w", S[p + ".linear.weight"].astype(np.float32)))
    bn_st("out_bn", "xvector.out_nonlinear.batchnorm")
    conv_bn("dense", "xvector.dense.linear", "xvector.dense.nonlinear.batchnorm", affine=False)
    out.append(("fbank_mel", kaldi_fbank_mel()))
    return out


def write_campplus(add: Callable[[str, np.ndarray], None], state: Mapping[str, np.ndarray],
                   src: str, prefix: str) -> None:
    for name, arr in campplus_tensors(state, src):
        add(f"{prefix}.{name}", arr)
