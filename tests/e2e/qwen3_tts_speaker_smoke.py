"""End-to-end parity check: codec_lm_speaker_encode (Qwen3-TTS ECAPA-TDNN)
vs vendored `Qwen3TTSSpeakerEncoder.forward(mel_spectrogram(pcm))`.

Drives the public C API at the same boundary llama.rn / llama.cpp will
use: ref-PCM (24 kHz mono) → (1, 1024) x-vector.

The reference path runs in PyTorch using the vendored Qwen3-TTS
package; codec.cpp's CPU ECAPA-TDNN must match end-to-end within
F16-projection noise (corr ≥ 0.9999).
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/e2e"))
sys.path.insert(0, str(REPO / ".model-src/Qwen3-TTS"))

from _codec_lm_ctypes import lib, codec_audio, CODEC_PCM_TYPE_F32, CodecLM

GGUF = REPO / "models/qwen3_tts/qwen3_tts_06b_base.gguf"


class CodecLMSpeakerInfo(ctypes.Structure):
    _fields_ = [
        ("needs_ref_pcm",           ctypes.c_bool),
        ("needs_ref_speech_tokens", ctypes.c_bool),
        ("needs_emotion_scalar",    ctypes.c_bool),
        ("ref_sample_rate",         ctypes.c_int32),
        ("emotion_default",         ctypes.c_float),
        ("n_rows",                  ctypes.c_int32),
        ("hidden_dim",              ctypes.c_int32),
        ("speaker_emb_dim",         ctypes.c_int32),
    ]


def setup_bindings():
    lib.codec_lm_speaker_get_info.argtypes = [ctypes.c_void_p]
    lib.codec_lm_speaker_get_info.restype  = ctypes.POINTER(CodecLMSpeakerInfo)
    lib.codec_lm_speaker_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(codec_audio),
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    lib.codec_lm_speaker_encode.restype = ctypes.c_int
    lib.codec_lm_get_last_error.argtypes = [ctypes.c_void_p]
    lib.codec_lm_get_last_error.restype = ctypes.c_char_p


def synth_ref_pcm(seconds=3.0, sr=24000):
    """Synthetic 3-second harmonic-rich signal."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    sig = (
        0.4 * np.sin(2 * np.pi * 200 * t) +
        0.3 * np.sin(2 * np.pi * 400 * t) +
        0.2 * np.sin(2 * np.pi * 800 * t) +
        0.1 * np.sin(2 * np.pi * 1600 * t * (1 + 0.05 * np.sin(2 * np.pi * 1.5 * t)))
    )
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t)
    return (sig * env).astype(np.float32)


def reference_xvector(pcm, sr):
    """Run HF Qwen3-TTS speaker_encoder on the same PCM."""
    import torch
    from huggingface_hub import hf_hub_download
    import safetensors
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSSpeakerEncoder, mel_spectrogram,
    )
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig

    enc = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig())
    # Load the speaker_encoder weights from the HF checkpoint.
    p = hf_hub_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "model.safetensors")
    sd = {}
    with safetensors.safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k.startswith("speaker_encoder."):
                sd[k.replace("speaker_encoder.", "", 1)] = f.get_tensor(k).float()
    enc.load_state_dict(sd)
    enc.eval()

    with torch.inference_mode():
        wav = torch.from_numpy(pcm).unsqueeze(0)
        # `extract_speaker_embedding` in upstream does `.transpose(1, 2)`
        # before handing to `speaker_encoder`, then the encoder's forward
        # does another `.transpose(1, 2)` — net-net the conv sees the
        # original `(B, n_mels, T)` layout.  Mirror that round-trip so
        # weights / biases pair with the right axes.
        mel = mel_spectrogram(
            wav, n_fft=1024, num_mels=128,
            sampling_rate=sr, hop_size=256, win_size=1024,
            fmin=0, fmax=12000, center=False,
        ).transpose(1, 2)
        xvec = enc(mel)[0]   # (1024,)
    return xvec.cpu().float().numpy()


def codec_cpp_xvector(lm_ptr, pcm, sr):
    audio = codec_audio()
    p = pcm.astype(np.float32, copy=False)
    audio.data        = p.ctypes.data_as(ctypes.c_void_p)
    audio.n_samples   = p.size
    audio.sample_rate = sr
    audio.n_channels  = 1
    audio.pcm_type    = CODEC_PCM_TYPE_F32

    info = lib.codec_lm_speaker_get_info(lm_ptr).contents
    out = (ctypes.c_float * (info.n_rows * info.hidden_dim))()
    rc = lib.codec_lm_speaker_encode(
        lm_ptr, ctypes.byref(audio), None, 0, None,
        out, info.n_rows * info.hidden_dim,
    )
    if rc != 0:
        err = lib.codec_lm_get_last_error(lm_ptr) or b""
        raise RuntimeError(f"speaker_encode rc={rc}: {err.decode()!r}")
    return np.frombuffer(out, dtype=np.float32).reshape(info.n_rows, info.hidden_dim).copy()


def main() -> int:
    setup_bindings()
    if not GGUF.is_file():
        print(f"FAIL: missing {GGUF}", file=sys.stderr); return 2

    pcm = synth_ref_pcm()
    sr  = 24000
    print(f"[ref] running HF Qwen3TTSSpeakerEncoder …")
    ref = reference_xvector(pcm, sr)
    print(f"      ref shape={ref.shape}")

    print(f"[cpp] running codec_lm_speaker_encode …")
    cpp_lm = CodecLM(str(GGUF))
    try:
        got = codec_cpp_xvector(cpp_lm.lm, pcm, sr)
    finally:
        cpp_lm.close()
    print(f"      got shape={got.shape}")

    row_ref = ref
    row_got = got[0]
    mad = float(np.max(np.abs(row_ref - row_got)))
    denom = float(np.linalg.norm(row_ref)) * float(np.linalg.norm(row_got))
    corr = float(np.dot(row_ref, row_got) / max(denom, 1e-12))
    ok = (mad <= 5e-2) and (corr >= 0.9999)
    tag = "OK " if ok else "FAIL"
    print(f"  [{tag}]  max_abs_diff={mad:.4g}  corr={corr:.6f}")

    if ok:
        print("\nQwen3-TTS speaker_encode parity test PASSED")
        return 0
    print("\nQwen3-TTS speaker_encode parity test FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
