"""End-to-end parity check: codec_lm_speaker_encode (Chatterbox) vs
vendored `VoiceEncoder.embeds_from_wavs` + `T3CondEnc.forward`.

Drives the public C API at the same boundary llama.rn will use:
ref-PCM + ref-speech-tokens + emotion → (34, 1024) cond_emb.

The reference path runs in PyTorch using the vendored Chatterbox
package; codec.cpp's CPU mel + LSTM + (graph) cond_enc must match the
HF output bit-for-bit within F16-projection noise (corr ≥ 0.9999).

Notes:
  * We supply the ref_speech_tokens directly (a small fixed array) so
    the test is deterministic — running S3T on the synthetic PCM would
    introduce additional drift.  In practice the caller (Phase B
    `examples/tts.py`) gets these from `conds.pt` or from `codec_encode`.
  * Mel front-end skips librosa.effects.trim(top_db=20) on both sides
    by passing already-trimmed PCM (synthetic signal, no silence
    boundary).
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/e2e"))
sys.path.insert(0, str(REPO / ".model-src/chatterbox/src"))

from _codec_lm_ctypes import (
    lib, codec_audio, CODEC_PCM_TYPE_F32, CodecLM,
)

GGUF = REPO / "models/chatterbox/chatterbox.gguf"

# Chatterbox emotion default + a fixed speech-token sample we feed in.
EMOTION       = 0.5
N_REF_SPEECH  = 16
SPEECH_TOKENS = np.arange(100, 100 + N_REF_SPEECH, dtype=np.int32)


class CodecLMSpeakerInfo(ctypes.Structure):
    _fields_ = [
        ("needs_ref_pcm",           ctypes.c_bool),
        ("needs_ref_speech_tokens", ctypes.c_bool),
        ("needs_emotion_scalar",    ctypes.c_bool),
        ("ref_sample_rate",         ctypes.c_int32),
        ("emotion_default",         ctypes.c_float),
        ("n_rows",                  ctypes.c_int32),
        ("hidden_dim",              ctypes.c_int32),
    ]


def setup_lib_bindings():
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


def synth_ref_pcm(seconds: float = 5.0, sr: int = 16000) -> np.ndarray:
    """Synthetic harmonic-rich signal at 16kHz mono.  Bandlimited so
    librosa STFT energies are well-behaved; non-trivial enough to
    exercise the VE LSTM across multiple partials."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    # Sweep mixture of three sinusoids with amplitude envelope.
    sig = (
        0.5 * np.sin(2 * np.pi * 220 * t * (1.0 + 0.1 * np.sin(2 * np.pi * 0.3 * t))) +
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.2 * np.sin(2 * np.pi * 660 * t)
    )
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    return (sig * env).astype(np.float32)


def reference_cond_emb(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Run the vendored Chatterbox VE + T3CondEnc forward in PyTorch."""
    import torch
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.t3.modules.cond_enc import T3Cond, T3CondEnc
    from chatterbox.models.t3.modules.t3_config import T3Config
    from safetensors.torch import load_file

    ck = REPO / "models/chatterbox"
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(ck / "ve.safetensors"))
    ve.eval()

    cond_enc = T3CondEnc(T3Config())
    t3_sd = load_file(ck / "t3_cfg.safetensors")
    cond_state = {k.replace("cond_enc.", "", 1): v for k, v in t3_sd.items() if k.startswith("cond_enc.")}
    cond_enc.load_state_dict(cond_state)
    # The perceiver needs the same speech_emb + pos table the LM owns.
    speech_emb_w     = t3_sd["speech_emb.weight"]            # (8194, 1024)
    speech_pos_emb_w = t3_sd["speech_pos_emb.emb.weight"]    # (4100, 1024)
    cond_enc.eval()

    with torch.inference_mode():
        ve_embed = torch.from_numpy(
            ve.embeds_from_wavs([pcm], sample_rate=sr, trim_top_db=None)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True)              # (1, 256)

        # Build cond_prompt_speech_emb the way prepare_conditioning does
        # in T3 (speech_emb lookup + speech_pos_emb add).
        tokens = torch.from_numpy(SPEECH_TOKENS).long().unsqueeze(0)  # (1, T)
        speech_emb = torch.nn.functional.embedding(tokens, speech_emb_w)
        pos        = torch.arange(N_REF_SPEECH).unsqueeze(0)
        speech_emb = speech_emb + torch.nn.functional.embedding(pos, speech_pos_emb_w)

        t3_cond = T3Cond(
            speaker_emb              = ve_embed,
            cond_prompt_speech_tokens= tokens.squeeze(0),
            cond_prompt_speech_emb   = speech_emb,                  # pre-computed
            emotion_adv              = EMOTION * torch.ones(1, 1, 1),
        )
        cond_emb = cond_enc(t3_cond)        # (1, 34, 1024)
    return cond_emb.squeeze(0).cpu().float().numpy()


def codec_cpp_cond_emb(lm_ptr, pcm: np.ndarray, sr: int) -> np.ndarray:
    """Run codec_lm_speaker_encode via the public C API."""
    audio = codec_audio()
    pcm_c = pcm.astype(np.float32, copy=False)
    audio.data        = pcm_c.ctypes.data_as(ctypes.c_void_p)
    audio.n_samples   = pcm_c.size
    audio.sample_rate = sr
    audio.n_channels  = 1
    audio.pcm_type    = CODEC_PCM_TYPE_F32

    tokens = np.ascontiguousarray(SPEECH_TOKENS, dtype=np.int32)
    emo    = ctypes.c_float(EMOTION)

    info_ptr = lib.codec_lm_speaker_get_info(lm_ptr)
    if not info_ptr:
        raise RuntimeError("codec_lm has no speaker section")
    info = info_ptr.contents
    n_elems = info.n_rows * info.hidden_dim
    out = (ctypes.c_float * n_elems)()

    rc = lib.codec_lm_speaker_encode(
        lm_ptr, ctypes.byref(audio),
        tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), tokens.size,
        ctypes.byref(emo),
        out, n_elems,
    )
    if rc != 0:
        err = lib.codec_lm_get_last_error(lm_ptr) or b""
        raise RuntimeError(f"codec_lm_speaker_encode rc={rc}, err={err.decode()!r}")
    return np.frombuffer(out, dtype=np.float32).reshape(info.n_rows, info.hidden_dim).copy()


def main() -> int:
    setup_lib_bindings()
    if not GGUF.is_file():
        print(f"FAIL: missing {GGUF}", file=sys.stderr)
        return 2

    pcm = synth_ref_pcm()
    sr  = 16000
    print(f"[ref] running PyTorch VE + T3CondEnc …")
    ref = reference_cond_emb(pcm, sr)
    print(f"      ref shape={ref.shape} dtype={ref.dtype}")

    print(f"[cpp] running codec_lm_speaker_encode …")
    cpp_lm = CodecLM(str(GGUF))
    try:
        got = codec_cpp_cond_emb(cpp_lm.lm, pcm, sr)
    finally:
        cpp_lm.close()
    print(f"      got shape={got.shape}")

    if got.shape != ref.shape:
        print(f"FAIL: shape mismatch {got.shape} vs {ref.shape}", file=sys.stderr)
        return 1

    # Per-row correlation + max-abs-diff.
    all_pass = True
    for r in range(ref.shape[0]):
        a = ref[r]; b = got[r]
        diff = float(np.max(np.abs(a - b)))
        denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        corr = float(np.dot(a, b) / max(denom, 1e-12))
        ok = (diff <= 5e-2) and (corr >= 0.9999)
        all_pass &= ok
        tag = "OK " if ok else "FAIL"
        if r < 4 or not ok:
            print(f"  [{tag}] row={r:2d}  max_abs_diff={diff:.4g}  corr={corr:.6f}")

    if all_pass:
        print("\nChatterbox speaker_encode parity test PASSED")
        return 0
    print("\nChatterbox speaker_encode parity test FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
