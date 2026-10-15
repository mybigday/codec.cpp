#!/usr/bin/env python3
"""End-to-end smoke test for the CosyVoice codec (flow + HiFT) through the C API.

For every converted CosyVoice GGUF found under models/cosyvoice/
(cosyvoice-300m, breezyvoice, cosyvoice2, cosyvoice3, cosyvoice-300m-sft):

  * codec_speaker_from_audio on a reference clip yields prompt_token /
    prompt_feat / embedding slots of the right shapes (24 kHz variants keep
    prompt_feat = 2 * prompt_token frames),
  * codec_decode with that speaker returns finite, non-silent PCM of the
    expected length, identically on a second call,
  * builtin (spk2info) speakers decode without reference audio.

Missing GGUFs are skipped. Convert with
    .venv/bin/python scripts/convert-to-gguf.py --checkpoint-path <dir> -o models/cosyvoice/<name>.gguf
Run from the repo root: .venv/bin/python tests/e2e/cosyvoice_decode_smoke.py
"""

from __future__ import annotations

import ctypes
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIB = ctypes.CDLL(str(REPO / "build" / "libcodec.so"))
REF_WAV = REPO / "input_audio" / "10_2_24k.wav"
MODELS = ["cosyvoice-300m", "breezyvoice", "cosyvoice2", "cosyvoice3", "cosyvoice-300m-sft"]


class ModelParams(ctypes.Structure):
    _fields_ = [("use_gpu", ctypes.c_bool), ("n_threads", ctypes.c_int32)]


class CtxParams(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int32)]


class DecodeParams(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int32), ("n_q", ctypes.c_int32), ("speaker", ctypes.c_void_p)]


class Audio(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("n_samples", ctypes.c_int32), ("sample_rate", ctypes.c_int32),
                ("n_channels", ctypes.c_int32), ("pcm_type", ctypes.c_int32)]


class Tokens(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_int32), ("n_frames", ctypes.c_int32),
                ("n_q", ctypes.c_int32), ("codebook_size", ctypes.c_int32), ("sample_rate", ctypes.c_int32),
                ("hop_size", ctypes.c_int32)]


class Pcm(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)), ("n_samples", ctypes.c_int32),
                ("sample_rate", ctypes.c_int32), ("n_channels", ctypes.c_int32)]


LIB.codec_model_load_from_file.restype = ctypes.c_void_p
LIB.codec_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParams]
LIB.codec_model_free.argtypes = [ctypes.c_void_p]
LIB.codec_init_from_model.restype = ctypes.c_void_p
LIB.codec_init_from_model.argtypes = [ctypes.c_void_p, CtxParams]
LIB.codec_free.argtypes = [ctypes.c_void_p]
LIB.codec_get_last_error.restype = ctypes.c_char_p
LIB.codec_get_last_error.argtypes = [ctypes.c_void_p]
LIB.codec_model_codebook_size.restype = ctypes.c_int32
LIB.codec_model_codebook_size.argtypes = [ctypes.c_void_p]
LIB.codec_model_sample_rate.restype = ctypes.c_int32
LIB.codec_model_sample_rate.argtypes = [ctypes.c_void_p]
LIB.codec_model_hop_size.restype = ctypes.c_int32
LIB.codec_model_hop_size.argtypes = [ctypes.c_void_p]
LIB.codec_model_n_speakers.restype = ctypes.c_int32
LIB.codec_model_n_speakers.argtypes = [ctypes.c_void_p]
LIB.codec_model_speaker_name.restype = ctypes.c_char_p
LIB.codec_model_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_int32]
LIB.codec_get_speaker.restype = ctypes.c_void_p
LIB.codec_get_speaker.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
LIB.codec_model_can_speaker_from_audio.restype = ctypes.c_bool
LIB.codec_model_can_speaker_from_audio.argtypes = [ctypes.c_void_p]
LIB.codec_speaker_from_audio.argtypes = [ctypes.c_void_p, ctypes.POINTER(Audio), ctypes.POINTER(ctypes.c_void_p)]
LIB.codec_speaker_free.argtypes = [ctypes.c_void_p]
LIB.codec_speaker_get_slot.restype = ctypes.c_bool
LIB.codec_speaker_get_slot.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
                                       ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_void_p)]
LIB.codec_decode.argtypes = [ctypes.c_void_p, ctypes.POINTER(Tokens), ctypes.POINTER(Pcm), DecodeParams]
LIB.codec_pcm_buffer_free.argtypes = [ctypes.POINTER(Pcm)]


def _slot_shape(spk, name):
    t = ctypes.c_int32()
    ne = (ctypes.c_int64 * 4)()
    d = ctypes.c_void_p()
    if not LIB.codec_speaker_get_slot(spk, name.encode(), ctypes.byref(t), ne, ctypes.byref(d)):
        return None
    return [ne[i] for i in range(4)]


def _decode(ctx, tokens, spk):
    tok = np.ascontiguousarray(tokens, dtype=np.int32)
    tb = Tokens(tok.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(tok), len(tok), 1, 0, 0, 0)
    out = Pcm()
    st = LIB.codec_decode(ctx, ctypes.byref(tb), ctypes.byref(out), DecodeParams(8, 1, spk))
    if st != 0:
        raise RuntimeError(f"codec_decode failed ({st}): {LIB.codec_get_last_error(ctx).decode()}")
    pcm = np.ctypeslib.as_array(out.data, shape=(out.n_samples,)).copy()
    LIB.codec_pcm_buffer_free(ctypes.byref(out))
    return pcm


def _check_pcm(pcm, n_expected, what):
    assert np.isfinite(pcm).all(), f"{what}: non-finite PCM"
    assert abs(len(pcm) - n_expected) <= 4, f"{what}: {len(pcm)} samples, expected ~{n_expected}"
    rms = float(np.sqrt((pcm ** 2).mean()))
    assert rms > 1e-3, f"{what}: silent output (rms={rms})"
    return rms


def _ref_audio():
    with wave.open(str(REF_WAV), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def run(name: str) -> None:
    path = REPO / "models" / "cosyvoice" / f"{name}.gguf"
    if not path.is_file():
        print(f"[SKIP] {name}: {path} not found")
        return
    model = LIB.codec_model_load_from_file(str(path).encode(), ModelParams(False, 8))
    assert model, f"{name}: model load failed"
    ctx = LIB.codec_init_from_model(model, CtxParams(0))
    sr = LIB.codec_model_sample_rate(model)
    hop = LIB.codec_model_hop_size(model)
    vocab = LIB.codec_model_codebook_size(model)
    rng = np.random.default_rng(5)
    target = rng.integers(0, vocab, size=60).astype(np.int32)
    try:
        if LIB.codec_model_can_speaker_from_audio(model):
            pcm, wsr = _ref_audio()
            a = Audio(pcm.ctypes.data, len(pcm), wsr, 1, 0)
            spk = ctypes.c_void_p()
            st = LIB.codec_speaker_from_audio(ctx, ctypes.byref(a), ctypes.byref(spk))
            assert st == 0, f"{name}: speaker_from_audio failed: {LIB.codec_get_last_error(ctx).decode()}"
            tok = _slot_shape(spk, "prompt_token")
            feat = _slot_shape(spk, "prompt_feat")
            emb = _slot_shape(spk, "embedding")
            assert tok and feat and emb and emb[0] == 192 and feat[0] == 80, f"{name}: bad slots {tok} {feat} {emb}"
            if sr == 24000:
                assert feat[1] == 2 * tok[0], f"{name}: prompt_feat {feat[1]} != 2 * prompt_token {tok[0]}"
            # v1: int(n / 50 * 22050 / 256) mel frames, x256 samples; v2/v3: 2 frames x 480.
            n_expected = int(len(target) / 50 * sr / 256) * 256 if sr == 22050 else len(target) * hop
            p1 = _decode(ctx, target, spk.value)
            rms = _check_pcm(p1, n_expected, name)
            p2 = _decode(ctx, target, spk.value)
            assert np.array_equal(p1, p2), f"{name}: decode is not deterministic"
            print(f"[PASS] {name}: speaker {tok[0]} tokens / {feat[1]} frames; decode {len(p1)} samples rms={rms:.4f}")
            LIB.codec_speaker_free(spk.value)
        for i in range(LIB.codec_model_n_speakers(model)):
            vname = LIB.codec_model_speaker_name(model, i).decode()
            spk = LIB.codec_get_speaker(model, vname.encode())
            prompt = _slot_shape(spk, "prompt_feat")
            if sr == 22050 and prompt is not None:
                continue   # joint length depends on the prompt; covered by the reference-audio case
            n_expected = int(len(target) / 50 * sr / 256) * 256 if sr == 22050 else len(target) * hop
            rms = _check_pcm(_decode(ctx, target, spk), n_expected, f"{name}/{vname}")
            print(f"[PASS] {name}: builtin speaker {vname} rms={rms:.4f}")
    finally:
        LIB.codec_free(ctx)
        LIB.codec_model_free(model)


def main() -> int:
    for name in MODELS:
        run(name)
    print("cosyvoice decode smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
