"""Pocket-TTS FlowLM teacher-forced parity smoke test.

Drives the codec_lm flow_lm C API on the converted GGUF and replays the exact
recorded-noise trajectory from the reference (tests/e2e/fixtures/pocket_tts/
gen_lm_fixtures.py):

  - tokenize "Hello world."         -> must match reference token ids
  - prefill (text-only, no voice)
  - for each step: feed the RECORDED noise, feed the RECORDED previous latent
    (teacher-forced), compare the emitted latent + EOS logit to the reference.

Teacher-forcing (feeding the reference's previous latent each step instead of
the codec's own) keeps the trajectory aligned so every emitted latent stays
comparable — free-running feedback diverges after a few steps under F16 weights.

Run:  .venv/bin/python tests/e2e/pocket_tts_lm_smoke.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIBCODEC = REPO / "build" / "libcodec.so"
GGUF = REPO / "models" / "pocket_tts" / "pocket_tts_en.gguf"
FIX = REPO / "tests" / "e2e" / "fixtures" / "pocket_tts"

LATENT_CORR_MIN = 0.99
EOS_AGREE_MIN = 1.0   # per-step EOS decision must match every step
TEXT = "Hello world."


class codec_model_params(ctypes.Structure):
    _fields_ = [("use_gpu", ctypes.c_bool), ("n_threads", ctypes.c_int32)]


class codec_context_params(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int32)]


class flow_info(ctypes.Structure):
    _fields_ = [
        ("d_model", ctypes.c_int32), ("ldim", ctypes.c_int32),
        ("n_txt_bins", ctypes.c_int32), ("insert_bos_before_voice", ctypes.c_int32),
        ("frames_after_eos", ctypes.c_int32), ("temperature", ctypes.c_float),
        ("eos_threshold", ctypes.c_float), ("lsd_decode_steps", ctypes.c_int32),
        ("has_tokenizer", ctypes.c_int32),
    ]


def _bind():
    lib = ctypes.CDLL(str(LIBCODEC))
    lib.codec_model_load_from_file.argtypes = [ctypes.c_char_p, codec_model_params]
    lib.codec_model_load_from_file.restype = ctypes.c_void_p
    lib.codec_lm_create.argtypes = [ctypes.c_void_p]
    lib.codec_lm_create.restype = ctypes.c_void_p
    lib.codec_lm_get_create_error.restype = ctypes.c_char_p
    lib.codec_lm_state_new.argtypes = [ctypes.c_void_p]
    lib.codec_lm_state_new.restype = ctypes.c_void_p
    lib.codec_lm_flow_get_info.argtypes = [ctypes.c_void_p]
    lib.codec_lm_flow_get_info.restype = ctypes.POINTER(flow_info)
    lib.codec_lm_flow_tokenize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    lib.codec_lm_flow_tokenize.restype = ctypes.c_int
    lib.codec_lm_flow_prefill.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    lib.codec_lm_flow_prefill.restype = ctypes.c_int
    lib.codec_lm_flow_step.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)]
    lib.codec_lm_flow_step.restype = ctypes.c_int
    lib.codec_lm_state_get_last_error.argtypes = [ctypes.c_void_p]
    lib.codec_lm_state_get_last_error.restype = ctypes.c_char_p
    return lib


def main() -> int:
    for p in (LIBCODEC, GGUF, FIX / "lm_tokens.npy", FIX / "lm_noise.npy",
              FIX / "lm_latents.npy", FIX / "lm_eos.npy", FIX / "lm_meta.npz"):
        if not p.is_file():
            print(f"FAIL: missing {p}")
            return 1

    ref_tokens = np.load(FIX / "lm_tokens.npy").astype(np.int32)
    ref_noise = np.load(FIX / "lm_noise.npy").astype(np.float32)     # [n, ldim]
    ref_lat = np.load(FIX / "lm_latents.npy").astype(np.float32)     # [n, ldim]
    ref_eos = np.load(FIX / "lm_eos.npy").astype(np.float32)         # [n]
    meta = np.load(FIX / "lm_meta.npz")
    eos_thr = float(meta["eos_threshold"])
    n_steps, ldim = ref_lat.shape

    lib = _bind()
    model = lib.codec_model_load_from_file(str(GGUF).encode(), codec_model_params(False, 1))
    if not model:
        print("FAIL: model load"); return 1
    lm = lib.codec_lm_create(model)
    if not lm:
        print("FAIL: codec_lm_create:", lib.codec_lm_get_create_error().decode()); return 1
    info = lib.codec_lm_flow_get_info(lm)
    if not info:
        print("FAIL: not a flow_lm model"); return 1
    fi = info.contents
    print(f"flow_lm: d_model={fi.d_model} ldim={fi.ldim} n_bins={fi.n_txt_bins} "
          f"temp={fi.temperature:.3f} eos_thr={fi.eos_threshold:.2f} "
          f"lsd_steps={fi.lsd_decode_steps} has_tok={fi.has_tokenizer}")

    # ---- tokenize ----
    cap = 256
    ids = (ctypes.c_int32 * cap)()
    n_out = ctypes.c_int32(0)
    rc = lib.codec_lm_flow_tokenize(lm, TEXT.encode(), ids, cap, ctypes.byref(n_out))
    if rc != 0:
        print("FAIL: tokenize rc", rc); return 1
    got_tokens = np.array([ids[i] for i in range(n_out.value)], dtype=np.int32)
    tok_ok = np.array_equal(got_tokens, ref_tokens)
    print(f"tokens: got {got_tokens.tolist()} ref {ref_tokens.tolist()}  "
          f"{'PASS' if tok_ok else 'FAIL'}")

    st = lib.codec_lm_state_new(lm)
    if not st:
        print("FAIL: state_new"); return 1

    # ---- prefill (text-only) ----
    tok_arr = (ctypes.c_int32 * len(got_tokens))(*got_tokens.tolist())
    rc = lib.codec_lm_flow_prefill(st, tok_arr, len(got_tokens), None, 0)
    if rc != 0:
        print("FAIL: prefill rc", rc, lib.codec_lm_state_get_last_error(st).decode()); return 1

    # ---- teacher-forced AR steps ----
    out_lat = (ctypes.c_float * ldim)()
    eos_logit = ctypes.c_float(0.0)
    is_eos = ctypes.c_int32(0)
    got_lat = np.zeros((n_steps, ldim), dtype=np.float32)
    got_eos = np.zeros((n_steps,), dtype=np.float32)

    for step in range(n_steps):
        # For step>0, seed out_lat with the REFERENCE previous latent (teacher).
        if step > 0:
            for d in range(ldim):
                out_lat[d] = float(ref_lat[step - 1, d])
        noise = (ctypes.c_float * ldim)(*ref_noise[step].tolist())
        rc = lib.codec_lm_flow_step(st, noise, out_lat, ctypes.byref(eos_logit),
                                    ctypes.byref(is_eos))
        if rc != 0:
            print("FAIL: step", step, "rc", rc,
                  lib.codec_lm_state_get_last_error(st).decode()); return 1
        got_lat[step] = np.array([out_lat[d] for d in range(ldim)], dtype=np.float32)
        got_eos[step] = float(eos_logit.value)

    lat_corr = float(np.corrcoef(got_lat.reshape(-1), ref_lat.reshape(-1))[0, 1])
    lat_max_abs = float(np.max(np.abs(got_lat - ref_lat)))
    eos_agree = float(np.mean((got_eos > eos_thr) == (ref_eos > eos_thr)))
    eos_max_abs = float(np.max(np.abs(got_eos - ref_eos)))

    print(f"latent: corr={lat_corr:.6f} max_abs={lat_max_abs:.4e}")
    print(f"eos:    agree={eos_agree:.3f} logit_max_abs={eos_max_abs:.4e}")
    print(f"  ref eos[:5]={np.round(ref_eos[:5],3).tolist()}")
    print(f"  got eos[:5]={np.round(got_eos[:5],3).tolist()}")

    ok = tok_ok and lat_corr >= LATENT_CORR_MIN and eos_agree >= EOS_AGREE_MIN
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
