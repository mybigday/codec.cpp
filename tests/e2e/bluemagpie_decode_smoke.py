"""BlueMagpie AudioVAE continuous-latent decode parity smoke test.

Drives the public C API (`codec_decode_quantized_representation`) on the
converted AudioVAE GGUF and compares the 48 kHz waveform against a PyTorch
`AudioVAE.decode` reference computed on the same fixed latent.

Fixtures (generated from the real audiovae.pth, see tests/e2e/fixtures/bluemagpie):
  audiovae_z.npy      (latent_dim=64, n_frames)  — channel-major latent
  audiovae_audio.npy  (n_frames*1920,)           — reference 48 kHz PCM

Run:  .venv/bin/python tests/e2e/bluemagpie_decode_smoke.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIBCODEC = REPO / "build" / "libcodec.so"
GGUF = REPO / "models" / "bluemagpie" / "bluemagpie.gguf"
FIX = REPO / "tests" / "e2e" / "fixtures" / "bluemagpie"

CORR_MIN = 0.9999
MSE_MAX = 1e-6


class codec_model_params(ctypes.Structure):
    _fields_ = [("use_gpu", ctypes.c_bool), ("n_threads", ctypes.c_int32)]


class codec_context_params(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int32)]


class codec_decode_params(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int32), ("n_q", ctypes.c_int32)]


class codec_pcm_buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("n_channels", ctypes.c_int32),
    ]


class codec_audio(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p), ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32), ("n_channels", ctypes.c_int32),
        ("pcm_type", ctypes.c_int32),
    ]


class codec_encode_params(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int32), ("frame_size", ctypes.c_int32),
                ("hop_size", ctypes.c_int32), ("n_q", ctypes.c_int32)]


class codec_token_buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("n_tokens", ctypes.c_int32),
                ("n_frames", ctypes.c_int32), ("n_q", ctypes.c_int32),
                ("codebook_size", ctypes.c_int32), ("sample_rate", ctypes.c_int32),
                ("hop_size", ctypes.c_int32)]


class codec_latent_buffer(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)), ("latent_dim", ctypes.c_int32),
                ("n_frames", ctypes.c_int32), ("sample_rate", ctypes.c_int32),
                ("hop_size", ctypes.c_int32)]


def _bind():
    lib = ctypes.CDLL(str(LIBCODEC))
    lib.codec_model_load_from_file.argtypes = [ctypes.c_char_p, codec_model_params]
    lib.codec_model_load_from_file.restype = ctypes.c_void_p
    lib.codec_init_from_model.argtypes = [ctypes.c_void_p, codec_context_params]
    lib.codec_init_from_model.restype = ctypes.c_void_p
    lib.codec_decode_quantized_representation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(codec_pcm_buffer), codec_decode_params,
    ]
    lib.codec_decode_quantized_representation.restype = ctypes.c_int
    lib.codec_get_last_error.argtypes = [ctypes.c_void_p]
    lib.codec_get_last_error.restype = ctypes.c_char_p
    lib.codec_pcm_buffer_free.argtypes = [ctypes.POINTER(codec_pcm_buffer)]
    lib.codec_free.argtypes = [ctypes.c_void_p]
    lib.codec_model_free.argtypes = [ctypes.c_void_p]
    lib.codec_encode_latent.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(codec_audio),
        ctypes.POINTER(codec_token_buffer), ctypes.POINTER(codec_latent_buffer),
        codec_encode_params]
    lib.codec_encode_latent.restype = ctypes.c_int
    lib.codec_latent_buffer_free.argtypes = [ctypes.POINTER(codec_latent_buffer)]
    return lib


def check_encode(lib, ctx):
    """AudioVAE encode parity (audio → latent mu) vs PyTorch fixture."""
    fx = FIX / "audiovae_encode.npz"
    if not fx.is_file():
        print("encode: SKIP (no fixture)")
        return True
    z = np.load(fx)
    audio = np.ascontiguousarray(z["audio"].astype(np.float32))
    ref = z["mu"]  # (latent_dim, n_frames)
    au = codec_audio(audio.ctypes.data_as(ctypes.c_void_p), audio.size, 16000, 1, 0)
    tb = codec_token_buffer()
    lb = codec_latent_buffer()
    st = lib.codec_encode_latent(ctx, ctypes.byref(au), ctypes.byref(tb), ctypes.byref(lb),
                                 codec_encode_params(1, 0, 0, 0))
    if st != 0:
        print(f"encode: FAIL status={st} err={lib.codec_get_last_error(ctx).decode()}")
        return False
    D, T = lb.latent_dim, lb.n_frames
    out = np.ctypeslib.as_array(lb.data, shape=(D * T,)).copy().reshape(D, T)
    lib.codec_latent_buffer_free(ctypes.byref(lb))
    n = min(T, ref.shape[1])
    corr = float(np.corrcoef(out[:, :n].reshape(-1), ref[:, :n].reshape(-1))[0, 1])
    # latent magnitudes are large (±12) so judge on correlation, not abs error.
    ok = corr >= 0.9999
    print(f"encode: latent_dim={D} n_frames={T} corr={corr:.8f}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    for p in (LIBCODEC, GGUF, FIX / "audiovae_z.npy", FIX / "audiovae_audio.npy"):
        if not p.is_file():
            print(f"FAIL: missing {p}")
            return 1

    lib = _bind()
    z = np.load(FIX / "audiovae_z.npy").astype(np.float32)        # (latent_dim, n_frames)
    ref = np.load(FIX / "audiovae_audio.npy").astype(np.float32)  # (n_frames*1920,)
    latent_dim, n_frames = z.shape
    # channel-major buffer (buffer[d*n_frames + t] = z[d][t]) — matches the
    # ne=(n_frames, latent_dim) graph input.
    buf = np.ascontiguousarray(z).reshape(-1)

    model = lib.codec_model_load_from_file(str(GGUF).encode(), codec_model_params(False, 1))
    if not model:
        print("FAIL: model load")
        return 1
    ctx = lib.codec_init_from_model(model, codec_context_params(0))
    if not ctx:
        print("FAIL: ctx init")
        lib.codec_model_free(model)
        return 1

    pcm = codec_pcm_buffer()
    st = lib.codec_decode_quantized_representation(
        ctx, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        latent_dim, n_frames, ctypes.byref(pcm), codec_decode_params(1, 0))
    if st != 0:
        print(f"FAIL: decode status={st} err={lib.codec_get_last_error(ctx).decode()}")
        lib.codec_free(ctx); lib.codec_model_free(model)
        return 1

    out = np.ctypeslib.as_array(pcm.data, shape=(pcm.n_samples,)).copy()
    out_sr, out_n = pcm.sample_rate, pcm.n_samples
    lib.codec_pcm_buffer_free(ctypes.byref(pcm))

    encode_ok = check_encode(lib, ctx)

    lib.codec_free(ctx)
    lib.codec_model_free(model)

    n = min(len(out), len(ref))
    out, ref = out[:n], ref[:n]
    corr = float(np.corrcoef(out, ref)[0, 1])
    mse = float(np.mean((out - ref) ** 2))
    max_abs = float(np.max(np.abs(out - ref)))
    print(f"sr={out_sr} n_samples={out_n} (ref {len(ref)})")
    print(f"corr={corr:.8f}  mse={mse:.3e}  max_abs={max_abs:.3e}")

    decode_ok = corr >= CORR_MIN and mse <= MSE_MAX and len(out) == len(np.load(FIX / 'audiovae_audio.npy'))
    print(f"decode: {'PASS' if decode_ok else 'FAIL'}")
    ok = decode_ok and encode_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
