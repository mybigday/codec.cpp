#!/usr/bin/env python3
"""End-to-end smoke test for repeated Chatterbox-S3G decode calls.

Loads the model with GPU offload and calls codec_decode twice on the same
context. This catches scheduler allocations retained across eval-graph releases,
which caused the second Metal decode to return silent PCM.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT = REPO_ROOT / "scripts" / "convert-to-gguf.py"
CHATTERBOX_DIR = REPO_ROOT / "models" / "chatterbox"


class codec_model_params(ctypes.Structure):
    _fields_ = [("use_gpu", ctypes.c_bool), ("n_threads", ctypes.c_int32)]


class codec_context_params(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int32)]


class codec_decode_params(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int32), ("n_q", ctypes.c_int32)]


class codec_token_buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_int32),
        ("n_frames", ctypes.c_int32),
        ("n_q", ctypes.c_int32),
        ("codebook_size", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("hop_size", ctypes.c_int32),
    ]


class codec_pcm_buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("n_channels", ctypes.c_int32),
    ]


def run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.decode(errors='replace')[:4096]}")
    return proc.stdout


def find_libcodec() -> Path | None:
    for name in ("libcodec.dylib", "libcodec.so"):
        path = REPO_ROOT / "build" / name
        if path.is_file():
            return path
    return None


def bind(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))
    lib.codec_model_load_from_file.argtypes = [ctypes.c_char_p, codec_model_params]
    lib.codec_model_load_from_file.restype = ctypes.c_void_p
    lib.codec_model_free.argtypes = [ctypes.c_void_p]
    lib.codec_init_from_model.argtypes = [ctypes.c_void_p, codec_context_params]
    lib.codec_init_from_model.restype = ctypes.c_void_p
    lib.codec_free.argtypes = [ctypes.c_void_p]
    lib.codec_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(codec_token_buffer),
        ctypes.POINTER(codec_pcm_buffer),
        codec_decode_params,
    ]
    lib.codec_decode.restype = ctypes.c_int
    lib.codec_pcm_buffer_free.argtypes = [ctypes.POINTER(codec_pcm_buffer)]
    lib.codec_get_last_error.argtypes = [ctypes.c_void_p]
    lib.codec_get_last_error.restype = ctypes.c_char_p
    return lib


def decode(lib: ctypes.CDLL, ctx: int, tokens: np.ndarray) -> tuple[np.ndarray, int]:
    token_buf = codec_token_buffer(
        tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        tokens.size,
        tokens.size,
        1,
        0,
        0,
        0,
    )
    pcm_buf = codec_pcm_buffer()
    status = lib.codec_decode(
        ctx, ctypes.byref(token_buf), ctypes.byref(pcm_buf), codec_decode_params(4, 0)
    )
    if status != 0:
        error = lib.codec_get_last_error(ctx).decode(errors="replace")
        raise RuntimeError(f"codec_decode failed ({status}): {error}")
    pcm = np.ctypeslib.as_array(pcm_buf.data, shape=(pcm_buf.n_samples,)).copy()
    sample_rate = pcm_buf.sample_rate
    lib.codec_pcm_buffer_free(ctypes.byref(pcm_buf))
    return pcm, sample_rate


def main() -> int:
    lib_path = find_libcodec()
    if lib_path is None:
        print("SKIP: build/libcodec shared library not built")
        return 0
    if not (CHATTERBOX_DIR / "s3gen.safetensors").is_file() or not (CHATTERBOX_DIR / "conds.pt").is_file():
        print(f"SKIP: {CHATTERBOX_DIR} not present")
        return 0

    with tempfile.TemporaryDirectory(prefix="codec-s3g-decode-") as td:
        td = Path(td)
        gguf = td / "s3g.gguf"
        run([sys.executable, str(CONVERT), "--checkpoint-path", str(CHATTERBOX_DIR), "--model-type", "chatterbox_s3g", "--output", str(gguf)])

        tokens = np.ascontiguousarray([
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500, 1600,
        ], dtype=np.int32)

        lib = bind(lib_path)
        model = lib.codec_model_load_from_file(
            str(gguf).encode(), codec_model_params(True, 4)
        )
        if not model:
            print("FAIL: model load")
            return 1
        ctx = lib.codec_init_from_model(model, codec_context_params(0))
        if not ctx:
            lib.codec_model_free(model)
            print("FAIL: context init")
            return 1

        try:
            results = [decode(lib, ctx, tokens) for _ in range(2)]
        finally:
            lib.codec_free(ctx)
            lib.codec_model_free(model)

    for index, (pcm, sample_rate) in enumerate(results, start=1):
        rms = float(np.sqrt(np.mean(pcm**2)))
        peak = float(np.max(np.abs(pcm)))
        print(
            f"  run {index}: decoded {pcm.size} samples at {sample_rate} Hz; "
            f"rms={rms:.6f}, peak={peak:.6f}"
        )
        if not np.all(np.isfinite(pcm)):
            print(f"FAIL: run {index} contains non-finite samples")
            return 1
        if peak > 1.0:
            print(f"FAIL: run {index} clip exceeded ±1.0")
            return 1
        if rms < 1e-4:
            print(f"FAIL: run {index} is silent")
            return 1

    if results[0][1] != results[1][1] or results[0][0].shape != results[1][0].shape:
        print("FAIL: repeated same-context decodes have different output shapes")
        return 1
    if not np.allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-6):
        print("FAIL: repeated same-context decodes differ")
        return 1
    print("chatterbox S3G decode end-to-end smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
