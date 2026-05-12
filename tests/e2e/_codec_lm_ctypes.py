"""Minimal ctypes wrapper around libcodec.so for parity tests.

Only exposes the codec_lm public API + codec_model load/free; sufficient
for tests/e2e/moss_ttsd_lm_gen_smoke.py to drive an AR loop in Python
without spawning a subprocess per step.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
LIBCODEC = REPO / "build" / "libcodec.so"


def _load() -> ctypes.CDLL:
    if not LIBCODEC.is_file():
        raise FileNotFoundError(
            f"libcodec.so not found at {LIBCODEC}; "
            f"build with cmake (it's added by default after the codec_objs/codec_shared "
            f"refactor)"
        )
    return ctypes.CDLL(str(LIBCODEC))


lib = _load()


# --- C structs ---

class codec_model_params(ctypes.Structure):
    _fields_ = [
        ("use_gpu",   ctypes.c_bool),
        ("n_threads", ctypes.c_int32),
    ]


class codec_lm_info(ctypes.Structure):
    _fields_ = [
        ("kind",                    ctypes.c_int),   # enum codec_lm_kind
        ("hidden_dim",              ctypes.c_int32),
        ("audio_embed_dim",         ctypes.c_int32),
        ("compose_audio_embed_dim", ctypes.c_int32),
        ("n_codebook",              ctypes.c_int32),
        ("codebook_sizes",          ctypes.POINTER(ctypes.c_int32)),
        ("delay_pattern",           ctypes.POINTER(ctypes.c_int32)),
        ("host_arch",               ctypes.c_char_p),
    ]


# --- prototypes ---

lib.codec_model_default_params.restype = codec_model_params

lib.codec_model_load_from_file.argtypes = [ctypes.c_char_p, codec_model_params]
lib.codec_model_load_from_file.restype  = ctypes.c_void_p

lib.codec_model_free.argtypes = [ctypes.c_void_p]
lib.codec_model_free.restype  = None

lib.codec_lm_create.argtypes = [ctypes.c_void_p]
lib.codec_lm_create.restype  = ctypes.c_void_p

lib.codec_lm_free.argtypes = [ctypes.c_void_p]
lib.codec_lm_free.restype  = None

lib.codec_lm_get_info.argtypes = [ctypes.c_void_p]
lib.codec_lm_get_info.restype  = ctypes.POINTER(codec_lm_info)

lib.codec_lm_kind_name.argtypes = [ctypes.c_int]
lib.codec_lm_kind_name.restype  = ctypes.c_char_p

lib.codec_lm_get_last_error.argtypes       = [ctypes.c_void_p]
lib.codec_lm_get_last_error.restype        = ctypes.c_char_p
lib.codec_lm_state_get_last_error.argtypes = [ctypes.c_void_p]
lib.codec_lm_state_get_last_error.restype  = ctypes.c_char_p
lib.codec_lm_get_create_error.argtypes     = []
lib.codec_lm_get_create_error.restype      = ctypes.c_char_p

lib.codec_lm_state_new.argtypes  = [ctypes.c_void_p]
lib.codec_lm_state_new.restype   = ctypes.c_void_p
lib.codec_lm_state_free.argtypes = [ctypes.c_void_p]
lib.codec_lm_state_free.restype  = None
lib.codec_lm_state_reset.argtypes = [ctypes.c_void_p]
lib.codec_lm_state_reset.restype  = None

lib.codec_lm_step_begin.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.codec_lm_step_begin.restype  = ctypes.c_int

lib.codec_lm_state_set_text_context.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.codec_lm_state_set_text_context.restype  = ctypes.c_int

lib.codec_lm_step_pending.argtypes = [ctypes.c_void_p]
lib.codec_lm_step_pending.restype  = ctypes.c_bool

lib.codec_lm_step_logits.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
]
lib.codec_lm_step_logits.restype  = ctypes.POINTER(ctypes.c_float)

lib.codec_lm_step_push_code.argtypes = [ctypes.c_void_p, ctypes.c_int32]
lib.codec_lm_step_push_code.restype  = ctypes.c_int

lib.codec_lm_step_finish.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
lib.codec_lm_step_finish.restype  = ctypes.c_int

lib.codec_lm_compose_audio_embd.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
]
lib.codec_lm_compose_audio_embd.restype  = ctypes.c_int


# --- pythonic wrapper ---

CODEC_STATUS_SUCCESS = 0


class CodecLM:
    """Convenience wrapper.  Owns the codec_model + codec_lm + codec_lm_state."""

    def __init__(self, gguf_path: Path | str):
        params = lib.codec_model_default_params()
        # Force CPU; parity tests are deterministic and the GGUF is
        # tiny enough that GPU offload isn't worth the variability.
        params.use_gpu   = False
        params.n_threads = max(1, (os.cpu_count() or 1) // 2)
        self.codec = lib.codec_model_load_from_file(str(gguf_path).encode("utf-8"), params)
        if not self.codec:
            raise RuntimeError(f"codec_model_load_from_file failed for {gguf_path}")
        self.lm = lib.codec_lm_create(self.codec)
        if not self.lm:
            err = lib.codec_lm_get_last_error(self.lm) or b""
            raise RuntimeError(f"codec_lm_create returned NULL ({err.decode()!r}); "
                               f"ensure GGUF has codec.lm.* metadata")
        info_ptr = lib.codec_lm_get_info(self.lm)
        self.info = info_ptr.contents
        self.n_cb        = int(self.info.n_codebook)
        self.hidden_dim  = int(self.info.hidden_dim)
        self.audio_embed_dim = int(self.info.audio_embed_dim)
        self.compose_audio_embed_dim = int(self.info.compose_audio_embed_dim)
        self.codebook_sizes  = [int(self.info.codebook_sizes[i]) for i in range(self.n_cb)]
        self.delay_pattern   = [int(self.info.delay_pattern[i])  for i in range(self.n_cb)]
        self.host_arch       = (self.info.host_arch or b"").decode()

    def state(self) -> "CodecLMState":
        s = lib.codec_lm_state_new(self.lm)
        if not s:
            raise RuntimeError("codec_lm_state_new failed")
        return CodecLMState(self, s)

    def compose_audio_embd(self, codes) -> "ctypes.Array":
        import numpy as np
        codes = np.asarray(codes, dtype=np.int32)
        if codes.shape != (self.n_cb,):
            raise ValueError(f"codes shape must be ({self.n_cb},), got {codes.shape}")
        out = (ctypes.c_float * self.audio_embed_dim)()
        rc = lib.codec_lm_compose_audio_embd(
            self.lm,
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out,
        )
        if rc != CODEC_STATUS_SUCCESS:
            err = (lib.codec_lm_get_last_error(self.lm) or b"").decode()
            raise RuntimeError(f"compose_audio_embd rc={rc}, err='{err}'")
        return out

    def close(self) -> None:
        if getattr(self, "lm", None):
            lib.codec_lm_free(self.lm); self.lm = None
        if getattr(self, "codec", None):
            lib.codec_model_free(self.codec); self.codec = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class CodecLMState:
    def __init__(self, parent: CodecLM, ptr):
        self.parent = parent
        self.ptr = ptr

    def step_begin(self, h_in) -> None:
        import numpy as np
        h = np.ascontiguousarray(np.asarray(h_in, dtype=np.float32))
        if h.shape != (self.parent.hidden_dim,):
            raise ValueError(
                f"h_in shape must be ({self.parent.hidden_dim},), got {h.shape}")
        rc = lib.codec_lm_step_begin(self.ptr, h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != CODEC_STATUS_SUCCESS:
            err = (lib.codec_lm_state_get_last_error(self.ptr) or b"").decode()
            raise RuntimeError(f"step_begin rc={rc}, err='{err}'")

    def set_text_context(self, text_token: int) -> None:
        rc = lib.codec_lm_state_set_text_context(self.ptr, ctypes.c_int32(int(text_token)))
        if rc != CODEC_STATUS_SUCCESS:
            err = (lib.codec_lm_state_get_last_error(self.ptr) or b"").decode()
            raise RuntimeError(f"set_text_context rc={rc}, err='{err}'")

    def step_logits(self):
        cb_idx = ctypes.c_int32(0)
        n      = ctypes.c_int32(0)
        ptr = lib.codec_lm_step_logits(self.ptr, ctypes.byref(cb_idx), ctypes.byref(n))
        if not ptr:
            err = (lib.codec_lm_state_get_last_error(self.ptr) or b"").decode()
            raise RuntimeError(f"step_logits returned NULL, err='{err}'")
        # ctypes-array view (no copy) — caller may copy if they want to keep it
        n_val = int(n.value)
        arr_t = ctypes.c_float * n_val
        return int(cb_idx.value), n_val, ctypes.cast(ptr, ctypes.POINTER(arr_t))[0]

    def step_push_code(self, code: int) -> None:
        rc = lib.codec_lm_step_push_code(self.ptr, ctypes.c_int32(int(code)))
        if rc != CODEC_STATUS_SUCCESS:
            err = (lib.codec_lm_state_get_last_error(self.ptr) or b"").decode()
            raise RuntimeError(f"step_push_code rc={rc}, err='{err}'")

    def step_finish(self):
        codes = (ctypes.c_int32 * self.parent.n_cb)()
        rc = lib.codec_lm_step_finish(self.ptr, codes)
        if rc != CODEC_STATUS_SUCCESS:
            err = (lib.codec_lm_state_get_last_error(self.ptr) or b"").decode()
            raise RuntimeError(f"step_finish rc={rc}, err='{err}'")
        return list(codes)

    def reset(self) -> None:
        lib.codec_lm_state_reset(self.ptr)

    def close(self) -> None:
        if getattr(self, "ptr", None):
            lib.codec_lm_state_free(self.ptr); self.ptr = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
