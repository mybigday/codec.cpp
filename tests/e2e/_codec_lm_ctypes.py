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
    # Field order MUST mirror struct codec_lm_info in include/codec_lm.h
    # exactly (ABI-append discipline: new fields at the end).
    _fields_ = [
        ("kind",                    ctypes.c_int),   # enum codec_lm_kind
        ("hidden_dim",              ctypes.c_int32),
        ("audio_embed_dim",         ctypes.c_int32),
        ("compose_audio_embed_dim", ctypes.c_int32),
        ("n_codebook",              ctypes.c_int32),
        ("codebook_sizes",          ctypes.POINTER(ctypes.c_int32)),
        ("delay_pattern",           ctypes.POINTER(ctypes.c_int32)),
        ("host_arch",               ctypes.c_char_p),
        # continuous-latent fields
        ("is_continuous",           ctypes.c_bool),
        ("patch_size",              ctypes.c_int32),
        ("latent_dim",              ctypes.c_int32),
        # end-of-audio metadata (Phase A)
        ("eos_code_c0",             ctypes.c_int32),
        ("eos_min_step",            ctypes.c_int32),
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

lib.codec_lm_step_is_eos.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
]
lib.codec_lm_step_is_eos.restype  = ctypes.c_int

lib.codec_lm_compose_audio_embd.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
]
lib.codec_lm_compose_audio_embd.restype  = ctypes.c_int


# --- codec_model decode side (used by per-stage e2e smokes to turn a
#     codec_lm-emitted code stream into PCM via codec_decode) ----------

class codec_decode_params(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int32),
        ("n_q",       ctypes.c_int32),
    ]


class codec_encode_params(ctypes.Structure):
    _fields_ = [
        ("n_threads",  ctypes.c_int32),
        ("frame_size", ctypes.c_int32),
        ("hop_size",   ctypes.c_int32),
        ("n_q",        ctypes.c_int32),
    ]


class codec_context_params(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_int32),
    ]


class codec_token_buffer(ctypes.Structure):
    _fields_ = [
        ("data",          ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens",      ctypes.c_int32),
        ("n_frames",      ctypes.c_int32),
        ("n_q",           ctypes.c_int32),
        ("codebook_size", ctypes.c_int32),
        ("sample_rate",   ctypes.c_int32),
        ("hop_size",      ctypes.c_int32),
    ]


class codec_audio(ctypes.Structure):
    _fields_ = [
        ("data",         ctypes.c_void_p),
        ("n_samples",    ctypes.c_int32),
        ("sample_rate",  ctypes.c_int32),
        ("n_channels",   ctypes.c_int32),
        ("pcm_type",     ctypes.c_int),     # enum codec_pcm_type
    ]


class codec_pcm_buffer(ctypes.Structure):
    _fields_ = [
        ("data",        ctypes.POINTER(ctypes.c_float)),
        ("n_samples",   ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("n_channels",  ctypes.c_int32),
    ]


CODEC_PCM_TYPE_F32 = 0
CODEC_PCM_TYPE_I16 = 1


lib.codec_context_default_params.restype = codec_context_params
lib.codec_decode_default_params.restype  = codec_decode_params
lib.codec_encode_default_params.restype  = codec_encode_params

lib.codec_init_from_model.argtypes = [ctypes.c_void_p, codec_context_params]
lib.codec_init_from_model.restype  = ctypes.c_void_p

lib.codec_free.argtypes = [ctypes.c_void_p]
lib.codec_free.restype  = None

lib.codec_get_last_error.argtypes = [ctypes.c_void_p]
lib.codec_get_last_error.restype  = ctypes.c_char_p

lib.codec_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(codec_token_buffer),
    ctypes.POINTER(codec_pcm_buffer),
    codec_decode_params,
]
lib.codec_decode.restype = ctypes.c_int

lib.codec_encode.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(codec_audio),
    ctypes.POINTER(codec_token_buffer),
    codec_encode_params,
]
lib.codec_encode.restype = ctypes.c_int

lib.codec_pcm_buffer_free.argtypes   = [ctypes.POINTER(codec_pcm_buffer)]
lib.codec_pcm_buffer_free.restype    = None
lib.codec_token_buffer_free.argtypes = [ctypes.POINTER(codec_token_buffer)]
lib.codec_token_buffer_free.restype  = None

lib.codec_model_has_encoder.argtypes = [ctypes.c_void_p]
lib.codec_model_has_encoder.restype  = ctypes.c_bool

lib.codec_model_sample_rate.argtypes  = [ctypes.c_void_p]
lib.codec_model_sample_rate.restype   = ctypes.c_int32
lib.codec_model_n_q.argtypes          = [ctypes.c_void_p]
lib.codec_model_n_q.restype           = ctypes.c_int32
lib.codec_model_hop_size.argtypes     = [ctypes.c_void_p]
lib.codec_model_hop_size.restype      = ctypes.c_int32
lib.codec_model_codebook_size.argtypes = [ctypes.c_void_p]
lib.codec_model_codebook_size.restype  = ctypes.c_int32
lib.codec_model_has_decoder.argtypes  = [ctypes.c_void_p]
lib.codec_model_has_decoder.restype   = ctypes.c_bool


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
        self.eos_code_c0     = int(self.info.eos_code_c0)
        self.eos_min_step    = int(self.info.eos_min_step)

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
        # When a model exposes a separate compose-output dim (e.g. LFM2-Audio
        # where audio_embed_dim=1024 but compose writes a 2048-wide backbone
        # embed), use that; fall back to audio_embed_dim otherwise.
        out_dim = (self.compose_audio_embed_dim
                   if self.compose_audio_embed_dim > 0
                   else self.audio_embed_dim)
        out = (ctypes.c_float * out_dim)()
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

    def step_is_eos(self, codes) -> bool:
        """Ask the model whether the just-emitted frame `codes` is EOS.
        Returns False for NOT_SUPPORTED kinds / absent sentinel."""
        import numpy as np
        c = np.ascontiguousarray(np.asarray(codes, dtype=np.int32))
        out = ctypes.c_int32(0)
        rc = lib.codec_lm_step_is_eos(
            self.ptr,
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(int(c.shape[0])),
            ctypes.byref(out),
        )
        # NOT_SUPPORTED (rc != SUCCESS) → treat as "not EOS".
        return rc == CODEC_STATUS_SUCCESS and out.value != 0

    def reset(self) -> None:
        lib.codec_lm_state_reset(self.ptr)

    def close(self) -> None:
        if getattr(self, "ptr", None):
            lib.codec_lm_state_free(self.ptr); self.ptr = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class CodecDecoder:
    """Wraps `codec_model + codec_context` for encode + decode.

    The audio codecs in this repo (Mimi, Qwen-codec, XY-Tokenizer, ...)
    expose `codec_encode` / `codec_decode` — this class hides the
    ctypes plumbing used by per-stage e2e smokes.  Name kept as
    "Decoder" for historical reasons; `.encode()` is also supported
    when the codec has an encoder.
    """

    def __init__(self, gguf_path, *, use_gpu: bool = False, n_threads: int | None = None):
        mp = lib.codec_model_default_params()
        mp.use_gpu   = bool(use_gpu)
        mp.n_threads = int(n_threads) if n_threads else max(1, (os.cpu_count() or 1) // 2)
        self.model = lib.codec_model_load_from_file(str(gguf_path).encode("utf-8"), mp)
        if not self.model:
            raise RuntimeError(f"codec_model_load_from_file failed for {gguf_path}")
        cp = lib.codec_context_default_params()
        self.ctx = lib.codec_init_from_model(self.model, cp)
        if not self.ctx:
            lib.codec_model_free(self.model); self.model = None
            raise RuntimeError("codec_init_from_model failed")
        self.has_encoder   = bool(lib.codec_model_has_encoder(self.model))
        self.has_decoder   = bool(lib.codec_model_has_decoder(self.model))
        self.sample_rate   = int(lib.codec_model_sample_rate(self.model))
        self.n_q           = int(lib.codec_model_n_q(self.model))
        self.hop_size      = int(lib.codec_model_hop_size(self.model))
        self.codebook_size = int(lib.codec_model_codebook_size(self.model))

    def decode(self, codes, *, n_q: int = 0, n_threads: int | None = None):
        """codes: int32 (n_frames, n_q) ndarray; returns (PCM, sr)."""
        import numpy as np
        if not self.has_decoder:
            raise RuntimeError("codec model has no decoder")
        c = np.ascontiguousarray(np.asarray(codes, dtype=np.int32))
        if c.ndim != 2:
            raise ValueError(f"codes must be 2D (T,Q), got shape {c.shape}")
        n_frames, q = c.shape

        tb = codec_token_buffer()
        tb.data          = c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        tb.n_q           = q
        tb.n_frames      = n_frames
        tb.n_tokens      = q * n_frames
        tb.codebook_size = self.codebook_size
        tb.sample_rate   = self.sample_rate
        tb.hop_size      = self.hop_size

        pcm = codec_pcm_buffer()
        dp = lib.codec_decode_default_params()
        dp.n_threads = int(n_threads) if n_threads else max(1, (os.cpu_count() or 1) // 2)
        dp.n_q       = int(n_q)
        rc = lib.codec_decode(self.ctx, ctypes.byref(tb), ctypes.byref(pcm), dp)
        if rc != CODEC_STATUS_SUCCESS:
            err = (lib.codec_get_last_error(self.ctx) or b"").decode()
            raise RuntimeError(f"codec_decode rc={rc}, err='{err}'")
        out = np.zeros(pcm.n_samples, dtype=np.float32)
        ctypes.memmove(out.ctypes.data, pcm.data, pcm.n_samples * 4)
        sr = int(pcm.sample_rate)
        nc = int(pcm.n_channels)
        lib.codec_pcm_buffer_free(ctypes.byref(pcm))
        if nc > 1:
            out = out.reshape(-1, nc)
        return out, sr

    def encode(self, pcm, sample_rate: int, *, n_q: int = 0,
               n_threads: int | None = None):
        """pcm: float32 (n_samples,) or (n_samples, n_channels); returns
        codes int32 (n_frames, n_q)."""
        import numpy as np
        if not self.has_encoder:
            raise RuntimeError("codec model has no encoder")
        if pcm.ndim == 1:
            n_ch = 1
            arr = np.ascontiguousarray(pcm.astype(np.float32, copy=False))
        elif pcm.ndim == 2:
            n_ch = int(pcm.shape[1])
            arr = np.ascontiguousarray(pcm.astype(np.float32, copy=False))
        else:
            raise ValueError(f"unsupported PCM shape {pcm.shape}")

        audio = codec_audio()
        audio.data        = arr.ctypes.data_as(ctypes.c_void_p)
        audio.n_samples   = int(arr.shape[0])
        audio.sample_rate = int(sample_rate)
        audio.n_channels  = n_ch
        audio.pcm_type    = CODEC_PCM_TYPE_F32

        tb = codec_token_buffer()
        ep = lib.codec_encode_default_params()
        ep.n_threads = int(n_threads) if n_threads else max(1, (os.cpu_count() or 1) // 2)
        ep.n_q       = int(n_q)
        rc = lib.codec_encode(self.ctx, ctypes.byref(audio), ctypes.byref(tb), ep)
        if rc != CODEC_STATUS_SUCCESS:
            err = (lib.codec_get_last_error(self.ctx) or b"").decode()
            raise RuntimeError(f"codec_encode rc={rc}, err='{err}'")
        q  = int(tb.n_q); nf = int(tb.n_frames)
        codes = np.empty((nf, q), dtype=np.int32)
        ctypes.memmove(codes.ctypes.data, tb.data, nf * q * 4)
        lib.codec_token_buffer_free(ctypes.byref(tb))
        return codes

    def close(self) -> None:
        if getattr(self, "ctx", None):
            lib.codec_free(self.ctx); self.ctx = None
        if getattr(self, "model", None):
            lib.codec_model_free(self.model); self.model = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
