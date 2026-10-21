"""High-level wrappers for the codec_lm adaptor API in include/codec_lm.h.

codec_lm is the piece between an AR backbone (run by llama.cpp) and the
codec: audio embedding / compose, per-step depth decoders, continuous-latent
heads, speaker encoders.  The backbone itself is not part of this package.
"""

from __future__ import annotations

import ctypes
import weakref
from dataclasses import dataclass

import numpy as np

from . import _ffi
from ._lib import lib
from .codec import CodecError, Model, _audio_struct, _f32, _fptr, _i32, _iptr, _str


def _struct_dict(s: ctypes.Structure) -> dict:
    out = {}
    for name, _t in s._fields_:
        v = getattr(s, name)
        out[name] = _str(v) if isinstance(v, bytes) else v
    return out


@dataclass(frozen=True)
class LMInfo:
    kind: int
    kind_name: str
    hidden_dim: int
    audio_embed_dim: int
    compose_audio_embed_dim: int
    n_codebook: int
    codebook_sizes: tuple[int, ...]
    delay_pattern: tuple[int, ...]
    host_arch: str
    is_continuous: bool
    patch_size: int
    latent_dim: int
    eos_code_c0: int
    eos_min_step: int


class LM:
    """codec_lm adaptor of a Model.

    Get it with Model.get_lm_adaptor(); there is one per Model and the Model
    frees it.  It keeps the Model alive while referenced.
    """

    def __init__(self, model: Model):
        self.model = model
        L = lib()
        self._states: weakref.WeakSet = weakref.WeakSet()
        self.ptr = L.codec_lm_create(model.ptr)
        if not self.ptr:
            raise CodecError("codec_lm_create failed", None, _str(L.codec_lm_get_create_error()))
        i = L.codec_lm_get_info(self.ptr).contents
        n = int(i.n_codebook)
        self.info = LMInfo(
            kind=int(i.kind),
            kind_name=_str(L.codec_lm_kind_name(i.kind)),
            hidden_dim=int(i.hidden_dim),
            audio_embed_dim=int(i.audio_embed_dim),
            compose_audio_embed_dim=int(i.compose_audio_embed_dim),
            n_codebook=n,
            codebook_sizes=tuple(int(i.codebook_sizes[k]) for k in range(n)) if i.codebook_sizes else (),
            delay_pattern=tuple(int(i.delay_pattern[k]) for k in range(n)) if i.delay_pattern else (),
            host_arch=_str(i.host_arch),
            is_continuous=bool(i.is_continuous),
            patch_size=int(i.patch_size),
            latent_dim=int(i.latent_dim),
            eos_code_c0=int(i.eos_code_c0),
            eos_min_step=int(i.eos_min_step),
        )

    @property
    def last_error(self) -> str:
        return _str(lib().codec_lm_get_last_error(self.ptr))

    def _check(self, what: str, st: int) -> None:
        if st != _ffi.CODEC_STATUS_SUCCESS:
            raise CodecError(what, st, self.last_error)

    def state(self) -> "LMState":
        return LMState(self)

    # --- embeddings ---------------------------------------------------

    @property
    def text_proj_dim(self) -> int:
        return int(lib().codec_lm_text_proj_dim(self.ptr))

    def project_text(self, text_token: int) -> np.ndarray:
        out = np.zeros(max(1, self.text_proj_dim), np.float32)
        if not lib().codec_lm_project_text(self.ptr, int(text_token), _fptr(out), out.size):
            raise CodecError("codec_lm_project_text", None, self.last_error)
        return out

    def codec_embd_row(self, code: int, dim: int | None = None) -> np.ndarray:
        out = np.zeros(dim or self.info.hidden_dim, np.float32)
        if not lib().codec_lm_codec_embd_row(self.ptr, int(code), _fptr(out), out.size):
            raise CodecError("codec_lm_codec_embd_row", None, self.last_error)
        return out

    def _compose_dim(self) -> int:
        return self.info.compose_audio_embed_dim or self.info.audio_embed_dim

    def compose_audio_embd(self, codes) -> np.ndarray:
        c = self._codes(codes)
        out = np.zeros(self._compose_dim(), np.float32)
        self._check("codec_lm_compose_audio_embd", lib().codec_lm_compose_audio_embd(self.ptr, _iptr(c), _fptr(out)))
        return out

    def compose_next_embd(self, codes, step: int) -> np.ndarray:
        c = self._codes(codes)
        out = np.zeros(self.info.hidden_dim, np.float32)
        self._check("codec_lm_compose_next_embd",
                    lib().codec_lm_compose_next_embd(self.ptr, _iptr(c), int(step), _fptr(out)))
        return out

    def _codes(self, codes) -> np.ndarray:
        c = _i32(codes).reshape(-1)
        if c.shape != (self.info.n_codebook,):
            raise ValueError(f"codes must have shape ({self.info.n_codebook},), got {c.shape}")
        return c

    # --- speaker encoder ----------------------------------------------

    def speaker_info(self) -> dict | None:
        p = lib().codec_lm_speaker_get_info(self.ptr)
        return _struct_dict(p.contents) if p else None

    def speaker_encode(self, *, ref_pcm=None, ref_sample_rate: int = 0, ref_speech_tokens=None,
                       emotion: float | None = None) -> np.ndarray:
        """Run the speaker-conditioning encoder -> (n_rows, hidden_dim)."""
        info = self.speaker_info()
        if info is None:
            raise CodecError("codec_lm_speaker_encode", _ffi.CODEC_STATUS_NOT_SUPPORTED, "no speaker section")
        audio_p = None
        if ref_pcm is not None:
            audio, _keep = _audio_struct(ref_pcm, ref_sample_rate or info["ref_sample_rate"])
            audio_p = ctypes.byref(audio)
        toks = None if ref_speech_tokens is None else _i32(ref_speech_tokens).reshape(-1)
        emo = None if emotion is None else np.array([emotion], np.float32)
        out = np.zeros((info["n_rows"], info["hidden_dim"]), np.float32)
        self._check("codec_lm_speaker_encode", lib().codec_lm_speaker_encode(
            self.ptr, audio_p, _iptr(toks), 0 if toks is None else toks.size, _fptr(emo), _fptr(out), out.size))
        return out

    def speaker_encode_from_embedding(self, speaker_emb, *, ref_speech_tokens=None,
                                      emotion: float | None = None) -> np.ndarray:
        info = self.speaker_info()
        if info is None:
            raise CodecError("codec_lm_speaker_encode_from_embedding", _ffi.CODEC_STATUS_NOT_SUPPORTED)
        emb = _f32(speaker_emb).reshape(-1)
        toks = None if ref_speech_tokens is None else _i32(ref_speech_tokens).reshape(-1)
        emo = None if emotion is None else np.array([emotion], np.float32)
        out = np.zeros((info["n_rows"], info["hidden_dim"]), np.float32)
        self._check("codec_lm_speaker_encode_from_embedding", lib().codec_lm_speaker_encode_from_embedding(
            self.ptr, _fptr(emb), emb.size, _iptr(toks), 0 if toks is None else toks.size, _fptr(emo),
            _fptr(out), out.size))
        return out

    # --- Chatterbox T3 helpers ----------------------------------------

    def _free(self) -> None:
        # Called by Model.close(); states created from this adaptor must not
        # be used afterwards.
        for st in list(self._states):
            st.close()  # codec_lm_state_free dereferences its lm
        if getattr(self, "ptr", None):
            lib().codec_lm_free(self.ptr)
            self.ptr = None


class LMState:
    """Per-sequence generation state (codec_lm_state).  Keeps the LM alive."""

    def __init__(self, lm: LM):
        self.lm = lm
        self.ptr = lib().codec_lm_state_new(lm.ptr)
        if not self.ptr:
            raise CodecError("codec_lm_state_new failed", None, lm.last_error)
        lm._states.add(self)

    @property
    def last_error(self) -> str:
        return _str(lib().codec_lm_state_get_last_error(self.ptr))

    def _check(self, what: str, st: int) -> None:
        if st != _ffi.CODEC_STATUS_SUCCESS:
            raise CodecError(what, st, self.last_error)

    def reset(self) -> None:
        lib().codec_lm_state_reset(self.ptr)

    def set_text_context(self, text_token: int) -> None:
        self._check("codec_lm_state_set_text_context",
                    lib().codec_lm_state_set_text_context(self.ptr, int(text_token)))

    # --- codebook step API --------------------------------------------

    def step_begin(self, h_in) -> None:
        h = _f32(h_in).reshape(-1)
        if h.size != self.lm.info.hidden_dim:
            raise ValueError(f"h_in must have {self.lm.info.hidden_dim} values, got {h.size}")
        self._check("codec_lm_step_begin", lib().codec_lm_step_begin(self.ptr, _fptr(h)))

    @property
    def step_pending(self) -> bool:
        return bool(lib().codec_lm_step_pending(self.ptr))

    def step_logits(self) -> tuple[int, np.ndarray]:
        """-> (codebook index, logits).  The logits array is a copy."""
        cb, n = ctypes.c_int32(0), ctypes.c_int32(0)
        p = lib().codec_lm_step_logits(self.ptr, ctypes.byref(cb), ctypes.byref(n))
        if not p:
            raise CodecError("codec_lm_step_logits returned NULL", None, self.last_error)
        return int(cb.value), np.ctypeslib.as_array(p, shape=(int(n.value),)).copy()

    def step_push_code(self, code: int) -> None:
        self._check("codec_lm_step_push_code", lib().codec_lm_step_push_code(self.ptr, int(code)))

    def step_finish(self) -> np.ndarray:
        out = np.zeros(self.lm.info.n_codebook, np.int32)
        self._check("codec_lm_step_finish", lib().codec_lm_step_finish(self.ptr, _iptr(out)))
        return out

    def step_is_eos(self, codes) -> bool:
        """False for models without an EOS sentinel (NOT_SUPPORTED)."""
        c = _i32(codes).reshape(-1)
        out = ctypes.c_int32(0)
        st = lib().codec_lm_step_is_eos(self.ptr, _iptr(c), c.size, ctypes.byref(out))
        return st == _ffi.CODEC_STATUS_SUCCESS and out.value != 0

    def step_greedy(self, h_in) -> np.ndarray:
        """step_begin + argmax over every codebook + step_finish."""
        self.step_begin(h_in)
        while self.step_pending:
            _, logits = self.step_logits()
            self.step_push_code(int(np.argmax(logits)))
        return self.step_finish()

    # --- continuous latent (CONTINUOUS_LATENT_CFM) --------------------

    def step_generate(self, h_in, *, cfg_value: float, n_timesteps: int,
                      noise=None) -> tuple[np.ndarray, bool]:
        """-> (patch (patch_size, latent_dim), stop flag)."""
        info = self.lm.info
        h = _f32(h_in).reshape(-1)
        nz = None if noise is None else _f32(noise).reshape(-1)
        out = np.zeros((info.patch_size, info.latent_dim), np.float32)
        stop = ctypes.c_int32(0)
        self._check("codec_lm_step_generate", lib().codec_lm_step_generate(
            self.ptr, _fptr(h), float(cfg_value), int(n_timesteps), _fptr(nz), _fptr(out), ctypes.byref(stop)))
        return out, bool(stop.value)

    def step_feedback_embd(self) -> np.ndarray:
        out = np.zeros(self.lm.info.hidden_dim, np.float32)
        self._check("codec_lm_step_feedback_embd", lib().codec_lm_step_feedback_embd(self.ptr, _fptr(out)))
        return out

    def text_prefill(self, hiddens) -> None:
        h = _f32(hiddens)
        h = h.reshape(-1, self.lm.info.hidden_dim)
        self._check("codec_lm_text_prefill",
                    lib().codec_lm_text_prefill(self.ptr, _fptr(h), int(h.shape[0]), int(h.shape[1])))

    def set_continuous_min_len(self, min_len: int) -> None:
        self._check("codec_lm_set_continuous_min_len", lib().codec_lm_set_continuous_min_len(self.ptr, int(min_len)))

    # --- FlowLM -------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "ptr", None):
            lib().codec_lm_state_free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
