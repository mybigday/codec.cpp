"""High-level wrapper for the audio-LM host layer in include/codec_audio_lm.h.

AudioLM is what a TTS host loop talks to: it loads a codec / codec_lm GGUF,
describes the model's prompt (prompt_info), builds speaker / conditioning
prompts, takes each backbone step (observe_*, the codebook step machine,
prompt / step embeddings) and decodes the accumulated codes to PCM.  The
backbone itself runs in llama.cpp (e.g. llama-cpp-python) and is not part of
this package.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import _ffi
from ._lib import lib
from .codec import CodecError, PathLike, Speaker, _f32, _fptr, _i32, _iptr, _make_kv_overrides, _str, _take_pcm

PASSTHROUGH = _ffi.CODEC_AUDIO_LM_OBSERVE_PASSTHROUGH
CONSUMED = _ffi.CODEC_AUDIO_LM_OBSERVE_CONSUMED
CONSUMED_EMBED = _ffi.CODEC_AUDIO_LM_OBSERVE_CONSUMED_EMBED
STOP = _ffi.CODEC_AUDIO_LM_OBSERVE_STOP

KIND_NAMES = {
    _ffi.CODEC_AUDIO_LM_KIND_UNKNOWN: "unknown",
    _ffi.CODEC_AUDIO_LM_KIND_RESIDUAL_DEPTH_AR: "residual_depth_ar",
    _ffi.CODEC_AUDIO_LM_KIND_PARALLEL_HEADS_DELAY: "parallel_heads_delay",
    _ffi.CODEC_AUDIO_LM_KIND_CONTINUOUS_CFM: "continuous_cfm",
    _ffi.CODEC_AUDIO_LM_KIND_TOKEN_SINGLE_CB: "token_single_cb",
}


@dataclass(frozen=True)
class SpeakerEntry:
    name: str
    ref_text: str
    is_default: bool


@dataclass(frozen=True)
class Prompt:
    tokens: np.ndarray                 # int32 (n,)
    embeds_prefix: np.ndarray | None   # float32 (rows, hidden)
    embeds_uncond: np.ndarray | None   # float32 (rows, hidden), CFG branch
    default_temperature: float
    default_top_p: float
    default_min_p: float
    default_repetition_penalty: float
    default_cfg_weight: float
    start_token: int
    stop_token: int


def _info_dict(s: _ffi.codec_audio_lm_prompt_info) -> dict:
    out = {}
    for name, _t in s._fields_:
        v = getattr(s, name)
        out[name] = _str(v) if isinstance(v, bytes) or (v is None and _t is ctypes.c_char_p) else v
    out["model_kind_name"] = KIND_NAMES.get(out["model_kind"], "unknown")
    return out


class AudioLM:
    """An audio-LM host context (codec_audio_lm) over a codec GGUF.

    ``kv_overrides`` maps GGUF metadata keys to int / float / bool / str
    values applied at load time.
    """

    def __init__(self, codec_path: PathLike, *, use_gpu: bool = False, n_threads: int = 0,
                 kv_overrides: Mapping[str, object] | None = None):
        L = lib()
        p = L.codec_audio_lm_default_params()
        p.codec_path = os.fsencode(codec_path)
        p.use_gpu = bool(use_gpu)
        p.n_threads = int(n_threads)
        self._kv = None
        if kv_overrides:
            self._kv = _make_kv_overrides(kv_overrides)
            p.kv_overrides = self._kv
            p.n_kv_overrides = len(kv_overrides)
        err = ctypes.create_string_buffer(1024)
        self.ptr = L.codec_audio_lm_init(ctypes.byref(p), err, len(err))
        if not self.ptr:
            raise CodecError(f"codec_audio_lm_init({codec_path})", None, _str(err.value))

    @classmethod
    def bare(cls) -> "AudioLM":
        """A context with no model (code accumulator + token ranges only)."""
        self = cls.__new__(cls)
        self._kv = None
        self.ptr = lib().codec_audio_lm_alloc_bare()
        if not self.ptr:
            raise CodecError("codec_audio_lm_alloc_bare failed")
        return self

    def close(self) -> None:
        if getattr(self, "ptr", None):
            lib().codec_audio_lm_free(self.ptr)
            self.ptr = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()

    def reset(self) -> None:
        """Drop per-sequence state (codes, step machine); weights stay."""
        lib().codec_audio_lm_reset(self.ptr)

    def _fail(self, what: str):
        raise CodecError(what, None, self.last_error)

    # --- capabilities --------------------------------------------------

    @property
    def last_error(self) -> str:
        return _str(lib().codec_audio_lm_last_error(self.ptr))

    @property
    def modality(self) -> int:
        return int(lib().codec_audio_lm_modality(self.ptr))

    @property
    def has_speaker_enc(self) -> bool:
        return bool(lib().codec_audio_lm_has_speaker_enc(self.ptr))

    @property
    def n_codebook(self) -> int:
        return int(lib().codec_audio_lm_n_codebook(self.ptr))

    @property
    def hidden_dim(self) -> int:
        return int(lib().codec_audio_lm_hidden_dim(self.ptr))

    @property
    def is_continuous(self) -> bool:
        return bool(lib().codec_audio_lm_is_continuous(self.ptr))

    def meta(self, key: str) -> str | None:
        """Raw codec GGUF metadata value (stringified), or None."""
        v = lib().codec_audio_lm_meta_str(self.ptr, key.encode("utf-8"))
        return None if v is None else _str(v)

    @property
    def lm_ptr(self) -> int | None:
        """Borrowed codec_lm handle for the raw codec_lm_* calls (None without one)."""
        return lib().codec_audio_lm_get_lm(self.ptr)

    def prompt_info(self) -> dict:
        """The model's prompt descriptor (codec_audio_lm_prompt_info) as a dict."""
        out = _ffi.codec_audio_lm_prompt_info()
        if not lib().codec_audio_lm_get_prompt_info(self.ptr, ctypes.byref(out)):
            self._fail("codec_audio_lm_get_prompt_info")
        return _info_dict(out)

    def grammar(self, text: str) -> str:
        """GBNF for the backbone sampler that the model derives for ``text`` ("" when none)."""
        L = lib()
        t = text.encode("utf-8")
        n = L.codec_audio_lm_grammar(self.ptr, t, None, 0)
        if n < 0:
            self._fail("codec_audio_lm_grammar")
        buf = ctypes.create_string_buffer(n + 1)
        L.codec_audio_lm_grammar(self.ptr, t, buf, n + 1)
        return _str(buf.value)

    # --- decoder speaker ----------------------------------------------

    @property
    def has_speakers(self) -> bool:
        return bool(lib().codec_audio_lm_codec_has_speakers(self.ptr))

    @property
    def can_speaker_from_audio(self) -> bool:
        return bool(lib().codec_audio_lm_codec_can_speaker_from_audio(self.ptr))

    def _listed(self, n: int) -> list[SpeakerEntry]:
        L = lib()
        out = []
        for i in range(n):
            name, ref, dflt = ctypes.c_char_p(), ctypes.c_char_p(), ctypes.c_bool()
            L.codec_audio_lm_listed_speaker(self.ptr, i, ctypes.byref(name), ctypes.byref(ref), ctypes.byref(dflt))
            out.append(SpeakerEntry(_str(name.value), _str(ref.value), bool(dflt.value)))
        return out

    def speakers(self) -> list[SpeakerEntry]:
        """Builtin speakers of the codec."""
        n = lib().codec_audio_lm_list_codec_speakers(self.ptr)
        if n < 0:
            self._fail("codec_audio_lm_list_codec_speakers")
        return self._listed(n)

    def speaker_file_entries(self, path: PathLike) -> list[SpeakerEntry]:
        """Speakers in a speaker file (checked against this codec)."""
        n = lib().codec_audio_lm_list_speaker_file(self.ptr, os.fsencode(path))
        if n < 0:
            self._fail(f"codec_audio_lm_list_speaker_file({path})")
        return self._listed(n)

    def set_speaker(self, name: str | None = None) -> None:
        """Pick a builtin decoder speaker (None = the model default)."""
        if not lib().codec_audio_lm_set_decode_speaker(self.ptr, None if name is None else name.encode("utf-8")):
            self._fail(f"set_decode_speaker({name})")

    def load_speaker(self, path: PathLike, name: str | None = None) -> None:
        """Use a speaker from a speaker file (name None = the file default)."""
        n = None if name is None else name.encode("utf-8")
        if not lib().codec_audio_lm_load_decode_speaker(self.ptr, os.fsencode(path), n):
            self._fail(f"load_decode_speaker({path})")

    def speaker_from_audio(self, pcm, sample_rate: int, ref_text: str | None = None) -> None:
        """Derive the decoder speaker from a reference clip (mono float)."""
        x = _f32(pcm).reshape(-1)
        t = None if ref_text is None else ref_text.encode("utf-8")
        if not lib().codec_audio_lm_decode_speaker_from_audio(self.ptr, _fptr(x), x.size, int(sample_rate), t):
            self._fail("decode_speaker_from_audio")

    def save_speaker(self, path: PathLike, name: str) -> None:
        if not lib().codec_audio_lm_save_decode_speaker(self.ptr, os.fsencode(path), name.encode("utf-8")):
            self._fail(f"save_decode_speaker({path})")

    @property
    def speaker(self) -> Speaker | None:
        """The speaker decode uses, borrowed from this context (None without speakers)."""
        p = lib().codec_audio_lm_decode_speaker(self.ptr)
        return Speaker(p, owner=self) if p else None

    # --- prompt --------------------------------------------------------

    def build_prompt(self, text: str | None = None, *, ref_pcm=None, ref_sample_rate: int = 0,
                     ref_text: str | None = None, speaker_emb=None, emotion: float | None = None,
                     extra: Mapping[str, str] | None = None) -> Prompt:
        """Speaker / conditioning prompt for the backbone (codec_audio_lm_build_prompt)."""
        L = lib()
        inp = L.codec_audio_lm_default_input()
        keep = []
        if ref_pcm is not None:
            x = _f32(ref_pcm).reshape(-1)
            keep.append(x)
            inp.ref_pcm, inp.ref_n_samples, inp.ref_sample_rate = _fptr(x), x.size, int(ref_sample_rate)
        if speaker_emb is not None:
            e = _f32(speaker_emb).reshape(-1)
            keep.append(e)
            inp.speaker_emb, inp.speaker_emb_dim = _fptr(e), e.size
        if emotion is not None:
            em = ctypes.c_float(emotion)
            keep.append(em)
            inp.emotion = ctypes.pointer(em)
        inp.ref_text = None if ref_text is None else ref_text.encode("utf-8")
        inp.text = None if text is None else text.encode("utf-8")
        if extra:
            ks = (ctypes.c_char_p * len(extra))(*[k.encode("utf-8") for k in extra])
            vs = (ctypes.c_char_p * len(extra))(*[str(v).encode("utf-8") for v in extra.values()])
            keep += [ks, vs]
            inp.extra_keys, inp.extra_values, inp.n_extra = ks, vs, len(extra)
        out = _ffi.codec_audio_lm_prompt()
        if not L.codec_audio_lm_build_prompt(self.ptr, ctypes.byref(inp), ctypes.byref(out)):
            self._fail("codec_audio_lm_build_prompt")
        rows, hid = int(out.embeds_prefix_rows), int(out.embeds_prefix_hidden)

        def mat(p):
            return np.ctypeslib.as_array(p, shape=(rows, hid)).copy() if p and rows > 0 and hid > 0 else None

        return Prompt(
            tokens=np.ctypeslib.as_array(out.tokens, shape=(out.n_tokens,)).copy() if out.n_tokens > 0
            else np.zeros(0, np.int32),
            embeds_prefix=mat(out.embeds_prefix),
            embeds_uncond=mat(out.embeds_uncond),
            default_temperature=float(out.default_temperature),
            default_top_p=float(out.default_top_p),
            default_min_p=float(out.default_min_p),
            default_repetition_penalty=float(out.default_repetition_penalty),
            default_cfg_weight=float(out.default_cfg_weight),
            start_token=int(out.start_token),
            stop_token=int(out.stop_token),
        )

    # --- token ranges / accumulator -----------------------------------

    def set_audio_token_range(self, offset: int, count: int, eos_id: int = -1) -> None:
        lib().codec_audio_lm_set_audio_token_range(self.ptr, offset, count, eos_id)

    def set_audio_token_ranges(self, offsets: Sequence[int], counts: Sequence[int], *,
                               frame_sentinel_id: int = -1, eos_id: int = -1) -> None:
        o, c = _i32(offsets), _i32(counts)
        if o.size != c.size:
            raise ValueError("offsets and counts must have the same length")
        lib().codec_audio_lm_set_audio_token_ranges(self.ptr, _iptr(o), _iptr(c), o.size, frame_sentinel_id, eos_id)

    def audio_token_range(self) -> tuple[int, int, int]:
        o, c, e = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
        lib().codec_audio_lm_get_audio_token_range(self.ptr, ctypes.byref(o), ctypes.byref(c), ctypes.byref(e))
        return int(o.value), int(c.value), int(e.value)

    @property
    def n_frames(self) -> int:
        """Complete audio frames accumulated so far."""
        return int(lib().codec_audio_lm_codes_n_frames(self.ptr))

    def codes(self) -> np.ndarray:
        """Accumulated codes, int32 (n_frames, n_q)."""
        q, t = ctypes.c_int32(), ctypes.c_int32()
        p = lib().codec_audio_lm_codes(self.ptr, ctypes.byref(q), ctypes.byref(t))
        if not p or t.value <= 0:
            return np.zeros((0, max(q.value, 1)), np.int32)
        return np.ctypeslib.as_array(p, shape=(t.value, q.value)).copy()

    def lm_eos(self) -> tuple[int, int]:
        """(eos_code_c0, eos_min_step) of the codec_lm; (-1, 0) when none."""
        c, s = ctypes.c_int32(), ctypes.c_int32()
        lib().codec_audio_lm_get_lm_eos(self.ptr, ctypes.byref(c), ctypes.byref(s))
        return int(c.value), int(s.value)

    @property
    def uses_embed_override(self) -> bool:
        return bool(lib().codec_audio_lm_get_uses_embed_override(self.ptr))

    def set_uses_embed_override(self, enabled: bool, start_step: int = 0) -> None:
        lib().codec_audio_lm_set_uses_embed_override(self.ptr, bool(enabled), int(start_step))

    # --- per-step hooks -------------------------------------------------

    def observe_token(self, token: int, hidden=None) -> int:
        """Classify a sampled backbone token; returns PASSTHROUGH / CONSUMED / CONSUMED_EMBED / STOP."""
        h = None if hidden is None else _f32(hidden).reshape(-1)
        return int(lib().codec_audio_lm_observe_token(self.ptr, int(token), _fptr(h), 0 if h is None else h.size))

    def next_embed(self) -> np.ndarray | None:
        """The next backbone input row after CONSUMED_EMBED (a copy)."""
        d = ctypes.c_int32()
        p = lib().codec_audio_lm_get_next_embed(self.ptr, ctypes.byref(d))
        return np.ctypeslib.as_array(p, shape=(d.value,)).copy() if p and d.value > 0 else None

    def observe_hidden(self, hidden, noise=None) -> int:
        """Continuous-latent step from a backbone hidden (noise: CFM init, or None to sample)."""
        h = _f32(hidden).reshape(-1)
        n = None if noise is None else _f32(noise).reshape(-1)
        return int(lib().codec_audio_lm_observe_hidden(self.ptr, _fptr(h), h.size, _fptr(n)))

    def set_continuous_params(self, cfg_value: float, n_timesteps: int, min_len: int = -1) -> None:
        lib().codec_audio_lm_set_continuous_params(self.ptr, float(cfg_value), int(n_timesteps), int(min_len))

    def text_prefill(self, hiddens) -> None:
        """Prime a continuous-latent model with the prompt hiddens (n_pos, hidden)."""
        h = _f32(hiddens)
        if h.ndim != 2:
            raise ValueError("hiddens must be (n_pos, hidden)")
        if not lib().codec_audio_lm_text_prefill(self.ptr, _fptr(h), h.shape[0], h.shape[1]):
            self._fail("codec_audio_lm_text_prefill")

    def observe_codes(self, codes, hidden=None) -> int:
        """Accumulate one full frame of codes (residual depth-AR / parallel heads)."""
        c = _i32(codes).reshape(-1)
        h = None if hidden is None else _f32(hidden).reshape(-1)
        return int(lib().codec_audio_lm_observe_codes(self.ptr, _iptr(c), c.size, _fptr(h),
                                                      0 if h is None else h.size))

    def push_codes(self, codes) -> None:
        """Append codes (n_frames, n_q) to the accumulator."""
        c = _i32(codes)
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        if not lib().codec_audio_lm_push_codes(self.ptr, _iptr(c), c.shape[0], c.shape[1]):
            self._fail("codec_audio_lm_push_codes")

    # codebook step machine
    def step_set_text_context(self, text_token: int) -> None:
        if not lib().codec_audio_lm_step_set_text_context(self.ptr, int(text_token)):
            self._fail("step_set_text_context")

    def step_begin(self, hidden) -> None:
        h = _f32(hidden).reshape(-1)
        if not lib().codec_audio_lm_step_begin(self.ptr, _fptr(h), h.size):
            self._fail("step_begin")

    def step_logits(self) -> tuple[int, np.ndarray]:
        """(codebook index, logits) for the next code of the current frame."""
        cb, n = ctypes.c_int32(), ctypes.c_int32()
        p = lib().codec_audio_lm_step_logits(self.ptr, ctypes.byref(cb), ctypes.byref(n))
        if not p:
            self._fail("step_logits")
        return int(cb.value), np.ctypeslib.as_array(p, shape=(n.value,)).copy()

    def step_push_code(self, code: int) -> None:
        if not lib().codec_audio_lm_step_push_code(self.ptr, int(code)):
            self._fail("step_push_code")

    def step_finish(self) -> np.ndarray:
        """Close the frame; returns its n_codebook codes."""
        out = np.zeros(max(self.n_codebook, 1), np.int32)
        if not lib().codec_audio_lm_step_finish(self.ptr, _iptr(out), out.size):
            self._fail("step_finish")
        return out

    def step_frame(self, hidden, sample=None) -> np.ndarray:
        """Run one whole frame of the step machine: sample(logits, cb) -> code
        (default: argmax).  Returns the frame's codes."""
        self.step_begin(hidden)
        for _ in range(self.n_codebook):
            cb, logits = self.step_logits()
            self.step_push_code(int(np.argmax(logits)) if sample is None else int(sample(logits, cb)))
        return self.step_finish()

    # --- prompt / step embeddings --------------------------------------

    @property
    def needs_composed_prompt(self) -> bool:
        return bool(lib().codec_audio_lm_prompt_needs_composed_embd(self.ptr))

    def compose_prompt_embd(self, text_token: int) -> np.ndarray:
        out = np.zeros(self.hidden_dim, np.float32)
        if not lib().codec_audio_lm_compose_prompt_embd(self.ptr, int(text_token), _fptr(out), out.size):
            self._fail("compose_prompt_embd")
        return out

    def compose_audio_codes_embd(self, codes) -> np.ndarray:
        c = _i32(codes).reshape(-1)
        out = np.zeros(self.hidden_dim, np.float32)
        if not lib().codec_audio_lm_compose_audio_codes_embd(self.ptr, _iptr(c), c.size, _fptr(out), out.size):
            self._fail("compose_audio_codes_embd")
        return out

    @property
    def has_text_lane(self) -> bool:
        """True when the codec projects the text itself (prefix + per-step text rows)."""
        return bool(lib().codec_audio_lm_has_text_lane(self.ptr))

    def build_prefix_embd(self, header_tokens, text_tokens, speaker_emb=None) -> tuple[np.ndarray, int]:
        """Text-lane prompt prefix rows (rows, hidden) and how many text tokens they consumed."""
        h, t = _i32(header_tokens).reshape(-1), _i32(text_tokens).reshape(-1)
        x = None if speaker_emb is None else _f32(speaker_emb).reshape(-1)
        cap = h.size + 5 + t.size + 1
        out = np.zeros((cap, self.hidden_dim), np.float32)
        n_rows, used = ctypes.c_int32(), ctypes.c_int32()
        if not lib().codec_audio_lm_build_prefix_embd(
                self.ptr, _iptr(h), h.size, _iptr(t), t.size, _fptr(x), 0 if x is None else x.size,
                _fptr(out), cap, ctypes.byref(n_rows), ctypes.byref(used)):
            self._fail("build_prefix_embd")
        return out[:n_rows.value].copy(), int(used.value)

    def step_text_embd(self, text_tokens, step: int) -> np.ndarray:
        """Text-lane row for AR step ``step``, added to next_embed()."""
        t = _i32(text_tokens).reshape(-1)
        out = np.zeros(self.hidden_dim, np.float32)
        if not lib().codec_audio_lm_step_text_embd(self.ptr, _iptr(t), t.size, int(step), _fptr(out), out.size):
            self._fail("step_text_embd")
        return out

    # --- end of sequence -----------------------------------------------

    def decode(self) -> tuple[np.ndarray, int]:
        """Decode the accumulated codes / latents: (float32 PCM, sample rate)."""
        pcm = _ffi.codec_pcm_buffer()
        if not lib().codec_audio_lm_decode_audio(self.ptr, ctypes.byref(pcm)):
            self._fail("codec_audio_lm_decode_audio")
        return _take_pcm(pcm)

    def decode_latents(self, latents) -> tuple[np.ndarray, int]:
        """Decode host-collected latents (n_frames, dim) in the codec's layout (Soprano)."""
        x = _f32(latents)
        if x.ndim != 2:
            raise ValueError("latents must be (n_frames, dim)")
        pcm = _ffi.codec_pcm_buffer()
        if not lib().codec_audio_lm_decode_latents(self.ptr, _fptr(x), x.shape[0], x.shape[1], ctypes.byref(pcm)):
            self._fail("codec_audio_lm_decode_latents")
        return _take_pcm(pcm)
