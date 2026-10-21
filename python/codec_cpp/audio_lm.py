"""Audio-LM inference description (include/codec_audio_lm.h).

codec_audio_lm tells a host how to run an audio LM whose backbone is a
llama.cpp model: what to feed the backbone, which logits to sample (the
backbone's, or a codec-side head's) and with which suggested constraints,
when to stop, and it turns the generation into audio.  The host owns every
llama.cpp call and samples every token, so any llama.cpp binding works
(llama-cpp-python, your own).  For the ready-made loop use codec_cpp.TTS.

    alm = codec_cpp.AudioLM("codec.gguf")
    vocab = codec_cpp.Vocab.from_llama_cpp(llama_model)    # or Vocab(tokenize=..., ...)
    voice = alm.voice("alice")                             # or alm.create_voice(pcm, sr, ref_text)
    alm.tts_prompt("Hello there.", vocab=vocab, voice=voice)
    c = alm.completion()                                   # chunks, constraints, head groups, flags
    ...                                                    # decode c.chunks with llama.cpp
    while True:                                            # per step
        v = alm.observe(tokens=[tok] if sampled else None, hiddens=hidden)
        while v.action == codec_cpp.AudioLM.SAMPLE_HEAD:   # codec-side heads: the host draws
            v = alm.head_token(draw(group[v.head_group], alm.head_logits(v.n_vocab)))
        if v.action == codec_cpp.AudioLM.STOP: break
        ...                                                # feed v.token / alm.next_embd()
    audio, sr = alm.decode()

Arrays come back as numpy copies; strings as str.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from . import _ffi
from ._lib import lib
from .codec import CodecError, PathLike, Speaker, _f32, _i32, _make_kv_overrides, _str, _take_pcm


def _b(s) -> bytes | None:
    return None if s is None else os.fsencode(s) if isinstance(s, os.PathLike) else str(s).encode("utf-8")


class Vocab:
    """The backbone vocabulary, as codec_audio_lm looks it up (it never decodes).

    ``tokenize(text, add_special, parse_special) -> list[int]``,
    ``piece(token) -> str`` (special tokens rendered), ``n_tokens``, ``eos``
    and optionally ``is_eog(token) -> bool``.
    """

    def __init__(self, *, tokenize: Callable[[str, bool, bool], Sequence[int]], piece: Callable[[int], str],
                 n_tokens: int, eos: int = -1, is_eog: Callable[[int], bool] | None = None):
        self._tokenize, self._piece, self._is_eog = tokenize, piece, is_eog

        def tok_cb(_ud, text, n, out, cap, add_special, parse_special):
            try:
                ids = list(self._tokenize(ctypes.string_at(text, n).decode("utf-8", "replace"),
                                          bool(add_special), bool(parse_special)))
                if len(ids) > cap:
                    return -len(ids)
                for i, t in enumerate(ids):
                    out[i] = int(t)
                return len(ids)
            except Exception:
                return 0

        def piece_cb(_ud, token, buf, cap):
            try:
                b = self._piece(int(token)).encode("utf-8")
                if len(b) > cap:
                    return -len(b)
                ctypes.memmove(buf, b, len(b))
                return len(b)
            except Exception:
                return 0

        def eog_cb(_ud, token):
            try:
                return bool(self._is_eog(int(token)))
            except Exception:
                return False

        self._cbs = (_ffi.codec_audio_lm_tokenize_fn(tok_cb), _ffi.codec_audio_lm_piece_fn(piece_cb),
                     _ffi.codec_audio_lm_is_eog_fn(eog_cb) if is_eog else _ffi.codec_audio_lm_is_eog_fn())
        self.struct = _ffi.codec_audio_lm_vocab(None, self._cbs[0], self._cbs[1], int(n_tokens), int(eos),
                                                self._cbs[2])

    @classmethod
    def from_llama_cpp(cls, model) -> "Vocab":
        """From a llama-cpp-python model: a ``llama_cpp.Llama`` or a low-level ``llama_model_p``."""
        import llama_cpp as lc
        m = getattr(model, "model", model)
        m = getattr(m, "model", m)   # Llama -> _LlamaModel -> llama_model_p
        v = lc.llama_model_get_vocab(m)

        def tokenize(text, add_special, parse_special):
            b = text.encode("utf-8")
            buf = (lc.llama_token * (len(b) + 16))()
            n = lc.llama_tokenize(v, b, len(b), buf, len(buf), add_special, parse_special)
            if n < 0:
                buf = (lc.llama_token * -n)()
                n = lc.llama_tokenize(v, b, len(b), buf, len(buf), add_special, parse_special)
            return list(buf[:max(n, 0)])

        def piece(token):
            buf = ctypes.create_string_buffer(256)
            n = lc.llama_token_to_piece(v, token, buf, len(buf), 0, True)
            if n < 0:
                buf = ctypes.create_string_buffer(-n)
                n = lc.llama_token_to_piece(v, token, buf, len(buf), 0, True)
            return buf.raw[:max(n, 0)].decode("utf-8", "replace")

        return cls(tokenize=tokenize, piece=piece, n_tokens=lc.llama_vocab_n_tokens(v),
                   eos=lc.llama_vocab_eos(v), is_eog=lambda t: bool(lc.llama_vocab_is_eog(v, t)))


@dataclass
class Chunk:
    type: int                       # AudioLM.CHUNK_TEXT / CHUNK_TOKENS / CHUNK_EMBD
    text: str = ""
    add_special: bool = False
    parse_special: bool = False
    tokens: np.ndarray | None = None   # int32 (n,)
    embd: np.ndarray | None = None     # float32 (n_rows, n_embd)


@dataclass
class Completion:
    """How to run the backbone for the prepared prompt (codec_audio_lm_completion)."""
    flow: int
    n_seq: int
    n_embd: int
    chunks: list[list[Chunk]]       # per sequence, decoded in order (empty chunks are padding)
    penalty_exempt: list[int]       # backbone tokens the host's penalties should not touch
    grammar: str
    allowed: list[tuple[int, int]]  # [begin, end) backbone token ranges; empty = all
    needs_embeddings: bool
    needs_prompt_hidden: bool
    sample_first: bool
    streamable: bool
    max_frames: int
    head_groups: list[dict] = field(default_factory=list)   # [{penalties, history, penalty_exempt}]


@dataclass
class Verdict:
    action: int                     # AudioLM.FEED_TOKEN / FEED_EMBD / STOP
    sample_backbone: bool
    is_audio: bool
    token: int                      # FEED_TOKEN: the token the model picked (-1 = the sampled one)
    head_group: int = 0             # SAMPLE_HEAD: the group to sample with
    n_vocab: int = 0                # SAMPLE_HEAD: the head's logits width


@dataclass
class Caps:
    modality: int
    needs_backbone: bool
    needs_backbone_path: bool
    text_needs_phonemes: bool
    takes_ref_audio: bool
    takes_ref_text: bool
    takes_instruct: bool
    needs_voice: bool
    sample_rate: int
    segment_bytes: int


class AudioLM:
    """A codec GGUF opened for audio-LM inference (one generation at a time)."""

    CHUNK_TEXT, CHUNK_TOKENS, CHUNK_EMBD = (_ffi.CODEC_AUDIO_LM_CHUNK_TEXT, _ffi.CODEC_AUDIO_LM_CHUNK_TOKENS,
                                            _ffi.CODEC_AUDIO_LM_CHUNK_EMBD)
    FLOW_TOKENS, FLOW_CODEC_HEAD, FLOW_HIDDEN, FLOW_SELF_CONTAINED = (
        _ffi.CODEC_AUDIO_LM_FLOW_TOKENS, _ffi.CODEC_AUDIO_LM_FLOW_CODEC_HEAD, _ffi.CODEC_AUDIO_LM_FLOW_HIDDEN,
        _ffi.CODEC_AUDIO_LM_FLOW_SELF_CONTAINED)
    FEED_TOKEN, FEED_EMBD, STOP, SAMPLE_HEAD = (_ffi.CODEC_AUDIO_LM_FEED_TOKEN, _ffi.CODEC_AUDIO_LM_FEED_EMBD,
                                                _ffi.CODEC_AUDIO_LM_STOP, _ffi.CODEC_AUDIO_LM_SAMPLE_HEAD)

    def __init__(self, codec: PathLike, *, backbone_path: PathLike | None = None, use_gpu: bool = False,
                 n_threads: int = 0, kv_overrides: Mapping[str, object] | None = None):
        L = lib()
        p = L.codec_audio_lm_default_params()
        p.codec_path = os.fsencode(codec)
        p.backbone_path = None if backbone_path is None else os.fsencode(backbone_path)
        p.use_gpu, p.n_threads = bool(use_gpu), int(n_threads)
        self._kv = None
        if kv_overrides:
            self._kv = _make_kv_overrides(kv_overrides)
            p.kv_overrides, p.n_kv_overrides = self._kv, len(kv_overrides)
        err = ctypes.create_string_buffer(2048)
        self.ptr = L.codec_audio_lm_init(ctypes.byref(p), err, len(err))
        if not self.ptr:
            raise CodecError("codec_audio_lm_init", None, _str(err.value))
        self._keep: list = []   # buffers the library reads during the current generation

    def close(self) -> None:
        if getattr(self, "ptr", None):
            lib().codec_audio_lm_free(self.ptr)
            self.ptr = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()

    def _fail(self, what: str):
        raise CodecError(what, None, _str(lib().codec_audio_lm_last_error(self.ptr)))

    # --- model ----------------------------------------------------------

    def meta(self, key: str) -> str | None:
        v = lib().codec_audio_lm_meta_str(self.ptr, key.encode())
        return None if v is None else _str(v)

    @property
    def caps(self) -> Caps:
        c = _ffi.codec_audio_lm_caps()
        lib().codec_audio_lm_get_caps(self.ptr, ctypes.byref(c))
        return Caps(int(c.modality), c.needs_backbone, c.needs_backbone_path, c.text_needs_phonemes,
                    c.takes_ref_audio, c.takes_ref_text, c.takes_instruct, c.needs_voice, int(c.sample_rate),
                    int(c.segment_bytes))

    def voices(self, speaker_file: PathLike | None = None) -> list[dict]:
        """Builtin voices, or those in a speaker file: [{name, ref_text, is_default}]."""
        L = lib()
        n = L.codec_audio_lm_list_voices(self.ptr, _b(speaker_file))
        if n < 0:
            self._fail("codec_audio_lm_list_voices")
        out = []
        for i in range(n):
            name, ref, d = ctypes.c_char_p(), ctypes.c_char_p(), ctypes.c_bool()
            L.codec_audio_lm_voice(self.ptr, i, ctypes.byref(name), ctypes.byref(ref), ctypes.byref(d))
            out.append({"name": _str(name.value), "ref_text": _str(ref.value), "is_default": bool(d.value)})
        return out

    def backbone_requirements(self, vocab: Vocab | None = None, max_frames: int = 0) -> dict:
        """llama.cpp context settings: n_ctx, n_ubatch (0 = n_ctx), n_seq_max, embeddings, flash_attn."""
        r = _ffi.codec_audio_lm_backbone_req()
        if not lib().codec_audio_lm_backbone_requirements(self.ptr, ctypes.byref(vocab.struct) if vocab else None,
                                                          int(max_frames), ctypes.byref(r)):
            self._fail("codec_audio_lm_backbone_requirements")
        return {"n_ctx": r.n_ctx, "n_ubatch": r.n_ubatch, "n_seq_max": r.n_seq_max,
                "embeddings": bool(r.embeddings), "flash_attn": bool(r.flash_attn)}

    # --- text to speech ---------------------------------------------------

    # --- voices -----------------------------------------------------------
    # A voice is a codec_cpp.Speaker carrying all of the model's voice
    # conditioning; pass it to tts_prompt / decode_generation.

    def voice(self, name: str | None = None) -> Speaker:
        """A builtin voice (None = the model's default), borrowed from the model."""
        p = lib().codec_audio_lm_get_voice(self.ptr, _b(name))
        if not p:
            self._fail("codec_audio_lm_get_voice")
        return Speaker(p, owner=self)

    def create_voice(self, pcm, sample_rate: int, ref_text: str | None = None) -> Speaker:
        """Clone a voice from a reference clip (mono float, any rate) and its transcript."""
        x = _f32(pcm).reshape(-1)
        p = lib().codec_audio_lm_create_voice(self.ptr, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x.size,
                                              int(sample_rate), _b(ref_text))
        if not p:
            self._fail("codec_audio_lm_create_voice")
        return Speaker(p, owned=True)

    def load_voice(self, path: PathLike, name: str | None = None) -> Speaker:
        p = lib().codec_audio_lm_load_voice(self.ptr, os.fsencode(path), _b(name))
        if not p:
            self._fail("codec_audio_lm_load_voice")
        return Speaker(p, owned=True)

    def save_voice(self, voice: Speaker, path: PathLike, name: str = "voice") -> None:
        if not lib().codec_audio_lm_save_voice(self.ptr, voice.ptr, os.fsencode(path), name.encode("utf-8")):
            self._fail("codec_audio_lm_save_voice")

    def tts_prompt(self, text: str, *, vocab: Vocab | None = None, instruct: str | None = None,
                   seed: int | None = None, min_frames: int | None = None, max_frames: int | None = None,
                   lm_guidance: float | None = None, flow_guidance: float | None = None,
                   flow_steps: int | None = None, voice: Speaker | None = None) -> None:
        """Prepare the prompt of one generation.  Sampling is the host's (every
        token, codec-side heads included); ``seed`` only drives the model's own
        randomness (flow init noise).  Decoder settings left None are the GGUF's;
        voice None = default."""
        L = lib()
        r = L.codec_audio_lm_default_tts_request()
        r.text, r.instruct = text.encode("utf-8"), _b(instruct)
        r.voice = voice.ptr if voice is not None else None
        self._voice = voice   # the library borrows it until the generation is decoded
        if seed is not None:
            r.seed = int(seed) & 0xFFFFFFFF
        for name, val, conv in (("min_frames", min_frames, int), ("max_frames", max_frames, int),
                                ("lm_guidance", lm_guidance, float), ("flow_guidance", flow_guidance, float),
                                ("flow_steps", flow_steps, int)):
            if val is not None:
                setattr(r, name, conv(val))
        if not L.codec_audio_lm_tts_prompt(self.ptr, ctypes.byref(r), ctypes.byref(vocab.struct) if vocab else None):
            self._fail("codec_audio_lm_tts_prompt")

    def split(self, text: str, max_bytes: int = 0) -> list[str]:
        """The model's segmentation of long text (one generation per segment)."""
        L = lib()
        n = L.codec_audio_lm_tts_split(self.ptr, text.encode("utf-8"), int(max_bytes))
        if n < 0:
            self._fail("codec_audio_lm_tts_split")
        return [_str(L.codec_audio_lm_tts_segment(self.ptr, i)) for i in range(n)]

    def join(self, pcm: np.ndarray, sample_rate: int, i: int, n: int) -> tuple[np.ndarray, int]:
        """Segment i of n's audio prepared for joining, and the samples of silence to put after it."""
        L = lib()
        x = _f32(pcm)
        nch = 1 if x.ndim == 1 else x.shape[1]
        buf = _ffi.codec_pcm_buffer()
        # The library resizes the buffer in place (malloc / free): hand it a malloc'd copy.
        libc = ctypes.CDLL(None)
        libc.malloc.restype = ctypes.c_void_p
        libc.malloc.argtypes = [ctypes.c_size_t]
        p = libc.malloc(max(1, x.size) * 4)
        ctypes.memmove(p, x.ctypes.data, x.size * 4)
        buf.data = ctypes.cast(p, ctypes.POINTER(ctypes.c_float))
        buf.n_samples, buf.n_channels, buf.sample_rate = x.size // nch, nch, int(sample_rate)
        gap = L.codec_audio_lm_tts_join(self.ptr, ctypes.byref(buf), int(i), int(n))
        out, _ = _take_pcm(buf)
        if gap < 0:
            self._fail("codec_audio_lm_tts_join")
        return out, int(gap)

    # --- completion and steps ----------------------------------------------

    def completion(self) -> Completion:
        L = lib()
        c = _ffi.codec_audio_lm_completion()
        if not L.codec_audio_lm_get_completion(self.ptr, ctypes.byref(c)):
            self._fail("codec_audio_lm_get_completion")
        seqs: list[list[Chunk]] = []
        for q in range(max(1, c.n_seq) if c.n_chunks else 0):
            row = []
            for i in range(c.n_chunks):
                ch = c.chunks[q * c.n_chunks + i]
                k = Chunk(int(ch.type), add_special=ch.add_special, parse_special=ch.parse_special)
                if ch.type == self.CHUNK_TEXT:
                    k.text = _str(ch.text)
                elif ch.type == self.CHUNK_TOKENS:
                    k.tokens = np.ctypeslib.as_array(ch.tokens, shape=(ch.n,)).copy() if ch.n else np.zeros(0, np.int32)
                elif ch.n:
                    k.embd = np.ctypeslib.as_array(ch.embd, shape=(ch.n, c.n_embd)).copy()
                else:
                    k.embd = np.zeros((0, c.n_embd), np.float32)
                row.append(k)
            seqs.append(row)
        def ids(p, n):
            return [p[i] for i in range(n)] if n else []
        groups = [{"penalties": bool(g.penalties), "history": ids(g.history, g.n_history),
                   "penalty_exempt": ids(g.penalty_exempt, g.n_penalty_exempt)}
                  for g in (c.head_groups[i] for i in range(c.n_head_groups))]
        return Completion(int(c.flow), int(c.n_seq), int(c.n_embd), seqs,
                          ids(c.penalty_exempt, c.n_penalty_exempt), _str(c.grammar) if c.grammar else "",
                          [(c.allowed[i].begin, c.allowed[i].end) for i in range(c.n_allowed)],
                          c.needs_embeddings, c.needs_prompt_hidden, c.sample_first, c.streamable,
                          int(c.max_frames), groups)

    def prefill(self, hiddens: np.ndarray) -> None:
        """Every prompt position's hidden state, (n_pos, n_embd) (when needs_prompt_hidden)."""
        h = _f32(hiddens)
        if not lib().codec_audio_lm_prefill(self.ptr, h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), h.shape[0]):
            self._fail("codec_audio_lm_prefill")

    def observe(self, tokens: Sequence[int] | None = None, hiddens: np.ndarray | None = None) -> Verdict:
        """One step: the sampled backbone token per sequence (None when none was sampled)
        and the last position's hidden state per sequence, (n_seq, n_embd) or (n_embd,)."""
        t = None if tokens is None else _i32(np.asarray(tokens, dtype=np.int32).reshape(-1))
        h = None if hiddens is None else _f32(hiddens).reshape(-1)
        v = _ffi.codec_audio_lm_verdict()
        if not lib().codec_audio_lm_observe(self.ptr,
                                            None if t is None else t.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                            None if h is None else h.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            ctypes.byref(v)):
            self._fail("codec_audio_lm_observe")
        return Verdict(int(v.action), bool(v.sample_backbone), bool(v.is_audio), int(v.token), int(v.head_group),
                       int(v.n_vocab))

    def head_logits(self, n_vocab: int) -> np.ndarray:
        """After a SAMPLE_HEAD verdict: the head's logits (constraints applied), a view."""
        p = lib().codec_audio_lm_head_logits(self.ptr)
        if not p:
            self._fail("codec_audio_lm_head_logits")
        return np.ctypeslib.as_array(p, shape=(n_vocab,))

    def head_token(self, token: int) -> Verdict:
        """The id the host drew from head_logits(); the next verdict."""
        v = _ffi.codec_audio_lm_verdict()
        if not lib().codec_audio_lm_head_token(self.ptr, int(token), ctypes.byref(v)):
            self._fail("codec_audio_lm_head_token")
        return Verdict(int(v.action), bool(v.sample_backbone), bool(v.is_audio), int(v.token), int(v.head_group),
                       int(v.n_vocab))

    def next_embd(self, seq: int = 0, n_embd: int | None = None) -> np.ndarray:
        """The next input row of sequence ``seq`` after FEED_EMBD."""
        if n_embd is None:
            n_embd = self._n_embd()
        p = lib().codec_audio_lm_next_embd(self.ptr, int(seq))
        if not p:
            self._fail("codec_audio_lm_next_embd")
        return np.ctypeslib.as_array(p, shape=(n_embd,)).copy()

    def _n_embd(self) -> int:
        c = _ffi.codec_audio_lm_completion()
        lib().codec_audio_lm_get_completion(self.ptr, ctypes.byref(c))
        return int(c.n_embd)

    @property
    def stop_reason(self) -> str:
        return _str(lib().codec_audio_lm_stop_reason(self.ptr))

    def generate_self_contained(self) -> None:
        if not lib().codec_audio_lm_generate_self_contained(self.ptr):
            self._fail("codec_audio_lm_generate_self_contained")

    # --- audio ----------------------------------------------------------------

    def decode(self) -> tuple[np.ndarray, int]:
        """The generation's audio: (float32 PCM, sample rate)."""
        buf = _ffi.codec_pcm_buffer()
        if not lib().codec_audio_lm_decode(self.ptr, ctypes.byref(buf)):
            self._fail("codec_audio_lm_decode")
        return _take_pcm(buf)

    def generation(self) -> np.ndarray:
        """The generation's decoder input, a copy: int32 codes or float32 latents, [n_frames, width]."""
        g = _ffi.codec_audio_lm_generation()
        if not lib().codec_audio_lm_get_generation(self.ptr, ctypes.byref(g)):
            self._fail("codec_audio_lm_get_generation")
        ptr = g.latents if g.kind == _ffi.CODEC_AUDIO_LM_DATA_LATENTS else g.codes
        return np.ctypeslib.as_array(ptr, shape=(g.n_frames, g.width)).copy()

    def decode_generation(self, gen: np.ndarray, voice: Speaker | None = None) -> tuple[np.ndarray, int]:
        """Audio of a generation() result (this model's), with the voice it was made with (None = default)."""
        g = _ffi.codec_audio_lm_generation()
        g.n_frames, g.width = int(gen.shape[0]), int(gen.shape[1])
        if np.issubdtype(gen.dtype, np.floating):
            arr = _f32(gen)
            g.kind, g.latents = _ffi.CODEC_AUDIO_LM_DATA_LATENTS, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            arr = _i32(gen)
            g.kind, g.codes = _ffi.CODEC_AUDIO_LM_DATA_CODES, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        buf = _ffi.codec_pcm_buffer()
        if not lib().codec_audio_lm_decode_generation(self.ptr, ctypes.byref(g), voice.ptr if voice else None,
                                                      ctypes.byref(buf)):
            self._fail("codec_audio_lm_decode_generation")
        return _take_pcm(buf)

    def stream_poll(self, chunk_ms: int = 1000) -> tuple[np.ndarray, int]:
        """Audio that became final since the last poll (possibly empty); call after audio steps."""
        buf = _ffi.codec_pcm_buffer()
        if not lib().codec_audio_lm_stream_poll(self.ptr, int(chunk_ms), ctypes.byref(buf)):
            self._fail("codec_audio_lm_stream_poll")
        return _take_pcm(buf)

    def stream_finish(self) -> tuple[np.ndarray, int]:
        buf = _ffi.codec_pcm_buffer()
        if not lib().codec_audio_lm_stream_finish(self.ptr, ctypes.byref(buf)):
            self._fail("codec_audio_lm_stream_finish")
        return _take_pcm(buf)

    @property
    def decodable_frames(self) -> int:
        return int(lib().codec_audio_lm_decodable_frames(self.ptr))
