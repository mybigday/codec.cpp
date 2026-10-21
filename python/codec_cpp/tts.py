"""One-call text to speech: the codec_audio_lm host loop in Python.

    tts = codec_cpp.TTS("model.gguf", "codec.gguf")
    audio, sr = tts.synthesize("Hello there.")
    audio, sr = tts.synthesize("Hello there.", ref_audio="me.wav")   # voice clone

The same loop as tts-cli / tts-server (common/tts_runner.cpp): the backbone
(model file) runs in the llama.cpp bundled with codec.cpp (libttsbackbone,
with its architecture / tokenizer patches) through ctypes, the codec file in
codec.cpp through codec_cpp.AudioLM.  For a loop around your own llama.cpp,
drive codec_cpp.AudioLM directly.
"""

from __future__ import annotations

import ctypes
import os
import wave
from typing import Mapping

import numpy as np

from . import _ffi
from ._lib import backbone_lib
from .audio_lm import AudioLM, Vocab
from .codec import CodecError, PathLike, Speaker

# The backbone GGUF's recommended sampling (general.sampling.*, as llama.cpp's
# common_init_sampler_from_model reads it), by codec_audio_lm_sampler name.
_GGUF_SAMPLING = {"samplers": "sequence", "temp": "temp", "top_k": "top_k", "top_p": "top_p", "min_p": "min_p",
                  "penalty_last_n": "penalty_last_n", "penalty_repeat": "penalty_repeat",
                  "penalty_freq": "penalty_freq", "penalty_present": "penalty_present",
                  "dry_multiplier": "dry_multiplier", "dry_base": "dry_base",
                  "dry_allowed_length": "dry_allowed_length", "dry_penalty_last_n": "dry_penalty_last_n"}


def _read_wav(path: PathLike) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV as mono float32."""
    with wave.open(os.fspath(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        return x.reshape(-1, w.getnchannels()).mean(axis=1), w.getframerate()


class _Backbone:
    """A llama.cpp model + context sized by codec_audio_lm_backbone_requirements."""

    def __init__(self, path: PathLike, alm: AudioLM, *, use_gpu: bool, n_threads: int, max_frames: int):
        L = self.L = backbone_lib()
        mp = L.llama_model_default_params()
        mp.n_gpu_layers = 999 if use_gpu else 0
        if not use_gpu:   # an empty device list, else llama.cpp still offloads batched ops to a GPU
            self._no_devices = (ctypes.c_void_p * 1)()
            mp.devices = ctypes.cast(self._no_devices, ctypes.c_void_p)
        self.model = L.llama_model_load_from_file(os.fsencode(path), mp)
        if not self.model:
            raise CodecError("llama_model_load_from_file", None, os.fspath(path))
        self.ctx = None
        self.vocab_ptr = L.llama_model_get_vocab(self.model)
        self.vocab = self._make_vocab()
        req = alm.backbone_requirements(self.vocab, max_frames)
        cp = L.llama_context_default_params()
        cp.n_ctx = cp.n_batch = req["n_ctx"]
        cp.n_ubatch = req["n_ubatch"] or req["n_ctx"]
        cp.n_seq_max = max(1, req["n_seq_max"])
        cp.embeddings = req["embeddings"]
        cp.pooling_type = _ffi.LLAMA_POOLING_TYPE_NONE
        if not req["flash_attn"]:
            cp.flash_attn_type = _ffi.LLAMA_FLASH_ATTN_TYPE_DISABLED
        if n_threads > 0:
            cp.n_threads = cp.n_threads_batch = n_threads
        cp.op_offload = use_gpu
        self.ctx = L.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise CodecError("llama_init_from_model", None, os.fspath(path))
        self.n_ctx = int(L.llama_n_ctx(self.ctx))
        self.n_vocab = int(L.llama_vocab_n_tokens(self.vocab_ptr))
        self.n_embd = int(L.llama_model_n_embd(self.model))   # hidden width (completion.n_embd is the input's)

    def _make_vocab(self) -> Vocab:
        L, v = self.L, self.vocab_ptr

        def piece(token):
            buf = ctypes.create_string_buffer(256)
            n = L.llama_token_to_piece(v, token, buf, len(buf), 0, True)
            if n < 0:
                buf = ctypes.create_string_buffer(-n)
                n = L.llama_token_to_piece(v, token, buf, len(buf), 0, True)
            return buf.raw[:max(n, 0)].decode("utf-8", "replace")

        return Vocab(tokenize=self.tokenize, piece=piece, n_tokens=L.llama_vocab_n_tokens(v),
                     eos=L.llama_vocab_eos(v), is_eog=lambda t: bool(L.llama_vocab_is_eog(v, t)))

    def tokenize(self, text: str, add_special: bool, parse_special: bool) -> list[int]:
        b = text.encode("utf-8")
        buf = (_ffi.llama_token * (len(b) + 8))()
        n = self.L.llama_tokenize(self.vocab_ptr, b, len(b), buf, len(buf), add_special, parse_special)
        if n < 0:
            buf = (_ffi.llama_token * -n)()
            n = self.L.llama_tokenize(self.vocab_ptr, b, len(b), buf, len(buf), add_special, parse_special)
        return list(buf[:max(n, 0)])

    def sampling(self) -> dict:
        """The general.sampling.* settings this GGUF carries."""
        out = {}
        buf = ctypes.create_string_buffer(512)
        for name, key in _GGUF_SAMPLING.items():
            if self.L.llama_model_meta_val_str(self.model, f"general.sampling.{key}".encode(), buf, len(buf)) > 0:
                v = buf.value.decode()
                out[name] = v if name == "samplers" else (int(v) if name in ("top_k", "penalty_last_n",
                                                                           "dry_allowed_length",
                                                                           "dry_penalty_last_n") else float(v))
        return out

    def clear(self) -> None:
        self.L.llama_memory_clear(self.L.llama_get_memory(self.ctx), True)

    def decode_tokens(self, tokens, pos: int, all_out: bool = False) -> None:
        n = len(tokens)
        b = self.L.llama_batch_init(n, 0, 1)
        try:
            for i, t in enumerate(tokens):
                b.token[i], b.pos[i], b.n_seq_id[i] = int(t), pos + i, 1
                b.seq_id[i][0] = 0
                b.logits[i] = 1 if (all_out or i == n - 1) else 0
            b.n_tokens = n
            if self.L.llama_decode(self.ctx, b) != 0:
                raise RuntimeError("llama_decode failed")
        finally:
            self.L.llama_batch_free(b)

    def decode_embd(self, rows: np.ndarray, pos: int) -> None:
        """rows (n_seq, n, n_embd): sequence q on seq_id q at positions pos..; outputs on each last row."""
        n_seq, n, dim = rows.shape
        if n == 0:
            return
        b = self.L.llama_batch_init(n_seq * n, dim, 1)
        try:
            e = np.ascontiguousarray(rows, np.float32)
            ctypes.memmove(b.embd, e.ctypes.data, e.nbytes)
            i = 0
            for q in range(n_seq):
                for r in range(n):
                    b.pos[i], b.n_seq_id[i] = pos + r, 1
                    b.seq_id[i][0] = q
                    b.logits[i] = 1 if r == n - 1 else 0
                    i += 1
            b.n_tokens = n_seq * n
            if self.L.llama_decode(self.ctx, b) != 0:
                raise RuntimeError("llama_decode failed")
        finally:
            self.L.llama_batch_free(b)

    def hidden(self, i: int) -> np.ndarray:
        p = self.L.llama_get_embeddings_ith(self.ctx, i)
        if not p:
            raise RuntimeError("the backbone returned no hidden state")
        return np.ctypeslib.as_array(p, shape=(self.n_embd,)).copy()

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.L.llama_free(self.ctx)
            self.ctx = None
        if getattr(self, "model", None):
            self.L.llama_model_free(self.model)
            self.model = None


class _Sampler:
    """A llama.cpp sampler chain as common_sampler builds it from common_params_sampling:
    logit bias, the stages in order, dist.  A grammar constrains the chain first."""

    _STAGES = {"penalties", "dry", "top_k", "top_p", "nucleus", "min_p", "typ_p", "typical_p", "typical",
               "temperature", "temp", "top_n_sigma", "xtc"}
    _DEFAULT = ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]

    def __init__(self, bb: _Backbone, sp: dict, *, grammar: str = "", bias: list[int] = ()):
        L = self.L = bb.L
        self.chain = L.llama_sampler_chain_init(L.llama_sampler_chain_default_params())
        self.penalties = None
        if grammar:
            g = L.llama_sampler_init_grammar(bb.vocab_ptr, grammar.encode("utf-8"), b"root")
            if not g:
                self.free()
                raise ValueError("the backbone grammar does not parse")
            L.llama_sampler_chain_add(self.chain, g)
        if len(bias):
            arr = (_ffi.llama_logit_bias * len(bias))(*[_ffi.llama_logit_bias(int(t), float("-inf")) for t in bias])
            L.llama_sampler_chain_add(self.chain, L.llama_sampler_init_logit_bias(bb.n_vocab, len(bias), arr))
        order = [s for s in sp["samplers"].split(";") if s] if sp["samplers"] else self._DEFAULT
        bad = [s for s in order if s not in self._STAGES]
        if bad:
            self.free()
            raise ValueError(f"unknown backbone sampler {bad[0]!r}")
        # llama.cpp's penalties clamp -1 to "off"; -1 ("the whole context") is the context size.
        last_n = bb.n_ctx if sp["penalty_last_n"] < 0 else sp["penalty_last_n"]
        for name in order:
            if name == "penalties":
                ex = list(sp["penalty_exempt"])
                exa = (_ffi.llama_token * max(1, len(ex)))(*ex)
                s = self.penalties = L.llama_sampler_init_penalties_exempt(
                    last_n, sp["penalty_repeat"], sp["penalty_freq"], sp["penalty_present"], exa, len(ex))
            elif name == "dry":
                s = L.llama_sampler_init_dry(bb.vocab_ptr, L.llama_model_n_ctx_train(bb.model), sp["dry_multiplier"],
                                             sp["dry_base"], sp["dry_allowed_length"], sp["dry_penalty_last_n"],
                                             None, 0)
            elif name == "top_k":
                s = L.llama_sampler_init_top_k(sp["top_k"])
            elif name in ("top_p", "nucleus"):
                s = L.llama_sampler_init_top_p(sp["top_p"], 0)
            elif name == "min_p":
                s = L.llama_sampler_init_min_p(sp["min_p"], 0)
            elif name in ("temperature", "temp"):
                s = L.llama_sampler_init_temp_ext(sp["temp"], 0.0, 1.0)
            else:
                continue   # typical / top-n-sigma / XTC: off at llama.cpp's defaults
            L.llama_sampler_chain_add(self.chain, s)
        L.llama_sampler_chain_add(self.chain, L.llama_sampler_init_dist(sp["seed"] & 0xFFFFFFFF))

    def accept_prompt(self, tokens) -> None:
        """Prompt tokens count for the penalties (not the grammar), as llama-server's init_sampler."""
        if self.penalties:
            for t in tokens:
                self.L.llama_sampler_accept(self.penalties, int(t))

    def sample(self, ctx) -> int:
        # llama_sampler_sample accepts the token into the whole chain (penalties, grammar).
        return int(self.L.llama_sampler_sample(self.chain, ctx, -1))

    def free(self) -> None:
        if self.chain:
            self.L.llama_sampler_free(self.chain)
            self.chain = None


class TTS:
    """A loaded TTS model: a llama.cpp backbone GGUF + a codec GGUF.

    ``model`` is None for self-contained models.  Load-time settings:
    ``use_gpu``, ``n_threads``, ``max_frames`` (longest generation, sizes the
    backbone context; 0 = model default) and ``kv_overrides`` (codec GGUF
    metadata).  ``audio_lm`` is the underlying codec_cpp.AudioLM.  Calls are
    not thread-safe; use one TTS per thread or serialize them.
    """

    def __init__(self, model: PathLike | None, codec: PathLike, *, use_gpu: bool = False,
                 n_threads: int = 0, max_frames: int = 0,
                 kv_overrides: Mapping[str, object] | None = None):
        self.audio_lm = AudioLM(codec, backbone_path=model, use_gpu=use_gpu, n_threads=n_threads,
                                kv_overrides=kv_overrides)
        self.max_frames = int(max_frames)
        self._bb = None
        self.last_result: dict = {}
        if self.audio_lm.caps.needs_backbone:
            if model is None:
                self.close()
                raise ValueError("this model needs its llama.cpp backbone GGUF (model=)")
            self._bb = _Backbone(model, self.audio_lm, use_gpu=use_gpu, n_threads=n_threads,
                                 max_frames=self.max_frames)

    def close(self) -> None:
        if getattr(self, "_bb", None):
            self._bb.close()
            self._bb = None
        if getattr(self, "audio_lm", None):
            self.audio_lm.close()
            self.audio_lm = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()

    # --- voices --------------------------------------------------------

    def voices(self, speaker_file: PathLike | None = None) -> list[str]:
        """Names ``voice=`` accepts: the builtin voices, or those in a speaker file."""
        return [v["name"] for v in self.audio_lm.voices(speaker_file)]

    @property
    def default_voice(self) -> str:
        return next((v["name"] for v in self.audio_lm.voices() if v["is_default"]), "")

    @property
    def takes_instruct(self) -> bool:
        """The model takes ``instruct=``."""
        return self.audio_lm.caps.takes_instruct

    def _voice(self, opts: dict) -> Speaker | None:
        """speaker_file > ref_audio (models that clone) > voice > the model's default."""
        alm = self.audio_lm
        voice, speaker_file = opts.pop("voice", None), opts.pop("speaker_file", None)
        ref_audio, ref_text = opts.pop("ref_audio", None), opts.pop("ref_text", None)
        save = opts.pop("save_speaker", None)
        if isinstance(voice, Speaker):
            spk = voice
        elif speaker_file is not None:
            spk = alm.load_voice(speaker_file, voice)
        elif ref_audio is not None and alm.caps.takes_ref_audio:
            pcm, sr = ref_audio if isinstance(ref_audio, tuple) else _read_wav(ref_audio)
            spk = alm.create_voice(pcm, int(sr), ref_text)
        else:
            if ref_audio is not None:
                import logging
                logging.getLogger("codec_cpp").warning(
                    "voice: this model cannot clone from reference audio; using the %s voice", voice or "default")
            spk = alm.voice(voice) if voice is not None else None
        if save is not None:
            alm.save_voice(spk if spk is not None else alm.voice(), save, "saved")
        return spk

    # --- synthesis -----------------------------------------------------

    def synthesize(self, text: str, **opts) -> tuple[np.ndarray, int]:
        """Synthesize ``text``: returns (float32 PCM, sample rate).

        Voice: ``speaker_file`` > ``ref_audio`` (clone; a WAV path or a
        ``(pcm, sample_rate)`` tuple, with ``ref_text`` its transcript) >
        ``voice`` (builtin name, or a codec_cpp.Speaker) > model default.
        Sampling: llama.cpp's settings by their names (``seed``, ``temp``,
        ``top_k``, ``top_p``, ``min_p``, ``penalty_last_n``,
        ``penalty_repeat``, ``penalty_freq``, ``penalty_present``,
        ``dry_*``, ``samplers``, ``penalty_exempt``); unset ones come from
        the backbone GGUF (general.sampling.*), then llama.cpp's defaults.
        Other options: ``instruct``, ``grammar``, ``min_frames``,
        ``lm_guidance``, ``flow_guidance``, ``flow_steps`` (decoder settings,
        unset = the codec GGUF's), ``segment_bytes``, ``save_speaker``.
        """
        pieces = list(self._run(text, opts, 0))
        if not pieces:
            return np.zeros(0, np.float32), self.audio_lm.caps.sample_rate
        sr = pieces[0][1]
        return np.concatenate([p for p, _ in pieces]), sr

    def stream(self, text: str, *, chunk_ms: int = 0, **opts):
        """Like synthesize, but yields (float32 PCM piece, sample rate) as the
        audio becomes final: per segment, or with ``chunk_ms`` > 0 about that
        much audio at a time while generating (models that support it)."""
        yield from self._run(text, opts, int(chunk_ms))

    def _run(self, text: str, opts: dict, chunk_ms: int):
        alm, opts = self.audio_lm, dict(opts)
        segment_bytes = int(opts.pop("segment_bytes", 0) or 0)
        grammar = opts.pop("grammar", None) or ""
        spk = self._voice(opts)
        sampling = AudioLM.default_sampling()
        if self._bb is not None:
            sampling.update(self._bb.sampling())
        sampling.update({k: opts.pop(k) for k in list(opts) if k in AudioLM.SAMPLING and opts[k] is not None})
        known = {"instruct", "min_frames", "lm_guidance", "flow_guidance", "flow_steps"}
        if set(opts) - known:
            raise TypeError(f"unknown synthesis option(s): {', '.join(sorted(set(opts) - known))}")
        segments = alm.split(text, segment_bytes)
        if not segments:
            raise ValueError("empty text")
        self.last_result = {"n_frames": 0, "stop_reason": ""}
        base_seed = int(sampling["seed"]) & 0xFFFFFFFF
        for i, seg in enumerate(segments):
            if base_seed != _ffi.CODEC_AUDIO_LM_DEFAULT_SEED:   # a random seed stays random
                sampling["seed"] = (base_seed + i) & 0xFFFFFFFF
            alm.tts_prompt(seg, vocab=self._bb.vocab if self._bb else None, voice=spk,
                           max_frames=self.max_frames or None, **opts, **sampling)
            c = None if self._bb is None else alm.completion()
            if c is None or c.flow == AudioLM.FLOW_SELF_CONTAINED:
                alm.generate_self_contained()
                frames, streamed = alm.decodable_frames, None
            else:
                frames, streamed = yield from self._generate(c, grammar, chunk_ms)
            self.last_result["n_frames"] += frames
            self.last_result["stop_reason"] = alm.stop_reason if streamed != "n_ctx" else "n_ctx"
            if streamed is True:
                continue   # went out as it was generated
            pcm, sr = alm.decode()
            pcm, gap = alm.join(pcm, sr, i, len(segments))   # the model's joining: edges, then silence
            if gap > 0:
                pcm = np.concatenate([pcm, np.zeros((gap,) + pcm.shape[1:], np.float32)])
            if len(pcm):
                yield pcm, sr

    def _generate(self, c, grammar: str, chunk_ms: int):
        """One segment's backbone loop.  Returns (audio frames, True when streamed / "n_ctx")."""
        alm, bb = self.audio_lm, self._bb
        n_seq = max(1, c.n_seq)
        if n_seq > 1 and c.sample_first:
            raise RuntimeError("backbone sampling over several sequences is not supported")
        allowed = np.zeros(bb.n_vocab, bool) if c.allowed else None
        for b, e in c.allowed:
            allowed[max(0, b):min(bb.n_vocab, e)] = True

        def bias(except_tok: int = -1) -> list[int]:
            if allowed is None:
                return []
            ok = allowed.copy()
            if 0 <= except_tok < bb.n_vocab:
                ok[except_tok] = False
            return np.nonzero(~ok)[0].tolist()

        smpl = _Sampler(bb, c.sampler, grammar=grammar or c.grammar, bias=bias())
        try:
            bb.clear()   # every generation starts from an empty context
            pos, prompt_hidden = 0, []
            for i, ch in enumerate(c.chunks[0]):
                if ch.type == AudioLM.CHUNK_EMBD:
                    if not len(ch.embd):
                        continue
                    rows = np.stack([c.chunks[q][i].embd for q in range(n_seq)])
                    if c.needs_prompt_hidden:
                        raise RuntimeError("prompt hidden states over embedding chunks are not supported")
                    if pos + rows.shape[1] + 1 >= bb.n_ctx:
                        raise RuntimeError("the prompt does not fit the backbone context")
                    bb.decode_embd(rows, pos)
                    pos += rows.shape[1]
                    continue
                if n_seq > 1:
                    raise RuntimeError("several sequences take embedding prompts only")
                toks = list(ch.tokens) if ch.type == AudioLM.CHUNK_TOKENS else \
                    bb.tokenize(ch.text, ch.add_special, ch.parse_special)
                if not toks:
                    continue
                if pos + len(toks) + 1 >= bb.n_ctx:
                    raise RuntimeError(f"the prompt ({pos + len(toks)} tokens) does not fit the backbone context "
                                       f"({bb.n_ctx}); shorten the reference or raise max_frames")
                smpl.accept_prompt(toks)
                bb.decode_tokens(toks, pos, all_out=c.needs_prompt_hidden)
                if c.needs_prompt_hidden:
                    prompt_hidden += [bb.hidden(k) for k in range(len(toks))]
                pos += len(toks)
            if c.needs_prompt_hidden:
                alm.prefill(np.stack(prompt_hidden))

            streaming = chunk_ms > 0 and c.streamable
            max_frames = self.max_frames or c.max_frames
            sample, frames, stop = c.sample_first, 0, None
            while frames < max_frames:
                if pos >= bb.n_ctx:
                    stop = "n_ctx"
                    break
                h = None
                if c.needs_embeddings:
                    # Sequence q's last output is -(n_seq - q) from the end.
                    h = np.stack([bb.hidden(-(n_seq - q)) for q in range(n_seq)])
                tok = None
                if sample:
                    tok = smpl.sample(bb.ctx)
                v = alm.observe(tokens=None if tok is None else [tok], hiddens=h)
                frames += v.is_audio
                if v.action == AudioLM.STOP:
                    break
                if v.action == AudioLM.FEED_TOKEN:
                    bb.decode_tokens([v.token if v.token >= 0 else tok], pos)
                else:
                    rows = np.stack([alm.next_embd(q, c.n_embd) for q in range(n_seq)])[:, None, :]
                    bb.decode_embd(rows, pos)
                pos += 1
                if v.is_audio and streaming:
                    pcm, sr = alm.stream_poll(chunk_ms)
                    if len(pcm):
                        yield pcm, sr
                sample = v.sample_backbone
            if streaming:
                pcm, sr = alm.stream_finish()
                if len(pcm):
                    yield pcm, sr
                return frames, True
            return frames, stop
        finally:
            smpl.free()
