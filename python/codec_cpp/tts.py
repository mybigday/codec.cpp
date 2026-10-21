"""One-call text to speech (common/codec_tts.h, libcodec_tts).

    tts = codec_cpp.TTS("model.gguf", "codec.gguf")
    audio, sr = tts.synthesize("Hello there.")
    audio, sr = tts.synthesize("Hello there.", ref_audio="me.wav")   # voice clone

This is the host loop tts-cli runs: the backbone (model file) runs in the
llama.cpp bundled with codec.cpp, the codec file in codec.cpp.  For a custom
loop around your own llama.cpp use codec_cpp.AudioLM instead.
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from typing import Mapping

import numpy as np

from . import _ffi
from ._lib import tts_lib
from .codec import CodecError, PathLike, _make_kv_overrides, _str, _take_pcm


def _b(s) -> bytes | None:
    return None if s is None else os.fsencode(s) if isinstance(s, os.PathLike) else str(s).encode("utf-8")


class TTS:
    """A loaded TTS model: a llama.cpp backbone GGUF + a codec GGUF.

    ``model`` is None for self-contained models.  Load-time settings:
    ``use_gpu``, ``n_threads``, ``max_frames`` (longest generation, sizes the
    backbone context; 0 = model default) and ``kv_overrides`` (codec GGUF
    metadata).  Calls are not thread-safe; use one TTS per thread or
    serialize them.
    """

    def __init__(self, model: PathLike | None, codec: PathLike, *, use_gpu: bool = False,
                 n_threads: int = 0, max_frames: int = 0,
                 kv_overrides: Mapping[str, object] | None = None):
        L = tts_lib()
        p = L.codec_tts_default_params()
        p.codec_path = os.fsencode(codec)
        p.backbone_path = None if model is None else os.fsencode(model)
        p.use_gpu = bool(use_gpu)
        p.n_threads = int(n_threads)
        p.max_frames = int(max_frames)
        self._kv = None
        if kv_overrides:
            self._kv = _make_kv_overrides(kv_overrides)
            p.kv_overrides, p.n_kv_overrides = self._kv, len(kv_overrides)
        err = ctypes.create_string_buffer(2048)
        self.ptr = L.codec_tts_load(ctypes.byref(p), err, len(err))
        if not self.ptr:
            raise CodecError("codec_tts_load", None, _str(err.value))
        self.last_result: dict = {}

    def close(self) -> None:
        if getattr(self, "ptr", None):
            tts_lib().codec_tts_free(self.ptr)
            self.ptr = None

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def __del__(self): self.close()

    # --- voices --------------------------------------------------------

    def voices(self, speaker_file: PathLike | None = None) -> list[str]:
        """Names ``voice=`` accepts: the builtin voices, or those in a speaker file."""
        L = tts_lib()
        n = L.codec_tts_voices(self.ptr, _b(speaker_file))
        if n < 0:
            raise CodecError("codec_tts_voices", None, _str(L.codec_tts_last_error(self.ptr)))
        return [_str(L.codec_tts_voice(self.ptr, i)) for i in range(n)]

    @property
    def default_voice(self) -> str:
        return _str(tts_lib().codec_tts_default_voice(self.ptr))

    @property
    def takes_instruct(self) -> bool:
        """The model takes ``instruct=``."""
        return bool(tts_lib().codec_tts_takes_instruct(self.ptr))

    # --- synthesis -----------------------------------------------------

    def synthesize(self, text: str, **opts) -> tuple[np.ndarray, int]:
        """Synthesize ``text``: returns (float32 PCM, sample rate).

        Voice: ``speaker_file`` > ``ref_audio`` (clone; a WAV path or a
        ``(pcm, sample_rate)`` tuple, with ``ref_text`` its transcript) >
        ``voice`` (builtin) > model default.  Other options (unset = the
        model's default): ``instruct``, ``preset``, ``grammar``, ``seed``,
        ``temperature``, ``top_p``, ``top_k``, ``min_p``,
        ``repetition_penalty``, ``min_frames``, ``lm_guidance``,
        ``flow_guidance``, ``flow_steps``, ``segment_bytes``,
        ``save_speaker``.
        """
        return self._run(text, opts, None, 0)

    def stream(self, text: str, *, chunk_ms: int = 0, **opts):
        """Like synthesize, but yields (float32 PCM piece, sample rate) as the
        audio becomes final: per segment, or with ``chunk_ms`` > 0 about that
        much audio at a time while generating (models that support it)."""
        import queue
        import threading
        q: "queue.Queue" = queue.Queue(maxsize=64)
        stop = threading.Event()
        done = object()

        def on_audio(pcm, sr):
            q.put((pcm, sr))
            return not stop.is_set()

        def work():
            try:
                self._run(text, opts, on_audio, chunk_ms)
                q.put(done)
            except BaseException as e:  # surfaced in the consumer
                q.put(e)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            while t.is_alive():
                try:
                    q.get_nowait()
                except Exception:
                    t.join(0.05)

    def _run(self, text: str, opts: dict, on_audio, chunk_ms: int) -> tuple[np.ndarray, int]:
        L = tts_lib()
        r = L.codec_tts_default_request()
        r.text = text.encode("utf-8")
        opts = dict(opts)
        ref_audio = opts.pop("ref_audio", None)
        for name in ("voice", "ref_text", "instruct", "speaker_file", "save_speaker", "preset", "grammar"):
            setattr(r, name, _b(opts.pop(name, None)))
        seed = opts.pop("seed", None)
        if seed is not None:
            r.seed = int(seed) & 0xFFFFFFFF
        for name in ("temperature", "top_p", "min_p", "repetition_penalty", "lm_guidance", "flow_guidance"):
            v = opts.pop(name, None)
            if v is not None:
                setattr(r, name, float(v))
        for name in ("top_k", "min_frames", "flow_steps", "segment_bytes"):
            v = opts.pop(name, None)
            if v is not None:
                setattr(r, name, int(v))
        if opts:
            raise TypeError(f"unknown synthesis option(s): {', '.join(sorted(opts))}")
        cb = None
        if on_audio is not None:
            def _cb(pcm, n, sr, nch, _ud):
                try:
                    x = np.ctypeslib.as_array(pcm, shape=(n * nch,)).copy()
                    return bool(on_audio(x if nch == 1 else x.reshape(n, nch), int(sr)))
                except Exception:
                    return False
            cb = _ffi.codec_tts_audio_callback(_cb)
            r.on_audio = cb
            r.stream_chunk_ms = int(chunk_ms)

        tmp = None
        try:
            if isinstance(ref_audio, tuple):
                from .codec import _f32
                pcm_in, sr_in = ref_audio
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                _write_wav(tmp.name, _f32(pcm_in), int(sr_in))
                r.ref_audio_path = tmp.name.encode()
            elif ref_audio is not None:
                r.ref_audio_path = _b(ref_audio)
            pcm = _ffi.codec_pcm_buffer()
            res = _ffi.codec_tts_result()
            if not L.codec_tts_run(self.ptr, ctypes.byref(r), ctypes.byref(pcm), ctypes.byref(res)):
                raise CodecError("codec_tts_run", None, _str(L.codec_tts_last_error(self.ptr)))
        finally:
            if tmp is not None:
                os.unlink(tmp.name)
        self.last_result = {"n_frames": int(res.n_frames), "stop_reason": _str(res.stop_reason),
                            "total_ms": float(res.total_ms), "decode_ms": float(res.decode_ms)}
        return _take_pcm(pcm)


def _write_wav(path: str, pcm: np.ndarray, sr: int) -> None:
    """16-bit WAV (interleaved when pcm is (n, channels))."""
    import wave
    x = np.clip(pcm, -1.0, 1.0)
    ch = 1 if x.ndim == 1 else x.shape[1]
    with wave.open(path, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())
