"""High-level wrappers for the codec API in include/codec.h."""

from __future__ import annotations

import ctypes
import os
import weakref
from pathlib import Path
from typing import Mapping, Union

import numpy as np

from . import _ffi
from ._lib import lib

PathLike = Union[str, "os.PathLike[str]"]

_STATUS_NAMES = {
    _ffi.CODEC_STATUS_SUCCESS: "SUCCESS",
    _ffi.CODEC_STATUS_INVALID_ARG: "INVALID_ARG",
    _ffi.CODEC_STATUS_INVALID_STATE: "INVALID_STATE",
    _ffi.CODEC_STATUS_IO_ERROR: "IO_ERROR",
    _ffi.CODEC_STATUS_NOT_SUPPORTED: "NOT_SUPPORTED",
    _ffi.CODEC_STATUS_INTERNAL_ERROR: "INTERNAL_ERROR",
}


class CodecError(RuntimeError):
    """A codec.cpp call failed; ``status`` is the codec_status value (or None)."""

    def __init__(self, what: str, status: int | None = None, detail: str = ""):
        self.status = status
        self.detail = detail
        msg = what
        if status is not None:
            msg += f": {_STATUS_NAMES.get(status, status)}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


def _str(b: bytes | None) -> str:
    return b.decode("utf-8", "replace") if b else ""


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def _fptr(a: np.ndarray | None):
    return None if a is None else a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _iptr(a: np.ndarray | None):
    return None if a is None else a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _audio_struct(pcm, sample_rate: int) -> tuple[_ffi.codec_audio, np.ndarray]:
    """PCM (n,) mono or (n, channels) interleaved -> codec_audio.

    int16 arrays are passed as CODEC_PCM_TYPE_I16; anything else as float32.
    """
    is_i16 = isinstance(pcm, np.ndarray) and pcm.dtype == np.int16
    arr = np.ascontiguousarray(pcm) if is_i16 else _f32(pcm)
    if arr.ndim == 1:
        n_ch = 1
    elif arr.ndim == 2:
        n_ch = int(arr.shape[1])
    else:
        raise ValueError(f"PCM must be (n,) or (n, channels), got {arr.shape}")
    audio = _ffi.codec_audio()
    audio.data = arr.ctypes.data_as(ctypes.c_void_p)
    audio.n_samples = int(arr.shape[0])
    audio.sample_rate = int(sample_rate)
    audio.n_channels = n_ch
    audio.pcm_type = _ffi.CODEC_PCM_TYPE_I16 if is_i16 else _ffi.CODEC_PCM_TYPE_F32
    return audio, arr  # keep `arr` alive while `audio` is in use


def _take_pcm(pcm: _ffi.codec_pcm_buffer) -> tuple[np.ndarray, int]:
    try:
        n, nc = int(pcm.n_samples), max(1, int(pcm.n_channels))
        out = np.ctypeslib.as_array(pcm.data, shape=(n,)).copy() if n else np.zeros(0, np.float32)
        if nc > 1:
            out = out.reshape(-1, nc)
        return out, int(pcm.sample_rate)
    finally:
        lib().codec_pcm_buffer_free(ctypes.byref(pcm))


def _make_kv_overrides(kv: Mapping[str, object]):
    arr = (_ffi.codec_model_kv_override * (len(kv) + 1))()
    for i, (key, val) in enumerate(kv.items()):
        k = key.encode("utf-8")
        if len(k) >= 128:
            raise ValueError(f"kv override key too long: {key!r}")
        o = arr[i]
        o.key = k
        if isinstance(val, bool):
            o.tag, o.val_bool = _ffi.CODEC_KV_OVERRIDE_TYPE_BOOL, val
        elif isinstance(val, int):
            o.tag, o.val_i64 = _ffi.CODEC_KV_OVERRIDE_TYPE_INT, val
        elif isinstance(val, float):
            o.tag, o.val_f64 = _ffi.CODEC_KV_OVERRIDE_TYPE_FLOAT, val
        elif isinstance(val, str):
            s = val.encode("utf-8")
            if len(s) >= 128:
                raise ValueError(f"kv override string too long for {key!r}")
            o.tag, o.val_str = _ffi.CODEC_KV_OVERRIDE_TYPE_STR, s
        else:
            raise TypeError(f"kv override {key!r}: unsupported type {type(val).__name__}")
    return arr  # last entry stays zeroed: key[0] == 0 terminates the list


class Speaker:
    """Reference-voice conditioning (codec_speaker).

    Speakers from :meth:`Model.speaker` / :meth:`SpeakerFile.speaker` are
    borrowed from their owner; speakers from :meth:`Context.speaker_from_audio`
    / :meth:`Model.load_speaker` are owned and freed by this object.
    """

    def __init__(self, ptr: int, *, owner: object | None = None, owned: bool = False):
        self.ptr = ptr
        self._owner = owner  # keeps the borrowing parent alive
        self._owned = owned

    @property
    def name(self) -> str:
        return _str(lib().codec_speaker_name(self.ptr))

    @property
    def ref_text(self) -> str | None:
        t = lib().codec_speaker_ref_text(self.ptr)
        return None if t is None else _str(t)

    @ref_text.setter
    def ref_text(self, text: str) -> None:
        st = lib().codec_speaker_set_ref_text(self.ptr, text.encode("utf-8"))
        if st != _ffi.CODEC_STATUS_SUCCESS:
            raise CodecError("codec_speaker_set_ref_text", st)

    def slot(self, name: str) -> np.ndarray | None:
        """Copy of one slot as an array shaped like the ggml tensor reversed
        (row-major, ne[3]..ne[0]), or None when absent."""
        typ = ctypes.c_int(0)
        ne = (ctypes.c_int64 * 4)()
        data = ctypes.c_void_p()
        if not lib().codec_speaker_get_slot(self.ptr, name.encode("utf-8"), ctypes.byref(typ), ne,
                                            ctypes.byref(data)):
            return None
        dtype = np.float32 if typ.value == _ffi.CODEC_SPEAKER_SLOT_F32 else np.int32
        ctype = ctypes.c_float if dtype is np.float32 else ctypes.c_int32
        shape = tuple(int(ne[i]) for i in (3, 2, 1, 0))
        n = int(np.prod(shape))
        flat = np.ctypeslib.as_array(ctypes.cast(data, ctypes.POINTER(ctype)), shape=(n,)).copy()
        while len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]
        return flat.reshape(shape)

    def close(self) -> None:
        if self._owned and self.ptr:
            lib().codec_speaker_free(self.ptr)
        self.ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class SpeakerFile:
    """A speaker file opened against a model (codec_speaker_file)."""

    def __init__(self, model: "Model", path: PathLike):
        self._model = model
        out = ctypes.c_void_p()
        st = lib().codec_speaker_file_open(model.ptr, os.fsencode(path), ctypes.byref(out))
        if st != _ffi.CODEC_STATUS_SUCCESS:
            raise CodecError(f"codec_speaker_file_open({path})", st)
        self.ptr = out.value

    def __len__(self) -> int:
        return int(lib().codec_speaker_file_n_speakers(self.ptr))

    @property
    def speaker_names(self) -> list[str]:
        return [_str(lib().codec_speaker_file_speaker_name(self.ptr, i)) for i in range(len(self))]

    def speaker(self, name: str | None = None) -> Speaker:
        p = lib().codec_speaker_file_get_speaker(self.ptr, None if name is None else name.encode("utf-8"))
        if not p:
            raise KeyError(name)
        return Speaker(p, owner=self)

    def close(self) -> None:
        if getattr(self, "ptr", None):
            lib().codec_speaker_file_free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class Model:
    """A loaded codec GGUF (codec_model).

    ``kv_overrides`` maps GGUF metadata keys to int / float / bool / str
    values applied at load time (see codec_model_load_from_file_ext).
    """

    def __init__(self, path: PathLike, *, use_gpu: bool | None = None, n_threads: int | None = None,
                 kv_overrides: Mapping[str, object] | None = None):
        L = lib()
        mp = L.codec_model_default_params()
        if use_gpu is not None:
            mp.use_gpu = bool(use_gpu)
        if n_threads is not None:
            mp.n_threads = int(n_threads)
        self.path = Path(path)
        self._lm = None
        self._contexts: weakref.WeakSet = weakref.WeakSet()
        if kv_overrides:
            self.ptr = L.codec_model_load_from_file_ext(os.fsencode(path), mp, _make_kv_overrides(kv_overrides))
        else:
            self.ptr = L.codec_model_load_from_file(os.fsencode(path), mp)
        if not self.ptr:
            raise CodecError(f"codec_model_load_from_file({path}) failed")

    # --- info ---------------------------------------------------------

    @property
    def arch(self) -> int: return int(lib().codec_model_arch(self.ptr))
    @property
    def arch_name(self) -> str: return _str(lib().codec_arch_name(self.arch))
    @property
    def name(self) -> str: return _str(lib().codec_model_name(self.ptr))
    @property
    def n_tensors(self) -> int: return int(lib().codec_model_n_tensors(self.ptr))
    @property
    def sample_rate(self) -> int: return int(lib().codec_model_sample_rate(self.ptr))
    @property
    def has_encoder(self) -> bool: return bool(lib().codec_model_has_encoder(self.ptr))
    @property
    def has_decoder(self) -> bool: return bool(lib().codec_model_has_decoder(self.ptr))
    @property
    def n_q(self) -> int: return int(lib().codec_model_n_q(self.ptr))
    @property
    def codebook_size(self) -> int: return int(lib().codec_model_codebook_size(self.ptr))
    @property
    def hop_size(self) -> int: return int(lib().codec_model_hop_size(self.ptr))
    @property
    def n_fft(self) -> int: return int(lib().codec_model_n_fft(self.ptr))
    @property
    def win_length(self) -> int: return int(lib().codec_model_win_length(self.ptr))
    @property
    def n_mels(self) -> int: return int(lib().codec_model_n_mels(self.ptr))
    @property
    def latent_dim(self) -> int: return int(lib().codec_model_latent_dim(self.ptr))

    def metadata(self) -> dict[str, str]:
        """GGUF metadata as string key/value pairs."""
        m = lib().codec_model_metadata(self.ptr)
        if not m:
            return {}
        md = m.contents
        return {_str(md.items[i].key): _str(md.items[i].value) for i in range(int(md.n_items))}

    # --- speakers -----------------------------------------------------

    @property
    def speaker_names(self) -> list[str]:
        L = lib()
        return [_str(L.codec_model_speaker_name(self.ptr, i)) for i in range(int(L.codec_model_n_speakers(self.ptr)))]

    def speaker(self, name: str | None = None) -> Speaker:
        """Builtin speaker (``None`` = the model's default)."""
        p = lib().codec_get_speaker(self.ptr, None if name is None else name.encode("utf-8"))
        if not p:
            raise KeyError(name)
        return Speaker(p, owner=self)

    @property
    def can_speaker_from_audio(self) -> bool:
        return bool(lib().codec_model_can_speaker_from_audio(self.ptr))

    def save_speaker(self, speaker: Speaker, path: PathLike, name: str | None = None) -> None:
        st = lib().codec_speaker_save(self.ptr, speaker.ptr, None if name is None else name.encode("utf-8"),
                                      os.fsencode(path))
        if st != _ffi.CODEC_STATUS_SUCCESS:
            raise CodecError(f"codec_speaker_save({path})", st)

    def load_speaker(self, path: PathLike, name: str | None = None) -> Speaker:
        out = ctypes.c_void_p()
        st = lib().codec_speaker_load(self.ptr, os.fsencode(path), None if name is None else name.encode("utf-8"),
                                      ctypes.byref(out))
        if st != _ffi.CODEC_STATUS_SUCCESS:
            raise CodecError(f"codec_speaker_load({path})", st)
        return Speaker(out.value, owned=True)

    def open_speaker_file(self, path: PathLike) -> SpeakerFile:
        return SpeakerFile(self, path)

    # --- contexts -----------------------------------------------------

    def context(self, *, seed: int | None = None) -> "Context":
        return Context(self, seed=seed)

    def get_lm_adaptor(self):
        """The model's codec_lm adaptor (see codec_cpp.lm.LM).

        Created on first call and reused afterwards; it lives as long as the
        Model and is freed by Model.close().  Per-sequence state comes from
        ``adaptor.state()``.
        """
        if self._lm is None:
            if not self.ptr:
                raise CodecError("get_lm_adaptor on a closed Model")
            from .lm import LM
            self._lm = LM(self)
        return self._lm

    def close(self) -> None:
        for c in list(getattr(self, "_contexts", ())):
            c.close()
        lm, self._lm = getattr(self, "_lm", None), None
        if lm is not None:
            lm._free()  # borrows the model: free it first
        if getattr(self, "ptr", None):
            lib().codec_model_free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


class Context:
    """Inference context (codec_context): encode / decode against a Model."""

    def __init__(self, model: Model, *, seed: int | None = None):
        self.model = model
        L = lib()
        cp = L.codec_context_default_params()
        if seed is not None:
            cp.seed = int(seed)
        self.ptr = L.codec_init_from_model(model.ptr, cp)
        if not self.ptr:
            raise CodecError("codec_init_from_model failed")
        model._contexts.add(self)

    @property
    def last_error(self) -> str:
        return _str(lib().codec_get_last_error(self.ptr))

    def _check(self, what: str, st: int) -> None:
        if st != _ffi.CODEC_STATUS_SUCCESS:
            raise CodecError(what, st, self.last_error)

    @staticmethod
    def _enc_params(n_q: int, n_threads: int | None) -> _ffi.codec_encode_params:
        ep = lib().codec_encode_default_params()
        ep.n_q = int(n_q)
        if n_threads is not None:
            ep.n_threads = int(n_threads)
        return ep

    @staticmethod
    def _dec_params(n_q: int, n_threads: int | None, speaker: Speaker | None) -> _ffi.codec_decode_params:
        dp = lib().codec_decode_default_params()
        dp.n_q = int(n_q)
        if n_threads is not None:
            dp.n_threads = int(n_threads)
        if speaker is not None:
            dp.speaker = speaker.ptr
        return dp

    def encode(self, pcm, sample_rate: int, *, n_q: int = 0, n_threads: int | None = None) -> np.ndarray:
        """PCM float32 / int16, (n,) or (n, channels) -> codes int32 (n_frames, n_q)."""
        audio, _keep = _audio_struct(pcm, sample_rate)
        tb = _ffi.codec_token_buffer()
        L = lib()
        self._check("codec_encode",
                    L.codec_encode(self.ptr, ctypes.byref(audio), ctypes.byref(tb), self._enc_params(n_q, n_threads)))
        try:
            nf, q = int(tb.n_frames), int(tb.n_q)
            return np.ctypeslib.as_array(tb.data, shape=(nf * q,)).copy().reshape(nf, q)
        finally:
            L.codec_token_buffer_free(ctypes.byref(tb))

    def encode_latent(self, pcm, sample_rate: int, *, n_q: int = 0,
                      n_threads: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """PCM -> (latent float32 (latent_dim, n_frames), codes (n_frames, n_q) or None)."""
        audio, _keep = _audio_struct(pcm, sample_rate)
        tb = _ffi.codec_token_buffer()
        lb = _ffi.codec_latent_buffer()
        L = lib()
        self._check("codec_encode_latent",
                    L.codec_encode_latent(self.ptr, ctypes.byref(audio), ctypes.byref(tb), ctypes.byref(lb),
                                          self._enc_params(n_q, n_threads)))
        try:
            d, t = int(lb.latent_dim), int(lb.n_frames)
            latent = np.ctypeslib.as_array(lb.data, shape=(d * t,)).copy().reshape(d, t)
            codes = None
            if tb.data and tb.n_frames > 0:
                nf, q = int(tb.n_frames), int(tb.n_q)
                codes = np.ctypeslib.as_array(tb.data, shape=(nf * q,)).copy().reshape(nf, q)
            return latent, codes
        finally:
            L.codec_latent_buffer_free(ctypes.byref(lb))
            L.codec_token_buffer_free(ctypes.byref(tb))

    def decode(self, codes, *, n_q: int = 0, n_threads: int | None = None,
               speaker: Speaker | None = None) -> tuple[np.ndarray, int]:
        """codes int32 (n_frames, n_q) -> (PCM float32, sample_rate).

        PCM is (n,) for mono, (n, channels) otherwise.
        """
        c = _i32(codes)
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        if c.ndim != 2:
            raise ValueError(f"codes must be (n_frames, n_q), got {c.shape}")
        m = self.model
        tb = _ffi.codec_token_buffer()
        tb.data = _iptr(c)
        tb.n_frames, tb.n_q = int(c.shape[0]), int(c.shape[1])
        tb.n_tokens = int(c.size)
        tb.codebook_size = m.codebook_size
        tb.sample_rate = m.sample_rate
        tb.hop_size = m.hop_size
        pcm = _ffi.codec_pcm_buffer()
        self._check("codec_decode", lib().codec_decode(self.ptr, ctypes.byref(tb), ctypes.byref(pcm),
                                                        self._dec_params(n_q, n_threads, speaker)))
        return _take_pcm(pcm)

    def decode_latent(self, latent, *, n_threads: int | None = None,
                      speaker: Speaker | None = None) -> tuple[np.ndarray, int]:
        """Continuous latent float32 (latent_dim, n_frames) -> (PCM, sample_rate)."""
        z = _f32(latent)
        if z.ndim != 2:
            raise ValueError(f"latent must be (latent_dim, n_frames), got {z.shape}")
        pcm = _ffi.codec_pcm_buffer()
        self._check("codec_decode_quantized_representation",
                    lib().codec_decode_quantized_representation(
                        self.ptr, _fptr(z), int(z.shape[0]), int(z.shape[1]), ctypes.byref(pcm),
                        self._dec_params(0, n_threads, speaker)))
        return _take_pcm(pcm)

    def decode_batch(self, seqs: list, *, latent: bool = False, n_q: int = 0,
                     n_threads: int | None = None) -> list[tuple[np.ndarray, int]]:
        """Decode several sequences in one call (codec_decode_batch).

        ``seqs`` are code arrays (n_frames, n_q), or latents (latent_dim,
        n_frames) when ``latent=True``.  Returns one (PCM, sample_rate) per
        sequence, in order.
        """
        L = lib()
        arrs = [_f32(s) if latent else _i32(s) for s in seqs]
        n = len(arrs)
        if n == 0:
            return []
        if latent:
            batch = L.codec_batch_init_latent(n, int(arrs[0].shape[0]), sum(a.size for a in arrs), n)
        else:
            batch = L.codec_batch_init_codes(n, sum(a.size for a in arrs), n)
        try:
            for i, a in enumerate(arrs):
                if latent:
                    rc = L.codec_batch_add_seq_latent(ctypes.byref(batch), i, int(a.shape[1]), _fptr(a),
                                                      int(a.shape[0]))
                else:
                    a = a.reshape(a.shape[0], -1)
                    rc = L.codec_batch_add_seq_codes(ctypes.byref(batch), i, int(a.shape[0]), int(a.shape[1]),
                                                     _iptr(a))
                if rc < 0:
                    raise CodecError(f"codec_batch_add_seq (seq {i}) failed")
            pcms = (_ffi.codec_pcm_buffer * n)()
            st = L.codec_decode_batch(self.ptr, ctypes.byref(batch), pcms, self._dec_params(n_q, n_threads, None))
            if st != _ffi.CODEC_STATUS_SUCCESS:
                for p in pcms:
                    L.codec_pcm_buffer_free(ctypes.byref(p))
                self._check("codec_decode_batch", st)
            return [_take_pcm(p) for p in pcms]
        finally:
            L.codec_batch_free(batch)

    def speaker_from_audio(self, pcm, sample_rate: int) -> Speaker:
        """Derive a speaker from reference audio (any rate; multi-channel is downmixed)."""
        audio, _keep = _audio_struct(pcm, sample_rate)
        out = ctypes.c_void_p()
        self._check("codec_speaker_from_audio",
                    lib().codec_speaker_from_audio(self.ptr, ctypes.byref(audio), ctypes.byref(out)))
        return Speaker(out.value, owned=True)

    def close(self) -> None:
        if getattr(self, "ptr", None):
            lib().codec_free(self.ptr)
            self.ptr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
