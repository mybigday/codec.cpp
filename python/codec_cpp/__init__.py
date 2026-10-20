"""Python bindings for codec.cpp (libcodec).

    import codec_cpp
    model = codec_cpp.Model("models/mimi/mimi.gguf")
    ctx = model.context()
    codes = ctx.encode(pcm, sample_rate)          # (n_frames, n_q) int32
    audio, sr = ctx.decode(codes)                  # float32 PCM

``codec_cpp.ffi`` exposes the raw ctypes declarations and ``codec_cpp.lib()``
the loaded library for calls the high-level layer does not wrap.
"""

from . import _ffi as ffi
from ._lib import lib, library_path, tts_lib
from .audio_lm import AudioLM, Prompt, SpeakerEntry, outetts_grammar
from .codec import CodecError, Context, Model, Speaker, SpeakerFile
from .lm import LM, LMInfo, LMState
from .tts import TTS

__version__ = "0.1.0"

__all__ = [
    "AudioLM", "CodecError", "Context", "LM", "LMInfo", "LMState", "Model", "Prompt", "Speaker",
    "SpeakerEntry", "SpeakerFile", "TTS", "ffi", "lib", "library_path", "tts_lib", "outetts_grammar",
]
