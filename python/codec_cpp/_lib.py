"""Locate and load libcodec.

The library is loaded lazily on first use, not at import time.  Processes
that also load llama-cpp-python must import ``llama_cpp`` before the first
codec_cpp call: llama-cpp-python ships its own libggml, and whichever
libggml is loaded first wins the process-wide symbol lookup.

Search order:
  1. ``$CODEC_CPP_LIBRARY`` — full path to the shared library.
  2. ``codec_cpp/lib/`` inside the installed package (bundled builds).
  3. ``<repo>/build/`` when running from a source checkout.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]

if sys.platform == "darwin":
    _LIB_NAMES = ("libcodec.dylib",)
elif sys.platform == "win32":
    _LIB_NAMES = ("codec.dll", "libcodec.dll")
else:
    _LIB_NAMES = ("libcodec.so",)

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None


def candidate_paths() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("CODEC_CPP_LIBRARY")
    if env:
        out.append(Path(env))
    for d in (_PKG_DIR / "lib", _REPO_ROOT / "build", _REPO_ROOT / "build" / "bin"):
        out.extend(d / n for n in _LIB_NAMES)
    return out


def library_path() -> Path:
    for p in candidate_paths():
        if p.is_file():
            return p
    tried = "\n  ".join(str(p) for p in candidate_paths())
    raise FileNotFoundError(
        "libcodec not found; build it with `cmake --build build -j` or set "
        f"CODEC_CPP_LIBRARY. Tried:\n  {tried}")


def lib() -> ctypes.CDLL:
    """Return the loaded libcodec with all prototypes declared."""
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                from . import _ffi
                handle = ctypes.CDLL(str(library_path()))
                _ffi.declare(handle)
                _lib = handle
    return _lib


# libcodec_tts (include/codec_tts.h): the one-call TTS session, with the
# bundled llama.cpp backbone.  Separate from libcodec so codec-only users
# never load llama.cpp; looked up next to libcodec.
if sys.platform == "darwin":
    _TTS_LIB_NAMES = ("libcodec_tts.dylib",)
elif sys.platform == "win32":
    _TTS_LIB_NAMES = ("codec_tts.dll", "libcodec_tts.dll")
else:
    _TTS_LIB_NAMES = ("libcodec_tts.so",)

_tts_lib: ctypes.CDLL | None = None


def tts_library_path() -> Path:
    env = os.environ.get("CODEC_CPP_TTS_LIBRARY")
    dirs = [Path(env).parent] if env else []
    if env and Path(env).is_file():
        return Path(env)
    dirs += [library_path().parent, _PKG_DIR / "lib", _REPO_ROOT / "build"]
    for d in dirs:
        for n in _TTS_LIB_NAMES:
            if (d / n).is_file():
                return d / n
    raise FileNotFoundError(
        "libcodec_tts not found; build codec.cpp with CODEC_TTS_BACKBONE=ON (the default) "
        "or set CODEC_CPP_TTS_LIBRARY.")


def tts_lib() -> ctypes.CDLL:
    """Return the loaded libcodec_tts with its prototypes declared."""
    global _tts_lib
    if _tts_lib is None:
        lib()  # libcodec first: libcodec_tts resolves codec symbols against it
        with _lock:
            if _tts_lib is None:
                from . import _ffi
                handle = ctypes.CDLL(str(tts_library_path()))
                _ffi.declare_tts(handle)
                _tts_lib = handle
    return _tts_lib
