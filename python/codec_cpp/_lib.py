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
