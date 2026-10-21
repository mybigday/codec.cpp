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
import logging
import os
import sys
import threading
from pathlib import Path

logger = logging.getLogger("codec_cpp")

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
                handle.codec_log_set(_log_cb, None)
                _lib = handle
    return _lib


# codec.cpp's log (and its ggml backends') goes to the "codec_cpp" logger:
# quiet below WARNING unless the application configures logging.
_LEVELS = {1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR, 4: logging.DEBUG}
_log_state = {"level": logging.INFO, "buf": ""}


def _on_log(level: int, text: bytes, _user) -> None:
    try:
        if level != 5:  # CODEC_LOG_LEVEL_CONT continues the previous message
            _log_state["level"] = _LEVELS.get(level, logging.INFO)
        buf = _log_state["buf"] + (text or b"").decode("utf-8", "replace")
        *lines, _log_state["buf"] = buf.split("\n")
        for line in lines:
            if line:
                logger.log(_log_state["level"], line)
    except Exception:  # never raise into C
        pass


from . import _ffi as _ffi_mod  # noqa: E402

_log_cb = _ffi_mod.codec_log_callback(_on_log)


# libttsbackbone: the llama.cpp bundled with codec.cpp (its architecture /
# tokenizer patches applied, its ggml hidden), for codec_cpp.TTS.  Separate
# from libcodec so codec-only users never load llama.cpp; looked up next to
# libcodec.
_BACKBONE_LIB_NAMES = ("libttsbackbone.so",)

_backbone_lib: ctypes.CDLL | None = None


def backbone_library_path() -> Path:
    env = os.environ.get("CODEC_CPP_BACKBONE_LIBRARY")
    if env and Path(env).is_file():
        return Path(env)
    dirs = [library_path().parent, library_path().parent / "ttsbackbone", _PKG_DIR / "lib",
            _REPO_ROOT / "build" / "ttsbackbone"]
    for d in dirs:
        for n in _BACKBONE_LIB_NAMES:
            if (d / n).is_file():
                return d / n
    raise FileNotFoundError(
        "libttsbackbone not found; build codec.cpp with CODEC_TTS_BACKBONE=ON (the default) "
        "or set CODEC_CPP_BACKBONE_LIBRARY.")


def backbone_lib() -> ctypes.CDLL:
    """Return the loaded bundled llama.cpp with the prototypes codec_cpp.TTS uses."""
    global _backbone_lib
    if _backbone_lib is None:
        lib()  # libcodec first: its log callback is reused for llama.cpp's log
        with _lock:
            if _backbone_lib is None:
                from . import _ffi
                handle = ctypes.CDLL(str(backbone_library_path()))
                _ffi.declare_llama(handle)
                handle.llama_log_set(_log_cb, None)
                handle.llama_backend_init()
                _backbone_lib = handle
    return _backbone_lib
