"""Shared helpers for the GGUF speaker format.

Builtin speakers are stored as `spk.<name>.<slot>` tensors (F32 or I32) with
metadata under the model architecture:

    <arch>.spk.default            name of the default speaker
    <arch>.spk.encoder_hash       hash of the encoders that produce speakers
    <arch>.spk.<name>.ref_text    transcript of the reference audio (optional)

The encoder hash identifies the reference-audio -> speaker pipeline (weights
plus preprocessing parameters). The runtime refuses speaker files whose hash
differs from the model's, so it must be computed from the source checkpoint
weights, independent of the GGUF storage dtype.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Mapping

import numpy as np

_NAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def speaker_name(name: str) -> str:
    """Normalize a voice name to the [a-z0-9_] form used in tensor names."""
    out = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    if not _NAME_RE.match(out):
        raise ValueError(f"cannot derive a valid speaker name from {name!r}")
    return out


def encoder_hash(tensors: Iterable[tuple[str, np.ndarray]], params: str) -> str:
    """sha256 over (name, shape, float32/int32 bytes) of each tensor, sorted by name, plus `params`."""
    h = hashlib.sha256()
    h.update(params.encode())
    for name, arr in sorted(tensors, key=lambda kv: kv[0]):
        arr = np.asarray(arr)
        arr = arr.astype(np.int32 if np.issubdtype(arr.dtype, np.integer) else np.float32)
        h.update(name.encode())
        h.update(str(tuple(arr.shape)).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def write_speaker_meta(writer, arch: str, *, default: str, enc_hash: str) -> None:
    writer.add_string(f"{arch}.spk.default", default)
    writer.add_string(f"{arch}.spk.encoder_hash", enc_hash)


def write_speaker(writer, arch: str, name: str, slots: Mapping[str, np.ndarray], ref_text: str | None = None) -> None:
    """Write one speaker. Integer slots are stored as I32, float slots as F32."""
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid speaker name {name!r}")
    for slot, arr in slots.items():
        arr = np.ascontiguousarray(arr)
        if np.issubdtype(arr.dtype, np.integer):
            writer.add_tensor(f"spk.{name}.{slot}", arr.astype(np.int32), "I32")
        else:
            writer.add_tensor(f"spk.{name}.{slot}", arr.astype(np.float32), "F32")
    if ref_text:
        writer.add_string(f"{arch}.spk.{name}.ref_text", ref_text)
