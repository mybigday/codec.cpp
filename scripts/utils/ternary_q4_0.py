#!/usr/bin/env python3
"""Losslessly repack a ternary-weight GGUF (bf16/f16/f32) to Q4_0.

Ternary checkpoints (BitNet-style, e.g. KittenTTS 2) store every linear weight
as {-s, 0, +s} with one scale `s` per group of >= 32 values. A Q4_0 block
(32 values, fp16 scale `d`, 4-bit q, value = d * (q - 8)) holds such a block
exactly with d = s and q in {7, 8, 9}. llama-quantize would instead pick
d = max / -8, which cannot represent -s exactly, so this writes the blocks
directly and refuses any tensor that is not exactly ternary.

Only 2-D tensors are packed (the token embedding is kept as-is: it is not
ternary). Usage: ternary_q4_0.py IN.gguf OUT.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

QK = 32


def _gguf():
    here = Path(__file__).resolve().parents[2] / "common" / "third-party" / "llama.cpp" / "gguf-py"
    if here.is_dir() and str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import gguf
    return gguf


def _to_f32(gguf, tensor) -> np.ndarray:
    t = tensor.tensor_type
    data = np.asarray(tensor.data)
    if t == gguf.GGMLQuantizationType.BF16:
        return (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if t in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        return data.astype(np.float32)
    raise ValueError(f"{tensor.name}: unsupported source type {t.name}")


def pack_q4_0(w: np.ndarray, name: str) -> np.ndarray | None:
    """[rows, cols] float32 → Q4_0 bytes [rows, cols / 32 * 18], or None if not exactly ternary."""
    rows, cols = w.shape
    if cols % QK:
        return None
    blocks = w.reshape(rows, cols // QK, QK)
    a = np.abs(blocks)
    d = a.max(axis=-1)
    nz = a > 0
    if np.any(nz & (a != d[..., None])):
        return None                                     # more than one magnitude in a block
    d16 = d.astype(np.float16)
    if np.any(d16.astype(np.float32) != d):
        return None                                     # scale not exactly representable in fp16
    q = (8 + np.sign(blocks)).astype(np.uint8)          # {7, 8, 9}
    qs = q[..., :16] | (q[..., 16:] << 4)               # low nibbles: 0..15, high: 16..31
    out = np.empty((rows, cols // QK, 18), dtype=np.uint8)
    out[..., :2] = d16.view(np.uint8).reshape(rows, cols // QK, 2)
    out[..., 2:] = qs
    return out.reshape(rows, -1)


def repack(src: Path, dst: Path) -> None:
    gguf = _gguf()
    reader = gguf.GGUFReader(str(src))
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(str(dst), arch=arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.FILE_TYPE:
            writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)

    payloads = []
    n_packed = 0
    for t in reader.tensors:
        shape = tuple(int(x) for x in reversed(t.shape))   # numpy order
        if len(shape) == 2 and t.name != "token_embd.weight":
            packed = pack_q4_0(_to_f32(gguf, t).reshape(shape), t.name)
            if packed is None:
                raise SystemExit(f"{t.name} is not exactly ternary per 32-block; refusing a lossy repack")
            payloads.append((t.name, packed, gguf.GGMLQuantizationType.Q4_0))
            n_packed += 1
        else:
            payloads.append((t.name, np.asarray(t.data), t.tensor_type))
    for name, data, ttype in payloads:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in payloads:
        writer.write_tensor_data(data)
    writer.close()
    print(f"[ternary-q4_0] packed {n_packed} tensors → {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    repack(Path(sys.argv[1]), Path(sys.argv[2]))
