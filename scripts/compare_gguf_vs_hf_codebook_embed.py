#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import MimiModel
from transformers.utils import logging as hf_logging

DTYPE_MAP_GGUF = {
    0: np.dtype("<f4"),  # GGML_TYPE_F32
    1: np.dtype("<f2"),  # GGML_TYPE_F16
}

GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12

SCALAR_SIZES = {
    GGUF_VALUE_TYPE_UINT8: 1,
    GGUF_VALUE_TYPE_INT8: 1,
    GGUF_VALUE_TYPE_UINT16: 2,
    GGUF_VALUE_TYPE_INT16: 2,
    GGUF_VALUE_TYPE_UINT32: 4,
    GGUF_VALUE_TYPE_INT32: 4,
    GGUF_VALUE_TYPE_FLOAT32: 4,
    GGUF_VALUE_TYPE_BOOL: 1,
    GGUF_VALUE_TYPE_UINT64: 8,
    GGUF_VALUE_TYPE_INT64: 8,
    GGUF_VALUE_TYPE_FLOAT64: 8,
}


def _read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_i32(f) -> int:
    return struct.unpack("<i", f.read(4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_i64(f) -> int:
    return struct.unpack("<q", f.read(8))[0]


def _read_str(f) -> str:
    n = _read_u64(f)
    return f.read(n).decode("utf-8")


def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _skip_gguf_value(f, vtype: int) -> None:
    if vtype == GGUF_VALUE_TYPE_STRING:
        _ = _read_str(f)
        return
    if vtype == GGUF_VALUE_TYPE_ARRAY:
        elem_type = _read_i32(f)
        n = _read_u64(f)
        if elem_type == GGUF_VALUE_TYPE_STRING:
            for _ in range(n):
                _ = _read_str(f)
        elif elem_type == GGUF_VALUE_TYPE_ARRAY:
            raise ValueError("nested GGUF arrays are unsupported")
        else:
            size = SCALAR_SIZES.get(elem_type)
            if size is None:
                raise ValueError(f"unsupported GGUF array element type: {elem_type}")
            f.read(size * n)
        return
    size = SCALAR_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"unsupported GGUF value type: {vtype}")
    f.read(size)


def read_gguf_tensor_info(path: Path) -> tuple[list[str], dict[str, tuple[list[int], int, int]], int]:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = _read_u32(f)
        if version != 3:
            raise ValueError(f"unsupported GGUF version: {version}")

        n_tensors = _read_u64(f)
        n_kv = _read_u64(f)

        for _ in range(n_kv):
            _ = _read_str(f)
            kv_type = _read_i32(f)
            _skip_gguf_value(f, kv_type)

        names: list[str] = []
        tensor_info: dict[str, tuple[list[int], int, int]] = {}
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = _read_u32(f)
            dims = [_read_i64(f) for _ in range(n_dims)]
            ggml_type = _read_i32(f)
            offset = _read_u64(f)
            names.append(name)
            tensor_info[name] = (dims, ggml_type, offset)

        data_start = _align_up(f.tell(), 32)
    return names, tensor_info, data_start


def load_gguf_tensor(path: Path, info: tuple[list[int], int, int], data_start: int) -> np.ndarray:
    dims, ggml_type, offset = info
    dt = DTYPE_MAP_GGUF.get(ggml_type)
    if dt is None:
        raise ValueError(f"unsupported GGUF tensor type: {ggml_type}")
    shape = tuple(reversed(dims))
    n_items = int(np.prod(shape, dtype=np.int64))
    with path.open("rb") as f:
        f.seek(data_start + offset)
        arr = np.frombuffer(f.read(n_items * dt.itemsize), dtype=dt)
    return arr.reshape(shape)


def choose_tensor_name(names: list[str]) -> tuple[str | None, list[str]]:
    exact_candidates = (
        "q.s.layers.0.cb.embed",
        "q.s.layers.0.codebook.embed",
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed",
    )
    for c in exact_candidates:
        if c in names:
            return c, []

    fuzzy = []
    for name in names:
        low = name.lower()
        has_sem = ("semantic" in low) or ("q.s." in low)
        has_layer0 = ("layers.0" in low) or ("layer.0" in low) or ("layer0" in low)
        has_embed = "embed" in low
        has_codebook = ("codebook" in low) or (".cb." in low) or ("cb." in low)
        if has_sem and has_layer0 and (has_embed or has_codebook):
            fuzzy.append(name)

    if len(fuzzy) == 1:
        return fuzzy[0], fuzzy
    return None, fuzzy


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare GGUF vs HF Mimi semantic layer0 codebook.embed")
    parser.add_argument("--model-dir", default="/home/node/.openclaw/workspace/checkpoints/mimi")
    parser.add_argument("--gguf", default="mimi.gguf")
    parser.add_argument("--row-index", type=int, default=1049)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    gguf_path = Path(args.gguf)

    print("=== GGUF vs HF Codebook Embed Comparison ===")
    print(f"HF model dir: {model_dir}")
    print(f"GGUF path   : {gguf_path}")
    print(f"row index   : {args.row_index}")

    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    model = MimiModel.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    hf = (
        model.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.embed.detach()
        .to(torch.float32)
        .cpu()
        .numpy()
    )
    print(f"HF tensor name: quantizer.semantic_residual_vector_quantizer.layers[0].codebook.embed")
    print(f"HF shape      : {tuple(hf.shape)}")

    names, tensor_info, data_start = read_gguf_tensor_info(gguf_path)
    selected_name, fuzzy = choose_tensor_name(names)

    if selected_name is None:
        print("GGUF tensor selection: NOT FOUND")
        print("Fuzzy candidates (semantic + layer0 + embed/codebook):")
        if fuzzy:
            for n in fuzzy:
                print(f"  - {n}")
        else:
            print("  (none)")
        print("Names containing semantic/layer/embed/codebook:")
        for n in names:
            low = n.lower()
            if ("semantic" in low) or ("layer" in low) or ("embed" in low) or ("codebook" in low) or ("cb." in low):
                print(f"  - {n}")
        return 2

    gguf_raw = load_gguf_tensor(gguf_path, tensor_info[selected_name], data_start).astype(np.float32, copy=False)
    print(f"GGUF tensor name: {selected_name}")
    print(f"GGUF raw shape  : {tuple(gguf_raw.shape)}")

    aligned_layout = "unknown"
    if gguf_raw.shape == hf.shape:
        gguf_aligned = gguf_raw
        aligned_layout = "direct"
    elif gguf_raw.ndim == 2 and gguf_raw.T.shape == hf.shape:
        gguf_aligned = gguf_raw.T
        aligned_layout = "transposed"
    else:
        print(f"ERROR: shape mismatch cannot align GGUF to HF: gguf={gguf_raw.shape} hf={hf.shape}")
        return 3

    print(f"GGUF aligned shape: {tuple(gguf_aligned.shape)}")
    print(f"layout alignment  : {aligned_layout}")

    full_max = max_abs_diff(gguf_aligned, hf)
    full_allclose = bool(np.allclose(gguf_aligned, hf, atol=args.atol, rtol=0.0, equal_nan=False))

    row = int(args.row_index)
    if row < 0 or row >= hf.shape[0]:
        raise ValueError(f"row-index out of range: {row} for shape {hf.shape}")
    row_gguf = gguf_aligned[row]
    row_hf = hf[row]
    row_max = max_abs_diff(row_gguf, row_hf)
    row_allclose = bool(np.allclose(row_gguf, row_hf, atol=args.atol, rtol=0.0, equal_nan=False))

    print("--- Full Tensor ---")
    print(f"shape check          : {tuple(gguf_aligned.shape)} vs {tuple(hf.shape)}")
    print(f"max_abs_diff         : {full_max:.9g}")
    print(f"allclose(atol=1e-5)  : {full_allclose}")

    print("--- Row Test [1049] ---")
    print(f"row max_abs_diff     : {row_max:.9g}")
    print(f"row allclose(atol=1e-5): {row_allclose}")
    print(f"row first8 gguf      : {np.array2string(row_gguf[:8], precision=7, separator=', ')}")
    print(f"row first8 hf        : {np.array2string(row_hf[:8], precision=7, separator=', ')}")

    ok = full_allclose and row_allclose
    print(f"RESULT: {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
