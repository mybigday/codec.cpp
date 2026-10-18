#!/usr/bin/env python3
"""
Re-quantize a codec.cpp GGUF tensor by tensor (mixed precision).

The precision of every tensor is the publisher's choice: codec.cpp runs any
mix of F32 / F16 / BF16 / block-quantized weights.  This tool rewrites an
existing GGUF (any model, metadata copied as is) with a per-tensor type plan.

Rules, first match wins:
  --tensor-type REGEX=TYPE   (repeatable; re.search on the tensor name)
  --type TYPE                for every other float tensor with >= --min-dims dims
  otherwise                  the source type is kept

A tensor whose row length (ne0) is not a multiple of the target block size
gets --fallback, else keeps its source type (reported).  Integer tensors are always copied.

--flatten REGEX re-declares matching 3-D conv1d kernels (K, C_in, C_out) as
(K*C_in, 1, C_out) before typing them: the same bytes, but rows long enough to
block-quantize (codec.cpp's conv1d takes either form).  Only for conv1d kernels.
--ci-major REGEX permutes matching ConvTranspose1d kernels (K, C_out, C_in) to
(C_in, C_out, K), the layout codec.cpp runs them in at any stored type.
Depthwise kernels have no quantizable layout.

Examples:
  # Q8_0 everywhere it fits, codebooks and norms kept F16
  python scripts/quantize-gguf.py mimi.gguf mimi-q8.gguf --type Q8_0 \\
      --tensor-type 'embed|codebook=F16' --tensor-type 'norm=F16'
  # only print the plan
  python scripts/quantize-gguf.py dac.gguf /dev/null --type Q4_0 --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402

from utils import quantization as repo_quant  # noqa: E402

FLOAT_TYPES = {T.F32, T.F16, T.BF16}
NATIVE_QUANT = {T.BF16, T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q8_0}  # gguf.quants.quantize
REPO_QUANT = {T.Q4_K: repo_quant.quantize_tensor_q4_k_m, T.Q5_K: repo_quant.quantize_tensor_q5_k_m}
TARGETS = {"F32": T.F32, "F16": T.F16, **{t.name: t for t in NATIVE_QUANT | set(REPO_QUANT)}}


def parse_type(s: str) -> T:
    t = TARGETS.get(s.upper())
    if t is None:
        raise argparse.ArgumentTypeError(f"unsupported type {s!r} (choose from {', '.join(sorted(TARGETS))})")
    return t


def parse_rule(s: str):
    pattern, sep, type_name = s.rpartition("=")
    if not sep or not pattern:
        raise argparse.ArgumentTypeError(f"expected REGEX=TYPE, got {s!r}")
    return re.compile(pattern), parse_type(type_name)


def to_f32(tensor) -> np.ndarray:
    """numpy array in ggml order reversed (last axis = ne0), float32."""
    shape = tuple(int(n) for n in reversed(tensor.shape))
    if tensor.tensor_type == T.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(shape)
    if tensor.tensor_type == T.F16:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(shape)
    return gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32).reshape(shape)


def encode(arr: np.ndarray, qtype: T) -> np.ndarray:
    if qtype == T.F32:
        return arr.astype(np.float32)
    if qtype == T.F16:
        return arr.astype(np.float16)
    if qtype in REPO_QUANT:
        raw = np.frombuffer(REPO_QUANT[qtype](arr, row_axis=-1), dtype=np.uint8)
        block, type_size = gguf.GGML_QUANT_SIZES[qtype]
        return raw.reshape(*arr.shape[:-1], arr.shape[-1] // block * type_size)
    return gguf.quants.quantize(arr, qtype)


def relayout(tensor, flatten, ci_major):
    """(ne as stored in the output, permutation of the ggml axes or None)."""
    ne = [int(n) for n in tensor.shape]
    if len(ne) != 3 or tensor.tensor_type not in FLOAT_TYPES:
        return ne, None
    if flatten is not None and flatten.search(tensor.name):
        return [ne[0] * ne[1], 1, ne[2]], None
    if ci_major is not None and ci_major.search(tensor.name):
        return [ne[2], ne[1], ne[0]], (2, 1, 0)
    return ne, None


def plan_type(tensor, rules, default, min_dims, ne, fallback=None):
    src = tensor.tensor_type
    if src not in FLOAT_TYPES and src not in NATIVE_QUANT and src not in REPO_QUANT:
        return src, None  # integer / exotic tensors are copied
    want = None
    for pattern, qtype in rules:
        if pattern.search(tensor.name):
            want = qtype
            break
    if want is None:
        if default is None or len(ne) < min_dims:
            return src, None
        want = default
    block, _ = gguf.GGML_QUANT_SIZES[want]
    ne0 = ne[0]
    if ne0 % block != 0:
        if fallback is not None and ne0 % gguf.GGML_QUANT_SIZES[fallback][0] == 0:
            return fallback, None
        return src, f"ne0={ne0} not a multiple of {block} for {want.name}"
    return want, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--type", type=parse_type, default=None, help="default target type for float tensors")
    ap.add_argument("--tensor-type", type=parse_rule, action="append", default=[], metavar="REGEX=TYPE",
                    help="per-tensor override, first match wins (repeatable)")
    ap.add_argument("--fallback", type=parse_type, default=None,
                    help="type for tensors whose rows don't fit the requested type's block (default: keep the source type)")
    ap.add_argument("--min-dims", type=int, default=2,
                    help="--type only applies to tensors with at least this many dims (default 2)")
    ap.add_argument("--flatten", type=re.compile, default=None, metavar="REGEX",
                    help="store matching 3-D conv1d kernels flattened to (K*C_in, 1, C_out)")
    ap.add_argument("--ci-major", type=re.compile, default=None, metavar="REGEX",
                    help="store matching 3-D ConvTranspose1d kernels as (C_in, C_out, K)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args()

    reader = gguf.GGUFReader(str(args.input))
    arch_field = reader.fields.get(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else "codec"

    plan = []
    for tensor in reader.tensors:
        ne, perm = relayout(tensor, args.flatten, args.ci_major)
        qtype, why = plan_type(tensor, args.tensor_type, args.type, args.min_dims, ne, args.fallback)
        plan.append((tensor, qtype, why, ne, perm))

    counts = {}
    for tensor, qtype, why, ne, _ in plan:
        counts[qtype.name] = counts.get(qtype.name, 0) + 1
        if why is not None:
            print(f"keep {tensor.tensor_type.name:5s} {tensor.name}: {why}")
        elif args.dry_run:
            print(f"{tensor.tensor_type.name:5s} -> {qtype.name:5s} {tensor.name} {ne}")
    print("tensor types:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.dry_run:
        return 0

    writer = gguf.GGUFWriter(str(args.output), arch=arch, endianess=reader.endianess)
    alignment = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = int(alignment.contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    for tensor, qtype, _, ne, perm in plan:
        moved = ne != [int(n) for n in tensor.shape] or perm is not None
        if qtype == tensor.tensor_type and not moved:
            data = tensor.data
        else:
            arr = to_f32(tensor)                                   # numpy axes = ggml axes reversed
            if perm is not None:
                arr = np.ascontiguousarray(np.transpose(arr, [2 - perm[2 - i] for i in range(3)]))
            data = encode(arr.reshape(tuple(reversed(ne))), qtype)
        # numpy F32/F16/integer arrays carry their type; BF16 and block types are raw bytes
        raw = data.dtype == np.uint8 and qtype not in (T.I8,)
        writer.add_tensor(tensor.name, data, raw_dtype=qtype if raw else None)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
