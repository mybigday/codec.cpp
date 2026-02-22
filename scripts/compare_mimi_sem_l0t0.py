#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def parse_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_vec(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"empty vector file: {path}")
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Mimi semantic layer0/t0 lookup vectors")
    parser.add_argument("--cpp-txt", default="/tmp/mimi_debug_sem_layer0_t0.txt")
    parser.add_argument("--hf-txt", default="/tmp/mimi_debug_sem_layer0_t0_hf.txt")
    parser.add_argument("--cpp-bin", default="/tmp/mimi_debug_sem_layer0_t0.bin")
    parser.add_argument("--hf-bin", default="/tmp/mimi_debug_sem_layer0_t0_hf.bin")
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    cpp_txt = parse_kv(Path(args.cpp_txt))
    hf_txt = parse_kv(Path(args.hf_txt))

    cpp = load_vec(Path(args.cpp_bin))
    hf = load_vec(Path(args.hf_bin))

    same_shape = cpp.shape == hf.shape
    max_abs_diff = float(np.max(np.abs(cpp - hf))) if same_shape else float("inf")
    is_close = bool(np.allclose(cpp, hf, atol=args.atol, rtol=0.0)) if same_shape else False

    print("=== Mimi Semantic Layer0/Timestep0 Lookup Comparison ===")
    print(f"code_index_cpp: raw={cpp_txt.get('code_index_raw', '?')} clamped={cpp_txt.get('code_index_clamped', '?')}")
    print(f"code_index_hf : raw={hf_txt.get('code_index_raw', '?')} clamped={hf_txt.get('code_index_clamped', '?')}")
    print(f"shape_cpp: {cpp.shape}")
    print(f"shape_hf : {hf.shape}")
    print(f"max_abs_diff: {max_abs_diff:.9g}")
    print(f"allclose(atol={args.atol:g}): {is_close}")

    print("first5:")
    for i in range(min(5, cpp.size, hf.size)):
        print(f"  [{i:3d}] cpp={cpp[i]: .9f} hf={hf[i]: .9f} diff={cpp[i]-hf[i]: .9f}")

    print("last5:")
    for i in range(max(0, min(cpp.size, hf.size) - 5), min(cpp.size, hf.size)):
        print(f"  [{i:3d}] cpp={cpp[i]: .9f} hf={hf[i]: .9f} diff={cpp[i]-hf[i]: .9f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
