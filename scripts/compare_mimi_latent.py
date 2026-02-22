#!/usr/bin/env python3
import argparse

import numpy as np


def load_latent(path: str, channels: int = 512) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"empty latent file: {path}")
    if arr.size % channels != 0:
        raise ValueError(f"file size is not divisible by {channels}: {path}")
    frames = arr.size // channels
    return arr.reshape(channels, frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Mimi latent dumps")
    parser.add_argument("--cpp", default="/tmp/mimi_latent_cpp.bin")
    parser.add_argument("--hf", default="/tmp/mimi_latent_hf.bin")
    args = parser.parse_args()

    cpp = load_latent(args.cpp)
    hf = load_latent(args.hf)

    if cpp.shape != hf.shape:
        print("Latent comparison:")
        print(f"  shape_cpp={cpp.shape}")
        print(f"  shape_hf={hf.shape}")
        print("  max_abs_diff=nan")
        print("  mse=nan")
        print("  corr=nan")
        print("  allclose=False")
        return 1

    diff = cpp - hf
    max_abs_diff = float(np.max(np.abs(diff)))
    mse = float(np.mean(diff * diff))

    cpp_flat = cpp.reshape(-1)
    hf_flat = hf.reshape(-1)
    cpp_std = float(np.std(cpp_flat))
    hf_std = float(np.std(hf_flat))
    if cpp_std == 0.0 or hf_std == 0.0:
        corr = 1.0 if np.array_equal(cpp_flat, hf_flat) else 0.0
    else:
        corr = float(np.corrcoef(cpp_flat, hf_flat)[0, 1])

    allclose = bool(np.allclose(cpp, hf, atol=1e-3, rtol=1e-4))

    print("Latent comparison:")
    print(f"  shape_cpp={cpp.shape}")
    print(f"  shape_hf={hf.shape}")
    print(f"  max_abs_diff={max_abs_diff}")
    print(f"  mse={mse}")
    print(f"  corr={corr}")
    print(f"  allclose={allclose}")

    return 0 if allclose else 1


if __name__ == "__main__":
    raise SystemExit(main())
