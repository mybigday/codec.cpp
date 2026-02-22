#!/usr/bin/env python3
import argparse

import numpy as np


CHECKPOINTS = [
    ("sem_decoded", "/tmp/mimi_debug_sem_decoded.bin", "/tmp/mimi_debug_sem_decoded_hf.bin", 256),
    ("acu_decoded", "/tmp/mimi_debug_acu_decoded.bin", "/tmp/mimi_debug_acu_decoded_hf.bin", 256),
    ("sem_acu_sum", "/tmp/mimi_debug_sem_acu_sum.bin", "/tmp/mimi_debug_sem_acu_sum_hf.bin", 256),
    ("latent_final", "/tmp/mimi_debug_latent_final.bin", "/tmp/mimi_debug_latent_final_hf.bin", 512),
]


def load_tensor(path: str, channels: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"empty file: {path}")
    if arr.size % channels != 0:
        raise ValueError(f"invalid file size for {path}: {arr.size} values not divisible by {channels}")
    frames = arr.size // channels
    return arr.reshape(channels, frames)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)
    a_std = float(np.std(a1))
    b_std = float(np.std(b1))
    if a_std == 0.0 or b_std == 0.0:
        return 1.0 if np.array_equal(a1, b1) else 0.0
    return float(np.corrcoef(a1, b1)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Mimi RVQ debug checkpoints")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--fail-on-divergence", action="store_true")
    args = parser.parse_args()

    rows = []
    first_divergence = None

    for name, cpp_path, hf_path, channels in CHECKPOINTS:
        cpp = load_tensor(cpp_path, channels)
        hf = load_tensor(hf_path, channels)

        if cpp.shape != hf.shape:
            max_abs_diff = float("nan")
            mse = float("nan")
            corr = float("nan")
            allclose = False
        else:
            diff = cpp - hf
            max_abs_diff = float(np.max(np.abs(diff)))
            mse = float(np.mean(diff * diff))
            corr = corrcoef(cpp, hf)
            allclose = bool(np.allclose(cpp, hf, atol=args.atol, rtol=args.rtol))

        if first_divergence is None and not allclose:
            first_divergence = name

        rows.append((name, str(cpp.shape), str(hf.shape), max_abs_diff, mse, corr, allclose))

    print("checkpoint     | shape_cpp   | shape_hf    | max_abs_diff | mse          | corr       | allclose")
    print("---------------+-------------+-------------+--------------+--------------+------------+---------")
    for name, shape_cpp, shape_hf, max_abs_diff, mse, corr, allclose in rows:
        print(f"{name:<14} | {shape_cpp:<11} | {shape_hf:<11} | {max_abs_diff:>12.6g} | {mse:>12.6g} | {corr:>10.6f} | {str(allclose):<8}")

    if first_divergence is None:
        print("FIRST_DIVERGENCE: none")
        return 0

    print(f"FIRST_DIVERGENCE: {first_divergence}")
    return 1 if args.fail_on_divergence else 0


if __name__ == "__main__":
    raise SystemExit(main())
