#!/usr/bin/env python3
import argparse

import numpy as np


STAGES = [
    ("z_before_upsample", "/tmp/mimi_dbg_z_before_upsample.bin", "/tmp/mimi_dbg_z_before_upsample_hf.bin", 512),
    ("z_after_upsample", "/tmp/mimi_dbg_z_after_upsample.bin", "/tmp/mimi_dbg_z_after_upsample_hf.bin", 512),
    ("z_after_transformer", "/tmp/mimi_dbg_z_after_transformer.bin", "/tmp/mimi_dbg_z_after_transformer_hf.bin", 512),
    ("z_dec_l0_conv", "/tmp/mimi_dbg_z_dec_l0_conv.bin", "/tmp/mimi_dbg_z_dec_l0_conv_hf.bin", 1024),
    ("z_dec_l1_elu", "/tmp/mimi_dbg_z_dec_l1_elu.bin", "/tmp/mimi_dbg_z_dec_l1_elu_hf.bin", 1024),
    ("z_dec_l2_convtr", "/tmp/mimi_dbg_z_dec_l2_convtr.bin", "/tmp/mimi_dbg_z_dec_l2_convtr_hf.bin", 512),
    ("z_dec_l3_resblock", "/tmp/mimi_dbg_z_dec_l3_resblock.bin", "/tmp/mimi_dbg_z_dec_l3_resblock_hf.bin", 512),
    ("z_dec_l4_elu", "/tmp/mimi_dbg_z_dec_l4_elu.bin", "/tmp/mimi_dbg_z_dec_l4_elu_hf.bin", 512),
    ("z_dec_l5_convtr", "/tmp/mimi_dbg_z_dec_l5_convtr.bin", "/tmp/mimi_dbg_z_dec_l5_convtr_hf.bin", 256),
    ("z_dec_l6_resblock", "/tmp/mimi_dbg_z_dec_l6_resblock.bin", "/tmp/mimi_dbg_z_dec_l6_resblock_hf.bin", 256),
    ("z_dec_l7_elu", "/tmp/mimi_dbg_z_dec_l7_elu.bin", "/tmp/mimi_dbg_z_dec_l7_elu_hf.bin", 256),
    ("z_dec_l8_convtr", "/tmp/mimi_dbg_z_dec_l8_convtr.bin", "/tmp/mimi_dbg_z_dec_l8_convtr_hf.bin", 128),
    ("z_dec_l9_resblock", "/tmp/mimi_dbg_z_dec_l9_resblock.bin", "/tmp/mimi_dbg_z_dec_l9_resblock_hf.bin", 128),
    ("z_dec_l10_elu", "/tmp/mimi_dbg_z_dec_l10_elu.bin", "/tmp/mimi_dbg_z_dec_l10_elu_hf.bin", 128),
    ("z_dec_l11_convtr", "/tmp/mimi_dbg_z_dec_l11_convtr.bin", "/tmp/mimi_dbg_z_dec_l11_convtr_hf.bin", 64),
    ("z_dec_l12_resblock", "/tmp/mimi_dbg_z_dec_l12_resblock.bin", "/tmp/mimi_dbg_z_dec_l12_resblock_hf.bin", 64),
    ("z_dec_l13_elu", "/tmp/mimi_dbg_z_dec_l13_elu.bin", "/tmp/mimi_dbg_z_dec_l13_elu_hf.bin", 64),
    ("y_pre_tanh", "/tmp/mimi_dbg_y_pre_tanh.bin", "/tmp/mimi_dbg_y_pre_tanh_hf.bin", 1),
]


def load_tensor(path: str, channels: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"empty file: {path}")
    if arr.size % channels != 0:
        raise ValueError(f"invalid file size for {path}: {arr.size} values not divisible by {channels}")
    return arr.reshape(channels, arr.size // channels)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)
    a_std = float(np.std(a1))
    b_std = float(np.std(b1))
    if a_std == 0.0 or b_std == 0.0:
        return 1.0 if np.array_equal(a1, b1) else 0.0
    return float(np.corrcoef(a1, b1)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Mimi post-latent checkpoints")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--fail-on-divergence", action="store_true")
    args = parser.parse_args()

    rows = []
    first_divergence = None

    for name, cpp_path, hf_path, channels in STAGES:
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

    print("stage              | shape_cpp   | shape_hf    | max_abs_diff | mse          | corr       | allclose")
    print("-------------------+-------------+-------------+--------------+--------------+------------+---------")
    for name, shape_cpp, shape_hf, max_abs_diff, mse, corr, allclose in rows:
        print(f"{name:<18} | {shape_cpp:<11} | {shape_hf:<11} | {max_abs_diff:>12.6g} | {mse:>12.6g} | {corr:>10.6f} | {str(allclose):<8}")

    if first_divergence is None:
        print("FIRST_DIVERGENCE: none")
        return 0

    print(f"FIRST_DIVERGENCE: {first_divergence}")
    return 1 if args.fail_on_divergence else 0


if __name__ == "__main__":
    raise SystemExit(main())
