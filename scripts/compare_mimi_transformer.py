#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np


STAGE_ORDER = ["ln1", "q", "k", "v", "q_rope", "k_rope", "attn_scores", "attn_ctx", "attn", "resid1", "ln2", "mlp_fc1", "mlp_act", "mlp_fc2", "resid2"]


def load_tensor(path: Path, channels: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"empty file: {path}")
    if arr.size % channels != 0:
        raise ValueError(f"invalid file size for {path}: {arr.size} not divisible by {channels}")
    return arr.reshape(channels, arr.size // channels)


def load_stage_tensor(path: Path, stage: str, channels: int, mlp_channels: int) -> np.ndarray:
    if stage != "attn_scores":
        stage_channels = mlp_channels if stage in ("mlp_fc1", "mlp_act") else channels
        return load_tensor(path, stage_channels)

    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"empty file: {path}")
    side = int(np.sqrt(arr.size))
    if side * side != arr.size:
        raise ValueError(f"invalid attn_scores size for {path}: {arr.size} is not a square matrix")
    return arr.reshape(side, side)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)
    a_std = float(np.std(a1))
    b_std = float(np.std(b1))
    if a_std == 0.0 or b_std == 0.0:
        return 1.0 if np.array_equal(a1, b1) else 0.0
    return float(np.corrcoef(a1, b1)[0, 1])


def discover_layers() -> list[int]:
    layers = set()
    pat = re.compile(r"_l(\d+)$")
    for p in Path("/tmp").glob("z_ln1_l*.bin"):
        m = pat.search(p.stem)
        if m is not None:
            layers.add(int(m.group(1)))
    return sorted(layers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Mimi transformer layer stage dumps (C++ vs HF)")
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--mlp-channels", type=int, default=2048)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--fail-on-divergence", action="store_true")
    args = parser.parse_args()

    layers = discover_layers()
    if not layers:
        raise FileNotFoundError("no C++ transformer dumps found (/tmp/z_ln1_l*.bin)")

    rows = []
    first_divergence = None

    for li in layers:
        for stage in STAGE_ORDER:
            cpp_path = Path(f"/tmp/z_{stage}_l{li}.bin")
            hf_path = Path(f"/tmp/mimi_dbg_z_{stage}_l{li}_hf.bin")

            if not cpp_path.exists() or not hf_path.exists():
                rows.append((li, stage, "missing", "missing", float("nan"), float("nan"), float("nan"), False))
                if first_divergence is None:
                    first_divergence = f"{stage}_l{li}"
                continue

            cpp = load_stage_tensor(cpp_path, stage, args.channels, args.mlp_channels)
            hf = load_stage_tensor(hf_path, stage, args.channels, args.mlp_channels)

            if cpp.shape != hf.shape:
                rows.append((li, stage, str(cpp.shape), str(hf.shape), float("nan"), float("nan"), float("nan"), False))
                if first_divergence is None:
                    first_divergence = f"{stage}_l{li}"
                continue

            diff = cpp - hf
            max_abs_diff = float(np.max(np.abs(diff)))
            mse = float(np.mean(diff * diff))
            corr = corrcoef(cpp, hf)
            ok = bool(np.allclose(cpp, hf, atol=args.atol, rtol=args.rtol))
            if first_divergence is None and not ok:
                first_divergence = f"{stage}_l{li}"
            rows.append((li, stage, str(cpp.shape), str(hf.shape), max_abs_diff, mse, corr, ok))

    print("layer | stage    | shape_cpp   | shape_hf    | max_abs_diff | mse          | corr       | allclose")
    print("------+----------+-------------+-------------+--------------+--------------+------------+---------")
    for li, stage, sc, sh, mad, mse, corr, ok in rows:
        print(f"{li:>5} | {stage:<8} | {sc:<11} | {sh:<11} | {mad:>12.6g} | {mse:>12.6g} | {corr:>10.6f} | {str(ok):<8}")

    if first_divergence is None:
        print("FIRST_DIVERGENCE: none")
        return 0

    print(f"FIRST_DIVERGENCE: {first_divergence}")
    return 1 if args.fail_on_divergence else 0


if __name__ == "__main__":
    raise SystemExit(main())
