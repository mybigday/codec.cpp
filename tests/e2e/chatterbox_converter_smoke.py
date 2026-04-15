#!/usr/bin/env python3
"""Smoke E2E for Chatterbox S3 converters.

Creates tiny fake checkpoints, converts them through the public converter CLI,
and verifies that codec.cpp can load the resulting GGUF files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT = REPO_ROOT / "scripts" / "convert-to-gguf.py"
INSPECT = REPO_ROOT / "build" / "inspect-codec"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_s3t_checkpoint(root: Path) -> Path:
    from safetensors.numpy import save_file

    ckpt = root / "s3t"
    ckpt.mkdir(parents=True, exist_ok=True)
    write_json(
        ckpt / "config.json",
        {
            "model_type": "chatterbox_s3t",
            "sample_rate": 24000,
            "encode_sample_rate": 16000,
            "hop_size": 960,
            "n_q": 1,
            "codebook_size": 6561,
            "n_fft": 400,
            "win_length": 400,
            "n_mels": 128,
        },
    )
    save_file(
        {
            "encoder.dummy.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
        },
        str(ckpt / "model.safetensors"),
    )
    return ckpt


def make_s3g_checkpoint(root: Path) -> Path:
    from safetensors.numpy import save_file

    ckpt = root / "s3g"
    ckpt.mkdir(parents=True, exist_ok=True)
    write_json(
        ckpt / "config.json",
        {
            "model_type": "chatterbox_s3g",
            "sample_rate": 24000,
            "hop_size": 960,
            "n_q": 1,
            "codebook_size": 6561,
        },
    )
    save_file(
        {
            "decoder.dummy.weight": np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4),
        },
        str(ckpt / "s3gen.safetensors"),
    )
    return ckpt


def assert_contains(output: str, needle: str) -> None:
    if needle not in output:
        raise AssertionError(f"missing expected text: {needle}\n--- output ---\n{output}")


def main() -> int:
    try:
        import safetensors.numpy  # noqa: F401
    except ModuleNotFoundError:
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])
        raise

    if not INSPECT.is_file():
        raise RuntimeError(f"Missing inspect binary: {INSPECT}. Build the project first.")

    with tempfile.TemporaryDirectory(prefix="codec-chatterbox-smoke-") as td:
        root = Path(td)
        s3t_ckpt = make_s3t_checkpoint(root)
        s3g_ckpt = make_s3g_checkpoint(root)
        s3t_gguf = root / "s3t.gguf"
        s3g_gguf = root / "s3g.gguf"

        run(
            [
                sys.executable,
                str(CONVERT),
                "--checkpoint-path",
                str(s3t_ckpt),
                "--model-type",
                "chatterbox_s3t",
                "--output",
                str(s3t_gguf),
            ]
        )
        run(
            [
                sys.executable,
                str(CONVERT),
                "--checkpoint-path",
                str(s3g_ckpt),
                "--model-type",
                "chatterbox_s3g",
                "--output",
                str(s3g_gguf),
            ]
        )

        out_s3t = run([str(INSPECT), str(s3t_gguf)])
        assert_contains(out_s3t, "arch:       Chatterbox-S3T")
        assert_contains(out_s3t, "sample_rate    24000")
        assert_contains(out_s3t, "has_encoder    true")
        assert_contains(out_s3t, "has_decoder    false")
        assert_contains(out_s3t, "hop_size       960")
        assert_contains(out_s3t, "codebook_size  6561")

        out_s3g = run([str(INSPECT), str(s3g_gguf)])
        assert_contains(out_s3g, "arch:       Chatterbox-S3G")
        assert_contains(out_s3g, "sample_rate    24000")
        assert_contains(out_s3g, "has_encoder    false")
        assert_contains(out_s3g, "has_decoder    true")
        assert_contains(out_s3g, "hop_size       960")
        assert_contains(out_s3g, "codebook_size  6561")

    print("chatterbox converter smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
