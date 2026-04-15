#!/usr/bin/env python3
"""Smoke test for Chatterbox S3G builtin conditioning export."""

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


def assert_contains(output: str, needle: str) -> None:
    if needle not in output:
        raise AssertionError(f"missing expected text: {needle}\n--- output ---\n{output}")


def main() -> int:
    try:
        import safetensors.numpy  # noqa: F401
        import torch
    except ModuleNotFoundError:
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])
        raise

    if not INSPECT.is_file():
        raise RuntimeError(f"Missing inspect binary: {INSPECT}. Build the project first.")

    from safetensors.numpy import save_file

    with tempfile.TemporaryDirectory(prefix="codec-chatterbox-s3g-conds-") as td:
        root = Path(td)
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

        torch.save(
            {
                "gen": {
                    "prompt_token": torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.int64),
                    "prompt_token_len": torch.tensor([5], dtype=torch.int64),
                    "prompt_feat": torch.zeros(1, 10, 80, dtype=torch.float32),
                    "prompt_feat_len": None,
                    "embedding": torch.ones(1, 192, dtype=torch.float32),
                }
            },
            ckpt / "conds.pt",
        )

        gguf = root / "s3g.gguf"
        run(
            [
                sys.executable,
                str(CONVERT),
                "--checkpoint-path",
                str(ckpt),
                "--model-type",
                "chatterbox_s3g",
                "--output",
                str(gguf),
            ]
        )

        out = run([str(INSPECT), str(gguf)])
        assert_contains(out, "arch:       Chatterbox-S3G")
        assert_contains(out, "chatterbox_s3g.has_builtin_conditioning = true")
        assert_contains(out, "chatterbox_s3g.cond.prompt_token_len = 5")
        assert_contains(out, "chatterbox_s3g.cond.prompt_feat_frames = 10")
        assert_contains(out, "chatterbox_s3g.cond.prompt_feat_dim = 80")
        assert_contains(out, "chatterbox_s3g.cond.embedding_dim = 192")
        assert_contains(out, "chatterbox_s3g.cond.prompt_token = <array:u32, n=5>")

    print("chatterbox S3G builtin-conditioning smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
