#!/usr/bin/env python3
"""Smoke test for Chatterbox S3G builtin conditioning export."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from _common import REPO, run_cmd

import codec_cpp

CONVERT = REPO / "scripts" / "convert-to-gguf.py"


def main() -> int:
    try:
        import safetensors.numpy  # noqa: F401
        import torch
    except ModuleNotFoundError:
        venv_python = REPO / ".venv" / "bin" / "python"
        if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])
        raise

    real_ckpt = REPO / "models" / "chatterbox"
    if not (real_ckpt / "s3gen.safetensors").is_file() or not (real_ckpt / "conds.pt").is_file():
        print(f"chatterbox S3G builtin-conditioning smoke test SKIPPED: {real_ckpt} not present")
        return 0

    with tempfile.TemporaryDirectory(prefix="codec-chatterbox-s3g-conds-") as td:
        root = Path(td)
        gguf = root / "s3g.gguf"
        run_cmd([sys.executable, CONVERT, "--checkpoint-path", real_ckpt, "--model-type", "chatterbox_s3g",
                 "--output", gguf], cwd=REPO)

        with codec_cpp.Model(gguf) as model:
            md = model.metadata()
            assert model.arch_name == "Chatterbox-S3G", model.arch_name
            assert model.speaker_names == ["default"], model.speaker_names
            assert md.get("chatterbox_s3g.spk.default") == "default", md.get("chatterbox_s3g.spk.default")
            assert md.get("chatterbox_s3g.spk.encoder_hash"), "missing chatterbox_s3g.spk.encoder_hash"

    print("chatterbox S3G builtin-conditioning smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
