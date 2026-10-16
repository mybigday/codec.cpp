#!/usr/bin/env python3
"""Smoke E2E for Chatterbox S3 converters.

Creates tiny fake checkpoints, converts them through the public converter CLI,
and verifies that codec.cpp can load the resulting GGUF files.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from _common import REPO, run_cmd

import codec_cpp

CONVERT = REPO / "scripts" / "convert-to-gguf.py"


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


def check_model(gguf: Path, *, arch: str, has_encoder: bool, has_decoder: bool) -> None:
    with codec_cpp.Model(gguf) as m:
        got = (m.arch_name, m.sample_rate, m.has_encoder, m.has_decoder, m.hop_size, m.codebook_size)
    want = (arch, 24000, has_encoder, has_decoder, 960, 6561)
    if got != want:
        raise AssertionError(f"{gguf.name}: (arch, sample_rate, has_encoder, has_decoder, hop, codebook) "
                             f"= {got}, want {want}")


def main() -> int:
    try:
        import safetensors.numpy  # noqa: F401
    except ModuleNotFoundError:
        venv_python = REPO / ".venv" / "bin" / "python"
        if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])
        raise

    with tempfile.TemporaryDirectory(prefix="codec-chatterbox-smoke-") as td:
        root = Path(td)
        s3t_ckpt = make_s3t_checkpoint(root)
        s3g_ckpt = make_s3g_checkpoint(root)
        s3t_gguf = root / "s3t.gguf"
        s3g_gguf = root / "s3g.gguf"

        for ckpt, kind, gguf in ((s3t_ckpt, "chatterbox_s3t", s3t_gguf), (s3g_ckpt, "chatterbox_s3g", s3g_gguf)):
            run_cmd([sys.executable, CONVERT, "--checkpoint-path", ckpt, "--model-type", kind, "--output", gguf],
                    cwd=REPO)

        check_model(s3t_gguf, arch="Chatterbox-S3T", has_encoder=True, has_decoder=False)
        check_model(s3g_gguf, arch="Chatterbox-S3G", has_encoder=False, has_decoder=True)

    print("chatterbox converter smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
