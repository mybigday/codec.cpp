#!/usr/bin/env python3
"""Exact-parity E2E for Chatterbox S3T.

Converts the real downloaded checkpoint to GGUF, runs `codec-cli encode`,
and checks that the emitted token IDs exactly match a PyTorch reference
implementation of the S3TokenizerV2 encoder path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT = REPO_ROOT / "scripts" / "convert-to-gguf.py"
CODEC = REPO_ROOT / "build" / "codec-cli"
CHECKPOINT = REPO_ROOT / "models" / "chatterbox"
S3GEN = CHECKPOINT / "s3gen.safetensors"
INPUT_WAV = REPO_ROOT / "input_audio" / "10_2.wav"
SLICE_OFFSET = 16000
SLICE_SAMPLES = 7040
ENCODE_TIME_LIMIT_SEC = 5.0


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


def read_wav_pcm16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        payload = wf.readframes(n)
    if ch != 1 or sw != 2:
        raise RuntimeError(f"{path}: expected mono PCM16 WAV, got channels={ch} sample_width={sw}")
    return sr, np.frombuffer(payload, dtype=np.int16).copy()


def write_wav_pcm16(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def load_existing_test_slice() -> tuple[np.ndarray, np.ndarray]:
    sr, pcm = read_wav_pcm16(INPUT_WAV)
    if sr != 16000:
        raise RuntimeError(f"{INPUT_WAV}: expected 16 kHz WAV, got {sr}")
    if pcm.size < SLICE_OFFSET + SLICE_SAMPLES:
        raise RuntimeError(
            f"{INPUT_WAV}: not enough audio for slice offset={SLICE_OFFSET} len={SLICE_SAMPLES}"
        )
    pcm = pcm[SLICE_OFFSET : SLICE_OFFSET + SLICE_SAMPLES].copy()
    wav_q = np.clip(pcm.astype(np.float32) / 32768.0, -1.0, 1.0)
    return pcm, wav_q


def compute_log_mel_like_codec(wav: np.ndarray, mel_filters: np.ndarray, window: np.ndarray) -> np.ndarray:
    n_fft = 400
    hop = 160
    token_hop = 640
    n_mels, n_bins = mel_filters.shape
    assert n_bins == n_fft // 2 + 1

    padded_pcm = ((wav.size + token_hop - 1) // token_hop) * token_hop
    mel_frames = padded_pcm // hop
    pcm_pad = np.zeros(padded_pcm, dtype=np.float32)
    pcm_pad[: wav.size] = wav

    center_pad = n_fft // 2

    def reflect_index(idx: int, length: int) -> int:
        if length <= 1:
            return 0
        while idx < 0 or idx >= length:
            if idx < 0:
                idx = -idx
            else:
                idx = 2 * length - 2 - idx
        return idx

    centered = np.empty(padded_pcm + center_pad * 2, dtype=np.float32)
    for i in range(centered.size):
        centered[i] = pcm_pad[reflect_index(i - center_pad, padded_pcm)]

    k = np.arange(n_bins, dtype=np.float64)[:, None]
    n = np.arange(n_fft, dtype=np.float64)[None, :]
    ang = 2.0 * np.pi * k * n / float(n_fft)
    cos_table = np.cos(ang)
    sin_table = np.sin(ang)

    out = np.empty((n_mels, mel_frames), dtype=np.float32)
    power = np.empty(n_bins, dtype=np.float32)
    global_max = -np.inf
    for ti in range(mel_frames):
        start = ti * hop
        frame = centered[start : start + n_fft].astype(np.float64) * window.astype(np.float64)
        re = cos_table @ frame
        im = -(sin_table @ frame)
        power[:] = (re * re + im * im).astype(np.float32)

        mel_vec = mel_filters @ power
        log_spec = np.log10(np.maximum(mel_vec, 1.0e-10)).astype(np.float32, copy=False)
        out[:, ti] = log_spec
        global_max = max(global_max, float(log_spec.max()))

    floor_val = global_max - 8.0
    out = np.maximum(out, floor_val)
    out = (out + 4.0) * 0.25
    return out


def compute_reference_codes(wav: np.ndarray) -> np.ndarray:
    import torch
    from safetensors import safe_open

    state: dict[str, torch.Tensor] = {}
    with safe_open(str(S3GEN), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("tokenizer."):
                state[key[len("tokenizer."):]] = f.get_tensor(key)

    mel_filters = state["_mel_filters"].cpu().numpy().astype(np.float32, copy=False)
    if "window" in state:
        window = state["window"].cpu().numpy().astype(np.float32, copy=False)
    else:
        window = np.array(
            [0.5 - 0.5 * np.cos(2.0 * np.pi * i / 400.0) for i in range(400)],
            dtype=np.float32,
        )
    x = torch.from_numpy(compute_log_mel_like_codec(wav, mel_filters, window)).unsqueeze(0)

    x = torch.nn.functional.gelu(
        torch.nn.functional.conv1d(
            x,
            state["encoder.conv1.weight"],
            state["encoder.conv1.bias"],
            stride=2,
            padding=1,
        )
    )
    x = torch.nn.functional.gelu(
        torch.nn.functional.conv1d(
            x,
            state["encoder.conv2.weight"],
            state["encoder.conv2.bias"],
            stride=2,
            padding=1,
        )
    )
    x = x.permute(0, 2, 1)

    head_dim = 64
    n_heads = 20
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(x.size(1))
    fr = torch.outer(t, freqs).float()
    cos = torch.cat((fr.cos(), fr.cos()), dim=-1).unsqueeze(0).unsqueeze(2)
    sin = torch.cat((fr.sin(), fr.sin()), dim=-1).unsqueeze(0).unsqueeze(2)

    def rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ql, qr = q[..., : head_dim // 2], q[..., head_dim // 2 :]
        kl, kr = k[..., : head_dim // 2], k[..., head_dim // 2 :]
        q_rot = torch.cat((-qr, ql), dim=-1)
        k_rot = torch.cat((-kr, kl), dim=-1)
        return q * cos + q_rot * sin, k * cos + k_rot * sin

    for li in range(6):
        p = f"encoder.blocks.{li}"
        h = torch.nn.functional.layer_norm(
            x.float(),
            (1280,),
            state[p + ".attn_ln.weight"].float(),
            state[p + ".attn_ln.bias"].float(),
            1e-5,
        ).to(x.dtype)
        q = torch.nn.functional.linear(h, state[p + ".attn.query.weight"], state[p + ".attn.query.bias"])
        k = torch.nn.functional.linear(h, state[p + ".attn.key.weight"], None)
        v = torch.nn.functional.linear(h, state[p + ".attn.value.weight"], state[p + ".attn.value.bias"])

        q = q.view(q.shape[0], q.shape[1], n_heads, head_dim)
        k = k.view(k.shape[0], k.shape[1], n_heads, head_dim)
        v4 = v.view(v.shape[0], v.shape[1], n_heads, head_dim)
        q, k = rope(q, k)

        scale = head_dim ** -0.25
        attn = torch.softmax(
            (q.permute(0, 2, 1, 3) * scale) @ (k.permute(0, 2, 3, 1) * scale).float(),
            dim=-1,
        ).to(q.dtype)
        wv = (attn @ v4.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).flatten(start_dim=2)
        attn_out = torch.nn.functional.linear(wv, state[p + ".attn.out.weight"], state[p + ".attn.out.bias"])

        fsmn = torch.nn.functional.pad(v.transpose(1, 2), (15, 15))
        fsmn = torch.nn.functional.conv1d(
            fsmn,
            state[p + ".attn.fsmn_block.weight"],
            None,
            groups=1280,
        ).transpose(1, 2)
        x = x + attn_out + fsmn + v

        m = torch.nn.functional.layer_norm(
            x.float(),
            (1280,),
            state[p + ".mlp_ln.weight"].float(),
            state[p + ".mlp_ln.bias"].float(),
            1e-5,
        ).to(x.dtype)
        ff = torch.nn.functional.linear(
            torch.nn.functional.gelu(
                torch.nn.functional.linear(m, state[p + ".mlp.0.weight"], state[p + ".mlp.0.bias"])
            ),
            state[p + ".mlp.2.weight"],
            state[p + ".mlp.2.bias"],
        )
        x = x + ff

    h = torch.nn.functional.linear(
        x,
        state["quantizer._codebook.project_down.weight"],
        state["quantizer._codebook.project_down.bias"],
    ).float()
    h = torch.tanh(h)
    h = h * 0.9990000128746033
    h = h.round() + 1
    powers = torch.pow(torch.tensor(3.0), torch.arange(8, dtype=h.dtype))
    return torch.sum(h * powers.view(1, 1, 8), dim=-1).int().cpu().numpy().reshape(-1)


def main() -> int:
    try:
        import safetensors  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError:
        venv_python = REPO_ROOT / ".venv" / "bin" / "python"
        if venv_python.is_file() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])
        raise

    if not CODEC.is_file():
        raise RuntimeError(f"Missing codec CLI: {CODEC}. Build the project first.")
    if not S3GEN.is_file():
        raise RuntimeError(f"Missing checkpoint: {S3GEN}")

    if not INPUT_WAV.is_file():
        raise RuntimeError(f"Missing input WAV: {INPUT_WAV}")

    pcm, wav_q = load_existing_test_slice()
    ref = compute_reference_codes(wav_q)

    with tempfile.TemporaryDirectory(prefix="codec-chatterbox-s3t-") as td:
        root = Path(td)
        wav_path = root / "input.wav"
        gguf_path = root / "model.gguf"
        out_path = root / "codes.npy"

        write_wav_pcm16(wav_path, pcm, 16000)
        run(
            [
                sys.executable,
                str(CONVERT),
                "--checkpoint-path",
                str(CHECKPOINT),
                "--model-type",
                "chatterbox_s3t",
                "--quantization",
                "F32",
                "--output",
                str(gguf_path),
            ]
        )
        t0 = time.perf_counter()
        run(
            [
                str(CODEC),
                "encode",
                "--model",
                str(gguf_path),
                "--in",
                str(wav_path),
                "--out",
                str(out_path),
            ]
        )
        encode_time_sec = time.perf_counter() - t0
        out = np.load(out_path).reshape(-1)

    if not np.array_equal(ref, out):
        diff = out.astype(np.int64) - ref.astype(np.int64)
        raise AssertionError(
            "Chatterbox S3T parity failed:\n"
            f"  ref[:16] = {ref[:16].tolist()}\n"
            f"  out[:16] = {out[:16].tolist()}\n"
            f"  mismatch = {int(np.count_nonzero(diff))}\n"
            f"  max_abs  = {int(np.max(np.abs(diff)))}"
        )

    if encode_time_sec > ENCODE_TIME_LIMIT_SEC:
        raise AssertionError(
            "Chatterbox S3T encode performance check failed:\n"
            f"  input_wav = {INPUT_WAV}\n"
            f"  slice_offset = {SLICE_OFFSET}\n"
            f"  slice_samples = {SLICE_SAMPLES}\n"
            f"  encode_time_sec = {encode_time_sec:.3f}\n"
            f"  limit_sec = {ENCODE_TIME_LIMIT_SEC:.3f}"
        )

    print(
        f"chatterbox S3T parity test passed ({ref.size} tokens, exact match, "
        f"encode_time={encode_time_sec:.3f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
