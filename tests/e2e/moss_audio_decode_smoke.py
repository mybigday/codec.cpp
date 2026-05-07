#!/usr/bin/env python3
"""End-to-end smoke test for MOSS-Audio-Tokenizer-Nano.

Drives the public C API via codec-cli (encode + decode) and compares the
reconstruction against a reference produced by the upstream HF model. The
encoder is allowed to flip a small fraction of codes due to argmax sensitivity
at codebook ties under FP noise; correctness is judged on reconstruction
correlation, not exact code match."""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODEC_CLI = REPO_ROOT / "build" / "codec-cli"
CONVERT = REPO_ROOT / "scripts" / "convert-to-gguf.py"
MOSS_DIR = REPO_ROOT / ".model-src" / "moss_audio_nano"
GGUF = REPO_ROOT / "models" / "moss_audio_nano" / "moss_audio_nano.gguf"


def run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.decode(errors='replace')[:4096]}")
    return proc.stdout


def read_wav_stereo_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw != 2:
        raise RuntimeError(f"{path}: expected PCM16, got sample_width={sw}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1, nch), sr


def write_wav_stereo_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def synth_test_input() -> tuple[np.ndarray, int]:
    """Use the repo's test.wav (mono 24 kHz speech) resampled to 48 kHz stereo.

    A real speech clip exercises a broadband portion of the codebook; pure
    sine inputs flip many codes under tiny FP noise and inflate amplitude
    error even when the codec is healthy."""
    src_path = REPO_ROOT / "test.wav"
    src, src_sr = read_wav_stereo_f32(src_path)
    mono = src.mean(axis=1) if src.ndim == 2 else src
    target_sr = 48000
    n_target = int(round(len(mono) * target_sr / src_sr))
    # Linear resample is enough for a smoke test.
    x_old = np.arange(len(mono), dtype=np.float64)
    x_new = np.linspace(0.0, len(mono) - 1, n_target, dtype=np.float64)
    resampled = np.interp(x_new, x_old, mono).astype(np.float32)
    # Take the first ~1 second so the encoder sees one bottleneck-aligned chunk.
    n = min(target_sr, len(resampled))
    resampled = resampled[:n]
    # Build a stereo signal: left = signal, right = signal delayed by 8 samples.
    # The delay distinguishes the channels so the per-channel comparison
    # would actually catch a channel-interleave swap bug.
    delay = 8
    right = np.concatenate([np.zeros(delay, dtype=np.float32), resampled[:-delay]])
    return np.stack([resampled, right], axis=1), target_sr


def hf_reconstruction(audio: np.ndarray, sr: int) -> np.ndarray:
    """Reconstruct via upstream HF MOSS-Audio-Tokenizer-Nano model."""
    sys.path.insert(0, str(MOSS_DIR))
    import importlib.util

    cfg_spec = importlib.util.spec_from_file_location(
        "configuration_moss_audio_tokenizer", MOSS_DIR / "configuration_moss_audio_tokenizer.py"
    )
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(cfg_mod)
    sys.modules["configuration_moss_audio_tokenizer"] = cfg_mod

    mod_spec = importlib.util.spec_from_file_location(
        "modeling_moss_audio_tokenizer", MOSS_DIR / "modeling_moss_audio_tokenizer.py"
    )
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)

    import torch
    cfg = cfg_mod.MossAudioTokenizerConfig.from_pretrained(MOSS_DIR)
    model = mod.MossAudioTokenizerModel.from_pretrained(MOSS_DIR, config=cfg).eval()

    x = torch.from_numpy(audio.T).unsqueeze(0)  # (1, 2, T)
    with torch.no_grad():
        # Mirror the cpp encode→decode path: use the public encode + decode
        # methods.  (model.forward() takes a `_decode_frame` shortcut that
        # passes the encoder's audio_codes_lengths into the decoder, applying
        # extra length-aware masking that the public decode API does not.)
        enc = model.encode(x, return_dict=True)
        dec = model.decode(enc.audio_codes, return_dict=True)
    return dec.audio[0].cpu().numpy().T  # (T, 2)


def main() -> int:
    if not CODEC_CLI.is_file():
        print(f"SKIP: {CODEC_CLI} not built")
        return 0
    if not GGUF.is_file():
        if not MOSS_DIR.is_dir():
            print(f"SKIP: {MOSS_DIR} missing — run scripts/converters/moss_audio.py first")
            return 0
        GGUF.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(CONVERT), "--checkpoint-path", str(MOSS_DIR), "--model-type", "moss_audio", "--output", str(GGUF)])

    with tempfile.TemporaryDirectory(prefix="codec-moss-") as td:
        td = Path(td)
        in_wav = td / "input.wav"
        cpp_wav = td / "cpp_e2e.wav"

        audio, sr = synth_test_input()
        write_wav_stereo_pcm16(in_wav, audio, sr)

        run([str(CODEC_CLI), "e2e", "--model", str(GGUF), "--in", str(in_wav), "--out", str(cpp_wav), "--nq", "16"])

        cpp_out, cpp_sr = read_wav_stereo_f32(cpp_wav)
        hf_out = hf_reconstruction(audio, sr)

    if cpp_sr != sr:
        print(f"FAIL: cpp output sample_rate {cpp_sr} != input {sr}")
        return 1
    if cpp_out.shape[1] != 2:
        print(f"FAIL: cpp output should be stereo, got {cpp_out.shape[1]} ch")
        return 1
    n = min(len(hf_out), len(cpp_out))
    cpp_out = cpp_out[:n]
    hf_out = hf_out[:n]

    # Compare cpp vs HF reference reconstruction (per channel).
    for c in range(2):
        corr = float(np.corrcoef(hf_out[:, c], cpp_out[:, c])[0, 1])
        max_diff = float(np.abs(hf_out[:, c] - cpp_out[:, c]).max())
        print(f"  ch{c}: corr_vs_hf={corr:.6f} max_diff={max_diff:.4f}")
        if corr < 0.95:
            print(f"FAIL: ch{c} cpp vs HF correlation {corr:.4f} < 0.95")
            return 1

    if not np.all(np.isfinite(cpp_out)):
        print("FAIL: cpp output contains non-finite samples")
        return 1
    print("MOSS-Audio-Nano e2e smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
