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

# Variant under test.  Set MOSS_VARIANT=full to test the 1.6B mono 24 kHz
# checkpoint instead.
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "nano"
if VARIANT == "full":
    MOSS_DIR = REPO_ROOT / ".model-src" / "moss_audio_full"
    GGUF = REPO_ROOT / "models" / "moss_audio_full" / "moss_audio_full.gguf"
    NUM_QUANTIZERS = 32
else:
    MOSS_DIR = REPO_ROOT / ".model-src" / "moss_audio_nano"
    GGUF = REPO_ROOT / "models" / "moss_audio_nano" / "moss_audio_nano.gguf"
    NUM_QUANTIZERS = 16


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


def synth_test_input(target_sr: int, n_channels: int) -> tuple[np.ndarray, int]:
    """Use the repo's test.wav (mono 24 kHz speech) resampled to the target.

    A real speech clip exercises a broadband portion of the codebook; pure
    sine inputs flip many codes under tiny FP noise and inflate amplitude
    error even when the codec is healthy."""
    src_path = REPO_ROOT / "test.wav"
    src, src_sr = read_wav_stereo_f32(src_path)
    mono = src.mean(axis=1) if src.ndim == 2 else src
    n_target = int(round(len(mono) * target_sr / src_sr))
    # Linear resample is enough for a smoke test.
    x_old = np.arange(len(mono), dtype=np.float64)
    x_new = np.linspace(0.0, len(mono) - 1, n_target, dtype=np.float64)
    resampled = np.interp(x_new, x_old, mono).astype(np.float32)
    # Use ~1.5s so the encoder sees a non-trivial sequence (enough to also
    # exercise frames past the bottleneck-alignment boundary).
    n = min(int(round(target_sr * 1.5)), len(resampled))
    resampled = resampled[:n]
    if n_channels == 1:
        return resampled.reshape(-1, 1), target_sr
    # Stereo: distinguish channels with a small delay so a channel swap bug
    # would surface in the per-channel comparison.
    delay = 8
    right = np.concatenate([np.zeros(delay, dtype=np.float32), resampled[:-delay]])
    return np.stack([resampled, right], axis=1), target_sr


def hf_reconstruction(audio: np.ndarray, sr: int) -> np.ndarray:
    """Reconstruct via upstream HF MOSS-Audio-Tokenizer model (nano or full)."""
    # Clear any previously-loaded MOSS modules so the variant-specific source
    # files actually get reloaded — these are remote-code modules and Python
    # caches them by module name, not by file path.
    for mn in list(sys.modules):
        if mn.startswith("configuration_moss_audio_tokenizer") or mn.startswith("modeling_moss_audio_tokenizer"):
            del sys.modules[mn]
    if str(MOSS_DIR) not in sys.path:
        sys.path.insert(0, str(MOSS_DIR))

    import importlib.util
    cfg_spec = importlib.util.spec_from_file_location(
        "configuration_moss_audio_tokenizer", MOSS_DIR / "configuration_moss_audio_tokenizer.py"
    )
    cfg_mod = importlib.util.module_from_spec(cfg_spec)
    sys.modules["configuration_moss_audio_tokenizer"] = cfg_mod
    cfg_spec.loader.exec_module(cfg_mod)

    mod_spec = importlib.util.spec_from_file_location(
        "modeling_moss_audio_tokenizer", MOSS_DIR / "modeling_moss_audio_tokenizer.py"
    )
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules["modeling_moss_audio_tokenizer"] = mod
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

    # Read variant config from the HF config.json so we use the right
    # sample rate and channel count without hardcoding it per variant.
    import json
    cfg_json = json.loads((MOSS_DIR / "config.json").read_text())
    sr = int(cfg_json.get("sampling_rate", 48000))
    n_channels = int(cfg_json.get("number_channels") or 1)

    with tempfile.TemporaryDirectory(prefix="codec-moss-") as td:
        td = Path(td)
        in_wav = td / "input.wav"
        cpp_wav = td / "cpp_e2e.wav"

        audio, _ = synth_test_input(sr, n_channels)
        write_wav_stereo_pcm16(in_wav, audio, sr)

        run([str(CODEC_CLI), "e2e", "--model", str(GGUF), "--in", str(in_wav), "--out", str(cpp_wav), "--nq", str(NUM_QUANTIZERS)])

        cpp_out, cpp_sr = read_wav_stereo_f32(cpp_wav)
        hf_out = hf_reconstruction(audio, sr)

    if cpp_sr != sr:
        print(f"FAIL: cpp output sample_rate {cpp_sr} != input {sr}")
        return 1
    if cpp_out.shape[1] != n_channels:
        print(f"FAIL: cpp output should have {n_channels} ch, got {cpp_out.shape[1]}")
        return 1
    n = min(len(hf_out), len(cpp_out))
    cpp_out = cpp_out[:n]
    hf_out = hf_out[:n]

    print(f"  variant={VARIANT} sr={sr} channels={n_channels} n_q={NUM_QUANTIZERS}")
    for c in range(n_channels):
        corr = float(np.corrcoef(hf_out[:, c], cpp_out[:, c])[0, 1])
        max_diff = float(np.abs(hf_out[:, c] - cpp_out[:, c]).max())
        print(f"  ch{c}: corr_vs_hf={corr:.6f} max_diff={max_diff:.4f}")
        if corr < 0.95:
            print(f"FAIL: ch{c} cpp vs HF correlation {corr:.4f} < 0.95")
            return 1

    if not np.all(np.isfinite(cpp_out)):
        print("FAIL: cpp output contains non-finite samples")
        return 1
    print(f"MOSS-Audio-Tokenizer ({VARIANT}) e2e smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
