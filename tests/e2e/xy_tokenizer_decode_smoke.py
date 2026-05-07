#!/usr/bin/env python3
"""End-to-end smoke test for XY-Tokenizer (16 kHz in / 24 kHz out).

The cpp encoder threads `n_valid` through the attention layers (matches
HF's `attention_mask`-based SDPA bias) and uses GELU-erf to match HF's
F.gelu, giving bit-exact-modulo-FP-noise parity in the encoder body.
RVQ argmin is FP-sensitive at codebook ties; we accept ≥ 95 % codes
matching HF on the smoke clip and ≥ 0.95 corr on the cpp e2e
reconstruction vs HF.
"""

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
XY_DIR = REPO_ROOT / ".model-src" / "xy_tokenizer"
GGUF = REPO_ROOT / "models" / "xy_tokenizer" / "xy_tokenizer.gguf"


def run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.decode(errors='replace')[:4096]}")
    return proc.stdout


def read_wav_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw != 2:
        raise RuntimeError(f"{path}: expected PCM16, got sample_width={sw}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1, nch).mean(axis=1), sr


def write_wav_mono_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def synth_test_input() -> tuple[np.ndarray, int]:
    """Linear-resample test.wav (24 kHz mono speech) to 16 kHz, take 1.5 s."""
    src, src_sr = read_wav_mono_f32(REPO_ROOT / "test.wav")
    target_sr = 16000
    n_target = int(round(len(src) * target_sr / src_sr))
    x_old = np.arange(len(src), dtype=np.float64)
    x_new = np.linspace(0.0, len(src) - 1, n_target, dtype=np.float64)
    resampled = np.interp(x_new, x_old, src).astype(np.float32)
    n = min(int(round(target_sr * 1.5)), len(resampled))
    return resampled[:n], target_sr


def hf_encode_decode(pcm: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Run HF reference: returns (codes (n_q, T), audio (samples,) at 24 kHz)."""
    # Re-import variant-specific modules (no relative-import support outside a package).
    for mn in list(sys.modules):
        if mn.startswith("configuration_xy_tokenizer") or mn.startswith("modeling_xy_tokenizer") or \
           mn.startswith("feature_extraction_xy_tokenizer"):
            del sys.modules[mn]
    if str(XY_DIR) not in sys.path:
        sys.path.insert(0, str(XY_DIR))

    import importlib.util
    for nm, fn in [
        ("feature_extraction_xy_tokenizer", "feature_extraction_xy_tokenizer.py"),
        ("configuration_xy_tokenizer", "configuration_xy_tokenizer.py"),
        ("modeling_xy_tokenizer", "modeling_xy_tokenizer.py"),
    ]:
        spec = importlib.util.spec_from_file_location(nm, XY_DIR / fn)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[nm] = mod
        spec.loader.exec_module(mod)

    import torch
    cfg_mod = sys.modules["configuration_xy_tokenizer"]
    fe_mod = sys.modules["feature_extraction_xy_tokenizer"]
    m_mod = sys.modules["modeling_xy_tokenizer"]
    cfg = cfg_mod.XYTokenizerConfig.from_pretrained(XY_DIR)
    fe = fe_mod.XYTokenizerFeatureExtractor(**cfg.params["feature_extractor_kwargs"])
    model = m_mod.XYTokenizerModel.from_pretrained(XY_DIR, config=cfg).eval()

    with torch.no_grad():
        feats = fe(torch.from_numpy(pcm).unsqueeze(0), sampling_rate=sr,
                   return_attention_mask=True, return_tensors="pt")
        enc = model.encode(feats)
        dec = model.decode(enc.audio_codes, overlap_seconds=10)
    codes = enc.audio_codes.cpu().numpy()[:, 0, :]   # (n_q=8, T)
    audio = dec["audio_values"][0].cpu().numpy().flatten()
    return codes, audio


def main() -> int:
    if not CODEC_CLI.is_file():
        print(f"SKIP: {CODEC_CLI} not built")
        return 0
    if not GGUF.is_file():
        if not XY_DIR.is_dir():
            print(f"SKIP: {XY_DIR} missing — run scripts/converters/xy_tokenizer.py first")
            return 0
        GGUF.parent.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(CONVERT),
             "--checkpoint-path", str(XY_DIR),
             "--model-type", "xy_tokenizer",
             "--output", str(GGUF)])

    with tempfile.TemporaryDirectory(prefix="codec-xy-") as td:
        td = Path(td)
        in_wav = td / "input.wav"

        pcm, sr = synth_test_input()
        write_wav_mono_pcm16(in_wav, pcm, sr)

        # ---- 1. HF reference codes & audio --------------------------------
        hf_codes, hf_audio = hf_encode_decode(pcm, sr)
        print(f"  hf codes: {hf_codes.shape}, hf audio: {hf_audio.shape} rms={hf_audio.std():.4f}")

        # ---- 2. cpp decode of HF codes (decoder parity) -------------------
        codes_npy = td / "hf_codes.npy"
        np.save(codes_npy, hf_codes.astype(np.int32))
        cpp_dec_wav = td / "cpp_dec.wav"
        run([str(CODEC_CLI), "decode", "--model", str(GGUF), "--codes", str(codes_npy),
             "--out", str(cpp_dec_wav), "--nq", str(hf_codes.shape[0])])
        cpp_dec, cpp_dec_sr = read_wav_mono_f32(cpp_dec_wav)
        n = min(len(cpp_dec), len(hf_audio))
        dec_corr = float(np.corrcoef(hf_audio[:n], cpp_dec[:n])[0, 1])
        dec_max = float(np.abs(hf_audio[:n] - cpp_dec[:n]).max())
        print(f"  cpp_decode(HF codes) vs HF: corr={dec_corr:.6f} max_diff={dec_max:.4f}")
        if dec_corr < 0.95:
            print(f"FAIL: decoder corr {dec_corr:.4f} < 0.95")
            return 1

        # ---- 2b. cpp encoder: codes match HF -----------------------------
        cpp_codes_npy = td / "cpp_codes.npy"
        run([str(CODEC_CLI), "encode", "--model", str(GGUF), "--in", str(in_wav),
             "--out", str(cpp_codes_npy), "--nq", str(hf_codes.shape[0])])
        cpp_codes = np.load(cpp_codes_npy)  # (n_q, T)
        n = min(hf_codes.shape[1], cpp_codes.shape[1])
        match = (hf_codes[:, :n] == cpp_codes[:, :n]).sum()
        total = int(hf_codes.shape[0]) * int(n)
        match_frac = match / max(1, total)
        print(f"  cpp encode vs HF codes: {match}/{total} = {match_frac:.2%}")
        if match_frac < 0.95:
            print(f"FAIL: code match {match_frac:.2%} < 95%")
            return 1

        # ---- 3. cpp e2e produces sane output ------------------------------
        cpp_e2e_wav = td / "cpp_e2e.wav"
        run([str(CODEC_CLI), "e2e", "--model", str(GGUF), "--in", str(in_wav),
             "--out", str(cpp_e2e_wav), "--nq", str(hf_codes.shape[0])])
        cpp_e2e, cpp_e2e_sr = read_wav_mono_f32(cpp_e2e_wav)
        if cpp_e2e_sr != 24000:
            print(f"FAIL: expected 24 kHz output, got {cpp_e2e_sr}")
            return 1
        if not np.all(np.isfinite(cpp_e2e)):
            print("FAIL: cpp e2e contains non-finite samples")
            return 1
        if cpp_e2e.std() < 1e-3:
            print(f"FAIL: cpp e2e is silent (rms={cpp_e2e.std():.4f})")
            return 1
        # cpp's audio length should be roughly proportional to input duration.
        expected_n = (len(pcm) // 1280) * 1920
        if abs(len(cpp_e2e) - expected_n) > 4 * 1920:
            print(f"FAIL: cpp e2e length {len(cpp_e2e)} != expected ~{expected_n}")
            return 1
        # Reconstruction parity vs HF e2e.
        n = min(len(cpp_e2e), len(hf_audio))
        e2e_corr = float(np.corrcoef(hf_audio[:n], cpp_e2e[:n])[0, 1])
        print(f"  cpp e2e: len={len(cpp_e2e)} rms={cpp_e2e.std():.4f} corr_vs_hf={e2e_corr:.4f}")
        if e2e_corr < 0.95:
            print(f"FAIL: e2e corr {e2e_corr:.4f} < 0.95")
            return 1

    print("XY-Tokenizer smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
