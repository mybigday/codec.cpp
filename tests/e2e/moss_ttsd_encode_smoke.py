"""End-to-end parity check: codec.cpp's XY-Tokenizer encode (bundled
inside moss_ttsd_v0_5.gguf) vs HF MOSS-TTSD's processor speech
tokenizer for a ref-audio WAV.

MOSS-TTSD's voice-clone flow uses ref audio as **speech tokens** spliced
into the prompt sequence — there isn't a separate speaker encoder.
codec.cpp's job for MOSS-TTSD ref audio is to encode the WAV into those
tokens; the application then interleaves them with text tokens following
the model's channel convention.

This test:
  1. Picks a short ref WAV.
  2. Runs codec.cpp's XY-Tokenizer encoder on the bundled
     `moss_ttsd_v0_5.gguf`.
  3. Runs HF's `AutoProcessor.from_pretrained("fnlp/MOSS-TTSD-v0.5",
     codec_path=...)` speech tokenizer on the same WAV.
  4. Confirms ≥95% code match (the same threshold the standalone
     xy_tokenizer_decode_smoke.py uses, since RVQ argmin is FP-sensitive
     at codebook ties).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CODEC_CLI = REPO / "build" / "codec-cli"
GGUF = REPO / "models/moss_ttsd_v0_5/moss_ttsd_v0_5.gguf"
HF_ID = "fnlp/MOSS-TTSD-v0.5"


def read_wav_mono_f32(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise RuntimeError(f"{path}: expected PCM16, got sample_width={sw}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1, nch).mean(axis=1), sr


def write_wav_mono_pcm16(path: Path, pcm: np.ndarray, sr: int) -> None:
    s = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr))
        w.writeframes(s.tobytes())


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n"
                            f"{p.stderr.decode(errors='replace')[:4096]}")
    return p.stdout


def hf_encode(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Run HF MOSS-TTSD processor's speech_tokenizer, return (n_q, T) codes."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(
        HF_ID,
        codec_path="fnlp/XY_Tokenizer_TTSD_V0_hf",
        trust_remote_code=True,
    )
    # `proc.feature_extractor` (XYTokenizerFeatureExtractor) does the
    # mel/STFT front-end; `proc.audio_tokenizer` (XYTokenizerModel)
    # runs encode → RVQ codes.  Same pipeline `MossTTSDSession` uses
    # under the hood when it asks HF to splice ref audio into the prompt.
    import torch
    with torch.inference_mode():
        feats = proc.feature_extractor(
            torch.from_numpy(pcm).unsqueeze(0), sampling_rate=sr,
            return_attention_mask=True, return_tensors="pt",
        )
        enc = proc.audio_tokenizer.encode(feats)
    return enc.audio_codes.cpu().numpy()[:, 0, :]   # (n_q, T)


def main() -> int:
    if not CODEC_CLI.is_file():
        print(f"SKIP: {CODEC_CLI} not built"); return 0
    if not GGUF.is_file():
        print(f"SKIP: {GGUF} missing — run scripts/convert-to-gguf.py first"); return 0

    src, sr = read_wav_mono_f32(REPO / "test.wav")
    # Downmix + resample to 16 kHz (MOSS-TTSD's XY-Tokenizer rate).
    target_sr = 16000
    if sr != target_sr:
        x_old = np.arange(len(src), dtype=np.float64)
        x_new = np.linspace(0.0, len(src) - 1,
                            int(round(len(src) * target_sr / sr)),
                            dtype=np.float64)
        src = np.interp(x_new, x_old, src).astype(np.float32)
    # Trim to 1.5 s for a focused test (matches the standalone
    # xy_tokenizer_decode_smoke.py duration so the RVQ tie-rate is
    # comparable).
    n = min(int(target_sr * 1.5), len(src))
    pcm = src[:n]

    with tempfile.TemporaryDirectory(prefix="moss-ttsd-enc-") as td:
        td = Path(td)
        in_wav = td / "input.wav"
        write_wav_mono_pcm16(in_wav, pcm, target_sr)

        # ---- HF reference codes -------------------------------------------
        print("[ref] running HF MOSS-TTSD processor speech_tokenizer …")
        hf_codes = hf_encode(pcm, target_sr)
        print(f"      hf codes: {hf_codes.shape}")

        # ---- codec.cpp encode via codec-cli -------------------------------
        print("[cpp] running codec.cpp xy_tokenizer encode (moss_ttsd_v0_5.gguf) …")
        cpp_npy = td / "cpp_codes.npy"
        run([str(CODEC_CLI), "encode", "--model", str(GGUF),
             "--in", str(in_wav), "--out", str(cpp_npy),
             "--nq", str(hf_codes.shape[0])])
        cpp_codes = np.load(cpp_npy)
        print(f"      cpp codes: {cpp_codes.shape}")

        # ---- Compare ------------------------------------------------------
        n = min(hf_codes.shape[1], cpp_codes.shape[1])
        match = int((hf_codes[:, :n] == cpp_codes[:, :n]).sum())
        total = int(hf_codes.shape[0]) * int(n)
        frac = match / max(1, total)
        print(f"  match: {match}/{total} = {frac:.2%}")
        if frac < 0.95:
            print(f"FAIL: code match {frac:.2%} < 95%", file=sys.stderr); return 1

        print("\nMOSS-TTSD ref-audio encode parity test PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
