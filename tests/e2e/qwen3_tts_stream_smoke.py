#!/usr/bin/env python3
"""E2E smoke for exact incremental decode (Qwen3-TTS-Tokenizer-12Hz).

The 12 Hz decoder is strictly causal (left-padded convs, right-cropped
ConvTranspose, sliding-window causal attention), so:

1. prefix equivalence: decoding the first n frames gives the first n * hop
   samples of decoding them all (up to float summation order);
2. streaming (codec_audio_lm_stream_poll with an incremental decoder) gives
   the samples of one batch decode, with an ICL voice whose reference is fed
   to the decoder first;
3. a second run with the same voice starts from the cached reference state
   and gives the same audio as the first.

Thresholds are SNRs: with F32 weights the paths agree to ~120 dB (float
rounding); with F16 / quantized weights the order-dependent rounding is
amplified through the decoder (measured 30-50 dB), so the checks use the
batch path's own prefix noise as the yardstick.

Run from the repo root: .venv/bin/python tests/e2e/qwen3_tts_stream_smoke.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec_cpp  # noqa: E402
from _common import MODEL_SAMPLING, read_wav_mono_f32  # noqa: E402

M = REPO / "models"
REF = REPO / "assets" / "refs"
CODEC = M / "qwen3_tts/qwen3_tts_06b_base.gguf"
BACKBONE = M / "qwen3_tts/qwen3_tts_talker.gguf"
TEXT = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."


def snr(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    e = float((d * d).sum())
    return float("inf") if e == 0 else 10 * np.log10(float((a.astype(np.float64) ** 2).sum()) / e)


def main() -> int:
    if not CODEC.exists() or not BACKBONE.exists():
        print("[skip] qwen3_tts_stream: missing GGUFs")
        return 0
    failed = 0

    # 1. prefix equivalence of the codec decode
    model = codec_cpp.Model(str(CODEC))
    ctx = model.context()
    pcm, sr = read_wav_mono_f32(REF / "jfk_en.wav")
    codes = ctx.encode(pcm, sr)            # (n_frames, n_q)
    full = ctx.decode(codes)[0]
    hop = model.hop_size
    n = codes.shape[0] // 2
    half = ctx.decode(codes[:n])[0]
    p_snr = snr(full[: len(half)], half)
    ok = len(half) == n * hop and p_snr >= 35
    print(f"[{'PASS' if ok else 'FAIL'}] prefix: decode(first {n} frames) = decode(all)[:{len(half)}] SNR {p_snr:.1f} dB")
    failed += not ok

    # 2-3. streaming vs batch with an ICL voice, and the cached reference state
    opts = dict(MODEL_SAMPLING.get("qwen3_tts", {}), seed=2, ref_audio=REF / "jfk_en.wav",
                ref_text=(REF / "jfk_en.txt").read_text().strip())
    with codec_cpp.TTS(BACKBONE, CODEC) as tts:
        batch, sr = tts.synthesize(TEXT, **opts)
        again, _ = tts.synthesize(TEXT, **opts)
        pieces = list(tts.stream(TEXT, chunk_ms=500, **opts))
    stream = np.concatenate([p for p, _ in pieces]) if pieces else np.zeros(0, np.float32)
    s_snr = snr(batch, stream) if len(stream) == len(batch) else float("-inf")
    ok = len(stream) == len(batch) and len(pieces) > 2 and s_snr >= p_snr - 15
    print(f"[{'PASS' if ok else 'FAIL'}] stream: {len(pieces)} pieces, {len(stream)} samples (batch {len(batch)}), "
          f"SNR vs batch {s_snr:.1f} dB")
    failed += not ok
    ok = len(again) == len(batch) and np.array_equal(again, batch)
    print(f"[{'PASS' if ok else 'FAIL'}] cached reference: second run {'identical' if ok else 'differs'}")
    failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
