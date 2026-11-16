#!/usr/bin/env python3
"""E2E smoke for streaming (codec_audio_lm_stream_poll).

* Causal decoders (Qwen3-TTS-Tokenizer, Mimi): codec memory. Each poll decodes
  only the new frames through the carried decoder state, so the stream equals
  one batch decode to float rounding (relative RMS <= 1e-3, as
  tests/e2e/codec_memory_smoke.py). An ICL voice's reference is decoded into
  the memory once; a second stream with the same voice (restored state) must
  give the same samples.
* Others (DAC): each poll decodes a window statelessly, [committed - look_back,
  avail) with the decoder's look_back / look_ahead (GGUF codec.decoder.look_*),
  and cross-fades the joins. The stream must match the batch decode
  perceptually: same length, log-mel distance well under the codec's own
  reconstruction error (3-5 dB against the input audio), no spectral jump at
  the joins beyond the batch decode's own.

Run from the repo root: .venv/bin/python tests/e2e/audio_lm_stream_smoke.py
"""

import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, stft

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec_cpp  # noqa: E402
from _common import MODEL_SAMPLING  # noqa: E402

M = REPO / "models"
REF = REPO / "assets" / "refs"
TEXT = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."
CASES = [   # name, path, codec gguf, backbone, options
    ("qwen3_tts_icl", "qwen3 12 Hz: memory", M / "qwen3_tts/qwen3_tts_06b_base.gguf", M / "qwen3_tts/qwen3_tts_talker.gguf",
     dict(MODEL_SAMPLING.get("qwen3_tts", {}), seed=2, ref_audio=REF / "jfk_en.wav",
          ref_text=(REF / "jfk_en.txt").read_text().strip() if (REF / "jfk_en.txt").exists() else None)),
    ("csm", "mimi: memory", M / "csm/csm.gguf", M / "csm/llama_backbone.gguf", dict(MODEL_SAMPLING.get("csm", {}), seed=7, temp=0)),
    ("outetts_1_0", "dac 24 kHz: left 0.32 s, right 0.32 s", M / "output/outetts/outetts-1.0-0.6b-codec-F32.gguf",
     M / "hf/outetts_1_0/OuteTTS-1.0-0.6B-Q8_0.gguf", dict(MODEL_SAMPLING.get("outetts_1_0", {}), seed=1)),
]
LSD_MAX, SEAM_MAX = 1.5, 2.0   # windowed path
REL_RMS_MAX = 1e-3             # memory path


def melfb(n_fft=400, n_mels=80, sr=16000):
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    pts = hz(np.linspace(0, mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, a:b] = (np.arange(a, b) - a) / max(1, b - a)
        fb[m - 1, b:c] = (c - np.arange(b, c)) / max(1, c - b)
    return fb


FB = melfb()


def logmel(x, sr):
    g = gcd(16000, sr)
    y = resample_poly(x.astype(np.float64), 16000 // g, sr // g)
    _, _, z = stft(y, 16000, nperseg=400, noverlap=240, boundary=None)   # 10 ms hop
    return 10 * np.log10(FB @ (np.abs(z) ** 2) + 1e-8)


def main() -> int:
    failed = 0
    for name, win, codec, backbone, opts in CASES:
        if not codec.exists() or not backbone.exists():
            print(f"[skip] {name}: missing GGUFs")
            continue
        opts = {k: v for k, v in opts.items() if v is not None}
        memory = "memory" in win
        with codec_cpp.TTS(backbone, codec) as tts:
            t0 = time.time(); batch, sr = tts.synthesize(TEXT, **opts); tb = time.time() - t0
            t0 = time.time(); pieces = list(tts.stream(TEXT, chunk_ms=500, **opts)); ts = time.time() - t0
            again = [p for p, _ in tts.stream(TEXT, chunk_ms=500, **opts)] if memory else None
        stream = np.concatenate([p for p, _ in pieces]) if pieces else np.zeros(0, np.float32)
        joins = np.cumsum([len(p) for p, _ in pieces])[:-1] / sr
        ok_len = abs(len(stream) - len(batch)) == 0
        A, B = logmel(batch, sr), logmel(stream, sr)
        k = min(A.shape[1], B.shape[1])
        lsd = float(np.mean(np.sqrt(np.mean((A[:, :k] - B[:, :k]) ** 2, axis=0))))
        flux = lambda X: np.sqrt(np.mean(np.diff(X, axis=1) ** 2, axis=0))  # noqa: E731
        fa, fb = flux(A[:, :k]), flux(B[:, :k])
        js = [int(j * 100) for j in joins if 0 < int(j * 100) < len(fa) - 1]
        seam = max([max(fb[j - 1:j + 2]) - max(fa[j - 1:j + 2]) for j in js], default=0.0)
        ok = ok_len and len(pieces) > 2
        if memory:
            n = min(len(stream), len(batch))
            rel = float(np.sqrt(np.mean((stream[:n] - batch[:n]) ** 2) / max(np.mean(batch[:n] ** 2), 1e-20)))
            same = again is not None and len(again) == len(pieces) and np.array_equal(np.concatenate(again), stream)
            ok = ok and rel <= REL_RMS_MAX and same
            detail = f"rel rms vs batch {rel:.1e}, repeat stream {'identical' if same else 'DIFFERS'}"
        else:
            ok = ok and lsd <= LSD_MAX and seam <= SEAM_MAX
            detail = f"worst seam {seam:+.2f} dB"
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({win}): {len(pieces)} pieces, {len(stream)} samples "
              f"(batch {len(batch)}), log-mel distance {lsd:.2f} dB, {detail}, "
              f"time stream {ts:.1f} s / batch {tb:.1f} s")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
