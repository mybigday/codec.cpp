#!/usr/bin/env python3
"""End-to-end smoke test for the CosyVoice codec (flow + HiFT) through the C API.

For every converted CosyVoice GGUF found under models/cosyvoice/
(cosyvoice-300m, breezyvoice, cosyvoice2, cosyvoice3, cosyvoice-300m-sft):

  * codec_speaker_from_audio on a reference clip yields prompt_token /
    prompt_feat / embedding slots of the right shapes (24 kHz variants keep
    prompt_feat = 2 * prompt_token frames),
  * codec_decode with that speaker returns finite, non-silent PCM of the
    expected length, identically on a second call,
  * builtin (spk2info) speakers decode without reference audio.

Missing GGUFs are skipped. Convert with
    .venv/bin/python scripts/convert-to-gguf.py --checkpoint-path <dir> -o models/cosyvoice/<name>.gguf
Run from the repo root: .venv/bin/python tests/e2e/cosyvoice_decode_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

from _common import REPO, read_wav_mono_f32

import codec_cpp

REF_WAV = REPO / "input_audio" / "10_2_24k.wav"
MODELS = ["cosyvoice-300m", "breezyvoice", "cosyvoice2", "cosyvoice3", "cosyvoice-300m-sft"]


def _slot_shape(spk, name):
    """Slot dims in ggml order (ne[0] first), or None when absent."""
    arr = spk.slot(name)
    return None if arr is None else list(reversed(arr.shape))


def _decode(ctx, tokens, spk):
    pcm, _sr = ctx.decode(np.asarray(tokens, dtype=np.int32).reshape(-1, 1), n_q=1, n_threads=8, speaker=spk)
    return pcm


def _check_pcm(pcm, n_expected, what):
    assert np.isfinite(pcm).all(), f"{what}: non-finite PCM"
    assert abs(len(pcm) - n_expected) <= 4, f"{what}: {len(pcm)} samples, expected ~{n_expected}"
    rms = float(np.sqrt((pcm ** 2).mean()))
    assert rms > 1e-3, f"{what}: silent output (rms={rms})"
    return rms


def run(name: str) -> None:
    path = REPO / "models" / "cosyvoice" / f"{name}.gguf"
    if not path.is_file():
        print(f"[SKIP] {name}: {path} not found")
        return
    with codec_cpp.Model(path, use_gpu=False, n_threads=8) as model, model.context(seed=0) as ctx:
        sr, hop, vocab = model.sample_rate, model.hop_size, model.codebook_size
        rng = np.random.default_rng(5)
        target = rng.integers(0, vocab, size=60).astype(np.int32)
        if model.can_speaker_from_audio:
            pcm, wsr = read_wav_mono_f32(REF_WAV)
            with ctx.speaker_from_audio(pcm, wsr) as spk:
                tok = _slot_shape(spk, "prompt_token")
                feat = _slot_shape(spk, "prompt_feat")
                emb = _slot_shape(spk, "embedding")
                assert tok and feat and emb and emb[0] == 192 and feat[0] == 80, f"{name}: bad slots {tok} {feat} {emb}"
                if sr == 24000:
                    assert feat[1] == 2 * tok[0], f"{name}: prompt_feat {feat[1]} != 2 * prompt_token {tok[0]}"
                # v1: int(n / 50 * 22050 / 256) mel frames, x256 samples; v2/v3: 2 frames x 480.
                n_expected = int(len(target) / 50 * sr / 256) * 256 if sr == 22050 else len(target) * hop
                p1 = _decode(ctx, target, spk)
                rms = _check_pcm(p1, n_expected, name)
                p2 = _decode(ctx, target, spk)
                assert np.array_equal(p1, p2), f"{name}: decode is not deterministic"
                print(f"[PASS] {name}: speaker {tok[0]} tokens / {feat[1]} frames; decode {len(p1)} samples rms={rms:.4f}")
        for vname in model.speaker_names:
            spk = model.speaker(vname)
            prompt = _slot_shape(spk, "prompt_feat")
            if sr == 22050 and prompt is not None:
                continue   # joint length depends on the prompt; covered by the reference-audio case
            n_expected = int(len(target) / 50 * sr / 256) * 256 if sr == 22050 else len(target) * hop
            rms = _check_pcm(_decode(ctx, target, spk), n_expected, f"{name}/{vname}")
            print(f"[PASS] {name}: builtin speaker {vname} rms={rms:.4f}")


def main() -> int:
    for name in MODELS:
        run(name)
    print("cosyvoice decode smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
