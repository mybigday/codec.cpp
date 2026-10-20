#!/usr/bin/env python3
"""CSM-1B text-to-speech from Python: llama-cpp-python runs the backbone,
codec_cpp.AudioLM the codec side (depth decoder, compose, Mimi decode).

Mirrors tts-cli's run_codebook_ar for a residual depth-AR model: tokenize
prefix + text + suffix from prompt_info, prefill, then per frame
step_frame(hidden) -> observe_codes -> next_embed -> backbone, until the
cb0 end-of-audio code.  Greedy; asserts the stop and the Whisper CER.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import llama_cpp  # noqa: F401  (load before the first codec_cpp call)

from _common import REPO, LlamaBackbone, asr_cer, must, write_wav_pcm16

import codec_cpp

CODEC = REPO / "models" / "csm" / "csm.gguf"
BACKBONE = REPO / "models" / "csm" / "llama_backbone.gguf"
TEXT = "Hello, this is a test of the emergency broadcast system."


def main() -> int:
    if not CODEC.is_file() or not BACKBONE.is_file():
        print(f"[skip] missing {CODEC.name} / {BACKBONE.name}")
        return 0
    alm = codec_cpp.AudioLM(CODEC)
    pi = alm.prompt_info()
    must(pi["model_kind_name"] == "residual_depth_ar", f"unexpected kind {pi['model_kind_name']}")
    alm.set_uses_embed_override(True, 1)

    bb = LlamaBackbone(BACKBONE, n_ctx=1024)
    toks = bb.tokenize(pi["prompt_prefix"] + TEXT + pi["prompt_suffix"],
                       add_bos=pi["add_bos"], special=pi["parse_special"])
    hidden = bb.feed_tokens(toks)
    stop = "max_frames"
    for _ in range(400):
        codes = alm.step_frame(hidden)
        if alm.observe_codes(codes, hidden) == codec_cpp.audio_lm.STOP:
            must(not alm.last_error, alm.last_error)
            stop = "eos"
            break
        hidden = bb.feed_embeds(alm.next_embed()[None, :])
    bb.close()
    pcm, sr = alm.decode()
    print(f"frames={alm.n_frames} stop={stop} audio={len(pcm) / sr:.2f}s")
    must(stop == "eos", "no end-of-audio code")

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "csm.wav"
        write_wav_pcm16(wav, pcm, sr)
        res = asr_cer(wav, TEXT, "en")
    if res is not None:
        print(f"ASR='{res[0]}' CER={res[1]:.3f}")
        must(res[1] <= 0.10, f"CER {res[1]:.3f}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
