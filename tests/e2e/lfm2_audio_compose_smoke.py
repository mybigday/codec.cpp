"""LFM2-Audio compose-audio-embd parity vs HF reference.

After sampling an audio frame in `LFM2AudioModel.generate_sequential`
the model composes the next-step backbone input as:

    in_emb = audio_embedding(next_token + codebook_offsets).sum(0)

i.e. one fused-table lookup per codebook (offset `i * (audio_vocab + 1)`),
summed across the N codebooks.  Our codec_lm now ships
`audio_embedding.embedding` as `lm.compose.audio_embd.weight` so
`codec_lm_compose_audio_embd` can produce this exact vector without
the caller having to dig into the original safetensors.

This smoke confirms the fused-compose runtime matches HF bit-exactly.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from _common import REPO, corr, must
from codec_cpp import Model

GGUF = REPO / "models/lfm2_audio/lfm2_audio.gguf"
HF   = "LiquidAI/LFM2-Audio-1.5B"


def main() -> int:
    must(GGUF.is_file(), f"missing {GGUF}; run convert-to-gguf first")

    import torch
    from liquid_audio import LFM2AudioModel  # type: ignore

    print("[hf] loading reference …", flush=True)
    m = LFM2AudioModel.from_pretrained(HF, dtype=torch.float32, device="cpu").eval()
    N = int(m.codebooks)

    print("[cpp] loading codec_lm …", flush=True)
    lm = Model(GGUF, use_gpu=False, n_threads=max(1, (os.cpu_count() or 1) // 2)).get_lm_adaptor()
    must(lm.info.n_codebook == N,
         f"n_cb mismatch lm={lm.info.n_codebook} hf={N}")
    must(lm.info.compose_audio_embed_dim > 0,
         "compose_audio_embed_dim not set — converter didn't write "
         "lm.compose.audio_embd / codec.lm.compose.audio_embed_dim")
    out_dim = lm.info.compose_audio_embed_dim
    print(f"  compose_audio_embed_dim = {out_dim}", flush=True)

    rng = np.random.default_rng(0)
    n_cases = 3
    all_pass = True
    for case in range(n_cases):
        # Per-cb code; cb 0..N-1 each in [0, audio_vocab+1) (+1 for EOS).
        codes = rng.integers(0, m.audio_vocab_size, size=N).astype(np.int64)
        with torch.no_grad():
            ref = m.audio_embedding(
                torch.from_numpy(codes) + m.codebook_offsets
            ).sum(0).cpu().float().numpy()

        cpp = lm.compose_audio_embd(codes)
        must(cpp.shape == (out_dim,), f"compose returned {cpp.shape} on case {case}")

        mad = float(np.max(np.abs(cpp - ref)))
        cc  = corr(cpp, ref)
        ok  = mad <= 1e-5 and cc >= 0.99999
        tag = "OK " if ok else "FAIL"
        print(f"  [{tag}] case={case}  codes={codes.tolist()}  "
              f"max_abs_diff={mad:.5g}  corr={cc:.6f}")
        all_pass &= ok

    lm.model.close()
    if all_pass:
        print("\nLFM2-Audio compose-embd parity PASSED")
        return 0
    print("\nLFM2-Audio compose-embd parity FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
