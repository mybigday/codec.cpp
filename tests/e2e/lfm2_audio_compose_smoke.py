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

import ctypes
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GGUF = REPO / "models/lfm2_audio/lfm2_audio.gguf"
HF   = "LiquidAI/LFM2-Audio-1.5B"

sys.path.insert(0, str(Path(__file__).parent))
from _codec_lm_ctypes import CodecLM, lib


def must(cond, msg):
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    must(GGUF.is_file(), f"missing {GGUF}; run convert-to-gguf first")

    import torch
    from liquid_audio import LFM2AudioModel  # type: ignore

    print("[hf] loading reference …", flush=True)
    m = LFM2AudioModel.from_pretrained(HF, dtype=torch.float32, device="cpu").eval()
    N = int(m.codebooks)

    print("[cpp] loading codec_lm …", flush=True)
    lm = CodecLM(GGUF)
    must(lm.n_cb == N,
         f"n_cb mismatch lm={lm.n_cb} hf={N}")
    must(lm.compose_audio_embed_dim > 0,
         "compose_audio_embed_dim not set — converter didn't write "
         "lm.compose.audio_embd / codec.lm.compose.audio_embed_dim")
    out_dim = lm.compose_audio_embed_dim
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

        out_buf = (ctypes.c_float * out_dim)()
        codes_i32 = codes.astype(np.int32)
        rc = lib.codec_lm_compose_audio_embd(
            lm.lm,
            codes_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_buf)
        must(rc == 0, f"compose rc={rc} on case {case}")
        cpp = np.frombuffer(out_buf, dtype=np.float32, count=out_dim).copy()

        mad = float(np.max(np.abs(cpp - ref)))
        a = cpp.astype(np.float64); b = ref.astype(np.float64)
        cc  = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
        ok  = mad <= 1e-5 and cc >= 0.99999
        tag = "OK " if ok else "FAIL"
        print(f"  [{tag}] case={case}  codes={codes.tolist()}  "
              f"max_abs_diff={mad:.5g}  corr={cc:.6f}")
        all_pass &= ok

    lm.close()
    if all_pass:
        print("\nLFM2-Audio compose-embd parity PASSED")
        return 0
    print("\nLFM2-Audio compose-embd parity FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
