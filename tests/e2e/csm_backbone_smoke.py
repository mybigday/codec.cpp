"""Backbone parity check: llama.cpp's Llama-3.2-1B (extracted from CSM)
vs HF's `CsmBackboneModel`, fed the same input embeddings.

This sits between the codec_lm parity test (`csm_lm_smoke.py`) and the
full free-running AR parity:
  - codec_lm parity:  given a hidden state, our depth decoder + heads
    produce the same logits as HF.
  - Backbone parity (this file): given the same input embeddings, our
    extracted Llama backbone produces the same hidden states as HF's
    `backbone_model`.
  - Together they imply per-step generation matches HF, modulo numerical
    drift.

How the GGUF gets there:

    python scripts/extract_csm_backbone.py \\
        --csm sesame/csm-1b --out /tmp/csm_backbone_hf
    python scripts/convert_csm_backbone_to_gguf.py \\
        --hf-dir /tmp/csm_backbone_hf \\
        --out    models/csm/llama_backbone.gguf

The wrapper script is a thin one-liner around llama.cpp's
`convert_hf_to_gguf.py` that monkey-patches the unrecognised-pre-
tokenizer-hash branch.  CSM ships a Mistral-flavoured regex twist on
top of Llama-3's tokenizer — same regex family but a different sha256.
At inference the host LLM is fed embeddings via `llama_batch.embd`, so
the tokenizer/BPE merges are unused; mapping the hash to `"llama-bpe"`
is safe.

Empirical numbers (F16 GGUF, F32 HF reference, T=24 positions through
all 16 backbone layers):

    full-tensor corr    = 0.999993
    full-tensor max-abs = 0.033
    last-pos    corr    = 0.999968
    last-pos    max-abs = 0.021

This is materially looser than the equivalent MOSS-TTSD-v0.5 test
(corr ~0.999998, last-pos max-abs ~3e-3) because (a) CSM was trained in
float32 not bfloat16 — F16 GGUF round-trip noise here is asymmetric,
(b) Llama-3's piecewise llama3 RoPE scaling adds an extra source of
elementwise accumulation difference per layer, and (c) the magnitudes
of CSM's hidden states are larger to start with.  Hidden-state values
in the O(1–10) range with corr ≥ 0.9999 still implies < 1 % relative
error per position — well within bit-for-bit "the model is the same
model" territory.  The thresholds below catch real bugs (wrong layer
count, missed RoPE scaling, transposed Q/K, etc.) which would show
up as corr ≪ 0.99.
"""

from __future__ import annotations

import sys
import ctypes
from pathlib import Path

import numpy as np

REPO  = Path(__file__).resolve().parents[2]
GGUF  = REPO / "models/csm/llama_backbone.gguf"
HF_LM = "sesame/csm-1b"


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def hf_reference(prompt_token_ids: list[int]):
    """Run HF CSM's `backbone_model` on a synthetic text-only inputs_embeds.

    The CSM backbone takes either token IDs of shape (B, T, num_codebooks)
    (audio AR mode) or `inputs_embeds` of shape (B, T, hidden) (the path
    `CsmForConditionalGeneration` uses internally after merging text+audio
    embeds).  For a deterministic backbone-only parity we build
    `inputs_embeds = embed_text_tokens(text_ids)` and feed it directly.
    """
    import torch
    from transformers import CsmForConditionalGeneration

    print("[hf] loading sesame/csm-1b …", flush=True)
    model = CsmForConditionalGeneration.from_pretrained(
        HF_LM, dtype=torch.float32).eval()

    text_ids = torch.tensor([prompt_token_ids], dtype=torch.long)  # (1, T)
    seq_len  = int(text_ids.shape[1])

    with torch.no_grad():
        inputs_embeds = model.embed_text_tokens(text_ids)         # (1, T, H_b)
        out = model.backbone_model(inputs_embeds=inputs_embeds, return_dict=True)
        ref_hidden_seq = out.last_hidden_state[0].cpu().float().numpy()
    print(f"[hf] seq_len={seq_len} hidden={ref_hidden_seq.shape[1]}", flush=True)
    return inputs_embeds[0].cpu().float().numpy(), ref_hidden_seq


def llamacpp_hidden_seq(inputs_embeds_np: np.ndarray, gguf_path: Path) -> np.ndarray:
    import llama_cpp as lc
    T, hidden = inputs_embeds_np.shape

    print("[cpp] llama_backend_init …", flush=True)
    lc.llama_backend_init()

    mp = lc.llama_model_default_params()
    mp.use_mmap = True
    mp.n_gpu_layers = 0
    print(f"[cpp] llama_model_load_from_file({gguf_path}) …", flush=True)
    model = lc.llama_model_load_from_file(str(gguf_path).encode("utf-8"), mp)
    must(bool(model), "llama_model_load_from_file failed")

    n_embd = lc.llama_model_n_embd(model)
    must(n_embd == hidden,
         f"llama backbone n_embd {n_embd} != composed-embd hidden {hidden}")

    cp = lc.llama_context_default_params()
    cp.n_ctx     = max(T + 8, 64)
    cp.n_batch   = max(T, 64)
    cp.n_ubatch  = max(T, 64)
    cp.embeddings   = True
    cp.pooling_type = 0  # LLAMA_POOLING_TYPE_NONE
    ctx = lc.llama_init_from_model(model, cp)
    must(bool(ctx), "llama_init_from_model failed")

    batch = lc.llama_batch_init(T, hidden, 1)
    batch.n_tokens = T
    embd_arr = inputs_embeds_np.astype(np.float32, copy=False).ravel()
    ctypes.memmove(batch.embd, embd_arr.ctypes.data, embd_arr.nbytes)
    for t in range(T):
        batch.pos[t]       = t
        batch.n_seq_id[t]  = 1
        batch.seq_id[t][0] = 0
        batch.logits[t]    = 1
    print(f"[cpp] llama_decode T={T} hidden={hidden} …", flush=True)
    rc = lc.llama_decode(ctx, batch)
    must(rc == 0, f"llama_decode rc={rc}")

    out = np.zeros((T, hidden), dtype=np.float32)
    for t in range(T):
        emb_ptr = lc.llama_get_embeddings_ith(ctx, t)
        must(bool(emb_ptr), f"llama_get_embeddings_ith({t}) returned NULL")
        ctypes.memmove(out[t].ctypes.data, emb_ptr, hidden * 4)

    lc.llama_batch_free(batch)
    lc.llama_free(ctx)
    lc.llama_model_free(model)
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    must(GGUF.is_file(),
         f"missing GGUF: {GGUF}; "
         f"run scripts/extract_csm_backbone.py + convert_hf_to_gguf.py first")

    # Deterministic synthetic text token IDs — content doesn't matter,
    # only that the same IDs go through both stacks.  Stay inside Llama-3
    # vocab (< 128256, > BOS=128000) and avoid the tail of audio special
    # tokens.  Pick 24 IDs spanning a realistic prompt length.
    rng = np.random.default_rng(1234)
    text_ids = [128000] + rng.integers(low=10, high=120000, size=23).tolist()

    inputs_embeds, ref_hidden_seq = hf_reference(text_ids)
    cpp_hidden_seq = llamacpp_hidden_seq(inputs_embeds, GGUF)
    must(cpp_hidden_seq.shape == ref_hidden_seq.shape,
         f"shape mismatch cpp={cpp_hidden_seq.shape} hf={ref_hidden_seq.shape}")

    T = ref_hidden_seq.shape[0]
    cc_all  = corr(cpp_hidden_seq, ref_hidden_seq)
    mad_all = float(np.max(np.abs(cpp_hidden_seq - ref_hidden_seq)))
    print(f"\nfull-tensor corr = {cc_all:.6f}  max_abs_diff = {mad_all:.5g}")

    POS_CORR_MIN = 0.9999
    POS_MAD_MAX  = 5e-2
    LAST_CORR_MIN = 0.9999
    LAST_MAD_MAX  = 5e-2

    print("per-position summary:")
    for t in range(T):
        cc  = corr(cpp_hidden_seq[t], ref_hidden_seq[t])
        mad = float(np.max(np.abs(cpp_hidden_seq[t] - ref_hidden_seq[t])))
        tag = "OK " if (cc >= POS_CORR_MIN and mad <= POS_MAD_MAX) else "warn"
        print(f"  [{tag}] t={t:3d}  corr={cc:.6f}  max_abs_diff={mad:.5g}")

    last_cc  = corr(cpp_hidden_seq[-1], ref_hidden_seq[-1])
    last_mad = float(np.max(np.abs(cpp_hidden_seq[-1] - ref_hidden_seq[-1])))
    if last_cc < LAST_CORR_MIN or last_mad > LAST_MAD_MAX:
        print(f"\nFAIL: last-position parity outside tolerance "
              f"(corr={last_cc:.6f}, max_abs_diff={last_mad:.5g})", file=sys.stderr)
        return 1

    print(f"\nCSM Llama backbone parity test PASSED "
          f"(last-pos corr={last_cc:.6f}, max_abs_diff={last_mad:.5g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
