"""Backbone parity check: llama.cpp's Qwen3 vs HF's Qwen3-TTS talker,
fed the same input embeddings.

Qwen3-TTS's talker uses MRoPE with an interleaved 3-section split
`[24, 20, 20]` for cross-modal (text + image/video + audio) sequences,
but for pure-text (or pure-audio, like TTS) sequences all three MRoPE
channels share the same position ids — per the upstream `get_rope_index`
docstring:

    "For pure text embedding sequence, the rotary position embedding
     has no difference with modern LLMs."

And the interleaved-MRoPE math reduces to vanilla 1D RoPE in that case
(the channel-mixing strided assignment becomes a no-op when all
channels are equal).  Llama.cpp's `qwen3` arch applies 1D RoPE — that's
the exact reduced case.  This test verifies the collapse holds
numerically.

How the GGUF gets there:

    python scripts/extract_qwen3_tts_backbone.py \\
        --qwen Qwen/Qwen3-TTS-12Hz-0.6B-Base --out /tmp/qwen3_tts_talker_hf
    python <llama.cpp>/convert_hf_to_gguf.py /tmp/qwen3_tts_talker_hf \\
        --outfile models/qwen3_tts/qwen3_tts_talker.gguf --outtype f16
"""

from __future__ import annotations

import sys
import ctypes
from pathlib import Path

import numpy as np

REPO  = Path(__file__).resolve().parents[2]
GGUF  = REPO / "models/qwen3_tts/qwen3_tts_talker.gguf"
HF_LM = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def hf_reference(input_embeds_np: np.ndarray):
    """Run HF Qwen3-TTS talker's `model.forward(inputs_embeds=...)`."""
    import torch
    sys.path.insert(0, str(REPO / ".model-src/Qwen3-TTS"))
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

    print("[hf] loading Qwen3-TTS-12Hz-0.6B-Base …", flush=True)
    cfg = Qwen3TTSConfig.from_pretrained(HF_LM)
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        HF_LM, config=cfg, torch_dtype=torch.float32).eval()
    talker = model.talker.model

    h = torch.from_numpy(input_embeds_np).float().unsqueeze(0)   # (1, T, H)
    T = int(h.shape[1])

    # For pure-text (or pure-audio TTS) sequences `get_rope_index` returns
    # all 3 MRoPE channels equal to a 1D cumsum-arange.  We replicate that
    # explicitly to keep position_ids deterministic.
    pos = torch.arange(T, dtype=torch.long).view(1, 1, T).expand(3, 1, T)

    with torch.no_grad():
        out = talker.forward(
            inputs_embeds=h,
            position_ids=pos,
            use_cache=False,
            past_key_values=None,
            return_dict=True,
        )
        ref_hidden_seq = out.last_hidden_state[0].cpu().float().numpy()
    print(f"[hf] seq_len={T} hidden={ref_hidden_seq.shape[1]}", flush=True)
    return ref_hidden_seq


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
         f"llama backbone n_embd {n_embd} != input hidden {hidden}")

    cp = lc.llama_context_default_params()
    cp.n_ctx       = max(T + 8, 64)
    cp.n_batch     = max(T, 64)
    cp.n_ubatch    = max(T, 64)
    cp.embeddings  = True
    cp.pooling_type = 0
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
         f"run scripts/extract_qwen3_tts_backbone.py + convert_hf_to_gguf.py first")

    # Deterministic synthetic inputs_embeds — content doesn't matter,
    # only that the same embeddings go through both stacks.
    rng = np.random.default_rng(1234)
    T = 16
    hidden = 1024
    inputs_embeds = rng.normal(0.0, 0.1, (T, hidden)).astype(np.float32)

    ref_hidden_seq = hf_reference(inputs_embeds)
    cpp_hidden_seq = llamacpp_hidden_seq(inputs_embeds, GGUF)
    must(cpp_hidden_seq.shape == ref_hidden_seq.shape,
         f"shape mismatch cpp={cpp_hidden_seq.shape} hf={ref_hidden_seq.shape}")

    cc_all  = corr(cpp_hidden_seq, ref_hidden_seq)
    mad_all = float(np.max(np.abs(cpp_hidden_seq - ref_hidden_seq)))
    print(f"\nfull-tensor corr = {cc_all:.6f}  max_abs_diff = {mad_all:.5g}")

    # Empirical numbers (T=16, F16 GGUF, F32 HF reference, 28L talker):
    #   full-tensor corr  = 0.999945     last-pos corr = 0.999988
    #   full-tensor max-abs = 0.85       last-pos max-abs = 0.21
    # Hidden-state magnitudes peak around O(30), so max_abs of 0.85
    # is < 3 % relative error per position — consistent with F16 weight
    # round-trip accumulating through 28 layers (vs CSM's 16L where the
    # same kind of noise stays under 0.03 because the network is half
    # as deep and the hidden magnitudes are smaller).  Tighter would
    # need a higher-precision GGUF (Q8/F32) or fix-ups in llama.cpp.
    POS_CORR_MIN  = 0.9999
    POS_MAD_MAX   = 1.5
    LAST_CORR_MIN = 0.9999
    LAST_MAD_MAX  = 1.5

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

    print(f"\nQwen3-TTS talker backbone parity test PASSED "
          f"(last-pos corr={last_cc:.6f}, max_abs_diff={last_mad:.5g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
