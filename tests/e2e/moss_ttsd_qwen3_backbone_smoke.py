"""Backbone parity check: llama.cpp's Qwen3 backbone vs HF's Qwen3 inside
MOSS-TTSD-v0.5, fed the same composed input embeddings.

This sits between the existing codec_lm parity test and the full AR
generation parity:
- codec_lm parity (already passing): given a hidden state, our parallel
  heads produce the same logits as HF's lm_heads.
- Backbone parity (this file): given the same input embeddings, our
  llama.cpp Qwen3 produces the same hidden states as HF's
  language_model.
- Together they imply the full per-step generation is identical, modulo
  numerical drift.
"""

from __future__ import annotations

import sys
import ctypes
from pathlib import Path

import numpy as np

REPO  = Path(__file__).resolve().parents[2]
GGUF  = REPO / "models/moss_ttsd_v0_5/qwen3_backbone.gguf"
HF_LM = "fnlp/MOSS-TTSD-v0.5"


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def hf_reference(prompt_text: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    print("[hf] loading MOSS-TTSD-v0.5 …", flush=True)
    tok   = AutoTokenizer.from_pretrained(HF_LM, trust_remote_code=True)
    model = AutoModel.from_pretrained(HF_LM, trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    base = model.model
    n_cb       = model.config.channels
    speech_pad = int(model.config.speech_pad_token)
    text_ids = tok(prompt_text, return_tensors="pt").input_ids[0]
    seq_len  = int(text_ids.shape[0])
    input_ids = torch.full((1, seq_len, n_cb), speech_pad, dtype=torch.long)
    input_ids[0, :, 0] = text_ids
    with torch.no_grad():
        inputs_embeds = base._prepare_multi_modal_inputs(input_ids)
        out = base.language_model(inputs_embeds=inputs_embeds, return_dict=True)
        ref_hidden_seq = out.last_hidden_state[0].cpu().float().numpy()
    print(f"[hf] seq_len={seq_len} hidden={ref_hidden_seq.shape[1]}", flush=True)
    return inputs_embeds[0].cpu().float().numpy(), ref_hidden_seq, n_cb


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
    must(GGUF.is_file(), f"missing GGUF: {GGUF}; "
                        f"run scripts/extract_qwen3_backbone.py + convert_hf_to_gguf.py first")
    prompt = "[S1] Hello, this is a parity test."
    inputs_embeds, ref_hidden_seq, n_cb = hf_reference(prompt)
    cpp_hidden_seq = llamacpp_hidden_seq(inputs_embeds, GGUF)
    must(cpp_hidden_seq.shape == ref_hidden_seq.shape,
         f"shape mismatch cpp={cpp_hidden_seq.shape} hf={ref_hidden_seq.shape}")

    T = ref_hidden_seq.shape[0]
    cc_all  = corr(cpp_hidden_seq, ref_hidden_seq)
    mad_all = float(np.max(np.abs(cpp_hidden_seq - ref_hidden_seq)))
    print(f"\nfull-tensor corr = {cc_all:.6f}  max_abs_diff = {mad_all:.5g}")

    print("per-position summary:")
    for t in range(T):
        cc  = corr(cpp_hidden_seq[t], ref_hidden_seq[t])
        mad = float(np.max(np.abs(cpp_hidden_seq[t] - ref_hidden_seq[t])))
        tag = "OK " if (cc >= 0.999 and mad <= 5e-3) else "warn"
        print(f"  [{tag}] t={t:3d}  corr={cc:.6f}  max_abs_diff={mad:.5g}")

    last_cc  = corr(cpp_hidden_seq[-1], ref_hidden_seq[-1])
    last_mad = float(np.max(np.abs(cpp_hidden_seq[-1] - ref_hidden_seq[-1])))
    if last_cc < 0.999 or last_mad > 5e-3:
        print(f"\nFAIL: last-position parity outside tolerance "
              f"(corr={last_cc:.6f}, max_abs_diff={last_mad:.5g})", file=sys.stderr)
        return 1

    print(f"\nMOSS-TTSD-v0.5 Qwen3 backbone parity test PASSED "
          f"(last-pos corr={last_cc:.6f}, max_abs_diff={last_mad:.5g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
