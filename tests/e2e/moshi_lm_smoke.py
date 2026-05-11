"""End-to-end parity check: codec_lm residual_depth_ar (flexible) vs HF Moshi.

Loads `kmhf/hf-moshiko` via HF transformers, picks a deterministic
synthetic backbone hidden state h_in and seed text_tok, then computes
per-codebook reference logits by replaying Moshi's `MoshiDepthDecoder`
forward (greedy AR loop).  Drives our codec_lm against the same h_in +
text_tok and teacher-forces the prefix with HF's codes, comparing
per-cb logits.

This exercises:

  - the converter's mapping of `depth_decoder.*` -> `lm.depth.*` 3D
    flexible tensors + `lm.depth.text_embd` + `lm.depth.audio_embd_*`.
  - the runtime's `is_flexible` init branch (loads 3D weights, single
    `flex_heads`, optional output norm absent).
  - the flexible depth-step graph: per-position in_proj via batched
    mul_mat on a 3D weight slice, per-position q/k/v/o + ffn_gate/up/down
    via the same pattern, no RoPE, no output norm, per-position lm_heads
    slice at the last prefix position.
  - the text-context API: `codec_lm_state_set_text_context` must be
    called before step_begin for `c0_input_modality=text` models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO  = Path(__file__).resolve().parents[2]
GGUF  = REPO / "models/moshi/moshiko.gguf"
HF_LM = "kmhf/hf-moshiko"

LOGITS_CORR_MIN     = 0.99999
LOGITS_MAX_ABS_DIFF = 2e-2

sys.path.insert(0, str(Path(__file__).parent))
from _codec_lm_ctypes import CodecLM


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def hf_reference(h_in_np: np.ndarray, text_token: int):
    """Greedy-AR through HF's depth decoder; returns per-cb logits + codes."""
    import torch
    from transformers import MoshiForConditionalGeneration

    print("[hf] loading kmhf/hf-moshiko …", flush=True)
    model = MoshiForConditionalGeneration.from_pretrained(
        HF_LM, torch_dtype=torch.float32).eval()
    depth = model.depth_decoder
    cfg   = model.config
    n_cb  = int(cfg.num_codebooks)

    h = torch.from_numpy(h_in_np).float().view(1, 1, -1)  # (1, 1, hidden)

    print(f"[hf] greedy-AR over {n_cb} codebooks …", flush=True)
    all_logits: list[np.ndarray] = []
    codes: list[int] = []
    with torch.no_grad():
        for k in range(n_cb):
            T = k + 1
            ids = torch.zeros(1, T, dtype=torch.long)
            ids[0, 0] = text_token
            for j in range(k):
                ids[0, j + 1] = int(codes[j])
            cache_pos = torch.arange(T)
            out = depth(
                input_ids=ids,
                last_hidden_state=h,
                cache_position=cache_pos,
                use_cache=False,
                past_key_values=None,
                return_dict=True,
            )
            # `out.logits` shape: (1, T, audio_vocab).  The cb-k prediction
            # is at the last position (index T-1, generated with
            # lm_heads.weight[k]).
            ck_logits = out.logits[0, T - 1]
            all_logits.append(ck_logits.cpu().numpy())
            codes.append(int(ck_logits.argmax().item()))

    return all_logits, codes


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    must(GGUF.is_file(), f"missing GGUF: {GGUF} "
                        f"(run convert-to-gguf.py --model-id {HF_LM} first)")

    cpp_lm = CodecLM(GGUF)
    must(cpp_lm.host_arch == "llama",
         f"unexpected host_arch={cpp_lm.host_arch!r}")

    # Deterministic synthetic h_in + text token.
    rng = np.random.default_rng(42)
    h_in       = rng.normal(0.0, 0.1, cpp_lm.hidden_dim).astype(np.float32)
    text_token = 1234

    ref_logits, ref_codes = hf_reference(h_in, text_token)
    n_cb = cpp_lm.n_cb
    must(len(ref_logits) == n_cb,
         f"hf produced {len(ref_logits)} logits, expected {n_cb}")

    print("[cpp] running codec_lm state machine (teacher-forced) …", flush=True)
    state = cpp_lm.state()
    state.set_text_context(text_token)
    state.step_begin(h_in)
    cpp_logits: list[np.ndarray] = []
    for k in range(n_cb):
        cb_idx, n, lg = state.step_logits()
        must(cb_idx == k, f"cb_idx out of order: got {cb_idx}, expected {k}")
        cpp_logits.append(np.array(lg, dtype=np.float32))
        state.step_push_code(int(ref_codes[k]))
    state.step_finish()
    state.close()

    vocab = ref_logits[0].shape[0]
    print(f"\nPer-codebook parity (n_cb={n_cb}, vocab={vocab}):")
    all_pass = True
    for k in range(n_cb):
        r = ref_logits[k]
        c = cpp_logits[k]
        must(c.shape == r.shape,
             f"cb={k} shape cpp={c.shape} hf={r.shape}")
        mad = float(np.max(np.abs(c - r)))
        cc  = corr(c, r)
        ok  = (mad <= LOGITS_MAX_ABS_DIFF) and (cc >= LOGITS_CORR_MIN)
        tag = "OK " if ok else "FAIL"
        print(f"  [{tag}] cb={k:2d}  max_abs_diff={mad:.5g}  corr={cc:.6f}")
        all_pass &= ok

    cpp_lm.close()

    if all_pass:
        print("\nMoshi codec_lm parity test PASSED")
        return 0
    print("\nMoshi codec_lm parity test FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
