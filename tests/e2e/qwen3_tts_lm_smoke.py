"""End-to-end parity check: codec_lm residual_depth_ar vs HF Qwen3-TTS.

Loads `Qwen/Qwen3-TTS-12Hz-0.6B-Base` via HF transformers, picks a
deterministic synthetic talker hidden state, computes per-codebook
reference logits by replaying Qwen3-TTS's actual code_predictor forward
(greedy AR loop), then drives our codec_lm against the same hidden +
the same teacher-forced codes and compares per-cb logits.

Layered against the M3 CSM parity tests, this verifies:

  - the Qwen3-TTS converter emits audio_embd_{i}, c0 head, depth heads,
    and the 5-layer Qwen3 depth transformer weights faithfully (per-cb
    embed table indices, q/k/v/o projection shapes asymmetric because
    n_heads*head_dim != hidden, q_norm/k_norm tensors),
  - the residual_depth_ar runtime handles the Qwen3-specific knobs
    (`has_qk_norm=true`, `has_in_proj=false` since talker.hidden ==
    depth.hidden for the 0.6B variant, rope_theta=1e6, no llama3 RoPE
    scaling),
  - the non-homogeneous codebook_sizes are honored (cb0 vocab is 3072
    spanning the talker's codec specials, cb1..15 are 2048 codec-only).

Tolerance: depth decoder F16 weights vs HF F32 reference, 5 layers of
matmul accumulation per step.  Empirically per-cb max_abs_diff stays
well under 5e-3 with correlation 1.000000 (rounded).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO  = Path(__file__).resolve().parents[2]
GGUF  = REPO / "models/qwen3_tts/qwen3_tts_06b_base.gguf"
HF_LM = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

LOGITS_CORR_MIN     = 0.99999
LOGITS_MAX_ABS_DIFF = 1e-2

sys.path.insert(0, str(Path(__file__).parent))
from _codec_lm_ctypes import CodecLM


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def hf_reference(h_in_np: np.ndarray):
    """Greedy-AR through HF's talker.codec_head + code_predictor,
    capturing per-cb logits."""
    import torch
    # Qwen3-TTS isn't in HF transformers proper — load via the vendored
    # `.model-src/Qwen3-TTS/qwen_tts/` package.
    sys.path.insert(0, str(REPO / ".model-src/Qwen3-TTS"))
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

    print("[hf] loading Qwen3-TTS-12Hz-0.6B-Base …", flush=True)
    cfg = Qwen3TTSConfig.from_pretrained(HF_LM)
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        HF_LM, config=cfg, torch_dtype=torch.float32).eval()
    talker = model.talker

    tk_cfg = cfg.talker_config
    n_cb   = int(tk_cfg.num_code_groups)

    c0_head     = talker.codec_head                        # Linear(H_t, V_c0)
    talker_embd = talker.model.codec_embedding             # Embedding(V_c0, H_t)
    cp          = talker.code_predictor                    # CodePredictorForGen
    cp_embds    = cp.model.codec_embedding                 # ModuleList[N-1] of Embedding
    cp_heads    = cp.lm_head                               # ModuleList[N-1] of Linear
    in_proj     = cp.small_to_mtp_projection               # Identity for 0.6B

    h = torch.from_numpy(h_in_np).float().unsqueeze(0)     # (1, H_t)

    print(f"[hf] greedy-AR over {n_cb} codebooks …", flush=True)
    all_logits: list[np.ndarray] = []
    codes: list[int] = []
    with torch.no_grad():
        # c0 — talker.codec_head over talker hidden.
        c0_logits = c0_head(h)[0]                          # (V_c0,)
        all_logits.append(c0_logits.cpu().numpy())
        codes.append(int(c0_logits.argmax().item()))

        # c1..c_{N-1} — replay prefix through depth decoder each step.
        for k in range(1, n_cb):
            # Build prefix embeds: pos 0 = h, pos 1 = talker_embd(c0),
            # pos i = cp_embds[i-2](c_{i-1}) for i = 2..k.
            embs = [h.unsqueeze(1)]                        # (1, 1, H_t)
            embs.append(talker_embd(
                torch.tensor([[codes[0]]], dtype=torch.long)))
            for j in range(1, k):
                embs.append(cp_embds[j - 1](
                    torch.tensor([[codes[j]]], dtype=torch.long)))
            inputs_embeds = torch.cat(embs, dim=1)         # (1, k+1, H_t/H_d)
            projected = in_proj(inputs_embeds)             # (1, k+1, H_d)

            out = cp.model.forward(
                inputs_embeds=projected,
                use_cache=False,
                past_key_values=None,
                return_dict=True,
            )
            depth_h_last = out.last_hidden_state[0, -1]    # (H_d,)
            ck_logits = cp_heads[k - 1](depth_h_last)      # (V,)
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
    must(GGUF.is_file(),
         f"missing GGUF: {GGUF} (run convert-to-gguf.py "
         f"--model-id {HF_LM} --output ... first)")

    rng = np.random.default_rng(42)
    cpp_lm = CodecLM(GGUF)
    must(cpp_lm.host_arch == "qwen3",
         f"unexpected host_arch={cpp_lm.host_arch!r}")

    h_in = rng.normal(0.0, 0.1, cpp_lm.hidden_dim).astype(np.float32)

    ref_logits, ref_codes = hf_reference(h_in)
    n_cb = cpp_lm.n_cb
    must(len(ref_logits) == n_cb,
         f"hf produced {len(ref_logits)} logits, expected {n_cb}")

    print("[cpp] running codec_lm state machine (teacher-forced) …", flush=True)
    state = cpp_lm.state()
    state.step_begin(h_in)
    cpp_logits: list[np.ndarray] = []
    for k in range(n_cb):
        cb_idx, n, lg = state.step_logits()
        must(cb_idx == k, f"cb_idx out of order: got {cb_idx}, expected {k}")
        cpp_logits.append(np.array(lg, dtype=np.float32))
        state.step_push_code(int(ref_codes[k]))
    state.step_finish()
    state.close()

    print(f"\nPer-codebook parity (n_cb={n_cb}):")
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
        print(f"  [{tag}] cb={k:2d}  vocab={r.shape[0]:5d}  "
              f"max_abs_diff={mad:.5g}  corr={cc:.6f}")
        all_pass &= ok

    cpp_lm.close()

    if all_pass:
        print("\nQwen3-TTS-0.6B-Base codec_lm parity test PASSED")
        return 0
    print("\nQwen3-TTS-0.6B-Base codec_lm parity test FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
