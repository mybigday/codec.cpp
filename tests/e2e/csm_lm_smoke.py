"""End-to-end parity check: codec_lm residual_depth_ar vs HF CSM.

Loads `sesame/csm-1b` via HF transformers, picks a deterministic
synthetic backbone hidden state, computes per-codebook reference logits
by replaying CSM's actual depth decoder forward (greedy AR loop), then
drives our codec_lm against the same hidden + the same teacher-forced
codes and compares per-cb logits.

Layered against the M3 phase-A smoke (which only checks plumbing /
shape / non-collapse), this verifies:

  - the CSM converter emits audio_embd_{i}, c0 head, depth heads, in_proj
    and the depth-decoder transformer weights faithfully (no axis
    transpose bugs, no missed RoPE freq factors),
  - the Llama depth decoder graph (RMSNorm + GQA + RoPE w/ llama3 piecewise
    scaling + SwiGLU) matches HF's `CsmDepthDecoderModel` numerically,
  - the state machine drives the right inputs at the right positions
    (in_proj(h) at pos 0, in_proj(audio_embd[k-1][c_{k-1}]) at pos k for
    k ≥ 1).

Tolerance: depth-decoder F16 weights vs HF F32 reference; four layers of
matmul accumulation per step plus an in_proj at every prefix position
(up to 32).  Empirically max_abs_diff stays under ~9e-3 across all 32
codebooks with correlation 1.000000 (rounded).  Thresholds below leave
a small buffer above that floor — tight enough to catch real bugs (wrong
tensor layout, off-by-one in the prefix, reversed permute, RoPE mode
flipped between NEOX and interleaved, missing in_proj on h, etc.), loose
enough to absorb F16-vs-F32 noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO  = Path(__file__).resolve().parents[2]
GGUF  = REPO / "models/csm/csm.gguf"
HF_LM = "sesame/csm-1b"

LOGITS_CORR_MIN     = 0.99999
LOGITS_MAX_ABS_DIFF = 2e-2

sys.path.insert(0, str(Path(__file__).parent))
from _codec_lm_ctypes import CodecLM


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def hf_reference(h_in_np: np.ndarray):
    """Greedy-AR through HF's depth decoder, capturing per-cb logits."""
    import torch
    from transformers import CsmForConditionalGeneration

    print("[hf] loading sesame/csm-1b …", flush=True)
    model = CsmForConditionalGeneration.from_pretrained(
        HF_LM, dtype=torch.float32).eval()

    cfg  = model.config
    n_cb = int(cfg.audio_num_codebooks)

    # CSM is in HF transformers proper, so attribute paths are stable.
    c0_head   = model.lm_head                                # Linear(H_b, V)
    depth_dec = model.depth_decoder.model                    # CsmDepthDecoderModel
    cb_head_w = model.depth_decoder.codebooks_head.weight    # (N-1, H_d, V)

    h = torch.from_numpy(h_in_np).float().unsqueeze(0)       # (1, H_b)

    print(f"[hf] greedy-AR over {n_cb} codebooks …", flush=True)
    all_logits: list[np.ndarray] = []
    codes: list[int] = []
    with torch.no_grad():
        # c0 — backbone-side codebook-0 head.
        c0_logits = c0_head(h)[0]                            # (V,)
        all_logits.append(c0_logits.cpu().numpy())
        codes.append(int(c0_logits.argmax().item()))

        # c1..c_{N-1} — replay full prefix through the depth decoder
        # each step (mirroring our prefix-recompute runtime).
        # CsmDepthDecoderModel.forward replaces inputs_embeds[:, 0] with
        # backbone_last_hidden_state, then projects everything via
        # inputs_embeds_projector.
        for k in range(1, n_cb):
            T = k + 1
            input_ids = torch.zeros(1, T, dtype=torch.long)
            for j in range(k):
                input_ids[0, j + 1] = int(codes[j])
            cache_pos = torch.arange(T)
            out = depth_dec.forward(
                input_ids=input_ids,
                backbone_last_hidden_state=h,
                cache_position=cache_pos,
                use_cache=False,
                past_key_values=None,
                return_dict=True,
            )
            depth_h_last = out.last_hidden_state[0, -1]      # (H_d,)
            head_w = cb_head_w[k - 1]                        # (H_d, V)
            ck_logits = depth_h_last @ head_w                # (V,)
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
    must(GGUF.is_file(), f"missing GGUF: {GGUF} (run convert-to-gguf first)")

    rng = np.random.default_rng(42)
    cpp_lm = CodecLM(GGUF)
    must(cpp_lm.host_arch == "llama",
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
        must(cb_idx == k,
             f"cb_idx out of order: got {cb_idx}, expected {k}")
        cpp_logits.append(np.array(lg, dtype=np.float32))
        # teacher-forced: push HF's greedy code, not ours
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
        print("\nCSM codec_lm parity test PASSED")
        return 0
    print("\nCSM codec_lm parity test FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
