"""End-to-end parity: codec_lm residual_depth_ar (LFM2-Audio shape) vs
LiquidAI/LFM2-Audio-1.5B's `_sample_audio_frame` reference.

LFM2-Audio's depth path lives in the `liquid_audio` pypi package
(`pip install liquid-audio`).  The reference loop in
`liquid_audio.model.lfm2_audio.LFM2AudioModel._sample_audio_frame`
is what we mirror here, deterministically replaying its greedy AR
through depthformer + per-cb heads given a synthetic backbone hidden
state.

This validates:
  - The converter mapping for `depth_linear` (8192, 2048) -> 3D
    `lm.depth.in_proj.weight` (8, 1024, 2048) + 2D `lm.depth.in_proj.bias`
    (1024, 8), the per-cb `depth_embeddings[i].{embedding, to_logits,
    embedding_norm}` -> `lm.depth.{audio_embd_i, heads_i, heads_i_norm}`,
    and the per-layer qkv_proj split (1536, 1024) -> separate q/k/v.
  - The runtime's new `run_depth_step_lfm2`: host-side prefix with
    zero at pos 0 + audio_embd[k-1] at pos k>=1, plus in-graph per-pos
    in_proj(h_in) + bias added before the shared transformer layers,
    plus per-cb pre-head RMSNorm + per-cb head at the last position.
  - The dispatcher: `step_begin` skips c0_head when depth_emits_c0
    (LFM2 has no separate c0_head), and `step_logits(k)` calls
    `run_depth_step_lfm2` for all k including k=0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO  = Path(__file__).resolve().parents[2]
GGUF  = REPO / "models/lfm2_audio/lfm2_audio.gguf"
HF_LM = "LiquidAI/LFM2-Audio-1.5B"

LOGITS_CORR_MIN     = 0.99999
LOGITS_MAX_ABS_DIFF = 2e-2

sys.path.insert(0, str(Path(__file__).parent))
from _codec_lm_ctypes import CodecLM


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


def hf_reference(h_in_np: np.ndarray):
    """Greedy-AR through liquid_audio's depthformer; per-cb logits + codes."""
    import torch
    from liquid_audio import LFM2AudioModel  # type: ignore

    print(f"[hf] loading {HF_LM} …", flush=True)
    model = LFM2AudioModel.from_pretrained(
        HF_LM, dtype=torch.float32, device="cpu").eval()
    n_cb = int(model.codebooks)
    audio_vocab = int(model.audio_vocab_size)  # 2049

    h = torch.from_numpy(h_in_np).float()  # (hidden_dim=2048,)

    print(f"[hf] greedy-AR over {n_cb} codebooks …", flush=True)
    all_logits: list[np.ndarray] = []
    codes: list[int] = []
    with torch.no_grad():
        # Mirrors `_sample_audio_frame` exactly, but recording per-cb
        # logits instead of just sampled tokens.  Uses uncached forward
        # (full prefix replayed each step) to mirror codec_lm's
        # prefix-recompute regime.
        from einops import rearrange  # liquid_audio dep
        depthformer_in = rearrange(
            model.depth_linear(h),
            "(C D) -> C D",
            C=n_cb, D=model.depthformer_dim,
        )  # (8, 1024)

        for k in range(n_cb):
            # Build prefix length T=k+1.  Pos 0 input = depthformer_in[0]
            # (no embedding lookup).  Pos p>=1 input = depthformer_in[p]
            # + depth_embeddings[p-1](codes[p-1]).
            steps = [depthformer_in[0:1]]
            for p in range(1, k + 1):
                emb = model.depth_embeddings[p - 1](
                    torch.tensor([int(codes[p - 1])]))
                steps.append(depthformer_in[p:p + 1] + emb)
            x = torch.cat(steps, dim=0).unsqueeze(0)  # (1, T, 1024)

            out = model.depthformer(x)                # (1, T, 1024)
            ck_logits = model.depth_embeddings[k].get_logits(out[0, -1])
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
         f"missing GGUF: {GGUF} (run convert-to-gguf.py --model-id {HF_LM} first)")

    cpp_lm = CodecLM(GGUF)
    must(cpp_lm.host_arch == "lfm2",
         f"unexpected host_arch={cpp_lm.host_arch!r}")

    rng = np.random.default_rng(42)
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
        print("\nLFM2-Audio codec_lm parity test PASSED")
        return 0
    print("\nLFM2-Audio codec_lm parity test FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
