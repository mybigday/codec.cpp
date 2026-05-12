"""Full AR generation parity test: codec.cpp + llama.cpp vs HF CSM.

Composes a text prompt's input embeddings via HF's `embed_text_tokens`,
feeds them into the converted Llama backbone GGUF via llama.cpp, runs
the per-step AR loop with greedy sampling on each of the 32 audio
codebooks (codec_lm.step_*), and compares the emitted code sequence
against the same forward arithmetic done on HF's
`CsmForConditionalGeneration` reference.

Layered against the codec_lm and backbone parity tests:
  - codec_lm parity (csm_lm_smoke.py): given a hidden, our depth
    decoder + heads produce HF-equivalent logits (bit-perfect).
  - backbone parity (csm_backbone_smoke.py): given input embeddings,
    our llama.cpp Llama-3.2-1B produces HF-equivalent hidden states
    (corr ~0.999968, last-pos max_abs ~0.021).
  - this test: the two pieces stitched together over many AR steps.

Two parity modes:
  - Teacher-forced: each AR step is fed HF's previous codes (not ours).
    Isolates per-step parity from compounding drift; what's left is the
    intrinsic numerical-implementation gap between llama.cpp's Llama
    and HF's Llama, plus the depth decoder's F16 quantization noise.
  - Free-running:   each AR step is fed our own previous codes.  Drift
    compounds; reported but not strictly asserted.

Expected behaviour:
  - Per-step argmax matches around 80–95 % per codebook in teacher-
    forced mode.  Compounding noise on close-call codebook logits
    causes occasional flips, especially on later codebooks where the
    depth decoder amplifies hidden-state drift.
  - Free-running typically diverges within the first AR step and then
    drifts unrecoverably.  This is intrinsic to running on different
    inference stacks; the audio-level similarity (separately measurable
    via the codec decode) is far more forgiving.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

import os as _os
REPO          = Path(__file__).resolve().parents[2]
GGUF_BACKBONE = Path(_os.environ.get(
    "GGUF_BACKBONE_OVERRIDE",
    str(REPO / "models/csm/llama_backbone.gguf")))
GGUF_CODEC_LM = Path(_os.environ.get(
    "GGUF_CODEC_LM_OVERRIDE",
    str(REPO / "models/csm/csm.gguf")))
HF_LM         = "sesame/csm-1b"

MAX_FRAMES_DEFAULT = 16

# Load order matters: llama-cpp-python ships its own libggml.so which
# is API-incompatible with the one the codec.cpp build uses.  Importing
# llama_cpp first locks its libllama.so + libggml.so into the process;
# `_codec_lm_ctypes` then dlopens libcodec.so so its ggml symbols stay
# private.
import llama_cpp  # noqa: F401  (ordering: must precede _codec_lm_ctypes)

sys.path.insert(0, str(Path(__file__).parent))
from _codec_lm_ctypes import CodecLM


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------
# HF reference: build prompt + run greedy AR for K steps
# ---------------------------------------------------------------------

def hf_reference(text_token_ids: list[int], max_frames: int):
    import torch
    from transformers import CsmForConditionalGeneration

    print("[hf] loading sesame/csm-1b …", flush=True)
    model = CsmForConditionalGeneration.from_pretrained(
        HF_LM, dtype=torch.float32).eval()
    cfg     = model.config
    n_cb    = int(cfg.audio_num_codebooks)
    audio_v = int(cfg.audio_vocab_size)

    text_ids = torch.tensor([text_token_ids], dtype=torch.long)  # (1, T)
    seq_len  = int(text_ids.shape[1])

    print(f"[hf] greedy AR T_prompt={seq_len} n_cb={n_cb} max_frames={max_frames} …",
          flush=True)
    ref_codes: list[list[int]] = []
    ref_hidden_per_step: list[np.ndarray] = []

    backbone   = model.backbone_model
    embed_text = model.embed_text_tokens
    embed_aud  = model.backbone_model.embed_tokens   # CsmBackboneModelEmbeddings
    c0_head    = model.lm_head                        # over audio vocab
    depth_dec  = model.depth_decoder.model
    depth_w    = model.depth_decoder.codebooks_head.weight  # (N-1, H_d, V)

    # Use HF's no-cache forward path.  We accumulate inputs_embeds and
    # re-run the full backbone each step — slow but matches the
    # prefix-recompute pattern our codec_lm uses on the depth side.
    with torch.no_grad():
        prompt_embeds = embed_text(text_ids)                     # (1, T, H_b)
        accum_embeds  = prompt_embeds.clone()

        for step in range(max_frames):
            backbone_out = backbone(inputs_embeds=accum_embeds, return_dict=True)
            last_hidden  = backbone_out.last_hidden_state[0, -1]  # (H_b,)
            ref_hidden_per_step.append(last_hidden.cpu().float().numpy())

            # c0 — greedy on backbone-side lm_head.
            c0_logits = c0_head(last_hidden.unsqueeze(0))[0]
            codes = [int(c0_logits.argmax().item())]

            # c1..c31 — greedy via depth decoder + per-position heads.
            for k in range(1, n_cb):
                T = k + 1
                depth_in_ids = torch.zeros(1, T, dtype=torch.long)
                for j in range(k):
                    depth_in_ids[0, j + 1] = int(codes[j])
                cache_pos = torch.arange(T)
                d_out = depth_dec.forward(
                    input_ids=depth_in_ids,
                    backbone_last_hidden_state=last_hidden.unsqueeze(0),
                    cache_position=cache_pos,
                    use_cache=False,
                    past_key_values=None,
                    return_dict=True,
                )
                dh = d_out.last_hidden_state[0, -1]
                ck_logits = dh @ depth_w[k - 1]                  # (V,)
                codes.append(int(ck_logits.argmax().item()))

            ref_codes.append(codes)

            # Compose next-step input embedding via embed_audio_tokens
            # over the just-sampled codes and append.  embed_aud expects
            # (B, T_seq, num_codebooks) and returns (B, T_seq, H_b)
            # already summed over codebooks.
            next_ids = torch.tensor(codes, dtype=torch.long).view(1, 1, n_cb)
            next_embd = embed_aud(next_ids)                      # (1, 1, H_b)
            accum_embeds = torch.cat([accum_embeds, next_embd], dim=1)

    return {
        "prompt_embeds": prompt_embeds[0].cpu().float().numpy(),
        "codes":         np.array(ref_codes, dtype=np.int32),
        "hidden":        np.stack(ref_hidden_per_step, 0),
        "n_cb":          n_cb,
        "audio_vocab":   audio_v,
    }


# ---------------------------------------------------------------------
# llama.cpp backbone wrapper for AR (same shape as MOSS-TTSD's helper)
# ---------------------------------------------------------------------

class LlamaBackbone:
    def __init__(self, gguf: Path, n_ctx: int):
        import llama_cpp as lc
        self.lc = lc

        lc.llama_backend_init()
        mp = lc.llama_model_default_params()
        mp.use_mmap     = True
        mp.n_gpu_layers = 0
        self.model = lc.llama_model_load_from_file(str(gguf).encode("utf-8"), mp)
        if not self.model:
            raise RuntimeError(f"llama_model_load_from_file failed: {gguf}")
        self.hidden = lc.llama_model_n_embd(self.model)

        cp = lc.llama_context_default_params()
        cp.n_ctx       = n_ctx
        cp.n_batch     = max(n_ctx, 64)
        cp.n_ubatch    = max(n_ctx, 64)
        cp.embeddings  = True
        cp.pooling_type = 0
        self.ctx = lc.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama_init_from_model failed")
        self.pos = 0

    def feed_embeds(self, input_embeds: np.ndarray) -> np.ndarray:
        T = input_embeds.shape[0]
        must(input_embeds.shape[1] == self.hidden,
             f"feed_embeds: width {input_embeds.shape[1]} != model n_embd {self.hidden}")
        batch = self.lc.llama_batch_init(T, self.hidden, 1)
        batch.n_tokens = T
        flat = np.ascontiguousarray(input_embeds.astype(np.float32, copy=False)).ravel()
        ctypes.memmove(batch.embd, flat.ctypes.data, flat.nbytes)
        for t in range(T):
            batch.pos[t]       = self.pos + t
            batch.n_seq_id[t]  = 1
            batch.seq_id[t][0] = 0
            batch.logits[t]    = 1 if t == T - 1 else 0
        rc = self.lc.llama_decode(self.ctx, batch)
        if rc != 0:
            self.lc.llama_batch_free(batch)
            raise RuntimeError(f"llama_decode rc={rc}")
        emb_ptr = self.lc.llama_get_embeddings_ith(self.ctx, 0)
        if not emb_ptr:
            self.lc.llama_batch_free(batch)
            raise RuntimeError("llama_get_embeddings_ith(0) NULL")
        out = np.zeros(self.hidden, dtype=np.float32)
        ctypes.memmove(out.ctypes.data, emb_ptr, self.hidden * 4)
        self.lc.llama_batch_free(batch)
        self.pos += T
        return out

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lc.llama_free(self.ctx); self.ctx = None
        if getattr(self, "model", None):
            self.lc.llama_model_free(self.model); self.model = None


# ---------------------------------------------------------------------
# Greedy step on codec_lm
# ---------------------------------------------------------------------

def codec_lm_greedy_step(state, h_in: np.ndarray) -> list[int]:
    state.step_begin(h_in)
    codes_out: list[int] = []
    for _ in range(state.parent.n_cb):
        cb_idx, n, lg = state.step_logits()
        arr = np.frombuffer(lg, dtype=np.float32)
        best = int(np.argmax(arr))
        codes_out.append(best)
        state.step_push_code(best)
    state.step_finish()
    return codes_out


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def run(text_token_ids: list[int], max_frames: int, mode: str) -> int:
    must(GGUF_BACKBONE.is_file(),
         f"missing {GGUF_BACKBONE}; run `python scripts/convert-backbone-to-gguf.py "
         f"--model-id sesame/csm-1b --output {GGUF_BACKBONE}` first")
    must(GGUF_CODEC_LM.is_file(),
         f"missing {GGUF_CODEC_LM}; run scripts/convert-to-gguf.py on "
         f"sesame/csm-1b first")

    ref = hf_reference(text_token_ids, max_frames)
    n_cb = ref["n_cb"]
    actual_steps = ref["codes"].shape[0]
    print(f"[hf] generated {actual_steps} frames (codes shape={ref['codes'].shape})",
          flush=True)

    print(f"[cpp] loading codec_lm + backbone …", flush=True)
    cpp_lm = CodecLM(GGUF_CODEC_LM)
    must(cpp_lm.n_cb == n_cb, f"n_cb mismatch cpp={cpp_lm.n_cb} hf={n_cb}")
    must(cpp_lm.hidden_dim == ref["prompt_embeds"].shape[1],
         f"hidden mismatch cpp={cpp_lm.hidden_dim} prompt={ref['prompt_embeds'].shape[1]}")

    n_ctx = ref["prompt_embeds"].shape[0] + max_frames + 8
    backbone = LlamaBackbone(GGUF_BACKBONE, n_ctx=n_ctx)

    state = cpp_lm.state()
    cur_hidden = backbone.feed_embeds(ref["prompt_embeds"])

    cpp_codes: list[list[int]] = []
    n_match_per_cb = [0] * n_cb
    first_mismatch_step = -1

    for step in range(actual_steps):
        codes = codec_lm_greedy_step(state, cur_hidden)
        cpp_codes.append(codes)

        ref_step = ref["codes"][step]
        for cb in range(n_cb):
            if codes[cb] == ref_step[cb]:
                n_match_per_cb[cb] += 1

        full_match = all(codes[cb] == int(ref_step[cb]) for cb in range(n_cb))
        if not full_match and first_mismatch_step < 0:
            first_mismatch_step = step
            print(f"  [step {step:3d}] mismatches:")
            for cb in range(n_cb):
                if codes[cb] != int(ref_step[cb]):
                    print(f"      cb={cb} cpp={codes[cb]:5d} hf={int(ref_step[cb]):5d}")

        feed_codes = codes if mode == "free" else [int(x) for x in ref_step]
        embd = cpp_lm.compose_audio_embd(feed_codes)
        next_embeds = np.frombuffer(embd, dtype=np.float32).reshape(1, -1).copy()
        cur_hidden = backbone.feed_embeds(next_embeds)

    cpp_codes_arr = np.array(cpp_codes, dtype=np.int32)
    print(f"\n--- {mode} mode results (steps={actual_steps}, n_cb={n_cb}) ---")
    print("Per-codebook match counts:")
    for cb in range(n_cb):
        rate = n_match_per_cb[cb] / max(1, actual_steps)
        print(f"  cb={cb:2d}  match={n_match_per_cb[cb]:3d}/{actual_steps}  "
              f"({rate*100:.1f}%)  vocab={cpp_lm.codebook_sizes[cb]}")
    print(f"first full-frame mismatch step: "
          f"{first_mismatch_step if first_mismatch_step >= 0 else 'none'}")

    state.close()
    cpp_lm.close()
    backbone.close()

    if mode == "teacher_forced":
        # Numerical-implementation gap budget: codec_lm side is bit-perfect
        # (csm_lm_smoke.py); backbone side is corr ~0.999968 / last-pos
        # max_abs ~0.021 (csm_backbone_smoke.py).  That much per-step drift
        # is enough to flip a fraction of the close-call codebook logits.
        # We require:
        #   - flip rate ≤ 25 % overall
        #   - per-codebook average match rate ≥ 70 %
        # Tighter would need either a quantization upgrade on the backbone
        # GGUF or upstream Llama-numerics work in llama.cpp.
        bad = int((cpp_codes_arr != ref["codes"]).sum())
        total = int(cpp_codes_arr.size)
        flip_rate = bad / total if total else 0.0
        avg_match = sum(n_match_per_cb) / max(1, n_cb * actual_steps)
        print(f"teacher-forced total flips: {bad}/{total} ({flip_rate*100:.2f}%); "
              f"avg per-cb match {avg_match*100:.2f}%")
        if flip_rate > 0.25 or avg_match < 0.70:
            print("FAIL: teacher-forced flip rate > 25 % or avg match < 70 %",
                  file=sys.stderr)
            return 1
    elif mode == "free":
        # Free-running drift is intrinsic and not asserted; we report
        # how many codebooks tracked HF for >= 25 % of the steps.
        n_tracking = sum(1 for cb in range(n_cb)
                         if n_match_per_cb[cb] >= max(1, actual_steps // 4))
        print(f"free-running: {n_tracking}/{n_cb} codebooks tracking HF "
              f"for >= 25 % of steps")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=MAX_FRAMES_DEFAULT)
    ap.add_argument("--mode", choices=["teacher_forced", "free", "both"],
                    default="both")
    args = ap.parse_args()

    # Deterministic synthetic text token IDs — same prompt for both modes
    # so results are reproducible.  Stay inside Llama-3 vocab (< 128256,
    # > BOS=128000) and avoid the audio special token tail.
    rng = np.random.default_rng(1234)
    text_token_ids = [128000] + rng.integers(low=10, high=120000, size=23).tolist()

    rc = 0
    if args.mode in ("teacher_forced", "both"):
        print("\n========== TEACHER-FORCED ==========\n")
        rc |= run(text_token_ids, args.steps, "teacher_forced")
    if args.mode in ("free", "both"):
        print("\n========== FREE-RUNNING ==========\n")
        rc |= run(text_token_ids, args.steps, "free")

    if rc == 0:
        print("\nCSM full AR generation parity test PASSED")
    else:
        print("\nCSM full AR generation parity test FAILED", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
