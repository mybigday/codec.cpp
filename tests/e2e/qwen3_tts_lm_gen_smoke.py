"""Full AR generation parity: codec.cpp + llama.cpp vs HF Qwen3-TTS.

Stitches the two halves together:

  - HF reference: synthetic deterministic inputs_embeds → talker.model
    (Qwen3 backbone) for the prompt → talker.codec_head + code_predictor
    greedy-AR for each AR frame, reusing the talker's hidden state for
    cb 0 (via lm_head) and the depth decoder for cb 1..N-1.
  - codec.cpp + llama.cpp: same inputs_embeds → llama.cpp Qwen3 backbone
    → codec_lm.step_* for greedy on each codebook → compose next-step
    input embedding via codec_lm_compose_audio_embd → feed back into
    llama.cpp.

Two modes:
  - teacher_forced: each AR step is fed HF's previous codes (not ours).
    Isolates per-step parity from compounding drift.  Asserts <= 30 %
    flip rate and >= 65 % avg per-cb match.
  - free        : each AR step is fed our own previous codes; drift
    compounds.  Reported only.

The Qwen3-TTS talker (28L Qwen3) is materially deeper than CSM's
Llama-3.2-1B (16L), so F16-vs-F32 hidden drift is ~10× larger in
absolute terms (backbone smoke: last-pos max_abs ~0.21 vs CSM's 0.02).
We expect more close-call cb logit flips per step — looser thresholds
than the CSM full-AR test.
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
    str(REPO / "models/qwen3_tts/qwen3_tts_talker.gguf")))
GGUF_CODEC_LM = Path(_os.environ.get(
    "GGUF_CODEC_LM_OVERRIDE",
    str(REPO / "models/qwen3_tts/qwen3_tts_06b_base.gguf")))
HF_LM         = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

MAX_FRAMES_DEFAULT = 8

# Order matters — llama_cpp ships its own libggml.so; importing it first
# locks its symbols into the process, then `_codec_lm_ctypes` dlopens
# libcodec.so with RTLD_LOCAL so the codec's ggml stays private.
import llama_cpp  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from _codec_lm_ctypes import CodecLM


def must(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------
# HF reference
# ---------------------------------------------------------------------

def hf_reference(prompt_embeds_np: np.ndarray, max_frames: int):
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
    talker = model.talker

    tk_cfg = cfg.talker_config
    n_cb   = int(tk_cfg.num_code_groups)

    c0_head     = talker.codec_head
    talker_embd = talker.model.codec_embedding
    cp          = talker.code_predictor
    cp_embds    = cp.model.codec_embedding
    cp_heads    = cp.lm_head
    in_proj     = cp.small_to_mtp_projection

    prompt = torch.from_numpy(prompt_embeds_np).float().unsqueeze(0)  # (1, T, H)
    seq_len  = int(prompt.shape[1])

    print(f"[hf] greedy AR T_prompt={seq_len} n_cb={n_cb} max_frames={max_frames} …",
          flush=True)
    ref_codes: list[list[int]] = []
    ref_hidden_per_step: list[np.ndarray] = []
    accum_embeds = prompt.clone()

    with torch.no_grad():
        for step in range(max_frames):
            T_step = int(accum_embeds.shape[1])
            pos = torch.arange(T_step, dtype=torch.long).view(1, 1, T_step).expand(3, 1, T_step)
            out = talker.model.forward(
                inputs_embeds=accum_embeds,
                position_ids=pos,
                use_cache=False,
                past_key_values=None,
                return_dict=True,
            )
            last_hidden = out.last_hidden_state[0, -1]
            ref_hidden_per_step.append(last_hidden.cpu().float().numpy())

            # c0 — talker.codec_head over talker hidden.
            c0_logits = c0_head(last_hidden.unsqueeze(0))[0]
            codes = [int(c0_logits.argmax().item())]

            # c1..c_{N-1} — greedy via depth decoder (prefix recompute).
            for k in range(1, n_cb):
                embs = [last_hidden.view(1, 1, -1)]
                embs.append(talker_embd(
                    torch.tensor([[codes[0]]], dtype=torch.long)))
                for j in range(1, k):
                    embs.append(cp_embds[j - 1](
                        torch.tensor([[codes[j]]], dtype=torch.long)))
                inputs_embeds = torch.cat(embs, dim=1)
                projected = in_proj(inputs_embeds)
                d_out = cp.model.forward(
                    inputs_embeds=projected,
                    use_cache=False,
                    past_key_values=None,
                    return_dict=True,
                )
                ck_logits = cp_heads[k - 1](d_out.last_hidden_state[0, -1])
                codes.append(int(ck_logits.argmax().item()))

            ref_codes.append(codes)

            # Compose next-step input embedding: sum over (talker.codec_embedding(c0)
            # + code_predictor.codec_embedding[i-1](c_i) for i=1..N-1).
            terms = [talker_embd(torch.tensor([[codes[0]]], dtype=torch.long))]
            for i in range(1, n_cb):
                terms.append(cp_embds[i - 1](
                    torch.tensor([[codes[i]]], dtype=torch.long)))
            next_embd = torch.stack(terms, dim=1).sum(dim=1)  # (1, 1, H_t)
            accum_embeds = torch.cat([accum_embeds, next_embd], dim=1)

    return {
        "prompt_embeds": prompt[0].cpu().float().numpy(),
        "codes":         np.array(ref_codes, dtype=np.int32),
        "hidden":        np.stack(ref_hidden_per_step, 0),
        "n_cb":          n_cb,
    }


# ---------------------------------------------------------------------
# llama.cpp backbone wrapper
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
        codes_out.append(int(np.argmax(arr)))
        state.step_push_code(codes_out[-1])
    state.step_finish()
    return codes_out


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def run(prompt_embeds: np.ndarray, max_frames: int, mode: str) -> int:
    must(GGUF_BACKBONE.is_file(), f"missing {GGUF_BACKBONE}")
    must(GGUF_CODEC_LM.is_file(), f"missing {GGUF_CODEC_LM}")

    ref = hf_reference(prompt_embeds, max_frames)
    n_cb = ref["n_cb"]
    actual_steps = ref["codes"].shape[0]
    print(f"[hf] generated {actual_steps} frames", flush=True)

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
        bad = int((cpp_codes_arr != ref["codes"]).sum())
        total = int(cpp_codes_arr.size)
        flip_rate = bad / total if total else 0.0
        avg_match = sum(n_match_per_cb) / max(1, n_cb * actual_steps)
        print(f"teacher-forced total flips: {bad}/{total} ({flip_rate*100:.2f}%); "
              f"avg per-cb match {avg_match*100:.2f}%")
        # Looser than CSM's 25%/70% — Qwen3-TTS talker is 28L vs CSM's
        # 16L, so backbone F16 drift is ~10x bigger in absolute terms.
        # That feeds straight into close-call cb-logit flips.
        if flip_rate > 0.30 or avg_match < 0.65:
            print("FAIL: teacher-forced flip rate > 30 % or avg match < 65 %",
                  file=sys.stderr)
            return 1
    elif mode == "free":
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

    # Deterministic synthetic prompt: random N(0, 0.1) input embeddings.
    rng = np.random.default_rng(1234)
    T = 16
    hidden = 1024
    prompt_embeds = rng.normal(0.0, 0.1, (T, hidden)).astype(np.float32)

    rc = 0
    if args.mode in ("teacher_forced", "both"):
        print("\n========== TEACHER-FORCED ==========\n")
        rc |= run(prompt_embeds, args.steps, "teacher_forced")
    if args.mode in ("free", "both"):
        print("\n========== FREE-RUNNING ==========\n")
        rc |= run(prompt_embeds, args.steps, "free")

    if rc == 0:
        print("\nQwen3-TTS full AR generation parity test PASSED")
    else:
        print("\nQwen3-TTS full AR generation parity test FAILED",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
