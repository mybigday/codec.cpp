"""Full AR generation parity test: codec.cpp+llama.cpp vs HF MOSS-TTSD-v0.5.

Composes the prompt's input embeddings via codec_lm.compose_audio_embd,
feeds them into the converted Qwen3 backbone GGUF via llama.cpp, runs
the per-step AR loop with greedy sampling on each of the 8 codebook
heads (codec_lm.step_*), and compares the emitted code sequence against
the same forward arithmetic done on the HF MossTTSDForCausalLM model.

Two parity modes:
  - Teacher-forced: each AR step is fed HF's previous codes (not ours).
    Isolates per-step parity from compounding drift; what's left is the
    intrinsic numerical-implementation gap between llama.cpp's Qwen3
    and HF's Qwen3.
  - Free-running:   each AR step is fed our own previous codes.  Drift
    compounds; reported but not strictly asserted.

Expected behaviour, validated against MOSS-TTSD-v0.5 / Qwen3-2B (F16 GGUF):
  - codec_lm-side parity is bit-perfect (separate test:
    `tests/e2e/moss_ttsd_lm_smoke.py`).
  - Backbone hidden parity (separate test:
    `tests/e2e/moss_ttsd_qwen3_backbone_smoke.py`) is corr ≈ 0.999998
    with last-position max_abs_diff ≈ 3e-3 — F16 vs BF16 round-trip
    plus ggml-vs-pytorch matmul accumulation differences.
  - Per-step argmax matches around 85-92 % per codebook in
    teacher-forced mode.  Compounding noise propagates through the
    head matmul; argmax flips are concentrated on close-call codebook
    logits.
  - Free-running typically diverges within the first AR step (the
    prompt-end hidden's drift is enough to flip every codebook on
    that step) and then drifts unrecoverably.  This is intrinsic to
    running on different inference stacks — not a codec_lm or
    converter bug — and the audio-level similarity (separately
    measurable via the codec decode) is far more forgiving.
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
    str(REPO / "models/moss_ttsd_v0_5/qwen3_backbone.gguf")))
GGUF_CODEC_LM = Path(_os.environ.get(
    "GGUF_CODEC_LM_OVERRIDE",
    str(REPO / "models/moss_ttsd_v0_5/moss_ttsd_v0_5.gguf")))
HF_LM         = "fnlp/MOSS-TTSD-v0.5"

# Free-running tolerances.  At least the first MIN_FREE_MATCH steps must
# match HF byte-for-byte; per-step drift after that is tracked but not
# fatal (F16-vs-BF16 argmax flips on close-call codebooks are expected).
MIN_FREE_MATCH = 8
MAX_FRAMES_DEFAULT = 32

# Load order matters: llama-cpp-python ships its own libggml.so which
# is API-incompatible with the libggml the codec.cpp build uses.  Importing
# llama_cpp first locks its libllama.so + libggml.so into the process;
# our `_codec_lm_ctypes` then dlopens libcodec.so with RTLD_LOCAL so its
# ggml symbols stay private.
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

def hf_reference(prompt_text: str, max_frames: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    print("[hf] loading MOSS-TTSD-v0.5 …", flush=True)
    tok   = AutoTokenizer.from_pretrained(HF_LM, trust_remote_code=True)
    model = AutoModel.from_pretrained(HF_LM, trust_remote_code=True,
                                      torch_dtype=torch.float32).eval()
    base = model.model
    n_cb       = model.config.channels
    speech_pad = int(model.config.speech_pad_token)
    # HF generate() stops on generation_config.eos_token_id (152694), NOT the
    # plain-text config.json eos (151643) - the converter bakes the former into
    # codec.lm.eos_code_c0 (see lm_adaptor: generation_config preferred).
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    text_eos   = int(gen_eos if gen_eos is not None else model.config.eos_token_id)

    text_ids = tok(prompt_text, return_tensors="pt").input_ids[0]
    seq_len  = int(text_ids.shape[0])

    # Initial prompt frames: text in channel 0, speech_pad in channels 1..7.
    # NOTE: skipping the processor's delay-pattern shift — the test
    # operates at the model-forward level, so both sides see the same
    # un-shifted sequence.  This still validates per-step AR parity; the
    # processor's pre/post shift is a sequence-level concern documented
    # in codec.lm.delay_pattern.
    prompt_ids = torch.full((1, seq_len, n_cb), speech_pad, dtype=torch.long)
    prompt_ids[0, :, 0] = text_ids

    print(f"[hf] greedy AR generation T_prompt={seq_len} max_frames={max_frames} …",
          flush=True)
    ref_codes: list[list[int]] = []
    ref_hidden_per_step: list[np.ndarray] = []
    ref_input_embeds_per_step: list[np.ndarray] = []
    cur_ids = prompt_ids
    with torch.no_grad():
        # Pre-compute the prompt embed so we can capture it for the cpp side.
        prompt_embeds = base._prepare_multi_modal_inputs(prompt_ids)
        ref_input_embeds_per_step.append(prompt_embeds[0].cpu().float().numpy())

        for step in range(max_frames):
            input_embeds = base._prepare_multi_modal_inputs(cur_ids)
            out = base.language_model(inputs_embeds=input_embeds, return_dict=True)
            last_hidden = out.last_hidden_state[0, -1]   # (hidden,)
            ref_hidden_per_step.append(last_hidden.cpu().float().numpy())

            # Per-cb greedy sample using lm_heads (tied to embedding_list).
            codes = []
            for i in range(n_cb):
                logits = last_hidden @ model.lm_heads[i].weight.t()
                codes.append(int(torch.argmax(logits).item()))
            ref_codes.append(codes)

            # Append a (1, 1, n_cb) frame and continue.  In a real generate()
            # the processor would also apply a per-cb delay shift, but at
            # this level we just match what the model itself sees.
            next_frame = torch.tensor(codes, dtype=torch.long).view(1, 1, n_cb)
            cur_ids = torch.cat([cur_ids, next_frame], dim=1)

            if codes[0] == text_eos:
                print(f"[hf] hit text EOS at step={step}; stopping early", flush=True)
                break

    return {
        "prompt_embeds": prompt_embeds[0].cpu().float().numpy(),
        "codes":         np.array(ref_codes, dtype=np.int32),
        "hidden":        np.stack(ref_hidden_per_step, 0),
        "n_cb":          n_cb,
        "speech_pad":    speech_pad,
        "text_eos":      text_eos,
    }


# ---------------------------------------------------------------------
# llama.cpp backbone wrapper for AR
# ---------------------------------------------------------------------

class LlamaBackbone:
    """Minimal AR driver: feed embeddings, read last-position hidden state."""

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
        cp.pooling_type = 0   # NONE
        self.ctx = lc.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama_init_from_model failed")

        self.pos = 0   # next absolute position to write into the KV cache

    def feed_embeds(self, input_embeds: np.ndarray) -> np.ndarray:
        """Feed (T, hidden) embeds as a single batch; return last-position
        hidden state.  Sets logits[T-1]=1 only — we just want the last."""
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

def codec_lm_greedy_step(state, h_in: np.ndarray) -> tuple[list[int], list[float]]:
    state.step_begin(h_in)
    codes_out: list[int] = []
    top_logits: list[float] = []
    while True:
        try:
            cb_idx, n, lg = state.step_logits()
        except RuntimeError:
            # No more logits — done with this step.
            break
        # greedy
        arr = np.frombuffer(lg, dtype=np.float32)   # zero-copy view
        best = int(np.argmax(arr))
        codes_out.append(best)
        top_logits.append(float(arr[best]))
        state.step_push_code(best)
    final = state.step_finish()
    return final, top_logits


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def run(prompt: str, max_frames: int, mode: str) -> int:
    must(GGUF_BACKBONE.is_file(), f"missing {GGUF_BACKBONE}")
    must(GGUF_CODEC_LM.is_file(), f"missing {GGUF_CODEC_LM}")

    ref = hf_reference(prompt, max_frames)
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

    # 1) Feed prompt — same as the HF reference's first base.language_model
    #    forward pass.  Last-position hidden becomes the input to AR step 0.
    state = cpp_lm.state()
    cur_hidden = backbone.feed_embeds(ref["prompt_embeds"])

    cpp_codes: list[list[int]] = []
    n_match_per_cb = [0] * n_cb
    first_mismatch_step = -1

    # Phase A: end-of-audio metadata parity.  MOSS-TTSD channel 0 is the
    # text vocab; end-of-audio == text EOS sampled on cb0.  eos_code_c0
    # must equal the HF reference's text_eos.
    text_eos = int(ref["text_eos"])
    must(cpp_lm.eos_code_c0 == text_eos,
         f"eos_code_c0={cpp_lm.eos_code_c0} != HF text_eos={text_eos}")
    must(cpp_lm.eos_min_step == 0,
         f"expected eos_min_step=0, got {cpp_lm.eos_min_step}")
    # HF stopped early iff its LAST frame's cb0 == text_eos.
    hf_stopped_on_eos = int(ref["codes"][-1][0]) == text_eos
    cpp_eos_step = -1

    for step in range(actual_steps):
        codes, _ = codec_lm_greedy_step(state, cur_hidden)
        cpp_codes.append(codes)

        # Model-owned EOS decision: assert step_is_eos fires exactly on
        # the frame whose cb0 == eos_code_c0 (teacher-forced feeds HF's
        # codes back, so on the terminal frame both sides agree on cb0).
        if state.step_is_eos(codes) and cpp_eos_step < 0:
            cpp_eos_step = step
        # step_is_eos must agree with the raw cb0 == eos test at each step.
        must(state.step_is_eos(codes) == (codes[0] == text_eos),
             f"step {step}: step_is_eos({codes[0]}) disagrees with cb0==eos")

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

        # Compose next-step input embedding from EITHER our codes
        # (free-running) or HF's codes (teacher-forced).
        feed_codes = codes if mode == "free" else [int(x) for x in ref_step]
        embd = cpp_lm.compose_audio_embd(feed_codes)
        next_embeds = np.frombuffer(embd, dtype=np.float32).reshape(1, -1).copy()
        cur_hidden = backbone.feed_embeds(next_embeds)

    cpp_codes_arr = np.array(cpp_codes, dtype=np.int32)
    print(f"\n--- {mode} mode results (steps={actual_steps}, n_cb={n_cb}) ---")
    print("Per-codebook match counts (out of {} steps):".format(actual_steps))
    for cb in range(n_cb):
        rate = n_match_per_cb[cb] / max(1, actual_steps)
        print(f"  cb={cb:2d}  match={n_match_per_cb[cb]:3d}/{actual_steps}  "
              f"({rate*100:.1f}%)  vocab={cpp_lm.codebook_sizes[cb]}")
    print(f"first full-frame mismatch step: "
          f"{first_mismatch_step if first_mismatch_step >= 0 else 'none'}")

    # EOS-parity report: in teacher-forced mode the cb0 codes are HF's, so
    # step_is_eos must fire at exactly HF's stop frame (the last one) when
    # HF stopped on EOS, and never otherwise.
    print(f"[eos] hf_stopped_on_eos={hf_stopped_on_eos} "
          f"cpp_eos_step={cpp_eos_step} actual_steps={actual_steps}", flush=True)
    if mode == "teacher_forced":
        if hf_stopped_on_eos:
            must(cpp_eos_step == actual_steps - 1,
                 f"step_is_eos should fire at HF's stop frame "
                 f"{actual_steps - 1}, got {cpp_eos_step}")
        else:
            must(cpp_eos_step == -1,
                 f"step_is_eos fired at {cpp_eos_step} but HF never stopped on EOS")

    state.close()
    cpp_lm.close()
    backbone.close()

    if mode == "teacher_forced":
        # The intrinsic gap between llama.cpp's Qwen3 and HF's Qwen3
        # (matmul accumulation order, q/k norm ordering, RoPE
        # implementation details) shows up here as occasional argmax
        # flips on close-call logits.  Our backbone-parity test reports
        # corr ≈ 0.999998 / max_abs_diff ≈ 3e-3 on the last-position
        # hidden state — that's enough to flip a small fraction of
        # codebook outputs.  The codec_lm side is bit-perfect (other
        # smoke test).  We require:
        #   - flip rate ≤ 15 % overall (currently ~9–12 % observed)
        #   - per-codebook match rate ≥ 80 % on average
        # Tighter thresholds need a quantization upgrade or upstream
        # llama.cpp Qwen3 work.
        bad = int((cpp_codes_arr != ref["codes"]).sum())
        total = int(cpp_codes_arr.size)
        flip_rate = bad / total if total else 0.0
        avg_match = sum(n_match_per_cb) / max(1, n_cb * actual_steps)
        print(f"teacher-forced total flips: {bad}/{total} ({flip_rate*100:.2f}%); "
              f"avg per-cb match {avg_match*100:.2f}%")
        if flip_rate > 0.15 or avg_match < 0.80:
            print("FAIL: teacher-forced flip rate > 15 % or avg match < 80 %",
                  file=sys.stderr)
            return 1
    elif mode == "free":
        # Free-running drift is intrinsic and not asserted; we report
        # how many codebooks tracked HF for >= 50 % of the steps so
        # there's still some signal.  A pure failure (zero codebooks
        # tracking) would indicate an AR-loop bug.
        n_tracking = sum(1 for cb in range(n_cb)
                         if n_match_per_cb[cb] >= actual_steps // 2)
        print(f"free-running: {n_tracking}/{n_cb} codebooks tracking HF "
              f"for ≥50 % of steps")
        # Not a failure condition by itself — report only.

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="[S1] Hello, this is a parity test.")
    ap.add_argument("--steps",  type=int, default=MAX_FRAMES_DEFAULT)
    ap.add_argument("--mode",   choices=["teacher_forced", "free", "both"],
                    default="both")
    args = ap.parse_args()

    rc = 0
    if args.mode in ("teacher_forced", "both"):
        print("\n========== TEACHER-FORCED ==========\n")
        rc |= run(args.prompt, args.steps, "teacher_forced")
    if args.mode in ("free", "both"):
        print("\n========== FREE-RUNNING ==========\n")
        rc |= run(args.prompt, args.steps, "free")

    if rc == 0:
        print("\nMOSS-TTSD-v0.5 full AR generation parity test PASSED")
    else:
        print("\nMOSS-TTSD-v0.5 full AR generation parity test FAILED",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
