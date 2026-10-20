"""Pocket-TTS FlowLM teacher-forced parity smoke test.

The FlowLM transformer runs in llama.cpp (stock starcoder2 backbone GGUF from
scripts/convert-backbone-to-gguf.py); codec_lm owns the step input embedding,
the EOS head and the LSD flow head.  Replays the recorded-noise trajectory from
the reference (tests/e2e/fixtures/pocket_tts/gen_lm_fixtures.py):

  - tokenize "Hello world."         -> must match reference token ids
  - prefill the backbone with the text tokens (no voice)
  - each step: feed codec_lm's input embedding, hand the backbone hidden to
    codec_lm_step_generate with the RECORDED noise, and teacher-force the
    RECORDED latent as the next step's input; compare the emitted latent + EOS
    decision to the reference.

Teacher-forcing keeps the trajectory aligned so every emitted latent stays
comparable — free-running feedback diverges after a few steps under F16 weights.

Run:  .venv/bin/python tests/e2e/pocket_tts_lm_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

from _common import REPO, LlamaBackbone   # imports llama_cpp before codec_cpp
import codec_cpp

GGUF = REPO / "models" / "pocket_tts" / "pocket_tts_en.gguf"
BACKBONE = REPO / "models" / "pocket_tts" / "backbone-f16.gguf"
FIX = REPO / "tests" / "e2e" / "fixtures" / "pocket_tts"

LATENT_CORR_MIN = 0.99
EOS_AGREE_MIN = 1.0   # per-step EOS decision must match every step
TEXT = "Hello world."


def main() -> int:
    for p in (GGUF, BACKBONE, FIX / "lm_tokens.npy", FIX / "lm_noise.npy",
              FIX / "lm_latents.npy", FIX / "lm_eos.npy", FIX / "lm_meta.npz"):
        if not p.is_file():
            print(f"FAIL: missing {p}")
            return 1

    ref_tokens = np.load(FIX / "lm_tokens.npy").astype(np.int32)
    ref_noise = np.load(FIX / "lm_noise.npy").astype(np.float32)     # [n, ldim]
    ref_lat = np.load(FIX / "lm_latents.npy").astype(np.float32)     # [n, ldim] normalized
    ref_eos = np.load(FIX / "lm_eos.npy").astype(np.float32)         # [n]
    meta = np.load(FIX / "lm_meta.npz")
    eos_thr = float(meta["eos_threshold"])
    # step_generate emits the denormalized latent (what Mimi decodes).
    ref_lat = ref_lat * meta["emb_std"].reshape(1, -1) + meta["emb_mean"].reshape(1, -1)
    n_steps, ldim = ref_lat.shape

    model = codec_cpp.Model(GGUF, use_gpu=False, n_threads=1)
    lm = model.get_lm_adaptor()
    fi = lm.flow_info()
    if fi is None:
        print("FAIL: not a flow_lm model"); return 1
    print(f"flow_lm: d_model={fi['d_model']} ldim={fi['ldim']} n_bins={fi['n_txt_bins']} "
          f"temp={fi['temperature']:.3f} eos_thr={fi['eos_threshold']:.2f} "
          f"lsd_steps={fi['lsd_decode_steps']} has_tok={fi['has_tokenizer']}")

    # ---- tokenize ----
    got_tokens = lm.flow_tokenize(TEXT, cap=256)
    tok_ok = np.array_equal(got_tokens, ref_tokens)
    print(f"tokens: got {got_tokens.tolist()} ref {ref_tokens.tolist()}  "
          f"{'PASS' if tok_ok else 'FAIL'}")

    st = lm.state()
    got_lat = np.zeros((n_steps, ldim), dtype=np.float32)
    got_eos = np.zeros((n_steps,), dtype=bool)
    with LlamaBackbone(BACKBONE, n_ctx=got_tokens.size + n_steps + 8) as bb:
        bb.feed_tokens(got_tokens)
        for step in range(n_steps):
            h = bb.feed_embeds(st.step_feedback_embd()[None, :])
            st.set_teacher_patch(ref_lat[step])   # the reference latent feeds the next step
            lat, got_eos[step] = st.step_generate(h, cfg_value=0.0, n_timesteps=0, noise=ref_noise[step])
            got_lat[step] = lat.reshape(-1)

    lat_corr = float(np.corrcoef(got_lat.reshape(-1), ref_lat.reshape(-1))[0, 1])
    lat_max_abs = float(np.max(np.abs(got_lat - ref_lat)))
    eos_agree = float(np.mean(got_eos == (ref_eos > eos_thr)))

    print(f"latent: corr={lat_corr:.6f} max_abs={lat_max_abs:.4e}")
    print(f"eos:    agree={eos_agree:.3f}  ref fires at {np.flatnonzero(ref_eos > eos_thr).tolist()}")

    ok = tok_ok and lat_corr >= LATENT_CORR_MIN and eos_agree >= EOS_AGREE_MIN
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
