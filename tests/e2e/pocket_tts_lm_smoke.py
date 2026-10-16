"""Pocket-TTS FlowLM teacher-forced parity smoke test.

Drives the codec_lm flow_lm C API on the converted GGUF and replays the exact
recorded-noise trajectory from the reference (tests/e2e/fixtures/pocket_tts/
gen_lm_fixtures.py):

  - tokenize "Hello world."         -> must match reference token ids
  - prefill (text-only, no voice)
  - for each step: feed the RECORDED noise, feed the RECORDED previous latent
    (teacher-forced), compare the emitted latent + EOS logit to the reference.

Teacher-forcing (feeding the reference's previous latent each step instead of
the codec's own) keeps the trajectory aligned so every emitted latent stays
comparable — free-running feedback diverges after a few steps under F16 weights.

Run:  .venv/bin/python tests/e2e/pocket_tts_lm_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

from _common import REPO
import codec_cpp

GGUF = REPO / "models" / "pocket_tts" / "pocket_tts_en.gguf"
FIX = REPO / "tests" / "e2e" / "fixtures" / "pocket_tts"

LATENT_CORR_MIN = 0.99
EOS_AGREE_MIN = 1.0   # per-step EOS decision must match every step
TEXT = "Hello world."


def main() -> int:
    for p in (GGUF, FIX / "lm_tokens.npy", FIX / "lm_noise.npy",
              FIX / "lm_latents.npy", FIX / "lm_eos.npy", FIX / "lm_meta.npz"):
        if not p.is_file():
            print(f"FAIL: missing {p}")
            return 1

    ref_tokens = np.load(FIX / "lm_tokens.npy").astype(np.int32)
    ref_noise = np.load(FIX / "lm_noise.npy").astype(np.float32)     # [n, ldim]
    ref_lat = np.load(FIX / "lm_latents.npy").astype(np.float32)     # [n, ldim]
    ref_eos = np.load(FIX / "lm_eos.npy").astype(np.float32)         # [n]
    meta = np.load(FIX / "lm_meta.npz")
    eos_thr = float(meta["eos_threshold"])
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

    # ---- prefill (text-only) ----
    st.flow_prefill(got_tokens)

    # ---- teacher-forced AR steps ----
    got_lat = np.zeros((n_steps, ldim), dtype=np.float32)
    got_eos = np.zeros((n_steps,), dtype=np.float32)

    for step in range(n_steps):
        # For step>0, seed the step with the REFERENCE previous latent (teacher).
        prev = ref_lat[step - 1] if step > 0 else None
        got_lat[step], got_eos[step], _ = st.flow_step(ref_noise[step], prev_latent=prev)

    lat_corr = float(np.corrcoef(got_lat.reshape(-1), ref_lat.reshape(-1))[0, 1])
    lat_max_abs = float(np.max(np.abs(got_lat - ref_lat)))
    eos_agree = float(np.mean((got_eos > eos_thr) == (ref_eos > eos_thr)))
    eos_max_abs = float(np.max(np.abs(got_eos - ref_eos)))

    print(f"latent: corr={lat_corr:.6f} max_abs={lat_max_abs:.4e}")
    print(f"eos:    agree={eos_agree:.3f} logit_max_abs={eos_max_abs:.4e}")
    print(f"  ref eos[:5]={np.round(ref_eos[:5],3).tolist()}")
    print(f"  got eos[:5]={np.round(got_eos[:5],3).tolist()}")

    ok = tok_ok and lat_corr >= LATENT_CORR_MIN and eos_agree >= EOS_AGREE_MIN
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
