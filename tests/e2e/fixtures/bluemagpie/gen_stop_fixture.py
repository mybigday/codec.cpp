"""One-off generator for the BlueMagpie stop-parity fixture (stop_gen.npz).

Runs the real BlueMagpie reference `_inference` on a short zero-shot ("null"
speaker) zh-TW prompt with deterministic (recorded) CFM noise, and dumps
everything the codec.cpp orchestration smoke test needs to teacher-force and
compare the stop head:

  prefill_hiddens  [T_text, h_barbet]   barbet prefill hiddens (RALM prefix input)
  step_hiddens     [n_steps, h_barbet]  per-iteration barbet step hiddens (i>=1)
  noise{i}         [D, P]               recorded CFM z injected at iteration i
  patch{i}         [D, P]               reference predicted latent patch
  stop_logits{i}   [2]                  raw stop_head logits at iteration i
  n_steps                               number of iterations produced
  stop_index                            first iteration where the ref broke
                                        (== n_steps when it hit natural stop),
  min_len, cfg, ts, h_barbet, D, P      scalars

Environment: needs the BlueMagpie + Barbet source packages importable.
  BM_SRC   default /tmp/BlueMagpie-TTS/src
  BARBET_SRC default /tmp/Barbet/src
Run from the repo root with the project venv:
  .venv/bin/python tests/e2e/fixtures/bluemagpie/gen_stop_fixture.py

This is scaffolding — the .npz it writes is the committed artefact; the E2E
smoke (bluemagpie_stop_smoke.py) is the standing test.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

BM_SRC = os.environ.get("BM_SRC", "/tmp/BlueMagpie-TTS/src")
BARBET_SRC = os.environ.get("BARBET_SRC", "/tmp/Barbet/src")
sys.path.insert(0, BM_SRC)
sys.path.insert(0, BARBET_SRC)

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
MODEL_DIR = REPO / "models" / "bluemagpie"
OUT = Path(__file__).resolve().parent / "stop_gen.npz"

# Short zero-shot zh-TW sentence; long enough to produce a natural stop within
# ~10-20 patches at patch_size=4.
TEXT = "你好，今天天氣真好。"  # 你好，今天天氣真好。
MIN_LEN = 2
CFG = 2.8
TS = 9
MAX_LEN = 40


def main() -> int:
    torch.manual_seed(0)
    from bluemagpie import BlueMagpieModel
    from bluemagpie._vendor.voxcpm.modules.locdit.unified_cfm import UnifiedCFM

    # The transformers AutoTokenizer wrapper chokes on this checkpoint's
    # `extra_special_tokens` list; load the fast tokenizer.json directly and adapt
    # it to the tiny `.encode(text, add_special_tokens=...)` interface _tokenize
    # needs (returns a list of ids).
    from tokenizers import Tokenizer
    class _TkAdapter:
        def __init__(self, path):
            self.tk = Tokenizer.from_file(path)
        def encode(self, text, add_special_tokens=False):
            return self.tk.encode(text, add_special_tokens=add_special_tokens).ids
    tokenizer = _TkAdapter(str(MODEL_DIR / "tokenizer.json"))

    model = BlueMagpieModel.from_local(str(MODEL_DIR), tokenizer=tokenizer, device="cpu")
    model = model.float().eval()
    # The residual_lm static KV cache was allocated at __init__ in the config
    # dtype (bf16); re-allocate it in float32 to match the float() weights.
    model.residual_lm.setup_cache(1, model.config.max_length, "cpu", torch.float32)
    model.config.dtype = "float32"

    h_barbet = model.tslm_adapter.norm.weight.shape[0]

    # ---- capture prefill + per-step barbet hiddens ----
    prefill_hiddens = {}
    step_hiddens = []

    real_prefill = model.base_lm.prefill
    def prefill_hook(inputs_embeds):
        hs, state = real_prefill(inputs_embeds)
        prefill_hiddens["h"] = hs[0].detach().to(torch.float32).cpu().numpy()  # [T, h_barbet]
        return hs, state
    model.base_lm.prefill = prefill_hook

    real_step = model.base_lm.forward_step
    def step_hook(x, state):
        h = real_step(x, state)
        step_hiddens.append(h[0].detach().to(torch.float32).cpu().numpy())  # [h_barbet]
        return h
    model.base_lm.forward_step = step_hook

    # ---- capture raw stop_head logits (forward hook on the module) ----
    stop_logits = []
    def stop_fwd_hook(_mod, _inp, out):
        stop_logits.append(out[0].detach().to(torch.float32).cpu().numpy())  # [2]
    model.stop_head.register_forward_hook(stop_fwd_hook)

    # ---- inject deterministic recorded noise into the CFM ----
    noises = []
    patches = []
    real_forward = UnifiedCFM.forward
    def cfm_forward(self, mu, n_timesteps, patch_size, cond, temperature=1.0,
                    cfg_value=1.0, sway_sampling_coef=1.0, use_cfg_zero_star=True):
        b, _ = mu.shape
        t = patch_size
        z = (torch.randn((b, self.in_channels, t), device=mu.device, dtype=mu.dtype)
             * temperature)
        noises.append(z[0].detach().to(torch.float32).cpu().numpy())  # [D, P]
        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        out = self.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond,
                               cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
        patches.append(out[0].detach().to(torch.float32).cpu().numpy())  # [D, P]
        return out
    UnifiedCFM.forward = cfm_forward

    with torch.inference_mode():
        _audio = model.generate(
            target_text=TEXT,
            use_null_speaker=True,
            min_len=MIN_LEN,
            max_len=MAX_LEN,
            inference_timesteps=TS,
            cfg_value=CFG,
        )

    n_steps = len(patches)
    assert n_steps == len(noises) == len(stop_logits), (n_steps, len(noises), len(stop_logits))
    # The reference loop appends the patch, then checks stop, then breaks: the
    # last captured iteration is the one that fired the stop (or max_len).  The
    # step_hiddens list has one entry per NON-final iteration (forward_step runs
    # after the stop check, so it is skipped on the stopping iteration) — pad it
    # so step_hiddens[i] is defined for the teacher-forced feed at iteration i>=1.
    P = patches[0].shape[1]
    D = patches[0].shape[0]

    # Determine the reference stop index by re-applying the guard to the logits.
    stop_index = n_steps  # sentinel: no stop (hit max_len)
    for i in range(n_steps):
        fired = int(np.argmax(stop_logits[i])) == 1
        if i > MIN_LEN and fired:
            stop_index = i
            break

    out = {
        "prefill_hiddens": prefill_hiddens["h"].astype(np.float32),
        "step_hiddens": np.stack(step_hiddens).astype(np.float32) if step_hiddens else np.zeros((0, D), np.float32),
        "n_steps": np.int32(n_steps),
        "stop_index": np.int32(stop_index),
        "min_len": np.int32(MIN_LEN),
        "cfg": np.float32(CFG),
        "ts": np.int32(TS),
        "h_barbet": np.int32(h_barbet),
        "D": np.int32(D),
        "P": np.int32(P),
    }
    for i in range(n_steps):
        out[f"noise{i}"] = noises[i].astype(np.float32)
        out[f"patch{i}"] = patches[i].astype(np.float32)
        out[f"stop_logits{i}"] = stop_logits[i].astype(np.float32)

    np.savez(OUT, **out)
    print(f"wrote {OUT}")
    print(f"  T_text={prefill_hiddens['h'].shape[0]} n_steps={n_steps} "
          f"stop_index={stop_index} n_step_hiddens={len(step_hiddens)} D={D} P={P}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
