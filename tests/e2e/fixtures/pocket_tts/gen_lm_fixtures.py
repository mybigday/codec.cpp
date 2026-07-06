"""Generate Pocket-TTS FlowLM teacher-forced parity fixtures from the reference.

Drives the reference FlowLMModel over a short English utterance with the AR
latent teacher-forced by the model's own greedy generation (recorded noise), so
codec.cpp can replay the exact trajectory deterministically:

  - text tokens (int32)                 -> tokens.npy
  - per-step recorded LSD init noise     -> noise.npy   [n_steps, ldim]
  - per-step generated latent (pre-norm) -> latents.npy [n_steps, ldim]
  - per-step EOS logit (scalar)          -> eos.npy     [n_steps]
  - config scalars                       -> meta.npz

The AR loop mirrors TTSModel._autoregressive_generation:
  - backbone_input starts as NaN (BOS), input_linear(NaN->bos_emb)
  - each step: transformer_out[-1] -> out_eos logit + lsd_decode(noise) -> latent
  - next backbone_input = latent
The KV cache holds [text embeds | AR latent embeds] (no voice; text-only cond,
matching TTSModel with a plain init_states and no audio prompt).

fp32 end to end.  Run:
  .venv/bin/python tests/e2e/fixtures/pocket_tts/gen_lm_fixtures.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / ".model-src" / "pocket-tts"))

from pocket_tts.conditioners.base import TokenizedText           # noqa: E402
from pocket_tts.models.flow_lm import FlowLMModel, lsd_decode    # noqa: E402
from pocket_tts.modules.stateful_module import (                 # noqa: E402
    increment_steps, init_states)
from pocket_tts.utils.config import CONFIGS_DIR, load_config     # noqa: E402

ST = REPO / "models" / "pocket_tts" / "model.safetensors"
OUT = Path(__file__).resolve().parent

TEXT = "Hello world."
N_STEPS = 24           # generate this many AR frames (teacher-forced)
TEMP = 0.7
EOS_THRESHOLD = -4.0
LSD_STEPS = 1
SEED = 4242


def build_flow_lm(cfg) -> FlowLMModel:
    import safetensors
    import torch.nn as nn
    flm = FlowLMModel.from_pydantic_config(
        cfg.flow_lm, latent_dim=cfg.mimi.quantizer.dimension,
        insert_bos_before_voice=cfg.flow_lm.insert_bos_before_voice)
    flm.speaker_proj_weight = nn.Parameter(
        torch.zeros((cfg.flow_lm.transformer.d_model,
                     cfg.mimi.inner_dim or cfg.mimi.seanet.dimension)))
    sd = {}
    with safetensors.safe_open(str(ST), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("flow_lm."):
                sd[k[len("flow_lm."):]] = f.get_tensor(k)
    missing, unexpected = flm.load_state_dict(sd, strict=False)
    assert not [m for m in missing], f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    # Register StatefulModule absolute names (TTSModel does this at load).
    from pocket_tts.modules.stateful_module import StatefulModule
    for name, mod in flm.named_modules():
        if isinstance(mod, StatefulModule):
            mod._module_absolute_name = name
    flm.eval()
    return flm


@torch.no_grad()
def main() -> int:
    torch.manual_seed(SEED)
    cfg = load_config(CONFIGS_DIR / "english.yaml")
    flm = build_flow_lm(cfg)
    ldim = flm.ldim

    tok = flm.conditioner.tokenizer
    tokens = torch.tensor(tok.sp.encode(TEXT, out_type=int))[None, :]  # [1, n_txt]
    n_txt = tokens.shape[1]

    max_seq = n_txt + N_STEPS + 4
    model_state = init_states(flm, batch_size=1, sequence_length=max_seq)

    # ---- text prefill (prompt): run the transformer over text embeddings,
    # filling the KV cache.  Mirrors _run_flow_lm's text-only path. ----
    text_embeddings = flm.conditioner(TokenizedText(tokens))     # [1, n_txt, d_model]
    flm.transformer(text_embeddings, model_state)                # fills KV; output discarded
    increment_steps(flm, model_state, increment=n_txt)

    noise_all = np.zeros((N_STEPS, ldim), dtype=np.float32)
    latents_all = np.zeros((N_STEPS, ldim), dtype=np.float32)
    eos_all = np.zeros((N_STEPS,), dtype=np.float32)

    empty_text = torch.zeros((1, 0), dtype=torch.int64)
    backbone_input = torch.full((1, 1, ldim), float("nan"), dtype=flm.dtype)

    for step in range(N_STEPS):
        te = flm.conditioner(TokenizedText(empty_text))          # [1,0,d_model]
        # --- replicate FlowLMModel.forward but record noise + eos logit ---
        sequence = backbone_input
        seq = torch.where(torch.isnan(sequence), flm.bos_emb, sequence)
        input_ = flm.input_linear(seq)
        transformer_out = flm.backbone(input_, te, sequence, model_state=model_state)
        transformer_out = transformer_out.to(torch.float32)[:, -1]    # [1, d_model]

        eos_logit = flm.out_eos(transformer_out)                      # [1,1]
        eos_all[step] = float(eos_logit.view(-1)[0])

        std = TEMP ** 0.5
        noise = torch.empty((1, ldim), dtype=transformer_out.dtype)
        torch.nn.init.normal_(noise, mean=0.0, std=std)
        noise_all[step] = noise[0].numpy()

        conditioned_flow = partial(flm.flow_net, transformer_out)
        latent = lsd_decode(conditioned_flow, noise, LSD_STEPS)       # [1, ldim]
        latents_all[step] = latent[0].numpy()

        increment_steps(flm, model_state, increment=1)
        backbone_input = latent[:, None, :]                          # feed back

    np.save(OUT / "lm_tokens.npy", tokens[0].numpy().astype(np.int32))
    np.save(OUT / "lm_noise.npy", noise_all)
    np.save(OUT / "lm_latents.npy", latents_all)
    np.save(OUT / "lm_eos.npy", eos_all)
    np.savez(OUT / "lm_meta.npz",
             n_txt=np.int32(n_txt), n_steps=np.int32(N_STEPS),
             ldim=np.int32(ldim), temp=np.float32(TEMP),
             eos_threshold=np.float32(EOS_THRESHOLD),
             emb_mean=flm.emb_mean.numpy().astype(np.float32),
             emb_std=flm.emb_std.numpy().astype(np.float32))
    n_eos = int((eos_all > EOS_THRESHOLD).sum())
    print(f"lm fixture: text={TEXT!r} n_txt={n_txt} n_steps={N_STEPS} "
          f"eos>{EOS_THRESHOLD}: {n_eos} steps")
    print(f"  latents range [{latents_all.min():.3f}, {latents_all.max():.3f}]")
    print(f"  eos logits range [{eos_all.min():.3f}, {eos_all.max():.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
