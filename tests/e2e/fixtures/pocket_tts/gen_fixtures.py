"""Generate Pocket-TTS Mimi-variant parity fixtures from the reference.

Builds the reference MimiModel from the english model.safetensors `mimi.*`
weights, then:
  - decode: fixed random latent [32, T] -> quantizer.output_proj ->
    decode_from_latent -> PCM.  Saved as (latent, pcm).
  - encode: fixed random audio [1, N] -> encode_to_latent -> latent [T, 32].
    Saved as (audio, mu channel-major [32, T]).

fp32 end to end.  Run:
  .venv/bin/python tests/e2e/fixtures/pocket_tts/gen_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / ".model-src" / "pocket-tts"))

from pocket_tts.models.mimi import MimiModel                      # noqa: E402
from pocket_tts.modules import mimi_transformer                    # noqa: E402
from pocket_tts.modules.dummy_quantizer import DummyQuantizer      # noqa: E402
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder # noqa: E402
from pocket_tts.modules.stateful_module import init_states, increment_steps  # noqa: E402

ST = REPO / "models" / "pocket_tts" / "model.safetensors"
OUT = Path(__file__).resolve().parent

MIMI_CFG = dict(
    sample_rate=24000, inner_dim=32, outer_dim=512, channels=1, frame_rate=12.5,
    seanet=dict(dimension=512, channels=1, n_filters=64, n_residual_layers=1,
                ratios=[6, 5, 4], kernel_size=7, residual_kernel_size=3,
                last_kernel_size=3, dilation_base=2, pad_mode="constant", compress=2),
    transformer=dict(d_model=512, num_heads=8, num_layers=2, layer_scale=0.01,
                     context=250, dim_feedforward=2048, input_dimension=512,
                     output_dimensions=(512,), max_period=10000.0),
    quantizer=dict(dimension=32, output_dimension=512),
)


def build_mimi() -> MimiModel:
    import safetensors
    sd = {}
    with safetensors.safe_open(str(ST), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("mimi."):
                sd[k[len("mimi."):]] = f.get_tensor(k)

    encoder = SEANetEncoder(**MIMI_CFG["seanet"])
    decoder = SEANetDecoder(**MIMI_CFG["seanet"])
    enc_tr = mimi_transformer.ProjectedTransformer(**MIMI_CFG["transformer"])
    dec_tr = mimi_transformer.ProjectedTransformer(**MIMI_CFG["transformer"])
    quant = DummyQuantizer(**MIMI_CFG["quantizer"])
    model = MimiModel(
        encoder, decoder, quant,
        channels=MIMI_CFG["channels"], sample_rate=MIMI_CFG["sample_rate"],
        frame_rate=MIMI_CFG["frame_rate"],
        encoder_frame_rate=MIMI_CFG["sample_rate"] / encoder.hop_length,
        inner_dim=MIMI_CFG["inner_dim"], outer_dim=MIMI_CFG["outer_dim"],
        encoder_transformer=enc_tr, decoder_transformer=dec_tr,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Register StatefulModule absolute names (TTSModel does this at load).
    from pocket_tts.modules.stateful_module import StatefulModule
    for name, mod in model.named_modules():
        if isinstance(mod, StatefulModule):
            mod._module_absolute_name = name
    missing = [m for m in missing if not m.endswith("_codebook") ]
    if missing:
        print("MISSING:", missing)
    if unexpected:
        print("UNEXPECTED:", unexpected)
    model.eval()
    return model


@torch.no_grad()
def main() -> int:
    torch.manual_seed(1234)
    model = build_mimi()

    # ---- decode fixture ----
    T = 20  # latent frames (~1.6 s)
    latent = torch.randn(1, T, 32, dtype=torch.float32)   # [B, T, ldim]
    mimi_steps = int(model.encoder_frame_rate / model.frame_rate)
    st = init_states(model, batch_size=1, sequence_length=T * mimi_steps)
    transposed = latent.transpose(-1, -2)                 # [B, 32, T]
    quantized = model.quantizer(transposed)               # [B, 512, T]
    audio = model.decode_from_latent(quantized, st)       # [B, 1, N]
    pcm = audio[0, 0].numpy().astype(np.float32)
    z = latent[0].transpose(0, 1).contiguous().numpy().astype(np.float32)  # [32, T] channel-major
    np.save(OUT / "decode_latent.npy", z)
    np.save(OUT / "decode_pcm.npy", pcm)
    print(f"decode fixture: latent {z.shape} pcm {pcm.shape}")

    # ---- encode fixture ----
    N = 24000 * 2  # 2 s
    from pocket_tts.modules.conv import pad_for_conv1d
    audio_in = torch.randn(1, 1, N, dtype=torch.float32) * 0.1
    mu = model.encode_to_latent(audio_in)                 # [B, 32, T] (channel-major)
    mu_cm = mu[0].contiguous().numpy().astype(np.float32)  # [32, T]
    # store the (possibly padded) audio actually consumed
    frame_size = model.frame_size
    padded = pad_for_conv1d(audio_in, frame_size, frame_size)
    np.savez(OUT / "encode.npz",
             audio=padded[0, 0].numpy().astype(np.float32),
             mu=mu_cm)
    print(f"encode fixture: audio {padded.shape[-1]} mu {mu_cm.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
