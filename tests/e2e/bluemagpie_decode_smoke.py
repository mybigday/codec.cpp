"""BlueMagpie AudioVAE continuous-latent decode parity smoke test.

Drives the public C API (`codec_decode_quantized_representation`) on the
converted AudioVAE GGUF and compares the 48 kHz waveform against a PyTorch
`AudioVAE.decode` reference computed on the same fixed latent.

Fixtures (generated from the real audiovae.pth, see tests/e2e/fixtures/bluemagpie):
  audiovae_z.npy      (latent_dim=64, n_frames)  — channel-major latent
  audiovae_audio.npy  (n_frames*1920,)           — reference 48 kHz PCM

Run:  .venv/bin/python tests/e2e/bluemagpie_decode_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

from _common import REPO
import codec_cpp

GGUF = REPO / "models" / "bluemagpie" / "bluemagpie.gguf"
FIX = REPO / "tests" / "e2e" / "fixtures" / "bluemagpie"

CORR_MIN = 0.9999
MSE_MAX = 1e-6


def check_encode(ctx: codec_cpp.Context) -> bool:
    """AudioVAE encode parity (audio → latent mu) vs PyTorch fixture."""
    fx = FIX / "audiovae_encode.npz"
    if not fx.is_file():
        print("encode: SKIP (no fixture)")
        return True
    z = np.load(fx)
    audio = z["audio"].astype(np.float32)
    ref = z["mu"]  # (latent_dim, n_frames)
    try:
        out, _ = ctx.encode_latent(audio, 16000, n_threads=1)
    except codec_cpp.CodecError as e:
        print(f"encode: FAIL {e}")
        return False
    D, T = out.shape
    n = min(T, ref.shape[1])
    corr = float(np.corrcoef(out[:, :n].reshape(-1), ref[:, :n].reshape(-1))[0, 1])
    # latent magnitudes are large (±12) so judge on correlation, not abs error.
    ok = corr >= 0.9999
    print(f"encode: latent_dim={D} n_frames={T} corr={corr:.8f}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    for p in (GGUF, FIX / "audiovae_z.npy", FIX / "audiovae_audio.npy"):
        if not p.is_file():
            print(f"FAIL: missing {p}")
            return 1

    z = np.load(FIX / "audiovae_z.npy").astype(np.float32)        # (latent_dim, n_frames)
    ref = np.load(FIX / "audiovae_audio.npy").astype(np.float32)  # (n_frames*1920,)

    model = codec_cpp.Model(GGUF, use_gpu=False, n_threads=1)
    ctx = model.context(seed=0)
    try:
        out, out_sr = ctx.decode_latent(z, n_threads=1)
    except codec_cpp.CodecError as e:
        print(f"FAIL: decode {e}")
        return 1
    out_n = len(out)

    encode_ok = check_encode(ctx)
    ctx.close()
    model.close()

    n = min(len(out), len(ref))
    out, ref = out[:n], ref[:n]
    corr = float(np.corrcoef(out, ref)[0, 1])
    mse = float(np.mean((out - ref) ** 2))
    max_abs = float(np.max(np.abs(out - ref)))
    print(f"sr={out_sr} n_samples={out_n} (ref {len(ref)})")
    print(f"corr={corr:.8f}  mse={mse:.3e}  max_abs={max_abs:.3e}")

    decode_ok = corr >= CORR_MIN and mse <= MSE_MAX and len(out) == len(np.load(FIX / 'audiovae_audio.npy'))
    print(f"decode: {'PASS' if decode_ok else 'FAIL'}")
    ok = decode_ok and encode_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
