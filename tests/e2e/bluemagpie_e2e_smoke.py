"""BlueMagpie end-to-end integration parity: codec.cpp generation + decode vs PyTorch.

Drives the codec.cpp audio path for a real generated utterance:
  per step  -> LocDiT/CFM diffusion (codec_bluemagpie_cfm_eval) on the reference
               (mu, cond, z) that PyTorch's _inference produced
  end       -> accumulate latent patches -> codec_decode_quantized_representation
               (AudioVAE) -> 48 kHz audio
and compares per-step patches + the final waveform against the reference produced
by the real BlueMagpieModel.generate (fixed injected noise).

Reference fixture: tests/e2e/fixtures/bluemagpie/e2e_gen.npz (PyTorch, f32).
This validates the chained codec.cpp generation→decode pipeline; the LM-side
(Barbet + tslm_adapter + RALM + projections) is verified separately and supplies
the per-step (mu, cond).
"""
from __future__ import annotations
import ctypes, os, sys
from pathlib import Path
import numpy as np

from _common import REPO
import codec_cpp

# CODEC_BM_GGUF overrides the GGUF path (e.g. to smoke a quantized variant);
# defaults to the F16 model shipped under models/bluemagpie/.
GGUF = Path(os.environ.get("CODEC_BM_GGUF") or (REPO / "models" / "bluemagpie" / "bluemagpie.gguf"))
FIX  = REPO / "tests" / "e2e" / "fixtures" / "bluemagpie" / "e2e_gen.npz"
PATCH_CORR_MIN = 0.99
AUDIO_CORR_MIN = 0.99


def bind_cfm_eval():
    """codec_bluemagpie_cfm_eval is an extern "C" test hook, not public API."""
    fp = ctypes.POINTER(ctypes.c_float)
    fn = codec_cpp.lib().codec_bluemagpie_cfm_eval
    fn.argtypes = [ctypes.c_void_p, fp, fp, fp, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, fp, ctypes.c_int32]
    fn.restype = ctypes.c_int
    return fn


def main() -> int:
    for p in (GGUF, FIX):
        if not p.is_file():
            print(f"FAIL missing {p}"); return 1
    fp = ctypes.POINTER(ctypes.c_float)

    z = np.load(FIX)
    n = int(z["n"]); ts = int(z["ts"]); cfg = float(z["cfg"]); ref_audio = z["audio"].astype(np.float32)
    P = z["patch0"].shape[1]; D = z["patch0"].shape[0]   # (D, P)

    model = codec_cpp.Model(GGUF, use_gpu=False, n_threads=1)
    ctx = model.context(seed=0)
    cfm_eval = bind_cfm_eval()

    def cp(a): a = np.ascontiguousarray(a.astype(np.float32)); return a, a.ctypes.data_as(fp)

    latents = np.zeros((D, n * P), dtype=np.float32)   # channel-major [D, T]
    patch_corrs = []
    for i in range(n):
        _z, zp = cp(z[f"z{i}"]); _c, cpp = cp(z[f"cond{i}"]); _m, mp = cp(z[f"mu{i}"])
        out = np.zeros(D * P, dtype=np.float32)
        rc = cfm_eval(ctx.ptr, zp, cpp, mp, P, ts, cfg, out.ctypes.data_as(fp), out.size)
        if rc != 0:
            print(f"FAIL step {i}: debug_cfm rc={rc}"); return 1
        patch_g = out.reshape(P, D).T   # (D, P)
        ref_patch = z[f"patch{i}"]      # (D, P)
        patch_corrs.append(float(np.corrcoef(patch_g.reshape(-1), ref_patch.reshape(-1))[0, 1]))
        latents[:, i * P:(i + 1) * P] = patch_g

    try:
        audio, out_sr = ctx.decode_latent(latents, n_threads=1)
    except codec_cpp.CodecError as e:
        print(f"FAIL decode {e}"); return 1

    m = min(len(audio), len(ref_audio))
    audio_corr = float(np.corrcoef(audio[:m], ref_audio[:m])[0, 1])
    pc_min = min(patch_corrs)
    print(f"steps={n}  per-step patch corr: min={pc_min:.6f} mean={np.mean(patch_corrs):.6f}")
    print(f"audio: {len(audio)} samples @ {out_sr} Hz (ref {len(ref_audio)})  corr={audio_corr:.6f}")
    ok = (pc_min >= PATCH_CORR_MIN and audio_corr >= AUDIO_CORR_MIN and len(audio) == len(ref_audio))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
