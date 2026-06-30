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
import ctypes, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIB  = REPO / "build" / "libcodec.so"
GGUF = REPO / "models" / "bluemagpie" / "bluemagpie.gguf"
FIX  = REPO / "tests" / "e2e" / "fixtures" / "bluemagpie" / "e2e_gen.npz"
PATCH_CORR_MIN = 0.99
AUDIO_CORR_MIN = 0.99


class MP(ctypes.Structure): _fields_ = [("use_gpu", ctypes.c_bool), ("n_threads", ctypes.c_int32)]
class CP(ctypes.Structure): _fields_ = [("seed", ctypes.c_int32)]
class DP(ctypes.Structure): _fields_ = [("n_threads", ctypes.c_int32), ("n_q", ctypes.c_int32)]
class PCM(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)), ("n_samples", ctypes.c_int32),
                ("sample_rate", ctypes.c_int32), ("n_channels", ctypes.c_int32)]


def main() -> int:
    for p in (LIB, GGUF, FIX):
        if not p.is_file():
            print(f"FAIL missing {p}"); return 1
    lib = ctypes.CDLL(str(LIB))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.codec_model_load_from_file.argtypes = [ctypes.c_char_p, MP]; lib.codec_model_load_from_file.restype = ctypes.c_void_p
    lib.codec_init_from_model.argtypes = [ctypes.c_void_p, CP]; lib.codec_init_from_model.restype = ctypes.c_void_p
    lib.codec_bluemagpie_cfm_eval.argtypes = [ctypes.c_void_p, fp, fp, fp, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, fp, ctypes.c_int32]
    lib.codec_bluemagpie_cfm_eval.restype = ctypes.c_int
    lib.codec_decode_quantized_representation.argtypes = [ctypes.c_void_p, fp, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(PCM), DP]
    lib.codec_decode_quantized_representation.restype = ctypes.c_int
    lib.codec_get_last_error.argtypes = [ctypes.c_void_p]; lib.codec_get_last_error.restype = ctypes.c_char_p
    lib.codec_pcm_buffer_free.argtypes = [ctypes.POINTER(PCM)]

    z = np.load(FIX)
    n = int(z["n"]); ts = int(z["ts"]); cfg = float(z["cfg"]); ref_audio = z["audio"].astype(np.float32)
    P = z["patch0"].shape[1]; D = z["patch0"].shape[0]   # (D, P)

    model = lib.codec_model_load_from_file(str(GGUF).encode(), MP(False, 1))
    ctx = lib.codec_init_from_model(model, CP(0))

    def cp(a): a = np.ascontiguousarray(a.astype(np.float32)); return a, a.ctypes.data_as(fp)

    latents = np.zeros((D, n * P), dtype=np.float32)   # channel-major [D, T]
    patch_corrs = []
    for i in range(n):
        _, zp = cp(z[f"z{i}"]); _, cpp = cp(z[f"cond{i}"]); _, mp = cp(z[f"mu{i}"])
        out = np.zeros(D * P, dtype=np.float32)
        rc = lib.codec_bluemagpie_cfm_eval(ctx, zp, cpp, mp, P, ts, cfg, out.ctypes.data_as(fp), out.size)
        if rc != 0:
            print(f"FAIL step {i}: debug_cfm rc={rc}"); return 1
        patch_g = out.reshape(P, D).T   # (D, P)
        ref_patch = z[f"patch{i}"]      # (D, P)
        patch_corrs.append(float(np.corrcoef(patch_g.reshape(-1), ref_patch.reshape(-1))[0, 1]))
        latents[:, i * P:(i + 1) * P] = patch_g

    T = n * P
    buf = np.ascontiguousarray(latents.reshape(-1))   # channel-major [D*T]
    pcm = PCM()
    rc = lib.codec_decode_quantized_representation(ctx, buf.ctypes.data_as(fp), D, T, ctypes.byref(pcm), DP(1, 0))
    if rc != 0:
        print(f"FAIL decode rc={rc} err={lib.codec_get_last_error(ctx).decode()}"); return 1
    audio = np.ctypeslib.as_array(pcm.data, shape=(pcm.n_samples,)).copy()
    out_sr = pcm.sample_rate
    lib.codec_pcm_buffer_free(ctypes.byref(pcm))

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
