"""BlueMagpie continuous-latent orchestration parity: prefill + stop head.

Verifies the codec.cpp continuous adaptor's RALM prefix prefill, primed first
step, and min_len stop guard against the real PyTorch reference.  The reference
_inference was run once (tests/e2e/fixtures/bluemagpie/gen_stop_fixture.py) on a
short zero-shot zh-TW prompt with recorded CFM noise; the fixture stores the
per-position barbet prefill hiddens, per-iteration barbet step hiddens, injected
noise, reference patches, raw stop_head logits, and the stop iteration index.

This test drives codec.cpp via ctypes teacher-forced:
  codec_lm_text_prefill(prefill hiddens)
  iteration 0            : step_generate(h_in unused, noise0)       [primed]
  iteration i>=1         : step_generate(h_in = step_hiddens[i-1], noise_i)
and compares:
  * each generated patch vs the reference (corr >= PATCH_CORR_MIN)
  * the stop-flag SEQUENCE: 0 for all iterations before the reference stop index
    and 1 exactly at it, honouring the same min_len (`i > min_len`) guard.

Fixture: tests/e2e/fixtures/bluemagpie/stop_gen.npz (PyTorch f32).
"""
from __future__ import annotations
import ctypes, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIB  = REPO / "build" / "libcodec.so"
GGUF = Path(os.environ.get("CODEC_BM_GGUF") or (REPO / "models" / "bluemagpie" / "bluemagpie.gguf"))
FIX  = REPO / "tests" / "e2e" / "fixtures" / "bluemagpie" / "stop_gen.npz"

# Per-patch corr threshold.  With teacher-forced trajectory + recorded noise the
# orchestration is exact; residual per-patch drift is codec.cpp quantized weights
# vs the PyTorch f32 reference.  F16: one patch ~0.9986, rest >= 0.9997 → 0.998.
# Q8_0 quantization noise is larger (min ~0.992) → looser bar when smoking Q8.
# Either way the assertion is very strong: uncorrelated patches sit near 0.
PATCH_CORR_MIN = 0.990 if "q8" in GGUF.name.lower() else 0.998


class MP(ctypes.Structure): _fields_ = [("use_gpu", ctypes.c_bool), ("n_threads", ctypes.c_int32)]
class CP(ctypes.Structure): _fields_ = [("seed", ctypes.c_int32)]


def _load_lib():
    lib = ctypes.CDLL(str(LIB))
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int32)
    lib.codec_model_load_from_file.argtypes = [ctypes.c_char_p, MP]
    lib.codec_model_load_from_file.restype  = ctypes.c_void_p
    lib.codec_model_free.argtypes = [ctypes.c_void_p]
    lib.codec_lm_create.argtypes = [ctypes.c_void_p]; lib.codec_lm_create.restype = ctypes.c_void_p
    lib.codec_lm_free.argtypes = [ctypes.c_void_p]
    lib.codec_lm_get_create_error.restype = ctypes.c_char_p
    lib.codec_lm_state_new.argtypes = [ctypes.c_void_p]; lib.codec_lm_state_new.restype = ctypes.c_void_p
    lib.codec_lm_state_free.argtypes = [ctypes.c_void_p]
    lib.codec_lm_state_reset.argtypes = [ctypes.c_void_p]
    lib.codec_lm_state_get_last_error.argtypes = [ctypes.c_void_p]
    lib.codec_lm_state_get_last_error.restype = ctypes.c_char_p
    lib.codec_lm_text_prefill.argtypes = [ctypes.c_void_p, fp, ctypes.c_int32, ctypes.c_int32]
    lib.codec_lm_text_prefill.restype = ctypes.c_int
    lib.codec_lm_set_continuous_min_len.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.codec_lm_set_continuous_min_len.restype = ctypes.c_int
    lib.codec_lm_set_teacher_patch.argtypes = [ctypes.c_void_p, fp, ctypes.c_int32]
    lib.codec_lm_set_teacher_patch.restype = ctypes.c_int
    lib.codec_lm_step_generate.argtypes = [ctypes.c_void_p, fp, ctypes.c_float, ctypes.c_int32, fp, fp, ip]
    lib.codec_lm_step_generate.restype = ctypes.c_int
    return lib


def run(lib, gguf, z, apply_min_len_override=None):
    """Drive one teacher-forced generation; returns (patch_corrs, stop_flags)."""
    fp = ctypes.POINTER(ctypes.c_float)
    n_steps   = int(z["n_steps"]); ts = int(z["ts"]); cfg = float(z["cfg"])
    D = int(z["D"]); P = int(z["P"]); h_barbet = int(z["h_barbet"])
    prefill = np.ascontiguousarray(z["prefill_hiddens"].astype(np.float32))  # [T, h_barbet]
    T_text = prefill.shape[0]
    step_hiddens = z["step_hiddens"].astype(np.float32)                      # [n_steps-1, h_barbet]

    model = lib.codec_model_load_from_file(str(gguf).encode(), MP(False, 1))
    if not model: raise RuntimeError("model load failed")
    lm = lib.codec_lm_create(model)
    if not lm:
        raise RuntimeError("codec_lm_create NULL: " + lib.codec_lm_get_create_error().decode())
    st = lib.codec_lm_state_new(lm)

    if apply_min_len_override is not None:
        lib.codec_lm_set_continuous_min_len(st, apply_min_len_override)

    # RALM prefix prefill.
    rc = lib.codec_lm_text_prefill(st, prefill.ctypes.data_as(fp), T_text, h_barbet)
    if rc != 0:
        raise RuntimeError("text_prefill rc=%d: %s" % (rc, lib.codec_lm_state_get_last_error(st).decode()))

    patch_corrs, stop_flags = [], []
    for i in range(n_steps):
        # codec.cpp bm.cfm.z is ggml (ne0=D, ne1=P): channel-fastest storage
        # (d + p*D).  numpy (D,P) is row-major (d*P + p), so flatten Fortran-order
        # to match, exactly as the reference z was recorded (b, D, P).
        noise = np.ascontiguousarray(z[f"noise{i}"].astype(np.float32).reshape(D, P).flatten(order="F"))
        if i == 0:
            h_in = np.zeros(h_barbet, dtype=np.float32)   # unused on the primed step
        else:
            h_in = np.ascontiguousarray(step_hiddens[i - 1])
        # Teacher-force the trajectory: arm this step with the reference patch
        # (codec ne0=D, ne1=P layout → Fortran flatten of the (D,P) ref patch)
        # so the cond + LocEnc feedback track the reference exactly; every
        # emitted patch then stays comparable at high corr.
        ref_patch_flat = np.ascontiguousarray(z[f"patch{i}"].astype(np.float32).reshape(D, P).flatten(order="F"))
        lib.codec_lm_set_teacher_patch(st, ref_patch_flat.ctypes.data_as(fp), D * P)

        out = np.zeros(D * P, dtype=np.float32)
        stop = ctypes.c_int32(-1)
        rc = lib.codec_lm_step_generate(st, h_in.ctypes.data_as(fp), cfg, ts,
                                        noise.ctypes.data_as(fp),
                                        out.ctypes.data_as(fp), ctypes.byref(stop))
        if rc != 0:
            raise RuntimeError("step_generate rc=%d step=%d: %s" %
                               (rc, i, lib.codec_lm_state_get_last_error(st).decode()))
        patch_g = out.reshape(P, D).T   # (D, P)
        ref_patch = z[f"patch{i}"]      # (D, P)
        patch_corrs.append(float(np.corrcoef(patch_g.reshape(-1), ref_patch.reshape(-1))[0, 1]))
        stop_flags.append(int(stop.value))

    lib.codec_lm_state_free(st); lib.codec_lm_free(lm); lib.codec_model_free(model)
    return patch_corrs, stop_flags


def main() -> int:
    for p in (LIB, GGUF, FIX):
        if not p.is_file():
            print(f"FAIL missing {p}"); return 1
    lib = _load_lib()
    z = np.load(FIX)
    n_steps    = int(z["n_steps"]); min_len = int(z["min_len"]); ref_stop = int(z["stop_index"])

    # Expected stop-flag sequence under our min_len guard: 0 everywhere except at
    # ref_stop (if the reference stopped before max_len).  Guard: stop honoured
    # only for patch index > min_len.
    patch_corrs, stop_flags = run(lib, GGUF, z)
    pc_min = min(patch_corrs)

    # Our first stop==1 index.
    our_stop = next((i for i, s in enumerate(stop_flags) if s == 1), n_steps)

    print(f"steps={n_steps} min_len={min_len}  ref_stop={ref_stop} our_stop={our_stop}")
    print(f"per-step patch corr: min={pc_min:.6f} mean={np.mean(patch_corrs):.6f}")
    print(f"stop flags: {stop_flags}")

    ok = True
    if pc_min < PATCH_CORR_MIN:
        print(f"  patch corr below {PATCH_CORR_MIN}"); ok = False
    # min_len guard must suppress any early stop.
    for i in range(min(min_len + 1, n_steps)):
        if stop_flags[i] != 0:
            print(f"  min_len guard violated: stop at patch {i} (<= {min_len})"); ok = False
    if our_stop != ref_stop:
        print(f"  stop timing mismatch: ours={our_stop} ref={ref_stop}"); ok = False

    # Determinism: a second identical run must be bit-identical.
    patch_corrs2, stop_flags2 = run(lib, GGUF, z)
    if stop_flags2 != stop_flags or patch_corrs2 != patch_corrs:
        # patch corr is derived from identical outputs, so any drift means
        # non-determinism in the generated patches.
        print("  non-deterministic: two runs differ"); ok = False
    else:
        print("determinism: two runs identical  OK")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
