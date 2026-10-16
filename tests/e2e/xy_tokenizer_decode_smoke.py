#!/usr/bin/env python3
"""End-to-end smoke test for XY-Tokenizer (16 kHz in / 24 kHz out).

The cpp encoder threads `n_valid` through the attention layers (matches
HF's `attention_mask`-based SDPA bias) and uses GELU-erf to match HF's
F.gelu, giving bit-exact-modulo-FP-noise parity in the encoder body.
RVQ argmin is FP-sensitive at codebook ties; we accept ≥ 95 % codes
matching HF on the smoke clip and ≥ 0.95 corr on the cpp e2e
reconstruction vs HF.
"""

from __future__ import annotations

import sys

import numpy as np

from _common import REPO as REPO_ROOT, read_wav_mono_f32, run_cmd
import codec_cpp

CONVERT = REPO_ROOT / "scripts" / "convert-to-gguf.py"
XY_DIR = REPO_ROOT / ".model-src" / "xy_tokenizer"
GGUF = REPO_ROOT / "models" / "xy_tokenizer" / "xy_tokenizer.gguf"
N_THREADS = 4


def synth_test_input() -> tuple[np.ndarray, int]:
    """Linear-resample test.wav (24 kHz mono speech) to 16 kHz, take 1.5 s."""
    src, src_sr = read_wav_mono_f32(REPO_ROOT / "test.wav")
    target_sr = 16000
    n_target = int(round(len(src) * target_sr / src_sr))
    x_old = np.arange(len(src), dtype=np.float64)
    x_new = np.linspace(0.0, len(src) - 1, n_target, dtype=np.float64)
    resampled = np.interp(x_new, x_old, src).astype(np.float32)
    n = min(int(round(target_sr * 1.5)), len(resampled))
    return resampled[:n], target_sr


def hf_encode_decode(pcm: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Run HF reference: returns (codes (n_q, T), audio (samples,) at 24 kHz)."""
    # Re-import variant-specific modules (no relative-import support outside a package).
    for mn in list(sys.modules):
        if mn.startswith("configuration_xy_tokenizer") or mn.startswith("modeling_xy_tokenizer") or \
           mn.startswith("feature_extraction_xy_tokenizer"):
            del sys.modules[mn]
    if str(XY_DIR) not in sys.path:
        sys.path.insert(0, str(XY_DIR))

    import importlib.util
    for nm, fn in [
        ("feature_extraction_xy_tokenizer", "feature_extraction_xy_tokenizer.py"),
        ("configuration_xy_tokenizer", "configuration_xy_tokenizer.py"),
        ("modeling_xy_tokenizer", "modeling_xy_tokenizer.py"),
    ]:
        spec = importlib.util.spec_from_file_location(nm, XY_DIR / fn)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[nm] = mod
        spec.loader.exec_module(mod)

    import torch
    cfg_mod = sys.modules["configuration_xy_tokenizer"]
    fe_mod = sys.modules["feature_extraction_xy_tokenizer"]
    m_mod = sys.modules["modeling_xy_tokenizer"]
    cfg = cfg_mod.XYTokenizerConfig.from_pretrained(XY_DIR)
    fe = fe_mod.XYTokenizerFeatureExtractor(**cfg.params["feature_extractor_kwargs"])
    model = m_mod.XYTokenizerModel.from_pretrained(XY_DIR, config=cfg).eval()

    with torch.no_grad():
        feats = fe(torch.from_numpy(pcm).unsqueeze(0), sampling_rate=sr,
                   return_attention_mask=True, return_tensors="pt")
        enc = model.encode(feats)
        dec = model.decode(enc.audio_codes, overlap_seconds=10)
    codes = enc.audio_codes.cpu().numpy()[:, 0, :]   # (n_q=8, T)
    audio = dec["audio_values"][0].cpu().numpy().flatten()
    return codes, audio


def main() -> int:
    if not GGUF.is_file():
        if not XY_DIR.is_dir():
            print(f"SKIP: {XY_DIR} missing — run scripts/converters/xy_tokenizer.py first")
            return 0
        GGUF.parent.mkdir(parents=True, exist_ok=True)
        run_cmd([sys.executable, str(CONVERT),
                 "--checkpoint-path", str(XY_DIR),
                 "--model-type", "xy_tokenizer",
                 "--output", str(GGUF)])

    model = codec_cpp.Model(GGUF, use_gpu=False, n_threads=N_THREADS)
    ctx = model.context()

    pcm, sr = synth_test_input()
    # The library sees the same PCM16-quantized input the reference sees via WAV.
    pcm = (np.round(np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16).astype(np.float32) / 32768.0)

    # ---- 1. HF reference codes & audio --------------------------------
    hf_codes, hf_audio = hf_encode_decode(pcm, sr)
    print(f"  hf codes: {hf_codes.shape}, hf audio: {hf_audio.shape} rms={hf_audio.std():.4f}")
    n_q = int(hf_codes.shape[0])

    # ---- 2. cpp decode of HF codes (decoder parity) -------------------
    cpp_dec, cpp_dec_sr = ctx.decode(hf_codes.T, n_q=n_q, n_threads=N_THREADS)
    n = min(len(cpp_dec), len(hf_audio))
    dec_corr = float(np.corrcoef(hf_audio[:n], cpp_dec[:n])[0, 1])
    dec_max = float(np.abs(hf_audio[:n] - cpp_dec[:n]).max())
    print(f"  cpp_decode(HF codes) vs HF: corr={dec_corr:.6f} max_diff={dec_max:.4f}")
    if dec_corr < 0.95:
        print(f"FAIL: decoder corr {dec_corr:.4f} < 0.95")
        return 1

    # ---- 2b. cpp encoder: codes match HF -----------------------------
    cpp_codes = ctx.encode(pcm, sr, n_q=n_q, n_threads=N_THREADS).T  # (n_q, T)
    n = min(hf_codes.shape[1], cpp_codes.shape[1])
    match = (hf_codes[:, :n] == cpp_codes[:, :n]).sum()
    total = int(hf_codes.shape[0]) * int(n)
    match_frac = match / max(1, total)
    print(f"  cpp encode vs HF codes: {match}/{total} = {match_frac:.2%}")
    if match_frac < 0.95:
        print(f"FAIL: code match {match_frac:.2%} < 95%")
        return 1

    # ---- 3. cpp e2e produces sane output ------------------------------
    cpp_e2e, cpp_e2e_sr = ctx.decode(cpp_codes.T, n_q=n_q, n_threads=N_THREADS)
    if cpp_e2e_sr != 24000:
        print(f"FAIL: expected 24 kHz output, got {cpp_e2e_sr}")
        return 1
    if not np.all(np.isfinite(cpp_e2e)):
        print("FAIL: cpp e2e contains non-finite samples")
        return 1
    if cpp_e2e.std() < 1e-3:
        print(f"FAIL: cpp e2e is silent (rms={cpp_e2e.std():.4f})")
        return 1
    # cpp's audio length should be roughly proportional to input duration.
    expected_n = (len(pcm) // 1280) * 1920
    if abs(len(cpp_e2e) - expected_n) > 4 * 1920:
        print(f"FAIL: cpp e2e length {len(cpp_e2e)} != expected ~{expected_n}")
        return 1
    # Reconstruction parity vs HF e2e.
    n = min(len(cpp_e2e), len(hf_audio))
    e2e_corr = float(np.corrcoef(hf_audio[:n], cpp_e2e[:n])[0, 1])
    print(f"  cpp e2e: len={len(cpp_e2e)} rms={cpp_e2e.std():.4f} corr_vs_hf={e2e_corr:.4f}")
    if e2e_corr < 0.95:
        print(f"FAIL: e2e corr {e2e_corr:.4f} < 0.95")
        return 1

    # ---- 4. long decode crossing the chunk boundary ------------------
    # The decode transformer pos_emb tables are sized for a single
    # chunk_code_length (=375) window; longer sequences must be decoded in
    # overlapping chunks (HF `decode(overlap_seconds=10)`).  Regression:
    # decode a 500-code synthetic sequence (2 HF chunks) and check it
    # (a) completes without the ggml_view_2d assert and (b) matches HF.
    T_long = 500
    rng = np.random.default_rng(1234)
    long_codes = rng.integers(0, 1024, size=(n_q, T_long)).astype(np.int32)
    long_hf_audio = None
    try:
        import torch as _torch
        m_mod2 = sys.modules["modeling_xy_tokenizer"]
        cfg2 = sys.modules["configuration_xy_tokenizer"].XYTokenizerConfig.from_pretrained(XY_DIR)
        model2 = m_mod2.XYTokenizerModel.from_pretrained(XY_DIR, config=cfg2).eval()
        with _torch.no_grad():
            dec2 = model2.decode(_torch.from_numpy(long_codes.astype(np.int64))[:, None, :],
                                 overlap_seconds=10)
        long_hf_audio = dec2["audio_values"][0].cpu().numpy().flatten()
    except Exception as exc:  # pragma: no cover
        print(f"  (skipping HF long-decode parity: {exc})")

    long_cpp, long_sr = ctx.decode(long_codes.T, n_q=n_q, n_threads=N_THREADS)
    expected_long = T_long * 1920
    print(f"  long decode: T={T_long} codes -> len={len(long_cpp)} (expected {expected_long})")
    if not np.all(np.isfinite(long_cpp)):
        print("FAIL: long decode contains non-finite samples")
        return 1
    if abs(len(long_cpp) - expected_long) > 1920:
        print(f"FAIL: long decode length {len(long_cpp)} != expected {expected_long}")
        return 1
    if long_hf_audio is not None:
        n = min(len(long_cpp), len(long_hf_audio))
        long_corr = float(np.corrcoef(long_hf_audio[:n], long_cpp[:n])[0, 1])
        print(f"  long decode corr_vs_hf={long_corr:.6f}")
        if long_corr < 0.95:
            print(f"FAIL: long decode corr {long_corr:.4f} < 0.95")
            return 1

    print("XY-Tokenizer smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
