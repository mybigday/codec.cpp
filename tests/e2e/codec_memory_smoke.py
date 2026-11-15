#!/usr/bin/env python3
"""Codec memory (docs/proposals/codec_memory.md) through the public C API.

For every causal side present under models/:
  * streamed (random hop-aligned chunks, down to one frame, one chunk longer
    than the attention window) vs batch on a clip longer than the window;
  * state get / set into a fresh context continues the stream bit-exactly;
  * a blob of another model / side, or a corrupted one, is rejected and the
    cache is unchanged;
  * clear == a fresh context.
Plus: a non-causal side's memory is a no-op of size 0, codec_decode_batch
neither reads nor updates memory, and the look_ahead GGUF key (via a load-time
override; the published GGUFs predate the keys and load with the per-arch
defaults) decides causality.

Tolerances. Chunked compute reorders float reductions (attention rows of
cache + chunk keys), so streamed output equals batch to rounding, not bit for
bit; these models amplify rounding to ~1e-4..1e-3 (the batch graph itself
differs that much between clip lengths). Decode: relative RMS <= 1e-3.
Encode: codes are argmaxes at near-ties in the deep RVQ levels; <= 3 % flips.
Convolution-only models (NeMo) are bit-exact.

Usage: codec_memory_smoke.py [model ...]   (default: every model found)
"""

from __future__ import annotations

import sys

import numpy as np

from _common import REPO, must, read_wav_mono_f32
import codec_cpp

MODELS = {
    "qwen3_tts_tokenizer": REPO / "models" / "qwen3_tts_tokenizer" / "qwen3_tts_tokenizer.gguf",
    "mimi": REPO / "models" / "mimi" / "mimi.gguf",
    "moss_audio_nano": REPO / "models" / "moss_audio_nano" / "moss_audio_nano.gguf",
    "nemo_nano_codec": REPO / "models" / "nemo_nano_codec" / "nemo_nano_codec.gguf",
}
NON_CAUSAL = REPO / "models" / "dac" / "dac.gguf"
SECONDS = 24.0          # longer than every window but MOSS's (whose windows exceed any test clip)
DEC_REL_RMS = 1e-3
ENC_FLIPS = 0.03


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def clip(model: codec_cpp.Model, stereo: bool) -> np.ndarray:
    x, sr = read_wav_mono_f32(REPO / "test.wav")
    x = resample(x, sr, model.sample_rate)
    reps = int(np.ceil(SECONDS * model.sample_rate / len(x)))
    x = np.tile(x, reps)[: int(SECONDS * model.sample_rate)]
    x = x[: len(x) // model.hop_size * model.hop_size]
    return np.stack([x, x], 1) if stereo else x


def plan(n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    sizes, out, i = [1, 1, min(100, n // 3), 3, 1], [], 0
    while i < n:
        k = sizes.pop(0) if sizes else int(rng.integers(1, 12))
        k = min(k, n - i)
        out.append(k)
        i += k
    return out


def pcm(res) -> np.ndarray:
    return np.asarray(res[0]).reshape(-1)


def stream_decode(ctx: codec_cpp.Context, codes: np.ndarray, sizes: list[int]) -> np.ndarray:
    out, i = [], 0
    for k in sizes:
        out.append(pcm(ctx.decode(codes[i:i + k])))
        i += k
    return np.concatenate(out)


def stream_encode(ctx: codec_cpp.Context, x: np.ndarray, hop: int, sr: int, sizes: list[int]) -> np.ndarray:
    out, i = [], 0
    for k in sizes:
        out.append(ctx.encode(x[i * hop:(i + k) * hop], sr))
        i += k
    return np.concatenate(out)


def check_state(name: str, m: codec_cpp.Model, side: str, run, other_blob: bytes) -> None:
    """run(ctx, lo, hi) feeds frames [lo, hi) and returns the output."""
    kw = {f"{side}_memory": True}
    T = run(None, 0, 0)   # total frames
    half = T // 2
    a = codec_cpp.Context(m, **kw)
    mem_a = getattr(a, f"{side}_memory")
    run(a, 0, half)
    blob = mem_a.get_state()
    must(len(blob) == mem_a.state_size() > 0, f"{name} {side}: state size")
    rest_a = run(a, half, T)

    b = codec_cpp.Context(m, **kw)
    mem_b = getattr(b, f"{side}_memory")
    must(mem_b.set_state(blob) == len(blob), f"{name} {side}: set_state rejected its own blob")
    rest_b = run(b, half, T)
    must(np.array_equal(rest_a, rest_b), f"{name} {side}: restored stream differs from the uninterrupted one")

    # rejected blobs leave the cache unchanged
    before = mem_b.get_state()
    bad = bytearray(blob)
    bad[20] ^= 0xFF   # fingerprint byte
    must(mem_b.set_state(bytes(bad)) == 0, f"{name} {side}: corrupted blob accepted")
    if other_blob:
        must(mem_b.set_state(other_blob) == 0, f"{name} {side}: foreign blob accepted")
    must(mem_b.set_state(blob[: len(blob) // 2]) == 0, f"{name} {side}: truncated blob accepted")
    must(mem_b.get_state() == before, f"{name} {side}: a rejected blob changed the cache")

    # clear == fresh context
    mem_b.clear()
    again = run(b, 0, half)
    fresh = run(codec_cpp.Context(m, **kw), 0, half)
    must(np.array_equal(again, fresh), f"{name} {side}: clear differs from a fresh context")
    print(f"  {side}: state {len(blob)} B, restore / reject / clear ok")


BLOBS: dict[str, bytes] = {}


def test_model(name: str) -> None:
    path = MODELS[name]
    if not path.is_file():
        print(f"[skip] {name}: {path} not found")
        return
    m = codec_cpp.Model(path)
    stereo = name == "moss_audio_nano"
    print(f"{name}: encoder causal {m.encoder_is_causal}, decoder causal {m.decoder_is_causal}")
    x = clip(m, stereo)
    sr, hop = m.sample_rate, m.hop_size
    c = codec_cpp.Context(m)
    codes = c.encode(x, sr) if m.has_encoder else None
    T = len(codes)
    sizes = plan(T)

    if m.decoder_is_causal:
        ref = pcm(c.decode(codes))
        st = stream_decode(codec_cpp.Context(m, decoder_memory=True), codes, sizes)
        must(st.shape == ref.shape, f"{name} decode: length {st.shape} vs {ref.shape}")
        d = st - ref
        rel = float(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(ref ** 2)))
        print(f"  decode stream vs batch ({T} frames, {len(sizes)} chunks): max|d| {np.abs(d).max():.2e}, rel rms {rel:.2e}")
        must(rel <= DEC_REL_RMS, f"{name} decode: rel rms {rel:.2e} > {DEC_REL_RMS}")

        def run_dec(ctx, lo, hi):
            if ctx is None:
                return T
            return stream_decode(ctx, codes[lo:hi], plan(hi - lo, seed=lo))
        check_state(name, m, "decoder", run_dec, BLOBS.get("decoder", b""))
        BLOBS["decoder"] = codec_cpp.Context(m, decoder_memory=True).decoder_memory.get_state()

    if m.encoder_is_causal:
        sc = stream_encode(codec_cpp.Context(m, encoder_memory=True), x, hop, sr, sizes)
        must(sc.shape == codes.shape, f"{name} encode: shape {sc.shape} vs {codes.shape}")
        flips = float((sc != codes).mean())
        print(f"  encode stream vs batch: {flips * 100:.2f} % codes differ")
        must(flips <= ENC_FLIPS, f"{name} encode: {flips * 100:.2f} % codes differ")

        def run_enc(ctx, lo, hi):
            if ctx is None:
                return T
            return stream_encode(ctx, x[lo * hop:], hop, sr, plan(hi - lo, seed=lo))
        check_state(name, m, "encoder", run_enc, BLOBS.get("encoder", b""))
        BLOBS["encoder"] = codec_cpp.Context(m, encoder_memory=True).encoder_memory.get_state()

    if m.decoder_is_causal:
        # codec_decode_batch neither reads nor updates the decoder memory
        ctx = codec_cpp.Context(m, decoder_memory=True)
        ctx.decode(codes[:7])
        before = ctx.decoder_memory.get_state()
        out = ctx.decode_batch([codes[:5], codes[5:9]])
        must(ctx.decoder_memory.get_state() == before, f"{name}: codec_decode_batch changed the memory")
        must(np.array_equal(np.asarray(out[0][0]).reshape(-1), pcm(c.decode(codes[:5]))),
             f"{name}: codec_decode_batch on a memory context is not stateless")
        print("  decode_batch bypasses memory: ok")


def test_non_causal() -> None:
    if not NON_CAUSAL.is_file():
        print(f"[skip] non-causal: {NON_CAUSAL} not found")
        return
    m = codec_cpp.Model(NON_CAUSAL)
    must(not m.decoder_is_causal, "DAC decoder reported causal")
    x = clip(m, False)[: m.sample_rate * 2]
    codes = codec_cpp.Context(m).encode(x, m.sample_rate)
    ctx = codec_cpp.Context(m, decoder_memory=True)
    mem = ctx.decoder_memory
    must(mem is not None and mem.state_size() == 0, "non-causal memory must have size 0")
    must(mem.get_state() == b"" and mem.set_state(b"") == 0, "empty get / set")
    mem.clear()
    a = pcm(ctx.decode(codes))
    b = pcm(codec_cpp.Context(m).decode(codes))
    must(np.array_equal(a, b), "memory on a non-causal decoder changed the output")
    print("dac: non-causal decoder memory is a size-0 no-op: ok")


def test_look_keys() -> None:
    path = MODELS["nemo_nano_codec"]
    if not path.is_file():
        return
    must(codec_cpp.Model(path).decoder_is_causal, "NeMo default (no key) should be causal")
    m = codec_cpp.Model(path, kv_overrides={"codec.decoder.look_ahead": 3})
    must(not m.decoder_is_causal, "codec.decoder.look_ahead = 3 should make the decoder non-causal")
    must(codec_cpp.Context(m, decoder_memory=True).decoder_memory.state_size() == 0,
         "non-causal by metadata: memory size must be 0")
    print("look_ahead GGUF key (override) decides causality: ok")


def main() -> int:
    names = sys.argv[1:] or list(MODELS)
    for n in names:
        test_model(n)
    test_non_causal()
    test_look_keys()
    print("codec_memory_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
