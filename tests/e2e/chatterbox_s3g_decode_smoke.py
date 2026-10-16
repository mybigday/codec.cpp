#!/usr/bin/env python3
"""End-to-end smoke test for repeated Chatterbox-S3G decode calls.

Loads the model with GPU offload and calls codec_decode twice on the same
context. This catches scheduler allocations retained across eval-graph releases,
which caused the second Metal decode to return silent PCM.

Also loads the same (flow-metadata-free) GGUF through
codec_model_load_from_file_ext with chatterbox_s3g.flow.* KV overrides, the
path published GGUFs use to pick a cheaper CFM schedule without re-conversion.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from _common import REPO, read_wav_mono_f32, run_cmd

import codec_cpp
from codec_cpp import CodecError

CONVERT = REPO / "scripts" / "convert-to-gguf.py"
CHATTERBOX_DIR = REPO / "models" / "chatterbox"
REF_WAV = REPO / "input_audio" / "10_2_24k.wav"


def load(gguf: Path, kv_overrides: dict | None = None) -> codec_cpp.Model:
    return codec_cpp.Model(gguf, use_gpu=True, n_threads=4, kv_overrides=kv_overrides)


def decode(ctx: codec_cpp.Context, tokens: np.ndarray, speaker=None) -> tuple[np.ndarray, int]:
    return ctx.decode(tokens.reshape(-1, 1), n_threads=4, speaker=speaker)


def check_speakers(model, ctx, tokens, implicit_pcm, spk_file: Path) -> str | None:
    """Builtin speaker lookup, explicit decode, save/load round trip. Returns a failure or None."""
    names = model.speaker_names
    print(f"  builtin speakers: {names}")
    if names != ["default"]:
        return f"expected one builtin speaker 'default', got {names}"
    spk = model.speaker()
    if spk.ptr != model.speaker("default").ptr:
        return "codec_get_speaker(NULL) did not return the default speaker"
    try:
        model.speaker("missing")
        return "codec_get_speaker returned a speaker for an unknown name"
    except KeyError:
        pass
    explicit, _ = decode(ctx, tokens, spk)
    if not np.array_equal(explicit, implicit_pcm):
        return "explicit default speaker decode differs from speaker=NULL decode"

    model.save_speaker(spk, spk_file, "saved_voice")
    with model.load_speaker(spk_file) as loaded:
        reloaded, _ = decode(ctx, tokens, loaded)
    if not np.array_equal(reloaded, implicit_pcm):
        return "decode with a saved+loaded speaker differs from the builtin speaker"
    print("  speaker save/load round trip is bit-exact")

    # Reference audio → speaker (S3 tokenizer + 24 kHz mel + CAMPPlus in-library).
    if not model.can_speaker_from_audio:
        return "converted GGUF has no speaker encoder"
    ref, sr = read_wav_mono_f32(REF_WAV)
    try:
        derived = ctx.speaker_from_audio(ref, sr)
    except CodecError as e:
        return f"codec_speaker_from_audio failed: {e}"
    derived_file = spk_file.with_name("derived.gguf")
    with derived:
        cloned, _ = decode(ctx, tokens, derived)
        model.save_speaker(derived, derived_file, "ref")
    from gguf import GGUFReader
    slots = {t.name.split(".")[-1]: np.array(t.data) for t in GGUFReader(str(derived_file)).tensors}
    n_tok, n_feat = slots["prompt_token"].size, slots["prompt_feat"].size // 80
    want_feat = int(ref.size / sr * 24000 + 1440 - 1920) // 480 + 1
    print(f"  speaker from {REF_WAV.name}: {n_tok} prompt tokens, {n_feat} mel frames, "
          f"|embedding|={np.linalg.norm(slots['embedding']):.3f}")
    if n_feat != want_feat or n_tok != n_feat // 2:
        return f"derived speaker shapes off: tokens={n_tok} frames={n_feat} (want {want_feat})"
    if not np.all(np.isfinite(cloned)) or np.array_equal(cloned, implicit_pcm):
        return "decode with the derived speaker is non-finite or identical to the builtin voice"

    # Multi-speaker file: list it and decode with speakers it owns.
    multi_file = spk_file.with_name("multi.gguf")
    merge_speaker_files(multi_file, "ref", [("saved_voice", spk_file), ("ref", derived_file)])
    with model.open_speaker_file(multi_file) as fh:
        names = fh.speaker_names
        print(f"  speaker file lists {names}")
        if sorted(names) != ["ref", "saved_voice"]:
            return f"speaker file listing off: {names}"
        if fh.speaker().ptr != fh.speaker("ref").ptr:
            return "codec_speaker_file_get_speaker(NULL) did not return the file's default"
        try:
            fh.speaker("missing")
            return "codec_speaker_file_get_speaker returned a speaker for an unknown name"
        except KeyError:
            pass
        from_file_saved, _ = decode(ctx, tokens, fh.speaker("saved_voice"))
        from_file_ref, _ = decode(ctx, tokens, fh.speaker("ref"))
    if not np.array_equal(from_file_saved, implicit_pcm) or not np.array_equal(from_file_ref, cloned):
        return "decode with a speaker from the multi-speaker file differs from the original"
    try:
        model.load_speaker(multi_file, "saved_voice").close()
    except CodecError:
        return "codec_speaker_load by name from a multi-speaker file failed"
    print("  multi-speaker file: listing + per-name decode bit-exact")
    return None


def merge_speaker_files(out: Path, default: str, parts: list[tuple[str, Path]]) -> None:
    """Writes one speaker file holding the default speaker of each `parts` file under a new name."""
    from gguf import GGUFReader, GGUFWriter

    def kv_str(reader, key):
        return bytes(reader.fields[key].parts[-1]).decode()

    w = None
    for name, path in parts:
        r = GGUFReader(str(path))
        arch = kv_str(r, "general.architecture")
        if w is None:
            w = GGUFWriter(str(out), arch)
            w.add_string("general.type", "speaker")
            w.add_uint32(f"{arch}.spk.format_version", 1)
            w.add_string(f"{arch}.spk.encoder_hash", kv_str(r, f"{arch}.spk.encoder_hash"))
            w.add_string(f"{arch}.spk.default", default)
        for t in r.tensors:
            w.add_tensor(f"spk.{name}.{t.name.split('.', 2)[2]}", t.data.copy(), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    try:
        codec_cpp.library_path()
    except FileNotFoundError:
        print("SKIP: build/libcodec shared library not built")
        return 0
    if not (CHATTERBOX_DIR / "s3gen.safetensors").is_file() or not (CHATTERBOX_DIR / "conds.pt").is_file():
        print(f"SKIP: {CHATTERBOX_DIR} not present")
        return 0

    with tempfile.TemporaryDirectory(prefix="codec-s3g-decode-") as td:
        td = Path(td)
        gguf = td / "s3g.gguf"
        run_cmd([sys.executable, CONVERT, "--checkpoint-path", CHATTERBOX_DIR, "--model-type", "chatterbox_s3g",
                 "--output", gguf])

        tokens = np.ascontiguousarray([
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500, 1600,
        ], dtype=np.int32)

        spk_file = td / "speaker.gguf"
        with load(gguf) as model, model.context(seed=0) as ctx:
            results = [decode(ctx, tokens) for _ in range(2)]
            speaker_fail = check_speakers(model, ctx, tokens, results[0][0], spk_file)
        if speaker_fail:
            print(f"FAIL: {speaker_fail}")
            return 1

        # A model whose speaker encoder differs must refuse the speaker file.
        with load(gguf, {"chatterbox_s3g.spk.encoder_hash": "other-encoder"}) as model:
            try:
                model.load_speaker(spk_file).close()
                print("FAIL: speaker file accepted by a model with a different encoder hash")
                return 1
            except CodecError:
                pass
        print("  speaker file rejected by mismatched encoder hash (expected)")

        # Bad schedules must be rejected at load, not at decode.
        try:
            load(gguf, {"chatterbox_s3g.flow.n_timesteps": 0}).close()
            print("FAIL: n_timesteps=0 override was accepted")
            return 1
        except CodecError:
            pass

        fast = {"chatterbox_s3g.flow.n_timesteps": 2, "chatterbox_s3g.flow.cfg_rate": 0.0}
        with load(gguf, fast) as model, model.context(seed=0) as ctx:
            fast_pcm, _ = decode(ctx, tokens)

    for index, (pcm, sample_rate) in enumerate(results, start=1):
        rms = float(np.sqrt(np.mean(pcm**2)))
        peak = float(np.max(np.abs(pcm)))
        print(
            f"  run {index}: decoded {pcm.size} samples at {sample_rate} Hz; "
            f"rms={rms:.6f}, peak={peak:.6f}"
        )
        if not np.all(np.isfinite(pcm)):
            print(f"FAIL: run {index} contains non-finite samples")
            return 1
        if peak > 1.0:
            print(f"FAIL: run {index} clip exceeded ±1.0")
            return 1
        if rms < 1e-4:
            print(f"FAIL: run {index} is silent")
            return 1

    if results[0][1] != results[1][1] or results[0][0].shape != results[1][0].shape:
        print("FAIL: repeated same-context decodes have different output shapes")
        return 1
    if not np.allclose(results[0][0], results[1][0], rtol=1e-5, atol=1e-6):
        print("FAIL: repeated same-context decodes differ")
        return 1
    fast_rms = float(np.sqrt(np.mean(fast_pcm**2)))
    print(f"  override n_timesteps=2 cfg_rate=0: {fast_pcm.size} samples; rms={fast_rms:.6f}")
    if fast_pcm.shape != results[0][0].shape or not np.all(np.isfinite(fast_pcm)) or fast_rms < 1e-4:
        print("FAIL: overridden flow schedule produced bad PCM")
        return 1
    if np.allclose(fast_pcm, results[0][0], rtol=1e-5, atol=1e-6):
        print("FAIL: flow KV overrides had no effect")
        return 1
    print("chatterbox S3G decode end-to-end smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
