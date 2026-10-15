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

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERT = REPO_ROOT / "scripts" / "convert-to-gguf.py"
CHATTERBOX_DIR = REPO_ROOT / "models" / "chatterbox"
REF_WAV = REPO_ROOT / "input_audio" / "10_2_24k.wav"


class codec_model_params(ctypes.Structure):
    _fields_ = [("use_gpu", ctypes.c_bool), ("n_threads", ctypes.c_int32)]


class codec_context_params(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int32)]


CODEC_KV_OVERRIDE_TYPE_INT = 0
CODEC_KV_OVERRIDE_TYPE_FLOAT = 1
CODEC_KV_OVERRIDE_TYPE_STR = 3


class _kv_override_value(ctypes.Union):
    _fields_ = [
        ("val_i64", ctypes.c_int64),
        ("val_f64", ctypes.c_double),
        ("val_bool", ctypes.c_bool),
        ("val_str", ctypes.c_char * 128),
    ]


class codec_model_kv_override(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("tag", ctypes.c_int), ("key", ctypes.c_char * 128), ("value", _kv_override_value)]


def kv_overrides(items: list[tuple[str, int, float | str]]) -> ctypes.Array:
    arr = (codec_model_kv_override * (len(items) + 1))()
    for slot, (key, tag, value) in zip(arr, items):
        slot.tag = tag
        slot.key = key.encode()
        if tag == CODEC_KV_OVERRIDE_TYPE_INT:
            slot.val_i64 = int(value)
        elif tag == CODEC_KV_OVERRIDE_TYPE_STR:
            slot.val_str = str(value).encode()
        else:
            slot.val_f64 = float(value)
    return arr


class codec_decode_params(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int32), ("n_q", ctypes.c_int32), ("speaker", ctypes.c_void_p)]


class codec_token_buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_int32),
        ("n_frames", ctypes.c_int32),
        ("n_q", ctypes.c_int32),
        ("codebook_size", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("hop_size", ctypes.c_int32),
    ]


class codec_audio(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("n_channels", ctypes.c_int32),
        ("pcm_type", ctypes.c_int),
    ]


class codec_pcm_buffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
        ("n_channels", ctypes.c_int32),
    ]


def run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.decode(errors='replace')[:4096]}")
    return proc.stdout


def find_libcodec() -> Path | None:
    for name in ("libcodec.dylib", "libcodec.so"):
        path = REPO_ROOT / "build" / name
        if path.is_file():
            return path
    return None


def bind(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))
    lib.codec_model_load_from_file.argtypes = [ctypes.c_char_p, codec_model_params]
    lib.codec_model_load_from_file.restype = ctypes.c_void_p
    lib.codec_model_load_from_file_ext.argtypes = [
        ctypes.c_char_p, codec_model_params, ctypes.POINTER(codec_model_kv_override)
    ]
    lib.codec_model_load_from_file_ext.restype = ctypes.c_void_p
    lib.codec_model_free.argtypes = [ctypes.c_void_p]
    lib.codec_init_from_model.argtypes = [ctypes.c_void_p, codec_context_params]
    lib.codec_init_from_model.restype = ctypes.c_void_p
    lib.codec_free.argtypes = [ctypes.c_void_p]
    lib.codec_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(codec_token_buffer),
        ctypes.POINTER(codec_pcm_buffer),
        codec_decode_params,
    ]
    lib.codec_decode.restype = ctypes.c_int
    lib.codec_pcm_buffer_free.argtypes = [ctypes.POINTER(codec_pcm_buffer)]
    lib.codec_get_last_error.argtypes = [ctypes.c_void_p]
    lib.codec_get_last_error.restype = ctypes.c_char_p
    lib.codec_model_n_speakers.argtypes = [ctypes.c_void_p]
    lib.codec_model_n_speakers.restype = ctypes.c_int32
    lib.codec_model_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.codec_model_speaker_name.restype = ctypes.c_char_p
    lib.codec_get_speaker.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.codec_get_speaker.restype = ctypes.c_void_p
    lib.codec_speaker_save.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.codec_speaker_save.restype = ctypes.c_int
    lib.codec_speaker_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.codec_speaker_load.restype = ctypes.c_int
    lib.codec_speaker_free.argtypes = [ctypes.c_void_p]
    lib.codec_speaker_file_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.codec_speaker_file_open.restype = ctypes.c_int
    lib.codec_speaker_file_n_speakers.argtypes = [ctypes.c_void_p]
    lib.codec_speaker_file_n_speakers.restype = ctypes.c_int32
    lib.codec_speaker_file_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.codec_speaker_file_speaker_name.restype = ctypes.c_char_p
    lib.codec_speaker_file_get_speaker.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.codec_speaker_file_get_speaker.restype = ctypes.c_void_p
    lib.codec_speaker_file_free.argtypes = [ctypes.c_void_p]
    lib.codec_model_can_speaker_from_audio.argtypes = [ctypes.c_void_p]
    lib.codec_model_can_speaker_from_audio.restype = ctypes.c_bool
    lib.codec_speaker_from_audio.argtypes = [ctypes.c_void_p, ctypes.POINTER(codec_audio), ctypes.POINTER(ctypes.c_void_p)]
    lib.codec_speaker_from_audio.restype = ctypes.c_int
    return lib


def decode(lib: ctypes.CDLL, ctx: int, tokens: np.ndarray, speaker: int | None = None) -> tuple[np.ndarray, int]:
    token_buf = codec_token_buffer(
        tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        tokens.size,
        tokens.size,
        1,
        0,
        0,
        0,
    )
    pcm_buf = codec_pcm_buffer()
    status = lib.codec_decode(
        ctx, ctypes.byref(token_buf), ctypes.byref(pcm_buf), codec_decode_params(4, 0, speaker)
    )
    if status != 0:
        error = lib.codec_get_last_error(ctx).decode(errors="replace")
        raise RuntimeError(f"codec_decode failed ({status}): {error}")
    pcm = np.ctypeslib.as_array(pcm_buf.data, shape=(pcm_buf.n_samples,)).copy()
    sample_rate = pcm_buf.sample_rate
    lib.codec_pcm_buffer_free(ctypes.byref(pcm_buf))
    return pcm, sample_rate


def check_speakers(lib, model, ctx, tokens, implicit_pcm, spk_file: Path) -> str | None:
    """Builtin speaker lookup, explicit decode, save/load round trip. Returns a failure or None."""
    n = lib.codec_model_n_speakers(model)
    names = [lib.codec_model_speaker_name(model, i).decode() for i in range(n)]
    print(f"  builtin speakers: {names}")
    if names != ["default"]:
        return f"expected one builtin speaker 'default', got {names}"
    spk = lib.codec_get_speaker(model, None)
    if not spk or spk != lib.codec_get_speaker(model, b"default"):
        return "codec_get_speaker(NULL) did not return the default speaker"
    if lib.codec_get_speaker(model, b"missing"):
        return "codec_get_speaker returned a speaker for an unknown name"
    explicit, _ = decode(lib, ctx, tokens, spk)
    if not np.array_equal(explicit, implicit_pcm):
        return "explicit default speaker decode differs from speaker=NULL decode"

    if lib.codec_speaker_save(model, spk, b"saved_voice", str(spk_file).encode()) != 0:
        return "codec_speaker_save failed"
    loaded = ctypes.c_void_p()
    if lib.codec_speaker_load(model, str(spk_file).encode(), None, ctypes.byref(loaded)) != 0:
        return "codec_speaker_load failed"
    try:
        reloaded, _ = decode(lib, ctx, tokens, loaded.value)
    finally:
        lib.codec_speaker_free(loaded)
    if not np.array_equal(reloaded, implicit_pcm):
        return "decode with a saved+loaded speaker differs from the builtin speaker"
    print("  speaker save/load round trip is bit-exact")

    # Reference audio → speaker (S3 tokenizer + 24 kHz mel + CAMPPlus in-library).
    if not lib.codec_model_can_speaker_from_audio(model):
        return "converted GGUF has no speaker encoder"
    import wave
    with wave.open(str(REF_WAV), "rb") as w:
        sr = w.getframerate()
        ref = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    audio = codec_audio(ref.ctypes.data, ref.size, sr, 1, 0)
    derived = ctypes.c_void_p()
    if lib.codec_speaker_from_audio(ctx, ctypes.byref(audio), ctypes.byref(derived)) != 0:
        return "codec_speaker_from_audio failed: " + lib.codec_get_last_error(ctx).decode(errors="replace")
    try:
        cloned, _ = decode(lib, ctx, tokens, derived.value)
        derived_file = spk_file.with_name("derived.gguf")
        if lib.codec_speaker_save(model, derived, b"ref", str(derived_file).encode()) != 0:
            return "codec_speaker_save of a derived speaker failed"
    finally:
        lib.codec_speaker_free(derived)
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
    fh = ctypes.c_void_p()
    if lib.codec_speaker_file_open(model, str(multi_file).encode(), ctypes.byref(fh)) != 0:
        return "codec_speaker_file_open failed"
    try:
        names = [lib.codec_speaker_file_speaker_name(fh, i).decode()
                 for i in range(lib.codec_speaker_file_n_speakers(fh))]
        print(f"  speaker file lists {names}")
        if sorted(names) != ["ref", "saved_voice"]:
            return f"speaker file listing off: {names}"
        if lib.codec_speaker_file_get_speaker(fh, None) != lib.codec_speaker_file_get_speaker(fh, b"ref"):
            return "codec_speaker_file_get_speaker(NULL) did not return the file's default"
        if lib.codec_speaker_file_get_speaker(fh, b"missing"):
            return "codec_speaker_file_get_speaker returned a speaker for an unknown name"
        from_file_saved, _ = decode(lib, ctx, tokens, lib.codec_speaker_file_get_speaker(fh, b"saved_voice"))
        from_file_ref, _ = decode(lib, ctx, tokens, lib.codec_speaker_file_get_speaker(fh, b"ref"))
    finally:
        lib.codec_speaker_file_free(fh)
    if not np.array_equal(from_file_saved, implicit_pcm) or not np.array_equal(from_file_ref, cloned):
        return "decode with a speaker from the multi-speaker file differs from the original"
    loaded = ctypes.c_void_p()
    if lib.codec_speaker_load(model, str(multi_file).encode(), b"saved_voice", ctypes.byref(loaded)) != 0:
        return "codec_speaker_load by name from a multi-speaker file failed"
    lib.codec_speaker_free(loaded)
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
    lib_path = find_libcodec()
    if lib_path is None:
        print("SKIP: build/libcodec shared library not built")
        return 0
    if not (CHATTERBOX_DIR / "s3gen.safetensors").is_file() or not (CHATTERBOX_DIR / "conds.pt").is_file():
        print(f"SKIP: {CHATTERBOX_DIR} not present")
        return 0

    with tempfile.TemporaryDirectory(prefix="codec-s3g-decode-") as td:
        td = Path(td)
        gguf = td / "s3g.gguf"
        run([sys.executable, str(CONVERT), "--checkpoint-path", str(CHATTERBOX_DIR), "--model-type", "chatterbox_s3g", "--output", str(gguf)])

        tokens = np.ascontiguousarray([
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
            1100, 1200, 1300, 1400, 1500, 1600,
        ], dtype=np.int32)

        lib = bind(lib_path)
        model = lib.codec_model_load_from_file(
            str(gguf).encode(), codec_model_params(True, 4)
        )
        if not model:
            print("FAIL: model load")
            return 1
        ctx = lib.codec_init_from_model(model, codec_context_params(0))
        if not ctx:
            lib.codec_model_free(model)
            print("FAIL: context init")
            return 1

        spk_file = td / "speaker.gguf"
        try:
            results = [decode(lib, ctx, tokens) for _ in range(2)]
            speaker_fail = check_speakers(lib, model, ctx, tokens, results[0][0], spk_file)
        finally:
            lib.codec_free(ctx)
            lib.codec_model_free(model)
        if speaker_fail:
            print(f"FAIL: {speaker_fail}")
            return 1

        # A model whose speaker encoder differs must refuse the speaker file.
        other = kv_overrides([("chatterbox_s3g.spk.encoder_hash", CODEC_KV_OVERRIDE_TYPE_STR, "other-encoder")])
        model = lib.codec_model_load_from_file_ext(str(gguf).encode(), codec_model_params(True, 4), other)
        if not model:
            print("FAIL: model load with encoder_hash override")
            return 1
        loaded = ctypes.c_void_p()
        st = lib.codec_speaker_load(model, str(spk_file).encode(), None, ctypes.byref(loaded))
        lib.codec_model_free(model)
        if st == 0:
            lib.codec_speaker_free(loaded)
            print("FAIL: speaker file accepted by a model with a different encoder hash")
            return 1
        print("  speaker file rejected by mismatched encoder hash (expected)")

        # Bad schedules must be rejected at load, not at decode.
        bad = kv_overrides([("chatterbox_s3g.flow.n_timesteps", CODEC_KV_OVERRIDE_TYPE_INT, 0)])
        if lib.codec_model_load_from_file_ext(str(gguf).encode(), codec_model_params(True, 4), bad):
            print("FAIL: n_timesteps=0 override was accepted")
            return 1

        fast = kv_overrides([
            ("chatterbox_s3g.flow.n_timesteps", CODEC_KV_OVERRIDE_TYPE_INT, 2),
            ("chatterbox_s3g.flow.cfg_rate", CODEC_KV_OVERRIDE_TYPE_FLOAT, 0.0),
        ])
        model = lib.codec_model_load_from_file_ext(str(gguf).encode(), codec_model_params(True, 4), fast)
        if not model:
            print("FAIL: model load with flow overrides")
            return 1
        ctx = lib.codec_init_from_model(model, codec_context_params(0))
        if not ctx:
            lib.codec_model_free(model)
            print("FAIL: context init with flow overrides")
            return 1
        try:
            fast_pcm, _ = decode(lib, ctx, tokens)
        finally:
            lib.codec_free(ctx)
            lib.codec_model_free(model)

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
