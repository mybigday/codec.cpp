#!/usr/bin/env python3
"""E2E smoke for codec_cpp.TTS: the Python host loop against tts-cli.

codec_cpp.TTS drives the llama.cpp bundled with codec.cpp (libttsbackbone)
through ctypes and codec_audio_lm through codec_cpp.AudioLM; tts-cli runs
the C++ loop (common/tts_runner.cpp) over the same two libraries.  With the
same seed and options both must produce the same audio: each case compares
the samples (as 16-bit PCM, the WAV tts-cli writes) and the stop reason.
Cases cover token / codec-head / hidden flows, a guidance pair (n_seq 2),
DRY + penalty-exempt sampling, a cloned voice and a self-contained model;
missing GGUFs skip a case.

Run from the repo root: .venv/bin/python tests/e2e/python_tts_smoke.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec_cpp  # noqa: E402
from _common import MODEL_SAMPLING, read_wav_mono_f32, sampling_flags  # noqa: E402

TTS_CLI = REPO / "build" / "tts-cli"
M = REPO / "models"
REF = REPO / "assets" / "refs"

_JFK = (REF / "jfk_en.txt").read_text().strip()


def _case(name, codec, backbone, text, card, seed, cli=(), opts=None, max_frames=0):
    """Both sides get the model card's sampling, the seed and the case's own options."""
    sp = dict(MODEL_SAMPLING.get(card, {}), seed=seed)
    return (name, codec, backbone, text, sampling_flags(sp) + list(cli), dict(sp, **(opts or {})), max_frames)


# name, codec, backbone (None = self-contained), text, tts-cli args, codec_cpp.TTS options, max_frames
CASES = [
    _case("csm", M / "csm/csm.gguf", M / "csm/llama_backbone.gguf",
          "Hello, this is a test of the emergency broadcast system.", "csm", 0xC0DEC1AB,
          ["--temp", "0"], dict(temp=0)),
    _case("chatterbox_t3", M / "chatterbox/chatterbox.gguf", M / "chatterbox/llama_backbone.gguf",
          "Hello, this is a test of the emergency broadcast system.", "chatterbox", 42),
    _case("kitten_tts2", M / "kitten_tts_2/kitten_tts2.gguf", M / "kitten_tts_2/kitten_tts2_backbone-q4_0.gguf",
          "One day, a little girl named Lily found a needle in her room.", "kitten_tts2", 42,
          ["--voice", "bruno"], dict(voice="bruno")),
    _case("qwen3_tts_icl", M / "qwen3_tts/qwen3_tts_06b_base.gguf", M / "qwen3_tts/qwen3_tts_talker.gguf",
          "The birch canoe slid on the smooth planks.", "qwen3_tts", 2,
          ["--ref-audio", str(REF / "jfk_en.wav"), "--ref-text", _JFK],
          dict(ref_audio=REF / "jfk_en.wav", ref_text=_JFK)),
    _case("outetts_1_0", M / "output/outetts/outetts-1.0-0.6b-codec-F32.gguf",
          M / "hf/outetts_1_0/OuteTTS-1.0-0.6B-Q8_0.gguf",
          "The birch canoe slid on the smooth planks.", "outetts_1_0", 1),
    _case("soprano", M / "soprano/soprano.gguf", M / "soprano/backbone-f16.gguf",
          "The birch canoe slid on the smooth planks.", "soprano", 42),
    _case("pocket_tts", M / "pocket_tts/pocket_tts_en.gguf", M / "pocket_tts/backbone-f16.gguf",
          "Hello world. This is a small text to speech model.", "pocket_tts", 11,
          ["--max-frames", "80", "--min-len", "24"], dict(min_frames=24), 80),
]


def tts_cli(codec, backbone, text, args, out):
    cmd = [str(TTS_CLI), "synthesize", "--model", str(codec), "--text", text, "--output", str(out)] + args
    if backbone is not None:
        cmd += ["--backbone", str(backbone)]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"tts-cli failed:\n{p.stderr[-2000:]}")
    lines = (p.stdout + p.stderr).splitlines()
    stop = next((ln.split("stop=")[1].split()[0] for ln in lines[::-1] if "stop=" in ln), "")
    return stop


def pcm16(x):
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int32)


def main() -> int:
    if not TTS_CLI.exists():
        print(f"tts-cli not built at {TTS_CLI}")
        return 1
    failed = 0
    for name, codec, backbone, text, args, opts, max_frames in CASES:
        if not codec.exists() or (backbone is not None and not backbone.exists()):
            print(f"[skip] {name}: missing GGUFs")
            continue
        try:
            with tempfile.TemporaryDirectory() as d:
                cli_stop = tts_cli(codec, backbone, text, args, Path(d) / "cli.wav")
                ref, ref_sr = read_wav_mono_f32(Path(d) / "cli.wav")
            with codec_cpp.TTS(backbone, codec, max_frames=max_frames) as tts:
                audio, sr = tts.synthesize(text, **opts)
                stop = tts.last_result["stop_reason"]
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
            continue
        a = audio if audio.ndim == 1 else audio.mean(axis=1)
        same_len = len(a) == len(ref) and sr == ref_sr
        diff = int(np.abs(pcm16(a) - pcm16(ref)).max()) if same_len and len(a) else -1
        ok = same_len and 0 <= diff <= 1 and (not cli_stop or cli_stop == stop)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: samples={len(a)} (tts-cli {len(ref)}) "
              f"max|d|={diff} lsb stop={stop} (tts-cli {cli_stop or '?'})")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
