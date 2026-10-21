#!/usr/bin/env python3
"""Text → `tts-cli synthesize` → Whisper → text round trip for the CosyVoice family.

Cases (each skipped when its GGUFs are missing under models/cosyvoice/):
  * zero-shot voice cloning (--ref-audio + --ref-text) for CosyVoice-300M,
    BreezyVoice, CosyVoice2 and CosyVoice3,
  * builtin voices (--voice): CosyVoice-300M-SFT (x-vector only) and
    BreezyVoice (prompt tokens / mel without a transcript),
  * cross-lingual (--ref-audio without --ref-text) for CosyVoice-300M and CosyVoice3,
  * a metadata override (cosyvoice.flow.speed = 1.5 via --override-kv).
Each must stop on the speech EOS, produce non-silent audio, and (when Whisper
is available) transcribe with CER below the threshold.

GGUFs: <name>.gguf from scripts/convert-to-gguf.py and <name>-backbone.gguf
from scripts/convert-backbone-to-gguf.py (see docs/models/cosyvoice.md).
Run from the repo root: .venv/bin/python tests/e2e/cosyvoice_synthesize_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from _common import MODEL_SAMPLING, REPO, sampling_flags
from ttscli_synthesize_smoke import _try_cer, _wav_stats
TTS_CLI = REPO / "build" / "tts-cli"
MODELS = REPO / "models" / "cosyvoice"
REF_WAV = REPO / "input_audio" / "10_2_24k.wav"
REF_TEXT = "在恶魔甩着头发，喷着口沫，张牙舞爪的扑在老夫人的镜子上，恶住他的脆弱的口管。"
ZH_TEXT = "今天天气很好，我们一起出去散步吧。"
EN_TEXT = "Hello, this is a cross lingual test of the speech synthesizer."


def case(label, name, text, lang, extra, cer_max, tmp) -> bool:
    codec, backbone = MODELS / f"{name}.gguf", MODELS / f"{name}-backbone.gguf"
    if not codec.is_file() or not backbone.is_file():
        print(f"[SKIP] {label}: {codec.name} / {backbone.name} not found")
        return True
    out = Path(tmp) / f"{label}.wav"
    cmd = [str(TTS_CLI), "synthesize", "--model", str(codec), "--backbone", str(backbone),
           "--text", text, "--output", str(out), "--n-threads", "8", "--seed", "1"]
    cmd += sampling_flags(MODEL_SAMPLING["breezyvoice" if name.startswith("breezy") else "cosyvoice"]) + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not out.is_file():
        print(f"[FAIL] {label}: exit {proc.returncode}\n{proc.stderr[-2000:]}")
        return False
    if "stop=eos" not in proc.stdout:
        print(f"[FAIL] {label}: generation did not stop on EOS")
        return False
    sr, dur, rms = _wav_stats(out)
    ok = 0.5 < dur < 30.0 and rms > 0.005
    cer_res = _try_cer(out, text, lang)
    cer_str = ""
    if cer_res is not None:
        asr, cer = cer_res
        cer_str = f" ASR='{asr}' CER={cer:.3f}"
        ok = ok and cer <= cer_max
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {sr} Hz dur={dur:.2f}s rms={rms:.3f}{cer_str}")
    return ok


def main() -> int:
    ref = ["--ref-audio", str(REF_WAV), "--ref-text", REF_TEXT]
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("cosyvoice-300m", "breezyvoice", "cosyvoice2", "cosyvoice3"):
            ok &= case(f"{name}-zero-shot", name, ZH_TEXT, "zh", ref, 0.3, tmp)
        ok &= case("cosyvoice-300m-sft-voice", "cosyvoice-300m-sft", ZH_TEXT, "zh", ["--voice", "zh_female"], 0.3, tmp)
        # BreezyVoice voices carry prompt tokens / mel but no transcript: cross-lingual prompting.
        ok &= case("breezyvoice-voice", "breezyvoice", ZH_TEXT, "zh", ["--voice", "zh_female"], 0.3, tmp)
        ok &= case("cosyvoice2-speed", "cosyvoice2", ZH_TEXT, "zh",
                   ref + ["--override-kv", "cosyvoice.flow.speed=float:1.5"], 0.3, tmp)
        for name in ("cosyvoice-300m", "cosyvoice3"):
            ok &= case(f"{name}-cross-lingual", name, EN_TEXT, "en", ["--ref-audio", str(REF_WAV)], 0.3, tmp)
    print("cosyvoice synthesize smoke test " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
