"""End-to-end smoke test for `tts-cli synthesize`.

Drives the built `tts-cli` binary through the FULL host AR loop (a
llama.cpp backbone + codec_common per-step hooks) for the models that
work end-to-end with local assets, and asserts:

  * the process exits 0 and writes a WAV,
  * generation stopped via the model's EOS signal (not the frame cap),
  * the WAV duration is within sane bounds and non-silent,
  * (optional, when transformers+whisper are available) the ASR CER
    against the input text is below a loose threshold.

CSM is the primary case: deterministic greedy generation stops on the
all-zero (eos_code_c0) frame, and English ASR matches the input exactly.
BlueMagpie (continuous CFM) is included when its scratch backbone is
present.  Missing assets → the sub-case is skipped, not failed.

Run from the repo root:  .venv/bin/python tests/e2e/ttscli_synthesize_smoke.py
"""

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TTS_CLI = REPO / "build" / "tts-cli"


def _run_synth(codec, backbone, text, out_wav, extra=None):
    cmd = [
        str(TTS_CLI), "synthesize",
        "--model", str(codec),
        "--backbone", str(backbone),
        "--text", text,
        "--output", str(out_wav),
        "--temp", "0",
        "--n-threads", "8",
    ]
    if extra:
        cmd += extra
    env = dict(os.environ)
    env["GGML_VK_DISABLE"] = env.get("GGML_VK_DISABLE", "0")
    proc = subprocess.run(cmd, capture_output=True, env=env)
    proc.stdout = proc.stdout.decode("utf-8", errors="replace")
    proc.stderr = proc.stderr.decode("utf-8", errors="replace")
    return proc


def _wav_stats(path):
    w = wave.open(str(path), "rb")
    n = w.getnframes()
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    w.close()
    return sr, len(x) / sr, float(np.sqrt((x ** 2).mean()))


def _try_cer(wav, ref, lang):
    """Best-effort ASR CER; returns None if deps/models unavailable."""
    try:
        import torch  # noqa
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except Exception:
        return None
    try:
        w = wave.open(str(wav), "rb")
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        w.close()
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        m = "openai/whisper-large-v3-turbo"
        proc = WhisperProcessor.from_pretrained(m)
        model = WhisperForConditionalGeneration.from_pretrained(m)
        inp = proc(x, sampling_rate=16000, return_tensors="pt")
        ids = model.generate(inp.input_features, language=lang, task="transcribe")
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    except Exception:
        return None

    def norm(s):
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFKC", s.lower()) if c.isalnum())

    a, b = norm(ref), norm(txt)
    d = np.zeros((len(a) + 1, len(b) + 1), int)
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return txt, d[len(a)][len(b)] / max(1, len(a))


def _case(name, codec, backbone, text, lang, want_stop, cer_max, extra=None,
          require_stop=True, cer_report_only=False, dur_max=30.0):
    """Drive one synthesize case.

    require_stop=False   → stop-reason is reported, not asserted (models
                           that don't reliably hit EOS free-running).
    cer_report_only=True → the ASR transcript + CER are printed but never
                           fail the case.  Used for models whose intrinsic
                           llama.cpp-vs-HF backbone numerical drift makes
                           free-running generation diverge (MOSS-TTSD) or
                           whose full talker prompt assembly isn't wired
                           yet (Qwen3-TTS voice clone).
    """
    if not codec.exists() or not backbone.exists():
        print(f"[skip] {name}: missing assets ({codec.name} / {backbone.name})")
        return None
    out = Path("/tmp") / f"ttscli_{name}.wav"
    if out.exists():
        out.unlink()
    proc = _run_synth(codec, backbone, text, out, extra)
    if proc.returncode != 0:
        print(f"[FAIL] {name}: exit {proc.returncode}\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
        return False
    line = [ln for ln in proc.stdout.splitlines() if "AR loop done" in ln]
    stop = line[0].split("stop=")[-1].strip() if line else "?"
    assert out.exists(), f"{name}: no wav written"
    sr, dur, rms = _wav_stats(out)
    ok = True
    if want_stop and stop != want_stop:
        if require_stop:
            print(f"[FAIL] {name}: stop={stop} (wanted {want_stop})"); ok = False
        else:
            print(f"[WARN] {name}: stop={stop} (wanted {want_stop})")
    if not (0.3 <= dur <= dur_max):
        print(f"[FAIL] {name}: duration {dur:.2f}s out of bounds"); ok = False
    if rms < 0.005:
        print(f"[FAIL] {name}: silent (rms={rms:.4f})"); ok = False
    cer_res = _try_cer(out, text, lang)
    cer_str = ""
    if cer_res is not None:
        asr, cer = cer_res
        cer_str = f" ASR='{asr}' CER={cer:.3f}"
        if cer > cer_max and not cer_report_only:
            print(f"[FAIL] {name}: CER {cer:.3f} > {cer_max}"); ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: stop={stop} dur={dur:.2f}s rms={rms:.3f}{cer_str}")
    return ok


def main():
    if not TTS_CLI.exists():
        print(f"tts-cli not built at {TTS_CLI}; run cmake --build build --target tts-cli")
        return 1

    results = []

    # CSM — deterministic, stops on eos_code_c0, English ASR exact.
    results.append(_case(
        "csm",
        REPO / "models" / "csm" / "csm.gguf",
        REPO / "models" / "csm" / "llama_backbone.gguf",
        "Hello, this is a test of the emergency broadcast system.",
        "en", want_stop="eos_code_c0", cer_max=0.15,
        extra=["--max-frames", "300"],
    ))

    # BlueMagpie — continuous CFM, stops via the diffusion stop head.
    bm_bb = Path.home() / "Projects" / "llama.rn" / ".scratch" / "gguf" / "BlueMagpie-Barbet-1B-f16.gguf"
    results.append(_case(
        "bluemagpie",
        REPO / "models" / "bluemagpie" / "bluemagpie.gguf",
        bm_bb,
        "今天天氣很好",
        "zh", want_stop="stop_head", cer_max=0.35,
        extra=["--cfg", "2.8", "--timesteps", "9", "--max-frames", "256"],
    ))

    # MOSS-TTSD v0.5 — parallel-heads-delay.  Exercises the full codes→PCM
    # transform (cb0 speech-range remap + delay un-shift, 8→codec n_q=8) and
    # the multi-modal composed prompt prefill.  The codec transform is
    # correct (audio is speech-shaped, non-silent), but free-running AR on
    # llama.cpp's F16 Qwen3 backbone vs HF's BF16 diverges on the first frame
    # (backbone corr 0.999998 but max_abs_diff ~6e-3 flips close-call parallel
    # heads → ~10 % per-step flips compound), so the transcript is babble.
    # Asserted: exit 0, non-silent, sane duration.  CER report-only; stop not
    # required (free-running rarely reaches text EOS).  Cap frames ≤ the
    # XY-Tokenizer post_rvq_adapter pos_emb limit (375).
    results.append(_case(
        "moss_ttsd_v05",
        REPO / "models" / "moss_ttsd_v0_5" / "moss_ttsd_v0_5.gguf",
        REPO / "models" / "moss_ttsd_v0_5" / "qwen3_backbone.gguf",
        "[S1]你好，欢迎使用语音合成系统。",
        "zh", want_stop="eos_code_c0", cer_max=1.0,
        extra=["--max-frames", "360"],
        require_stop=False, cer_report_only=True,
    ))

    # Qwen3-TTS-0.6B-Base — residual depth-AR + ECAPA-TDNN speaker encoder.
    # With a --ref-audio x-vector prefix the talker stops on eos_code_c0
    # (2150) — the conditioning path is structurally wired.  Full voice-clone
    # intelligibility needs the talker's projected prompt assembly
    # (text_projection MLP + codec control-tag interleaving) which is not
    # dumped into the GGUF, so the raw-x-vector + ChatML approximation is not
    # yet intelligible.  Asserted: exit 0, non-silent.  CER report-only.
    results.append(_case(
        "qwen3_tts_refaudio",
        REPO / "models" / "qwen3_tts" / "qwen3_tts_06b_base.gguf",
        REPO / "models" / "qwen3_tts" / "qwen3_tts_talker.gguf",
        "你好，欢迎使用语音合成。",
        "zh", want_stop="eos_code_c0", cer_max=1.0,
        extra=["--ref-audio", str(REPO / "test.wav"), "--max-frames", "300"],
        require_stop=False, cer_report_only=True,
    ))

    ran = [r for r in results if r is not None]
    if not ran:
        print("no cases ran (all assets missing)")
        return 0
    if all(ran):
        print(f"\nOK: {len(ran)}/{len(ran)} synthesize cases passed")
        return 0
    print(f"\nFAILED: {sum(1 for r in ran if not r)}/{len(ran)} cases failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
