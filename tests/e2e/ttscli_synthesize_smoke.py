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


def _run_synth(codec, backbone, text, out_wav, extra=None, greedy=True):
    cmd = [
        str(TTS_CLI), "synthesize",
        "--model", str(codec),
        "--backbone", str(backbone),
        "--text", text,
        "--output", str(out_wav),
        "--n-threads", "8",
    ]
    if greedy:
        cmd += ["--temp", "0"]
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
          require_stop=True, cer_report_only=False, dur_max=30.0,
          audio_report_only=False, greedy=True, stop_line="AR loop done"):
    """Drive one synthesize case.

    require_stop=False    → stop-reason is reported, not asserted (models
                            that don't reliably hit EOS free-running).
    cer_report_only=True  → the ASR transcript + CER are printed but never
                            fail the case.  Used for models whose full
                            prompt assembly isn't wired yet (Qwen3-TTS
                            talker needs MRoPE) or that need streaming the
                            host loop doesn't do (MOSS-TTS-Realtime).
    audio_report_only=True→ duration/silence are reported, not asserted.
                            For models whose sampled output isn't reliable
                            end-to-end yet (MOSS-TTS-Realtime): the case
                            proves the decode path runs + exits 0, but the
                            audio content is not guaranteed.
    """
    if not codec.exists() or not backbone.exists():
        print(f"[skip] {name}: missing assets ({codec.name} / {backbone.name})")
        return None
    out = Path("/tmp") / f"ttscli_{name}.wav"
    if out.exists():
        out.unlink()
    proc = _run_synth(codec, backbone, text, out, extra, greedy=greedy)
    if proc.returncode != 0:
        print(f"[FAIL] {name}: exit {proc.returncode}\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
        return False
    line = [ln for ln in proc.stdout.splitlines() if stop_line in ln]
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
        if audio_report_only:
            print(f"[WARN] {name}: duration {dur:.2f}s out of bounds")
        else:
            print(f"[FAIL] {name}: duration {dur:.2f}s out of bounds"); ok = False
    if rms < 0.005:
        if audio_report_only:
            print(f"[WARN] {name}: near-silent (rms={rms:.4f})")
        else:
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

    # Chatterbox T3 — embd-driven Llama backbone with the T3 LM adaptor
    # (own EnTokenizer BPE + cond_enc perceiver + text/speech learned PEs +
    # speech_head), CFG dual-sequence sampling, builtin voice conds, decode
    # via Chatterbox-S3G.  Sampled (not greedy) with the reference regime
    # (temp 0.8 / min_p 0.05 / rep 1.2 / cfg 0.5); a fixed seed keeps it
    # deterministic.  Stops on the T3 stop-speech token (eos_code_c0=6562)
    # and English ASR should match the input closely.
    results.append(_case(
        "chatterbox_t3",
        REPO / "models" / "chatterbox" / "chatterbox.gguf",
        REPO / "models" / "chatterbox" / "llama_backbone.gguf",
        "Hello, this is a test of the emergency broadcast system.",
        "en", want_stop="eos_code_c0", cer_max=0.3,
        extra=["--seed", "42", "--max-frames", "600"],
        greedy=False, stop_line="chatterbox AR done",
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
    # the multi-modal composed prompt prefill.  NOW INTELLIGIBLE: two fixes
    # landed — (a) the BF16 qwen3 backbone (the F16 downcast's ~6e-3 hidden
    # deltas flipped close-call parallel-head argmaxes → babble; BF16 matches
    # HF's native precision), and (b) the converter now bakes the correct
    # end-of-audio EOS (generation_config.eos_token_id=152694, not
    # config.eos_token_id=151643) so cb0 stops the audio.  Greedy (--temp 0)
    # on BF16 reproduces the input sentence and stops via eos_code_c0.
    results.append(_case(
        "moss_ttsd_v05",
        REPO / "models" / "moss_ttsd_v0_5" / "moss_ttsd_v0_5.gguf",
        REPO / "models" / "moss_ttsd_v0_5" / "qwen3_backbone_bf16.gguf",
        "[S1]你好，欢迎使用语音合成系统。",
        "zh", want_stop="eos_code_c0", cer_max=0.35,
        extra=["--max-frames", "360"],
    ))

    # MOSS-TTS-Realtime — residual depth-AR, 16 RVQ codebooks decoded through
    # the MOSS-Audio codec (native 32-quantizer RVQ, truncated to the LM's 16
    # via the decode-time n_q remap).  The 16→32 decode mismatch is FIXED
    # (moss_audio honours a partial n_q).  The streaming text↔audio interleave
    # is now implemented (run_realtime_streaming in tts-cli): the spoken text
    # is fed one token per audio frame into the ASSISTANT turn, the per-step
    # backbone input is text_embd[token] + compose_audio_embd(prev_frame), the
    # first prefill_text_len=12 text tokens open the audio channel (cb0 BOS),
    # and generation stops on cb0 == audio EOS (1026).  Now intelligible: this
    # case asserts stop==eos_code_c0, non-silent audio, and (when whisper is
    # available) an ASR CER within bounds.  Uses the reference sampling regime
    # (temp 0.8 / top-p 0.6 / top-k 30 / rep-penalty 1.1).
    results.append(_case(
        "moss_tts_realtime",
        REPO / "models" / "moss_tts_realtime" / "moss_tts_realtime.gguf",
        REPO / "models" / "moss_tts_realtime" / "qwen3_backbone_bf16.gguf",
        "你好，欢迎使用语音合成。",
        "zh", want_stop="eos_code_c0", cer_max=0.3,
        extra=["--temp", "0.8", "--top-p", "0.6", "--top-k", "30",
               "--seed", "42", "--max-frames", "500"],
        require_stop=True, cer_report_only=False, audio_report_only=False,
        dur_max=35.0,
    ))

    # Qwen3-TTS-0.6B-Base — residual depth-AR + ECAPA-TDNN speaker encoder.
    # The faithful talker prompt is assembled (converter bakes text_projection
    # + text_embedding + control-tag ids; codec_common builds the additive
    # dual-lane prefix — projected text lane summed with the codec control-tag
    # lane, ECAPA x-vector inserted between the think-tags and pad/bos; tts-cli
    # injects the projected trailing text per-step).  The talker backbone runs
    # as plain `qwen3` (1D RoPE) which is the exact reduction of the reference's
    # interleaved-MRoPE for pure-audio TTS — see qwen3_tts_backbone_smoke.py.
    # Intelligibility was fixed by carrying the depth-decoder FFN matmuls at F32
    # (the SwiGLU intermediate reaches ~1.4e5, which overflowed the F16 mul_mat
    # activation downcast to inf -> NaN codes; codec_op_lm_per_pos_linear now
    # dequants F16 weights to F32 so activations stay F32).  Asserted: stop via
    # eos_code_c0 and CER within bound.
    results.append(_case(
        "qwen3_tts_refaudio",
        REPO / "models" / "qwen3_tts" / "qwen3_tts_06b_base.gguf",
        REPO / "models" / "qwen3_tts" / "qwen3_tts_talker.gguf",
        "你好，欢迎使用语音合成。",
        "zh", want_stop="eos_code_c0", cer_max=0.3,
        extra=["--ref-audio", str(REPO / "test.wav"),
               "--temp", "0.9", "--top-k", "50", "--seed", "42",
               "--max-frames", "300"],
        require_stop=True, cer_report_only=False,
    ))

    # LFM2-Audio-1.5B — sequential text→audio TTS (residual depth-AR, 8 Mimi
    # codebooks).  The backbone (lfm2 arch) first free-runs in TEXT modality
    # off its tied-embedding lm_head; the "Perform TTS. Use the US male voice."
    # system prompt makes it emit <|audio_start|> as the first token, switching
    # to AUDIO_OUT where the depth decoder emits 8-codebook frames fed back via
    # compose_audio_codes_embd, stopping on cb0 == EOAudio (2048).  llama.cpp
    # omits the output head when embeddings are enabled, so tts-cli recomputes
    # text logits as hidden · token_embd (the tied head).  GREEDY TTS is
    # degenerate for this model (the reference itself gets stuck on "Hello."),
    # so this case samples (temp 0.8 / top-k 64, fixed seed) per the reference
    # regime — it stops on eos_code_c0 at ~4-6s and English ASR matches.
    results.append(_case(
        "lfm2_audio",
        REPO / "models" / "lfm2_audio" / "lfm2_audio.gguf",
        REPO / "models" / "lfm2_audio" / "lfm_backbone.gguf",
        "Hello, this is a test of the emergency broadcast system.",
        "en", want_stop="eos_code_c0", cer_max=0.3,
        extra=["--temp", "0.8", "--top-k", "64", "--seed", "42",
               "--max-frames", "400"],
        greedy=False, require_stop=True,
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
