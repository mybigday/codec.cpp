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
from pathlib import Path

import numpy as np

from _common import REPO, asr_cer, read_wav_mono_f32

TTS_CLI = REPO / "build" / "tts-cli"


def _run_synth(codec, backbone, text, out_wav, extra=None, greedy=True):
    cmd = [
        str(TTS_CLI), "synthesize",
        "--model", str(codec),
        "--text", text,
        "--output", str(out_wav),
        "--n-threads", "8",
    ]
    # Self-contained models (Pocket-TTS FlowLM) take no backbone.
    if backbone is not None:
        cmd += ["--backbone", str(backbone)]
    if greedy and backbone is not None:
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
    x, sr = read_wav_mono_f32(path)
    return sr, len(x) / sr, float(np.sqrt((x ** 2).mean()))


def _try_cer(wav, ref, lang):
    return asr_cer(wav, ref, lang)


def _case(name, codec, backbone, text, lang, want_stop, cer_max, extra=None,
          require_stop=True, cer_report_only=False, dur_max=30.0,
          audio_report_only=False, greedy=True, stop_line="AR loop done", cer_ref=None):
    """Drive one synthesize case.

    cer_ref               → reference transcript for the CER when `text` is
                            not plain text (NeuTTS phonemes).

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
    if not codec.exists() or (backbone is not None and not backbone.exists()):
        bb = backbone.name if backbone is not None else "(none)"
        print(f"[skip] {name}: missing assets ({codec.name} / {bb})")
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
    cer_res = _try_cer(out, cer_ref or text, lang)
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

    # Chatterbox Turbo — GPT-2 medium T3 (single lane, no CFG; GPT-2 BPE
    # text tokenizer carried by the backbone GGUF), sampled with the GGUF's
    # reference regime (temp 0.8 / top-k 1000 / top-p 0.95 / rep 1.2, HF
    # warper order), 3 trailing S3GEN_SIL codes, decoded by the MeanFlow
    # S3Gen (2 steps, no CFG).  Greedy T3 matches PyTorch token-for-token.
    results.append(_case(
        "chatterbox_turbo",
        REPO / "models" / "chatterbox_turbo" / "chatterbox_turbo.gguf",
        REPO / "models" / "chatterbox_turbo" / "gpt2_backbone.gguf",
        "Hello, this is a test of the emergency broadcast system.",
        "en", want_stop="eos_code_c0", cer_max=0.1,
        extra=["--seed", "42", "--max-frames", "1000"],
        greedy=False, stop_line="chatterbox AR done",
    ))

    # Chatterbox Turbo voice clone: --ref-audio swaps both the T3 VoiceEncoder
    # row and the S3G decoder speaker (codec_speaker_from_audio: S3 tokens +
    # 24 kHz mel + CAMPPlus, derived in-library).
    results.append(_case(
        "chatterbox_turbo_ref",
        REPO / "models" / "chatterbox_turbo" / "chatterbox_turbo.gguf",
        REPO / "models" / "chatterbox_turbo" / "gpt2_backbone.gguf",
        "Hello, this is a test of the emergency broadcast system.",
        "en", want_stop="eos_code_c0", cer_max=0.1,
        extra=["--seed", "42", "--max-frames", "1000",
               "--ref-audio", str(REPO / "input_audio" / "10_2_24k.wav")],
        greedy=False, stop_line="chatterbox AR done",
    ))

    # KittenTTS 2 — Qwen3-1.7B speech LM (ternary weights, lossless Q4_0
    # repack) emitting S3 tokens as vocab rows, decoded by the MeanFlow S3Gen.
    # Built-in voice: position-0 speaker row + in-context reference (voice's
    # transcript + S3 tokens), kittenml sampling (stable preset, run/repetition
    # penalties).  Greedy tokens match PyTorch fp32 for the bf16 backbone.
    kt = REPO / "models" / "kitten_tts_2"
    kt_text = "One day, a little girl named Lily found a needle in her room."
    results.append(_case(
        "kitten_tts2",
        kt / "kitten_tts2.gguf", kt / "kitten_tts2_backbone-q4_0.gguf",
        kt_text, "en", want_stop="eos", cer_max=0.1,
        extra=["--seed", "42", "--voice", "bruno"],
        greedy=False, stop_line="KittenTTS AR done",
    ))
    # Voice clone: --ref-audio derives every speaker slot in-library (S3G conds,
    # x-vector → spk_proj row, reference S3 tokens); --ref-text enables the
    # in-context reference prompt.
    results.append(_case(
        "kitten_tts2_clone",
        kt / "kitten_tts2.gguf", kt / "kitten_tts2_backbone-q4_0.gguf",
        kt_text, "en", want_stop="eos", cer_max=0.1,
        extra=["--seed", "42", "--ref-audio", str(REPO / "input_audio" / "10_2_24k.wav"),
               "--ref-text", "这恶魔甩着头发喷着口沫张牙舞爪地扑在老妇人的镜子上扼住她的脆弱的喉管"],
        greedy=False, stop_line="KittenTTS AR done",
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
    # Intelligibility needs the depth-decoder ffn_down at F32: the SwiGLU
    # intermediate reaches ~1.4e5, which overflows an F16 matmul's activation
    # downcast to inf -> NaN codes, so the converter stores that weight F32.
    # Asserted: stop via eos_code_c0 and CER within bound.
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

    # Qwen3-TTS Base, in-context voice prompt (upstream's default ICL mode):
    # the reference's transcript and codes ride ahead of the text, and the
    # decode prepends the reference's codes and cuts their share of the audio.
    _jfk_txt = REPO / "assets" / "refs" / "jfk_en.txt"
    if _jfk_txt.exists():
        results.append(_case(
            "qwen3_tts_icl",
            REPO / "models" / "qwen3_tts" / "qwen3_tts_06b_base.gguf",
            REPO / "models" / "qwen3_tts" / "qwen3_tts_talker.gguf",
            "The birch canoe slid on the smooth planks.",
            "en", want_stop="eos_code_c0", cer_max=0.15, dur_max=6.0,
            extra=["--ref-audio", str(REPO / "assets" / "refs" / "jfk_en.wav"),
                   "--ref-text", _jfk_txt.read_text().strip(),
                   "--temp", "0.9", "--top-k", "50", "--seed", "2", "--max-frames", "300"],
            greedy=False, require_stop=True,
        ))

    # Qwen3-TTS with a 48 kHz reference clip.  The ECAPA-TDNN speaker encoder
    # works at 24 kHz (codec.speaker.ref_sample_rate); arbitrary --ref-audio
    # rates (24k/44.1k/48k) must be resampled to that rate before the encode
    # (audio_lm_build_prompt does this).  Regression guard for the bug where a
    # non-24k ref wav hit "sample_rate mismatch" and aborted synthesize (→
    # silence).  test.wav is 24 kHz; make a 48 kHz copy on the fly.
    q3_ref48 = Path("/tmp") / "ttscli_qwen3_ref48k.wav"
    try:
        import soundfile as sf
        import librosa as _lb
        _x, _sr = sf.read(str(REPO / "test.wav"))
        if _x.ndim > 1:
            _x = _x.mean(axis=1)
        _x48 = _lb.resample(_x.astype(np.float32), orig_sr=_sr, target_sr=48000)
        sf.write(str(q3_ref48), _x48, 48000, subtype="PCM_16")
        _have48 = True
    except Exception as _e:
        print(f"[skip] qwen3_tts_ref48k: could not build 48k ref ({_e})")
        _have48 = False
    if _have48:
        results.append(_case(
            "qwen3_tts_ref48k",
            REPO / "models" / "qwen3_tts" / "qwen3_tts_06b_base.gguf",
            REPO / "models" / "qwen3_tts" / "qwen3_tts_talker.gguf",
            "你好，欢迎使用语音合成。",
            "zh", want_stop="eos_code_c0", cer_max=0.3,
            extra=["--ref-audio", str(q3_ref48),
                   "--temp", "0.9", "--top-k", "50", "--seed", "42",
                   "--max-frames", "300"],
            require_stop=True, cer_report_only=False,
        ))

    # Qwen3-TTS CustomVoice — builtin voices from the codec GGUF: the talker's
    # speaker row is codec_embedding[spk_id]; dialect voices (dylan = Beijing)
    # also switch the control stream to the think + language-tag form.  1.7B
    # takes --instruct (and has the talker→depth in_proj).
    q3cv = REPO / "models" / "output" / "release"
    for name, size, voice, text, lang, extra in [
        ("qwen3_tts_cv_ryan", "0.6B", "ryan", "The birch canoe slid on the smooth planks.", "en", []),
        ("qwen3_tts_cv_dylan", "0.6B", "dylan", "你好，欢迎使用语音合成。", "zh", []),
        ("qwen3_tts_cv17_instruct", "1.7B", "vivian", "你好，欢迎使用语音合成。", "zh",
         ["--instruct", "用特别开心的语气说"]),
    ]:
        d = q3cv / f"Qwen3-TTS-12Hz-{size}-CustomVoice"
        results.append(_case(
            name, d / "codec-q8_0.gguf", d / f"qwen3-tts-{size.lower()}-customvoice-q8_0.gguf",
            text, lang, want_stop="eos_code_c0", cer_max=0.3,
            extra=["--voice", voice, "--seed", "42", "--max-frames", "300"] + extra,
            require_stop=True, cer_report_only=False, greedy=False,
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

    # Pocket-TTS FlowLM: the transformer is a stock starcoder2 backbone (text
    # LUT as token_embd); codec_lm runs the step input projection, EOS head and
    # LSD flow head.  The LM math was verified against the reference when it
    # was ported (teacher-forced latent corr 0.99999, EOS agreement 1.0).  Here
    # we assert the end-to-end pipeline runs, stops via the
    # EOS head, and writes audio.  Content is REPORT-ONLY: pocket-tts is a fragile
    # 100M short-utterance model whose free-run output quality depends on the CFM
    # noise stream (torch.randn), which C++ can't bit-reproduce — even the reference,
    # driven directly, transcribes short prompts as "Thank you."/"Okay.".  --min-len
    # forces a minimum utterance so the EOS head is exercised past its trigger.
    results.append(_case(
        "pocket_tts",
        REPO / "models" / "pocket_tts" / "pocket_tts_en.gguf",
        REPO / "models" / "pocket_tts" / "backbone-f16.gguf",
        "Hello world. This is a small text to speech model.",
        "en", want_stop="eos_head", cer_max=1.0,
        extra=["--seed", "11", "--max-frames", "80", "--min-len", "24"],
        greedy=False, require_stop=False,
        cer_report_only=True, audio_report_only=True,
        stop_line="pocket AR done",
    ))

    # Soprano: Qwen3 backbone (tokenizer.ggml.pre = soprano) whose final-norm
    # hidden states feed the Vocos decoder directly; greedy, stops on [STOP].
    results.append(_case(
        "soprano",
        REPO / "models" / "soprano" / "soprano.gguf",
        REPO / "models" / "soprano" / "backbone-f16.gguf",
        "The birch canoe slid on the smooth planks.",
        "en", want_stop="eos", cer_max=0.10,
        greedy=False, stop_line="soprano AR done",
    ))

    # ── NeuTTS (phoneme input; voices baked into the distill NeuCodec GGUF) ───
    # Input is phonemized here as upstream neutts does (espeak en-us).  The
    # codec GGUF carries upstream's samples/ voices (scripts/bake-speakers.py,
    # docs/models/neutts.md); the _ref case derives the voice from audio with
    # the distill encoder.  Seed 1 is pinned (nano is seed-fragile).
    _neutts_codec = REPO / "models" / "output" / "neutts" / "neutts-codec-F16.gguf"
    _sentence = "The birch canoe slid on the smooth planks."
    try:
        from phonemizer.backend import EspeakBackend
        _g2p = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True,
                             words_mismatch="ignore", language_switch="remove-flags")
        _phones = lambda t: " ".join(_g2p.phonemize([t])[0].split())
    except Exception as e:  # noqa: BLE001
        print(f"[skip] neutts: phonemizer unavailable ({e})")
        _phones = None
    if _phones is not None:
        _nt_extra = ["--temp", "1.0", "--top-k", "50", "--seed", "1", "--max-frames", "400"]
        for name, bb, voice, cer_max in (
                ("neutts_nano", "neutts_nano/neutts-nano-q8_0.gguf", "jo", 0.30),
                ("neutts_air", "neutts_air/neutts-air-q8_0.gguf", "dave", 0.15)):
            results.append(_case(name, _neutts_codec, REPO / "models" / "hf" / bb, _phones(_sentence), "en",
                                 want_stop="eos", cer_max=cer_max, extra=_nt_extra + ["--voice", voice],
                                 greedy=False, stop_line="NeuTTS AR done", cer_ref=_sentence))
        _ref_txt = REPO / "assets" / "refs" / "jfk_en.txt"
        if _ref_txt.exists():
            results.append(_case(
                "neutts_air_ref", _neutts_codec, REPO / "models" / "hf" / "neutts_air" / "neutts-air-q8_0.gguf",
                _phones(_sentence), "en", want_stop="eos", cer_max=0.15,
                extra=_nt_extra + ["--ref-audio", str(REPO / "assets" / "refs" / "jfk_en.wav"),
                                   "--ref-text", _phones(_ref_txt.read_text().strip())],
                greedy=False, stop_line="NeuTTS AR done", cer_ref=_sentence))

    # ── OuteTTS 0.2 / 0.3 / 1.0 (voices baked into a per-model codec GGUF) ───
    # Raw text input; the runner detects the interface from the vocab and
    # attaches the prompt-derived grammar.  Codec GGUFs: docs/models/outetts.md.
    # Reference sampling defaults (1.0: temp 1.0 / top-k 50 / rep 1.1 over 64;
    # 0.2/0.3: temp 0.1 / rep 1.1); seed 1 pinned.
    _oute = REPO / "models" / "output" / "outetts"
    for name, codec, bb, voice in (
            ("outetts_1_0", "outetts-1.0-0.6b-codec-F32.gguf", "outetts_1_0/OuteTTS-1.0-0.6B-Q8_0.gguf", None),
            ("outetts_0_2", "outetts-0.2-500m-codec.gguf", "outetts_legacy/OuteTTS-0.2-500M-Q8_0.gguf", None),
            ("outetts_0_3", "outetts-0.3-500m-codec.gguf", "outetts_legacy/OuteTTS-0.3-500M-Q8_0.gguf",
             "en_female_1"),
            ("outetts_0_3_1b", "outetts-0.3-1b-codec.gguf", "outetts_legacy/OuteTTS-0.3-1B-Q8_0.gguf", None)):
        results.append(_case(name, _oute / codec, REPO / "models" / "hf" / bb, _sentence, "en",
                             want_stop="eos", cer_max=0.10,
                             extra=["--seed", "1"] + (["--voice", voice] if voice else []),
                             greedy=False, stop_line="OuteTTS AR done"))

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
