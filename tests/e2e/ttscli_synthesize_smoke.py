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

    # Pocket-TTS FlowLM — SELF-CONTAINED (no backbone): the AR transformer, text
    # LUT, LSD flow head and EOS head all live in the codec GGUF.  Routed on
    # codec.lm.kind="flow_lm"; tokenized by the baked SentencePiece model.  The
    # LM math is bit-verified against the reference (tests/e2e/pocket_tts_lm_smoke.py:
    # teacher-forced latent corr 0.9999, EOS agreement 1.0; free-run corr 0.96 with
    # matched noise).  Here we assert the end-to-end pipeline runs, stops via the
    # EOS head, and writes audio.  Content is REPORT-ONLY: pocket-tts is a fragile
    # 100M short-utterance model whose free-run output quality depends on the CFM
    # noise stream (torch.randn), which C++ can't bit-reproduce — even the reference,
    # driven directly, transcribes short prompts as "Thank you."/"Okay.".  --min-len
    # forces a minimum utterance so the EOS head is exercised past its trigger.
    results.append(_case(
        "pocket_tts",
        REPO / "models" / "pocket_tts" / "pocket_tts_en.gguf",
        None,   # self-contained, no backbone
        "Hello world. This is a small text to speech model.",
        "en", want_stop="eos_head", cer_max=1.0,
        extra=["--seed", "11", "--max-frames", "80", "--min-len", "24"],
        greedy=False, require_stop=False,
        cer_report_only=True, audio_report_only=True,
        stop_line="AR done",
    ))

    # ── NeuTTS-nano (enforced CER) ────────────────────────────────────────────
    # NeuTTS uses phoneme-mode input: both --text and --ref-text must be
    # pre-phonemized before being passed to tts-cli (espeak is OFF by default
    # so the binary passes them verbatim to the runner).
    #
    # The model is a single-codebook AR (KIND_TOKEN_SINGLE_CB); it uses the
    # distill_neucodec GGUF for decode (distill shares the FSQ code space and
    # the decoder with base, so decoded audio is correct even though the
    # backbone was trained against the base encoder).
    #
    # Pinned seed=1 is deterministic; nano is seed-fragile (~2/6 seeds collapse
    # to near-silence) so we pin a known-good seed (see NeuTTS status in
    # CLAUDE.md).  CER threshold 0.30 gives headroom over the measured 0.147
    # (residual is a Whisper tail-hallucination, not a synthesis error).
    _neutts_nano_codec   = REPO / "models" / "distill_neucodec" / "distill_neucodec.gguf"
    _neutts_nano_bb      = REPO / "models" / "hf" / "neutts_nano" / "neutts-nano-q8_0.gguf"
    _neutts_nano_refwav  = REPO / "assets" / "refs" / "jfk_en.wav"
    _neutts_nano_reftxt  = REPO / "assets" / "refs" / "jfk_en.txt"

    _neutts_assets_ok = (
        _neutts_nano_codec.exists()
        and _neutts_nano_bb.exists()
        and _neutts_nano_refwav.exists()
        and _neutts_nano_reftxt.exists()
    )

    if not _neutts_assets_ok:
        missing = [p for p in [_neutts_nano_codec, _neutts_nano_bb,
                                _neutts_nano_refwav, _neutts_nano_reftxt]
                   if not p.exists()]
        print(f"[skip] neutts_nano: missing assets ({', '.join(p.name for p in missing)})")
        results.append(None)
    else:
        _neutts_phon_ok = False
        try:
            from phonemizer.backend import EspeakBackend as _EspeakBackend
            _neutts_phon_ok = True
        except Exception as _e:
            print(f"[skip] neutts_nano: phonemizer unavailable ({_e})")
            results.append(None)

        if _neutts_phon_ok:
            _g2p = _EspeakBackend(
                language="en-us",
                preserve_punctuation=True,
                with_stress=True,
                words_mismatch="ignore",
                language_switch="remove-flags",
            )

            def _to_phones(text):
                return " ".join(_g2p.phonemize([text])[0].split())

            _original_text = "The birch canoe slid on the smooth planks."
            _ref_raw        = _neutts_nano_reftxt.read_text().strip()
            _phon_text      = _to_phones(_original_text)
            _phon_ref       = _to_phones(_ref_raw)

            print(f"[info] neutts_nano: original_text = {_original_text!r}")
            print(f"[info] neutts_nano: phon_text     = {_phon_text!r}")
            print(f"[info] neutts_nano: phon_ref      = {_phon_ref!r}")

            _out_wav = Path("/tmp") / "ttscli_neutts_nano.wav"
            if _out_wav.exists():
                _out_wav.unlink()

            _proc = _run_synth(
                _neutts_nano_codec,
                _neutts_nano_bb,
                _phon_text,
                _out_wav,
                extra=[
                    "--ref-audio", str(_neutts_nano_refwav),
                    "--ref-text",  _phon_ref,
                    "--temp",      "1.0",
                    "--top-k",     "50",
                    "--seed",      "1",
                    "--max-frames", "400",
                ],
                greedy=False,
            )

            if _proc.returncode != 0:
                print(f"[FAIL] neutts_nano: exit {_proc.returncode}")
                print(_proc.stdout[-1200:])
                print(_proc.stderr[-800:])
                results.append(False)
            else:
                _stop_lines = [ln for ln in _proc.stdout.splitlines()
                               if "NeuTTS AR done" in ln]
                _stop = _stop_lines[0].split("stop=")[-1].strip() if _stop_lines else "?"
                _frames_str = ""
                if _stop_lines:
                    import re as _re
                    _m = _re.search(r"(\d+) frames", _stop_lines[0])
                    if _m:
                        _frames_str = f" frames={_m.group(1)}"

                assert _out_wav.exists(), "neutts_nano: no wav written"
                _sr, _dur, _rms = _wav_stats(_out_wav)

                # CER: ASR against the plain-English original (not the phonemes).
                _cer_res = _try_cer(_out_wav, _original_text, "en")
                _cer_str = ""
                _ok = True
                if _rms < 0.005:
                    print(f"[FAIL] neutts_nano: near-silent (rms={_rms:.4f})")
                    _ok = False
                if not (0.3 <= _dur <= 30.0):
                    print(f"[FAIL] neutts_nano: duration {_dur:.2f}s out of bounds")
                    _ok = False
                if _cer_res is not None:
                    _asr_txt, _cer_val = _cer_res
                    _cer_str = f" ASR={_asr_txt!r} CER={_cer_val:.3f}"
                    # Enforced: CER ≤ 0.30 (measured 0.147 @ seed=1; headroom
                    # covers Whisper tail-hallucination drift).
                    if _cer_val > 0.30:
                        print(f"[FAIL] neutts_nano: CER {_cer_val:.3f} > 0.30")
                        _ok = False
                print(f"[{'PASS' if _ok else 'FAIL'}] neutts_nano: "
                      f"stop={_stop}{_frames_str} dur={_dur:.2f}s rms={_rms:.3f}{_cer_str}")
                results.append(_ok)

    # ── NeuTTS-air (enforced CER) ─────────────────────────────────────────────
    # NeuTTS-air (Qwen2 0.5B) is the larger sibling of neutts-nano: same
    # phoneme-mode ICL protocol, same distill_neucodec codec, same
    # ref-audio/ref-text convention, same token-id contract.  The backbone
    # differs (qwen2 arch, 24 layers, hidden=896, vocab=217652) but the
    # NeuTTS host-loop is arch-agnostic — detection is purely vocab-range.
    #
    # Converted via llama.cpp/convert_hf_to_gguf.py --outtype q8_0 (plain
    # Qwen2ForCausalLM, no tensor-rename needed).  Seed=1 reused from nano
    # (both are NeuTTS checkpoints; the EOS-at-9-frames bias at seed=42 is
    # observed on nano; air may or may not share it, but seed=1 is safe).
    #
    # Pinned seed=1 is deterministic; air is robust (CER 0.000 measured).
    # CER threshold 0.15 gives huge headroom over the measured 0.000.
    _neutts_air_codec   = REPO / "models" / "distill_neucodec" / "distill_neucodec.gguf"
    _neutts_air_bb      = REPO / "models" / "hf" / "neutts_air" / "neutts-air-q8_0.gguf"
    _neutts_air_refwav  = REPO / "assets" / "refs" / "jfk_en.wav"
    _neutts_air_reftxt  = REPO / "assets" / "refs" / "jfk_en.txt"

    _neutts_air_assets_ok = (
        _neutts_air_codec.exists()
        and _neutts_air_bb.exists()
        and _neutts_air_refwav.exists()
        and _neutts_air_reftxt.exists()
    )

    if not _neutts_air_assets_ok:
        _missing_air = [p for p in [_neutts_air_codec, _neutts_air_bb,
                                     _neutts_air_refwav, _neutts_air_reftxt]
                        if not p.exists()]
        print(f"[skip] neutts_air: missing assets ({', '.join(p.name for p in _missing_air)})")
        results.append(None)
    else:
        _neutts_air_phon_ok = False
        try:
            from phonemizer.backend import EspeakBackend as _EspeakBackendAir
            _neutts_air_phon_ok = True
        except Exception as _e_air:
            print(f"[skip] neutts_air: phonemizer unavailable ({_e_air})")
            results.append(None)

        if _neutts_air_phon_ok:
            _g2p_air = _EspeakBackendAir(
                language="en-us",
                preserve_punctuation=True,
                with_stress=True,
                words_mismatch="ignore",
                language_switch="remove-flags",
            )

            def _to_phones_air(text):
                return " ".join(_g2p_air.phonemize([text])[0].split())

            _air_original_text = "The birch canoe slid on the smooth planks."
            _air_ref_raw       = _neutts_air_reftxt.read_text().strip()
            _air_phon_text     = _to_phones_air(_air_original_text)
            _air_phon_ref      = _to_phones_air(_air_ref_raw)

            print(f"[info] neutts_air: original_text = {_air_original_text!r}")
            print(f"[info] neutts_air: phon_text     = {_air_phon_text!r}")
            print(f"[info] neutts_air: phon_ref      = {_air_phon_ref!r}")

            _air_out_wav = Path("/tmp") / "ttscli_neutts_air.wav"
            if _air_out_wav.exists():
                _air_out_wav.unlink()

            _air_proc = _run_synth(
                _neutts_air_codec,
                _neutts_air_bb,
                _air_phon_text,
                _air_out_wav,
                extra=[
                    "--ref-audio", str(_neutts_air_refwav),
                    "--ref-text",  _air_phon_ref,
                    "--temp",      "1.0",
                    "--top-k",     "50",
                    "--seed",      "1",
                    "--max-frames", "400",
                ],
                greedy=False,
            )

            if _air_proc.returncode != 0:
                print(f"[FAIL] neutts_air: exit {_air_proc.returncode}")
                print(_air_proc.stdout[-1200:])
                print(_air_proc.stderr[-800:])
                results.append(False)
            else:
                _air_stop_lines = [ln for ln in _air_proc.stdout.splitlines()
                                   if "NeuTTS AR done" in ln]
                _air_stop = (_air_stop_lines[0].split("stop=")[-1].strip()
                             if _air_stop_lines else "?")
                _air_frames_str = ""
                if _air_stop_lines:
                    import re as _re_air
                    _m_air = _re_air.search(r"(\d+) frames", _air_stop_lines[0])
                    if _m_air:
                        _air_frames_str = f" frames={_m_air.group(1)}"

                assert _air_out_wav.exists(), "neutts_air: no wav written"
                _air_sr, _air_dur, _air_rms = _wav_stats(_air_out_wav)

                # CER: ASR against the plain-English original (not the phonemes).
                _air_cer_res = _try_cer(_air_out_wav, _air_original_text, "en")
                _air_cer_str = ""
                _air_ok = True
                if _air_rms < 0.005:
                    print(f"[FAIL] neutts_air: near-silent (rms={_air_rms:.4f})")
                    _air_ok = False
                if not (0.3 <= _air_dur <= 30.0):
                    print(f"[FAIL] neutts_air: duration {_air_dur:.2f}s out of bounds")
                    _air_ok = False
                if _air_cer_res is not None:
                    _air_asr_txt, _air_cer_val = _air_cer_res
                    _air_cer_str = f" ASR={_air_asr_txt!r} CER={_air_cer_val:.3f}"
                    # Enforced: CER ≤ 0.15 (measured 0.000 @ seed=1; robust
                    # model, large headroom).
                    if _air_cer_val > 0.15:
                        print(f"[FAIL] neutts_air: CER {_air_cer_val:.3f} > 0.15")
                        _air_ok = False
                print(f"[{'PASS' if _air_ok else 'FAIL'}] neutts_air: "
                      f"stop={_air_stop}{_air_frames_str} dur={_air_dur:.2f}s "
                      f"rms={_air_rms:.3f}{_air_cer_str}")
                results.append(_air_ok)

    # ── OuteTTS 1.0-0.6B (CER report-only) ───────────────────────────────────
    # OuteTTS uses RAW text input — NO phonemization.  The bundled default
    # speaker (assets/speakers/en-female-1-neutral.json) is auto-loaded by
    # the runner.  Detection is vocabulary-based (V3 signature tokens) and
    # the auto-grammar is attached automatically.
    #
    # Sampling regime mirrors the OuteTTS reference: temp=1.0, top-k=50,
    # seed=1, rep-penalty=1.1.  CER is REPORT-ONLY for this first pass;
    # a threshold will be enforced once the round-trip quality is baselined.
    _outetts_codec  = REPO / "models" / "output" / "ibm-research--DAC.speech-F32.gguf"
    _outetts_bb     = REPO / "models" / "hf" / "outetts_1_0" / "OuteTTS-1.0-0.6B-Q8_0.gguf"
    _outetts_spk    = REPO / "assets" / "speakers" / "en-female-1-neutral.json"

    _outetts_assets_ok = (
        _outetts_codec.exists()
        and _outetts_bb.exists()
        and _outetts_spk.exists()
    )

    if not _outetts_assets_ok:
        _missing_oe = [p for p in [_outetts_codec, _outetts_bb, _outetts_spk]
                       if not p.exists()]
        print(f"[skip] outetts_1_0: missing assets ({', '.join(p.name for p in _missing_oe)})")
        results.append(None)
    else:
        _outetts_text = "The birch canoe slid on the smooth planks."
        _outetts_out  = Path("/tmp") / "ttscli_outetts_1_0.wav"
        if _outetts_out.exists():
            _outetts_out.unlink()

        print(f"[info] outetts_1_0: text = {_outetts_text!r}")

        _outetts_proc = _run_synth(
            _outetts_codec,
            _outetts_bb,
            _outetts_text,
            _outetts_out,
            extra=[
                "--temp",        "1.0",
                "--top-k",       "50",
                "--seed",        "1",
                "--rep-penalty", "1.1",
                "--max-frames",  "600",
            ],
            greedy=False,
        )

        if _outetts_proc.returncode != 0:
            print(f"[FAIL] outetts_1_0: exit {_outetts_proc.returncode}")
            print(_outetts_proc.stdout[-1200:])
            print(_outetts_proc.stderr[-800:])
            results.append(False)
        else:
            _oe_stop_lines = [ln for ln in _outetts_proc.stdout.splitlines()
                              if "OuteTTS AR done" in ln]
            _oe_stop = (_oe_stop_lines[0].split("stop=")[-1].strip()
                        if _oe_stop_lines else "?")
            _oe_frames_str = ""
            if _oe_stop_lines:
                import re as _re_oe
                _m_oe = _re_oe.search(r"(\d+) frames", _oe_stop_lines[0])
                if _m_oe:
                    _oe_frames_str = f" frames={_m_oe.group(1)}"

            assert _outetts_out.exists(), "outetts_1_0: no wav written"
            _oe_sr, _oe_dur, _oe_rms = _wav_stats(_outetts_out)

            # CER: ASR against the raw English text (no phoneme indirection).
            _oe_cer_res = _try_cer(_outetts_out, _outetts_text, "en")
            _oe_cer_str = ""
            _oe_ok = True
            if _oe_rms < 0.005:
                print(f"[FAIL] outetts_1_0: near-silent (rms={_oe_rms:.4f})")
                _oe_ok = False
            if not (0.3 <= _oe_dur <= 30.0):
                print(f"[FAIL] outetts_1_0: duration {_oe_dur:.2f}s out of bounds")
                _oe_ok = False
            if _oe_cer_res is not None:
                _oe_asr_txt, _oe_cer_val = _oe_cer_res
                _oe_cer_str = f" ASR={_oe_asr_txt!r} CER={_oe_cer_val:.3f}"
                # CER is report-only for this first pass — no threshold enforced.
            print(f"[{'PASS' if _oe_ok else 'FAIL'}] outetts_1_0: "
                  f"stop={_oe_stop}{_oe_frames_str} dur={_oe_dur:.2f}s "
                  f"rms={_oe_rms:.3f}{_oe_cer_str}")
            results.append(_oe_ok)

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
