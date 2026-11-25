#!/usr/bin/env python3
"""E2E smoke for MOSS-TTS-Nano-100M: voice-clone synthesis through codec_cpp.

Assets (missing → skip):
  models/moss_tts_nano/moss_tts_nano_codec.gguf   codec + LM adaptor
      scripts/convert-to-gguf.py --checkpoint-path <MOSS-Audio-Tokenizer-Nano>
          --lm-source <MOSS-TTS-Nano-100M> --outtype F16 --lm-outtype BF16
  models/moss_tts_nano/backbone-bf16.gguf         the global transformer (starcoder2)
      scripts/convert-backbone-to-gguf.py --model-id <MOSS-TTS-Nano-100M> --outtype bf16

Checks:
  1. The voice: codec_speaker_from_audio gives ref_codes I32 [16, floor(n48 / 3840)].
  2. Synthesis (codec_cpp.TTS, the bundled llama.cpp backbone, fixed seeds) of
     an English and a Chinese sentence in the cloned JFK voice: stops on the
     text slot's audio end, 48 kHz, plausible duration, not silent; the Whisper
     CER (when transformers + whisper are available) is reported (sampled
     output varies with the seed and the SIMD path).
  3. With MOSS_TTS_NANO_SRC=<the HF checkpoint dir> and torch: teacher-forced
     frame 0 of the depth decoder on PyTorch's backbone hidden (all 17
     argmaxes equal, logits close) and the voice codes vs the reference's
     batch_encode (MOSS_AUDIO_TOKENIZER_SRC=<the codec dir>; >= 90% equal with
     models/moss_tts_nano/moss_tts_nano_codec-f32.gguf, reported otherwise).

Run from the repo root: .venv/bin/python tests/e2e/moss_tts_nano_smoke.py
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codec_cpp  # noqa: E402
from _common import MODEL_SAMPLING, asr_cer, read_wav_mono_f32  # noqa: E402

M = REPO / "models" / "moss_tts_nano"
CODEC = M / "moss_tts_nano_codec.gguf"
BACKBONE = M / "backbone-bf16.gguf"
CODEC_F32 = M / "moss_tts_nano_codec-f32.gguf"   # optional: F32 codec part (--outtype F32) for the voice parity
REF_WAV = REPO / "assets" / "refs" / "jfk_en.wav"

CASES = [   # name, text, lang, seed, CER flagged above (sampled; the reference varies as much from seed to seed)
    ("en", "The birch canoe slid on the smooth planks.", "en", 3, 0.35),
    ("zh", "你也可以咨询除贵国以外其他政府的建议。", "zh", 3, 0.35),
]


def check_voice(model) -> bool:
    x, sr = read_wav_mono_f32(REF_WAV)
    spk = codec_cpp.Context(model).speaker_from_audio(x, sr)
    rc = spk.slot("ref_codes")
    n48 = (len(x) * 48000 + sr - 1) // sr
    want_t = n48 // 3840
    ok = rc is not None and rc.dtype == np.int32 and rc.size == 16 * want_t and \
        int(rc.min()) >= 0 and int(rc.max()) < 1024
    print(f"[{'PASS' if ok else 'FAIL'}] voice: ref_codes {None if rc is None else rc.shape} (want 16 x {want_t})")
    return ok


def check_synth() -> bool:
    ok_all = True
    with codec_cpp.TTS(BACKBONE, CODEC) as tts:
        for name, text, lang, seed, cer_max in CASES:
            audio, sr = tts.synthesize(text, ref_audio=REF_WAV, seed=seed, **MODEL_SAMPLING["moss_tts_nano"])
            stop = tts.last_result["stop_reason"]
            a = audio if audio.ndim == 1 else audio.mean(axis=1)
            dur, rms = len(a) / sr, float(np.sqrt((a ** 2).mean()))
            ok = stop == "audio_end" and sr == 48000 and 0.8 <= dur <= 15.0 and rms > 0.005
            msg = f"stop={stop} sr={sr} dur={dur:.2f}s rms={rms:.3f}"
            out = Path("/tmp") / f"moss_tts_nano_{name}.wav"
            import wave
            with wave.open(str(out), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
            res = asr_cer(out, text, lang)
            if res is not None:
                asr, cer = res
                msg += f" ASR='{asr}' CER={cer:.3f}" + ("" if cer <= cer_max else " (high, reported)")
            print(f"[{'PASS' if ok else 'FAIL'}] synth_{name}: {msg}")
            ok_all &= ok
    return ok_all


def check_parity(model) -> bool | None:
    if CODEC_F32.exists():
        model = codec_cpp.Model(CODEC_F32, n_threads=8)
    src = os.environ.get("MOSS_TTS_NANO_SRC")
    if not src:
        print("[skip] parity: set MOSS_TTS_NANO_SRC to the HF checkpoint dir")
        return None
    try:
        import torch
        import torchaudio
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as e:  # noqa: BLE001
        print(f"[skip] parity: {e}")
        return None
    import soundfile as sf
    torchaudio.load = lambda p, *a, **k: (lambda x: (torch.from_numpy(x[0].T.copy()), x[1]))(
        sf.read(str(p), dtype="float32", always_2d=True))
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    cfg.attn_implementation = "sdpa"
    cfg.local_transformer_attn_implementation = "sdpa"
    ref = AutoModelForCausalLM.from_pretrained(src, config=cfg, trust_remote_code=True, dtype=torch.float32).eval()
    tok = ref._load_text_tokenizer(text_tokenizer_path=src)
    ok = True
    codec_src = os.environ.get("MOSS_AUDIO_TOKENIZER_SRC")
    x, sr = read_wav_mono_f32(REF_WAV)
    if codec_src:
        at = ref._load_audio_tokenizer(audio_tokenizer_pretrained_name_or_path=codec_src, device="cpu")
        wav, s = ref._load_reference_audio(REF_WAV, ref._resolve_audio_tokenizer_sample_rate(at),
                                           ref._resolve_audio_tokenizer_channels(at))
        with torch.no_grad():
            codes = ref._normalize_audio_codes(ref._call_audio_encode(at, wav, s)).numpy()
        got = codec_cpp.Context(model).speaker_from_audio(x, sr).slot("ref_codes").reshape(-1, 16)
        n = min(len(got), len(codes))
        frac = float((got[:n] == codes[:n]).mean())
        # Exact up to deep-RVQ near ties with an F32 codec part; another
        # encoder precision only reports.
        strict = model is not None and CODEC_F32.exists()
        good = len(got) == len(codes) and (frac >= 0.90 or not strict)
        print(f"[{'PASS' if good else 'FAIL'}] parity_voice: {len(got)} vs {len(codes)} frames, {frac:.1%} codes equal"
              f"{'' if strict else ' (no F32 codec GGUF: reported only)'}")
        ok &= good
    else:
        codes = np.zeros((20, 16), np.int64)   # any reference rows exercise the prompt
    ids, mask = ref.build_inference_input_ids(text=CASES[0][1], text_tokenizer=tok, mode="voice_clone",
                                              prompt_audio_codes=torch.from_numpy(codes).long(), device="cpu")
    with torch.no_grad():
        h = ref.transformer(inputs_embeds=ref._build_inputs_embeds(ids), attention_mask=mask).last_hidden_state[:, -1]
        lt = ref.local_transformer
        seq = h.unsqueeze(1)
        st = codec_cpp.LM(model).state()
        st.step_begin(h[0].numpy())
        worst, agree = 0.0, 0
        for k in range(17):   # teacher-forced on PyTorch's picks
            out = lt(inputs_embeds=seq, attention_mask=torch.ones(seq.shape[:2], dtype=torch.bool)).last_hidden_state[:, -1]
            _, g = st.step_logits()
            if k == 0:   # the text slot: assistant slot vs audio end
                pair = [cfg.audio_assistant_slot_token_id, cfg.audio_end_token_id]
                r, g = ref.text_lm_head(out)[0, pair].numpy(), g[pair]
                code = cfg.audio_assistant_slot_token_id
                nxt = ref.transformer.wte(torch.tensor([code]))
            else:
                r = ref.audio_lm_heads[k - 1](out)[0].numpy()
                code = int(r.argmax())
                nxt = ref.audio_embeddings[k - 1](torch.tensor([code]))
            agree += int(np.argmax(g) == np.argmax(r))
            worst = max(worst, float(np.abs(g - r).max()))
            st.step_push_code(code)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
    good = agree == 17 and worst < 0.25
    print(f"[{'PASS' if good else 'FAIL'}] parity_depth: {agree}/17 argmax equal, logits max|d| {worst:.3f}")
    return ok and good


def main() -> int:
    if not CODEC.exists() or not BACKBONE.exists():
        print(f"[skip] moss_tts_nano: missing {CODEC.name} / {BACKBONE.name} in {M}")
        return 0
    model = codec_cpp.Model(CODEC, n_threads=8)
    results = [check_voice(model), check_synth(), check_parity(model)]
    ran = [r for r in results if r is not None]
    print("OK" if all(ran) else "FAILED")
    return 0 if all(ran) else 1


if __name__ == "__main__":
    sys.exit(main())
