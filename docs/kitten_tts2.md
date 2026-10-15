# KittenTTS 2

[KittenML/kitten-tts-2](https://huggingface.co/KittenML/kitten-tts-2) — Qwen3-1.7B
speech LM with ternary linear weights. It reads text and writes S3 tokens as
ordinary vocab rows (`<|a_0|>` = 151675, 6561 codes), decoded by
Chatterbox-Turbo's MeanFlow S3Gen (2 steps). Weights: Stellon Labs Community
License (`speaker/` pyannote model: MIT).

## Conversion

```bash
# codec side: S3Gen (chatterbox-turbo s3gen_meanflow.safetensors, found next to the
# checkpoint or fetched from the Hub) + speaker encoders + 47 voices + LM contract
python scripts/convert-to-gguf.py --checkpoint-path models/kitten_tts_2 \
    --model-type kitten_tts2 --output models/kitten_tts_2/kitten_tts2.gguf

# backbone: lm/model.safetensors (spk_proj dropped) → bf16, then lossless Q4_0
python scripts/convert-backbone-to-gguf.py --model-id models/kitten_tts_2 \
    --output models/kitten_tts_2/kitten_tts2_backbone-bf16.gguf --outtype bf16
python scripts/utils/ternary_q4_0.py models/kitten_tts_2/kitten_tts2_backbone-bf16.gguf \
    models/kitten_tts_2/kitten_tts2_backbone-q4_0.gguf
```

The repo's `cpp/model-tq2_1.gguf` uses the fork-only ggml type 43 (TQ2_1) and
cannot be loaded. Every 128-wide group is {-s, 0, +s}, so Q4_0 (d = s,
q = 8 ± 1) stores the weights exactly; `ternary_q4_0.py` refuses any tensor that
is not. `llama-quantize` Q4_0 is lossy here (d = max/-8 cannot hit -s).

## Runtime contract (`common/tts_runner.cpp::run_kitten_tts2`)

Detected from the backbone vocab (`<|a_0|>`, `<final_seg>`, `[START]`);
everything else is `codec.lm.kitten.*` metadata in the codec GGUF.

- Prompt (`kittenml/kittentts2/prompt.py`):
  `[START] <ref_text_start> transcript <ref_text_end> <ref_speech_start> ref audio <ref_speech_end> [{emo: 1}] <text_start> text <final_seg> <speech_start>`.
  Position 0 is the speaker's `lm_cond` row instead of the `[START]` embedding.
  Without a transcript (`--ref-audio` without `--ref-text`) the reference block is
  dropped and only `lm_cond` carries the voice.
- `{emo: 1}` is inserted when the text has a known `[emotion]`, `<event>` or
  `(((emphasis)))` tag.
- Sampling (`logits.py`): token-run penalty (grace 10, 1.3/step over the grace)
  → repetition penalty 1.1 over the last 50 generated tokens, exempting
  `<|speech_end|>`, `[STOP]` and silence (4299) → temp → top-k → top-p → min-p.
  Presets `stable` (0.8 / 50 / 0.8) and `expressive` (0.9 / 50 / 0.9), `--preset`.
- Stops on `<|speech_end|>` or `[STOP]`; non-audio tokens are dropped; 3 silence
  tokens are appended before S3Gen.
- Text > 380 chars is split on sentences (min 130), each chunk generated with a
  fresh KV cache and joined with trimmed edges, 8 ms fades and a 0.16 s gap.
- Not ported: the `kitten_text_processing` written→spoken normalizer (pass text
  with numbers/dates already spelled out) and Whisper auto-transcription of the
  clone reference.

## Parity

- Prompt token ids identical to PyTorch.
- Greedy LM tokens: bf16 backbone matches transformers **fp32** for all 92
  tokens of the test sentence (transformers bf16 itself diverges from fp32 at
  token 10). The Q4_0 backbone quantizes activations to Q8_0 in its matmuls and
  diverges at the same point as transformers bf16.
- Speaker from audio: S3G slots as in [codec_speaker.md](codec_speaker.md);
  `lm_cond` cosine 1.0000 vs PyTorch fp32 (0.997 vs the shipped voice, which was
  computed in bf16); `ref_tokens` 94 % equal to the shipped ones (they were
  tokenized after a librosa/soxr resample, here torchaudio's sinc).
- E2E (`tests/e2e/ttscli_synthesize_smoke.py`): builtin voice and clone, Whisper
  CER 0 (Q4_0 backbone); VoiceEncoder cosine to the reference (bf16 runs) 0.92 (builtin Bruno), 0.88 / 0.90
  (clone with / without transcript).
