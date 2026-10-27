# Pocket-TTS

[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts): a 100M
continuous-latent AR model (FlowLM) over a small 24 kHz Mimi variant. Each
12.5 Hz frame is a 32-d latent sampled by a one-step LSD flow head on the
transformer's hidden state; an EOS head ends the utterance.

## Conversion

```bash
# Mimi variant + FlowLM adaptor (input projection, voice prefix, EOS + flow heads, tokenizer)
python scripts/convert-to-gguf.py --checkpoint-path models/pocket_tts \
    --model-type pocket_tts --output models/pocket_tts/pocket_tts_en.gguf

# FlowLM transformer → stock starcoder2 backbone GGUF, with the SentencePiece
# text tokenizer as its llama.cpp vocab (UGM; byte fallback needs
# patches/ugm-byte-fallback-llamacpp.patch). Older backbones without the vocab
# fall back to the codec GGUF's tokenizer copy (deprecated).
python scripts/convert-backbone-to-gguf.py --model-id models/pocket_tts \
    --output models/pocket_tts/backbone-f16.gguf --outtype f16
```

`models/pocket_tts` holds `model.safetensors` + `tokenizer.model` (the HF
repo's `languages/<lang>/`). The FlowLM transformer is pre-LN affine LayerNorm,
non-gated GELU FFN, no linear biases, interleaved RoPE (θ 10000), final
`out_norm`: starcoder2 with zero biases. starcoder2 ropes NEOX, so the
backbone converter reorders each head's q/k rows from interleaved to
half-split. Its `token_embd` is the text LUT (4001 rows) under
`tokenizer.ggml.model = none`; codec_lm tokenizes (SentencePiece unigram).
The checkpoint is BF16: `--outtype f16` / `bf16`; `f32` is capped at bf16.

## Runtime

```bash
tts-cli synthesize --backbone backbone-f16.gguf --model pocket_tts_en.gguf \
    --text "Hello world. This is a small text to speech model." --output out.wav \
    --ref-audio voice.wav   # or --speaker-file voice.gguf
```

A voice is required: without one the format call fails ("Pocket-TTS needs a
voice") and `caps.needs_voice` is set, since the unconditioned model's EOS head
may never fire.

The codec_audio_lm family (`common/audio_lm_family_flow.cpp`) follows
`pocket_tts` `TTSModel`; any host loop over `include/codec_audio_lm.h` runs
it (tts-cli, `codec_cpp.TTS` / `AudioLM`, llama.rn):

- `prepare_text_prompt` (strip, capitalize, trailing period), then the backbone
  sequence `[bos_before_voice + speaker_proj(voice latents) | text tokens |
  per-frame input embeddings]` (the reference fills its KV cache in that
  order: `get_state_for_audio_prompt`, then the text, then the frames);
- per frame: the frame input (`input_linear` of `bos_emb` ending the prompt,
  then of the previous latent, fed back as an embedding) → backbone hidden →
  the EOS head (logit > −4) and the LSD flow head (latent from noise
  ~ N(0, temperature) seeded by `--seed`; the temperature is the codec
  GGUF's `codec.lm.temperature`, 0.7 as upstream, a model setting changed
  with `--override-kv codec.lm.temperature=float:...`, not token sampling);
- after EOS, `frames_after_eos` more frames (word-count guess + 2), dropping
  the frame it stops on; `--min-len` ignores EOS before that frame.

A voice is a codec speaker (docs/codec_speaker.md) with one slot, `latents`
F32 `[n_frames, 32]`: the Mimi encoder latent of the reference, which
`codec_speaker_from_audio` computes from the first 30 s (resampled to 24 kHz
in-graph), as the reference server does (`get_state_for_audio_prompt(...,
truncate=True)` → `_encode_audio`). `--save-speaker` / `--speaker-file` store
and reuse it; files carry `pocket_mimi.spk.encoder_hash` (GGUFs converted
before it have none, so their speakers are not hash-checked).

Free-run quality depends on the noise stream (the reference too); short
prompts can stop after a frame or two. Parity when the backbone move landed:
llama.cpp hidden vs the reference min cos 0.99998, teacher-forced latent corr
0.999999, EOS agreement 1.0; `tests/e2e/pocket_tts_decode_smoke.py` checks
Mimi.
