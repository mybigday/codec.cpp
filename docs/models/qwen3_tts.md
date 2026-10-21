# Qwen3-TTS (12 Hz)

[Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) speech LMs:
a Qwen3 talker (the backbone, run by llama.cpp as plain `qwen3`) that
emits codebook 0, a 5-layer depth decoder (codec_lm `residual_depth_ar`)
for codebooks 1..15, and the Qwen3-TTS-Tokenizer-12Hz codec.

| Model | Speaker | Instruct |
|---|---|---|
| 0.6B Base | cloned: `--ref-audio` (+ `--ref-text` for the in-context prompt) | no |
| 0.6B / 1.7B CustomVoice | builtin voices (`--voice`) | 1.7B only |

The codec GGUF holds the codec plus that model's adaptor (talker codec
embeddings, text projection, depth decoder, speaker encoder or voices), so
it is converted once per model:

```bash
python scripts/convert-to-gguf.py --model-id Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --lm-source Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --outtype F16 --quantization Q8_0 --lm-outtype F16 -o codec-q8_0.gguf
python scripts/convert-backbone-to-gguf.py --model-id Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --output qwen3-tts-0.6b-customvoice-f16.gguf --outtype f16
```

## Talker prompt

`audio_lm_build_talker_prefix` assembles upstream's two-lane prefix
(`Qwen3TTSForConditionalGeneration.generate`): projected text tokens summed
with codec control tags, `[nothink, think_bos, think_eos, SPEAKER, pad, bos]`
for the auto language. The speaker row is the Base x-vector, or for
CustomVoice `codec_embedding[spk_id]`.

## Sampling

Defaults follow upstream's `generation_config.json`: temperature 0.9 and
top-k 50 for every codebook, top-p 1.0, and `repetition_penalty` 1.05 on
codebook 0 only (over the generated frames, as the talker's
`generate`); the depth codebooks take no penalty. Without the penalty an
ICL clone on a quantized talker (Q4_K_M) can loop and never reach the end
code. `--rep-penalty` / `repetition_penalty` overrides it.

## Cloned voices (Base)

A Base voice is a codec speaker (docs/codec_speaker.md) made by
`codec_speaker_from_audio` in one graph, as `create_voice_clone_prompt` does:
the reference resampled to 24 kHz gives `xvector` F32 [hidden] (the bundled
ECAPA-TDNN, `extract_speaker_embedding`) and `ref_codes` I32 [16, T] (the
codec's own encoder, `speech_tokenizer.encode`); `--ref-text` is stored as
the speaker's ref_text. `--save-speaker` / `--speaker-file` keep and reuse it.

- No ref_text: `x_vector_only_mode` — the x-vector is the SPEAKER row.
- With ref_text: upstream's default ICL mode (`generate_icl_prompt`, streaming
  text). The text lane is `ref_text` followed by the text; after `codec_bos`
  the next T rows add the reference frames' summed codebook embeddings to it;
  the rest of the text then rides one token per step, `tts_eos` after it and
  `tts_pad` from then on. The decode prepends `ref_codes` to the generated
  codes and cuts their share of the audio (`generate_voice_clone`).

Against `qwen_tts` on a 5.5 s clip: x-vector cosine 1.000000, ref_codes 98 %
(codebook 0 100 %); output speaker similarity (VoiceEncoder, 3 seeds) 0.80
x-vector-only / 0.83 ICL, the reference 0.79 / 0.83.

## CustomVoice voices

The converter writes `talker_config.spk_id` as builtin speakers
(docs/codec_speaker.md): `spk.<name>.spk_id` I32 [1], plus
`spk.<name>.language_id` I32 [1] for the dialect voices, whose language tag
upstream uses when the language is auto (control stream
`[think, think_bos, LANG, think_eos, SPEAKER, pad, bos]`):

| Voice | Native language |
|---|---|
| `vivian` (default), `serena`, `uncle_fu` | Chinese |
| `dylan` | Chinese, Beijing dialect |
| `eric` | Chinese, Sichuan dialect |
| `ryan`, `aiden` | English |
| `ono_anna` | Japanese |
| `sohee` | Korean |

`vivian` is upstream's README example; `generate_custom_voice` itself has
no default. `--instruct` (e.g. `用特别愤怒的语气说`) is prepended as a
`<|im_start|>user` turn on 1.7B (`codec.lm.qwen3tts.takes_instruct`);
0.6B ignores it, as upstream does.

```bash
tts-cli synthesize --backbone qwen3-tts-1.7b-customvoice-q8_0.gguf --model codec-q8_0.gguf \
    --voice vivian --instruct "用特别愤怒的语气说" \
    --text "其实我真的有发现，我是一个特别善于观察别人情绪的人。" --output out.wav
```

1.7B differs from 0.6B in widths only: the talker (2048) is wider than the
depth decoder (1024), so the depth inputs (talker-width embeddings) go
through `small_to_mtp_projection` (`lm.depth.in_proj`).
