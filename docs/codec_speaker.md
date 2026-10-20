# Codec speakers (`codec_speaker`)

Some decoders are conditioned on a reference voice, not only on tokens
(Chatterbox-S3G: prompt tokens + prompt mel + x-vector). `codec_speaker` is
the cross-model handle for that conditioning: a set of named tensors
("slots") whose layout each architecture defines and validates. The LM-side
counterpart for prompt embeddings is `codec_lm_speaker_*`
([audio_speaker_encoders.md](audio_speaker_encoders.md)).

## API (`include/codec.h`)

| Call | Ownership |
|---|---|
| `codec_model_n_speakers` / `codec_model_speaker_name` / `codec_get_speaker(model, name)` | builtin speakers; owned by the model, `name = NULL` → default |
| `codec_model_can_speaker_from_audio` / `codec_speaker_from_audio(ctx, audio)` | derived from reference audio (any rate); caller frees |
| `codec_speaker_save(model, spk, name, path)` / `codec_speaker_load(model, path, name)` | speaker files; loaded speakers are caller-owned |
| `codec_speaker_file_open` / `codec_speaker_file_n_speakers` / `codec_speaker_file_speaker_name` / `codec_speaker_file_get_speaker(file, name)` | list a speaker file (it may hold several); speakers owned by the file handle, `name = NULL` → the file's default |
| `codec_speaker_get_slot` / `codec_speaker_ref_text` / `codec_speaker_set_ref_text` | model-specific data for host runners |
| `codec_decode_params.speaker` | decode with this speaker (`NULL` → default) |

Every decode/load checks the speaker against the model: same
`general.architecture`, same `<arch>.spk.encoder_hash`, required slots with
valid shapes (`codec_model_vtable::speaker_validate`). Mismatches are refused,
never coerced: a speaker made by a different encoder lives in a different
embedding space.

## GGUF format

Builtin speakers (model GGUF) and speaker files share one layout:

```
tensors:   spk.<name>.<slot>                     F32 or I32, name ∈ [a-z0-9_]{1,32}
metadata:  <arch>.spk.default        = <name>
           <arch>.spk.encoder_hash   = sha256 of the reference-audio → speaker pipeline
           <arch>.spk.<name>.ref_text = transcript (optional)
speaker files additionally:
           general.type = "speaker", general.architecture = <arch>,
           <arch>.spk.format_version = 1
```

The encoder hash is computed by the converter from the *source* weights of
every encoder `codec_speaker_from_audio` runs, plus a string describing the
preprocessing (`scripts/converters/speaker.py::encoder_hash`), so F16/F32
conversions of the same checkpoint accept each other's speakers.

GGUFs converted before this format (`chatterbox_s3g.has_builtin_conditioning`
+ `s3g.cond.*`) load their single voice as speaker `default`, bit-exact. They
bundle no speaker encoders and carry no encoder hash, so they cannot derive a
speaker from audio (tts-cli `--ref-audio` then only conditions the LM side and
says so), and speaker files are not interchangeable between them and
re-converted GGUFs. Re-convert to get voice cloning.

## Chatterbox-S3G slots

| Slot | Shape | From audio (`S3Gen.embed_ref`) |
|---|---|---|
| `prompt_token` | I32 [T] | S3TokenizerV2 at 16 kHz (`s3t.*` bundled) |
| `prompt_feat` | F32 [80, F], F ∈ {2T, 2T+1} | Matcha log-mel at 24 kHz (`s3g.spk.mel_fb`) |
| `embedding` | F32 [192] | CAMPPlus on Kaldi fbank (`s3g.cp.*`, BN folded) |

`codec_speaker_from_audio` is one cached graph per (input length, sample
rate): torchaudio's sinc resampler, the 10 s truncation, both mel front-ends
and the encoders (`src/blocks/`) all run inside it; the DFT/filterbank
front-ends are pinned to the CPU backend for F32 accuracy, the networks
offload. Parity vs PyTorch on a KittenTTS voice clip: tokens 100 % (F32 GGUF;
one FSQ boundary flip in 248 with the default F16 S3T), prompt_feat max |Δ|
4e-5, embedding cosine 0.9999998 (CPU and Vulkan).

Chatterbox codec GGUFs that bundle T3's VoiceEncoder (`speaker.voice_encoder.*`,
`codec.speaker.encoder_arch = chatterbox_voice_encoder`) add T3's
conditioning (`prepare_conditionals`), so a speaker file carries the whole
voice:

| Slot | Shape | From audio |
|---|---|---|
| `t3_speaker_emb` | F32 [256] | VoiceEncoder of the clip at 16 kHz (via 24 kHz), silence-trimmed (`codec.speaker.ve.trim_top_db`) |
| `t3_cond_tokens` | I32 [≤ N] | S3 tokens of the first N / 25 s, N = the builtin cond prompt length (150, Turbo 375) |

Turbo scales the reference to `chatterbox_s3g.spk.ref_lufs` (-27 LUFS) before
all of it. Builtin voices without these slots use the GGUF's builtin T3 conds.

`chatterbox_s3g.spk.lm_encoder = "xvector_sincnet"` (KittenTTS 2) adds:

| Slot | Shape | From audio |
|---|---|---|
| `lm_cond` | F32 [2048] | pyannote XVectorSincNet → L2 norm → `spk_proj` (Linear + LayerNorm), full clip at 16 kHz |
| `ref_tokens` | I32 [N] (optional) | S3 tokens of the full clip (in-context reference prompt) |

## CosyVoice slots

| Slot | Shape | From audio (`CosyVoiceFrontEnd`) |
|---|---|---|
| `prompt_token` | I32 [T] | S3 tokenizer v1 / v2 / v3 at 16 kHz (`s3t.*` bundled) |
| `prompt_feat` | F32 [80, F] | Matcha log-mel of the 16 kHz clip resampled to 22.05 / 24 kHz (`cv.spk.mel_fb`); 24 kHz models keep F = 2T |
| `embedding` | F32 [192] | CAMPPlus on Kaldi fbank (`cv.cp.*`, from `campplus.onnx`) |

SFT voices (`spk2info.pt`) carry only `embedding` (BreezyVoice also ships
`prompt_token` / `prompt_feat`). The transcript (`ref_text`) selects zero-shot
over cross-lingual prompting in the LM; see [cosyvoice.md](cosyvoice.md).

## Token-codec speakers (TTS LM prompts)

Some codecs only decode tokens that a TTS LM samples; their speakers condition
the LM prompt, not the decoder. Such a codec GGUF is published for one TTS
model, so a speaker may hold that backbone's token ids.

| Codec | Slots | Model |
|---|---|---|
| DAC, WavTokenizer | `audio_prompt` I32 [N] (backbone token ids of the reference's audio section) + `ref_text` (text section) | OuteTTS ([outetts.md](models/outetts.md)) |
| NeuCodec, distill NeuCodec | `ref_codes` I32 [T] (codec tokens) + `ref_text` (phonemes) | NeuTTS ([neutts.md](models/neutts.md)) |
| Pocket-Mimi | `latents` F32 [T, 32] (Mimi encoder latent of the first 30 s; the FlowLM applies `speaker_proj`) | Pocket-TTS ([pocket_tts.md](models/pocket_tts.md)) |
| Qwen3-TTS-Tokenizer | `spk_id` I32 [1] (talker codec-vocab row used as the speaker row) + optional `language_id` I32 [1] (dialect tag) | Qwen3-TTS CustomVoice ([qwen3_tts.md](models/qwen3_tts.md)) |
| Qwen3-TTS-Tokenizer + ECAPA-TDNN | `xvector` F32 [hidden] (speaker row) + `ref_codes` I32 [16, T] + `ref_text` (in-context prompt) | Qwen3-TTS Base, cloned |

`scripts/bake-speakers.py` writes them into a copy of a codec GGUF (or a
speaker file). OuteTTS speakers cannot be derived from audio;
`codec_speaker_from_audio` on a distill NeuCodec runs its encoder, and its
`encoder_hash` covers the distill encoder + DistilHuBERT weights. OuteTTS
codec GGUFs hash the interface and backbone id instead, so speaker files do
not cross between models. Qwen3-TTS CustomVoice voices come from the
checkpoint's `talker_config.spk_id` at conversion; their hash covers the
talker codec_embedding table the ids index; Base GGUFs hash the ECAPA weights. Pocket-TTS hashes the Mimi encoder (SEANet, encoder transformer, downsample).

## tts-cli

Decoder speaker precedence: `--speaker-file` > `--ref-audio` (when the codec
can derive one) > `--voice NAME` > model default. With `--speaker-file`,
`--voice` picks a speaker inside the file. `--save-speaker PATH` writes the
speaker in use; `--ref-text` is stored as its `ref_text`.

`tts-cli voices --model CODEC.gguf [--speaker-file PATH]` lists the builtin
speakers (or a file's), `*` marking the default.
