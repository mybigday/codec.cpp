# Audio speaker encoders — codec.cpp framework state

The `codec_lm_speaker_encode*` C API exposes per-model "ref audio →
conditioning embedding" pipelines through one generic interface.  Each
codec_model loaded from `.gguf` declares its requirements (input
flags, working sample rate, output shape) via `codec_lm_speaker_info`;
the runtime dispatches on a `codec.speaker.encoder_arch` GGUF metadata
string to the per-arch implementation.

This document tracks which models have which level of support.

## Boundary

| Layer | Owns |
|---|---|
| **codec.cpp** | audio encoder weights + forward (PCM → conditioning embedding) |
| **llama.cpp** | LM backbone, tokenizer, prompt assembly, AR sampling |
| **Application (llama.rn / examples/tts.py / …)** | wiring: load audio → call `codec_lm_speaker_encode` → hand resulting `(n_rows, hidden_dim)` embedding to llama.cpp as part of `inputs_embeds` |

The output of `codec_lm_speaker_encode` is a plain F32 matrix of
shape `(info.n_rows, info.hidden_dim)`, ready to be concatenated /
spliced into the backbone's input embedding stream — exactly the same
contract MTMD vision tokens already satisfy on the llama.cpp side.

## Public API (`include/codec_lm.h`)

```c
const struct codec_lm_speaker_info * codec_lm_speaker_get_info(const codec_lm *);

codec_status codec_lm_speaker_encode(
    codec_lm *,
    const codec_audio * ref_pcm,           // OPTIONAL per info
    const int32_t     * ref_speech_tokens, // OPTIONAL per info
    int32_t             n_ref_speech_tokens,
    const float       * emotion,           // NULL = info.emotion_default
    float             * out,               // [n_rows × hidden_dim]
    int32_t             out_n_elems);

codec_status codec_lm_speaker_encode_from_embedding(
    codec_lm *,
    const float       * speaker_emb,       // pre-computed (skip audio encoder)
    int32_t             speaker_emb_dim,   // must match info.speaker_emb_dim
    const int32_t     * ref_speech_tokens,
    int32_t             n_ref_speech_tokens,
    const float       * emotion,
    float             * out,
    int32_t             out_n_elems);
```

## Per-model status

### 1. Chatterbox (`encoder_arch = "chatterbox_voice_encoder"`) — **SHIPPED**

- **Audio encoder**: 3-layer LSTM voice encoder + linear projection → 256-d
  L2-normalized speaker embedding.
- **Conditioning pipeline**: `cond_enc` (spkr_enc + emotion_adv_fc) + Perceiver
  resampler (32 learned queries cross-attend over `speech_emb +
  speech_pos_emb` of `cond_speech_tokens`).
- **Output**: `(34, 1024)` = `[spkr_proj | perceiver_resampled (32) | emotion_proj]`.
- **Runtime**: `src/lm/speaker_chatterbox.cpp`.  Hybrid CPU (mel + LSTM,
  ~5.5 MB weights, runs once per ref audio) + ggml graph (cond_enc +
  perceiver, cached by `n_ref_speech_tokens`).
- **Parity**: `tests/e2e/chatterbox_speaker_smoke.py` — 34 rows corr =
  1.000000 vs `VoiceEncoder.embeds_from_wavs` + `T3CondEnc.forward`.
- **Application example**: `examples/tts.py` ChatterboxSession (`--ref-audio
  path.wav` runs the full pipeline; default falls back to the bundled
  `conds.pt` speaker via `codec_lm_speaker_encode_from_embedding`).

### 2. Qwen3-TTS (`encoder_arch = "qwen3_tts_ecapa_tdnn"`) — **SCAFFOLDED**

- **Audio encoder**: ECAPA-TDNN (TDNN + SE-Res2Net stack + Attentive
  Statistical Pooling + final Conv1d) → 1024-d x-vector (from
  `Qwen3TTSSpeakerEncoder` in upstream).
- **Conditioning pipeline**: x-vector spliced as a single row into
  `talker_input_embed` between `tts_pad_embed` placeholders and the
  `tts_bos_embed`.
- **Output target**: `(1, 1024)`.
- **Runtime**: stub.  `lm.cpp` recognises the `encoder_arch`, exposes
  `codec_lm_speaker_get_info` correctly, but `codec_lm_speaker_encode`
  returns `CODEC_STATUS_NOT_SUPPORTED` with a clear error message.
- **Today's workaround**: `examples/tts.py` Qwen3TTSSession runs the HF
  `Qwen3TTSModel.extract_speaker_embedding` Python-side and feeds the
  result into the backbone via `inputs_embeds` (the same way it does for
  the entire voice-clone prompt).  Switching this to the C API is the
  follow-up.

#### Port checklist (ECAPA-TDNN)

1. **Converter** (`scripts/converters/lm_adaptor/qwen3_tts.py`):
   - Locate the HF `model.talker.speaker_encoder` parameters.
   - Bundle as `speaker.qwen3_tts.blocks_{l}.conv.{weight,bias}`,
     `speaker.qwen3_tts.mfa.…`, `speaker.qwen3_tts.asp.…`,
     `speaker.qwen3_tts.fc.{weight,bias}`.
   - Bake mel basis (slaney, librosa-style, fmin=0 fmax=8000, 128 mels)
     + Hann window into `speaker.qwen3_tts.mel_basis` /
     `speaker.qwen3_tts.window`.
   - Write `codec.speaker.encoder_arch = "qwen3_tts_ecapa_tdnn"`,
     `codec.speaker.n_rows = 1`, `codec.speaker.hidden_dim = 1024`,
     `codec.speaker.speaker_emb_dim = 1024`,
     `codec.speaker.ref_sample_rate = 24000`,
     plus `codec.speaker.ecapa.*` (enc_channels, enc_kernel_sizes,
     enc_dilations, enc_attention_channels, enc_res2net_scale,
     enc_se_channels).

2. **Runtime** (`src/lm/speaker_qwen3_tts.cpp`):
   - Slaney mel front-end (CPU helper in `audio_dsp.cpp`, exists for
     other models — reuse).
   - `TimeDelayNetBlock` = `ggml_conv_1d` (reflect pad) + ReLU.
   - `Res2NetBlock` = chunk along channels, chain residuals through
     `scale - 1` TDNN sub-blocks, re-concat.
   - `SqueezeExcitationBlock` = mean-pool over time + Conv1d + ReLU +
     Conv1d + sigmoid + multiplicative gate.
   - `SE-Res2NetBlock` = TDNN + Res2Net + TDNN + SE + skip.
   - `AttentiveStatisticsPooling` = compute (mean, std) → TDNN(channels×3 →
     attn_channels) → tanh → Conv1d → softmax over time → weighted
     (mean, std) → cat.
   - Final `fc` Conv1d → squeeze to `(1, 1024)`.

3. **Dispatch** (`lm.cpp`): replace the stub branch in
   `speaker_arch_init`/`codec_lm_speaker_encode*` with calls into the
   new file's API.

4. **Parity test** (`tests/e2e/qwen3_tts_speaker_smoke.py`):
   feed a synthetic 5 s waveform; compare against the vendored
   `Qwen3TTSSpeakerEncoder.forward(mel)` reference.  Threshold: corr ≥
   0.9999.

### 3. MOSS-TTSD (no `encoder_arch` registered) — **N/A by design**

- **Audio encoder**: there isn't a separate speaker encoder.  MOSS-TTSD's
  voice-clone path interleaves the **speech-tokenizer codes** of the ref
  audio directly into the prompt (as audio-channel tokens), alongside
  the ref transcript on the text channel.
- **No speaker_encode needed**: the existing `codec_encode(ref_pcm)`
  yields the speech tokens; the application splices them into the
  prompt.
- **Today**: `examples/tts.py` MossTTSDSession does exactly this through
  HF.  A future codec.cpp-only path would call `codec_encode` directly
  (which already works on the bundled `moss_ttsd_*.gguf` codec) — no
  new API surface required.

### 4. LFM2-Audio, MOSS-TTS-Realtime, CSM, MOSS-TTS-Nano — **no speaker encoder**

These models either don't accept ref-audio voice clone (zero-shot
or speaker-id based) or their conditioning lives entirely in the LM
prompt as text.  No `codec.speaker.has_encoder` metadata; the speaker
API returns `NOT_SUPPORTED`.

## Adding a new speaker encoder

For an audio encoder X that returns `(n_rows, hidden_dim)`:

1. Pick an `encoder_arch` string (snake_case, model-prefixed).
2. Add `CODEC_SPEAKER_ARCH_<X>` enum value in `src/lm/lm_internal.h`.
3. Write the converter (`scripts/converters/.../<model>.py`):
   - Bundle the encoder weights under `speaker.<arch>.…` namespace.
   - Bake any constants (mel basis, window) the runtime needs.
   - Write the `codec.speaker.*` metadata block (n_rows, hidden_dim,
     needs_*, ref_sample_rate, emotion_default, speaker_emb_dim,
     encoder_arch).
4. Write the runtime (`src/lm/speaker_<arch>.cpp`):
   - `<arch>_speaker_init` / `_free` / `_encode` / (optional)
     `_encode_from_emb`.
   - One cached ggml graph for the audio-encoder forward.
5. Wire dispatch (`src/lm/lm.cpp`):
   - Add the new `encoder_arch` string in `codec_lm_populate_info`.
   - Add a case in each of `speaker_arch_init`, `speaker_arch_free`,
     `codec_lm_speaker_encode`, `codec_lm_speaker_encode_from_embedding`.
6. Add a parity test (`tests/e2e/<model>_speaker_smoke.py`).

That's all the framework requires.
