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

### 2. Qwen3-TTS (`encoder_arch = "qwen3_tts_ecapa_tdnn"`) — **SHIPPED**

- **Audio encoder**: ECAPA-TDNN (TDNN + 3× SE-Res2Net + MFA TDNN +
  Attentive Statistical Pooling + final Conv1d) → 1024-d x-vector
  (from `Qwen3TTSSpeakerEncoder` in upstream).
- **Conditioning pipeline**: x-vector spliced as a single row into
  `talker_input_embed` between `tts_pad_embed` placeholders and the
  `tts_bos_embed`.
- **Output**: `(1, 1024)`.
- **Runtime**: `src/lm/speaker_qwen3_tts.cpp`.  Pure CPU forward
  (~5 GFLOPs once per ref clip; no big matmul, ggml graph would add
  complexity without perf wins).  Reflect-pad reflection helper handles
  TDNN k=5 / k=3 with arbitrary dilation.
- **Parity**: `tests/e2e/qwen3_tts_speaker_smoke.py` — corr=1.000000,
  max_abs_diff < 1e-3 vs HF reference.
- **Notes**: `codec_lm_speaker_encode_from_embedding` is a memcpy
  pass-through (the x-vector IS the cond_emb), so callers that cached
  the 1024-d vector elsewhere can feed it back without re-running ECAPA.

### 3. MOSS-TTSD (uses `codec_encode`, no `encoder_arch`) — **SHIPPED**

- **Audio encoder**: no separate speaker encoder.  MOSS-TTSD's voice-
  clone path interleaves the **speech-tokenizer codes** of the ref
  audio directly into the prompt (as audio-channel tokens), alongside
  the ref transcript on the text channel.
- **codec.cpp surface**: the existing `codec_encode` API on the
  bundled `moss_ttsd_v0_5.gguf` (which carries the XY-Tokenizer codec
  + LM adaptor sections in one file) produces speech tokens that the
  application splices into the prompt.  No `codec_lm_speaker_encode`
  variant is registered because there's nothing audio→embedding to
  compute; the LM consumes the codes through its own embedding tables.
- **Parity**: `tests/e2e/moss_ttsd_encode_smoke.py` — `codec.cpp`'s
  encode (via `codec-cli encode`) matches HF's MossTTSDProcessor
  speech-tokenizer codes 100 % over a 1.5 s clip.
- **Today**: `examples/tts.py` MossTTSDSession uses HF for full prompt
  assembly (including ref-audio encoding).  Switching to codec.cpp's
  encoder is mechanical (call `codec_encode` then splice the resulting
  tokens into the prompt token tensor manually) and is a separate
  follow-up that doesn't affect the framework — the encoder itself is
  bit-parity-verified.

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
