# Audio-to-* implementation plan for codec.cpp

Extends `docs/audio_lm_extensions.md` (which surveyed 9 reference models)
with a broader **37-model sweep** covering ASR, audio chat, S2S
translation, voice conversion, audio-mixed multimodal, and music
continuation.  The output here is a concrete order of work for fitting
all of these into the existing `LLM-backbone (llama.cpp) + lm_adaptor
(codec_lm) + codec` shape.

## 0. Critical update — llama.cpp MTMD already handles audio

**Discovery:** `llama.cpp/tools/mtmd` (the "MultiModal Transformer
Decoder" library that replaced `llava.cpp`) already supports audio
input alongside images.  Public C API in `tools/mtmd/mtmd.h`:

```c
enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,   // ← yes
};

MTMD_API bool          mtmd_support_audio(mtmd_context * ctx);
MTMD_API int           mtmd_get_audio_sample_rate(mtmd_context * ctx);
MTMD_API mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples,
                                                    const float * data);
MTMD_API bool          mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap);
```

Audio projector types already wired in `clip.cpp` and exposed via
the standard `mmproj` GGUF artefact emitted by
`convert_hf_to_gguf.py --mmproj`:

| Projector | Model family |
|---|---|
| `PROJECTOR_TYPE_ULTRAVOX` | Ultravox (Whisper enc + stack-frames + MLP) |
| `PROJECTOR_TYPE_VOXTRAL` | Voxtral-Mini, Voxtral-Small |
| `PROJECTOR_TYPE_QWEN2A` | Qwen2-Audio |
| `PROJECTOR_TYPE_GLMA` | GLM-Audio (GLM-4-Voice tokenizer path) |
| `PROJECTOR_TYPE_LFM2A` | LFM2-Audio (audio-in side) |
| `PROJECTOR_TYPE_MUSIC_FLAMINGO` | Music Flamingo |
| `PROJECTOR_TYPE_QWEN25O` | Qwen2.5-Omni (dispatches to QWEN2A / QWEN25VL) |

Plus the audio preprocessor (mel-spec parameters, chunk-length, FFT
size, hop) is configurable from the mmproj GGUF — same workflow as
for vision projectors today.

### What this changes about the plan

The clusters A+B+M from §2 (Whisper-encoder + MLP / FastConformer + MLP
/ AuT) — which I'd scoped as **the most expensive Tier-1 work** in
codec.cpp — are **already done in llama.cpp**.  Specifically:

  - Cluster A (Whisper-large-v3 enc + 4× MLP) = `PROJECTOR_TYPE_VOXTRAL`
    or `PROJECTOR_TYPE_ULTRAVOX` depending on stack-frames variant.
  - Cluster B (FastConformer + MLP) = covered for LFM2-Audio via
    `PROJECTOR_TYPE_LFM2A`.
  - Cluster M (AuT) — needs to land in MTMD when Qwen3-Omni's full
    arch is supported; structurally fits the existing projector flow.

The remaining gaps (audio-OUTPUT side, codec_lm semantics, Talker
transformer, duplex streaming, dual-codec input) are still squarely
codec.cpp's responsibility — MTMD only covers the audio→hidden-embed
path on the LLM-input side.

### Revised division of labour

```
                    audio in        →   backbone        →   audio out
                    ─────────           ───────────         ─────────
LLM input adapter:  llama.cpp MTMD     llama.cpp           — (n/a for audio out)
                    (clip.cpp /
                     mtmd-audio.cpp)

Backbone LLM:       —                  llama.cpp           —

Per-step AR:        —                  codec_lm/llama.cpp  codec_lm
                                       talker (new)

Codec encode/dec:   — (raw PCM /        —                  codec.cpp codec_model
                       mel input)                          (Mimi/SNAC/XY/…)
```

### Adjusted Tier 1 (was: "ship Whisper-encoder in codec.cpp")

**Now:** ship a Python ctypes wrapper around `libmtmd.so` (parallels
`tests/e2e/_codec_lm_ctypes.py`) and rewire `examples/asr.py` so
each profile can run via either:

  1. **Stage A (current):** stock HF transformers — what's in this
     repo's `asr.py` today.
  2. **Stage A.5 (new):** MTMD via the ctypes wrapper, consuming a
     standalone GGUF + mmproj GGUF.  No HF dependency at inference.
     Whisper/Voxtral/Qwen2-Audio/Ultravox/LFM2-Audio audio-in all
     fall under this path.

The original "Stage B = native encoder in codec.cpp" is now mostly
deferred — only useful when MTMD doesn't yet cover a target encoder
(e.g. Granite-Speech Q-former, Phi-4-MM Conformer+LoRA route,
Gemma-3n USM, Qwen3-Omni AuT).  Those still need codec.cpp-side work
*or* a parallel MTMD contribution.

### What stays in codec.cpp regardless

  - `codec_lm` for all audio-output AR (existing).
  - Codec encode/decode for Mimi / SNAC / XY-Tokenizer / etc. (existing).
  - New API surfaces from §4 that aren't about *encoder graphs* but
    about *codec_lm semantics*: `position_kind` (sequential mode-switch),
    `TALKER_TRANSFORMER` kind, dual user-audio-stream, streaming KV
    persistence.  None of these are duplicated by MTMD.

Status today (the 7 wired TTS profiles in `docs/tts_cli.md`): the
output-side audio path is well-covered for both `parallel_heads_delay`
and `residual_depth_ar` kinds.  **There is no audio-input adapter, no
streaming audio context, no Talker abstraction, and no dual-stream
output.**  This doc proposes the minimum new abstractions to fix that.

## 1. Per-model coverage

The 37 models cluster on five axes: (a) audio-input encoder family,
(b) how that encoder's output reaches the backbone, (c) streaming
property, (d) output mode (text / audio / interleaved / parallel /
duplex), (e) whether a codec_lm step loop applies at all.

### 1.1 ASR / audio understanding (text out, no codec_lm)

| Model | Encoder → projector | Frame rate (post) | Streaming | Backbone | HF readiness | Params |
|---|---|---|---|---|---|---|
| Whisper-large-v3 / turbo / distil | log-mel 128 × 3000 → 32-layer bidir Transformer | (encoder-only path) | whole 30 s | own dec | stock | 0.8–1.55B |
| Voxtral-Mini-3B-2507 | Whisper-large-v3 + 4× MLP downsample | 12.5 Hz inline embed | whole 30 s, 32 k ctx ≈ 40 min | Ministral-3B | **stock** | 3B |
| Voxtral-Small-24B-2507 | same adapter as above | 12.5 Hz | same | Mistral-Small-3.1 | **stock** | 24.3B |
| Qwen2-Audio-7B-Instruct | Whisper-large-v3 + MLP | inline | whole-utt | Qwen2 | **stock** | 8.4B |
| Qwen2.5-Omni-7B (audio-in subset) | Whisper-derived + MLP, TMRoPE aligned | streaming 2 s blocks | streaming | Qwen2.5 (Thinker) | trust_remote_code | 7B |
| Phi-4-Multimodal-Instruct | 80-d log-mel → 3 conv (8× ds) → 24 Conformer blocks → MLP + **speech LoRA route** on decoder | whole-utt | trust_remote_code | Phi-4-Mini | 5.6B |
| SenseVoice-Small (FunASR) | 80-d log-mel → stride-6 stack → SAN-M → CTC | streaming chunks | text + 5 tag tokens | own (non-AR) | FunASR pkg | 234M |
| Parakeet-TDT-0.6b / 1.1b | FastConformer + TDT decoder (joint token + duration) | streaming chunks | own RNN-T-ish | NeMo / recent transformers | 0.6 / 1.1B |
| Granite-Speech-3.3-8B | 10-block Conformer + CTC → 2-layer **Q-former** projector (5× ds, 10 Hz) + always-on speech-LoRA | whole-utt | Granite-3.3 | **stock** | 8B |
| Aero-1-Audio | Whisper-like enc → projector → inline | ≤15 min | Qwen2.5-1.5B | **stock** | 1.5B |
| Gemma-3n-Audio (E2B / E4B) | USM-based enc, 160 ms / frame → 6.5 Hz × 1536-d | streaming 160 ms chunks | Gemma-3n | **stock** | 2B / 4B |
| Kyutai STT (stt-2.6b-en, stt-1b-en_fr) | **Mimi encoder used as adapter** → 12.5 Hz codec tokens | true-streaming, 0.5–2.5 s delay | Helium | **stock** | 1–2.6B |

### 1.2 Audio chat / duplex / S2S translation

| Model | Audio-in | Audio-out | Output mode | Backbone | Params |
|---|---|---|---|---|---|
| Moshi (full duplex) | Mimi 8-cb @ 12.5 Hz on user PCM | Mimi 8-cb model stream | duplex (parallel user + model + text) | Helium | 7B |
| Hibiki | identical Moshi shape | identical | streaming S2S | Helium | 2 / 7B |
| LFM2-Audio-1.5B (chat) | FastConformer + projector inline | Mimi 8-cb model | sequential mode-switch text↔audio | LFM2 | 1.5B |
| GLM-4-Voice-9B | Whisper-enc + **VQ → 12.5 Hz tokens** (cluster F) | CosyVoice-derived flow-matching dec | interleaved text+audio | GLM-4-9B | 9B |
| MiniCPM-o-2.6 | Whisper-medium enc (300M) + SigLip; **TDM** (time-division mux) | ChatTTS codec | streaming text+audio | Qwen2.5-7B | 8B |
| Llama-Omni | Whisper-large-v3 enc → MLP → Llama-3.1-8B | discrete-unit HiFi-GAN (HuBERT-units) | text + NAR speech parallel | Llama-3.1-8B | 8B+ |
| SpeechGPT-7B | HuBERT k-means units as **vocab tokens** | HuBERT-unit HiFi-GAN | sequence-mode-switch | LLaMA-13B | 7–13B |
| Spirit-LM-7B | HuBERT phonetic tokens (+ pitch/style) as **vocab tokens** | HuBERT-unit HiFi-GAN | text+speech word-level interleave | LLaMA-2 | 7B |
| SeamlessM4T-v2-Large | w2v-BERT 2.0 Conformer enc (635M) | non-AR T2U → HiFi-GAN | text + waveform | own seq2seq | 2.3B |
| SeamlessExpressive | w2v-BERT + PRETSSel | own | waveform | own | research-only |
| StreamSpeech | chunk Conformer + CTC alignment + NAR T2U + HiFi-GAN | true-streaming, configurable chunk | text + waveform | own | custom repo, <1B |
| Qwen2.5-Omni-7B / Qwen3-Omni-MoE-30B-A3B | Whisper-derived (2.5) or **AuT** (3) → MLP | Talker emits Qwen3-TTS-Tokenizer | text + audio parallel | Qwen2.5 / Qwen3-MoE | 7B / 30B(A3B) |
| Mini-Omni / Mini-Omni-2 | Whisper + vision enc → Qwen2-0.5B | SNAC 7-cb + text head co-emitted | Text-Instruct **Delay Parallel** 8-wide | Qwen2-0.5B | 0.5B + adapters |
| Step-Audio-Chat-130B | **dual codec input** (linguistic 16.7 Hz × 1024 + semantic 25 Hz × 4096, 2:3 interleave) | hybrid flow-matching + neural vocoder | text + audio | Step-1 (130B, not in llama.cpp) | 130B |

### 1.3 Voice conversion (no LM)

| Model | Content enc | Decoder | Notes |
|---|---|---|---|
| OpenVoice v2 | MeloTTS base | ToneColorConverter (VITS flow) | no LLM |
| RVC v2 | HuBERT-12L 768-d + RMVPE f0 | flow prior → NSF-HiFiGAN | no LLM, streaming-ish |
| Diff-VC / Diff-HierVC | source-filter content + DiffPitch | DiffVoice (two diffusions) | no LLM |
| Seed-VC | XLSR / Whisper / HuBERT content | DiT flow-matching + BigVGAN | no LLM, streaming option |
| StyleTTS-VC | content + style (KD from StyleTTS) | adversarial | no LLM |

These are pure `src/models/*.cpp` graph kinds — no `codec_lm` content.

### 1.4 Music / sound continuation

| Model | Cond | Decode | Notes |
|---|---|---|---|
| MusicGen-Medium/Melody | T5 text cross-attn (+ chromagram audio cross-attn for Melody) | EnCodec 32 kHz 4 cb @ 50 Hz, codebook-delay AR | `parallel_heads_delay` with delay matching cb offsets |
| Stable Audio Open 1.0 | T5 text cond | DiT-over-VAE-latents (64-d @ 21.5 Hz), one-shot diffusion | not `codec_lm` — pure graph-kind |
| MAGNeT | T5 text cond | masked NAR Transformer over EnCodec codes | new kind `non_ar_masked_iter` |
| YuE | lyrics + style | Stage-1 LM (LLaMA-2 over text + cb-0 audio) + Stage-2 LM (residual cb) | two `residual_depth_ar` instances chained |

### 1.5 Architecturally novel models — expanded notes

**Step-Audio-Chat-130B** runs *two parallel codecs at different rates*
on user audio (16.7 Hz linguistic + 25 Hz semantic) interleaved 2:3
into a single backbone-input stream.  No other surveyed model has
dual-codec input.  Backbone (Step-1 130B) is not in llama.cpp ⇒ defer.
Source: [arXiv 2502.11946](https://arxiv.org/abs/2502.11946).

**Kyutai STT** is structurally Moshi with `has_model_audio_stream=false`
plus a positional delay on text emission (0.5–2.5 s).  Generalisation of
MOSS-TTSD's `delay_pattern[]` to *text* instead of audio output.
Source: [DSM paper arXiv 2509.08753](https://arxiv.org/abs/2509.08753).

**Phi-4-Multimodal** is the only model with **per-modality LoRA routing**
— the same Phi-4-Mini decoder runs with a 460M speech-LoRA when audio
is present in the prompt.  Per-utterance, not per-token.

**Qwen3-Omni** swaps Whisper → **AuT** (Qwen's own from-scratch audio
Transformer).  Output dim and frame rate match the Whisper-family
adapter API; only the encoder graph is new.

**Granite-Speech** is the only model with a **Q-former projector**
(window query Transformer over 15-acoustic-embedding blocks, 5× ds),
not an MLP.

**Mini-Omni-2** does the most aggressive parallel decoding in the
survey: 8 logit streams per step (1 text + 7 SNAC) with 1-step
inter-stream delay.  Structurally `parallel_heads_delay` + a text head
co-emitted at position 0.

**SeamlessM4T-v2 / StreamSpeech** have a **non-AR Text-to-Unit (T2U)**
stage between the LM decoder and HiFi-GAN.  No per-step audio AR — the
"codec_lm" analogue is degenerate.  These are full codec.cpp graph
pipelines, not `codec_lm` shapes.

**MusicGen-Melody** is the only audio-conditioned model in the music
family — the melody prompt is a chromagram cross-attended into the LM,
not an inline-embedded input.  Cross-attention isn't currently in
`codec_lm`; if MusicGen is in scope, this is new graph wiring.

**Stable Audio Open / MAGNeT** are diffusion / non-AR — no `codec_lm`
step loop applies; pure `codec.cpp` graph kinds.

### 1.6 Variants noted in one line

- Distil-Whisper, Whisper-turbo: same encoder as Whisper-large-v3.
- Voxtral-Small-24B: identical adapter shape to Voxtral-Mini-3B.
- Hibiki = Moshi shape, task only differs.
- Aero-1-Audio = Qwen2-Audio shape, smaller (Qwen2.5-1.5B backbone).
- LFM2.5-Audio = LFM2-Audio retrained.
- MAGNeT ≈ MusicGen non-AR variant.

## 2. Audio-input adapter clusters

The MVP insight: **one new API call covers 10+ models if its underlying
encoder graph is swappable**.  Clusters A–E and M all produce
`[T_audio, hidden]` continuous embeddings spliced inline into the
backbone prompt; only the encoder weights / projector differ.

| Cluster | Encoder | Projector | Frame rate post | Inline / token? | Models |
|---|---|---|---|---|---|
| **A** | Whisper-large-v3 (1280-d, 50 Hz) | MLP, 4× temporal ds | 12.5 Hz | inline | Voxtral-Mini/Small, Qwen2-Audio, Qwen2.5-Omni audio-in, Aero-1, Llama-Omni, MiniCPM-o-2.6, distil-whisper enc |
| **B** | FastConformer (8× depthwise conv ds) | MLP | ~12.5 Hz | inline | LFM2-Audio, Parakeet-enc-only |
| **C** | 24-block Conformer over log-mel | MLP + per-modality LoRA on decoder | ~12.5 Hz | inline (+ LoRA swap) | Phi-4-MM |
| **D** | 10-block Conformer + CTC | 2-layer window Q-former (5× ds) | 10 Hz | inline (+ always-on LoRA) | Granite-Speech-3.3 |
| **E** | USM 160 ms tokenizer | (none) | 6.5 Hz | inline | Gemma-3n-Audio |
| **F** | Whisper enc + VQ codebook | embedding table | 12.5 Hz | as ordinary vocab token | GLM-4-Voice |
| **G** | codec.cpp Mimi `codec_encode` (already shipped) | none — codes ↦ N-cb embed sum | 12.5 Hz | summed audio embed | Moshi, Hibiki, Kyutai STT |
| **H** | HuBERT + k-means (units in vocab) | embedding (vocab expanded) | 50 Hz pre-dedup | ordinary vocab token | SpeechGPT, Spirit-LM |
| **I** | w2v-BERT 2.0 Conformer (635M) | own dec cross-attn | 50 Hz | encoder-decoder cross-attn | SeamlessM4T-v2, StreamSpeech |
| **J** | log-mel + SAN-M | CTC head (no LLM splice) | ~16 Hz | (text emitted directly) | SenseVoice |
| **K** | FastConformer + TDT | RNN-T joint net | ~12.5 Hz | (text + duration emitted directly) | Parakeet |
| **L** | HuBERT / ContentVec / XLSR + f0 | flow / diffusion decoder | various | (no LM) | RVC, OpenVoice, Diff-VC, Seed-VC, StyleTTS-VC |
| **M** | AuT | MLP | 12.5 Hz | inline | Qwen3-Omni |
| **N** | linguistic + semantic codecs in parallel, 2:3 temporal interleave | none | mixed | summed | Step-Audio |
| **O** | chromagram extractor (CPU) | cross-attn | — | cross-attention | MusicGen-Melody |

**Coverage rule:** A+B+C+D+E+M → one API call (`compose_audio_input_embd`)
+ pluggable encoder graphs.  G+H are reuses of the existing
token-based audio embed sum.  F is essentially G with a different
codec.  I, J, K, L are standalone graph pipelines that don't go
through `codec_lm` at all.  N is the only multi-codec input shape; O
is the only cross-attention conditioning.

## 3. Output mode taxonomy

| Output mode | codec_lm support today | Models |
|---|---|---|
| Text-only (no codec_lm) | n/a — skip codec_lm | Whisper, Voxtral, Qwen2-Audio, Phi-4-MM, Granite-Speech, Aero-1, Gemma-3n-Audio, SenseVoice, Parakeet, Kyutai STT |
| Audio-only AR | ✅ `residual_depth_ar` + `parallel_heads_delay` | CSM, Qwen3-TTS, MOSS-TTSD, MOSS-TTS-Realtime, Chatterbox |
| Sequential mode-switch text↔audio | ❌ need `set_position_kind` | LFM2-Audio chat, GLM-4-Voice, SpeechGPT, Spirit-LM |
| Parallel text+audio (text head co-emitted) | ❌ need `step_text_logits` | Mini-Omni-2 (1 text + 7 SNAC), Moshi, partially LFM2 |
| Talker-driven dual stream (persistent KV) | ❌ need new kind `talker_transformer` | Qwen2.5-Omni, Qwen3-Omni |
| Duplex (user audio fed continuously + model audio + text co-emitted) | ❌ need user-stream embed + `POS_DUPLEX` | Moshi, Hibiki |
| Music codec out | ✅ `parallel_heads_delay` w/ delay pattern fits | MusicGen, YuE-stage-2 |
| Non-AR (masked or diffusion) | ❌ not a `codec_lm` shape | MAGNeT, Stable Audio Open |

## 4. Proposed C API extensions

Ordered by how many models each unlocks.

### 4.1 Audio-input adapter — unlocks A+B+C+D+E+M (10+ models)

```c
// in include/codec_audio_input.h

enum codec_audio_input_kind {
    CODEC_AUDIO_IN_NONE          = 0,
    CODEC_AUDIO_IN_PCM_F32       = 1,   // raw waveform, in-graph mel
    CODEC_AUDIO_IN_LOG_MEL       = 2,   // pre-computed log-mel
    CODEC_AUDIO_IN_CODEC_TOKENS  = 3,   // Mimi / HuBERT / VQ tokens
};

// New fields on codec_lm_info:
//   bool                              has_audio_input_adapter;
//   enum codec_audio_input_kind       audio_input_kind;
//   int32_t                           audio_input_sample_rate_hz;
//   int32_t                           audio_input_frame_rate_hz;   // post-projector
//   int32_t                           audio_input_token_dim;       // == hidden_dim

enum codec_status codec_lm_compose_audio_input_embd(
    struct codec_lm * lm,
    const float *     pcm_or_features,    // [n_samples] or [n_frames, n_mel]
    int32_t           n_in,
    float *           out_embd,            // [T_audio_max, hidden_dim], caller-allocated
    int32_t *         out_T_audio);
```

Internal dispatch via metadata `codec.lm.audio_in.encoder_kind ∈
{whisper, fastconformer, conformer, qformer, usm, aut}` and bundled
weights under `lm.audio_in.encoder.*` / `lm.audio_in.projector.*` in
the codec_lm GGUF.

### 4.2 Streaming audio context — unlocks Moshi / Hibiki / Kyutai STT / true-streaming chat

```c
enum codec_status codec_lm_state_push_audio_input_chunk(
    struct codec_lm_state * st,
    const float *           pcm_chunk,
    int32_t                 n_samples,
    int32_t                 chunk_id);  // ordering / cache reuse

const float * codec_lm_state_get_pending_audio_embd(
    struct codec_lm_state * st,
    int32_t *               out_T);
```

State owns a rolling encoder KV cache or feature cache; chunks
accumulate until the next `step_begin` consumes them.

### 4.3 Token-based user audio stream — unlocks Moshi / Hibiki / Step-Audio

```c
// codec_lm_info additions:
//   bool      has_user_audio_stream;
//   int32_t   n_user_codebook;
//   const int32_t * user_codebook_sizes;
//   int32_t   n_user_codec;                  // 1 normally; Step-Audio = 2
//   const int32_t * user_codec_frame_rates_hz;

enum codec_status codec_lm_state_set_user_audio_codes(
    struct codec_lm_state * st,
    int32_t                 codec_id,
    const int32_t *         codes,
    int32_t                 frame_offset);
```

### 4.4 Position-kind / dual-stream output — unlocks LFM2 chat / GLM-4-Voice / Spirit-LM / Mini-Omni-2 / Qwen-Omni

```c
enum codec_lm_position_kind {
    CODEC_LM_POS_TEXT    = 0,   // backbone text only, no codec_lm step
    CODEC_LM_POS_AUDIO   = 1,   // codec_lm runs full step
    CODEC_LM_POS_BOTH    = 2,   // Mini-Omni-2 / Moshi: text head + N-cb audio
    CODEC_LM_POS_DUPLEX  = 3,   // Moshi: text + model audio + user audio fed in
};

enum codec_status codec_lm_state_set_position_kind(
    struct codec_lm_state *      st,
    enum codec_lm_position_kind  kind);

const float * codec_lm_step_text_logits(  // valid when POS_BOTH / POS_DUPLEX
    const struct codec_lm_state * st,
    int32_t *                     out_n);
```

### 4.5 New kind: `TALKER_TRANSFORMER` — unlocks Qwen2.5-Omni / Qwen3-Omni

```c
enum codec_lm_kind {
    CODEC_LM_KIND_PARALLEL_HEADS_DELAY = 1,
    CODEC_LM_KIND_RESIDUAL_DEPTH_AR    = 2,
    CODEC_LM_KIND_TALKER_TRANSFORMER   = 3,
};

// Talker kind owns a multi-step KV cache that does NOT reset per
// backbone step.  step_begin appends one position; the cache is
// cleared only on state_reset / state_kv_clear.
enum codec_status codec_lm_state_kv_clear(struct codec_lm_state * st);
```

### 4.6 Modality-routed LoRA hint — unlocks Phi-4-MM

Metadata-only (host LLM applies the LoRA caller-side):

```c
// codec_lm_info additions:
//   const char * host_modality_lora_name;   // "speech"
//   const char * host_modality_lora_blob;   // tensor namespace prefix
```

### 4.7 Cross-attention prompt features — needed only by MusicGen-Melody

Defer.  This is a `codec.cpp` graph wiring concern (chromagram in
cross-attention KV), not a `codec_lm` extension.

## 5. Phased implementation order

**Tier 1 — MVP audio-in (Voxtral as reference).**

1. Add `codec_lm_compose_audio_input_embd` + Whisper-encoder graph
   under `src/models/whisper_encoder.cpp`.  In-graph log-mel via the
   STFT/iSTFT recipes already documented in `CLAUDE.md`.
2. Wire **Voxtral-Mini-3B-2507** as the smoke test — simplest Whisper
   + 4× MLP downsample exemplar, stock-llama.cpp Ministral-3B backbone.
3. New `examples/asr.py` driver (analogue of `examples/tts.py`): audio
   in → encoder → splice into prompt → llama.cpp text-emission AR loop
   → decode tokens.

Once (1)+(2) ship, Qwen2-Audio-7B / Voxtral-Small-24B / Aero-1-Audio
ride on the same adapter — only projector weights / backbone GGUFs
differ.

**Tier 2 — adapter variants (one model = one new encoder family).**

4. **LFM2-Audio audio-in mode** — FastConformer encoder (cluster B);
   reuses existing LFM2-Audio audio-out wiring.
5. **Granite-Speech-3.3-8B** — Conformer + Q-former projector (D).
6. **Phi-4-Multimodal** — Conformer + speech-LoRA route (C) +
   `host_modality_lora_name` metadata.
7. **Gemma-3n-Audio** — USM encoder (E).

**Tier 3 — streaming + duplex.**

8. **Kyutai STT (stt-2.6b-en)** — first true-streaming.  Uses Mimi
   codec as adapter (cluster G), already supported.  Validates
   `state_push_audio_input_chunk` + delayed text emission.
9. **Moshi (full duplex)** — adds user-stream embed + `POS_DUPLEX`.
   Hibiki rides on this with zero new work.
10. **LFM2-Audio chat mode** — sequential mode-switch.  Adds
    `set_position_kind`.

**Tier 4 — parallel decoding extensions.**

11. **Mini-Omni-2** — text-head-co-emitted parallel decoding over SNAC
    (codec already converted).  Adds `step_text_logits` + `POS_BOTH`.
12. **GLM-4-Voice** — Whisper-enc + VQ → tokens (F).  New codec
    converter (GLM-4-Voice-Tokenizer); reuses token-input compose.
    New `src/models/glm4voice_decoder.cpp` for the CosyVoice-derived
    flow-matching decoder.

**Tier 5 — Talker transformer.**

13. **Qwen2.5-Omni-7B** — adds `TALKER_TRANSFORMER` kind.
14. **Qwen3-Omni-MoE** — same kind, swaps Whisper → AuT, swaps Talker
    depth-AR → MTP parallel heads (already `parallel_heads_delay`).

**Tier 6 — encoder-decoder S2S translation.**

15. **SeamlessM4T-v2-Large** — full codec.cpp graph pipeline (w2v-BERT
    encoder + seq2seq decoder + non-AR T2U + HiFi-GAN).  Not a
    `codec_lm` model; sits beside it.
16. **StreamSpeech** — chunk Conformer + CTC alignment.  Defer.

**Tier 7 — Voice conversion (no LM).**

17. **RVC v2** — HuBERT content + RMVPE f0 + NSF-HiFiGAN.  Reuses
    Chatterbox-S3G's NSF + in-graph STFT helpers; new pieces are
    ContentVec encoder, flow prior, optional retrieval (CPU-side KNN).
18. **OpenVoice v2 / Seed-VC / Diff-HierVC / StyleTTS-VC** — each adds
    a new decoder family.  Defer unless explicitly requested.

**Tier 8 — music / sound (defer).**

19. **MusicGen-Medium** — EnCodec 32 kHz codec converter +
    `parallel_heads_delay` with codebook-delay.  Adds T5 text
    cross-attention to the AR transformer (new wiring).
20. **YuE** — two `residual_depth_ar` instances chained.  Needs YuE
    codec converter.
21. **Stable Audio Open** — DiT-over-VAE-latents.  Pure `codec.cpp`
    graph kind; needs DiT block builder + VAE.
22. **MAGNeT** — masked NAR Transformer over EnCodec.  New kind
    `non_ar_masked_iter`.  Pure `codec.cpp`.
23. **Step-Audio-Chat-130B** — dual-codec interleave + Step-1 backbone
    (not in llama.cpp).  Defer.

## 6. Risk / open questions

**HF availability.**  Stock `AutoModel`: Whisper variants, Voxtral,
Qwen2-Audio, Qwen2.5-Omni (`trust_remote_code`), Phi-4-MM
(`trust_remote_code`), Granite-Speech, Aero-1, Gemma-3n-Audio,
LFM2-Audio, Moshi / Hibiki, Kyutai STT, MusicGen, MAGNeT, Stable Audio
(via diffusers), SeamlessM4T-v2.  Custom-repo-only or research-only:
SpeechGPT, Spirit-LM, Llama-Omni, MiniCPM-o-2.6 (model loads but
ChatTTS codec needs custom path), StreamSpeech, OpenVoice / RVC /
Diff-VC / Seed-VC / StyleTTS-VC, YuE, Step-Audio.

**Codec coverage gaps.**  Need new converters for: EnCodec 32 kHz (4 cb
@ 50 Hz), HuBERT-unit HiFi-GAN, Seamless T2U + HiFi-GAN, GLM-4-Voice
flow-matching decoder, ChatTTS, YuE codec, Step-Audio dual codec, Stable
Audio VAE.  EnCodec is easiest (well-documented AE + RVQ); HuBERT-unit
HiFi-GAN is medium; Seamless T2U is hardest (NAR seq2seq with custom
attention pattern).

**Streaming vs whole-utterance.**  For Tier 1 MVP we can ship
**whole-utterance only** — Voxtral / Qwen2-Audio / Phi-4-MM all accept
that.  True streaming requires (4.2) `state_push_audio_input_chunk` +
a rolling encoder KV cache.  Whisper's encoder is fundamentally
non-streaming (bidir attn over fixed 30 s); for Whisper-based streaming
models (Llama-Omni, MiniCPM-o-2.6) the only path is chunk + overlap +
caller-side stitch.  Tier 3's Kyutai STT side-steps this by using Mimi
(already streaming-capable) as the encoder.

**Continuous-feature vs token-feature input — does `compose_audio_embd`
grow?** No — keep them separate.  Existing `compose_audio_embd` is
*output-side* (codes → next-step hidden).  The new
`compose_audio_input_embd` is *input-side* (PCM → `[T, h]` prompt
span).  Conflating would force the API to track which codebook is
input vs output.  For models that have both (Moshi, Hibiki,
LFM2-Audio chat, Qwen-Omni, Mini-Omni-2) the caller assembles the
prompt = `[text_embd; audio_input_embd; ...]` then drives the
per-step output loop normally.

**KV-cache regimes.**  Three to support:
1. Per-step (today's `residual_depth_ar`, reset every step).
2. Persistent multi-step (proposed `TALKER_TRANSFORMER`).
3. Rolling-window-with-eviction (true streaming Whisper / USM).

(1) and (2) are covered by the proposed API.  (3) is deferred to
Tier 6 if ever — most models we care about either fit (1)+(2) or
accept whole-utterance chunking.

## 7. Implementation strategy — Python HF first, native incrementally

Same pattern as MOSS-TTS-Nano-100M's HF-fallback profile: each new
audio-in model can land as a Python `examples/asr.py` (or
`examples/audio_chat.py`) profile that uses HF transformers for the
encoder + backbone today, with `bypass_standard_run = True` to skip
codec.cpp's standard AR loop.  Once the corresponding C-side audio
encoder / runtime kind lands, the profile migrates to native — same
public CLI, no behaviour change for the user.

Concretely:

- **Stage A (1 hour per model, no C++ changes):** HF-fallback profile
  for Voxtral / Qwen2-Audio / Phi-4-MM / Granite-Speech / Aero-1-Audio
  shipped as `examples/asr.py` entries.  Validates the prompt-format
  + tokenizer-output path.
- **Stage B (~1 model-week):** `codec_lm_compose_audio_input_embd` +
  Whisper-encoder graph in C++.  Move Voxtral to native; the rest of
  cluster A automatically benefits.
- **Stage C (per-cluster):** add FastConformer / Conformer / Q-former /
  USM / AuT encoder graphs as cluster-B…M models join the native path.

This staging avoids the trap of waiting for the whole C++ stack to be
ready before shipping anything — every Stage-A profile delivers user
value immediately, Stage-B/C migrations are pure performance wins.

## 8. Sources

Per-model references:

- [Whisper (HF docs)](https://huggingface.co/docs/transformers/en/model_doc/whisper),
  [Distil-Whisper](https://github.com/huggingface/distil-whisper)
- [Voxtral paper arXiv 2507.13264](https://arxiv.org/abs/2507.13264),
  [Voxtral-Mini HF](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507),
  [Voxtral-Small HF](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)
- [Qwen2-Audio HF](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct),
  [arXiv 2407.10759](https://arxiv.org/abs/2407.10759)
- [Qwen2.5-Omni arXiv 2503.20215](https://arxiv.org/abs/2503.20215),
  [HF](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [Qwen3-Omni arXiv 2509.17765](https://arxiv.org/abs/2509.17765)
- [Phi-4-MM HF](https://huggingface.co/microsoft/Phi-4-multimodal-instruct),
  [arXiv 2503.01743](https://arxiv.org/abs/2503.01743)
- [SenseVoice HF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall),
  [arXiv 2407.04051](https://arxiv.org/abs/2407.04051)
- [Parakeet-TDT HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2),
  [NeMo ASR docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [Granite-Speech HF](https://huggingface.co/ibm-granite/granite-speech-3.3-8b),
  [HF docs](https://huggingface.co/docs/transformers/model_doc/granite_speech)
- [Aero-1-Audio HF](https://huggingface.co/lmms-lab/Aero-1-Audio)
- [Gemma-3n dev guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/),
  [HF docs](https://huggingface.co/docs/transformers/en/model_doc/gemma3n)
- [Kyutai STT](https://kyutai.org/stt),
  [DSM paper arXiv 2509.08753](https://arxiv.org/abs/2509.08753),
  [stt-2.6b-en HF](https://huggingface.co/kyutai/stt-2.6b-en)
- [Moshi paper arXiv 2410.00037](https://arxiv.org/abs/2410.00037),
  [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- [Hibiki HF](https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16),
  [paper arXiv 2502.03382](https://arxiv.org/abs/2502.03382)
- [LFM2-Audio HF](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B),
  [LFM2 arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- [GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice),
  [paper arXiv 2412.02612](https://arxiv.org/abs/2412.02612),
  [tokenizer HF](https://huggingface.co/zai-org/glm-4-voice-tokenizer)
- [MiniCPM-o-2.6 HF](https://huggingface.co/openbmb/MiniCPM-o-2_6)
- [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni),
  [paper arXiv 2409.06666](https://arxiv.org/abs/2409.06666)
- [SpeechGPT arXiv 2305.11000](https://arxiv.org/abs/2305.11000)
- [Spirit-LM arXiv 2402.05755](https://arxiv.org/abs/2402.05755),
  [facebookresearch/spiritlm](https://github.com/facebookresearch/spiritlm)
- [SeamlessM4T-v2 HF docs](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)
- [StreamSpeech](https://github.com/ictnlp/StreamSpeech),
  [paper arXiv 2406.03049](https://arxiv.org/abs/2406.03049)
- [OpenVoice](https://github.com/myshell-ai/OpenVoice),
  [v2 HF](https://huggingface.co/myshell-ai/OpenVoiceV2)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI),
  [annotated RVC](https://gudgud96.github.io/2024/09/26/annotated-rvc/)
- [Diff-HierVC arXiv 2311.04693](https://arxiv.org/abs/2311.04693),
  [GitHub](https://github.com/hayeong0/Diff-HierVC)
- [Seed-VC](https://github.com/Plachtaa/seed-vc),
  [DiT-VC paper arXiv 2411.09943](https://arxiv.org/abs/2411.09943)
- [StyleTTS-VC arXiv 2212.14227](https://arxiv.org/abs/2212.14227)
- [Mini-Omni-2 arXiv 2410.11190](https://arxiv.org/abs/2410.11190)
- [Step-Audio arXiv 2502.11946](https://arxiv.org/abs/2502.11946),
  [GitHub](https://github.com/stepfun-ai/Step-Audio)
- [MusicGen arXiv 2306.05284](https://arxiv.org/abs/2306.05284),
  [audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
- [Stable Audio Open HF](https://huggingface.co/stabilityai/stable-audio-open-1.0),
  [paper arXiv 2407.14358](https://arxiv.org/abs/2407.14358)
- [MAGNeT arXiv 2401.04577](https://arxiv.org/abs/2401.04577)
- [YuE](https://github.com/multimodal-art-projection/YuE),
  [paper arXiv 2503.08638](https://arxiv.org/abs/2503.08638)
