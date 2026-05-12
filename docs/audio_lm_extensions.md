# codec_lm API growth survey — audio-in & audio-in/out models

Goal: identify the minimum API extensions needed so the current
`codec_lm` (TTS-only: backbone → hidden → N audio-codebook tokens →
composed embed) can grow into speech-in chat (A), full audio-in /
audio-out chat (B), and Moshi-style streaming dual-stream (C).

Current API surface (see [`include/codec_lm.h`](../include/codec_lm.h)):
kinds = `PARALLEL_HEADS_DELAY` + `RESIDUAL_DEPTH_AR`; compose is
`sum_i audio_embd_i[codes[i]] -> [hidden_dim]`; no audio-in concept
exists.

## 1. Moshi (full dual-stream mode)

Backbone: **Helium-7B** Temporal Transformer + small **Depth
Transformer** (6 layers, Llama-style) for intra-step codebooks. Codec:
**Mimi**, 12.5 Hz, **Q=8** quantizers per stream (codebook 0 =
semantic, 1–7 = acoustic with a small acoustic delay between semantic
and acoustic). Per-step structure: **3 parallel streams summed at the
input**: (a) **user-audio Mimi codes** (8 codebooks), (b)
**assistant-audio Mimi codes** (8 codebooks), (c) **inner-monologue
text token** (assistant's transcript, time-aligned at 12.5 Hz). Output
side: the model predicts text + assistant-audio (8 codebooks);
user-audio is conditioning-only, never emitted. Codebook 0 (semantic)
and text are predicted from the Temporal Transformer hidden directly;
codebooks 1–7 go through the Depth Transformer sequentially conditioned
on prior codebooks at the same step. Compose for next step: `embed(text)
+ sum_q audio_embd_user_q[user_codes_q] + sum_q
audio_embd_model_q[model_codes_q]` — i.e. **two separate audio-embed
tables** plus the text embedding, summed. "Flexible per-position
weights" refers to having per-codebook-position depth-decoder weights /
per-position output projections — every depth position has its own
linear (and possibly own transformer-layer weights). Refs: [Moshi
paper](https://arxiv.org/abs/2410.00037),
[kyutai-labs/moshi](https://github.com/kyutai-labs/moshi),
[HF transformers docs](https://huggingface.co/docs/transformers/model_doc/moshi).

## 2. Hibiki (Kyutai speech-translation)

Architecturally a Moshi clone: dual-stream Temporal+Depth Transformer,
Mimi codec, **same 12.5 Hz frame rate**, text + 2× 8-codebook audio
streams. Difference is task-only: source-language audio occupies the
"user audio" stream, target-language audio + inner-monologue text
occupy the "model" stream. Decoder-only simultaneous translation:
user-audio stream is fed *continuously* (codes only — never
logits-predicted) and the model emits text+target-audio as soon as
enough context has accumulated. Adds Classifier-Free Guidance scaling
for voice-transfer control (a runtime sampler concern, not an
architectural codec_lm concern). Backbone family: same Helium-derived
Temporal+Depth pair as Moshi. From the codec_lm API perspective Hibiki
is identical to Moshi — same shape of `set_user_audio_codes` + same
`RESIDUAL_DEPTH_AR` depth-decoder pattern with text c0. Ref:
[Hibiki HF card](https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16),
[paper 2502.03382](https://arxiv.org/abs/2502.03382).

## 3. Qwen2-Audio (speech-in, text-out)

Backbone family: **Qwen2** (Qwen2AudioForConditionalGeneration). Audio
input: **Whisper-large-v3 encoder** producing continuous 1280-d
features, projected into the LLM embedding space by a small adapter
MLP and **spliced inline** into the prompt sequence (no cross-attention;
the encoded audio is just a span of "audio embeddings" in the
otherwise-text prompt). Whole-utterance (30-s chunks); not streaming.
Output: **text only**, no audio output, no codebook heads. From
codec_lm's perspective this model has **no `codec_lm` at all** — only
a speech-encoder adapter that produces `[T_audio, hidden]`
token-equivalent embeddings for the backbone. That adapter is the new
"audio-input compose" surface. Refs:
[Qwen2-Audio HF card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct),
[arXiv 2407.10759](https://arxiv.org/abs/2407.10759).

## 4. Qwen2.5-Omni-7B (Thinker-Talker)

Backbone family: **Qwen2.5**. Input audio: Whisper-derived encoder,
continuous features, fed inline into the **Thinker** (the LLM "brain")
alongside text/vision via TMRoPE positional alignment. Output is
**interleaved text + speech**: the Thinker streams text tokens and
hidden states; the **Talker** is a second autoregressive transformer
that consumes (a) Thinker hidden states + (b) Thinker-emitted text
tokens, and emits **codec tokens** from the custom **qwen-tts-tokenizer**
codec. Talker does **not** require word-level/timestamp alignment — it
consumes thinker outputs as a stream. The actual codec token output is
decoded by a streaming **sliding-window DiT → mel → BigVGAN**
detokenizer (that's codec-decode, lives in codec.cpp, not codec_lm).
Per-step structure in the Talker is uncertain from public materials —
appears to be effectively single-codebook (DiT handles the rest of the
fidelity), no depth-decoder. Compose: Thinker hidden + text-token
embedding flow into the Talker; previous Talker codes feed back into
Talker via its own embed table. Refs: [Qwen2.5-Omni
paper](https://arxiv.org/abs/2503.20215),
[blog](https://qwenlm.github.io/blog/qwen2.5-omni/),
[HF card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B).

## 5. Qwen3-Omni MoE (Thinker-Talker)

Backbone family: **Qwen3-MoE** for **both** Thinker and Talker (both
are MoE). Audio input encoder: new **AuT (Audio Transformer)**, trained
from scratch on 20M h of supervised audio — replaces Whisper. AuT still
produces continuous features that are inline-spliced into the Thinker
(same shape of integration as Qwen2.5-Omni, just a different encoder).
Output: text + audio streamed in parallel. Key shift vs 2.5: Talker is
**multi-track / multi-codebook**, autoregressively predicting multiple
codebook layers **via MTP (multi-token-prediction) modules** — i.e. the
Talker emits one frame of N codebooks per step using N parallel MTP
heads, not via a depth-decoder. Code rate dropped to **12.5 Hz** (input
and output). Waveform stage swapped from DiT to lightweight **Code2Wav**
ConvNet (codec.cpp concern). For codec_lm: this is closer to
**`PARALLEL_HEADS_DELAY`** (N parallel heads off a backbone hidden) but
the "backbone" is the MoE Talker, not the Thinker — and the Thinker
hidden state is what feeds the Talker. Refs: [Qwen3-Omni
paper](https://arxiv.org/abs/2509.17765),
[GitHub](https://github.com/QwenLM/Qwen3-Omni).

## 6. Voxtral (Mistral speech-in)

Backbone family: **Ministral-3B** / **Mistral-Small-3.1-24B**. Audio
input: **Whisper-large-v3 encoder** → 50 Hz audio embeddings → 4× MLP
downsampler → 12.5 Hz audio embeddings spliced inline into the LLM
prompt (same inline-embed pattern as Qwen2-Audio/2.5-Omni). 30-second
windows, whole-utterance; supports up to ~40 minutes via the 32k
context. Output: **text only**. No audio output, no codec_lm needed for
output. From codec_lm's perspective Voxtral is structurally identical
to Qwen2-Audio: it's an audio-input *adapter* (Whisper encoder + MLP)
and nothing more. Ref:
[Voxtral paper 2507.13264](https://arxiv.org/abs/2507.13264),
[HF card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507).

## 7. Phi-4-Multimodal-Instruct

Backbone family: **Phi-4-Mini**. Audio encoder: 3 conv layers + 24
**Conformer** blocks (1024 attn dim, 1536 FFN, 16 heads) on 80-d
log-mel @ 10 ms; conv stack subsamples 8× → ~80 ms / token rate fed to
the decoder. Audio projector is a vanilla MLP into the LLM embedding
space. Integration is again **inline embedding** (audio tokens slotted
into the prompt where `<|audio_1|>` lives) but additionally the speech
path activates a **speech-specific LoRA** (~460 M params) on the
language decoder via a Mixture-of-LoRAs design — i.e. the LLM weights
themselves change behaviour depending on whether speech is in the
prompt. Output: **text only**, no audio output. Vision encoder is
analogous (separate encoder + projector + LoRA). For codec_lm: the
audio adapter is again the relevant surface, plus a
**modality-routed LoRA switch** that codec_lm doesn't currently model.
Refs:
[Phi-4-Multimodal HF card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct),
[arXiv 2503.01743](https://arxiv.org/abs/2503.01743).

## 8. LFM2-Audio (full audio-chat mode)

Backbone family: **LFM2** (1.5B). Audio input: **FastConformer encoder**
→ continuous audio embeddings inline-spliced into the LFM2 backbone
(the speech-in compose looks just like Phi-4/Voxtral: encoder +
projector, no codec tokens on the input side). Audio output: **Mimi
codes, Q=8** (1 semantic + 7 acoustic), 12.5 Hz, predicted by an
**RQ-Transformer** (depth-decoder, sequential — same
`RESIDUAL_DEPTH_AR` flavour as CSM/Moshi). Output is **interleaved
text+audio at frame granularity**: text tokens are 1-wide tensors;
audio frames are 8-wide tensors; the LFM2 backbone emits a mixed
sequence of "text token" and "audio frame" positions. Compose for next
backbone input: when current emission was an 8-code audio frame, **sum
the 8 codebook embeddings** (each looked up via
`audio_embedding.embedding` with codebook-id offsets baked into the row
index) into a single 2048-d embedding; when it was text, use the
regular text embed. Already supported by codec_lm output side;
**audio-input** path is missing. Refs: [LFM2-Audio HF
card](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B), [LFM2 tech
report 2511.23404](https://arxiv.org/abs/2511.23404).

## 9. Whisper / WhisperX (reference encoder)

Pure **encoder-decoder ASR**, no codec_lm relevance directly — included
as the reference for "Whisper-style encoder = continuous-feature
audio-input adapter". Input: 30 s log-mel @ 10 ms (80 mel bins for v2,
128 for v3) → 3000 frames → conv stride-2 → 1500 frames → bidirectional
transformer → **`[1500, hidden]`** (1280 for large). That
`[1500, hidden]` tensor is what every model above (Qwen2-Audio,
Qwen2.5-Omni, Voxtral, by replacement of encoder also Qwen3-Omni's AuT
and Phi-4's Conformer and LFM2's FastConformer) consumes — typically
after an MLP projector and a 4× temporal downsample to ~12.5 Hz. **The
codec_lm "audio-in" surface is just "produce `[T_audio, hidden_dim]`
from PCM/mel"**; the encoder identity varies per model. Ref:
[Whisper HF model doc](https://huggingface.co/docs/transformers/en/model_doc/whisper).

---

## Gap analysis — concrete codec_lm extensions

The current API only models the **output** side of an audio LM (hidden
→ codes → next-hidden embed). The survey shows three orthogonal axes
the API currently can't express:

### A. Audio-input adapter surface

Needed for: Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Voxtral, Phi-4-MM,
LFM2-Audio audio-in, Moshi user stream, Hibiki user stream.

Today, codec_lm has `codec_lm_compose_audio_embd(codes -> hidden)`. Add
the dual:

- `codec_lm_compose_audio_input_embd(lm, pcm_or_features, n_frames,
  out_embd[T, hidden])` — runs the model's audio-input adapter
  (Whisper/AuT/Conformer/FastConformer + projector + downsampler) and
  writes a `[T, hidden_dim]` block the caller splices into the
  llama.cpp prompt. The adapter weights live in the GGUF under a new
  namespace `lm.audio_in.*`.
- For **token-based** audio input (Moshi user stream / LFM2-Audio
  audio-in if Mimi-coded):
  `codec_lm_compose_user_audio_codes_embd(lm, codes[n_user_cb],
  out_embd[hidden])` — second audio-embed table, separate from the
  model-audio table. Models needing this: Moshi, Hibiki, LFM2-Audio
  (when prompted with codec tokens directly).
- `codec_lm_info.has_audio_input_adapter` (bool),
  `codec_lm_info.audio_input_frame_rate_hz`,
  `codec_lm_info.audio_input_feature_kind ∈ {pcm, mel, codec_tokens}`
  so the caller knows which entry point to use.
- `codec_lm_info.n_user_codebook` + `codec_lm_info.user_codebook_sizes[]`
  for dual-stream models (Moshi/Hibiki).

### B. Dual-stream / interleaved text+audio output

Needed for: Moshi, Hibiki, LFM2-Audio, Qwen2.5-Omni, Qwen3-Omni.

The current step loop assumes "hidden → N audio codebook heads". Two
things break:

1. **Text head co-emitted at the same step** (Moshi inner-monologue,
   Qwen-Omni Thinker text alongside Talker codes). Today the text head
   lives in the host LLM (llama.cpp) and codec_lm only handles audio.
   For Moshi this is fine — the text token is just the backbone's
   normal output and is passed to codec_lm via the existing
   `codec_lm_state_set_text_context`. But for Qwen2.5/3-Omni, the
   **Thinker** (llama.cpp side) produces text + hidden, and the
   **Talker** is a *separate* transformer that needs its own KV state.
   That isn't a single backbone-hidden-in / codes-out call — it's
   `talker_hidden = talker_forward(thinker_hidden, text_tok); codes =
   talker_heads(talker_hidden)`. Either model this as a second
   `codec_lm` instance (a "talker" kind) or add:
   - `codec_lm_step_begin_with_text(state, h_in, text_tok)` — already
     approximated by `step_begin` + `state_set_text_context`, just make
     text mandatory for talker kinds.
   - A new kind `CODEC_LM_KIND_TALKER_TRANSFORMER` whose state owns a
     full streaming KV cache (not reset per backbone step, unlike
     `RESIDUAL_DEPTH_AR`).

2. **Frame-level interleaving** (LFM2-Audio, Qwen-Omni): the caller
   needs to know per-step whether this position is a *text* step (1
   token from the backbone's text head, no codec_lm call) or an *audio
   frame* step (codec_lm produces N codebook tokens). Add:
   - `codec_lm_state_set_position_kind(state, CODEC_LM_POS_AUDIO |
     CODEC_LM_POS_TEXT)` — controls whether `step_begin` runs the audio
     path or short-circuits.
   - On the compose side: `codec_lm_compose_text_position_embd(lm,
     text_tok, out_embd[hidden])` so the same compose call covers both
     branches uniformly; lets the caller treat positions as opaque.

### C. Streaming KV / multi-cache lifecycle

Needed for: Moshi, Hibiki, Qwen2.5-Omni Talker, Qwen3-Omni Talker.

`RESIDUAL_DEPTH_AR` already resets the depth KV every `step_begin`. The
Moshi/Talker variants need a **persistent** KV cache that lives across
backbone steps (the depth decoder there is a per-step transformer, but
the talker is a multi-step transformer of its own). Add:

- `codec_lm_state_kv_persist(state, bool)` or distinct kinds
  (`RESIDUAL_DEPTH_AR` keeps reset-per-step semantics;
  `TALKER_TRANSFORMER` keeps a persistent KV).
- `codec_lm_state_kv_clear(state)` for explicit end-of-utterance /
  start-of-new-conversation resets.
- `codec_lm_state_clone(state)` for speculative-sampling style
  branching (cheap KV clone via copy-on-write; useful for CFG cond/uncond
  pairs Moshi-style if those move out of caller-managed sampling and
  into the runtime).

### D. Modality-routed LoRA / dual audio-embed tables

- Phi-4-MM activates a speech-specific LoRA on the LLM decoder when
  audio is in the prompt. That's a **llama.cpp** concern, not codec_lm
  — but codec_lm should publish `codec_lm_info.host_modality_lora_name`
  ("speech" / "vision") so the caller knows which LoRA to swap.
- Moshi has two audio-embed tables (user vs model). The current
  `codec_lm_audio_embd(lm, cb_idx, code)` indexes a single table; add a
  stream-id argument: `codec_lm_audio_embd_stream(lm, stream_id,
  cb_idx, code)` with
  `stream_id ∈ {CODEC_LM_STREAM_MODEL, CODEC_LM_STREAM_USER}`.
  `codec_lm_info.has_user_audio_stream` /
  `.has_model_audio_stream` published in `codec_lm_info`.

### E. Minimum new info fields, all together

```c
codec_lm_info {
    // existing fields ...
    bool      has_audio_input_adapter;     // Voxtral, Phi-4, Qwen-Audio, Qwen-Omni, LFM2-Audio
    int32_t   audio_input_frame_rate_hz;   // 12.5, 50, etc.
    int32_t   audio_input_feature_kind;    // pcm / mel / codec_tokens
    bool      has_user_audio_stream;       // Moshi, Hibiki
    bool      has_model_audio_stream;      // Moshi, Hibiki, LFM2-Audio, CSM, Qwen-Omni
    int32_t   n_user_codebook;             // Moshi: 8, Hibiki: 8, else 0
    const int32_t * user_codebook_sizes;   // Moshi: 8x2048
    bool      interleaves_text_audio_positions; // LFM2-Audio, Qwen-Omni
    bool      has_talker_transformer;      // Qwen2.5-Omni, Qwen3-Omni
    const char * host_modality_lora_name;  // Phi-4-MM: "speech"
}
```

### F. Which extension which model needs (summary)

| Extension | Models |
|---|---|
| `compose_audio_input_embd` (continuous features) | Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Voxtral, Phi-4-MM, LFM2-Audio |
| `compose_user_audio_codes_embd` (token features) | Moshi, Hibiki |
| `has_user_audio_stream` + dual embed tables | Moshi, Hibiki |
| `interleaves_text_audio_positions` + position-kind setter | LFM2-Audio, Qwen2.5-Omni, Qwen3-Omni |
| `TALKER_TRANSFORMER` kind (persistent KV) | Qwen2.5-Omni, Qwen3-Omni |
| `host_modality_lora_name` | Phi-4-MM (and any future Mixture-of-LoRAs adapter) |
| Existing `RESIDUAL_DEPTH_AR` (no change) | CSM, Qwen3-TTS, LFM2-Audio output, Moshi codebooks 1–7, Hibiki |
| Existing `PARALLEL_HEADS_DELAY` (no change) | MOSS-TTSD, Qwen3-Omni Talker (MTP heads) |

Note: Whisper itself maps to "audio-input adapter, output-only-text" —
no codec_lm structures change for it, it's just the reference template
for the encoder portion of `lm.audio_in.*` weights.

## Sources

- [Moshi paper (arXiv 2410.00037)](https://arxiv.org/abs/2410.00037),
  [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi),
  [HF Moshi docs](https://huggingface.co/docs/transformers/model_doc/moshi)
- [Hibiki HF card](https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16),
  [Hibiki paper (arXiv 2502.03382)](https://arxiv.org/abs/2502.03382)
- [Qwen2-Audio HF card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct),
  [arXiv 2407.10759](https://arxiv.org/abs/2407.10759)
- [Qwen2.5-Omni paper (arXiv 2503.20215)](https://arxiv.org/abs/2503.20215),
  [blog](https://qwenlm.github.io/blog/qwen2.5-omni/),
  [HF Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [Qwen3-Omni paper (arXiv 2509.17765)](https://arxiv.org/abs/2509.17765),
  [GitHub](https://github.com/QwenLM/Qwen3-Omni)
- [Voxtral paper (arXiv 2507.13264)](https://arxiv.org/abs/2507.13264),
  [HF Voxtral Mini](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)
- [Phi-4-Multimodal HF card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct),
  [Phi-4-Mini paper (arXiv 2503.01743)](https://arxiv.org/abs/2503.01743)
- [LFM2-Audio HF card](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B),
  [LFM2.5-Audio HF card](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B),
  [LFM2 tech report (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404)
- [HF Whisper docs](https://huggingface.co/docs/transformers/en/model_doc/whisper)
