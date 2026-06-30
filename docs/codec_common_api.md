# codec_common — generic audio-LM API

Status: landed.  Reference implementation in `common/audio_lm.cpp`,
exercised by `examples/tts-cli` (`build/tts-cli {info,decode,synthesize,
trace,simulate-typeA,simulate-typeB,simulate-multicb}`).

A small `codec_common::` namespace in codec.cpp that wraps the
per-model audio-LM machinery (speaker_encode, prompt prefix assembly,
depth-AR / parallel-heads bookkeeping, codes accumulation, audio
decode) behind one interface, in the same spirit as llama.cpp's
`common/` layer.

The application (llama.rn's rn-tts, examples/tts-cli, …) keeps full
ownership of the `llama_decode` loop, the sampler, KV management, and
the host-side batch — `codec_common` only provides:
  * **build-time**: ref-audio encoding + prompt-prefix construction
  * **per-step**: a `observe_token` hook that returns what the host
    should feed at the next step (text passthrough, audio-consumed
    token path, audio-consumed embed-override path, or stop)
  * **end-of-sequence**: codes → PCM decode

The host's own AR loop (e.g. `rn-completion.cpp`'s `doCompletion`) does
NOT get wrapped.  That is the explicit non-negotiable constraint —
codec_common provides per-step hooks, not a loop replacement.

## Boundary

| Layer | Owns |
|---|---|
| **llama.cpp / host AR loop** | tokenize, `llama_decode`, sampler, KV cache, batch construction |
| **codec_common (new)** | ref-audio encoding (speaker_encode), prompt prefix (cond_emb), per-step codes bookkeeping (depth-AR for residual_depth_ar, delay for parallel_heads_delay), `compose_audio_embd`, codes → PCM decode, modality detection |
| **App glue (rn-tts.cpp / examples/tts-cli.cpp)** | decides when to call which codec_common entry, plus minimal per-step branching on `observe_action` result (e.g. switch the next `llama_batch` between token and embd path) |

## Inference patterns

Every audio-output LM currently in scope falls into one of four
categories.  The `observe_token` return type encodes which path the
host should take next:

| Type | Examples | Per-step shape | Next backbone input |
|---|---|---|---|
| **A** Token-based single-cb | OuteTTS, Orpheus, VibeVoice | sample one token from a unified vocab where some IDs encode audio | `embd.push_back(tok)` — standard token path |
| **B** Embed-based single-cb | Chatterbox T3 (once llama.cpp's `chatterbox_t3` arch lands) | sample one speech token from a dedicated speech head | `speech_emb[tok] + speech_pos_emb[step]` via `inputs_embeds` |
| **C** Residual depth-AR multi-cb | CSM, Qwen3-TTS, Moshi, LFM2-Audio | backbone samples cb0; a small depth decoder runs to sample cb1..N-1 conditioned on cb0 | `compose_audio_embd(cb0..cb_{N-1})` via `inputs_embeds` |
| **D** Parallel heads delay multi-cb | MOSS-TTSD | backbone hidden feeds N parallel heads that sample all N codebooks in one step (delay pattern shifts emission per cb) | `compose_audio_embd(cb0..cb_{N-1})` via `inputs_embeds` |

A walks the standard token path; B/C/D need to override the next
input with a precomputed embedding.  That is the only behavioural
split the API needs to expose.

## Public API

```cpp
// include/codec_common.h
namespace codec_common {

// ──────────────────────────────────────────────────────────────────
// Modality
// ──────────────────────────────────────────────────────────────────
enum modality_flag : uint32_t {
    INPUT_TEXT   = 1 << 0,   // model consumes text prompt
    INPUT_AUDIO  = 1 << 1,   // model consumes ref / prompt audio
    OUTPUT_TEXT  = 1 << 2,   // model emits text (audio chat)
    OUTPUT_AUDIO = 1 << 3,   // model emits speech (TTS / audio chat)
};

// ──────────────────────────────────────────────────────────────────
// Params / I/O types
// ──────────────────────────────────────────────────────────────────
struct audio_lm_params {
    std::string codec_path;         // codec / codec_lm GGUF
    bool        use_gpu   = false;
    int32_t     n_threads = 0;
};

// Generic input descriptor — fill whatever fields the caller has;
// model uses what it needs (declared via modality + speaker info).
struct audio_lm_input {
    const float * ref_pcm         = nullptr;
    int32_t       ref_n_samples   = 0;
    int32_t       ref_sample_rate = 0;

    const float * speaker_emb     = nullptr;  // pre-computed (skip VE / ECAPA)
    int32_t       speaker_emb_dim = 0;

    const float * emotion         = nullptr;  // [0, 1] scalar

    std::string   ref_text;                    // ICL voice clone
    std::string   text;                        // synthesis target

    // Model-specific knobs (language_id, speaker_id, …) — small bag.
    std::map<std::string, std::string> extra;
};

// Built prompt: tokens (host feeds via llama_decode) + optional
// prefix embeddings (cond_emb etc.) for the inputs_embeds path.
struct audio_lm_prompt {
    std::vector<llama_token> tokens;

    // Optional: prefix embeddings (rows × hidden) prepended to the
    // sequence via inputs_embeds.  Empty when the model doesn't need
    // a prefix (Type A models).
    std::vector<float> embeds_prefix;
    int32_t            embeds_prefix_rows   = 0;
    int32_t            embeds_prefix_hidden = 0;

    // CFG-using models (Chatterbox): unconditional branch for the
    // pair-decode classifier-free-guidance sampling.
    std::vector<float> embeds_uncond;

    // Sampler hints — training-time defaults; host may override.
    float       default_temperature        = 0.0f;
    float       default_top_p              = 0.0f;
    float       default_min_p              = 0.0f;
    float       default_repetition_penalty = 0.0f;
    float       default_cfg_weight         = 0.0f;
    llama_token start_token                = -1;
    llama_token stop_token                 = -1;
};

struct audio_lm_audio_output {
    std::vector<float> pcm;
    int32_t            sample_rate = 0;
    int32_t            n_channels  = 1;
};

// ──────────────────────────────────────────────────────────────────
// Per-step observe action
// ──────────────────────────────────────────────────────────────────
enum observe_action {
    OBSERVE_PASSTHROUGH,      // text token — host renders, embd.push_back(tok)
    OBSERVE_CONSUMED,         // audio token (Type A) — no render, embd.push_back(tok)
    OBSERVE_CONSUMED_EMBED,   // audio token (Type B/C/D) — no render, next step
                              //                            uses get_next_embed()
                              //                            via inputs_embeds path
    OBSERVE_STOP,             // model emitted stop / codec_lm done
};

// ──────────────────────────────────────────────────────────────────
// Lifecycle
// ──────────────────────────────────────────────────────────────────
struct audio_lm_context;

audio_lm_context * audio_lm_init (const audio_lm_params &);
void               audio_lm_free (audio_lm_context *);
void               audio_lm_reset(audio_lm_context *);   // for next sequence

// ──────────────────────────────────────────────────────────────────
// Capability queries (read once at init from GGUF metadata)
// ──────────────────────────────────────────────────────────────────
uint32_t audio_lm_modality       (const audio_lm_context *);  // modality_flag bitmask
bool     audio_lm_has_speaker_enc(const audio_lm_context *);
int32_t  audio_lm_n_codebook     (const audio_lm_context *);
int32_t  audio_lm_hidden_dim     (const audio_lm_context *);

// ──────────────────────────────────────────────────────────────────
// Prompt build (one shot, before the AR loop)
// ──────────────────────────────────────────────────────────────────
bool audio_lm_build_prompt(audio_lm_context *,
                            const audio_lm_input  &,
                            audio_lm_prompt       *);

// ──────────────────────────────────────────────────────────────────
// Per-step hook (called by host after sampling each backbone token)
// ──────────────────────────────────────────────────────────────────
//
// `last_hidden` is the backbone's hidden state at the just-sampled
// position, fetched by the host via `llama_get_embeddings_ith(-1)`.
// codec_common needs it for Type C/D where it has to push the hidden
// into the depth decoder before sampling cb1..N-1.  Pass nullptr for
// Type A/B (the function will ignore it).
//
// The action return tells the host how to set up the NEXT
// llama_decode call:
//   OBSERVE_PASSTHROUGH / OBSERVE_CONSUMED:
//     standard `embd.push_back(tok)` + `llama_batch_get_one(&tok, 1)`
//   OBSERVE_CONSUMED_EMBED:
//     host calls `get_next_embed()` and feeds the returned vector via
//     `llama_batch_init(...).embd`; the audio token is NOT pushed onto
//     `embd` for KV continuity, the embed is.
//   OBSERVE_STOP:
//     host breaks out of the AR loop and calls `decode_audio`.
observe_action audio_lm_observe_token(
        audio_lm_context *,
        llama_token tok,
        const float * last_hidden,    // may be nullptr for Type A/B
        int32_t       hidden_dim);

// Only valid after OBSERVE_CONSUMED_EMBED; points into audio_lm_context's
// internal buffer; remains valid until the next observe_token / reset call.
const float * audio_lm_get_next_embed(const audio_lm_context *,
                                       int32_t * out_dim);

// ──────────────────────────────────────────────────────────────────
// End of sequence
// ──────────────────────────────────────────────────────────────────
//
// Decode accumulated codes → PCM.  Valid only when modality has
// OUTPUT_AUDIO and the AR loop has produced at least one audio frame.
bool audio_lm_decode_audio(audio_lm_context *, audio_lm_audio_output *);

}  // namespace codec_common
```

## `observe_action` semantics in detail

```
OBSERVE_PASSTHROUGH
  ├─ host: render token text to user
  ├─ host: embd.push_back(tok)
  └─ next llama_decode: batch of {tok}, token path
     (modality.OUTPUT_TEXT branch — text in an audio-chat reply)

OBSERVE_CONSUMED
  ├─ host: do NOT render (audio token, not human-readable)
  ├─ host: embd.push_back(tok)   ← still needed for KV continuity
  └─ next llama_decode: batch of {tok}, token path
     (Type A: OuteTTS / Orpheus — audio rides on the regular token stream)

OBSERVE_CONSUMED_EMBED
  ├─ host: do NOT render
  ├─ host: do NOT embd.push_back(tok)   ← embed replaces it
  ├─ host: call get_next_embed() → (vec, dim)
  └─ next llama_decode: batch with `.embd = vec, .n_tokens = 1`,
                         inputs_embeds path
     (Type B/C/D: codec_common already composed the right embedding
      — for B it's `speech_emb[tok] + speech_pos_emb[step]`; for C/D
      it's `compose_audio_embd(cb0..cb_{N-1})` from the just-completed
      depth-AR / parallel-heads step)

OBSERVE_STOP
  ├─ host: break out of AR loop
  └─ host: call decode_audio() → PCM
```

The `last_hidden` parameter is what makes Type C/D fit inside a single
`observe_token` call: codec_common pushes that hidden into its own
codec_lm step machine, samples cb1..N-1 internally, runs the depth
decoder, and composes the next embed.  The host never sees cb1..N-1
tokens — they live entirely inside codec_common.

## Modality detection

Read once at init from GGUF metadata under `codec.lm.modality.*`:

```
codec.lm.modality.input_text   = bool
codec.lm.modality.input_audio  = bool
codec.lm.modality.output_text  = bool
codec.lm.modality.output_audio = bool
```

`audio_lm_modality()` returns the bitmask.  The host uses it to:

* decide whether to enable the audio path at all (skip for pure text LM)
* enable ref-audio input UI (`INPUT_AUDIO` set)
* set up `decode_audio` at end-of-sequence (`OUTPUT_AUDIO` set)
* honour text-output renderers (`OUTPUT_TEXT` set, e.g. audio-chat
  models where some tokens are text and some are audio — the
  `OBSERVE_PASSTHROUGH` path handles the text ones)

Each model declares its own modality at convert time.  No host-side
hardcoded model lists.

## Per-model fit

| Model | Modality | observe_action pattern | Notes |
|---|---|---|---|
| OuteTTS / Orpheus / VibeVoice | `INPUT_TEXT \| OUTPUT_AUDIO` | mostly CONSUMED, STOP at EOS | Type A — token path throughout |
| Chatterbox T3 (post-llama.cpp arch) | `INPUT_TEXT \| INPUT_AUDIO \| OUTPUT_AUDIO` | CONSUMED_EMBED per step, STOP at speech_eos | Type B — `speech_emb + speech_pos_emb` override |
| CSM | `INPUT_TEXT \| OUTPUT_AUDIO` | CONSUMED_EMBED per step (depth-AR internal), STOP at cb0=audio_eos | Type C — depth decoder hidden inside |
| Qwen3-TTS | `INPUT_TEXT \| INPUT_AUDIO \| OUTPUT_AUDIO` | CONSUMED_EMBED per step, STOP at codec_eos_token_id | Type C — voice clone with x-vector prefix |
| MOSS-TTSD | `INPUT_TEXT \| INPUT_AUDIO \| OUTPUT_AUDIO` | CONSUMED_EMBED per step, STOP at <\|end_of_speech\|> | Type D — delay-pattern multi-cb; some cb0 codes are text → may surface as PASSTHROUGH |
| Audio-chat (Qwen2-Audio, future) | `INPUT_AUDIO \| INPUT_TEXT \| OUTPUT_TEXT` | PASSTHROUGH (text only), STOP at EOS | INPUT_AUDIO triggers ref-audio encode at build time |
| MoshiVoice / full duplex (future) | `INPUT_AUDIO \| INPUT_TEXT \| OUTPUT_TEXT \| OUTPUT_AUDIO` | interleaved PASSTHROUGH + CONSUMED_EMBED per step | needs both branches handled in one loop |

## Host integration (rn-completion shape)

The application loop stays in app code.  rn-completion's `doCompletion`
gains a small per-step branch — minimal, model-agnostic — to honour
`observe_action`:

```cpp
// Inside the existing AR loop, after common_sampler_sample produced `tok`:
const float * h = nullptr;
if (audio_lm_ctx) {
    h = llama_get_embeddings_ith(ctx, -1);
}

auto action = audio_lm_ctx
    ? audio_lm_observe_token(audio_lm_ctx, tok, h, hidden_dim)
    : OBSERVE_PASSTHROUGH;

switch (action) {
    case OBSERVE_PASSTHROUGH:
        generated_text += token_text;        // existing rendering
        embd.push_back(tok);
        break;

    case OBSERVE_CONSUMED:
        // skip rendering; KV-continuity push remains
        embd.push_back(tok);
        break;

    case OBSERVE_CONSUMED_EMBED: {
        int32_t dim;
        const float * vec = audio_lm_get_next_embed(audio_lm_ctx, &dim);
        // Next iteration: build llama_batch via .embd path with `vec`,
        // not via .token path.  The existing batch construction picks
        // this up via a small flag on the completion state.
        pending_embed_override = vec;
        pending_embed_dim      = dim;
        break;
    }

    case OBSERVE_STOP:
        has_next_token = false;
        break;
}
```

The "small flag" + embed-path batch construction is what already exists
in the host as the temporary audio-token patch; it gets reused, just
now driven by codec_common's verdict instead of model-specific
hardcoding.

`tts_wrapper` / `mtmd_wrapper` in llama.rn shrink to thin shims:

| llama.rn surface | Implementation |
|---|---|
| `tts_wrapper::getTTSType()` | reads `audio_lm_modality()` |
| `tts_wrapper::tryAddAudioToken(...)` | calls `audio_lm_observe_token(...)`, returns the action; host loop honours it |
| `tts_wrapper::reset()` | `audio_lm_reset(...)` |
| `mtmd_wrapper::processMedia(audio_path)` | calls `audio_lm_build_prompt(...)` with `ref_pcm`, then prepends `embeds_prefix` via the existing image-embedding inject path |
| end-of-completion → emit WAV | `audio_lm_decode_audio(...)` |

## Why this satisfies the "don't wrap llama_decode" constraint

* `audio_lm_*` never calls `llama_decode`.  All backbone forwards
  happen in app code.
* `audio_lm_*` never owns the AR loop control flow (`while
  (has_next_token)`).  It's purely callback-shaped: build_prompt at
  the start, observe_token per step, decode_audio at the end.
* The `last_hidden` parameter is a pull from app code (`llama_get_
  embeddings_ith`) — codec_common doesn't reach into the llama
  context.  This keeps the dependency one-way: app pulls hidden,
  hands it to codec_common; codec_common returns codes / next embed.

The minor patches the host already applied to `rn-completion.cpp`
(switching between token batch and embd batch based on a flag) stay —
they become the integration surface for codec_common's
`OBSERVE_CONSUMED_EMBED` path, just generalised so they're no longer
model-specific.

## Landing roadmap

1. **`include/codec_common.h` + lifecycle / modality / decode_audio**.
   Wraps existing codec.cpp / codec_lm.h.  No new concepts.  Smoke test
   against an already-working model.
2. **`build_prompt` for Type A**.  Reference impl: OuteTTS or Orpheus.
   Validates token-path round-trip end-to-end through the new API.
3. **`observe_token` Type A path** (`OBSERVE_PASSTHROUGH` /
   `OBSERVE_CONSUMED` / `OBSERVE_STOP`).  Validates the same model still
   works when driven through the API.
4. **`observe_token` Type B path** (`OBSERVE_CONSUMED_EMBED` +
   `get_next_embed`).  Reference impl: Chatterbox once llama.cpp's
   `chatterbox_t3` arch lands.  Validates `inputs_embeds` override.
5. **`observe_token` Type C / D** (depth decoder + parallel heads
   inside `observe_token`).  Reference impls: CSM (C), MOSS-TTSD (D).
   Heaviest milestone — touches the depth-AR KV cache wiring.
6. **llama.rn integration**.  Replace tts_wrapper / mtmd_wrapper
   internals with the codec_common bridge.  rn-completion gets the
   minimal per-step branch + embed-path batch construction generalised
   from the existing temporary patch.

## Open questions / follow-ups

* **`extra` schema** — what keys does each model accept?  Document as
  a registry (`docs/codec_common_input_keys.md`) once the first 2–3
  models are wired so the schema is grounded in real use.
* **CFG pair handling** — for Type B Chatterbox, `embeds_uncond` is the
  unconditional branch.  Does the host run that as a second
  `llama_seq` in the same context, or two contexts?  Decide once
  llama.cpp's `chatterbox_t3` arch lands (depends on whether it
  supports the CFG pair natively).
* **`get_next_embed` ownership** — points into internal buffer, valid
  until next call.  Document the lifetime contract explicitly; hosts
  that need to retain across calls memcpy it out.
* **Streaming PCM output** — `decode_audio` is one-shot.  For long
  generations, expose a `decode_audio_chunk(n_frames)` that returns
  partial PCM after every K codes.  Defer to v2; one-shot is fine for
  reference impl.
* **Thread safety** — `audio_lm_context` per session, single-threaded.
  Document; do not pretend to be re-entrant.
