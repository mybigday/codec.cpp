# codec_common — generic audio-LM API

Status: landed.  Reference implementation in `common/audio_lm.cpp`,
exercised by `examples/tts-cli` (`build/tts-cli {info,decode,synthesize}`).
Dev-time trace/simulate-* subcommands were removed once `synthesize`
covered every model end-to-end; the e2e smokes (`tests/e2e/`) drive the
same paths via ctypes.

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

### Phase B (landed): `tts-cli synthesize` reference host

`examples/tts-cli.cpp` now drives the full host AR loop against a real
llama.cpp backbone (`--backbone LLAMA.gguf`), proving codec_common's
per-step hooks end-to-end.  Verified E2E (whisper ASR): **CSM** (English,
CER 0.0, stops on `eos_code_c0`) and **BlueMagpie** (continuous CFM,
zh, stops on the diffusion stop head).  Smoke: `tests/e2e/
ttscli_synthesize_smoke.py`.

New codec_common surface (`audio_lm_get_prompt_info` + step-machine
passthroughs `audio_lm_step_{set_text_context,begin,logits,push_code,
finish}`) keeps model-specific prompt assembly + the codebook step
machine inside codec_common, keyed on `codec.lm.host_arch` /
`codec.lm.kind` metadata — the host stays model-agnostic (tokenizer +
llama_decode loop only).

Build integration: the pinned llama.cpp submodule bundles ggml 0.15
under the SAME `libggml.so.0` soname as codec's ggml 0.9 — a runtime ABI
collision.  `cmake/SetupTtsBackbone.cmake` builds llama.cpp + its ggml
STATIC/PIC via ExternalProject, then wraps the archives into one
`libttsbackbone.so` whose dynamic table exports only `llama_*` (ggml_*
hidden via version script).  tts-cli links codec's ggml 0.9 (shared) +
the isolated backbone; the two ggml instances never see each other.
Gated by `-DCODEC_TTS_BACKBONE=ON` (default; CPU-only).

Wired + whisper-verified end-to-end (6/6 synthesize smoke): CSM,
Chatterbox-T3, BlueMagpie, MOSS-TTSD v0.5, MOSS-TTS-Realtime, Qwen3-TTS,
and **LFM2-Audio** (see below).

### LFM2-Audio: sequential text→audio TTS (Flow 5)

LFM2-Audio-1.5B is a *sequential* multimodal LM (`host_arch=lfm2`,
`residual_depth_ar`, `depth_emits_c0=true`, 8 Mimi codebooks).  Unlike the
other depth-AR models it does not start audio immediately — it first
free-runs in TEXT modality and only switches to AUDIO_OUT when it samples
`<|audio_start|>` (id 128).  The wiring (`run_lfm2_sequential` in
`tts-cli.cpp` + the `lfm2` branch of `audio_lm_get_prompt_info`):

* **Prompt** (the flip into audio): system turn
  `"Perform TTS. Use the US male voice."` + user turn with the text +
  assistant open.  With this system prompt the model emits `<|audio_start|>`
  as its very first generated token.  (`Liquid4All/liquid-audio` TTS
  recipe; the ChatState/generate_sequential path in
  `liquid_audio.model.lfm2_audio`.)
* **Text logits from a tied head** — llama.cpp's `lfm2` graph omits the
  output head when embeddings are enabled (`if (!cparams.embeddings)` in
  `models/lfm2.cpp`), and we need embeddings for the depth decoder.  Since
  LFM2 ties its lm_head to `token_embd`, tts-cli recomputes text logits as
  `hidden · token_embd[v]` over the vocab (the returned embeddings are
  already post-final-norm, matching HF's `last_hidden_state`).
* **Audio phase** — the depth decoder emits all 8 codebooks per step; each
  frame feeds back through `compose_audio_codes_embd`
  (`audio_embedding(codes + codebook_offsets).sum(0)`); stop on
  cb0 == EOAudio (2048).
* **Sampling, not greedy** — greedy TTS is *degenerate* for this model
  (the reference itself gets stuck on "Hello."); the reference regime is
  `audio_temperature=0.8, audio_top_k=64`.  With sampling it stops on
  eos_code_c0 at ~4–6 s and English ASR reproduces the input (CER 0.0 with
  a fixed seed).  The existing F16 `lfm_backbone.gguf` is sufficient (no
  BF16 backbone needed once sampled).

### Moshi: why not in `synthesize`

Moshi is **formally out of scope** for `tts-cli synthesize`, for three
concrete blockers:

1. **No backbone asset, no arch support.**  Moshi's backbone is Helium-7B,
   which the pinned llama.cpp submodule does not implement (no `helium`
   arch — grep is empty).  The converter approximates `host_arch=llama`,
   but the real Moshi backbone is not a plain Llama and there is no local
   backbone GGUF (`models/moshi/` ships only `moshiko.gguf` = the Mimi
   codec + depth-decoder adaptor; `~/.cache/huggingface/.../moshiko-*` has
   only a README).
2. **No single-shot audio EOS.**  `moshiko.gguf` carries
   `eos_code_c0 = None` — Moshi is a **full-duplex dialogue** model whose
   audio stream never terminates on a cb0 sentinel.  One-shot TTS is the
   wrong frame; the host AR loop has no stop condition to key on.
3. **Duplex protocol needs a different host loop.**  Moshi interleaves
   input-audio + input-text + output-text + output-audio every step
   (the "MoshiVoice / full duplex (future)" row in *Per-model fit*); that
   is a bidirectional loop, not the unidirectional prompt→audio flow
   `synthesize` implements.  Kyutai's one-shot-TTS sibling is a *different*
   model (`kyutai/dsm`, delayed-streams TTS), not Moshi.

The codec_lm side stays validated: `tests/e2e/moshi_lm_smoke.py` checks the
`depth_decoder.* → lm.depth.*` flexible-tensor conversion + the
`c0_input_modality=text` depth-step graph against HF Moshi (no backbone /
no full generation).  `tts-cli synthesize` now **refuses a Moshi codec_lm
with a clear message** (missing Helium backbone arch + duplex protocol)
instead of running a stop-less loop against an unloadable backbone.

Historic note — the earlier "refuses these cleanly (exit 10)" set (MOSS-*,
Qwen3-TTS, LFM2, Chatterbox) is now all wired; only Moshi is refused.

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
