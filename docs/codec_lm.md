# codec_lm — LM Adaptor design

A thin layer that handles the audio-token side of "non-typical" autoregressive
TTS LLMs that emit one text token plus N audio codebook tokens per AR step.
The host LLM runs in **llama.cpp**; the audio codec runs in **codec.cpp**;
this layer is the bridge that turns the host LLM's hidden state into audio
codes and turns codes back into the embedding the host LLM expects at the
next step.

## Goals & non-goals

**Design principle** — the same split llama.cpp itself uses for vision
(MTMD splits ViT/DPT out of the core LLM): **main AR backbone runs in
llama.cpp; small / specialized / non-typical auxiliary transformer modules
run here.** Concretely, depth decoders for CSM/Qwen3-TTS/Moshi (4-6 layers,
KV reset per backbone step, within-step AR over codebooks) and the
audio-specialized decoders for MusicGen-family (cross-attn + parallel
codebook heads + delay-pattern input) all qualify as auxiliary specialized
modules — they live in codec_lm. The "main backbone" of each model — the
thing that drives the outer AR loop on a token-by-token basis as a
general-purpose LLM — stays in llama.cpp.

**Goals**:
- One narrow C API (`include/codec_lm.h`) that covers every audio-AR LM
  whose main backbone is a stock transformer llama.cpp can load (or a T5
  encoder for MusicGen-family).
- Single GGUF, single load: the LM-adaptor weights ship in the same GGUF as
  the codec they pair with, under the `lm.*` tensor namespace, and are
  picked up via the existing `codec_model` loader. No second mmap.
- Sampling is the caller's job (use `llama_sampler_apply` on a constructed
  `llama_token_data_array`, or anything else); codec_lm exposes raw logits.
- Path selection (when is a position text vs. audio? mixed?) is per-model
  glue in the caller, not generic library logic.

**Non-goals**:
- We do not link llama.cpp from codec.cpp. The integration boundary is
  pure data: caller hands us `float * h_in`, we hand back logits and codes.
- We do not wrap the main-backbone LLM. Caller manages `llama_context`,
  KV cache, text token sampling, etc.
- We do not handle text token embeddings. Most models can use `b.token` and
  let llama.cpp do `tok_embd` lookup natively; the few that mix text + audio
  at the same position (e.g. Moshi) require the caller to extract the
  text-embedding table from the backbone GGUF themselves.
- We do not own VALL-E-style two-stage pipelines or Bark's three independent
  decoder LLMs. Those are different paradigms; if needed they get their own
  kinds (`nar_refine`, etc.) and explicit roadmap entries — never stubs.

## Architecture

```
caller process
├── llama_context                     (host backbone, e.g. csm-backbone.gguf)
│       embeddings = true
│       pooling_type = NONE
│       Inputs:  b.token  for text-only positions
│                b.embd   for audio positions (caller composed via codec_lm)
│       Outputs: hidden state via llama_get_embeddings_ith
│
├── codec_model                       (codec + lm in one GGUF, e.g. csm-codec.gguf)
│   ├── tensors: standard codec.* (Mimi/EnCodec/XY-Tokenizer/...)
│   └── tensors: lm.*                 (audio embed table, c0 head, depth decoder, …)
│
├── codec_lm                          (borrows from codec_model)
│   └── codec_lm_state                (per-generation; KV cache, delay register)
│
└── (caller's sampler stack — typically llama_sampler chains)
```

Per AR step:

1. Caller asks llama.cpp for the backbone's hidden state at the last position.
2. `codec_lm_step_begin(state, h)`.
3. Loop k = 0..n_codebook-1:
   - `codec_lm_step_logits(state, &cb, &n)` → logits pointer.
   - Caller samples (e.g. `llama_sampler_apply`) → code.
   - `codec_lm_step_push_code(state, code)`.
4. `codec_lm_step_finish(state, codes_out)`.
5. Caller composes the next-step input embedding:
   - `codec_lm_compose_audio_embd(lm, codes, audio_embd)`
   - For mixed text + audio positions, caller adds a text embedding row on
     top (model-specific path; see "Per-position composition" below).
6. Caller feeds via `b.embd` and `llama_decode`s once. Repeat.

## The two kinds

Source-verified taxonomy (see `docs/codec_lm_verification.md` for file:line
citations). **No stubs**: a kind only exists once a real model has been
integrated against it.

### `parallel_heads_delay`

N independent `Linear(hidden, codebook_size_i)` heads off the same backbone
hidden state. No intra-step dependency. Optional per-codebook delay shift
register (`delay[N]`); `delay[0] = 0` and `delay[i] >= 0`.

`step_begin(h)` runs all N heads at once. `step_logits` just hands out the
already-computed pointers in order; `step_push_code` records into the delay
register. Compose is `sum_i audio_embd_i[codes[i]]`.

Codebook sizes can be heterogeneous — MOSS-TTSD has `c0 = text_vocab_size
(152697)` and `c1..c7 = 1025` — so `codebook_sizes` is per-codebook, not a
scalar.

Models confirmed to fit with a llama.cpp-compatible backbone: **MOSS-TTSD-v0.5**
(host arch = `qwen3`, delay `[0,1,2,3,4,5,6,7]`, c0 vocab = text + speech
combined so the regular sum-of-cb compose handles everything).

### `residual_depth_ar`

Backbone emits c0 from a single linear head. A small AR transformer with its
own KV cache (reset every backbone step) emits c1..c_{N-1} sequentially.
Each step in the inner loop:
- depth decoder consumes the previously emitted code's audio embedding (or
  for k=1, the `[backbone_hidden, c0_embed]` length-2 sequence)
- per-codebook linear head turns the decoder's hidden into c_k logits

Depth decoder is a Llama-style transformer: RMSNorm + RoPE + GQA + SwiGLU,
with `tok_embeddings` and final output projection set to `nn.Identity()` —
it consumes embeddings and produces hidden states; codebook heads live
outside it.

A **shape projection** `lm.depth.in_proj.weight` is allowed when the depth
decoder's hidden dim differs from the backbone's (CSM: 2048 → 1024; absent
in Qwen3-TTS-style models where dims match).

Two `weight_layout` variants:

- `shared` — depth decoder is one transformer reused for every codebook
  position with the same weights (CSM, Qwen3-TTS, Qwen3-Omni-MoE Talker).
- `flexible` — every linear weight is `[n_codebook, out, in]`, slice
  `[step_idx]` is used for each depth position (Moshi's
  `MoshiFlexibleLinear`). Same graph topology as `shared`, just one
  `ggml_get_rows`-style gather on each weight per step.

Models confirmed to fit with `weight_layout = "shared"`: CSM (Sesame),
Qwen3-TTS, Qwen3-Omni-MoE Talker. Note for Qwen3 family: layers carry
per-head `q_norm` / `k_norm` (Qwen3 RMSNorm), and the talker uses 3D MRoPE
on the backbone — this is a llama.cpp-side capability check, not a
codec_lm concern, but ships are gated on llama.cpp's `qwen3` arch
supporting MRoPE.

Models confirmed to fit with `weight_layout = "flexible"`: Moshi. Moshi
also carries an extra "depth-decoder position 0 is text vocab" wrinkle —
encoded via `codebook_sizes[0] = text_vocab` and `c0_input_modality = "text"`
metadata — and a dual-stream backbone audio embed split (user + model) that
the caller composes outside codec_lm.

LFM2-Audio is conjectured to fit `shared` based on the published
config (depthformer = 6 layers, dim 1024, tied embeds), but the actual
modeling code is not in HF transformers; tensor names are pending a
safetensors-key inspection.

## GGUF schema

Tensor namespace: `lm.*`. Metadata namespace: `codec.lm.*`. Sits alongside
the regular codec.* tensors and metadata in a single GGUF.

GGUF stores `codec.lm.kind` as a string for forward compatibility (old
runtimes fail gracefully on unknown values from newer GGUFs); the C API
exposes it via `enum codec_lm_kind` (`codec_lm_info.kind`). Mapping is
done at `codec_lm_create` time — unrecognised strings yield
`CODEC_LM_KIND_UNKNOWN` and `codec_lm_create` returns NULL. Use
`codec_lm_kind_name(kind)` for the round-trip string form.

```
codec.lm.has_adaptor:        bool
codec.lm.kind:               string                  # "parallel_heads_delay" / "residual_depth_ar"
codec.lm.host_arch:          string                  # informational; "llama" / "qwen3" / "lfm2"
codec.lm.hidden_dim:         i32
codec.lm.audio_embed_dim:    i32                     # usually = hidden_dim
codec.lm.n_codebook:         i32
codec.lm.codebook_sizes:     array<i32>              # [n_codebook]
codec.lm.delay_pattern:      array<i32>              # [n_codebook]; all-zero = no delay

# parallel_heads_delay only:
codec.lm.parallel.fused_audio_embd: bool             # true → single (sum * vocab, dim) table
                                                     # indexed by `code + cb_idx * vocab`
                                                     # false → per-cb tables `lm.audio_embd_{i}.weight`

# residual_depth_ar only:
codec.lm.residual.depth_layers:           i32
codec.lm.residual.depth_hidden:           i32
codec.lm.residual.depth_n_heads:          i32
codec.lm.residual.depth_n_kv_heads:       i32
codec.lm.residual.depth_rope_theta:       f32
codec.lm.residual.depth_rope_scaling:     f32        # set to 1.0 if not used
codec.lm.residual.depth_has_qk_norm:      bool       # Qwen3-style per-head RMSNorm on q/k
codec.lm.residual.depth_has_in_proj:      bool       # true → lm.depth.in_proj.weight present
codec.lm.residual.weight_layout:          string     # "shared" / "flexible"
codec.lm.residual.c0_input_modality:      string     # "audio" / "text" — Moshi has "text"

```

Tensor naming, **parallel_heads_delay**:
```
lm.audio_embd_{i}.weight              # if !fused_audio_embd, i = 0..n_codebook-1
lm.audio_embd.weight                  # if fused_audio_embd; (n_codebook*vocab, dim)
lm.head_{i}.weight                    # i = 0..n_codebook-1; (codebook_size_i, hidden)
lm.head_{i}.bias                      # optional
```

Tensor naming, **residual_depth_ar**:

Schema standardizes on **unfused** per-codebook tables (converter splits
fused weights from CSM/Dia/Parler at convert time):

```
lm.audio_embd_{i}.weight              # i = 0..n_codebook-1; per-cb input embed table
                                      # (CSM: tied between backbone use & depth_decoder
                                      # input — converter writes once and the runtime
                                      # uses the same tensor in both graphs)
lm.c0_head.weight                     # (codebook_sizes[0], hidden) — backbone c0 head
lm.depth.heads_{i}.weight             # i = 0..n_codebook-2; per-cb depth-output head
                                      # (codebook_sizes[i+1], depth_hidden)
lm.depth.in_proj.weight               # (depth_hidden, hidden); only when
                                      #   depth_has_in_proj=true
lm.depth.in_proj.bias                 # optional

lm.depth.blk_{l}.attn_norm.weight     # input_layernorm
lm.depth.blk_{l}.q.weight             # self_attn.q_proj
lm.depth.blk_{l}.k.weight             # self_attn.k_proj
lm.depth.blk_{l}.v.weight             # self_attn.v_proj
lm.depth.blk_{l}.o.weight             # self_attn.o_proj
lm.depth.blk_{l}.q_norm.weight        # only when depth_has_qk_norm=true (Qwen3 family)
lm.depth.blk_{l}.k_norm.weight        # only when depth_has_qk_norm=true
lm.depth.blk_{l}.ffn_norm.weight      # post_attention_layernorm
lm.depth.blk_{l}.ffn_gate.weight      # mlp.gate_proj
lm.depth.blk_{l}.ffn_up.weight        # mlp.up_proj
lm.depth.blk_{l}.ffn_down.weight      # mlp.down_proj
lm.depth.output_norm.weight           # final RMSNorm
```

For `weight_layout = "flexible"` (Moshi), every `lm.depth.blk_{l}.*`
weight gains a leading dimension `[n_codebook, ...]`; the runtime gathers
slice `step_idx` per depth step. `ffn_gate` and `ffn_up` may be fused into
one `[n_codebook, 2*ffn, hidden]` tensor at convert time (Moshi's source
ships them fused as `fc1`); for the unified `shared` schema they're split.

## Per-position composition (caller side)

The choice of how to feed input embeddings is model-specific and lives in
the caller's integration code, not in codec_lm. Three patterns observed:

| Pattern | Models | Caller flow |
|---|---|---|
| **Disjoint by frame** | CSM, MusicGen / Parler-TTS / Dia (decoder side) | Text-only positions: `b.token`. Audio-only positions: `b.embd = codec_lm_compose_audio_embd(codes)`. |
| **Mixed within position** | Moshi (text + 2×audio at same step), Qwen3-Omni Talker (per its forward) | Caller extracts `tok_embd` from backbone GGUF at startup, looks up text row themselves, adds to `codec_lm_compose_audio_embd(codes)`, feeds via `b.embd`. |
| **Cross-attn encoder/decoder** | MusicGen, Parler-TTS, Dia | Encoder is a separate LLM (T5 / similar). Caller runs encoder → cross-attn cache; decoder side is "audio-only by frame" → `b.embd`. |

codec_lm offers no help with text-side composition. The model-specific
caller code is where path selection, mixing, and (if needed) backbone-GGUF
text-embedding extraction live.

## Why no callback for sampling

Earlier drafts had a `codec_lm_sampler_fn` callback. Removed because:

- llama.cpp's `llama_sampler_apply(smpl, llama_token_data_array * cur_p)`
  is vocab-opaque on the statistical samplers (top_k, top_p, min_p, typical,
  temp, dist, penalties, dry — see `llama.h:1269` and the `selected` field
  protocol). The caller can wrap audio logits in 6 lines.
- Forcing a callback hides the construction of `llama_token_data_array`
  from the user. Users who don't want llama_sampler can still use any
  sampler trivially without writing an adapter.
- The state-machine API (`step_logits` returns logits, caller pushes back
  the sampled code) is uniform across `parallel_heads_delay` (all N
  pre-computed) and `residual_depth_ar` (lazy: depth decoder advances
  between codebooks). Either can sample however they want without code
  flowing through codec_lm.

## Why no `codec_lm_load_from_file`

The codec already mmaps and parses the GGUF. Both `codec.*` and `lm.*`
tensors land in the same `model->weights` ggml_context. Re-mmapping or
re-parsing would just bloat memory and complicate cache-key logic for the
existing `codec_graph_cache_get_or_build` infrastructure. `codec_lm_create`
borrows the codec_model's backend, mmap, and metadata; it just registers a
vtable based on `codec.lm.kind`.

## Implementation layout

```
include/codec_lm.h           ← public C API (this file)
src/lm/
  lm.cpp / lm.h              ← public API entry, kind dispatch, state lifecycle
  lm_internal.h              ← codec_lm / codec_lm_state struct + per-kind vtable
  parallel_heads_delay.cpp   ← M1
  residual_depth_ar.cpp      ← M3 (then M4/M5/M6 reuse with metadata-only diffs)
```

Reuses existing runtime:
- `codec_graph_cache_get_or_build` / `codec_graph_compute` (graph build,
  galloc-based allocation, scheduler).
- `codec_perf_log` (so `tools/benchmark.py` picks up codec_lm phases
  automatically — `lm_step_begin`, `lm_depth_step`, etc.).
- `codec_runtime_write_tensor` / `codec_runtime_read_tensor` (input/output
  marshalling — same patterns as the codec models).

New `codec_graph_kind` enum entries are added under the existing dispatch:
`LM_PARALLEL_EXPAND`, `LM_PARALLEL_EMBED`, `LM_DEPTH_BACKBONE_HEAD`,
`LM_DEPTH_DECODER_STEP`.

## Roadmap

Reordered per per-model verification (`docs/codec_lm_verification.md`).
M1 is the simpler kind first; M3 is the bigger architectural test (depth
decoder).

| Milestone | Kind | First model end-to-end | Status |
|---|---|---|---|
| M1 | `parallel_heads_delay` + skeleton + GGUF metadata loader + converter scaffold | **MOSS-TTSD-v0.5** (XY-Tokenizer codec already in repo; backbone via llama.cpp `qwen3`) | go |
| M3 | `residual_depth_ar` with `weight_layout = "shared"` | **CSM** (Mimi codec already in repo; backbone = Llama-3.2-1B via llama.cpp `llama`) | go |
| M4 | M3 reused; verify llama.cpp `qwen3` MRoPE first | **Qwen3-TTS** / **Qwen3-Omni-MoE Talker** | gated on llama.cpp 3D MRoPE check |
| M5 | M3 + `weight_layout = "flexible"` + `c0_input_modality = "text"` + dual-stream backbone audio embeds | **Moshi** | gated on M3 + flexible-weight runtime |
| M6 | M3 reused | **LFM2-Audio** | gated on inspecting safetensors keys for tensor names |

Out of scope for v1:
- **MusicGen** — old, not worth dedicated effort.
- **Parler-TTS / Dia** — both have a custom Bart/Llama-style audio decoder
  with cross-attn to a T5 encoder. The three (MusicGen / Parler-TTS / Dia)
  decoders differ on three axes (LayerNorm vs RMSNorm × sinusoidal vs RoPE
  × ungated FFN vs SwiGLU), which makes a single codec_lm "xattn decoder"
  kind effectively a small generic transformer runtime — past the
  "specialized auxiliary module" line. Cleaner future paths if either is
  wanted: (a) **Dia**: contribute "Llama-with-cross-attn" arch upstream to
  llama.cpp (general-purpose extension, not audio-specific) — Dia's
  decoder is RMSNorm + RoPE + SwiGLU + cross-attn, so this drops it
  straight in; codec_lm provides only the parallel heads + sum-embed.
  (b) **Parler-TTS**: revisit if/when there's a use case; either upstream
  Bart-with-RoPE to llama.cpp or extend codec_lm with the third kind.
- **VALL-E** — separate AR + NAR networks, different paradigm; needs its
  own kind (`nar_refine`) when the time comes.
- **Bark** — three independent decoder LLMs; pipeline of kinds, not one
  codec_lm instance.
- **Higgs Audio v2** — fits parallel_heads_delay on the audio side but
  uses DualFFN host-LLM modifications that llama.cpp doesn't support
  out of the box.

Each milestone ends with a `tests/e2e/<model>_lm_smoke.py` that drives the
public C API end-to-end against the HF reference, scored by audio waveform
correlation. No per-stage parity test scaffolding lands in tree (per the
project's "E2E-only validation" feedback rule).

## Verification appendix

See `docs/codec_lm_verification.md` for file:line citations confirming each
in-scope model fits the design.
