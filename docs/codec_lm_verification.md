# codec_lm — per-model verification

Source-code audits backing the kind taxonomy and tensor schema in
`docs/codec_lm.md`. Every claim below cites a specific file:line in the
canonical implementation. Sources lived under
`.model-src/transformers/src/transformers/models/<name>/` (HF transformers
mirror) and `.model-src/<name>/` (vendor checkouts) at audit time.

## A. CSM (Sesame) — `residual_depth_ar` (weight_layout=`shared`)

Canonical source: `SesameAILabs/csm@daed31e6` `models.py` (203 lines).
HF mirror: `.model-src/transformers/src/transformers/models/csm/`.

**Backbone**: stock Llama-3.2-1B (16 layers, hidden 2048, GQA 32/8,
RoPE base 500k). `models.py:10-23` constructs it via torchtune. llama.cpp
`llama` arch loads it as-is.

**Per-AR-step** (`models.py:132-184`):
1. `embeds = audio_embeddings[input_ids + offsets].sum(dim=2)` — input is
   `(b, s, n_codebook+1, dim)` summed across the channel axis. Per-frame
   mask zeros out unused channels (text-only frame: only channel −1 is
   text; audio-only frame: channels 0..N-1 are audio). Per `generator.py:65,89`
   masks are disjoint, so per position it's either `text_embd[token]` or
   `sum_i audio_embd[code_i + i*audio_vocab]`.
2. `h = backbone(h, ...)` → `(b, s, 2048)`.
3. `last_h = h[:, -1, :]`; `c0_logits = codebook0_head(last_h)`.
4. Sample c0; `c0_embed = audio_embeddings[c0 + 0*audio_vocab]`.
5. `decoder.reset_caches()` (`models.py:170`); `curr_h = cat([last_h, c0_embed], dim=1)`.
6. For i in 1..N-1: `decoder_h = decoder(projection(curr_h), ...)`;
   `ci_logits = decoder_h[:,-1,:] @ audio_head[i-1]`; sample ci;
   `curr_h = audio_embeddings[ci + i*audio_vocab]` (single-token append).

**Two towers, both stripped** (`models.py:48-52`):
```python
def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
```

**HF restructure** (`modeling_csm.py:777-793`): `embed_text_tokens` lives
at the `CsmForConditionalGeneration` top level; `CsmBackboneModelEmbeddings`
holds the audio table; `_tied_weights_keys` ties
`backbone_model.embed_tokens.embed_audio_tokens.weight` to
`depth_decoder.model.embed_tokens.weight` — i.e. one audio embedding
table is shared between backbone-input composition and depth-decoder
input. The math is identical to the canonical version.

**Tensor mapping** (HF source attribute → `lm.*` target):

| HF attribute | Target tensor |
|---|---|
| `backbone_model.embed_tokens.embed_audio_tokens.weight` `(N*V, H=2048)` | converter splits into `lm.audio_embd_{i}.weight` `(V, H)` for i = 0..N-1 |
| `lm_head.weight` `(V, 2048)` | `lm.c0_head.weight` |
| `depth_decoder.codebooks_head.weight` `(N-1, H_d=1024, V)` | converter splits into `lm.depth.heads_{i}.weight` `(V, H_d)` for i = 0..N-2 |
| `depth_decoder.model.inputs_embeds_projector.weight` `(H_d=1024, H=2048)` | `lm.depth.in_proj.weight` |
| `depth_decoder.model.embed_tokens.weight` (tied to backbone audio table) | converter writes once; runtime references the same tensor |
| `depth_decoder.model.layers.{l}.self_attn.{q,k,v,o}_proj.weight` | `lm.depth.blk_{l}.{q,k,v,o}.weight` |
| `depth_decoder.model.layers.{l}.mlp.{gate,up,down}_proj.weight` | `lm.depth.blk_{l}.{ffn_gate,ffn_up,ffn_down}.weight` |
| `depth_decoder.model.layers.{l}.input_layernorm.weight` | `lm.depth.blk_{l}.attn_norm.weight` |
| `depth_decoder.model.layers.{l}.post_attention_layernorm.weight` | `lm.depth.blk_{l}.ffn_norm.weight` |
| `depth_decoder.model.norm.weight` | `lm.depth.output_norm.weight` |

**Metadata**: `n_codebook=32`, `codebook_sizes=[2051]*32` (homogeneous, all
Mimi audio vocab), `delay_pattern=[0]*32`, `weight_layout="shared"`,
`depth_layers=4`, `depth_hidden=1024`, `depth_n_heads=8`, `depth_n_kv_heads=2`,
`depth_rope_theta=500000.0`, `depth_has_in_proj=true`,
`depth_has_qk_norm=false`, `c0_input_modality="audio"`.

**Caller composition pattern**: disjoint by frame. Text-only positions
use llama.cpp `b.token` (backbone's `tok_embd` is set to CSM's
`embed_text_tokens` at convert time); audio-only positions use `b.embd =
codec_lm_compose_audio_embd(codes)`.

## B. MOSS-TTSD-v0.5 — `parallel_heads_delay`

Source: `https://huggingface.co/fnlp/MOSS-TTSD-v0.5` `config.json` +
`modeling_moss_ttsd.py`.

**Backbone**: 28-layer Qwen3-style (hidden 2048, head_dim 128, GQA 16/8,
RoPE 1e6, q_norm/k_norm). llama.cpp `qwen3` loads it as-is.

**Per-step heads** (HF custom modeling): 8 parallel `Linear(hidden,
codebook_size_i)` heads on the backbone's last hidden state. No depth
decoder. Logits computed in parallel; sampling order is independent.

**Heterogeneous codebook sizes**:
`vocab_size_list=[152697, 1025, 1025, 1025, 1025, 1025, 1025, 1025]`.
Channel 0 is the **Qwen3 text vocab + speech-token range** (text+speech
in one vocab, ids `< 151665` = text, ids `[151665, 152689]` = speech).
Channels 1..7 are pure speech vocab (1024 codes + 1 pad).

**Delay pattern**: not stored as a config field; applied at processor
level by `_shift_inputs` which pre-shifts column j backwards by j
positions. Effective `delay_pattern = [0,1,2,3,4,5,6,7]`. Caller is
expected to apply the same shift externally.

**Tensor mapping**:

| HF attribute | Target tensor |
|---|---|
| `model.embedding_list.{i}.weight` for i=0..7 | `lm.audio_embd_{i}.weight` (per-cb sizes) |
| `lm_heads.{i}.weight` for i=0..7 | `lm.heads_{i}.weight` (per-cb output sizes) |
| `model.layers.{l}.{...}` | host-LLM namespace (not `lm.*`) — converter writes a separate `qwen3` GGUF for the backbone |

`tie_word_embeddings=true`: `model.embedding_list[0].weight` ties to
`lm_heads[0].weight`. Converter handles tie at write time.

**Metadata**: `n_codebook=8`,
`codebook_sizes=[152697, 1025, 1025, 1025, 1025, 1025, 1025, 1025]`,
`delay_pattern=[0,1,2,3,4,5,6,7]`, `host_arch="qwen3"`.

**Caller composition pattern**: every backbone position is `sum_i
audio_embd_i[input_ids[..., i]]`. Channel 0's input id is either text
or speech (one combined vocab); channels 1..7 are speech or pad. The
caller does NOT need to extract `tok_embd` from the backbone GGUF — the
sum-of-cb-embeds compose handles all positions because text lives in
channel 0's table.

## C. Qwen3-TTS — `residual_depth_ar` (weight_layout=`shared`)

Source: `.model-src/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py`
`:1015-1268, 1564-1610`; `configuration_qwen3_tts.py:70-258, 391`.

**Backbone**: 20-layer Qwen3-style talker (hidden 1024, head_dim 128, GQA
16/8) + 3D MRoPE on the backbone (multimodal RoPE, time/height/width).
llama.cpp `qwen3` loads layer weights, but **MRoPE on `qwen3` arch
needs verification** — it currently lives in `qwen3vl` / `qwen2vl`. M4
ship is gated on this.

**Code predictor** (the depth decoder, `:1118-1190`): 5-layer Qwen3
transformer with own `codec_embedding[i]` per position, `lm_head[i]` per
position output, plus a `small_to_mtp_projection` for hidden-dim shaping.
Same `weight_layout="shared"` as CSM (one transformer reused at every
depth position).

**Tensor mapping** (selected; full list mirrors CSM with q_norm/k_norm
added):

| HF attribute | Target tensor |
|---|---|
| `talker.model.codec_embedding.weight` (single Embedding for c0 input) | `lm.audio_embd_0.weight` |
| `talker.code_predictor.model.codec_embedding.{i}.weight` for i=0..30 | `lm.audio_embd_{i+1}.weight` |
| `talker.codec_head.weight` | `lm.c0_head.weight` |
| `talker.code_predictor.lm_head.{i}.weight` for i=0..30 | `lm.depth.heads_{i}.weight` |
| `talker.code_predictor.small_to_mtp_projection.{weight,bias}` | `lm.depth.in_proj.{weight,bias}` |
| `talker.code_predictor.model.layers.{l}.self_attn.{q,k,v,o}_proj.weight` | `lm.depth.blk_{l}.{q,k,v,o}.weight` |
| `talker.code_predictor.model.layers.{l}.self_attn.{q,k}_norm.weight` | `lm.depth.blk_{l}.{q,k}_norm.weight` |
| `talker.code_predictor.model.layers.{l}.mlp.{gate,up,down}_proj.weight` | `lm.depth.blk_{l}.{ffn_gate,ffn_up,ffn_down}.weight` |
| `talker.code_predictor.model.layers.{l}.{input,post_attention}_layernorm.weight` | `lm.depth.blk_{l}.{attn,ffn}_norm.weight` |
| `talker.code_predictor.model.norm.weight` | `lm.depth.output_norm.weight` |

**Metadata**: `n_codebook=32`, `codebook_sizes=[4199]*32` (default; verify
from checkpoint), `delay_pattern=[0]*32`, `weight_layout="shared"`,
`depth_layers=5`, `depth_hidden=1024`, `depth_has_qk_norm=true`,
`depth_has_in_proj=true`, `host_arch="qwen3"`,
`c0_input_modality="audio"`.

**Caller composition pattern**: backbone positions are exclusive — each
is either text or audio, not summed. Text positions use `b.token`; audio
positions use `b.embd = codec_lm_audio_embd(0, c0_token)` (only c0 enters
the backbone — c1..c31 are predicted by the depth decoder, not fed back
into the backbone).

## D. Qwen3-Omni-MoE Talker — `residual_depth_ar` (weight_layout=`shared`)

Source: `.model-src/transformers/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`,
classes `Qwen3OmniMoeTalkerForConditionalGeneration` (`:3086-3303`),
`Qwen3OmniMoeCodePredictor` (`:2569-2740`).

**Backbone**: Qwen3-MoE talker — `Qwen3OmniMoeTalkerTextSparseMoeBlock`
with `gate` + `experts.gate_up_proj`/`down_proj` (3D `[n_experts, ...]`)
+ `shared_expert` + `shared_expert_gate`. llama.cpp `qwen3moe` loads it.
Same MRoPE caveat as Qwen3-TTS.

**Code predictor** (`:2569-2664`): structurally identical to Qwen3-TTS's
code_predictor — dense (NOT MoE), small (~5 layers), per-position
embedding tables and lm_head, q_norm/k_norm, in_proj.

Tensor mapping mirrors Qwen3-TTS one-for-one. Metadata identical except
`host_arch="qwen3moe"`.

## E. Moshi (Kyutai) — `residual_depth_ar` (weight_layout=`flexible`)

Source: `.model-src/transformers/src/transformers/models/moshi/modeling_moshi.py`,
classes `MoshiFlexibleLinear` (`:211-247`), `MoshiDepthDecoder` (`:837-846,
955-969`), forward (`:1286-1405`).

**Critical structural difference**: `MoshiFlexibleLinear` is `Linear`-like
but stores weights as `[num_codebooks, out, in]` and gathers
`weight[layer_idx]` per call. Every `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`fc1`, `fc2`, plus `input_projections` and `lm_heads` use this shape.
There is **no shared transformer** reused N times — there are N transformers
with different weights stacked under shared norm parameters.

This breaks the canonical `weight_layout="shared"` runtime: graph builder
must take a `step_idx` and gather the corresponding slice of each weight
tensor. Plumbing-wise this is small (one `ggml_get_rows`-style gather per
linear in the depth-step graph), but it requires runtime support in
`src/lm/residual_depth_ar.cpp`.

**Two further wrinkles** beyond `weight_layout="flexible"`:

1. **Depth-decoder position 0 is text-vocab.** `MoshiDepthDecoder` has
   `text_embed_tokens.weight` of shape `(text_vocab+1, hidden)` for the
   step-0 input, and `embed_tokens[N-1]` `(audio_vocab+1, hidden)` for
   audio steps 1..N-1. Encoded via `codebook_sizes[0] = text_vocab` and
   `c0_input_modality = "text"` metadata.

2. **Backbone audio embeds are dual-stream.** Backbone has both `user`
   audio channels and `model` audio channels (`embed_tokens.{0..2N-1}`),
   summed along with `model.embed_tokens(text_tok)` at every position
   for full-duplex chat. For TTS-only use the `user` half can be omitted
   from the GGUF; for full duplex it's caller's job to compose
   `text_embd + sum_i user_audio_embd_i + sum_i model_audio_embd_i`.

**Tensor mapping** (after `convert_moshi_transformers.py:79-176`):

| HF attribute | Target tensor |
|---|---|
| `decoder.model.embed_tokens.weight` → `depth_decoder.text_embed_tokens.weight` after convert | `lm.depth.text_embd.weight` |
| `depth_decoder.embed_tokens.{i}.weight` for i=0..6 | `lm.audio_embd_{i+1}.weight` |
| `depth_decoder.input_projections.weight` `[N, hidden, in]` | `lm.depth.in_proj.weight` (3D, flexible) |
| `depth_decoder.layers.{l}.self_attn.{q,k,v,o}_proj.linear.weight` `[N, ...]` | `lm.depth.blk_{l}.{q,k,v,o}.weight` (3D) |
| `depth_decoder.layers.{l}.mlp.fc1.weight` `[N, 2*ffn, hidden]` (fused gate+up) | converter splits to `lm.depth.blk_{l}.ffn_gate.weight` + `lm.depth.blk_{l}.ffn_up.weight` (3D each) |
| `depth_decoder.layers.{l}.mlp.fc2.weight` `[N, hidden, ffn]` | `lm.depth.blk_{l}.ffn_down.weight` (3D) |
| `depth_decoder.lm_heads.weight` `[N, audio_vocab, hidden]` | `lm.depth.heads.weight` (3D, flexible — runtime gathers slice `[step_idx]`) |

**Metadata**: `n_codebook=8`,
`codebook_sizes=[text_vocab, audio_vocab, ..., audio_vocab]`,
`delay_pattern=[0]*8`, `weight_layout="flexible"`,
`c0_input_modality="text"`, `host_arch="llama"` (Helium = Llama-style).

**Caller composition pattern**: mixed text + audio at the same position.
Caller has to extract `tok_embd` from the backbone GGUF, look up
`text_embd[text_tok]`, sum with `codec_lm_compose_audio_embd(codes)`, feed
via `b.embd`.

## F. LFM2-Audio-1.5B — `residual_depth_ar` (likely `shared`, blocked)

Source: `https://huggingface.co/LiquidAI/LFM2-Audio-1.5B` `config.json`.
Modeling code is **not published** in the HF repo at time of audit — only
weights and config. From config:

- `codebooks: 8`
- `interleaved_n_text: 6, interleaved_n_audio: 12` — text and audio
  positions interleave at fixed ratio
- `tie_audio_embeddings: false` — all 8 audio embed tables explicit
- `semantic_codebook_factor: 100` — semantic c0 has scaled vocab (TBD)
- `depthformer: {layers: 6, dim: 1024, tie: true}` — depth decoder is
  6-layer, 1024-hidden; embeddings tied (likely `lm.audio_embd_i` ↔
  `lm.depth.embed`)
- backbone: LFM2-1.2B hybrid SSM + attention (16 layers; 6 are
  `full_attention`, rest are short-conv). llama.cpp `lfm2` arch
  supports it.

**Blockers**:
1. No published modeling code → exact tensor naming convention not
   confirmable from source. Must inspect safetensors keys before writing
   the converter.
2. Codec is unidentified (8-codebook). Needs identification before
   the codec side of the GGUF can be authored.

**Estimated metadata**: `n_codebook=8`, `weight_layout="shared"`,
`depth_layers=6`, `depth_hidden=1024`, `depth_has_in_proj=true` (likely;
backbone hidden ≠ 1024), `host_arch="lfm2"`,
`c0_input_modality="audio"`.

**Caller composition pattern**: per-frame interleaving — caller dispatches
text or audio per position (same flow as CSM).

## G. Deferred: MusicGen / Parler-TTS / Dia

All three are encoder-decoder TTS with a stock T5 encoder (loadable in
llama.cpp via `LLM_ARCH_T5ENCODER`) plus a custom audio decoder with
cross-attention. They diverge architecturally on three axes:

| Model | Decoder norm | Decoder FFN | Decoder pos embed | Source |
|---|---|---|---|---|
| MusicGen | LayerNorm + bias | ungated `fc1`/`fc2` | sinusoidal | `musicgen/modeling_musicgen.py:294-326,430-457` |
| Parler-TTS | LayerNorm + bias | ungated `fc1`/`fc2` | sinusoidal **or** rotary (config flag) | `parler-tts/modeling_parler_tts.py:940-981,326-373,1338-1380` |
| Dia | **RMSNorm** | **SwiGLU** (`gate_proj`/`up_proj`/`down_proj`) | **rotary** | `dia/modeling_dia.py:99-114,117-138,513-521,572-584` |

Building one codec_lm kind that handles all three means LayerNorm /
RMSNorm × sinusoidal / rotary × ungated / SwiGLU = effectively a small
general transformer runtime, which exceeds the "specialized auxiliary
module" line the design draws.

Cleaner future paths:

- **Dia is essentially "Llama with cross-attention"** (RMSNorm + RoPE +
  SwiGLU + cross-attn). The right home for this is **upstream
  llama.cpp** as a generic encoder-decoder Llama variant; it's a
  general-purpose extension that benefits more than just audio. Once
  that lands, codec_lm only needs to provide the parallel heads +
  sum-embed adaptor (i.e. `parallel_heads_delay`-equivalent on top of a
  caller-managed second llama_context for the decoder).

- **Parler-TTS** (Bart-with-RoPE variant) and **MusicGen** (Bart) don't
  fit any natural llama.cpp extension. If wanted, revisit at that time —
  decide between upstream contribution or extending codec_lm. MusicGen
  specifically isn't worth dedicated effort given its age.

**Out of scope for v1**: no `xattn_decoder_*` kind, no
`codec_lm_state_set_encoder_kv` API, no `step_begin(NULL)` semantics. The
two-kind design (`parallel_heads_delay` + `residual_depth_ar`) covers
everything M1-M6 actually ships.

## H. Out-of-scope (different paradigms)

These don't fit the codec_lm model under any kind:

- **VALL-E**: AR-on-c0 + NAR-on-c1..N is a different paradigm. Two
  separate networks, NAR runs N-1 forward passes. Needs a third kind
  (`nar_refine`) when implemented — not a codec_lm v1 concern.
- **Bark**: three independent decoder LLMs (semantic, coarse, fine).
  Pipeline-of-codec_lms, not a single codec_lm instance. Out of scope.
- **Higgs Audio v2**: audio side is `parallel_heads_delay`, but DualFFN
  modifies the host LLM transformer layers (audio-specific FFN expert
  per layer). Stock llama.cpp does not load DualFFN; needs llama.cpp
  arch work first.

## Schema decisions taken from this audit

- **Per-cb codebook sizes are required** (MOSS-TTSD c0 is text-vocab
  152697, c1..c7 are 1025).
- **Standardize on unfused `lm.audio_embd_{i}` / `lm.heads_{i}`** in the
  GGUF. Converter splits fused source weights (CSM, Dia, Parler).
- **`lm.depth.in_proj.weight`** is conditional, gated on
  `depth_has_in_proj` metadata (CSM, Qwen3-TTS, Moshi → true; Qwen3-Omni-MoE
  → conditional on hidden dims; LFM2 → likely true).
- **`weight_layout`** has two values: `shared` (CSM, Qwen3-TTS,
  Qwen3-Omni-MoE) and `flexible` (Moshi). Runtime path differs: flexible
  adds `ggml_get_rows` on every weight per depth step. Both are within
  reach; `shared` lands first (M3), `flexible` lands as part of M5 alongside
  Moshi.
- **`c0_input_modality`** captures the Moshi quirk where depth-decoder
  position 0 takes text instead of audio embedding.
- **Per-head `q_norm` / `k_norm`** flagged via `depth_has_qk_norm` (Qwen3
  family).
- **MusicGen / Parler-TTS / Dia** all deferred (see section G). One
  codec_lm kind covering the three would have to span LayerNorm/RMSNorm
  × sinusoidal/rotary × ungated/SwiGLU — past the "specialized auxiliary
  module" line. Dia's natural home is upstream llama.cpp ("Llama with
  cross-attention"); MusicGen / Parler-TTS revisit only if there's a
  concrete need.

## Go/No-Go summary

```
CSM             → SHIP M3 first; residual_depth_ar weight_layout=shared
                  via stock llama.cpp `llama` backbone
MOSS-TTSD       → SHIP M1; parallel_heads_delay delay=[0..7];
                  qwen3 backbone via stock llama.cpp
Qwen3-TTS       → M4 candidate; gated on llama.cpp `qwen3` arch
                  supporting 3D MRoPE
Qwen3-Omni-MoE  → M4 candidate; same MRoPE gate; backbone via `qwen3moe`
Moshi           → M5; needs weight_layout="flexible" runtime + dual-stream
                  text+audio composition in caller
LFM2-Audio      → M6; blocked on safetensors-key inspection (no published
                  modeling code) and codec identification
MusicGen        → out of scope: old, not worth dedicated effort
Parler-TTS      → deferred: revisit only if there's a concrete need;
                  Bart-with-RoPE variant has no natural llama.cpp home
Dia             → deferred: natural home is upstream llama.cpp
                  ("Llama with cross-attention" — Dia's decoder is
                  RMSNorm + RoPE + SwiGLU + cross-attn). Once that
                  arch exists, codec_lm only needs parallel heads.
VALL-E          → out of scope: separate AR+NAR paradigm; needs new kind
Bark            → out of scope: pipeline of three decoder LLMs
Higgs Audio v2  → out of scope: DualFFN host-LLM modification not in llama.cpp
```
