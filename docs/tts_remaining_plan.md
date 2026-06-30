# Plan: wiring MOSS-TTS-Nano, MOSS-TTS-Realtime, LFM2-Audio
        into the existing `backbone + lm_adaptor + codec` structure

## Status snapshot

| Model | Status | Commit |
|---|---|---|
| MOSS-TTS-Realtime | ✅ end-to-end working via codec_lm + llama.cpp | c1c5f68 (profile), f14c771 (converters) |
| LFM2-Audio (TTS-only) | ✅ pipeline working; quality needs tuning | a714dea |
| MOSS-TTS-Nano-100M | 🟡 HF-driven AR loop + codec.cpp decode; native codec_lm pending | 12d5a53 |

The Realtime + LFM2-Audio commits also produced shared infrastructure:
  - `decode_n_q` hook on TTSSession (lets profiles whose codec_lm has
    fewer codebooks than the codec exposes pass that down to
    `codec.decode` — used by LFM2: 8 vs Mimi's 32).
  - Bug fix in `_codec_lm_ctypes.CodecLM.compose_audio_embd` —
    allocate the output buffer at `compose_audio_embed_dim` (not
    `audio_embed_dim`) when the model publishes a separate compose dim.
  - `_layout_moss_tts_local` shared between Realtime + Nano (Nano half
    is stubbed `NotImplementedError` until the GPT-2 depth-block lands).

Status as of this writing: 4 profiles wired (`csm`, `qwen3-tts`,
`moss-ttsd-v0.5`, `moss-ttsd-v0.7`).  This doc maps the 3 remaining
under-3B targets onto the existing structure without inventing a new
top-level kind: each fits as either
(a) a new `lm_adaptor` layout under one of the existing
`codec_lm_kind`s (`residual_depth_ar` / `parallel_heads_delay`), or
(b) a small driver-side extension in the host (examples/tts-cli +
llama.rn's rn-tts) that hangs off codec_common's existing
build_prompt / observe_codes hooks.

## 1. Decompose what each model needs

### MOSS-TTS-Realtime  (~2.3B; Qwen3 backbone + 17-channel emission)

```
HF MossTTSRealtime:
    embed_tokens     : ModuleList([text_embed_152k] + [audio_embed_1027]*16)   # 17 tables
    language_model   : Qwen3Model (28L, 2048d) — already-known shape
    local_transformer: MossTTSRealtimeLocalTransformer (4L, 2048d, Qwen3-flavour)
                       ↳ produces 16 audio codebook logits sequentially
    text head        : (implicit) language_model + tied wte
```

Per AR step the model emits **17 tokens in parallel**:
- cb 0   = TEXT token, sampled by the language_model's lm_head over the
           Qwen3 text vocab (151936-wide).
- cb 1..16 = AUDIO codes (1027-wide each), sampled sequentially by the
             local_transformer depth decoder conditioned on the backbone
             hidden + the already-sampled cb 0..k-1.

Compose for next step: sum of `text_embed(cb_0) + audio_embed_i(cb_i)`
across all 17 tables.

Maps cleanly to **`residual_depth_ar`** with:
- `n_cb = 17`,
- `codebook_sizes = [151936, 1027, 1027, ..., 1027]` (cb-0 is text vocab),
- depth decoder = `local_transformer` (Qwen3-style, 4 layers).

The c0 head ("cb 0 produced by the backbone's lm_head") is already how
CSM works (where c0 is the audio LM head off `backbone_model.embed_tokens`);
here it's the text LM head off `language_model.embed_tokens` instead.
No new codec_lm kind needed.

### MOSS-TTS-Nano-100M (~100M; GPT-2 backbone + 17-channel emission) — DEFERRED

After reading the actual modeling code, two surprises vs the original sketch:

1. `position_embedding_type = "rope"`, NOT absolute.  The GPT-2 backbone
   wraps RoPE inside the attention block (no `wpe` tensor in the
   checkpoint).  This is not llama.cpp's default `gpt2` arch.  Either:
   - Extend the standalone-GGUF converter to emit a custom `gpt2_rope`
     arch (small fork from `gpt2`), OR
   - Treat the backbone as a flavour of llama (RMSNorm → LayerNorm
     swap, SwiGLU → GELU, RoPE kept) so llama.cpp can consume it via
     a per-tensor mapping — but that mis-names ops in ggml.
2. `local_transformer_layers = 1` (not 4 as the side-by-side claimed).
   Smaller scope on the depth side: only 1 GPT-2 block to support
   in `codec_op_lm_gpt2_depth_block` once added.

Concrete remaining work:

```
HF MossTTSNano:
    transformer      : GPT2Model (12L, 768d) — backbone is GPT-2, not Llama
    audio_embeddings : ModuleList([audio_embed_1024]*16)
    text_lm_head     : tied to transformer.wte
    audio_lm_heads   : tied to audio_embeddings[i] for i in 0..15
    local_transformer: MossTTSNanoGPT2Model (4L, 768d, GPT-2-flavour)
                       — local_transformer.wte = Identity (consumes embeds)
```

Same 17-channel emission shape as Realtime, just smaller.  The two
substantive deltas from Realtime are:

1. **Backbone arch is GPT-2** (LayerNorm + GELU + standard MHA), not Qwen3.
   llama.cpp already has a `gpt2` arch — converter just needs to rename
   `transformer.*` into the standalone GPT-2 layout.

2. **Depth decoder is GPT-2-flavoured** (LayerNorm + GELU + standard MHA),
   not Llama-flavoured (RMSNorm + SwiGLU + GQA).  Today
   `src/lm/residual_depth_ar.cpp`'s depth-block helper
   (`codec_op_lm_llama_depth_block`) only emits Llama-shape blocks.

Still maps to **`residual_depth_ar`** in shape; runtime just needs to
dispatch the depth block by `lm.depth.arch` metadata.

### LFM2-Audio (~1.5B; interleaved text/audio chat → TTS)

```
HF Lfm2Audio:
    lfm                : LFM2 backbone (hybrid Mamba+attn, 1.5B)
    audio_embedding    : Embedding(2049 * 8, 2048)   # one fused row per (cb,code)
    audio_lm_heads     : 8 separate heads sharing weights with audio_embedding rows
    text path          : lfm.embed_tokens + tied lm_head (LFM2 text vocab ~65k)
```

LFM2-Audio is the *only* model in the bundle that emits **mode-switched
single-track output** (one position is *either* a text token *or* an
8-codebook audio frame).  Per step the AR loop picks a modality, samples
from the matching head, embeds via the matching table, and the next step
runs the backbone with that single new row.

Codec_lm side fits **`residual_depth_ar` with `n_cb = 8`** unchanged
(this is exactly what `models/lfm2_audio/lfm2_audio.gguf` already does,
proven by `tests/e2e/lfm2_audio_compose_smoke.py`).  The missing piece
is on the **driver** side: the AR loop needs to ask "what modality is
this step?", which today is hard-wired to "always audio".

## 2. What lands where in the existing structure

### A. `scripts/convert-backbone-to-gguf.py` — backbone converters

Add two new preparers next to the existing CSM/Qwen3-TTS/LFM2-Audio/Moshi ones:

- `prep_moss_tts_realtime(src, dst, cfg)`
  - cfg has `language_config` nested; flatten it as the standalone Qwen3
    config (the same way `prep_moss_ttsd` flattens `cfg['language_config']`).
  - Rename `model.language_model.*` → `model.*`.
  - Output: standalone Qwen3 GGUF that llama.cpp's existing `qwen3` arch loads.

- `prep_moss_tts_nano(src, dst, cfg)`
  - cfg has `gpt2_config` nested; flatten it as a standalone GPT-2 config
    (`architectures: ["GPT2LMHeadModel"]`, model_type `gpt2`).
  - Rename `transformer.*` → standalone GPT-2 layout
    (`transformer.h.{l}.*`, `transformer.wte.weight`, `transformer.wpe.weight`,
     `transformer.ln_f.weight`).
  - llama.cpp's `convert_hf_to_gguf.py` handles `GPT2LMHeadModel` natively.

LFM2-Audio backbone is already converted (`models/lfm2_audio/lfm_backbone.gguf`),
done by the existing `prep_lfm2_audio`.

### B. `scripts/converters/lm_adaptor/moss_ttsd.py` — LM adaptor layouts

Today this file covers `MossTTSDForCausalLM` (parallel_heads_delay v0.5/v0.7),
`MossTTSDelayModel` (parallel_heads_delay v1.0/MOSS-TTS), and
`AsteroidTTSModel`.  Add two new layouts:

- `_layout_moss_tts_realtime(cfg, sd)`:
  - `n_codebook = 1 + 16 = 17`
  - `codebook_sizes = [text_vocab=151936, 1027, 1027, ..., 1027]`
  - Per-cb embed: `embed_tokens.{0..16}` (text vocab on cb 0, audio on cb 1..16).
  - Per-cb head:
    - cb 0: tied to `language_model.embed_tokens.weight` (backbone's lm_head).
    - cb 1..16: live inside `local_transformer.lm_heads.{0..15}`.
  - Depth decoder weights: `local_transformer.layers.{0..3}.*` (Qwen3-flavour).
  - Writes `lm.depth.*` tensors + metadata exactly the same as CSM/Qwen3-TTS
    (codec_lm kind stays `residual_depth_ar`).

- `_layout_moss_tts_nano(cfg, sd)`:
  - Same shape as Realtime (`n_codebook = 17`).
  - `codebook_sizes = [text_vocab=gpt2_vocab, 1024, 1024, ..., 1024]`.
  - Depth decoder is GPT-2-flavoured.  Need one new metadata key:
    `codec.lm.depth.arch = "gpt2"` (default `"llama"` for backward compat).
  - Tensor name mapping: `local_transformer.h.{l}.*` → `lm.depth.layers.{l}.*`
    with GPT-2 attn naming (`c_attn` = q+k+v fused, `c_proj` = output, ...).

### C. `src/lm/residual_depth_ar.cpp` — runtime depth-block dispatch

Today `rda_depth_layer` is a thin shim over `codec_op_lm_llama_depth_block`.
To support GPT-2-style depth decoders (MOSS-TTS-Nano only), generalise:

```cpp
// In rda_impl::init(): read codec.lm.depth.arch, default "llama".
// In rda_build_depth_step(): dispatch the per-layer call:
if (impl->depth_arch == DEPTH_ARCH_LLAMA) {
    h = codec_op_lm_llama_depth_block(ctx, h, &layer, ...);
} else if (impl->depth_arch == DEPTH_ARCH_GPT2) {
    h = codec_op_lm_gpt2_depth_block(ctx, h, &layer, ...);
}
```

The new helper lives in `src/ops/lm_attn.{cpp,h}`:

```cpp
// codec_op_lm_gpt2_depth_block:
//   LayerNorm (with bias) → MHA (qkv fused, with bias)
//   LayerNorm (with bias) → FFN (Linear + GELU + Linear, with bias)
// Reuses codec_op_lm_attn_ctx_dth for the attn body (no GQA, no QK-norm).
```

LFM2-Audio + MOSS-TTS-Realtime both use Llama/Qwen3-style depth blocks,
so they reuse the existing path.

### D. Host driver hooks (codec_common + tts-cli / rn-tts)

Three things to add, in increasing order of size:

1. **`info.cb0_is_text` awareness** (small)
   - codec_common's `observe_codes` should default to stripping cb 0
     before handing the frame to `audio_lm_decode_audio` when
     `info.cb0_is_text` is set in the GGUF.  Same goes for the stop
     heuristic: default to `codes[0] in {backbone_eos,
     audio_end_special}` unless the host overrides.

2. **MOSS-TTS-Realtime / -Nano host glue** (medium)
   - Mirrors the MOSS-TTSD wiring in `rn-tts.cpp` / `tts-cli
     synthesize`: the codec_lm side already produces the right N
     codebook codes per step; the host just needs the right prompt
     assembly (BOS / segment tokens) plus a cb-0 mask in the sampler
     so the free-running AR loop stays in speech-space.  No
     codec_common API change required.

3. **LFM2-Audio mode-switching** (largest)
   - The host (llama.cpp via rn-completion / tts-cli) already owns
     both logits and hidden state on every decode, so the runtime
     plumbing is free.  The framework piece is a `position_kind` /
     `step_mode` accessor on codec_lm so the host can ask "does this
     position emit TEXT or an AUDIO_FRAME?" — see
     `docs/audio_in_implementation.md` for the planned C-side API.
     The host then dispatches:
       - TEXT mode: sample 1 token from the backbone's text logits,
         feed back via the token batch path.  No codec_lm call.
       - AUDIO_FRAME mode: existing path — codec_lm step +
         compose_next_embd.
     Mode transitions for LFM2 specifically: detect `<|audio_start|>`
     (128) on the text side → AUDIO_FRAME, detect `2048` on cb-0 →
     TEXT, mirroring `liquid_audio.generate_sequential`.
   - Output assembly:
     - LFM2's pack_codes drops the text-only AR steps from the audio
       stream (only AUDIO_FRAME positions carry decodable codes); decode
       via Mimi as usual.

## 3. Order of attack

Recommended order, smallest blast radius first:

1. **MOSS-TTS-Realtime** (~3–4 h):
   reuses Llama-flavoured depth-block runtime + existing
   `codec.lm.depth.arch="llama"` path; only adds a backbone preparer +
   one new lm_adaptor layout + a new profile class.  Validates the
   "17-channel residual_depth_ar with cb-0 = text head" pattern that
   MOSS-TTS-Nano will also need.

2. **LFM2-Audio** (~4–5 h):
   no converter work needed (everything is already converted); only
   adds the driver-side mode-switch infrastructure
   (`step_mode` hook, `last_text_logits()` on `LlamaBackbone`).  This
   lays the foundation for any future audio-chat model (Qwen-Omni Talker,
   Hibiki, etc. — see `docs/audio_lm_extensions.md` §B).

3. **MOSS-TTS-Nano-100M** (~6–8 h):
   largest lift because of the GPT-2 depth block.  Once Realtime is
   wired, the only new code is `codec_op_lm_gpt2_depth_block` +
   `prep_moss_tts_nano` + `_layout_moss_tts_nano` + a one-line dispatch
   in `rda_build_depth_step`.

After all three: the TTS CLI covers the user-side ask of
"MOSS-TTS 全系列 + LFM2 (<3B)".  Larger MOSS-TTS variants
(v1.0 / Local-Transformer / base) are out of scope per the <3B cap,
but their architectures share the `MossTTSDelayModel` adaptor that's
already wired — adding them is then just a converter run + a registry
entry.

## 4. Risk / dependency notes

- **MOSS-TTS-Nano depth-block parity**: GPT-2's `c_attn` is a fused
  `[q;k;v]` linear, but our existing lm_attn helpers expect separate
  `q_proj` / `k_proj` / `v_proj`.  Either split at convert time
  (preferred — same trick we already do for the Moshi backbone's
  `MoshiGatingMLP.fc1`) or add a fused-qkv variant of
  `codec_op_lm_attn_ctx_dth`.  Splitting at convert time is bit-perfect
  and avoids touching the runtime.

- **LFM2 `last_text_logits()` is a transitional hack**: it works because
  LFM2 ties `lm_head` to `embed_tokens`.  For untied models we'd need
  to extract `lm_head.weight` separately from the backbone GGUF — easy,
  just an extra tensor read at session init.

- **HF dependency**: each new profile still imports HF for tokenizer +
  text-embed extraction (and for MOSS-TTSD-shaped decode, also the HF
  processor).  Pre-computing a "compiled speaker" artefact
  (`alice.compiled.json` carrying `ref_code.npz` + `spk_emb.npy`) is
  the next-natural step to drop HF at inference time, but doesn't
  block the architecture work above.

- **`codec.lm.cb0_is_text` metadata key**: currently inferred via the
  oversize codebook_sizes[0] heuristic in MOSS-TTSD's profile.  Making
  it an explicit GGUF key is cheap and removes a class of footgun for
  the new profiles.
