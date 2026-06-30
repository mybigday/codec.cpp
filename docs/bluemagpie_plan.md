# BlueMagpie-TTS integration plan (backlog)

Status: **backlog only — not implemented.** Adds support for OpenFormosa's
[BlueMagpie-TTS](https://huggingface.co/OpenFormosa/BlueMagpie-TTS), a
Taiwanese-Mandarin TTS model. BlueMagpie is **VoxCPM2 with its Text-Semantic
LM swapped from MiniCPM-4 to Barbet** (a Mamba2 + attention hybrid).

This model establishes a **new paradigm** for the repo: a *continuous-latent
autoregressive-diffusion* TTS, as opposed to every model shipped so far, which
is *discrete-codebook*. It does not emit codes, does not sample logits, and its
audio tokenizer is a **continuous VAE**, not a discrete quantizer. The plan
below shows it still decomposes cleanly into the project's three-component
architecture (LM ↔ LM-adaptor ↔ audio tokenizer), at the cost of one new
`codec_lm` kind and a continuous-latent codec output path.

Sources read (verbatim, not summaries):
- `github.com/OpenFormosa/BlueMagpie-TTS` — `src/bluemagpie/{model,tslm,config,adapter,conditioning}.py`
- vendored `src/bluemagpie/_vendor/voxcpm/` — `model/voxcpm2.py`, `modules/{locenc,locdit,audiovae,minicpm4,layers}/`
- `github.com/OpenFormosa/Barbet` — `modeling_barbet.py`, `configuration_barbet.py`
- HF repo `config.json`, `release_metadata.json`, `USAGE.md`

---

## 1. Architecture summary

BlueMagpie is an **AR loop with a data-dependent length** (`model.py::_inference`,
`for i in range(max_len)`, terminated by a stop head). Every step produces **one
continuous latent patch** `[p=4, d=64]`; all patches are concatenated and decoded
to a 48 kHz waveform by AudioVAE at the end.

```
text tokens ──Barbet.embed_tokens──────────────┐
                                                ├─ interleave ─> Barbet backbone ─> H_b
ref/prompt latents ─LocEnc─┬─ enc_to_tslm_proj──┘                   │
                           └─ enc_to_lm_proj (H_v) ──────┐    tslm_adapter (H_b→H_v)
                                                         │          │
                                                         │        FSQ (audio positions)
                                                         │          │ enc_outputs (H_v)
                                 fusion_concat_proj(cat) ┴──────────┤
                                           │                        │
                                      RALM (MiniCPM4 8L)      lm_to_dit_proj
                                           │ res_to_dit_proj        │
                                           └────────── concat ──────┘
                                                       │ mu
                                              LocDiT / UnifiedCFM  ◀── z ~ N(0,1), prev patch (cond)
                                                       │ latent patch [p=4, d=64]
                                                  AudioVAE.decode → 48 kHz
```

### Per-step loop (the heart of it)

Each AR step (`_inference`, after a one-time prefill):

1. **Barbet TSLM** `forward_step` → hidden `H_b` `[1536]`. Mamba2 + attention
   hybrid; `BarbetCache` holds attention K/V (rolling window for sliding-window
   layers), the trailing `d_conv-1` causal-conv inputs per Mamba layer, and the
   Mamba2 selective-scan SSM state.
2. `tslm_adapter` (1536→2048) → **FSQ** (`ScalarQuantizationLayer`, deterministic
   `round(tanh(in_proj(h))·9)/9` then `out_proj`) → `lm_hidden` `[2048]`.
3. **RALM** (`residual_lm`, MiniCPM4 8L, `no_rope`, persistent static KV) →
   `residual_hidden` `[2048]`.
4. `lm_to_dit_proj(lm_hidden)` ‖ `res_to_dit_proj(residual_hidden)` → `mu`.
5. **LocDiT / `UnifiedCFM`**: Euler ODE over `n_timesteps` (default ~9–10),
   each timestep runs the `VoxCPMLocDiTV2` estimator (12L, hidden 1024, in=64)
   over a **2× CFG batch** (cond + uncond), with `cfg_zero_star` scaling and a
   `sway_sampling` `t_span`. Init noise `z ~ N(0,1)`, cond = previous patch.
   Output: one latent patch `[p=4, d=64]`.
6. **Feedback**: `LocEnc(patch)` → `enc_to_tslm_proj` → next-step Barbet input
   embedding; `enc_to_lm_proj` → next-step RALM fusion input.
7. **Stop head** on `lm_hidden` (`stop_proj`→SiLU→`stop_head`→argmax); break if
   `i > min_len` and stop==1.

Prefill (once): `LocEnc` over prompt/ref latents, `Barbet.prefill`,
`RALM` prefill. Then loop. Then `AudioVAE.decode(concat(patches))`.

### Component inventory & config (from `config.json`)

| Module | Shape / config | Role |
|---|---|---|
| **Barbet TSLM** | hidden 1536, 28 L, 16 heads / 2 KV, head_dim 128, `qk_norm`, `qk_clip` (α 0.5, thr 100, off by default), RoPE θ 1e7, **mamba layers [3,7,11,15,19,23,27]** (d_state 64, d_conv 4, expand 2), **global-attn layers [0,4,8,12,16,20,24]**, sliding_window 8192 elsewhere, vocab 114944, tie embeddings | text-semantic LM (the recurrent backbone) |
| `tslm_adapter` | `ProjectionAdapter` 1536→2048, 1 residual block, ffn_mult 2 | bridge H_b→H_v |
| **FSQ** | `ScalarQuantizationLayer` 2048→512(latent)→2048, scale 9, **deterministic** | hidden-state bottleneck (NOT a codebook) |
| **RALM** | MiniCPM4, 8 L (`residual_lm_num_layers`), hidden 2048, `no_rope`, persistent KV | residual acoustic LM |
| `fusion_concat_proj` | Linear 2·2048→2048 | fuse FSQ hidden + LocEnc-lm feat |
| **LocEnc** | `VoxCPMLocEnc`, 12 L, hidden 1024, input_dim 64 | encode latent patch → LM input space |
| **LocDiT** | `VoxCPMLocDiTV2`, 12 L, hidden 1024, in=64; `UnifiedCFM` Euler, σ_min 1e-6, cfg 2.0–2.8 | per-step CFM diffusion → latent patch |
| stop head | Linear 2048→2048 → SiLU → Linear 2048→2 | termination |
| **AudioVAE V2** | enc rates [2,5,8,8] dim 128; dec rates [8,6,5,2,2,2] dim 2048; latent 64; depthwise; cond `scale_bias`; 16 kHz in → 48 kHz out; `use_noise_block=false` | continuous audio ↔ latent |
| `SpeakerProjector` | 192 → 1536 | ECAPA centroid → `[spk]` slot (additive) |

Total weights ≈ 7.75 GB (bf16) + 377 MB AudioVAE.

### Four input modes (`_build_inputs`)
Plain / continuation (`prompt_text`+`prompt_wav`) / reference clip
(`reference_wav`) / speaker vector (`speaker_centroid` [192]). Layout:
`[spk?] [ref prefix?] [text + audio_start] [prompt audio?]`. Speaker slot is
`none`/`null`(learned null)/`centroid`.

### What this is NOT
- **No discrete codes.** FSQ quantizes the LM hidden inline; it is not sampled
  and not fed to the decoder. The decoder consumes the **CFM diffusion latent**.
- **No retrieval.** "RALM" = *Residual Acoustic LM* (an 8-layer MiniCPM), not a
  retrieval-augmented LM. There is no datastore.
- **No logits / no sampling** on the audio path. Generation is diffusion + a
  binary stop classifier.

---

## 2. Mapping onto the three-component architecture

Confirmed clean split (the reason this is buildable as pure ggml):

```
┌─ Box 1: LM (llama.cpp) ─┐  ┌── Box 2: LM-adaptor (codec_lm, NEW kind) ──┐  ┌─ Box 3: tokenizer (codec) ─┐
│ Barbet TSLM             │h │ tslm_adapter→FSQ; RALM(8L); → mu;           │  │ AudioVAE                   │
│ Mamba2+attn hybrid      ├─▶│ LocDiT/CFM (unroll N×2 CFG) → latent patch  ├─▶│ decode: latent seq → 48k   │
│ embd-in, hidden-out     │  │ LocEnc(patch)→proj → feedback embd          │  │ encode: ref/prompt → latent│
│ recurrent state         │◀─┤ stop head                                  │  └────────────────────────────┘
└─────────────────────────┘e │ + speaker encoder (existing API)           │
                              └────────────────────────────────────────────┘
```

| Boundary | discrete models (today) | BlueMagpie (continuous) |
|---|---|---|
| llama.cpp → codec_lm | hidden `h` via `step_begin(h)` | **same**: Barbet hidden `h` |
| codec_lm internal | hidden → logits → sample code | RALM step + CFM solve + RNG → latent patch (no sampling) |
| codec_lm → llama.cpp | `compose_audio_embd(codes)` (row sum) | `LocEnc(patch)`-derived `[hidden_dim]` feedback embd |
| codec_lm → codec | token buffer (int codes) | continuous latent buffer (f32) |
| codec (tokenizer) | discrete decode | AudioVAE continuous decode; encode for ref/prompt |
| speaker | `codec_lm_speaker_*` | **reused as-is** |

**Topology is identical to the existing adaptor; only the data types on the
boundaries change** (codes → continuous latents). That is why this is an
*additive* new kind, not a rewrite of the contract.

### Component placement decisions
- **RALM → Box 2 (codec_lm), not llama.cpp.** It is an 8-layer auxiliary
  transformer tightly coupled to the adaptor's per-step compute (input =
  `fusion_concat_proj(lm_hidden, LocEnc_lm)`, output → DiT). Keeps the llama.cpp
  boundary clean: llama.cpp runs only the "real" semantic LM (Barbet). Matches
  the module-split principle (auxiliary specialized transformers live in
  codec_lm). `codec_lm_state` already holds a depth KV cache — here it becomes
  utterance-persistent instead of per-step-reset.
- **Barbet → Box 1 (llama.cpp)** in **embedding-in / hidden-out** mode
  (`llama_batch.embd` input, embeddings/hidden output, no logits→sampling).
  llama.cpp supports both embedding input and hybrid recurrent memory.

### The one structural cost
The feedback loop crosses all three boxes **every AR step** (Barbet →
codec_lm produces patch → LocEnc → embd back to Barbet). llama.cpp ↔ codec_lm
run in **strict lockstep per step** — no batching across steps like some
discrete models allow. This is an orchestration/perf consideration, not a
blocker; the API shape (`step_begin(h)` → fetch feedback embd → feed llama.cpp
next step) supports it.

---

## 3. codec_lm API extension (the authorized expansion)

New kind alongside `PARALLEL_HEADS_DELAY` / `RESIDUAL_DEPTH_AR`:

```c
enum codec_lm_kind {
    ...
    CODEC_LM_KIND_CONTINUOUS_LATENT_CFM = 3,   // VoxCPM / BlueMagpie family
};
```

New per-step entry points (replace the logits/sample/push dance for this kind):

```c
// hidden -> internally run RALM step + CFM diffusion (+RNG) -> one latent patch.
// `patch` is [patch_size * latent_dim] f32 (e.g. 4*64=256). The diffusion
// timesteps × CFG passes are unrolled inside a single cached graph.
enum codec_status codec_lm_step_generate(
    struct codec_lm_state * st,
    const float *           h_in,        // [hidden_dim] Barbet hidden
    float                   cfg_value,
    int32_t                 n_timesteps,
    const float *           noise,       // [patch_size*latent_dim] or NULL (host RNG)
    float *                 out_patch,   // [patch_size*latent_dim]
    int32_t *               out_stop);   // stop-head argmax (0/1)

// feedback embedding for the NEXT llama.cpp step (LocEnc(patch) -> proj).
// Replaces compose_audio_embd for this kind.
enum codec_status codec_lm_step_feedback_embd(
    struct codec_lm_state * st,
    float *                 out_embd);   // [hidden_dim]
```

`codec_lm_info` gains: `patch_size`, `latent_dim`, `is_continuous` flag (so
callers branch). The discrete fields (`codebook_sizes`, etc.) are left zeroed.

Continuous latent output buffer + codec decode path (Box 3):

```c
// parallel to codec_token_buffer; holds [n_frames, patch_size, latent_dim] f32
struct codec_latent_buffer { float * data; int32_t n_frames, patch_size, latent_dim; };

// AudioVAE continuous decode (and encode for ref/prompt prefill)
enum codec_status codec_decode_latents(struct codec_model *, const struct codec_latent_buffer *, struct codec_audio * out);
enum codec_status codec_encode_latents(struct codec_model *, const struct codec_audio *, struct codec_latent_buffer * out);
```

Speaker path: **no change.** `codec_lm_speaker_*` already returns an
`(n_rows, hidden_dim)` matrix; BlueMagpie's `SpeakerProjector(centroid[192])`
maps onto it, and the ECAPA front-end is the one already shipped for Qwen3-TTS.

---

## 4. ggml graph plan (Box 2 + Box 3, codec.cpp side)

Per the "one model = one graph" rule, **the outer AR loop is host-side** (it is
data-dependent / stop-terminated — the same exemption llama.cpp's decode loop
gets). Fixed-shape compute is unrolled into cached graphs:

- **`CODEC_GRAPH_BLUEMAGPIE_PREFILL`** — LocEnc over prompt latents + RALM
  prefill (Barbet prefill is llama.cpp's job). Fills persistent caches.
- **`CODEC_GRAPH_BLUEMAGPIE_STEP`** — one cached graph: `tslm_adapter` → FSQ →
  RALM `forward_step` (reads/writes persistent KV) → `mu` → **LocDiT CFM
  unrolled over `n_timesteps` × 2 CFG passes** (Chatterbox-S3G's 10-step CFM is
  the precedent) → latent patch → LocEnc(patch) → feedback embd + stop logits.
  RNG `z` injected as an input tensor (host-side, per `feedback_*` reproducible
  noise pattern).
- **`CODEC_GRAPH_BLUEMAGPIE_VAE_DECODE`** — AudioVAE decode of the accumulated
  latent sequence (conv stack, depthwise, `scale_bias` cond, `use_noise_block`
  off → deterministic). Plus `..._VAE_ENCODE` for ref/prompt.

Persistent recurrent state across the host loop:
- Barbet attn-KV window + Mamba conv ring + SSM state → **in llama.cpp** memory.
- RALM static KV → in `codec_lm_state` (backend buffer; watch scheduler-reset /
  allocation-lifetime semantics — see the `codec_graph_prepare_io` warning).

Reusable ops already in the repo: CFM unroll (`codec_op_cfm_causal_resnet_block_tc`
& friends from Chatterbox-S3G), basic transformer blocks, ECAPA speaker encoder.

---

## 5. llama.cpp side: the Barbet arch — gap analysis

Verified against `Barbet/modeling_barbet.py` (789 L) + `configuration_barbet.py`
and the vendored `ggml/include/ggml.h`. **Headline: no new ggml kernels are
required.** `ggml_ssm_conv` and `ggml_ssm_scan` are present, and `ssm_scan` is
the **Mamba2 variant** (`s, x, dt, A, B, C, ids` — the per-sequence state-slot
form llama.cpp's hybrid recurrent models already use). Every other piece is an
existing llama.cpp building block. The work is **arch assembly + a converter**,
not kernel authoring.

### Block motif
Repeating `global, sliding, sliding, mamba2` over 28 layers (7 groups):
`layer_type(i)` = global if `i ∈ {0,4,8,12,16,20,24}`, mamba if
`i ∈ {3,7,11,15,19,23,27}`, else sliding. SwiGLU MLP + pre/post RMSNorm on every
layer; tied embed/LM-head.

### Per-feature gap

| Barbet feature (from source) | llama.cpp equivalent | Gap |
|---|---|---|
| GQA 16 h / 2 kv, **head_dim 128** (q-dim 2048 ≠ hidden 1536, Gemma-style) | `n_embd_head_{k,v}` decoupled from `n_embd` | none |
| per-head **qk RMSNorm** (`q_norm`/`k_norm` on head_dim) | Qwen3 / Gemma3 q/k-norm | none |
| RoPE θ=1e7, optional linear scaling | standard RoPE + linear | none |
| **SWA 8192 vs global** per-layer schedule | per-layer SWA config (Gemma3 alternation) | config plumbing only |
| SwiGLU MLP, RMSNorm, tied embeddings | standard | none |
| **Mamba2 SSD scan** (per-head A/D/dt_bias, 24 h × head_dim 128, n_group 2, d_state 64) | `ggml_ssm_scan` (Mamba2) | none — direct |
| xBC depthwise conv + SiLU | `ggml_ssm_conv` + silu | none |
| **gated RMSNorm** (`norm_before_gate=false`, group_size = inner/groups) | Mamba2 gated norm | none |
| `qk_logit_clip` = `thr·tanh(score/thr)` | Gemma2 attn-logit-softcap | **off in ckpt** (`false`) → skip |
| `attention_sink` (per-head learned sink) | gpt-oss attention sink | **off in ckpt** (`false`) → skip |
| `qk_clip_alpha` | — (muP **training-only**, never referenced at inference) | skip |
| MTP heads | — | **off in ckpt** (`mtp_enabled=false`) → skip |

### The only convert-time reconciliations
Barbet stores the mixer in a non-canonical (but mathematically standard) layout:
- **5 separate in-projs** (`in_proj_{z,x,b,c,dt}`) → fuse into llama.cpp's single
  Mamba2 in_proj in `[z, x, B, C, dt]` order (verify ggml's expected order).
- **3 separate depthwise convs** (`conv_x` 3072 ch, `conv_b`/`conv_c` 128 ch each)
  → concat into one xBC conv weight. Channel total `3072+128+128 = 3328` matches
  llama.cpp's `d_inner + 2·n_group·d_state` exactly; the model's own step-cache
  path already concatenates these weights, confirming equivalence.
- SiLU is applied per-conv in Barbet and post-xBC-conv in llama.cpp Mamba2 — same
  result (depthwise ⇒ per-channel ⇒ split/concat invariant).

### Remaining work (medium, not high)
1. **`LLM_ARCH_BARBET`** graph builder dispatching global / sliding / mamba2 per
   the GGUF layer schedule — assembly of existing ops, modeled on
   `llm_build_falcon_h1` / `llm_build_granite_hybrid` (hybrid attention + recurrent
   memory in one model via `llama_memory_hybrid`).
2. **`convert_hf_to_gguf.py` BarbetModel**: fuse in-projs + convs as above; map
   attention/MLP names; write KV for the layer-type schedule, SWA window, global
   set, head_dim, `qk_norm`, RoPE θ. Audit the `strict=False`-style key set.
3. **embedding-in / hidden-out** decode: feed Barbet via `llama_batch.embd`, read
   the final-norm hidden (not logits); lm_head + MTP unused on the audio path. The
   tied `embed_tokens` *is* used (text-token → embedding at text positions).

The fiddliest part is not any single op but **`llama_memory_hybrid`'s per-sequence
SSM state slots (`ids`) coexisting with rolling-window attention KV** — but that
plumbing already ships for Granite-hybrid / Falcon-H1 / Bamba / Jamba, so it is a
"wire up an existing pattern" task, not new infrastructure.

### GGUF packaging (whole model)
Barbet tensors under the host-LLM namespace; Box-2 modules (RALM, LocEnc, LocDiT,
FSQ, projections, stop head, SpeakerProjector) under `lm.*`; AudioVAE under the
codec namespace; metadata `codec.lm.kind = "continuous_latent_cfm"`,
`codec.lm.has_adaptor = true`.

---

## 6. Conversion notes & parity hooks

- Two checkpoints: `pytorch_model.bin` (everything except AudioVAE) +
  `audiovae.pth`. `from_local` loads with `strict=False` — **cross-check
  missing/unexpected keys** (recurring pitfall: silent default-init under
  `strict=False`).
- FSQ is deterministic at inference (`round(tanh(·)·9)/9`); bake nothing, just
  reproduce the rounding.
- AudioVAE `use_noise_block=false` → deterministic decode (mirror SNAC's
  NoiseBlock-as-identity treatment).
- `cfg_zero_star` (`optimized_scale` = dot/‖·‖²) and the `sway_sampling` t_span
  are small reductions — fold into the CFM graph.
- Determinism for parity: expose a path that takes precomputed `z` noise per
  step (the established RNG-injection pattern), so end-to-end vs PyTorch becomes
  bit-comparable up to the diffusion solver.

### Parity staging (then delete the scaffolding per project convention)
1. AudioVAE decode (continuous, deterministic) vs PyTorch — bit-level.
2. LocEnc + LocDiT single-patch CFM (fixed `z`, fixed `mu`/cond) vs PyTorch.
3. RALM single `forward_step` vs PyTorch.
4. Barbet prefill + `forward_step` vs PyTorch (covered by upstream
   `tests/test_step_equivalence.py` — mirror it).
5. Full host AR loop with injected noise → deterministic e2e.
6. End-to-end smoke: `tests/e2e/bluemagpie_decode_smoke.py` driving the public
   C API (the only artefact that stays).

---

## 7. Effort & risk

| Item | Effort | Risk |
|---|---|---|
| AudioVAE continuous decode graph | M | low (conv stack, deterministic) |
| `codec_latent_buffer` + `codec_decode_latents` | S | low |
| LocEnc + LocDiT CFM step graph (unrolled) | M | medium (precedent: Chatterbox-S3G) |
| RALM 8L + projections + FSQ + stop | S–M | low |
| `CODEC_LM_KIND_CONTINUOUS_LATENT_CFM` + new step API | M | low (additive) |
| Persistent RALM KV across host loop | S | medium (alloc lifetime / scheduler reset) |
| Speaker (ECAPA → SpeakerProjector) | S | low (reuse Qwen3-TTS) |
| **Barbet Mamba2-hybrid arch in llama.cpp** | **L** | **medium** — no new ggml kernels (`ssm_scan`/`ssm_conv` present, Mamba2 variant); assembly + converter only; softcap/sink/MTP all off in ckpt. Fiddliest = `llama_memory_hybrid` wiring (already ships for Granite-hybrid/Falcon-H1) |
| GGUF converter (two stacks, two namespaces) | M | medium (`strict=False` key audit) |

**Verdict.** Buildable as pure ggml with **no** non-neural component and **no**
contract rewrite — it slots into the three-component architecture as an
additive `codec_lm` kind plus a continuous-latent codec path. The dominant cost
and risk is the **Barbet hybrid arch on the llama.cpp side**, which is
orthogonal to the split. Everything on the codec.cpp side reuses existing
patterns (CFM unroll, ECAPA speaker encoder, conv decode graphs).
