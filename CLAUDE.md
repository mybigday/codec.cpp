# codec.cpp

This repository is a C/C++ library + CLI that runs several neural audio codecs (currently WavTokenizer-Large, DAC, Mimi, Soprano, NeuCodec, NeMo Nano, Qwen3-TTS-Tokenizer, Chatterbox-S3T, Chatterbox-S3G, XCodec2) using **ggml** graphs so execution can be offloaded via **ggml backends** (CPU/CUDA/Vulkan/Metal/etc.).

The intended architecture is **llama.cpp-style**:
- Build model forward passes as **ggml graphs (ops)**.
- Execute via **ggml_backend + ggml_backend_sched** so backends can offload.
- Avoid bespoke CPU-side tensor math buffers when possible.

---

## High-level layout

- `include/codec.h` — public C API (model load/init, encode/decode, batch decode)
- `src/codec.cpp` — top-level dispatch + model loading + backend selection
- `src/models/` — per-architecture graph builders and glue
  - `wavtokenizer.cpp/.h`
  - `dac.cpp/.h`
  - `mimi.cpp/.h`
- `src/runtime/` — graph cache + execution runtime
  - `graph.cpp/.h` — graph cache keyed by (kind, n_frames, n_q, hop, etc.)
  - `graph_exec.cpp` — ggml_backend scheduler init + graph compute
  - `tensor_utils.*`, `gguf_kv.*` — tensor helpers / metadata
- `src/ops/` — small wrappers around ggml ops + a few custom compositions
  - `ggml_ops.*` — layernorm/groupnorm/linear/unary/snake/pad/crop helpers
  - `conv1d.*`, `convtr1d.*` — conv wrappers (keep only if ggml lacks needed variant)
- `src/batch/` — sequence-level batch container + decode loop (MVP)
- `examples/` — demo/inspection binaries (e.g. batch decode)
- `ggml/` — ggml submodule/subproject

---

## Build

CMake project with ggml as a subdirectory.

Typical CPU build:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Enable GPU backend (example: CUDA):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

Backends are intended to be selected via ggml backend selection logic.

---

## Runtime / backend philosophy

### Backend selection
In `src/codec.cpp` the backend is selected roughly as:
- if `codec_model_params.use_gpu = true`: call `ggml_backend_load_all()` then `ggml_backend_init_best()`
- else: CPU backend

### Scheduler-based execution (important)
`src/runtime/graph_exec.cpp` uses:
- `ggml_backend_sched_new(...)`
- `ggml_backend_sched_graph_compute(...)`

This is the core mechanism enabling CPU/GPU split + offload.

**Key rule:** graphs should be constructed in a way that ggml can place tensors on supported backends; avoid pulling intermediate tensors out to CPU buffers.

---

## Graph caching

Graphs are cached by a small key (see `codec_graph_cache_key` in internal headers), typically including:
- graph kind (encode/decode per model)
- `n_frames`, `n_q`, `hop`, input sizes, latent_dim, etc.

Flow:
1. `codec_graph_cache_get_or_build(...)` builds graph in an eval arena (`ggml_init(no_alloc=true)`).
2. `codec_graph_prepare_io(...)` allocates tensors for the graph in a backend buffer.
3. `codec_graph_compute(...)` runs scheduler compute.

Important constraints:
- When switching to a different graph allocation, scheduler reset may be required to avoid dangling allocations (see comments in `codec_graph_prepare_io`).

---

## ggml op usage

Prefer directly using ggml ops when available.

`src/ops/ggml_ops.cpp` provides small helpers that are either:
- thin wrappers over ggml primitives (`ggml_norm`, `ggml_group_norm`, `ggml_mul_mat`, activations)
- composed ops built from primitives (e.g. DAC `snake` implemented as `x + sin(ax)^2 / a`)

If a needed op is missing in ggml:
1. First try composing from existing ops.
2. If impossible/perf critical, add a custom op (CPU SIMD first) and keep a path to backend support later.

---

## Model files / GGUF

Models are loaded from `.gguf`.

Some tensors that could be generated at runtime should instead be baked into GGUF during conversion (for reproducibility + avoiding runtime FP32→FP16 conversions).

If conversion scripts are involved, regenerate gguf after changes (stale gguf is a common source of “missing tensor” errors).

---

## Conventions / guardrails for changes

- Keep encode/decode numerics stable (unit/regression tests where possible).
- Avoid introducing new CPU-only intermediate buffers; build everything as ggml tensors.
- When touching graph execution / backend scheduler: be careful with allocation lifetimes (`eval_ctx`, scheduler reset semantics).
- Prefer small, reviewable commits.

### One model = one graph

Every model in this repo decodes through a **single cached ggml graph** per public call. CPU-side orchestration is allowed only for: (a) RNG sampling, (b) deterministic LUT precompute that can't be expressed as ggml ops, (c) trivial output marshalling (allocating the output buffer, applying short trim/fade tails).

Whenever you find yourself running graph A → CPU loop → graph B, that is a refactor signal. Loops over fixed iteration counts (CFM ODE Euler steps, CFG cond/uncond passes, n-step diffusion solvers) **must be unrolled into the graph** — they are not "loops" semantically, just fixed-shape compute. The Chatterbox-S3G decode unrolls 10 Euler steps × 2 CFG passes (20 estimator subgraphs) inside a single graph; eval-arena memory grows to multi-GB but graph cache absorbs the build cost.

CPU computation that *looks* like a loop but is really feedforward (NSF source generation, forward STFT, OLA-based iSTFT) belongs in the graph too. The recipes:
- **STFT** → `ggml_conv_1d` against a `[k=n_fft, in=1, out=n_bins]` cos/-sin basis kernel (window pre-baked).
- **iSTFT OLA** → matmul against `[n_bins, n_fft]` synthesis basis, then `ggml_conv_transpose_1d` with an identity `[k, out=1, in=n_fft]` kernel and `stride=hop` to scatter-accumulate frames; divide by an envelope reconstructed by the same convtranspose on a constant `window^2` matrix.
- **Cumsum + nearest-upsample + step-mask** are all ggml primitives; `ggml_arange + sin/cos + concat` builds sinusoidal/rel-pos PE in-graph.

### Lessons from prior model integrations

These recur often enough that they're worth checking *before* you debug a parity failure:

- **Memory literals overflow**. `8u * 1024u * 1024u * 1024u` is `unsigned int` arithmetic and wraps to 0; the graph cache then fails with "ggml_new_object: not enough space". Always cast: `(size_t) 8 * 1024 * 1024 * 1024`.
- **`ggml_mul_mat(a, b)` contracts on `ne[0]`** — both operands' inner dim must agree. For attention `attn @ v` you must permute `v_dth (d, t, h)` → `(t, d, h)` first; see `codec_op_lm_attn_ctx_dth` for the canonical pattern.
- **`ggml_conv_transpose_1d` weight shape is `(k, out, in)`**, not `(k, in, out)`. PyTorch's `(in, out, k)` maps to ggml `ne[0]=k, ne[1]=out, ne[2]=in`; an "identity" OLA kernel needs `weight[k=i, 0, in=i] = 1`.
- **Embedding-table layout**. PyTorch `nn.Embedding.weight` saved as `(V, hidden)` row-major lands in ggml as `ne[0]=hidden, ne[1]=V`. A token's embedding row is at flat offset `ci + token_id * hidden` — easy to invert; use `ggml_get_rows` rather than indexing manually.
- **`F.normalize(x)` ≈ `ggml_rms_norm(x) / sqrt(N)`**. ggml_rms_norm divides by `sqrt(mean(x²))`; PyTorch divides by `sqrt(sum(x²))`. The two differ by `sqrt(N)`.
- **Snake activation expects `[t, c]`**. `codec_op_snake` reshapes alpha as `(1, ne[1])` and broadcasts; if you pass it `[c, t]` it'll silently misalign or assert.
- **PyTorch parametrized weights**. Newer checkpoints (`.parametrizations.weight.original0/1` style weight_norm) won't deserialize via the legacy `weight_g/weight_v` API. Bake `g * v / ||v||` at converter time so the runtime never sees the parametrization.
- **HiFi-GAN iSTFT trim ≠ Vocos iSTFT trim**. `codec_runtime_istft_from_head`'s default trim is `(n_fft − hop)/2` (Vocos/Wavtokenizer). HiFi-GAN's `torch.istft` with `center=True` removes `n_fft/2` per side; pass `trim_pad_override = n_fft/2`.
- **periodic vs symmetric Hann**. `scipy.get_window("hann", n, fftbins=True)` is `0.5 − 0.5·cos(2πn/N)` (periodic). The default Hann in `codec_runtime_istft_from_head` is symmetric (`/(N-1)`); pass an explicit window if the reference uses periodic.
- **Espnet rel-pos PE is interleaved**. `pe[r, 2k] = sin`, `pe[r, 2k+1] = cos` (not `concat([sin, cos])`). To build it in-graph, stack into `[half, 2, n_rows]` then `permute(1, 0, 2, 3) → cont` so the contiguous flatten gives the interleaved layout.
- **strict=False state-dict loads can hide silent default-init**. The HKUSTAudio/xcodec2 checkpoint stores `act.beta` (per BigVGAN) but the upstream `SnakeBeta` was renamed to `act.bias`; HF's `load_state_dict(strict=False)` silently drops the unmatched key and runs the activation with `bias=0` → effective `inv_beta = 1`. If you bake the trained beta into your converter, you'll diverge from the reference. Always cross-check `load_state_dict` reports for unexpected keys, especially when the upstream code does the load with `strict=False`.
- **clangd noise is NOT real**. The codebase shows constant `Adding 'string' to a string does not append` and `lambda has no matching call` diagnostics from clangd because there's no compile_commands.json wired up. **Trust `cmake --build`** — if cmake compiles, the code is fine.

### Parity testing strategy

When matching PyTorch numerically:

- Stage parity: validate each subgraph against PyTorch *before* chaining them. Bit-perfect deterministic stages (encoder, single CFM step) catch shape/permute bugs early; once chained, only RNG-dependent stages will diverge.
- For RNG-dependent paths (CFM noise init, NSF random phase + Gaussian noise), expose a path that takes a precomputed noise tensor instead of sampling — then PyTorch-reference parity becomes deterministic.
- Once the full pipeline matches end-to-end, **delete the standalone parity test binaries and tests**. This project standardises on E2E smoke tests (`tests/e2e/<model>_decode_smoke.py`) that drive the public C API. Per-stage parity scripts are scaffolding, not artefacts.

### Build flow

`cmake --build build -j` is the source of truth. Run it after every non-trivial edit; the linter diagnostics surfaced inline are unreliable. After build, run `tests/e2e/<model>_*_smoke.py` from the repo root via `.venv/bin/python` (PyTorch deps live in that venv).

---

## Useful entry points for Codex

If you need to understand execution:
- `src/runtime/graph_exec.cpp` (scheduler + compute)
- `src/runtime/graph.cpp` (cache + arena)

If you need to understand a model forward:
- `src/models/mimi.cpp` / `dac.cpp` / `wavtokenizer.cpp`

If you need to add/replace an op:
- `src/ops/ggml_ops.cpp` (+ possibly ggml upstream)

---

## Mimi Encoder Migration Status

- Mimi encode path is consolidated into one canonical graph kind: `CODEC_GRAPH_MIMI_ENCODE`.
- The unified graph builder is the only Mimi encode graph path (`frontend -> transformer -> downsample -> unrolled RVQ`).
- Split/legacy graph kinds for Mimi encode stages are removed from runtime graph enums.
- Mimi encode weight writing now targets only the canonical encode graph path.

## XCodec2 Status

- Full encode + decode wired up: PCM → CPU mel-fbank → 16 conformer layers (relative-key Shaw attention) + SemanticEncoder + BigCodec acoustic stack → fc_prior → FSQ encode → indices; decoder mirrors NeuCodec (FSQ → fc_post_a → Vocos backbone with 12 RoFormer layers → ISTFT head).
- Decoder bit-perfect (corr 0.99999981); encoder ≥99.6% codes match HF reference. The drift is FSQ rounding near bucket boundaries — single-feature flips, not systematic bias.
- Mel-fbank feature extractor lives CPU-side in `audio_dsp.cpp` (`codec_runtime_w2v_bert_features`) — matches `SeamlessM4TFeatureExtractor` (Kaldi-2^15 scale, per-frame remove-DC, 0.97 preemphasis, Povey window, log-mel + per-bin ddof=1 normalize, stride-2 stack).
- New shareable graph ops landed: `codec_op_alias_free_snake_beta_tc` (BigVGAN/alias-free SnakeBeta with 12-tap Kaiser FIR) and `codec_op_lm_attn_rel_key_dth` (Shaw rel-key attention via per-t_q batched matmul).
- Encode graph for ≥7s audio fits in CPU RAM; the e2e test caps input at 5s via `max_input_seconds` to keep the runner under iGPU/RAM pressure.
- Quantizer's `act.beta` mismatch (HF reference loads with `strict=False` and silently uses `bias=0`) is matched at convert time by baking `inv_beta = 1/(1+1e-9)` for every Activation1d.

## Chatterbox-S3G Status

- Decode path is a single graph: tokens → encoder → unrolled CFM ODE (10 steps × CFG cond/uncond) → mel → f0_predictor → in-graph NSF source → STFT → main HiFT → in-graph iSTFT → PCM. Builtin-conds path only (no ref_wav embedding yet).
- Reusable building blocks landed in `src/ops/ggml_ops` (`codec_op_basic_transformer_block_tc`, `codec_op_cfm_causal_resnet_block_tc`, `codec_op_causal_block1d_tc`, `codec_op_hifigan_resblock_branch_ct`, `codec_op_sinusoidal_time_emb`, `codec_op_espnet_rel_pos_emb`) and `src/ops/lm_attn` (`codec_op_lm_attn_rel_pos_dth`, `codec_op_rel_shift_espnet`).
- meanflow checkpoints convert and load metadata but `init` rejects them — the meanflow ODE schedule + `time_embed_mixer` aren't wired into the unrolled graph.
