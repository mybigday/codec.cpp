# codec.cpp

This repository is a C/C++ library + CLI that runs neural audio codecs (and the codec side of several TTS models) using **ggml** graphs so execution can be offloaded via **ggml backends** (CPU/CUDA/Vulkan/Metal/etc.). Per-model implementation status lives in `docs/model_status.md`; per-model notes (conversion, quantization, runtime contract) in `docs/models/<model>.md`.

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
  - `audio_frontend.*` — in-graph resampling, STFT/mel, Kaldi fbank
- `src/blocks/` — reusable network blocks (S3 tokenizer v1/v2, CAMPPlus,
  XVectorSincNet, Chatterbox VoiceEncoder, ECAPA-TDNN, HiFT vocoder, CFM
  U-Net + Euler/CFG solver, DiT estimator, WeNet Conformer): `tensor → tensor`
  subgraph builders that a model's or codec_lm's graph composes, never
  standalone graphs
- `src/batch/` — sequence-level batch container + decode loop (MVP)
- `examples/` — demo/inspection binaries (e.g. batch decode)
- `tools/` — installed tools: `tts-cli`, `tts-server` (OpenAI-compatible `/v1/audio/speech`) over `common/tts_runner`; `tools/benchmark/` — perf / TTS timing harnesses
- `python/codec_cpp/` — Python API over the public C headers (ctypes; packaged by the root `pyproject.toml`). `_ffi.py` mirrors `include/*.h` by hand — update it in the same commit as any public struct/function change
- `tests/e2e/` — E2E smokes; drive the library through `codec_cpp`, shared test helpers (assertions, WAV I/O, llama.cpp backbone driver) live in `tests/e2e/_common.py`
- `ggml/` — ggml submodule/subproject

---

## Build

CMake project with ggml as a subdirectory.

Env-var debug hooks (`CODEC_PERF_LOG`, `CODEC_OP_PROFILE`, `CODEC_FORCE_COMPOSITE_OPS`, `CODEC_NO_OP_OFFLOAD`, `CODEC_AUDIT_UPCAST`) are compiled in only with `-DCODEC_DEBUG_TOOLS=ON`; production builds carry none of them. New debug env flags go behind the same `#ifdef CODEC_DEBUG_TOOLS`.

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
- Lookup tables a builder needs (DFT bases, resampler kernels) are graph constants: `codec_graph_const*` inside the build_fn; `codec_graph_prepare_io` writes them. Blocks stay self-contained, callers don't fill tables by hand.
- `codec_graph_cpu_scope` pins the ops built inside it to the CPU backend while the rest of the graph offloads. Use it for precision-critical front-ends only (see the Vulkan matmul lesson below); end the scope on a non-view tensor.

---

## ggml op usage

Prefer directly using ggml ops when available.

`src/ops/ggml_ops.cpp` provides small helpers that are either:
- thin wrappers over ggml primitives (`ggml_norm`, `ggml_group_norm`, `ggml_mul_mat`, activations)
- composed ops built from primitives (e.g. DAC `snake` implemented as `x + sin(ax)^2 / a`)

If a needed op is missing in ggml:
1. First try composing from existing ops.
2. If impossible/perf critical, add a custom op (CPU SIMD first) and keep a path to backend support later.

**Pick ops every target backend runs — Metal and Vulkan, not just CPU.** An unsupported op does not fail: the scheduler silently moves it to CPU, splitting the graph (and a view crossing that boundary can crash). Check `ggml_metal_device_supports_op` (`ggml/src/ggml-metal/ggml-metal-device.m`) and `ggml_backend_vk_device_supports_op` before choosing an op, including their dtype/contiguity conditions. Known gaps and the replacements in `src/ops`:
- Metal: no `ROUND`/`FLOOR` unary (`codec_op_round_bounded`: STEP-based, bit-exact with `ggml_round` incl. ties, for FSQ-style bounded inputs), `PAD` only pads at the end (`codec_op_pad_1d` prepends a zero constant), binary ops need an F32 src0 with contiguous rows.
- Vulkan: no `PAD_REFLECT_1D` (`codec_op_pad_reflect_tc`), no `ELU` (`codec_op_unary(..., CODEC_UNARY_ELU)`), no `POOL_1D` (`codec_op_pool_1d`).
- OpenCL (llama.rn's Android GPU; Adreno only, ggml-opencl refuses Mali): no `SIN`/`COS`/`LEAKY_RELU`/`ARANGE`/`POOL_*`/`ARGMAX`/`CONV_TRANSPOSE_1D`/`LOG`. Use `codec_op_sin`/`codec_op_cos`/`codec_op_leaky_relu`/`codec_op_arange`/`codec_op_pool_1d` and `codec_convtr1d` (GEMM path); they keep the native op where `codec_graph_backend_supports` says it runs and compose otherwise. `CODEC_FORCE_COMPOSITE_OPS=1` exercises the fallbacks on CPU (debug-tools build).
- Both: `ggml_clamp` is in-place (`codec_op_clamp_min`).

---

## Model files / GGUF

Models are loaded from `.gguf`.

Some tensors that could be generated at runtime should instead be baked into GGUF during conversion (for reproducibility + avoiding runtime FP32→FP16 conversions).

If conversion scripts are involved, regenerate gguf after changes (stale gguf is a common source of “missing tensor” errors).

Storage precision is a converter decision (`scripts/utils/precision.py`): `--outtype` (F32/F16/BF16/AUTO) and `--quantization` (Q8_0/Q5_K_M/Q4_K_M) for the codec part, `--lm-outtype` / `--lm-quantization` for the LM-side adaptor (`lm.*`). A float type is never wider than the checkpoint (F32 from a BF16 checkpoint stays BF16); generated DSP tables stay F32. In converter code `"F16"` means "this part's float type" and `"F32"` means "F32 capped at the checkpoint's precision".

---

## Conventions / guardrails for changes

- Keep encode/decode numerics stable (unit/regression tests where possible).
- Compute at the stored precision: a weight runs at its GGUF type (F32 / F16 / BF16 / block-quantized) in every matmul, conv, convtr and codebook search. Never cast a weight operand (`codec_graph_stored_weight` recovers the stored tensor behind a `codec_graph_weight` cast; widening a bias / norm scale for an elementwise op is lossless and fine). Precision is chosen at conversion: a tensor whose computation needs a precision is stored at it (mel filterbanks / windows, codebooks, weights fed activations beyond F16 range such as the Qwen3-TTS depth ffn_down, SNAC's encoder), and `scripts/quantize-gguf.py` writes any per-tensor mix (including the quantizable conv layouts: conv1d `(K*C_in, 1, C_out)`, ConvTranspose `(C_in, C_out, K)`). Where ggml cannot run a type in a layout, fail loudly; never dequantize silently.
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

### Backbone vs codec.cpp boundary (llama.cpp ↔ codec_lm)

Classify every transformer in a new TTS / audio-LM model before writing code:

- **Backbone = the main AR generator's transformer stack** (the model that emits audio tokens / latents step by step from text). It **always runs in llama.cpp** — even when its arch is non-stock (relative-position attention, cross-attention, custom norms). Missing arch → add a llama.cpp patch under `patches/<model>-llamacpp/`; never build a parallel AR runtime for it in codec.cpp. llama.cpp runs it embd-in / hidden-out (`docs/codec_lm.md`).
- **codec_lm (codec.cpp)** = the adaptor between backbone and codec: audio embedding tables / compose, logit heads, and small transformers **downstream of the backbone** — per-step depth decoders (CSM / Qwen3-TTS / Moshi), flow/CFM heads, etc. Not generators in their own right.
- **Run-once encoders upstream of the backbone** (speaker encoders, `cond_enc`, ViT-style prompt encoders) live in codec.cpp and hand the backbone prefill embeddings, as llama.cpp's MTMD does for vision encoders.
- Self-contained exceptions (Pocket-TTS FlowLM) predate this rule; don't cite them as precedent.

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
- **`codec_op_snake` clamps alpha to positive — wrong for SNAC**.  The shared helper does `clamp(alpha, eps, FLT_MAX)` (BigVGAN convention), which flips negative-alpha channels into a near-singular `1/eps` factor.  SNAC's reference formula is `(alpha + 1e-9).reciprocal()` and the trained checkpoints have negative alphas.  Use a local Snake variant that preserves sign (multiply-then-add-eps in the denominator) instead of reaching for `codec_op_snake`.
- **codec_token_buffer uses (T, Q) interleaved layout**, not (Q, T).  `codec_example_load_npy_i32_2d_tq` transposes a (n_q, n_frames) NPY into `data[t*n_q + q]`, and the save helper expects the same.  When implementing a multi-codebook model, mirror NeuCodec/XCodec2's `tokens->data[t*n_q + q]` access pattern — `data[q*n_frames + t]` reads garbage.
- **strict=False state-dict loads can hide silent default-init**. The HKUSTAudio/xcodec2 checkpoint stores `act.beta` (per BigVGAN) but the upstream `SnakeBeta` was renamed to `act.bias`; HF's `load_state_dict(strict=False)` silently drops the unmatched key and runs the activation with `bias=0` → effective `inv_beta = 1`. If you bake the trained beta into your converter, you'll diverge from the reference. Always cross-check `load_state_dict` reports for unexpected keys, especially when the upstream code does the load with `strict=False`.
- **Offloaded F32 matmuls are not F32-exact**. ggml-vulkan's `matmul_f32_f32` stages tiles in fp16 shared memory whenever the device has fp16 (`GGML_PREC_F32` does not change it): ~2.5e-4 relative error. Networks shrug it off, but framed DFTs / filterbanks on raw audio don't — quiet bins come from cancelling large terms, and the following log/InstanceNorm amplifies them (speaker mel max |Δ| 0.1 → 4e-5 once pinned to CPU). Put such front-ends in a `codec_graph_cpu_scope`.
- **ONNX exports can differ from the PyTorch module**. CosyVoice's `campplus.onnx` averages a partial last 100-frame CAM segment over the full window (ONNX `AveragePool` divides by the kernel size; PyTorch `avg_pool1d(ceil_mode=True)` by the frames present) — x-vector cos 0.996 until `codec_block_campplus(..., onnx_seg_pooling=true)`. Compare the ONNX against the PyTorch module at a few input lengths before assuming they are the same model.
- **WeNet/CosyVoice encoders with `static_chunk_size = 1` are causal**. A call like `encoder(x, len, decoding_chunk_size=1)` builds a chunk-of-one mask (each frame sees itself + all left frames) even though the module looks bidirectional; check `add_optional_chunk_mask` arguments before porting any WeNet encoder.
- **llama.cpp's HF converter mangles non-ASCII added tokens**. `get_vocab_base` round-trips added tokens through `tokenizer.decode(tokenizer.encode(t))`; Qwen2's byte-level decoder turns `[iāng]` into `[i\x01ng]`, the collisions then trip `GGML_ASSERT(id_to_token.size() == token_to_id.size())` at load. Patch the round trip out when the added tokens are used for tokenization (`prep_cosyvoice` in `scripts/convert-backbone-to-gguf.py`).
- **Metal `UPSCALE` is nearest-only**. Bilinear `ggml_interpolate` silently splits the graph to CPU there; express linear resampling as a matmul with a constant weight matrix (arbitrary sizes) or broadcast blends of neighbour differences (integer factors), and nearest picks as `ggml_get_rows` with host-computed indices (exact PyTorch rounding).
- **clangd noise is NOT real**. The codebase shows constant `Adding 'string' to a string does not append` and `lambda has no matching call` diagnostics from clangd because there's no compile_commands.json wired up. **Trust `cmake --build`** — if cmake compiles, the code is fine.
- **Fixed-size positional-embedding tables ⇒ the model chunks long inputs**. A `sinusoids(max_pos, d)` buffer sliced as `pos_emb[:tgt_len]` (XY-Tokenizer, Whisper-style encoders) only has `max_pos` rows; feed it a sequence longer than that and `ggml_view_2d(pos, ne0, t, …)` overruns the source and trips `GGML_ASSERT(data_size + view_offs <= ggml_nbytes(view_src))` at *graph-build* time (not compute). Before assuming a resource-sizing bug, check the reference's `forward`: if it does `pos[:tgt_len] if tgt_len < pos.shape[0] else pos`, the model is *designed* to never exceed the table — it chunks. XY-Tokenizer decode caps at `chunk_code_length = chunk_len_s·enc_sr/enc_downsample = 375` codes and HF `decode(overlap_seconds=10)` walks overlapping windows (stride `duration_code_length = 250`, keep leading `250·1920` samples per window, final trim to `total·1920`). Mirror that CPU-side loop — each window is one full decode graph — rather than growing the table (which would desync from the reference numerics). Parity check: the chunk seam has *zero* diff when the stitch is correct.

### Parity testing strategy

When matching PyTorch numerically:

- Stage parity: validate each subgraph against PyTorch *before* chaining them. Bit-perfect deterministic stages (encoder, single CFM step) catch shape/permute bugs early; once chained, only RNG-dependent stages will diverge.
- For RNG-dependent paths (CFM noise init, NSF random phase + Gaussian noise), expose a path that takes a precomputed noise tensor instead of sampling — then PyTorch-reference parity becomes deterministic.
- Once the full pipeline matches end-to-end, **delete the standalone parity test binaries and tests**. This project standardises on E2E smoke tests (`tests/e2e/<model>_decode_smoke.py`) that drive the public C API. Per-stage parity scripts are scaffolding, not artefacts.

### Build flow

`cmake --build build -j` is the source of truth. Run it after every non-trivial edit; the linter diagnostics surfaced inline are unreliable. After build, run `tests/e2e/<model>_*_smoke.py` from the repo root via `.venv/bin/python` (PyTorch deps live in that venv). Smokes reach libcodec through `codec_cpp` (it falls back to `python/` + `build/libcodec.so` when not installed); never add per-test ctypes struct/prototype copies — extend `python/codec_cpp` instead, and put generic test helpers in `tests/e2e/_common.py`.

---

## Useful entry points for Codex

If you need to understand execution:
- `src/runtime/graph_exec.cpp` (scheduler + compute)
- `src/runtime/graph.cpp` (cache + arena)

If you need to understand a model forward:
- `src/models/mimi.cpp` / `dac.cpp` / `wavtokenizer.cpp`

If you need to add/replace an op:
- `src/ops/ggml_ops.cpp` (+ possibly ggml upstream)
