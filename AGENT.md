# codec.cpp

This repository is a C/C++ library + CLI that runs several neural audio codecs (currently WavTokenizer-Large, DAC, Mimi, Qwen3-TTS-Tokenizer) using **ggml** graphs so execution can be offloaded via **ggml backends** (CPU/CUDA/Vulkan/Metal/etc.).

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
  - `qwen3_tts_tokenizer.cpp/.h`
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

## ggml constraints (quick list)

- `ggml_conv_transpose_1d` requires `p0 == 0` and `d0 == 1`; use crop for padding.
- `conv1d` weight layout is `[k, in, out]`.
- `conv_transpose_1d` weight layout is `[k, out, in]`.
- Prefer `ggml_cont` only when needed; many ops require contiguous tensors.
- Keep all math inside ggml; do not add CPU-only tensor math paths.

---

## Model files / GGUF

Models are loaded from `.gguf`.

Some tensors that could be generated at runtime should instead be baked into GGUF during conversion (for reproducibility + avoiding runtime FP32→FP16 conversions).

If conversion scripts are involved, regenerate gguf after changes (stale gguf is a common source of “missing tensor” errors).

---

## Model source code (local)

If a model’s original PyTorch code is needed during conversion or reference:
- Put the upstream source under `.model-src/` (local-only).
- Converters should read from `.model-src/<repo>/...` rather than importing from the network.
- Runtime must not depend on the original Python source.

---

## ggml submodule constraints

`ggml/` is a submodule. Do **not** edit it directly unless explicitly asked to update the submodule.

If an op is missing:
- Prefer composition in `src/ops/`.
- If a true kernel is required, plan a **submodule update** (upstream or fork) rather than editing files in-place.
- Avoid CPU-only fallbacks; keep the path compatible with ggml backends.

---

## Conventions / guardrails for changes

- Keep encode/decode numerics stable (unit/regression tests where possible).
- Avoid introducing new CPU-only intermediate buffers; build everything as ggml tensors.
- **Never reshape/transpose weights at runtime.** If weights need reshape, do it in the GGUF converter or via gguf transpose ops during conversion.
- When touching graph execution / backend scheduler: be careful with allocation lifetimes (`eval_ctx`, scheduler reset semantics).
- Prefer small, reviewable commits.

---

## Python dependencies

We track Python deps in two files:
- `requirements.txt` for conversion/build utilities.
- `requirements-e2e.txt` for end-to-end tests (HF refs + audio).

Keep them minimal and deterministic (pin versions when CI is sensitive).

---

## Model/Op Implementation Playbook (condensed)

### New model (encode/decode)
1. **GGUF converter first**
   - Add converter in `scripts/converters/`.
   - Bake all needed weights/metadata into GGUF. No runtime weight transforms.
   - Confirm tensor layout: ggml expects `[k, in, out]` for conv1d and `[k, out, in]` for conv_transpose_1d (see `ggml_conv_transpose_1d` constraints).
2. **Runtime model struct**
   - Add metadata fields in `codec_*` struct and initialize in `codec_*_init`.
   - Read GGUF keys with sane defaults but validate shapes early.
3. **Graph build**
   - Build encode/decode forward graphs using ggml ops only.
   - Cache graph with a compact key (kind, n_frames, n_q, hop, n_in, latent_dim).
   - If graph is large, ensure graph size + backend scheduler capacity are adequate (see `src/runtime/graph.cpp` and `src/runtime/graph_exec.cpp`).
4. **Weights and IO**
   - Use `codec_*_copy_*` helpers to map GGUF tensors into graph tensors.
   - Avoid any CPU-only math or bespoke tensor loops unless absolutely necessary.
5. **E2E tests**
   - Add/update model entry in `tests/e2e/config.json` (sample rate, n_q, gguf path).
   - Ensure HF reference runs with the same sample rate/hop.
   - Run `python tests/e2e/runner.py --models <name>`.

### New op (ggml)
1. **Prefer ggml primitives**
   - Implement as a composition in `src/ops/ggml_ops.cpp` when possible.
2. **If a custom op is needed**
   - Implement in ggml backend (CPU + optional GPU stubs).
   - Keep API minimal and add a small targeted test.
3. **Respect ggml constraints**
   - Many ops impose shape/stride constraints; confirm in `ggml/` sources.
   - Example: `ggml_conv_transpose_1d` enforces `p0==0` and `d0==1`; use crop for padding.

---

## Common pitfalls

- **Graph size assertions**: if you hit `GGML_ASSERT(cgraph->n_nodes < cgraph->size)` or scheduler hash-set asserts, increase graph/scheduler capacity.
- **Sample rate mismatches**: ensure model `sample_rate` in GGUF and E2E config match HF reference.
- **Silent tensor layout mistakes**: verify tensor shapes against ggml expectations and PyTorch definitions.
- **Runtime weight fixes**: do not reshape/transpose weights at runtime; fix converter instead.

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

## Local skills

- `codec-model-dev` — end-to-end guide for adding a model (converter → ggml graphs → tests).
- `codec-op-dev` — guidance for adding/adjusting ggml ops safely.

Use the skill files for step-by-step workflows; they encode the preferred design constraints for this repo.
## Model registry (vtable architecture)

Models are wired via a **switch-based vtable registry** in `src/codec.cpp`. `codec_model` stores:
- shared/core metadata fields (sample rate, hop, n_q, etc.)
- `impl` (opaque model-specific struct)
- `vtable` (init/encode/decode/decode_latent)

Model-specific structs live in `src/models/<model>.h` and are **not** defined in `src/codec_internal.h`. Core code must not cast `impl`; only model files cast their own `impl`.

### Adding a new model
1. Define the model struct in `src/models/<model>.h`.
2. Implement model graph + init in `src/models/<model>.cpp`.
3. Provide a `codec_<model>_vtable()` with `create_impl/destroy_impl/init/encode/decode`.
4. Register it in the switch in `codec_model_vtable_for_arch()` in `src/codec.cpp`.

## Reuse / composite models

If a model reuses another model’s encoder (e.g. Qwen3 reuses Mimi), put both configs in a model-specific `impl` struct and call the shared encoder helpers (`codec_mimi_encode_with(...)`).
