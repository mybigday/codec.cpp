# codec.cpp

This repository is a C/C++ library + CLI that runs several neural audio codecs (currently WavTokenizer-Large, DAC, Mimi) using **ggml** graphs so execution can be offloaded via **ggml backends** (CPU/CUDA/Vulkan/Metal/etc.).

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
