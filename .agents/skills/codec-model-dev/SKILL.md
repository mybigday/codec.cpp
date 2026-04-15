---
name: codec-model-dev
description: Guide for adding a new codec model (GGUF conversion + ggml graph integration) in codec.cpp.
---

# codec-model-dev

Use this skill when adding a **new model architecture** or a large variant that needs GGUF conversion + ggml graphs.

## Checklist (short)
1. Converter + GGUF metadata
2. Runtime model struct + vtable init
3. Graph build (encode/decode) + exact graph sizing
4. Exact weight upload + IO wiring
5. E2E test + HF parity

---

## 1) Converter + GGUF

**Goal:** all weight transforms happen *during conversion*, never at runtime.

- Add a converter in `scripts/converters/`.
- If you need upstream source, place it under `.model-src/` and import locally.
- Add GGUF keys for:
  - `codec.sample_rate`, `codec.hop_size`, `codec.n_q`, `codec.codebook_size`, `codec.latent_dim`, `codec.codebook_dim`
  - `codec.has_encoder`, `codec.has_decoder`
- Ensure tensor layout is compatible with ggml:
  - `conv1d` weights: ggml expects `[k, in, out]`.
  - `conv_transpose_1d` weights: ggml expects `[k, out, in]` and **p0==0, d0==1**; use crop for padding.
- If weight needs transpose/reshape: do it in converter or use a GGUF transpose op during conversion.
- Never reshape/transpose weights at runtime.

**Validate** with a small inspection script (shape + basic stats).

---

## 2) Runtime model struct + vtable init

Files:
- `src/models/<model>.h` (model struct + vtable declaration)
- `src/models/<model>.cpp` (init + vtable)
- `src/codec.cpp` (register in switch-based vtable registry)

Actions:
- Add model struct fields for metadata + any fixed architectural constants in `src/models/<model>.h`.
- Implement `codec_<model>_vtable()` (create_impl/destroy_impl/init/encode/decode).
- In `codec_<model>_init`, load GGUF keys with safe defaults and sanity checks.
- Propagate model-level fields to `codec_model` (sample_rate, hop_size, n_q, etc.).
- Set the model vtable `graph_size` callback. Default policy should be the shared exact DAG-based sizing helper unless the model truly needs something else.
- Register the vtable in `codec_model_vtable_for_arch()` in `src/codec.cpp`.

---

## 3) Graph build (encode/decode) + exact graph sizing

Files:
- `src/models/<model>.cpp`
- `src/runtime/graph.cpp`
- `src/runtime/graph_exec.cpp`

Rules:
- Build graphs using **ggml ops only**.
- Avoid CPU-side tensor math.
- Add graph cache keys: `(kind, n_frames, n_q, hop, n_in, latent_dim)`.
- Do not add `kind`-based graph-size heuristics.
- Eval graph capacity should be exact for the built DAG.
- Scheduler capacity must also be exact, but its requirement is not necessarily identical to eval-graph capacity. Follow ggml's scheduler contract instead of reusing old blanket multipliers.
- If you need a model-specific graph sizing policy, put it behind the vtable callback rather than in shared conditional logic.

---

## 4) Exact weight upload + IO

Patterns:
- Runtime helpers should only do exact validated upload of already-converted tensors.
- Shared runtime helpers should validate source existence, type, and exact shape compatibility, then upload.
- Keep naming consistent: `codec_<model>_*_tensor_name`.
- Validate shapes early and return actionable errors.
- Never transpose/reshape/reorder weights at runtime. If layout is wrong, fix the converter.

---

## 5) E2E test + HF parity

Files:
- `tests/e2e/config.json`
- `tests/e2e/runner.py`

Actions:
- Add model entry (sample_rate, n_q, gguf path, local_path, converter).
- Ensure input resampling matches model.
- Run:
  - `python tests/e2e/runner.py --models <name>`

If mismatch:
- Compare HF vs ggml outputs step-by-step.
- Check padding/stride and tensor layouts first.
- For graph allocation failures, check the vtable graph sizing policy and scheduler sizing contract before increasing anything.
