---
name: codec-model-dev
description: Guide for adding a new codec model (GGUF conversion + ggml graph integration) in codec.cpp.
---

# codec-model-dev

Use this skill when adding a **new model architecture** or a large variant that needs GGUF conversion + ggml graphs.

## Checklist (short)
1. Converter + GGUF metadata
2. Runtime model struct + init
3. Graph build (encode/decode)
4. Weight copy + IO wiring
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

## 2) Runtime model struct + init

Files:
- `src/codec_internal.h` (model struct)
- `src/models/<model>.cpp` (init)

Actions:
- Add model struct fields for metadata + any fixed architectural constants.
- In `codec_<model>_init`, load GGUF keys with safe defaults and sanity checks.
- Propagate model-level fields to `codec_model` (sample_rate, hop_size, n_q, etc.).

---

## 3) Graph build (encode/decode)

Files:
- `src/models/<model>.cpp`
- `src/runtime/graph.cpp` (graph sizing if needed)
- `src/runtime/graph_exec.cpp` (scheduler sizing if needed)

Rules:
- Build graphs using **ggml ops only**.
- Avoid CPU-side tensor math.
- Add graph cache keys: `(kind, n_frames, n_q, hop, n_in, latent_dim)`.
- If graph is large, scale `ggml_new_graph_custom` size and ensure scheduler capacity.

---

## 4) Weight copy + IO

Patterns:
- Use `codec_*_copy_*` helpers to map GGUF tensors into graph tensors.
- Keep naming consistent: `codec_<model>_*_tensor_name`.
- Validate shapes early and return actionable errors.

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
