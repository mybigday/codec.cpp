---
name: codec-op-dev
description: Guide for adding or modifying ggml ops in codec.cpp with backend-safe patterns.
---

# codec-op-dev

Use this skill when adding a **new op** or modifying ggml op usage.

## Decision flow
1. **Can this be built from existing ggml ops?**
   - If yes: implement in `src/ops/ggml_ops.cpp` (or small helper file).
2. **If not possible or too slow**:
   - Implement a custom ggml op (CPU path first; keep a plan for GPU backends).

---

## Composition-first approach

Preferred location:
- `src/ops/ggml_ops.cpp`

Guidelines:
- Keep ops pure ggml tensor graph (no CPU-side math).
- Use `ggml_cont` only when needed (avoid extra copies).
- Respect ggml shape constraints.

---

## Custom op approach (only if needed)

Important: `ggml/` is a submodule in this repo and must not be edited directly for commits.

Preferred steps:
1. Try to implement the op as a composition in `src/ops/ggml_ops.cpp`.
2. If true custom op is required:
   - First, verify there is no composition possible with existing ggml ops.
   - If a new kernel is unavoidable, **plan a ggml submodule update** (upstream or fork) that adds the op in `ggml/`.
   - Do not add CPU-only fallbacks; keep ops backend-compatible.
3. If someone asks for a CPU kernel in `ggml/src/ggml-cpu/`, explain that it requires a submodule bump (no direct edits).
4. Add a focused test in `tests/` or a local harness within this repo.

Note: ops changes should not touch the model registry (vtable) in `src/codec.cpp` unless a new model is being added.

---

## Constraints to check

Common ggml constraints you must verify in `ggml/`:
- `ggml_conv_transpose_1d`: **p0 == 0**, **d0 == 1** (use crop for padding).
- Layout expectations for conv/linear.
- Backend limitations on op combos (e.g., certain ops only on CPU).

---

## Testing + diagnostics

Recommended:
- Create a minimal graph test (input → op → output).
- Compare against a PyTorch reference for numeric parity.
- Validate shapes and memory use.

Failure patterns:
- Graph size assertion: increase graph size or scheduler capacity.
- Wrong layout: fix in converter or add GGUF transpose op.
