---
name: codec-dev
description: Guidelines for developing the codec.cpp project, ensuring parity with llama.cpp architecture and conventions.
---

# codec.cpp Development Guidelines

Implement the Audio Encoder/Decoder framework following 'llama.cpp' backbone patterns.

## Core Rules:
1. **Architecture**: Use 'ggml' as the backend. Maintain 'codec_model' (shared weight data) and 'codec_context' (stateful inference) separation.
2. **GPU Support**: Use a simple 'bool use_gpu' flag in params instead of layer-based offloading, optimized for smaller audio models.
3. **Format**: Use GGUF exclusively for model files. Do not create new file structures.
4. **Style**: Pure C header (extern "C" for C++) with 4-space indentation, mirroring llama.cpp's coding style.
5. **Implementation**:
   - Support 'WavTokenizer-Large', 'DAC', 'Mimi', and 'Qwen3-TTS-Tokenizer'.
   - Models are registered via a vtable in `src/codec.cpp`; model structs live in `src/models/<model>.h`.
   - If an operator is missing in GGML, implement it as a custom operator using existing GGML primitives where possible.
6. **Submodule**: `ggml/` is a submodule; do not edit it directly unless explicitly asked to update the submodule.

## Reference:
- Always refer to the latest 'llama.h' and 'ggml.h' patterns from the llama.cpp repository.
