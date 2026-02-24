cmake_minimum_required(VERSION 3.16)
project(codec C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GGML Backend Options (pass-through to ggml subproject)
# Goal: allow building with any ggml-native accelerator backend on other machines.
# These options mirror ggml/llama.cpp-style flags.
option(GGML_CUDA        "Enable CUDA backend"       OFF)
option(GGML_HIP         "Enable HIP/ROCm backend"   OFF)
option(GGML_VULKAN      "Enable Vulkan backend"     OFF)
option(GGML_METAL       "Enable Metal backend"      OFF)
option(GGML_SYCL        "Enable SYCL backend"       OFF)
option(GGML_OPENCL      "Enable OpenCL backend"     OFF)
option(GGML_CANN        "Enable CANN backend"       OFF)
option(GGML_MUSA        "Enable MUSA backend"       OFF)
option(GGML_WEBGPU      "Enable WebGPU backend"     OFF)
option(GGML_ZDNN        "Enable zDNN backend"       OFF)
option(GGML_VIRTGPU     "Enable VirtGPU backend"    OFF)
option(GGML_RPC         "Enable RPC backend"        OFF)
option(GGML_BLAS        "Enable BLAS backend"       OFF)

# CPU SIMD optimizations (default: auto-detect enabled)
option(GGML_AVX         "Enable AVX"                ON)
option(GGML_AVX2        "Enable AVX2"               ON)
option(GGML_AVX512      "Enable AVX512"             ON)
option(GGML_FMA         "Enable FMA"                ON)
option(GGML_NEON        "Enable ARM NEON"           ON)
option(GGML_SVE         "Enable ARM SVE"            ON)

set(GGML_CPU ON)

add_subdirectory(ggml)

add_library(codec STATIC
    src/codec.cpp
    src/runtime/gguf_kv.cpp
    src/runtime/tensor_utils.cpp
    src/runtime/graph.cpp
    src/runtime/graph_exec.cpp
    src/ops/safe_math.cpp
    src/ops/ggml_ops.cpp
    src/ops/rope.cpp
    src/ops/rvq.cpp
    src/ops/conv1d.cpp
    src/ops/convtr1d.cpp
    src/batch/batch.cpp
    src/models/wavtokenizer.cpp
    src/models/dac.cpp
    src/models/mimi.cpp
    src/models/qwen3_tts_tokenizer.cpp
)

target_include_directories(codec
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

target_link_libraries(codec
    PRIVATE
        ggml
)

add_executable(inspect-codec
    examples/inspect-codec.cpp
)

target_link_libraries(inspect-codec PRIVATE codec)

add_executable(codec-cli
    examples/codec-cli.cpp
)

target_link_libraries(codec-cli PRIVATE codec)

add_executable(codec-batch-decode
    examples/codec-batch-decode.cpp
)

target_include_directories(codec-batch-decode PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(codec-batch-decode PRIVATE codec ggml ggml-cpu)
