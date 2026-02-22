#!/usr/bin/env bash
set -euo pipefail

backends=(
    "CPU-only::"
    "CUDA:-DGGML_CUDA=ON"
    "HIP:-DGGML_HIP=ON"
    "Vulkan:-DGGML_VULKAN=ON"
    "Metal:-DGGML_METAL=ON"
    "SYCL:-DGGML_SYCL=ON"
    "OpenCL:-DGGML_OPENCL=ON"
    "CANN:-DGGML_CANN=ON"
    "MUSA:-DGGML_MUSA=ON"
    "WebGPU:-DGGML_WEBGPU=ON"
    "zDNN:-DGGML_ZDNN=ON"
    "VirtGPU:-DGGML_VIRTGPU=ON"
)

for entry in "${backends[@]}"; do
    name="${entry%%:*}"
    flags="${entry#*:}"
    build_dir="build_${name}"

    echo "Building: $name ($flags)"
    cmake -B "$build_dir" $flags
    cmake --build "$build_dir" -j || echo "SKIP: $name (missing deps)"
done
