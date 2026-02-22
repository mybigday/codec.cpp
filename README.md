## Quick Start

### 1. Convert Models to GGUF

```bash
cd scripts

# From HuggingFace
python convert-to-gguf.py --model-id kyutai/mimi --output mimi.gguf

# From local checkpoint
python convert-to-gguf.py --input-dir ./mimi-checkpoint --output mimi.gguf

# With quantization (Q4_K_M, Q5_K_M, Q8_0)
python convert-to-gguf.py --model-id kyutai/mimi --output mimi-q4.gguf --quantization Q4_K_M
```

### 2. Decode Audio

```bash
./build/codec-cli decode --model mimi.gguf --codes input.npy --out output.wav

# With GPU acceleration (if built with CUDA/Vulkan/Metal)
./build/codec-cli decode --model mimi.gguf --codes input.npy --out output.wav --use-gpu
```

## Build with GPU Acceleration

### CUDA (NVIDIA)
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
./build/codec-cli --model model.gguf --codes in.npy --out out.wav --use-gpu
```

### Vulkan (Cross-platform)
```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j
```

### Metal (macOS)
```bash
cmake -B build -DGGML_METAL=ON
cmake --build build -j
```

### SYCL (Intel GPUs)
```bash
cmake -B build -DGGML_SYCL=ON
cmake --build build -j
```

### OpenCL
```bash
cmake -B build -DGGML_OPENCL=ON
cmake --build build -j
```

### CANN (Ascend)
```bash
cmake -B build -DGGML_CANN=ON
cmake --build build -j
```

### HIP/ROCm (AMD GPUs)
```bash
cmake -B build -DGGML_HIP=ON
cmake --build build -j
```

### MUSA
```bash
cmake -B build -DGGML_MUSA=ON
cmake --build build -j
```

### WebGPU
```bash
cmake -B build -DGGML_WEBGPU=ON
cmake --build build -j
```

### zDNN
```bash
cmake -B build -DGGML_ZDNN=ON
cmake --build build -j
```

### VirtGPU
```bash
cmake -B build -DGGML_VIRTGPU=ON
cmake --build build -j
```

### Multiple backends (fallback chain)
```bash
cmake -B build -DGGML_CUDA=ON -DGGML_VULKAN=ON
cmake --build build -j
# Runtime auto-selects: CUDA > Vulkan > CPU
```

### CPU-only (default)
```bash
cmake -B build
cmake --build build -j
./build/codec-cli --model model.gguf --codes in.npy --out out.wav
```
