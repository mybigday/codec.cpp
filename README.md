codec.cpp
===

Neural audio codec inference in C/C++

The main goal of codec.cpp is to enable neural audio codec inference with minimal setup and state-of-the-art performance on a wide range of hardware — locally and in the cloud. Supports Mimi, DAC, WavTokenizer with quantization and multi-backend GPU acceleration.


## Quick Start

### 1. Convert Models to GGUF

```bash
cd scripts

# From HuggingFace
python convert-to-gguf.py --model-id kyutai/mimi --output mimi.gguf

# From local checkpoint
python convert-to-gguf.py --input-dir ./mimi-checkpoint --output mimi.gguf

# Float precision (F32 / F16 / BF16 / AUTO = the checkpoint's; default F16, never wider than the checkpoint)
python convert-to-gguf.py --model-id kyutai/mimi --output mimi-bf16.gguf --outtype BF16

# With quantization (Q4_K_M, Q5_K_M, Q8_0)
python convert-to-gguf.py --model-id kyutai/mimi --output mimi-q4.gguf --quantization Q4_K_M

# A bundled LM-side adaptor (lm.* tensors) has its own settings
python convert-to-gguf.py --model-id Qwen/Qwen3-TTS-Tokenizer-12Hz --lm-source Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output qwen3-tts-codec.gguf --lm-outtype F16 --lm-quantization Q8_0

# Requantize any GGUF per tensor
python scripts/quantize-gguf.py in.gguf out.gguf --type Q8_0 --tensor-type 'codebook=F32'
```

codec.cpp computes every tensor at the type stored in the GGUF, so precision is
chosen here: converters store F32 where a computation needs it.

### 2. Decode Audio

```bash
./build/codec-cli decode --model mimi.gguf --codes input.npy --out output.wav

# With GPU acceleration (if built with CUDA/Vulkan/Metal)
./build/codec-cli decode --model mimi.gguf --codes input.npy --out output.wav --use-gpu
```

### 3. Python

```python
import codec_cpp
ctx = codec_cpp.Model("mimi.gguf").context()
audio, sr = ctx.decode(ctx.encode(pcm, 24000))
```

`pip install .` builds libcodec and installs the `codec_cpp` package; see [python/README.md](python/README.md).

## Text-to-speech

`tts-cli` runs TTS models end to end: a text-to-speech-token model (run by the
bundled llama.cpp) plus a codec GGUF (run by codec.cpp). Ready-made GGUFs are
on [Hugging Face](https://huggingface.co/BricksDisplay); each model card lists
which two files to download. Supported models: [docs/model_status.md](docs/model_status.md#tts-tts-cli-synthesize).

Build (clone with `--recursive`: it fetches the llama.cpp that runs the
text-side model; codec.cpp adds the architectures upstream llama.cpp lacks):

```bash
git clone --recursive https://github.com/mybigday/codec.cpp
cd codec.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # GPU: add -DGGML_CUDA=ON / -DGGML_VULKAN=ON / -DGGML_METAL=ON
cmake --build build -j
```

Synthesize:

```bash
./build/tts-cli synthesize --backbone MODEL.gguf --model CODEC.gguf \
    --text "Hello there." --output out.wav
```

| Option | What it does |
|---|---|
| `--voice NAME` | use a builtin voice; list them with `tts-cli voices --model CODEC.gguf` |
| `--ref-audio FILE.wav` | clone the voice in a short clip (models that support it) |
| `--ref-text TEXT` | transcript of `--ref-audio`, for models that use one |
| `--save-speaker FILE.gguf` / `--speaker-file FILE.gguf` | save a voice once, reuse it later |
| `--seed N` | repeatable output |
| `--temp F`, `--top-k N`, `--top-p F` | sampling (defaults come from the model file) |
| `--max-frames N` | cap the output length |
| `--gpu` | run on the GPU backend the build has (CUDA / Vulkan / Metal) |
| `--n-threads N` | CPU threads |
| `--override-kv KEY=TYPE:VALUE` | change a setting stored in the codec file (keys are listed in [docs/models](docs/models)) |

Run `./build/tts-cli` without arguments for the full list. Model-specific text
input rules (language tags, phonemes) are on each model card.

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

## License

MIT

Third-party code bundled in this repository:

- [pocketfft](https://github.com/mreineck/pocketfft) (header-only FFT, used by the host ISTFT) — BSD-3-Clause, see `src/third_party/pocketfft/LICENSE.md`.

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
