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
codec-cli decode --model mimi.gguf --codes input.npy --out output.wav

# With GPU acceleration (if built with CUDA/Vulkan/Metal)
codec-cli decode --model mimi.gguf --codes input.npy --out output.wav --use-gpu
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
tts-cli synthesize --backbone MODEL.gguf --model CODEC.gguf \
    --text "Hello there." --output out.wav
```

| Option | What it does |
|---|---|
| `--voice NAME` | use a builtin voice; list them with `tts-cli voices --model CODEC.gguf` |
| `--ref-audio FILE.wav` | clone the voice in a short clip (models that support it) |
| `--ref-text TEXT` | transcript of `--ref-audio`, for models that use one |
| `--save-speaker FILE.gguf` / `--speaker-file FILE.gguf` | save a voice once, reuse it later |
| `--seed N` | repeatable output |
| `--temp F`, `--top-k N`, `--top-p F`, `--min-p F`, `--repeat-penalty F`, ... | llama.cpp's sampling flags; unset ones come from the backbone file (`general.sampling.*`, the model card's), then llama.cpp's defaults |
| `--max-frames N` | cap the output length |
| `--gpu` | run on the GPU backend the build has (CUDA / Vulkan / Metal) |
| `--n-threads N` | CPU threads |
| `--override-kv KEY=TYPE:VALUE` | change a setting stored in the codec file (keys are listed in [docs/models](docs/models)) |

Run `tts-cli` without arguments for the full list. Model-specific text
input rules (language tags, phonemes) are on each model card.

### OpenAI-compatible server

`tts-server` loads one model pair and serves OpenAI's
[`POST /v1/audio/speech`](https://platform.openai.com/docs/api-reference/audio/createSpeech),
so the `openai` SDKs and other compatible clients can use it:

```bash
tts-server --backbone MODEL.gguf --model CODEC.gguf --port 8080   # also --gpu, --n-threads, --api-key KEY

curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"model": "tts", "input": "Hello there.", "voice": "NAME", "response_format": "wav"}' -o out.wav
```

How it differs from OpenAI's API:

- `response_format` is required. `wav` and `pcm` are always available.
  `mp3`, `opus`, `aac` and `flac` each need their encoder at build time:
  `-DCODEC_TTS_SERVER_MP3=ON` (libmp3lame), `_OPUS` (libopus + libogg),
  `_AAC` (libfdk-aac) or `_FLAC` (libFLAC). Asking for a format the build
  lacks returns a 400 error.
- `voice` takes the model's own voice names (`GET /v1/audio/voices` lists
  them). Leave it out to get the model's default voice; `--override-kv
  <arch>.spk.default=str:NAME` changes the default. Speakers saved with
  `tts-cli --save-speaker` can be served with `--speaker-file FILE.gguf`.
- Custom voices, as in OpenAI's
  [custom voices](https://developers.openai.com/api/docs/guides/custom-voices),
  for models that clone (`--ref-audio` in tts-cli). With `--voice-dir DIR`
  the server serves the voices stored there; adding `--allow-voice-create`
  also enables `POST /v1/audio/voices` (multipart `name` + `audio_sample`, a
  WAV of at most 30 s, plus an optional `ref_text` transcript, which models
  such as Qwen3-TTS and CosyVoice use). It answers
  `{"object": "audio.voice", "id": "voice_...", ...}`, stores the voice as
  `DIR/<id>.gguf`, and requests use it with `"voice": {"id": "voice_..."}`.
  `GET /v1/audio/voices` lists them under `custom`. `consent` is not checked.
- `instructions` works only on models with an instruct mode (CosyVoice), and
  is passed through as the model expects it (CosyVoice-300M-Instruct: end it
  with `<|endofprompt|>`). `speed` must be 1.0.
- Sampling (an extension): llama-server's request fields (`temperature`,
  `top_k`, `top_p`, `min_p`, `repeat_last_n`, `repeat_penalty`,
  `presence_penalty`, `frequency_penalty`, `dry_*`, `samplers`, `seed`) override
  the server's `--temp` / `--top-k` / ... flags, which override the backbone
  file's `general.sampling.*` (the model card's), then llama.cpp's defaults.
- Every response is 24 kHz mono, matching OpenAI's formats (`pcm` is raw
  16-bit little-endian). Requests run one at a time.

Responses stream: audio is sent as it is generated, with chunked transfer
encoding, or as `speech.audio.delta` / `speech.audio.done` events with
`"stream_format": "sse"`. Two settings shape it:

- `--segment-bytes N` (default 160, about 10 seconds of speech): long input
  is synthesized a piece at a time, since TTS models get worse on long text.
  Pieces end at sentence ends and always at blank lines; a sentence longer
  than N is split at commas, then spaces, and kept whole if it has neither.
  `0` sends the whole input to the model at once. (`tts-cli synthesize`
  takes the same option, off by default.) Tags written into the text, such
  as MOSS-TTSD's `[S1]` or CosyVoice's language tags, stay with the piece
  they are in, so repeat them per sentence or use `0`.
- `--stream-chunk-ms N` (default 1000): on models whose decoder can run on
  partial output (CSM, LFM2-Audio, MOSS-TTSD, MOSS-TTS-Realtime, NeuTTS,
  OuteTTS), audio is sent in pieces of about this length while a segment is
  still being generated. Other models (CosyVoice, Chatterbox, KittenTTS,
  Soprano, Pocket-TTS) send each segment when it is done. `0` sends whole
  segments for every model. With NeuTTS, MOSS-TTSD and OuteTTS 0.2 / 0.3,
  whose decoders look at the whole segment, the streamed audio differs
  slightly from the non-streamed one.

A request that fails before any audio is ready gets an error status as
usual. One that fails after audio has started ends early: with an `error`
event under SSE, or as a cut-off body otherwise. Keep `--segment-bytes` on
when streaming: with `0`, each chunk re-decodes everything generated so far.

## Build with GPU Acceleration

### CUDA (NVIDIA)
```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j
codec-cli --model model.gguf --codes in.npy --out out.wav --use-gpu
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
codec-cli --model model.gguf --codes in.npy --out out.wav
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
