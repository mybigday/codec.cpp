# codec-cpp (Python bindings)

`codec_cpp` is the Python API for codec.cpp. It wraps the public C API
(`include/codec.h`, `codec_speaker.h`, `codec_lm.h`, `codec_audio_lm.h`, and `codec_tts.h` from
libcodec_tts) with ctypes. It has no compiled extension, and its only
dependency is numpy.

## Install

The package is not on PyPI yet. There are two ways to get it:

**Wheel with libcodec bundled.** The root `pyproject.toml` builds libcodec with
CMake (scikit-build-core) and puts it inside the package:

```bash
pip install .                                      # CPU
pip install . -C cmake.define.GGML_VULKAN=ON       # any ggml backend flag works
uv build --wheel                                   # wheel file for another machine
```

Wheel builds link ggml statically. `codec_cpp/lib/` holds `libcodec.so`,
`libcodec_tts.so` (`codec_cpp.TTS`) and `libttsbackbone.so`, the bundled
llama.cpp with codec.cpp's architecture patches, which the build applies to
the llama.cpp submodule (clone with `--recursive`, or pip a git URL, which
fetches submodules). ggml is built with `GGML_NATIVE` on, so the wheel is
tuned to the machine that builds it.

**From a source checkout, with no install.** Build the library the usual way
(`cmake --build build -j`) and put `python/` on `sys.path`. The loader finds
`build/libcodec.so` on its own. This is how `tests/e2e` runs.

The loader looks for the library in this order:

1. `$CODEC_CPP_LIBRARY`, the full path to a libcodec you built yourself
2. `codec_cpp/lib/` (wheel installs)
3. `<repo>/build/` (source checkouts)

The library is loaded the first time it is used, not at import. If the same
process also uses llama-cpp-python, `import llama_cpp` before the first
codec_cpp call. llama-cpp-python ships its own libggml, and the first one loaded
wins.

## Text to speech

`codec_cpp.TTS` runs the whole pipeline, the same host loop as `tts-cli`: the
backbone GGUF in the llama.cpp bundled with codec.cpp (with the architecture
patches the models need), the codec GGUF in codec.cpp. It loads
`libcodec_tts`, which wheels bundle and CMake builds put next to libcodec
(`CODEC_TTS_BACKBONE=ON`, the default).

```python
import codec_cpp

tts = codec_cpp.TTS("backbone.gguf", "codec.gguf")    # use_gpu=, n_threads=, max_frames=, kv_overrides=
audio, sr = tts.synthesize("Hello there.")
audio, sr = tts.synthesize("Hello there.", ref_audio="me.wav", ref_text="What the clip says.")
audio, sr = tts.synthesize("Hello there.", voice=tts.voices()[0], seed=7, temperature=0.8)
for piece, sr in tts.stream(long_text, chunk_ms=500):   # audio as it becomes final
    play(piece, sr)
```

Options left unset keep the model's defaults (from its GGUF): `instruct`,
`preset`, `grammar`, `seed`, `temperature`, `top_p`, `top_k`, `min_p`,
`repetition_penalty`, `min_frames`, `lm_guidance`, `flow_guidance`,
`flow_steps`, `segment_bytes`, `speaker_file`, `save_speaker`.

## Your own llama.cpp loop: AudioLM

`codec_cpp.AudioLM` mirrors `include/codec_audio_lm.h`
([docs/codec_common_api.md](../docs/codec_common_api.md)): codec.cpp says what
to feed the backbone, how to sample and when to stop; your code makes every
llama.cpp call. With llama-cpp-python (stock architectures only):

```python
import llama_cpp                                     # before codec_cpp (own libggml)
import codec_cpp

alm = codec_cpp.AudioLM("codec.gguf")                # backbone_path= for models that read its embeddings
vocab = codec_cpp.Vocab.from_llama_cpp(llama_model)
req = alm.backbone_requirements(vocab, max_frames=500)   # n_ctx, n_ubatch, n_seq_max, embeddings, flash_attn
voice = alm.create_voice(pcm, sr, ref_text)         # or alm.voice("name") / alm.load_voice(path)
alm.tts_prompt("Hello there.", vocab=vocab, voice=voice)
c = alm.completion()      # c.chunks (TEXT / TOKENS / EMBD), c.sampler, c.allowed, c.grammar, flags
# decode the chunks, then per step:
v = alm.observe(tokens=[tok] if sampled else None, hiddens=last_hidden)
# STOP -> done; FEED_TOKEN -> decode v.token (or tok); FEED_EMBD -> decode alm.next_embd(0)
audio, sr = alm.decode()  # or alm.stream_poll() after audio steps + alm.stream_finish()
```

`tests/e2e/audio_lm_host_smoke.py` is a complete loop. `alm.split` /
`alm.join` give the model's segmentation of long text.

## Codecs

```python
import codec_cpp

model = codec_cpp.Model("models/mimi/mimi.gguf")    # use_gpu=, n_threads=, kv_overrides={...}
ctx = model.context()                                # seed=

codes = ctx.encode(pcm, 24000)                       # float32 (n,) -> int32 (n_frames, n_q)
audio, sr = ctx.decode(codes)                        # -> float32 (n,), sample rate

latent, _ = ctx.encode_latent(pcm, sr)               # continuous-latent codecs: (latent_dim, n_frames)
audio, sr = ctx.decode_latent(latent)

outs = ctx.decode_batch([codes_a, codes_b])          # [(pcm, sr), ...]
```

### GGUF metadata overrides

These work like llama.cpp's `--override-kv`:

```python
codec_cpp.Model(path, kv_overrides={"chatterbox_s3g.flow.n_timesteps": 10})
```

### Speakers

```python
spk = model.speaker()                     # builtin default; model.speaker_names lists the rest
spk = ctx.speaker_from_audio(ref_pcm, sr) # when model.can_speaker_from_audio
model.save_speaker(spk, "voice.gguf", name="alice")
spk = model.load_speaker("voice.gguf")
audio, sr = ctx.decode(codes, speaker=spk)
spk.slot("embedding")                     # raw conditioning tensor, numpy copy
```

### codec_lm (the adaptor between an LLM backbone and the codec)

```python
lm = model.get_lm_adaptor()                # one per Model, reused; freed with the Model
lm.info                                    # LMInfo: kind, hidden_dim, n_codebook, ...
st = lm.state()
codes = st.step_greedy(hidden)             # depth decoder: one frame of codes from a backbone hidden
embd = lm.compose_audio_embd(codes)        # next backbone input row
```

The other step calls (`st.step_generate` for continuous latents,
`st.text_prefill`, ...) mirror their C functions one to one. The backbone
itself runs in llama.cpp and is not part of this package.

The high-level layer does not wrap every call. For the rest, use
`codec_cpp.lib()` (the loaded `ctypes.CDLL`, with every prototype already
declared) and `codec_cpp.ffi` (structs and enums).

Errors raise `codec_cpp.CodecError`. Its `.status` is the `codec_status` value,
and the message includes the library's last-error string.

## Keeping the bindings in sync

`codec_cpp/_ffi.py` mirrors the headers by hand. When a public struct or
function changes, update `_ffi.py` in the same commit. The headers only ever
append fields at the end, and `_ffi.py` must keep the same order.

## Distribution notes

To publish wheels later, solve these first:

- **Portable CPU wheels.** They need `GGML_NATIVE=OFF`, plus either a
  conservative ISA baseline or `GGML_BACKEND_DL=ON` with
  `GGML_CPU_ALL_VARIANTS=ON`. The second option ships one `libggml-cpu-*.so`
  per ISA, and the install rule would then have to copy those libraries too.
- **GPU wheels.** Build one wheel per backend (CUDA, Vulkan, Metal) in CI,
  following llama-cpp-python's index-per-backend layout.
- **Bundled shared libraries.** Linux wheels need `auditwheel repair` and macOS
  wheels need `delocate`, to vendor libgomp and similar runtime dependencies.
