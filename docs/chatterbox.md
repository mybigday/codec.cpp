# Chatterbox Integration Notes

This document describes the parts of Chatterbox that `codec.cpp` and `llama.cpp`
need to understand in order to support the model family cleanly.

The intended split is:

- `codec.cpp`: `S3Tokenizer` (`S3T`) and `S3Gen` (`S3G`)
- `llama.cpp`: `T3` speech-token generator

The design goal is strict discrete-token interchange between the two libraries:

- `llama.cpp` emits discrete speech token IDs
- `codec.cpp` consumes discrete speech token IDs
- no continuous latents, speaker embeddings, prompt mels, or hidden states cross
  the boundary between the two runtimes


## 1. Chatterbox Components

Chatterbox is not one model. It is a stack of components:

1. Text tokenizer
2. `T3` token generator
3. `S3Tokenizer` speech tokenizer
4. `S3Gen` speech-token decoder
5. Voice encoder and reference-conditioning path

Relevant upstream files in `.model-src/chatterbox`:

- `src/chatterbox/models/s3tokenizer/s3tokenizer.py`
- `src/chatterbox/models/s3gen/s3gen.py`
- `src/chatterbox/models/t3/t3.py`
- `src/chatterbox/models/t3/modules/t3_config.py`
- `src/chatterbox/models/t3/modules/cond_enc.py`
- `src/chatterbox/tts.py`
- `src/chatterbox/tts_turbo.py`


## 2. Recommended Runtime Split

### `codec.cpp`

`codec.cpp` should implement the shared speech codec layer:

- `S3Tokenizer`: audio to discrete S3 token IDs
- `S3Gen`: discrete S3 token IDs to audio

This is the right layer for:

- audio tokenization for end-to-end audio systems
- speech detokenization after `T3`
- model conversion of the speech codec and decoder

### `llama.cpp`

`llama.cpp` should implement `T3`:

- text tokens plus voice conditioning to discrete S3 token IDs
- autoregressive generation of S3 token IDs
- BOS/EOS handling for the speech-token stream

`T3` is a separate task-specific model, even when its backbone is Llama-like or
GPT-like. It is not the same thing as a general main dialogue tower.

For a voice assistant, the practical architecture is usually:

1. main dialogue model
2. `T3`
3. `codec.cpp` `S3G`

For TTS only, the stack can be:

1. `T3`
2. `codec.cpp` `S3G`


## 3. S3 Token Contract

This section is the most important part of the interop design.

### Payload token IDs

Upstream `S3Tokenizer` defines:

- sample rate: 16 kHz
- mel hop: 160 samples
- token hop: 640 samples
- token rate: 25 tokens/sec
- payload vocabulary size: 6561

Source: `.model-src/chatterbox/src/chatterbox/models/s3tokenizer/s3tokenizer.py`

The shared payload token range is:

- valid payload IDs: `0..6560`

These payload IDs are the only token IDs that should cross the
`llama.cpp` <-> `codec.cpp` boundary.

### Special tokens

Original Chatterbox `T3Config` defines:

- `start_speech_token = 6561`
- `stop_speech_token = 6562`
- `speech_tokens_dict_size = 8194`

Turbo overrides this with a smaller speech vocabulary:

- `speech_tokens_dict_size = 6563`
- payload IDs still remain `< 6561`

Sources:

- `.model-src/chatterbox/src/chatterbox/models/t3/modules/t3_config.py`
- `.model-src/chatterbox/src/chatterbox/tts_turbo.py`

### Decode-side filtering

Upstream TTS code filters generated speech tokens before `S3Gen` decode:

- remove invalid or OOV IDs
- keep only IDs `< 6561`

Sources:

- `.model-src/chatterbox/src/chatterbox/tts.py`
- `.model-src/chatterbox/src/chatterbox/tts_turbo.py`
- `.model-src/chatterbox/src/chatterbox/models/s3gen/s3gen.py`

`codec.cpp` should match this behavior semantically:

- payload token stream passed to `S3G` must contain only `0..6560`
- BOS, EOS, and other OOV values must not be silently clamped into the codebook
- decode should either reject invalid IDs or drop them explicitly

### Token layout

Unlike multi-codebook RVQ codecs already in `codec.cpp`, S3 uses a single token
stream:

- `n_q = 1`
- `codebook_size = 6561`

This should be represented in `codec.cpp` as:

- `codec_token_buffer.n_q = 1`
- `codec_token_buffer.codebook_size = 6561`


## 4. Timebase and Sample Rate Mismatch

Chatterbox uses different sample rates on the encode and decode sides.

### Encode side

`S3Tokenizer` consumes:

- 16 kHz waveform
- token hop = 640 samples
- token rate = 25 Hz

### Decode side

`S3Gen` emits:

- 24 kHz waveform
- same token rate = 25 Hz
- effective output-side hop = 960 samples per token

This is a real API concern for `codec.cpp`.

Current `codec.cpp` model metadata and token buffer design mostly assume one
public sample rate and one public hop size:

- `codec_model.sample_rate`
- `codec_model.hop_size`
- `codec_token_buffer.sample_rate`
- `codec_token_buffer.hop_size`

Existing support for `encode_sample_rate` is not enough by itself because
decode currently validates token-buffer sample rate against the model decode
sample rate.

Relevant files:

- `include/codec.h`
- `src/codec.cpp`
- `src/models/neucodec.cpp`

### Required metadata direction

Chatterbox support should carry both sides explicitly in metadata.

Recommended GGUF metadata:

- `codec.encode_sample_rate = 16000`
- `codec.sample_rate = 24000`
- `codec.token_rate_hz = 25`
- `codec.encode_hop_size = 640`
- `codec.decode_hop_size = 960`
- `codec.n_q = 1`
- `codec.codebook_size = 6561`

If `codec.cpp` keeps only one public `hop_size`, then Chatterbox support will
need either:

- a small API extension for separate encode/decode hop sizes, or
- clear internal rules that token buffers carry the encode-side timebase while
  decode uses model-side decode timing

The first option is cleaner.


## 5. `T3` Responsibilities and Model Shape

`T3` is a token-to-token model. It does not synthesize waveform directly.

It consumes:

- text token IDs
- conditioning prefix embeddings
- an initial speech BOS token

It emits:

- autoregressive speech token IDs

Core implementation details from upstream:

- separate text and speech embeddings
- separate speech output head
- custom conditioning encoder (`T3CondEnc`)
- optional learned speech/text positional embeddings
- custom generation loop over speech tokens

Relevant files:

- `.model-src/chatterbox/src/chatterbox/models/t3/t3.py`
- `.model-src/chatterbox/src/chatterbox/models/t3/modules/cond_enc.py`
- `.model-src/chatterbox/src/chatterbox/models/t3/llama_configs.py`

### Original Chatterbox

Original Chatterbox uses:

- Llama-style backbone (`Llama_520M`)
- learned positional embeddings for text and speech
- speaker embedding conditioning
- prompt speech-token conditioning
- emotion conditioning
- CFG-like dual-batch generation path

### Turbo

Turbo is simpler and is the recommended first target:

- GPT-2 medium style backbone
- no learned speech/text position embeddings in the wrapper
- no perceiver resampler
- no emotion conditioning in practice
- no CFG path
- smaller speech vocab size setup

Source:

- `.model-src/chatterbox/src/chatterbox/tts_turbo.py`

### Practical implication

For `llama.cpp`, Turbo is the better first implementation target because it has:

- a simpler conditioning path
- a cleaner speech-token contract
- a simpler generation loop


## 6. S3Gen Responsibilities

`S3Gen` is not a plain discrete codec decoder. It is a conditioned decoder.

It needs:

- speech token IDs
- reference prompt speech tokens
- reference prompt mel features
- speaker embedding

Those are built by `embed_ref(...)` from a reference waveform.

Source:

- `.model-src/chatterbox/src/chatterbox/models/s3gen/s3gen.py`

The reference path internally uses:

- 24 kHz mel extraction for prompt features
- 16 kHz speaker embedding extraction
- 16 kHz S3 tokenization of the reference clip

This means `S3Gen` decode is a function of:

- payload token IDs
- side conditioning derived from a reference clip

### Important boundary rule

Even though `S3Gen` uses continuous conditioning internally, those continuous
values should stay fully inside `codec.cpp`.

The clean architecture is:

- application provides reference audio to `codec.cpp`
- `codec.cpp` computes all decode-side conditioning locally
- `llama.cpp` never sees prompt mels, speaker embeddings, or `ref_dict`


## 7. What Must Cross the Library Boundary

Between `llama.cpp` and `codec.cpp`, transfer only:

- payload speech token IDs as `int32`
- sequence length
- optional model-specific generation metadata if needed by the application,
  but not continuous model state

Do not transfer:

- speaker embeddings
- prompt mel tensors
- prompt speech embedding tensors
- hidden states
- quantized continuous latents

For voice cloning, both runtimes may consume the same raw reference audio, but
they should derive their own conditioning locally:

- `llama.cpp` derives `T3` conditioning
- `codec.cpp` derives `S3Gen` conditioning


## 8. What `codec.cpp` Must Implement

### `S3Tokenizer` (`S3T`)

`codec.cpp` encoder responsibilities:

- accept 16 kHz mono PCM
- reproduce upstream log-mel extraction behavior
- reproduce upstream tokenizer output IDs
- emit `n_q = 1`, `codebook_size = 6561`

Converter responsibilities:

- load upstream S3 tokenizer checkpoint
- convert all tensor layout changes at conversion time
- write GGUF metadata for encode/decode sample rates and token rate

### `S3Gen`

`codec.cpp` decoder responsibilities:

- accept payload speech token IDs `0..6560`
- compute or load reference conditioning locally
- decode to 24 kHz mono PCM

There are two practical decoder modes:

1. Fixed-voice mode
   Conditioning is baked into the GGUF or loaded once at runtime.
   This is the simplest first decode target.

2. Reference-conditioned mode
   A reference clip is provided at runtime and `codec.cpp` builds decode-side
   conditioning from it.

The second mode is required for full Chatterbox-style zero-shot voice cloning.


## 9. What `llama.cpp` Must Implement

### `T3`

`llama.cpp` responsibilities:

- text tokenization for the chosen model variant
- conditioning encoder path used by `T3`
- speech BOS/EOS handling
- autoregressive generation of speech token IDs
- filtering of generated IDs before handoff to `codec.cpp`

For the handoff to `codec.cpp`:

- remove BOS/EOS from the final payload stream
- remove or reject tokens `>= 6561`
- pass only payload IDs `0..6560`

### Model split for assistant systems

For a full audio/chat assistant this is usually not a single LLM.

Typical stack:

1. main dialogue model
2. `T3`
3. `codec.cpp` `S3G`

`T3` is specialized speech-token generation, not a replacement for the main
dialogue tower.


## 10. GGUF Conversion Notes

Chatterbox support will likely need multiple converters or converter modes.

### Recommended conversion units

1. `s3tokenizer`
2. `s3gen`
3. `t3_turbo`
4. optionally `t3_original`

The cleanest split is not one monolithic GGUF for all of Chatterbox.

### Why separate conversion units

- `codec.cpp` and `llama.cpp` own different runtime pieces
- `S3T` and `S3G` share token semantics but not the same runtime code path
- Turbo and original Chatterbox differ enough in `T3` that separate converter
  handling is likely cleaner

### Metadata to preserve

At minimum, converted models should preserve:

- encode sample rate
- decode sample rate
- token rate
- token codebook size
- `n_q`
- BOS/EOS and speech-vocab configuration for `T3`
- conditioning dimensions for speaker/prompt paths


## 11. Current Gaps in Local Source Tree

This repository currently has the Chatterbox wrapper code but not everything
needed to implement conversion immediately.

Notable missing pieces in the local tree:

- upstream `s3tokenizer` package source used by the wrapper
- locally cached Chatterbox or Chatterbox-Turbo checkpoints under
  `.model-src/models/hf`

Implication:

- architecture planning can be done now
- actual converter implementation needs the upstream checkpoint files and, for
  `S3Tokenizer`, the model package or a copied local source snapshot


## 12. Recommended Implementation Order

1. Extend `codec.cpp` metadata/API to represent separate encode and decode
   sample rates and hops cleanly.
2. Implement `S3Tokenizer` encode parity in `codec.cpp`.
3. Implement fixed-condition `S3Gen` decode in `codec.cpp`.
4. Implement runtime reference-conditioned `S3Gen` decode API.
5. Implement `T3` Turbo in `llama.cpp`.
6. Add end-to-end tests for:
   - `S3T` parity
   - `S3G` parity
   - `T3 -> payload IDs -> S3G`


## 13. Summary

The key design rules are:

- Chatterbox is a stack, not a single model.
- `codec.cpp` should own the S3 speech-token codec layer.
- `llama.cpp` should own `T3`.
- The only value that should cross the boundary is the discrete payload speech
  token stream.
- Payload speech token IDs are `0..6560`.
- Chatterbox uses 16 kHz encode and 24 kHz decode, so `codec.cpp` needs a clean
  dual-timebase representation.

