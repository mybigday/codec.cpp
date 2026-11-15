# RFC: codec memory (cached context in `codec_context`)

Status: **proposal, not implemented**. Nothing in `include/` or `common/*.h`
changes until this document is approved.

## 1. Summary

Minimal change: a `codec_context` can optionally own an encoder memory and a
decoder memory. Each is three things:

1. a cache of the context that side needs from earlier input, as a
   `llama_context` caches K/V;
2. a clear;
3. state save and restore.

What the cache holds:

* the K/V of a sliding attention window;
* the left tails of convolutions and the overlap partial sums of transposed
  convolutions;
* the position counter.

With a memory on, each `codec_encode` / `codec_decode` call on that side uses
the cached context of the previous calls instead of starting from silence.

Nothing else changes:

* each call processes exactly the input it is given, as today, including the
  padding it applies;
* each call returns the same output shape as today;
* the caller decides what to emit, trim or crossfade.

For a causal side, consecutive calls produce the matching slices of the batch
output.

Each side's context reach is recorded in the GGUF as `look_back` /
`look_ahead` frames (§5).

Out of scope:

* the LLM's own context (its KV cache) stays in llama.cpp;
* reference-voice decoder state (§6).

## 2. Motivation

Every public call is stateless today. A streaming host that wants batch-quality
output therefore either decodes from frame 0 on every chunk, whose cost grows
with the square of the stream length, or re-decodes a left window around each
chunk (#6). Both recompute context the model already computed.

The cost on a Pixel 7 Pro (4 big cores, F32; RTF = decode time / audio length):

| Codec | Batch RTF | #6 window, 0.5 s chunks | #6 window, 2 s chunks |
|---|---|---|---|
| Qwen3-TTS 12 Hz | 1.48 | 6.2 | 3.7 |
| Mimi | 0.24 | 0.83 | 0.40 |

For a causal model the context is bounded:

* each convolution needs its last (k − 1) · dilation inputs;
* each transposed convolution needs the overlap it has not yet added;
* each sliding-window attention layer needs its last W keys and values.

Caching these makes streaming cost the same as batch, with exact output.

The same applies on the encoder side:

* full duplex (Moshi style), where the microphone is encoded continuously by a
  causal Mimi encoder while the model speaks;
* streaming speech input to Mimi-based audio LMs such as LFM2-Audio.

## 3. API (header diff sketch)

All additions go in `include/codec.h`. Nothing changes in `common/*.h`,
`include/codec_audio_lm.h` or `include/codec_speaker.h`.

```diff
@@ include/codec.h
+typedef struct codec_memory * codec_memory_t;
+
 struct codec_context_params {
     int32_t seed;
+    // Cache context across calls (default false: every call independent, as today).
+    bool encoder_memory;   // codec_encode / codec_encode_latent
+    bool decoder_memory;   // codec_decode / codec_decode_quantized_representation
 };
@@
+// True when that side's output for a frame depends only on the current and
+// earlier input: its look_ahead (GGUF codec.<side>.look_ahead) is 0. Memory
+// then makes streamed output equal batch output; a non-causal side's memory
+// has size 0.
+bool codec_model_encoder_is_causal(const struct codec_model * model);
+bool codec_model_decoder_is_causal(const struct codec_model * model);
+
+// NULL when that memory is not enabled in the context params.
+codec_memory_t codec_get_encoder_memory(const struct codec_context * ctx);
+codec_memory_t codec_get_decoder_memory(const struct codec_context * ctx);
+
+// Empty the cache: the next call behaves as on a fresh context.
+void codec_memory_clear(codec_memory_t mem);
+
+// Save / restore the cache (§7). get_data returns the bytes written (0 = dst
+// too small); set_data returns the bytes read, or 0 when the blob does not fit
+// this model and side, leaving the cache unchanged.
+size_t codec_memory_state_get_size(codec_memory_t mem);
+size_t codec_memory_state_get_data(codec_memory_t mem, uint8_t * dst, size_t size);
+size_t codec_memory_state_set_data(codec_memory_t mem, const uint8_t * src, size_t size);
```

* **One handle per side.** There is no part selector: a context has an
  encoder memory, a decoder memory, both or neither, and every memory call acts
  on the one handle it is given.
* **State lives on the memory handle, not on the context.** This deliberately
  departs from `llama_state_*(ctx, …)`, so that it is always clear which side's
  state is saved.
* **`codec_model_*_is_causal`** is a model property. A host such as the #6
  streamer needs the answer before it creates a context, to choose its path.
* The `codec_encode` / `codec_decode` signatures, their parameter structs and
  the status codes are unchanged.
* `python/codec_cpp/_ffi.py` mirrors the new fields and functions in the same
  commit.

**Mapping to llama.cpp**

| llama.cpp | codec.cpp |
|---|---|
| `llama_context_params` (KV settings) | `codec_context_params.encoder_memory` / `decoder_memory` |
| `llama_get_memory(ctx)` | `codec_get_encoder_memory(ctx)` / `codec_get_decoder_memory(ctx)` |
| `llama_memory_clear(mem, data)` | `codec_memory_clear(mem)` |
| `llama_state_get_size / get_data / set_data(ctx, …)` | `codec_memory_state_get_size / get_data / set_data(mem, …)` (on the handle, by design) |
| `llama_memory_seq_rm / cp / keep / add` | none (one sequence) |

**ABI.** `codec_context_params` grows (two `bool`s; 8 bytes with padding) and is
passed by value to `codec_init_from_model`, so binaries built against the old
header must be rebuilt. Hosts that start from `codec_context_default_params()`
(all in-tree code, llama.rn) need no source change. Everything else is
additive.

## 4. Semantics

* **Calls.** With a side's memory enabled, that side's calls read its cache in
  place of the zero history a fresh call starts from. Each call updates the
  cache with the context its input leaves behind, in the same graph. Input
  handling, padding and output shape are exactly as today.
* **Batch output.** On a causal side, the outputs of consecutive calls equal
  the matching slices of the batch output on the concatenated input, to rounding
  (§8).
* **Clear.** `codec_memory_clear` empties one side's cache. Use it before input
  that does not continue the previous input. One case is the caller's
  responsibility: after a chunk the call padded (not a whole number of hops),
  the cache ends in padding, so clear before continuing.
* **Non-causal sides.** On a side with `look_ahead > 0`, enabling memory is
  valid and changes nothing:
  * the state size is 0;
  * get and set of the empty blob succeed;
  * clear does nothing.
* **One required restriction.** `codec_decode_batch` neither reads nor updates
  the decoder memory. A batch holds several independent sequences, and memory
  holds one.

Implementation note: the cache tensors are graph inputs, and the updated cache
is written back in the same graph, as llama.cpp writes its KV cache. That keeps
*one model = one graph per public call*.

## 5. GGUF metadata: look_back / look_ahead

Four keys, in frames of that side (codec frames, at the hop rate), written by
the converters:

| Key | Type | Meaning |
|---|---|---|
| `codec.encoder.look_back` | i32 | past frames of context the encoder uses |
| `codec.encoder.look_ahead` | i32 | future frames the encoder's output for a frame depends on |
| `codec.decoder.look_back` | i32 | as above, for the decoder |
| `codec.decoder.look_ahead` | i32 | as above, for the decoder |

* **Causality.** `is_causal` ⇔ `look_ahead == 0`. The value −1 means
  unbounded (global attention, flow): the side is non-causal and no window
  covers its reach.
* **On a causal side, `look_back` is the cached context length**: the number
  of past frames the memory keeps.
  * For an attention layer that is its sliding window, converted to codec
    frames: the cache holds that window, minus the current step, of K/V per
    layer, at the layer's own rate. Mimi's transformers run at twice the
    codec frame rate, so their 250-step window is 125 codec frames.
  * Convolution tails are shorter than one look-back and are kept per layer as
    the kernel needs.
  * Memory size is therefore about `n_layers × look_back × kv_dim × 2 ×
    sizeof(type)` plus small conv tails. Caching that window exactly reproduces
    batch output, even where the impulse response reaches further, because each
    layer's cached K/V already folds in what came before.
* **On a non-causal side**, `look_back` / `look_ahead` are the left and right
  windows a host re-decodes around each chunk without memory: #6's `left` /
  `right`, in frames instead of seconds. Memory size is 0.
* **Loading old GGUFs.** When the keys are missing, the loader falls back to
  per-arch defaults. New code must load old GGUFs. #6's hard-coded table in
  `src/audio_lm_window.h` becomes those defaults (for non-causal sides), and
  the streamer reads the metadata first.

Per-arch values (these become the defaults; frames at each model's own rate):

| Arch | Encoder look_back / look_ahead | Decoder look_back / look_ahead | Source |
|---|---|---|---|
| Mimi (12.5 Hz) | 125 / 0 | 125 / 0 | transformer sliding window: 250 steps (HF `sliding_window`) at the transformer's 25 Hz = 125 codec frames; causal convs |
| Qwen3-TTS tokenizer (12.5 Hz) | 125 / 0 (Mimi encoder, same 250-step window at 25 Hz) | 72 / 0 | decoder `qwen3.decoder.sliding_window` 72 (its transformer runs at 12.5 Hz) |
| MOSS-Audio | per-stage `context_duration` in frames / 0 | same / 0 | converter computes it from the stage windows |
| NeMo nano codec | (no encoder streaming) | 27 / 0 | conv receptive field (impulse: 2.2 s) |
| DAC 16 kHz (50 Hz) | n/a | 16 / 16 | #6 window 0.32 s / 0.32 s (reach ±10 frames) |
| WavTokenizer large (75 Hz) | n/a | 96 / 24 | #6's accepted window 1.28 s / 0.32 s |
| SNAC | n/a | 16 / 16 (proposed) | reach ±9 frames; not yet in #6 |
| XCodec2, NeuCodec, Chatterbox S3G, latent decoders | n/a | −1 / −1 | global attention or flow |

* "n/a" encoder entries: the converter writes the measured reach of the
  encoder when that side is converted. Until then, a missing encoder key on a
  non-causal arch defaults to −1 / −1.
* #6's 1.28 s windows for Mimi and Qwen3 decoders do not carry over: those
  decoders are causal, so the streamer takes the memory path (§9) and needs no
  window.
* DAC is per file: the 24 kHz and 44 kHz variants have other frame rates, so
  their converter writes their own frame counts.

## 6. Reference voices

The decoder state for a reference voice (Qwen3-TTS ICL) belongs to
`codec_speaker`. It would be precomputed when the speaker GGUF is built and
applied internally when a decode uses that speaker. That is a future
`codec_speaker` proposal.

## 7. State save / restore

Per memory handle, the blob is little-endian and self-describing:

```
header   char[4]  "CDCM"
         u32      format_version        (1)
         u32      arch                  (enum codec_arch)
         u32      side                  (0 = encoder, 1 = decoder)
         u8[32]   model_fingerprint     (SHA-256 over tensor names, types, shapes and the
                                         graph-relevant GGUF keys; computed once at load)
         u32      impl_version          (per-arch cache layout version)
         i64      position
         u32      n_sections
section  u32 tag, u32 ggml_type, i64 ne[4], u64 nbytes, u8 data[nbytes]   (8-byte aligned)
```

* A non-causal side's memory has size 0: nothing is written, and setting a
  0-byte blob succeeds.
* Section tags name the cached tensors: `conv_tail[i]`, `convtr_overlap[i]`,
  `k[l]`, `v[l]`, `ring_head`.
* Data is copied with `ggml_backend_tensor_get`, so a blob saved on the GPU
  restores on the CPU and the other way round. It is tied to the model file and
  the side through the fingerprint and the `side` field.
* `set_data` validates the blob before touching the cache. A mismatch returns 0
  and leaves the cache unchanged.
* **Versioning**: no migration. The cache can always be rebuilt by re-feeding
  the input. Old `format_version` / `impl_version` blobs are rejected.

## 8. Thread safety and numerics

* Thread rules are unchanged: one `codec_context` is used by one thread at a
  time, and its memories are part of it.
* Contexts of one model are independent, since each has its own backend
  instance. Full duplex uses two contexts, one per thread.
* A memory handle is valid until `codec_free`.
* **Numerics.** Chunked compute changes matmul tiling, so streamed output
  matches batch to rounding, not bit for bit. Smokes assert:
  * PCM: max |Δ| ≤ 1e-4;
  * codes: at most 0.1 % flipped, and only at quantizer ties.

## 9. Relation to #6 (windowed stateless streamer)

#6 merges first, on its own terms: it changes no public header beyond a comment
and needs nothing from this RFC. Afterwards:

* When `codec_model_decoder_is_causal(model)` is true, `alm_streamer`
  (`common/audio_lm_c.cpp`) takes the incremental path: it opens its codec
  context with `decoder_memory = true` and decodes only the new frames each
  poll. Consecutive outputs join exactly, so no window or crossfade is needed.
  * Expected win on the phone: Qwen3 streaming goes from RTF 6.2 to the batch
    RTF of about 1.5.
* Otherwise it keeps #6's windowed path. The window becomes the decoder's
  `look_back` / `look_ahead`, from metadata or the per-arch default. With −1, it
  decodes from frame 0 as today.
* Emission, trimming and crossfading stay with the streamer in both paths.

## 10. Rollout

1. Approve this RFC.
2. Implement the API, the loader keys with per-arch defaults, and the cache for
   the causal archs. Old GGUFs keep working through the defaults.
3. Update the converters to write the four keys.
4. Update the published GGUFs in place with the new metadata, which changes no
   weights. Each upload is approved separately.

## 11. Test plan

* **Stream vs batch** on random hop-aligned chunk sizes down to one frame:
  * Mimi encode and decode;
  * Qwen3-TTS tokenizer encode and decode;
  * MOSS-Audio-Nano encode and decode;
  * NeMo decode.

  Tolerances are those of §8.
* **`set_data(get_data())` round trip**: the continuation is identical to an
  uninterrupted stream. A blob from another model or side is rejected and the
  cache is unchanged.
* **`clear`** on one side equals a fresh context for that side, and the other
  side is untouched.
* **Non-causal side** (DAC decoder) with memory on: output is byte-identical to
  memory off, the state size is 0, and an empty get/set succeeds.
* **Old GGUF** (no look keys): the loader gives the per-arch defaults, and the
  #6 smoke output is unchanged.
* **Phone** (Pixel 7 Pro): streaming RTF vs batch for Qwen3 and Mimi decode.
