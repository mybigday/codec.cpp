# RFC: codec memory (cached context in `codec_context`)

Status: **proposal, not implemented**. Nothing in `include/` or `common/*.h`
changes until this document is approved.

## 1. Summary

Minimal change: a `codec_context` can optionally own *memory*, which is three
things:

1. a cache of the context a model needs from earlier input, as a
   `llama_context` caches K/V;
2. a clear;
3. state save and restore.

What the cache holds:

* the K/V of a sliding attention window;
* the left tails of convolutions and the overlap partial sums of transposed
  convolutions;
* the position counter.

With memory on, each `codec_encode` / `codec_decode` call uses the cached
context of the previous calls instead of starting from silence.

Nothing else changes:

* each call processes exactly the input it is given, as today, including the
  padding it applies;
* each call returns the same output shape as today;
* the caller decides what to emit, trim or crossfade.

For a causal model, consecutive calls then produce the matching slices of the
batch output.

Encoder memory and decoder memory are separate parts, enabled, cleared and
saved independently. One sequence per context. Naming follows llama.cpp's
`llama_memory_t`.

Out of scope:

* the LLM's own context (its KV cache) stays in llama.cpp;
* reference-voice decoder state (§5).

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
+// Parts of a codec: a bit set.
+enum codec_part {
+    CODEC_PART_NONE    = 0,
+    CODEC_PART_ENCODER = 1,
+    CODEC_PART_DECODER = 2,
+    CODEC_PART_ALL     = 3,   // ENCODER | DECODER
+};
+
+typedef struct codec_memory * codec_memory_t;
+
 struct codec_context_params {
     int32_t seed;
+    // Parts that cache context across codec_encode / codec_decode calls.
+    // Default CODEC_PART_NONE: every call is independent, as today.
+    enum codec_part memory;
 };
@@
+// True when `part`'s (ENCODER or DECODER) output for a frame depends only on the
+// current and earlier input (no look-ahead). Then memory makes streamed output
+// equal batch output. A non-causal part has memory size 0.
+bool codec_model_is_causal(const struct codec_model * model, enum codec_part part);
+
+// NULL when the context was created with memory == CODEC_PART_NONE.
+codec_memory_t codec_get_memory(const struct codec_context * ctx);
+
+// Empty the cache of `parts`. Other parts are untouched.
+void codec_memory_clear(codec_memory_t mem, enum codec_part parts);
+
+// Save / restore the cached context of `parts` (§6), as llama_state_*.
+// get_data returns the bytes written (0 = dst too small); set_data returns the
+// bytes read, or 0 when the blob does not fit this model / parts, leaving the
+// cache unchanged.
+size_t codec_state_get_size(struct codec_context * ctx, enum codec_part parts);
+size_t codec_state_get_data(struct codec_context * ctx, enum codec_part parts, uint8_t * dst, size_t size);
+size_t codec_state_set_data(struct codec_context * ctx, enum codec_part parts, const uint8_t * src, size_t size);
```

* **`codec_model_is_causal`** is a model property. A host such as the #6
  streamer needs the answer before it creates a context, to choose its path.
  For a causal part, memory makes streamed output equal batch output. A
  non-causal part has memory size 0.
* **`enum codec_part`** names a codec's parts. It is shared by the `memory`
  field, `codec_memory_clear`, the state functions and `codec_model_is_causal`,
  so it is not memory-specific.
* The `codec_encode` / `codec_decode` signatures, their parameter structs and
  the status codes are unchanged.
* `python/codec_cpp/_ffi.py` mirrors the new field and functions in the same
  commit.

**Mapping to llama.cpp**

| llama.cpp | codec.cpp |
|---|---|
| `llama_context_params` (KV types, `n_ctx`) | `codec_context_params.memory` |
| `llama_get_memory(ctx)` | `codec_get_memory(ctx)` |
| `llama_memory_clear(mem, data)` | `codec_memory_clear(mem, parts)` |
| `llama_state_get_size / get_data / set_data(ctx, …)` | `codec_state_get_size / get_data / set_data(ctx, parts, …)` |
| `llama_memory_seq_rm / cp / keep / add` | none (one sequence) |

**ABI.** `codec_context_params` grows from 4 to 8 bytes and is passed by value
to `codec_init_from_model`, so binaries built against the old header must be
rebuilt. Hosts that start from `codec_context_default_params()` (all in-tree
code, llama.rn) need no source change. Everything else is additive.

## 4. Semantics

* **Calls.** With a part enabled, `codec_encode` / `codec_encode_latent` or
  `codec_decode` / `codec_decode_quantized_representation` reads that part's
  cache in place of the zero history a fresh call starts from. It updates the
  cache with the context its input leaves behind, in the same graph. Input
  handling, padding and output shape are exactly as today.
* **Batch output.** For a model with memory, the outputs of consecutive calls
  equal the matching slices of the batch output on the concatenated input, to
  rounding (§7).
* **Clear.** `codec_memory_clear` empties the cache, so the next call behaves
  like a call on a fresh context. Use it before input that does not continue
  the previous input. One case is the caller's responsibility: after a chunk
  the call padded (not a whole number of hops), the cache ends in padding, so
  clear before continuing.
* **Non-causal parts.** When `codec_model_is_causal` is false for a part, its
  memory size is 0, and enabling memory on it is valid and changes nothing:
  * the state size is 0;
  * get and set of the empty blob succeed;
  * clear does nothing.
* **One required restriction.** `codec_decode_batch` neither reads nor updates
  the memory. A batch holds several independent sequences, and memory holds
  one.

Implementation note: the cache tensors are graph inputs, and the updated cache
is written back in the same graph, as llama.cpp writes its KV cache. That keeps
*one model = one graph per public call*.

## 5. Per codec family

Reach is from an impulse test on the host: one frame's codes changed, and the
change in output measured before ("look-ahead") and after ("memory") that frame,
at −40 dB.

| Family (archs) | Encoder causal (memory) | Decoder causal (memory) | What the cache holds |
|---|---|---|---|
| **Mimi** (`MIMI`; LFM2-Audio, Moshi; `POCKET_MIMI` decoder) | yes: causal SEANet convs, stride-2 downsample, causal transformer with a 250-frame window | yes: mirror image, causal, −40 dB memory 0.24 s (Moshi's Mimi 2.0 s) | conv tails (< 1 MB); K/V 8 layers × 250 frames × 512 × 2 ≈ 8 MB F32 per part; position |
| **Qwen3-TTS 12 Hz tokenizer** (`QWEN3_TTS_TOKENIZER`) | yes: Mimi encoder (`qwen3.encoder.*`) | yes: causal convs, 8 transformer layers with a 72-frame window, causal upsampling convtr | K/V 8 × 71 frames × 1024 × 2 ≈ 4.6 MB F32; conv tails and convtr overlaps; position. The same contents as upstream llama.cpp's `mtmd_gen_state` |
| **MOSS-Audio** (`MOSS_AUDIO`; Nano, full, TTS-Realtime) | yes: causal sliding-window transformers per stage | yes: mirror image, −40 dB memory 17–24 s | K/V per stage, window = `context_duration × stage frame rate`; conv tails; position |
| **NeMo nano codec** | | yes: causal conv, memory 2.2 s | conv tails |
| **DAC, SNAC, WavTokenizer**, XCodec2, NeuCodec, Chatterbox S3G, latent decoders, CosyVoice, XY-Tokenizer, Whisper-style encoders | no (memory 0) | no (memory 0) | nothing |

* **Models that look ahead** (non-causal convs, global attention) produce
  output that depends on later input. They have no earlier context to remember,
  so their memory is 0 and the caller's windowed re-decode (#6) remains how they
  stream.
* **Reference voices** (Qwen3-TTS ICL): the decoder state for a reference voice
  belongs to `codec_speaker`. It would be precomputed when the speaker GGUF is
  built and applied internally when a decode uses that speaker. That is a future
  `codec_speaker` proposal.

## 6. State save / restore

The blob is little-endian and self-describing:

```
header   char[4]  "CDCM"
         u32      format_version        (1)
         u32      arch                  (enum codec_arch)
         u8[32]   model_fingerprint     (SHA-256 over tensor names, types, shapes and the
                                         graph-relevant GGUF keys; computed once at load)
         u32      parts                 (codec_part bits present)
per part u32      part                  (ENCODER / DECODER)
         u32      impl_version          (per-arch cache layout version)
         i64      position
         u32      n_sections
section  u32 tag, u32 ggml_type, i64 ne[4], u64 nbytes, u8 data[nbytes]   (8-byte aligned)
```

* For a part with no memory, the size is 0 and no bytes are written. Setting a
  0-byte blob succeeds.
* Section tags name the cached tensors: `conv_tail[i]`, `convtr_overlap[i]`,
  `k[l]`, `v[l]`, `ring_head`.
* Data is copied with `ggml_backend_tensor_get`, so a blob saved on the GPU
  restores on the CPU and the other way round. It is tied to the model file
  through the fingerprint.
* `set_data` validates the blob before touching the cache. A mismatch returns 0
  and leaves the cache unchanged.
* **Versioning**: no migration. The cache can always be rebuilt by re-feeding
  the input. Old `format_version` / `impl_version` blobs are rejected.

## 7. Thread safety and numerics

* Thread rules are unchanged: one `codec_context` is used by one thread at a
  time, and its memory is part of it.
* Contexts of one model are independent, since each has its own backend
  instance. Full duplex uses two contexts, one per thread.
* The memory handle is valid until `codec_free`.
* **Numerics.** Chunked compute changes matmul tiling, so streamed output
  matches batch to rounding, not bit for bit. Smokes assert:
  * PCM: max |Δ| ≤ 1e-4;
  * codes: at most 0.1 % flipped, and only at quantizer ties.

## 8. Relation to #6 (windowed stateless streamer)

#6 merges first, on its own terms: it changes no public header beyond a comment
and needs nothing from this RFC. Afterwards:

* When `codec_model_is_causal(model, CODEC_PART_DECODER)` is true,
  `alm_streamer` (`common/audio_lm_c.cpp`) takes the incremental path: it opens
  its codec context with `memory = CODEC_PART_DECODER` and decodes only the new
  frames each poll. Consecutive outputs then
  join exactly, so the left window and crossfade are unnecessary.
  * Expected win on the phone: Qwen3 streaming goes from RTF 6.2 to the batch
    RTF of about 1.5.
* For every other codec, #6's host-side windowing stays unchanged as the
  fallback.
* Emission, trimming and crossfading stay with the streamer in both paths.

## 9. Test plan

* **Stream vs batch** on random hop-aligned chunk sizes down to one frame:
  * Mimi encode and decode;
  * Qwen3-TTS tokenizer encode and decode;
  * MOSS-Audio-Nano encode and decode;
  * NeMo decode.

  Tolerances are those of §7.
* **`set_data(get_data())` round trip**: the continuation is identical to an
  uninterrupted stream. A blob from another model or part is rejected and the
  cache is unchanged.
* **`clear`** equals a fresh context, and the other part is untouched.
* **No-memory model** (DAC) with memory on: output is byte-identical to memory
  off, the state size is 0, and an empty get/set succeeds.
* **Phone** (Pixel 7 Pro): streaming RTF vs batch for Qwen3 and Mimi decode.
