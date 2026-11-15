# RFC: codec memory (cached context in `codec_context`)

Status: **proposal, not implemented**. Nothing in `include/` or `common/*.h`
changes until this document is approved.

## 1. Summary

A `codec_context` can optionally own *memory*: the context a model needs from
earlier input, cached so it is not recomputed, as a `llama_context` caches K/V.
The cached context is:

* the K/V of a sliding attention window;
* the left tails of convolutions and the overlap partial sums of transposed
  convolutions;
* the position counter.

With memory on, each `codec_encode` / `codec_decode` call sees the cached
context of the previous calls instead of starting from silence.

Memory schedules no output. Each call returns exactly the output for its own
input: as many frames or samples as a stateless call on that input would
return, and the caller decides what to emit, trim or crossfade, as today. For
the models that support memory (causal ones, §5), the output of consecutive
calls equals the matching slice of the batch output.

Encoder memory and decoder memory are separate parts, enabled, cleared and
serialized independently. One sequence per context. Naming follows llama.cpp's
`llama_memory_t`.

Out of scope:

* the LLM's own context (its KV cache) is not codec memory; it stays in
  llama.cpp;
* reference-voice decoder state (§5).

## 2. Motivation

Today every public call is stateless. A streaming host that wants batch-quality
output therefore either decodes from frame 0 on every chunk, whose cost grows
with the square of the stream length, or re-decodes a left window around each
chunk (#6). Both recompute context the model already computed.

The cost on a Pixel 7 Pro (4 big cores, F32; RTF = decode time / audio length):

| Codec | Batch RTF | #6 window, 0.5 s chunks | #6 window, 2 s chunks |
|---|---|---|---|
| Qwen3-TTS 12 Hz | 1.48 | 6.2 | 3.7 |
| Mimi | 0.24 | 0.83 | 0.40 |

For a causal model the context is bounded and cheap to keep:

* each convolution needs its last (k − 1) · dilation input samples;
* each transposed convolution needs the k − stride overlap it has not yet
  added;
* each sliding-window attention layer needs its last W keys and values.

Caching these makes streaming cost the same as batch, and the output is the
batch output, not an approximation.

The encoder side needs the same thing:

* **Full duplex (Moshi style)**: the user's microphone is encoded continuously
  by a causal Mimi encoder and fed to the LLM frame by frame while the model
  speaks.
* **Streaming speech input to an audio LM** (LFM2-Audio and other Mimi-based
  LMs): the same pattern without the reply audio.

## 3. API (header diff sketch)

All additions go in `include/codec.h`. Nothing changes in `common/*.h`,
`include/codec_audio_lm.h` or `include/codec_speaker.h`.

```diff
@@ include/codec.h
+// Which parts of a context's memory: a bit set.
+enum codec_memory_type {
+    CODEC_MEMORY_NONE   = 0,
+    CODEC_MEMORY_ENCODE = 1,
+    CODEC_MEMORY_DECODE = 2,
+    CODEC_MEMORY_BOTH   = 3,   // ENCODE | DECODE
+};
+
+typedef struct codec_memory * codec_memory_t;
+
 struct codec_context_params {
     int32_t seed;
+    // Parts that cache context across codec_encode / codec_decode calls.
+    // Default CODEC_MEMORY_NONE: every call is independent, as today.
+    enum codec_memory_type memory;
 };
@@
+// Whether the model supports memory for `part` (ENCODE or DECODE).
+bool codec_model_has_memory(const struct codec_model * model, enum codec_memory_type part);
+
+// NULL when the context was created with memory == CODEC_MEMORY_NONE.
+codec_memory_t codec_get_memory(const struct codec_context * ctx);
+
+// Empty the cached context of the parts in `which` (start of a new stream).
+// The other part is untouched.
+void codec_memory_clear(codec_memory_t mem, enum codec_memory_type which);
+
+// Serialized cached context of the parts in `which` (§7). get_data returns the
+// bytes written (0 = dst too small); set_data returns the bytes read, or 0 when
+// the blob does not fit this model / part, leaving the memory unchanged.
+size_t codec_memory_state_get_size(codec_memory_t mem, enum codec_memory_type which);
+size_t codec_memory_state_get_data(codec_memory_t mem, enum codec_memory_type which, uint8_t * dst, size_t size);
+size_t codec_memory_state_set_data(codec_memory_t mem, enum codec_memory_type which, const uint8_t * src, size_t size);
```

* The `codec_encode` / `codec_decode` signatures are unchanged, and so are
  their parameter structs.
* No status code is added.
* `python/codec_cpp/_ffi.py` mirrors the new field and functions in the same
  commit.

**Mapping to llama.cpp**

| llama.cpp | codec.cpp |
|---|---|
| `llama_context_params` (KV types, `n_ctx`) | `codec_context_params.memory` |
| `llama_get_memory(ctx)` | `codec_get_memory(ctx)` |
| `llama_memory_clear(mem, data)` | `codec_memory_clear(mem, which)` |
| `llama_state_seq_get_size / get_data / set_data` (on ctx, per seq) | `codec_memory_state_*` (on mem, per part) |
| `llama_memory_seq_rm / cp / keep / add` | none (one sequence) |

**ABI.** `codec_context_params` grows from 4 to 8 bytes and is passed by value
to `codec_init_from_model`, so binaries built against the old header must be
rebuilt. Hosts that start from `codec_context_default_params()` (all in-tree
code, llama.rn) need no source change. Everything else is additive.

## 4. Semantics

### 4.1 General

* With a part enabled, each call reads that part's cached context, computes its
  input with it, and updates the cache in the same graph.
* The output shape is that of a stateless call on the same input.
* Memory never holds raw samples or frames that are waiting to be output. What
  it caches is context for future input, never output that has not been
  returned yet.
* The output of calls 1..n equals the slice of the batch output on the
  concatenated input, to rounding (§8). This holds for every model that reports
  memory support, because only causal models do (§5).
* `codec_memory_clear` returns a part to the start of a stream. The next call
  behaves like the first call of a batch: same left padding, position 0.

### 4.2 Encode

* **Every chunk must be hop-aligned**: `n_samples % hop == 0`, where hop is the
  product of the encoder's strides (`codec_model_hop_size`). Otherwise the call
  returns `CODEC_STATUS_INVALID_ARG`.
* **The caller pads the last chunk.** Nothing is buffered between calls.
* **What batch encode does at the tail** (checked on Mimi, the Qwen3-TTS
  tokenizer and MOSS-Audio-Nano):
  * Batch encode does not zero-pad the raw input to the hop. Following the
    reference (Moshi / HF `MimiConv1d`), each strided convolution right-pads its
    own input to a multiple of its stride, with zeros or, for Mimi's downsample
    convolution, by replicating the last value. The padding therefore happens at
    every resolution, after nonlinearities.
  * On hop-aligned input these extra paddings are all zero length. That is what
    makes chunking exact: `encode(x[: k·hop])` equals the first k frames of
    `encode(x)` bit for bit, for all three encoders (measured on a 6 s speech
    clip, k = 50).
  * Consequence for the caller's zero-padding of the last chunk:
    * the frame count matches batch (51 frames for 50⅓ hops);
    * every frame before the last is identical;
    * the last frame can differ, because batch pads per layer and the caller pads
      raw samples. Measured: Mimi 1.96 % of codes differ (all of them in the
      last frame), Qwen3 tokenizer 0.92 %, MOSS-Audio-Nano 1.96 %.
  * A host that needs batch codes for the final partial frame can re-encode that
    frame's window statelessly. Q2 covers making the library do it.
* `codec_encode_latent` shares the encode part.

### 4.3 Decode

* `codec_decode(ctx, tokens, out, p)` returns `tokens->n_frames × hop` samples,
  as a stateless call does. With memory, those samples are the matching slice
  of the batch decode:
  * the convolutions see the previous frames' tails instead of zero padding;
  * attention sees the cached K/V window;
  * the transposed convolutions' overlap from the previous chunk is added at the
    start. That overlap is a partial sum (an activation), cached like a K/V
    entry, and it is not held-back output.
* Batch decode trims the overlap that the last frame's transposed convolutions
  would add past the end. A streaming chunk does the same, and the trimmed part
  stays in the cache as context for the next chunk.
* `n_q`, the token layout and the decode parameters must stay the same within a
  stream. Changing them before a clear returns `CODEC_STATUS_INVALID_ARG`.
* `codec_decode_quantized_representation` shares the decode part.
* `codec_decode_batch` on a context with DECODE memory returns
  `CODEC_STATUS_NOT_SUPPORTED`. A batch holds several sequences, and memory
  holds one.

### 4.4 Lifecycle and graphs

* `codec_init_from_model` allocates the memory:
  * device buffers sized from the model's metadata (window lengths, channels,
    kernel sizes);
  * freed with the context.
* Creating a context fails, with `codec_get_last_error` naming the part, if a
  requested part is unsupported.
* With `memory == NONE`, behaviour and numerics are byte-identical to today.
* *One model = one graph per public call* still holds:
  * the cache tensors are graph inputs;
  * the updated cache is written back in the same graph (`ggml_cpy` into the
    memory buffer), as llama.cpp writes its KV cache;
  * ring offsets and positions are small input tensors, so a graph is reused
    for every chunk of the same length.

  The graph cache key gains the memory flag. Hosts get the best cache hit rate
  with a fixed chunk size.

## 5. Per codec family (v1 = causal models only)

Reach is from an impulse test on the host: one frame's codes changed, and the
change in output measured before ("look-ahead") and after ("memory") that frame,
at −40 dB.

| Family (archs) | Encode | Decode | What the memory caches |
|---|---|---|---|
| **Mimi** (`MIMI`; LFM2-Audio, Moshi; `POCKET_MIMI` for the decoder) | causal SEANet convs, stride-2 downsample, causal transformer with a 250-frame window: **supported** | mirror image, causal, look-ahead 0, −40 dB memory 0.24 s (Moshi's Mimi 2.0 s): **supported** | conv tails (< 1 MB), with the replicate-padded downsample cache seeded from the first input value after a clear, as batch pads; K/V 8 layers × 250 frames × 512 × 2 ≈ 8 MB F32 (4 MB F16) per part; position |
| **Qwen3-TTS 12 Hz tokenizer** (`QWEN3_TTS_TOKENIZER`) | Mimi encoder (`qwen3.encoder.*`): **supported**, same as Mimi | causal convs, 8 transformer layers with a 72-frame window, causal upsampling convtr; look-ahead 0: **supported** | K/V 8 × 71 frames × 1024 × 2 ≈ 4.6 MB F32; conv tails and convtr overlaps; position. The same contents as upstream llama.cpp's `mtmd_gen_state` |
| **MOSS-Audio** (`MOSS_AUDIO`; Nano, full, TTS-Realtime) | patched pretransform + causal sliding-window transformers per stage: **supported** | mirror image, look-ahead 0, −40 dB memory 17–24 s: **supported** | K/V per stage, window = `context_duration × stage frame rate` (the largest of the set); conv tails; position. The hop is a whole patch (3840 samples for Nano at 48 kHz stereo) |
| **NeMo nano codec** | not a streaming target | causal conv, memory 2.2 s: **supported** | conv tails only |
| **DAC, SNAC** | non-causal | non-causal, reach ±0.13–0.20 s: **unsupported** in v1 | |
| **WavTokenizer** | non-causal | global attention, whole-clip reach: **unsupported** in v1 | |
| XCodec2, NeuCodec, Chatterbox S3G, latent decoders, CosyVoice, XY-Tokenizer, Whisper-style encoders | | **unsupported** | |

**Non-causal models.** A non-causal model's output for a frame depends on
frames after it, so cached context alone cannot reproduce batch output. The
caller's windowed re-decode (#6) stays the method:

* right look-ahead, trimming and crossfade remain the caller's;
* memory could at most cache the left-context intermediates of the last L
  frames (activations, K/V), so the re-decode does not recompute them;
* the saving is bounded by L / (L + chunk + R), and the cached left
  activations of a non-causal network still depend on the old right padding,
  which needs model-specific handling.

**Proposal:** unsupported in v1, revisited per model when a host shows the
re-decode cost matters. The phone numbers say it matters most for DAC: RTF 2.5
windowed vs 1.11 batch at 0.5 s chunks.

**Reference voices** (Qwen3-TTS ICL): the decoder state for a reference voice
is owned by `codec_speaker`, precomputed when the speaker GGUF is built and
applied internally when a decode uses that speaker. That is a separate future
`codec_speaker` proposal, not part of this API.

## 6. Relation to #6 (windowed stateless streamer)

#6 merges first, on its own terms: it changes no public header beyond a comment
and needs nothing from this RFC. Afterwards:

* For a codec where `codec_model_has_memory(model, DECODE)` is true,
  `alm_streamer` (`common/audio_lm_c.cpp`) opens its codec context with DECODE
  memory and decodes only the new frames each poll. The left window and the
  20 ms crossfade are then unnecessary, because consecutive outputs join
  exactly.
  * Expected win on the phone: Qwen3 streaming goes from RTF 6.2 to the batch
    RTF of about 1.5.
  * The Qwen3 ICL gap goes away once the reference voice state exists (future
    `codec_speaker` proposal).
* For every other codec, #6's host-side windowing (L / R table, crossfade)
  stays unchanged as the fallback, and decoding from frame 0 stays the default
  for codecs with no window entry.
* Emission, trimming and crossfading stay with the streamer in both paths. The
  library never takes them over.

`tests/e2e/audio_lm_stream_smoke.py` covers both paths. With memory, the stream
must match batch to rounding.

## 7. State serialization

The blob is little-endian and self-describing:

```
header   char[4]  "CDCM"
         u32      format_version        (1)
         u32      arch                  (enum codec_arch)
         u8[32]   model_fingerprint     (SHA-256 over tensor names, types, shapes and the
                                         graph-relevant GGUF keys; computed once at load)
         u32      parts                 (codec_memory_type bits present)
per part u32      part                  (ENCODE / DECODE)
         u32      impl_version          (per-arch cache layout version)
         i64      n_frames              (position)
         i32      n_q
         u32      n_sections
section  u32 tag, u32 ggml_type, i64 ne[4], u64 nbytes, u8 data[nbytes]   (8-byte aligned)
```

* Section tags name the cached tensors: `conv_tail[i]`, `convtr_overlap[i]`,
  `k[l]`, `v[l]`, `ring_head`.
* Data is copied with `ggml_backend_tensor_get`, so a blob saved on the GPU
  restores on the CPU and the other way round. It is tied to the model file
  through the fingerprint, not to the backend.
* `set_data` validates everything (magic, versions, arch, fingerprint, part,
  section shapes) before touching the memory. Any mismatch returns 0 and leaves
  the cache unchanged.
* **Versioning**: no migration. The cache can always be recomputed by
  re-feeding the input. `format_version` changes with the header and
  `impl_version` with an arch's layout. Old blobs are rejected.

## 8. Thread safety and numerics

* Thread rules are unchanged: one `codec_context` is used by one thread at a
  time, and its memory is part of it.
* Calling `state_get_data` while another thread encodes on the same context is
  a data race.
* Contexts of one model are independent, since each has its own backend
  instance. Full duplex therefore uses two contexts: encode on one thread,
  decode on another.
* One context with `BOTH` is allowed, but its calls are serialized because they
  share the scheduler and eval arena.
* The memory handle is valid until `codec_free`. It is not reference counted.
* **Numerics.** Chunked compute changes matmul tiling, so streamed output
  matches batch to rounding, not bit for bit. Smokes assert:
  * PCM: max |Δ| ≤ 1e-4;
  * codes: at most 0.1 % flipped, and only at quantizer ties.
* Cached tensors keep the precision the batch graph computes them at.

## 9. Open questions

* **Q1. Non-causal support.** The proposal is "later, reported as unsupported
  in v1" (§5). Confirm, or name a model that must be in v1. DAC is the
  candidate, since its windowed re-decode costs 2.5× batch on the phone.
* **Q2. The final partial frame.** Hop-aligned input only, with the caller
  zero-padding the last chunk, means the last frame's codes can differ from
  batch (§4.2). The alternative: accept an unaligned last chunk, apply batch's
  per-layer tail padding, and leave the part needing a clear before further
  input. Is the last-frame difference acceptable, or should v1 offer that?
* **Q3. Sample rate.** Streamed encode input must be at the model's encode
  rate. Resampling in a stream needs its own cached tail, so v1 leaves it to the
  host. OK?
* **Q4. Graph cache with irregular chunks.** Should the library pad chunks up
  to bucket sizes, to cap the number of cached graphs? Padding would conflict
  with the exact-output rule, so the alternative is to tell hosts to use a fixed
  chunk size and accept more graphs otherwise.

## 10. Test plan

* **Stream vs batch** on hop-aligned random chunk sizes down to one frame:
  * Mimi encode and decode;
  * Qwen3-TTS tokenizer encode and decode;
  * MOSS-Audio-Nano encode and decode;
  * NeMo decode.

  Tolerances are those of §8.
* **Final chunk**: caller zero-padding gives batch codes for every frame but
  the last.
* **`set_data(get_data())` round trip**: the continuation is identical to an
  uninterrupted stream. A blob from another model or part is rejected and the
  cache is unchanged.
* **`clear`** equals a fresh context, and the other part is untouched.
* **Phone** (Pixel 7 Pro): streaming RTF vs batch for Qwen3 and Mimi decode.
  Target: within 10 % of batch at 0.5 s chunks.
