# RFC: codec memory (streaming encode / decode state in `codec_context`)

Status: **proposal, not implemented**. Nothing in `include/` or `common/*.h`
changes until this document is approved.

## 1. Summary

A `codec_context` can optionally own *memory*: the state an encoder or decoder
carries from one call to the next, the way a `llama_context` owns its KV cache.
With memory on, each `codec_encode` / `codec_decode` call continues the stream
from the previous call instead of starting from silence, and returns only
output that is final. Encoder memory and decoder memory are separate parts,
enabled, cleared and serialized independently. One sequence per context.

Naming follows llama.cpp's `llama_memory_t` (`llama_get_memory`,
`llama_memory_clear`, `llama_state_seq_get_size / get_data / set_data`).

Out of scope: the LLM's own context (its KV cache) is not codec memory. It
stays in llama.cpp.

## 2. Motivation

Today every public call is stateless. Streaming hosts get around this in
two ways, and both are costly:

* **Decode from frame 0 each time** (the streamer on main). Cost grows with
  the square of the stream length. On a Pixel 7 Pro (4 big cores, F32), a
  Qwen3-TTS reply with a reference voice took 89 s to stream against 29 s in batch.
* **Overlapping windows** (#6). Each chunk re-decodes L frames of left context
  and holds back R frames of right context. The cost is linear but inflated by
  (L + chunk + R) / chunk, and the output is only approximate when the codec's
  reach is longer than L. The table below is from the same phone (RTF = decode
  time / audio length):

  | Codec | Batch RTF | Windowed RTF, 0.5 s chunks | Windowed RTF, 2 s chunks |
  |---|---|---|---|
  | Qwen3-TTS 12 Hz | 1.48 | 6.2 | 3.7 |
  | DAC 16 kHz | 1.11 | 2.5 | 1.6 |
  | Mimi | 0.24 | 0.83 | 0.40 |
  | WavTokenizer | 0.18 | 0.74 | |

  For Qwen3 and DAC, windowing pushes a decoder that is near or above real
  time well past it.

Causal codecs don't need either workaround. Their output at time t depends only
on the past, and the past is summarized by a bounded state:

* the left tails of the convolutions;
* the overlap tails of the transposed convolutions;
* the K/V of a sliding attention window;
* the position counter.

Carrying that state makes streaming cost the same as batch and makes the output
equal to the batch output.

The encoder side needs the same thing:

* **Full duplex (Moshi style)**: the user's microphone is encoded continuously
  by a causal Mimi encoder and fed to the LLM frame by frame while the model
  speaks. Re-encoding a growing buffer, or windowing it, wastes the same compute
  as on the decode side.
* **Streaming speech-to-LLM input** (LFM2-Audio and Mimi-based audio LMs): the
  same pattern without the reply audio.
* **Reference voices (Qwen3-TTS ICL)**: the reference clip's codes condition
  the decoder. A decoder state computed once from those codes and then restored
  per request is exactly what upstream llama.cpp keeps in `mtmd_gen_state`
  (`mtmd_gen_state_set_ref`). It also removes the 2–2.8 dB stream-vs-batch gap
  measured in #6, which comes from the streamer never seeing the reference
  codes.

Non-causal encoders and decoders (Whisper-style encoders, DAC, SNAC,
WavTokenizer) carry no state. For them the overlap window stays an internal
implementation detail behind the same API (§5) and never appears in it.

## 3. API (header diff sketch)

All additions go in `include/codec.h`. Nothing changes in `common/*.h` or
`include/codec_audio_lm.h`.

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
+// How a model implements a part's memory (codec_model_memory_kind).
+enum codec_memory_kind {
+    CODEC_MEMORY_KIND_NONE   = 0,  // not supported: enabling it fails
+    CODEC_MEMORY_KIND_STATE  = 1,  // carried state; output == batch output
+    CODEC_MEMORY_KIND_WINDOW = 2,  // internal overlap window; == batch when the
+                                   // window covers the model's reach, else approximate
+};
+
+typedef struct codec_memory * codec_memory_t;
+
 struct codec_context_params {
     int32_t seed;
+    // Parts whose state persists across codec_encode / codec_decode calls.
+    // Default CODEC_MEMORY_NONE: every call is independent, as today.
+    enum codec_memory_type memory;
 };
@@
+enum codec_memory_kind codec_model_memory_kind(const struct codec_model * model, enum codec_memory_type part);
+
+// NULL when the context was created with memory == CODEC_MEMORY_NONE.
+codec_memory_t codec_get_memory(const struct codec_context * ctx);
+
+// Back to the start of a stream (silence history, position 0) for the parts
+// in `which`. The other part is untouched.
+void codec_memory_clear(codec_memory_t mem, enum codec_memory_type which);
+
+// Serialized state of the parts in `which` (§7). get_data returns the bytes
+// written (0 = dst too small); set_data returns the bytes read, or 0 when the
+// blob does not fit this model / part, leaving the memory unchanged.
+size_t codec_memory_state_get_size(codec_memory_t mem, enum codec_memory_type which);
+size_t codec_memory_state_get_data(codec_memory_t mem, enum codec_memory_type which, uint8_t * dst, size_t size);
+size_t codec_memory_state_set_data(codec_memory_t mem, enum codec_memory_type which, const uint8_t * src, size_t size);
+
+// Stream position of a part: frames emitted by the encoder, or frames consumed
+// by the decoder since the last clear.
+int64_t codec_memory_n_frames(codec_memory_t mem, enum codec_memory_type part);
```

The `codec_encode` / `codec_decode` signatures are unchanged.
`python/codec_cpp/_ffi.py` mirrors the struct field and the new functions in the
same commit.

**Mapping to llama.cpp**

| llama.cpp | codec.cpp |
|---|---|
| `llama_context_params` (KV types, `n_ctx`) | `codec_context_params.memory` |
| `llama_get_memory(ctx)` | `codec_get_memory(ctx)` |
| `llama_memory_clear(mem, data)` | `codec_memory_clear(mem, which)` |
| `llama_state_seq_get_size / get_data / set_data` (on ctx, per seq) | `codec_memory_state_*` (on mem, per part) |
| `llama_memory_seq_rm / cp / keep / add` | none (one sequence; see §9) |

Unlike llama, the state functions take the memory handle. The part selector
plays the role llama's `seq_id` plays there.

**ABI.** `codec_context_params` grows from 4 to 8 bytes and is passed by value
to `codec_init_from_model`, so binaries built against the old header must be
rebuilt. Hosts that start from `codec_context_default_params()` (all in-tree
code, llama.rn) need no source change. Everything else is additive.

## 4. Semantics

### 4.1 Encode with memory (`ENCODE` on)

* `codec_encode(ctx, audio, out, p)` appends `audio` to the stream and returns
  the frames that are now final, possibly zero (`out->n_frames == 0` is
  success). Leftover samples, fewer than one hop, stay in the memory.
* A frame is final when all of its input is in:
  * **STATE kind**: as soon as its hop of samples has arrived. Causal, no added
    latency beyond one hop.
  * **WINDOW kind**: once R samples of right context have also arrived.
* **Flush**: `audio->n_samples == 0` pads the tail the way batch encode pads a
  clip's end and returns the remaining frames.
  * After a flush, the part is *ended*. A non-empty call returns
    `CODEC_STATUS_INVALID_STATE` until `codec_memory_clear`.
* **Invariant**: concatenating every call's frames, flush included, gives the
  codes `codec_encode` gives in one call on the concatenated audio:
  * STATE kind: identical codes, except that a rounding difference at a
    quantizer decision boundary may flip a code (§8, tolerance);
  * WINDOW kind: the same, when the window covers the reach.
* `audio->sample_rate` must equal the model's encode rate. Resampling in a
  stream needs its own state; that is open question Q3.
* `codec_encode_latent` shares the encode part.

### 4.2 Decode with memory (`DECODE` on)

* `codec_decode(ctx, tokens, out, p)` appends the frames and returns the PCM
  that is now final:
  * **STATE kind**: all of it;
  * **WINDOW kind**: everything except the last R frames, which are held back.
  * The output can be empty, for example when the frames fit inside R.
* **Flush**: `tokens->n_frames == 0` returns the held-back tail (a no-op for
  STATE kind) and ends the part, as for encode.
* **Invariant**: the concatenated PCM equals the batch decode of the
  concatenated codes. That holds for STATE kind and for WINDOW kind with full
  reach. Otherwise the gap is bounded by the per-model numbers in §5.
* Joins between WINDOW chunks crossfade internally, as #6 does today. The host
  never sees a seam.
* Parameters fixed for the life of a stream: `n_q`, `speaker`, and the layout
  of the token buffer. Changing one before a clear returns
  `CODEC_STATUS_INVALID_ARG`.
* `codec_decode_quantized_representation` shares the decode part.
* `codec_decode_batch` on a context with DECODE memory returns
  `CODEC_STATUS_NOT_SUPPORTED`. A batch holds several sequences, and memory
  holds one.

### 4.3 Lifecycle

* Memory is allocated by `codec_init_from_model`:
  * device buffers sized from the model's metadata (window lengths, channels);
  * freed with the context;
  * zeroed state, position 0.
* If a part's kind is NONE, `codec_init_from_model` fails and
  `codec_get_last_error` names the part.
* `codec_memory_clear(mem, which)` is cheap: a buffer memset and counters reset.
* With `memory == NONE`, behaviour and numerics are byte-identical to today.

### 4.4 Graphs

The rule *one model = one graph per public call* still holds:

* the state tensors are inputs of the call's graph;
* the updated state is written back in the same graph (`ggml_cpy` into the
  memory buffer), as llama.cpp writes its KV cache;
* ring-buffer offsets and positions are small input tensors, so a graph is
  reused for every chunk of the same length.

The graph cache key gains the memory flag and the chunk length. Hosts get the
best cache hit rate with a fixed chunk size.

## 5. Per codec family

Reach below is from an impulse test on the host: one frame's codes changed, and
the change in output measured before ("look-ahead") and after ("memory") that
frame, at −40 dB.

| Family (archs) | Encoder | Decoder | State per part |
|---|---|---|---|
| **Mimi** (`MIMI`; LFM2-Audio, Moshi; `POCKET_MIMI` for the decoder) | causal SEANet convs, stride-2 downsample, causal transformer with a 250-frame sliding window: **STATE** | mirror image, measured causal, −40 dB memory 0.24 s (Moshi's Mimi 2.0 s): **STATE** | conv / convtr tails (< 1 MB); K/V 8 layers × 250 frames × 512 × 2 ≈ 8 MB F32 (4 MB F16) per part; position; pending samples < 1920 |
| **Qwen3-TTS 12 Hz tokenizer** (`QWEN3_TTS_TOKENIZER`) | Mimi encoder (`qwen3.encoder.*`): **STATE**, same as Mimi | causal convs + 8 transformer layers, sliding window 72 frames, then causal upsampling convtr; −40 dB memory 5.8–7.7 s, reach to the end of the clip: **STATE** | K/V 8 × 71 frames; conv and convtr tails; position. The same contents as upstream llama.cpp's `mtmd_gen_state` |
| **MOSS-Audio** (`MOSS_AUDIO`; Nano, full, TTS-Realtime) | patched pretransform + causal sliding-window transformers per stage: **STATE** | mirror image; −40 dB memory 17–24 s: **STATE** | K/V per stage, window = `context_duration × stage frame rate`, so the largest state of the set; pending samples to a whole patch; Nano needs 48 kHz stereo |
| **DAC** (`DAC`) | non-causal convs, finite receptive field: **WINDOW**, exact | non-causal; reach ±0.13–0.20 s (10 frames): **WINDOW** L = R = 0.32 s, **exact** (#6 measured 0.00 dB vs batch) | the last L codes / samples and R held back; no tensors |
| **SNAC** (`SNAC`) | non-causal convs: **WINDOW**, exact | reach ±0.19 s (9 frames): **WINDOW**, exact | as DAC; three code rates, so the window is counted at the coarsest rate |
| **WavTokenizer** (`WAVTOKENIZER_LARGE`) | non-causal convs with reflect padding: **WINDOW**, exact | Vocos backbone with global attention, reach is the whole clip: **WINDOW** L 1.28 / R 0.32 s, **approximate** (1.8 dB vs batch on a 4.8 dB codec error; equal WER) | last L codes, R held back |
| NeMo nano codec | (not a streaming target) | causal conv, memory 2.2 s: **STATE** | conv tails only |
| XCodec2, NeuCodec, Chatterbox S3G, latent decoders (Soprano, Pocket-TTS, BlueMagpie), CosyVoice, XY-Tokenizer | | global attention, flow, or not measured: **NONE** for now | the streamer keeps decoding from frame 0 |

Notes:

* **Whisper-style encoders** (S3 tokenizers, XY-Tokenizer) are non-causal with
  a fixed input window. They get WINDOW kind if someone needs them streamed. The
  overlap stays internal, as Hans asked.
* **Qwen3-TTS reference voice**:
  1. decode the reference codes on a context with DECODE memory;
  2. discard the PCM;
  3. call `codec_memory_state_get_data(mem, DECODE)`;
  4. per request, call `set_data` and decode the new codes.

  The output then matches the batch decode of `ref + new` with the reference
  audio trimmed off, which is the official model's ICL path. Encoding the
  reference clip itself is a one-shot `codec_encode`. ENCODE memory matters there
  only when the clip itself arrives as a stream.
* **Full duplex**: one context with ENCODE memory on the microphone thread and
  one with DECODE memory on the playback thread (§8).

## 6. Relation to #6 (windowed stateless streamer)

#6 merges first, on its own terms: it changes no public header beyond a comment
and needs nothing from this RFC. Afterwards:

1. **Phase 1, STATE kind** for Mimi, the Qwen3 tokenizer and MOSS. In
   `common/audio_lm_c.cpp`, `alm_streamer` opens its codec context with DECODE
   memory when `codec_model_memory_kind(model, DECODE) == STATE` and just
   forwards new frames. Otherwise it keeps #6's window. Measurable win: Qwen3
   streaming on the phone goes from RTF 6.2 to the batch RTF of about 1.5.
2. **Phase 2, WINDOW kind inside the library.** #6's internal
   `src/audio_lm_window.h` table and crossfade move into the codec memory as its
   WINDOW implementation. `alm_streamer` is then a single path (feed frames,
   flush at the end), and #6's host-side window code is deleted.
3. Decoding from frame 0 survives only for NONE-kind codecs.

So #6 becomes the internal fallback, first in the streamer and then inside the
library, and is not exposed. Its smoke (`tests/e2e/audio_lm_stream_smoke.py`)
becomes the regression test for both kinds.

## 7. State serialization

The blob is little-endian and self-describing:

```
header   char[4]  "CDCM"
         u32      format_version        (1)
         u32      arch                  (enum codec_arch)
         u8[32]   model_fingerprint     (SHA-256 over tensor names, types, shapes and the
                                         decode-relevant GGUF keys; computed once at load)
         u32      parts                 (codec_memory_type bits present)
per part u32      part                  (ENCODE / DECODE)
         u32      kind                  (STATE / WINDOW)
         u32      impl_version          (per-arch layout version)
         i64      n_frames, i64 n_samples_pending
         i32      n_q, i32 flags        (ended, ...)
         u32      n_sections
section  u32 tag, u32 ggml_type, i64 ne[4], u64 nbytes, u8 data[nbytes]   (8-byte aligned)
```

* Section tags name the tensors: `conv_tail[i]`, `k[l]`, `v[l]`, `ring_head`,
  `pending_pcm`, `window_codes`.
* Data is copied with `ggml_backend_tensor_get`, so a blob saved on the GPU
  restores on the CPU and the other way round. It is tied to the model file
  through the fingerprint, not to the backend.
* `set_data` validates everything (magic, versions, arch, fingerprint, part,
  section shapes) before touching the memory. Any mismatch returns 0 and leaves
  the state unchanged.
* **Versioning**: no migration. State is a cache that can always be recomputed.
  `format_version` changes with the header and `impl_version` with an arch's
  layout. The old blob is rejected, and hosts recompute it (Qwen3: re-decode
  the reference codes once).
* Size: Qwen3 decode state is about 8 × 71 × 1024 × 2 × 4 B ≈ 4.6 MB F32, plus
  tails. That is small enough to keep one per saved voice.

## 8. Thread safety and numerics

* Thread rules are unchanged: one `codec_context` is used by one thread at a
  time, and its memory is part of it.
* Calling `state_get_data` while another thread encodes on the same context is
  a data race.
* Contexts of one model are independent, since each has its own backend
  instance. Full duplex therefore uses two contexts: encode on one thread,
  decode on another.
* One context with `BOTH` is allowed, but its encode and decode calls are
  serialized because they share the scheduler and eval arena.
* The memory handle is valid until `codec_free`. It is not reference counted.
* **Numerics.** Chunked compute changes matmul tiling, so STATE kind matches
  batch to rounding, not bit for bit. Smokes assert:
  * PCM: max |Δ| ≤ 1e-4;
  * codes: at most 0.1 % flipped, and only at quantizer ties;
  * WINDOW kind: the per-model LSD limits of #6.
* State tensors keep the precision the batch graph computes them at (F32 K/V
  unless the graph already uses F16).

## 9. Open questions

* **Q1. Flush ends the stream**, rather than letting it continue after a
  flush. Proposed: ends, which is simpler and matches the "end of utterance"
  use. The alternative needs a "pad but keep going" mode that no family needs
  today.
* **Q2. `codec_model_memory_kind` and `codec_memory_n_frames`** are beyond the
  list Hans gave. Without the first, a host can only learn the kind by failing
  `codec_init_from_model`. The second gives frame alignment for full duplex.
  Keep both?
* **Q3. Resampling.** Proposed: streamed input must be at the model's rate in
  v1. An in-graph streaming resampler with its own tail state comes later.
* **Q4. Voice files.** Should a Qwen3 reference decode state be cached in the
  `codec_speaker` / voice file (`codec_audio_lm_save_voice`)? That would touch
  `include/codec_speaker.h`, and only by a separate proposal.
* **Q5. Sequence ids.** Proposed: none. One context per sequence, which is how
  llama.rn uses contexts. Multi-sequence memory, as with `llama_memory_seq_*`,
  would only matter for a server batching many streams through one decoder.
* **Q6. Approximate WINDOW kinds.** Should WavTokenizer stay WINDOW, which is
  approximate, or report NONE unless the host opts into approximation? A
  `codec_context_params` flag such as `allow_approx_memory` would make the
  choice explicit.
* **Q7. Chunk-length graph cache.** Should the library pad short chunks up to
  a set of bucket sizes, to cap the number of cached graphs when a host sends
  irregular chunks?

## 10. Test plan (per kind)

* **Stream vs batch** on random chunk sizes, including 1-frame and sub-hop
  chunks: Mimi encode and decode, Qwen3 decode, MOSS decode, DAC, WavTokenizer.
* **`set_data(get_data())` round trip**: the continuation is identical to an
  uninterrupted stream. A blob from another model or part is rejected and the
  state is unchanged.
* **`clear`** equals a fresh context, and the other part is untouched.
* **Phone benchmark** (Pixel 7 Pro): streaming RTF vs batch for Qwen3 and Mimi
  with STATE kind. Target: within 10 % of batch at 0.5 s chunks.
