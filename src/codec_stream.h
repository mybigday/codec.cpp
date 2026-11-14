#ifndef CODEC_STREAM_H
#define CODEC_STREAM_H

// Internal (not part of the public C API): exact incremental decode.
//
// Strictly causal decoders (no lookahead anywhere) can decode a generation
// piece by piece: the stream carries the decoder's state between pushes, each
// push decodes only its new frames, and the concatenated output equals one
// decode of all frames (up to float summation order). Used by the audio_lm
// streamer and to start generations from a cached reference prefix.

#include "codec.h"

#include <cstdint>

struct codec_decode_stream;

// nullptr when the context's model has no incremental decode (or on error).
struct codec_decode_stream * codec_decode_stream_new(struct codec_context * ctx, int32_t n_q);
// An independent copy (same context), e.g. a cached state after a reference prefix.
struct codec_decode_stream * codec_decode_stream_clone(const struct codec_decode_stream * stream);
void codec_decode_stream_free(struct codec_decode_stream * stream);
// Decodes tokens->n_frames new frames (T x Q interleaved) into out_pcm (n_frames * hop samples).
enum codec_status codec_decode_stream_push(
    struct codec_decode_stream * stream,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm);
// Frames decoded so far.
int64_t codec_decode_stream_n_frames(const struct codec_decode_stream * stream);
bool codec_model_has_decode_stream(const struct codec_model * model);

#endif
