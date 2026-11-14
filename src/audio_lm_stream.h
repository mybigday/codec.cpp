#ifndef CODEC_AUDIO_LM_STREAM_H
#define CODEC_AUDIO_LM_STREAM_H

// Internal to libcodec (the audio_lm layer's own .cpp files include it; hosts
// never do, and nothing here is part of include/ or common/*.h).
//
// Exact incremental decode for codecs whose decoder is strictly causal
// (src/codec_stream.h): codec_audio_lm_stream_poll decodes only new frames
// through a carried decoder state, and an in-context reference voice is decoded
// once per voice and reused. Codecs without it keep the window path.

#include "codec_common.h"

namespace codec_common {

struct audio_lm_stream;

// nullptr when the codec has no incremental decode. Starts from the decoder
// state after the current voice's in-context reference, if it has one (cached
// per voice in the context).
audio_lm_stream * audio_lm_stream_begin(audio_lm_context * ctx);
// Decodes the accumulated frames from where the stream stands up to frame_end.
bool audio_lm_stream_next(audio_lm_context * ctx, audio_lm_stream * s, int32_t frame_end, audio_lm_audio_output * out);
void audio_lm_stream_end(audio_lm_stream * s);

// Decodes codec codes (n_frames x width) that follow `prefix` (n_prefix x
// width, its audio dropped), from the cached state after the prefix. false:
// no incremental decode for this codec (the caller decodes prefix + codes and
// cuts), or an error (last_error set).
bool audio_lm_decode_after_prefix(audio_lm_context * ctx, const int32_t * prefix, int32_t n_prefix, const int32_t * codes,
                                  int32_t n_frames, int32_t width, audio_lm_audio_output * out);

}  // namespace codec_common

#endif
