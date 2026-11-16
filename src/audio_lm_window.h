#ifndef CODEC_AUDIO_LM_WINDOW_H
#define CODEC_AUDIO_LM_WINDOW_H

// Internal to libcodec (the audio_lm objects' private include dir; not installed, not part of include/ or
// common/*.h).

#include "codec_common.h"

namespace codec_common {

// The stateless streaming window of the context's codec, in codec frames: each poll decodes
// [committed - left, avail), holds back `right` (the frames the decoder may still revise), and cross-fades the joins.
// Read from the decoder's look_back / look_ahead (GGUF codec.decoder.look_*, else the per-arch defaults). false: an
// unbounded look_back or look_ahead; the streamer decodes from the generation's first frame (exact, quadratic).
// Causal decoders with memory do not use it (src/audio_lm_stream.h).
bool audio_lm_stream_window(const audio_lm_context * ctx, int32_t * left, int32_t * right);

}  // namespace codec_common

#endif
