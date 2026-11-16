#ifndef CODEC_AUDIO_LM_STREAM_H
#define CODEC_AUDIO_LM_STREAM_H

// Internal to libcodec (the audio_lm objects' private include dir; not installed, not part of include/ or
// common/*.h).
//
// Streaming decode with codec memory (docs/proposals/codec_memory.md): for a causal decoder, each
// codec_audio_lm_stream_poll decodes only the new frames on a second codec context whose decoder memory carries
// the decoder's state. An in-context reference voice is decoded into the memory first, once per voice (the state
// after it is kept and restored for the next generation).

#include "codec_common.h"

namespace codec_common {

// Starts a stream for the current generation: clears the stream context's decoder memory, or restores the state
// after the current voice's in-context reference. 1: started; 0: the codec has no memory streaming (a non-causal
// decoder, no state to carry, continuous latents); -1: an error (last_error set).
int32_t audio_lm_stream_begin(audio_lm_context * ctx);
// Decodes accumulated frames [frame_begin, frame_end) on the stream context; frame_begin must be where the
// previous call ended (0 after begin).
bool audio_lm_stream_decode(audio_lm_context * ctx, int32_t frame_begin, int32_t frame_end, audio_lm_audio_output * out);

}  // namespace codec_common

#endif
