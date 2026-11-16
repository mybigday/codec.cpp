#ifndef CODEC_LOOK_H
#define CODEC_LOOK_H

// Internal to libcodec: the model's decoder look_back / look_ahead in codec frames (GGUF
// codec.decoder.look_back / look_ahead, else the per-arch defaults; -1 = unbounded).

#include <stdint.h>

struct codec_model;

void codec_model_decoder_look(const struct codec_model * model, int32_t * look_back, int32_t * look_ahead);

#endif
