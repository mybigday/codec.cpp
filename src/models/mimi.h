#ifndef CODEC_MODEL_MIMI_H
#define CODEC_MODEL_MIMI_H

#include "../codec_internal.h"

enum codec_status codec_mimi_init(struct codec_model * model);

enum codec_status codec_mimi_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

enum codec_status codec_mimi_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params);

#endif
