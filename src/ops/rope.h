#ifndef CODEC_OPS_ROPE_H
#define CODEC_OPS_ROPE_H

#include "../codec_internal.h"

ggml_tensor * codec_op_rope(
    ggml_context * ctx,
    ggml_tensor * x_dth,
    int32_t n_dims,
    float freq_base,
    float freq_scale);

#endif
