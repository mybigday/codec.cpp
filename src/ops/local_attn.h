#ifndef CODEC_OPS_LOCAL_ATTN_H
#define CODEC_OPS_LOCAL_ATTN_H

#include <ggml.h>

struct codec_local_attn_params {
    const float * bias = nullptr; // [heads, max_dist]
    int32_t heads = 0;
    int32_t head_dim = 0;
    int32_t window = 0;
    int32_t max_dist = 0;
};

ggml_tensor * codec_op_local_attn(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const codec_local_attn_params * params);

#endif
