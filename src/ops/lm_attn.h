#pragma once

#include "../codec_internal.h"

struct codec_lm_attn_params {
    float scale = 0.0f;   // if <= 0, use 1 / sqrt(head_dim)
    bool causal = false;
};

// q_dth, k_dth, v_dth are [head_dim, t, n_heads]
// returns context tensor [head_dim, t, n_heads]
ggml_tensor * codec_op_lm_attn_ctx_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const codec_lm_attn_params * params);

