#include "lm_attn.h"

#include <cmath>

ggml_tensor * codec_op_lm_attn_ctx_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const codec_lm_attn_params * params) {

    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr) {
        return nullptr;
    }
    if (q_dth->ne[0] != k_dth->ne[0] || q_dth->ne[0] != v_dth->ne[0] ||
        q_dth->ne[1] != k_dth->ne[1] || q_dth->ne[1] != v_dth->ne[1] ||
        q_dth->ne[2] != k_dth->ne[2] || q_dth->ne[2] != v_dth->ne[2]) {
        return nullptr;
    }

    const int32_t head_dim = (int32_t) q_dth->ne[0];
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max(1, head_dim)));
    const bool causal = params != nullptr && params->causal;

    ggml_tensor * k_cont = ggml_cont(ctx, k_dth);
    ggml_tensor * attn_scores = ggml_mul_mat(ctx, k_cont, q_dth); // [t, t, h]
    if (attn_scores == nullptr) {
        return nullptr;
    }

    attn_scores = ggml_scale_inplace(ctx, attn_scores, scale);
    if (causal) {
        attn_scores = ggml_diag_mask_inf_inplace(ctx, attn_scores, 0);
    }

    ggml_tensor * attn_probs = ggml_soft_max(ctx, attn_scores);
    if (attn_probs == nullptr) {
        return nullptr;
    }

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_tdh, attn_probs); // [d, t, h]
}

