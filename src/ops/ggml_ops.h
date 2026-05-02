#ifndef CODEC_OPS_GGML_OPS_H
#define CODEC_OPS_GGML_OPS_H

#include "../codec_internal.h"

enum codec_unary_op {
    CODEC_UNARY_SIGMOID = 0,
    CODEC_UNARY_ELU = 1,
    CODEC_UNARY_SILU = 2,
    CODEC_UNARY_GELU_ERF = 3,
};

ggml_tensor * codec_op_unary(ggml_context * ctx, ggml_tensor * x, codec_unary_op op);
ggml_tensor * codec_op_layer_norm(ggml_context * ctx, ggml_tensor * x, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_layer_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_layer_norm_tc(ggml_context * ctx, ggml_tensor * x_tc, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_rms_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma);
ggml_tensor * codec_op_group_norm(ggml_context * ctx, ggml_tensor * x, int32_t n_groups, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * codec_op_linear_tc(ggml_context * ctx, ggml_tensor * x_tc, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * codec_op_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps);
ggml_tensor * codec_op_snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * inv_beta, float eps);
ggml_tensor * codec_op_pad_1d(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right);
ggml_tensor * codec_op_pad_1d_replicate(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right);
ggml_tensor * codec_op_causal_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t target_t);
ggml_tensor * codec_op_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t crop_left, int32_t crop_right);
ggml_tensor * codec_op_channel_scale(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale);

ggml_tensor * codec_op_tokens_to_features(ggml_context * ctx, ggml_tensor * tokens, int32_t out_channels);

// ConvNeXt block (Vocos-style): residual + (depthwise conv → LayerNorm → linear → GELU → linear → channel-scale).
// `x_ct` is `[c, t]`; biases and `gamma` are optional (pass nullptr to skip).
// `dw_padding` is the symmetric (non-causal) padding for the depthwise conv;
// for causal variants compose your own block.
ggml_tensor * codec_op_convnext_block_ct(
    ggml_context * ctx,
    ggml_tensor * x_ct,
    ggml_tensor * dw_w,
    ggml_tensor * dw_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b,
    ggml_tensor * pw1_w,
    ggml_tensor * pw1_b,
    ggml_tensor * pw2_w,
    ggml_tensor * pw2_b,
    ggml_tensor * gamma,
    int32_t dw_padding);

#endif
