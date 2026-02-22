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
ggml_tensor * codec_op_group_norm(ggml_context * ctx, ggml_tensor * x, int32_t n_groups, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * codec_op_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps);
ggml_tensor * codec_op_pad_1d(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right);
ggml_tensor * codec_op_causal_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t target_t);
ggml_tensor * codec_op_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t crop_left, int32_t crop_right);
ggml_tensor * codec_op_channel_scale(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale);

ggml_tensor * codec_op_tokens_to_features(ggml_context * ctx, ggml_tensor * tokens, int32_t out_channels);

#endif
