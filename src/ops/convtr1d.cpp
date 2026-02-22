#include "convtr1d.h"

#include "ggml_ops.h"

#include <algorithm>

ggml_tensor * codec_convtr1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_convtr1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    ggml_tensor * y = codec_convtr1d(ctx, x, w, b, stride, 0, dilation);
    if (y == nullptr) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t crop_right = std::max(0, kernel - stride);
    return codec_op_crop_1d(ctx, y, 0, crop_right);
}
