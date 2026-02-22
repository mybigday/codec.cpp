#include "conv1d.h"

#include "ggml_ops.h"

ggml_tensor * codec_conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    ggml_tensor * w_conv = w->type == GGML_TYPE_F16 ? w : ggml_cast(ctx, w, GGML_TYPE_F16);
    ggml_tensor * y = ggml_conv_1d(ctx, w_conv, x, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_conv1d_depthwise(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    ggml_tensor * w_conv = w->type == GGML_TYPE_F16 ? w : ggml_cast(ctx, w, GGML_TYPE_F16);
    ggml_tensor * y = ggml_conv_1d_dw(ctx, w_conv, x, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_conv1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    if (dilation == 1) {
        if (w->ne[0] < stride) {
            return nullptr;
        }

        const int32_t pad_left = (int32_t) w->ne[0] - stride;
        ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, 0);
        if (x_pad == nullptr) {
            return nullptr;
        }
        ggml_tensor * w_conv = w->type == GGML_TYPE_F16 ? w : ggml_cast(ctx, w, GGML_TYPE_F16);
        ggml_tensor * y = ggml_conv_1d(ctx, w_conv, x_pad, stride, 0, 1);
        if (b != nullptr) {
            ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
            y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
        }
        return ggml_cont(ctx, y);
    }

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t pad = (kernel - 1) * dilation;
    ggml_tensor * y = codec_conv1d(ctx, x, w, b, stride, dilation, pad);
    if (y == nullptr) {
        return nullptr;
    }

    const int32_t target_t = ((int32_t) x->ne[0] + stride - 1) / stride;
    return codec_op_causal_crop_1d(ctx, y, target_t);
}
