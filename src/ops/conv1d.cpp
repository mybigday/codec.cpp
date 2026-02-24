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

    if (w->ne[0] < stride) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * w_conv = w->type == GGML_TYPE_F16 ? w : ggml_cast(ctx, w, GGML_TYPE_F16);
    ggml_tensor * y = ggml_conv_1d(ctx, w_conv, x_pad, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_conv1d_depthwise_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0) {
        return nullptr;
    }

    if (w->ne[0] < stride) {
        return nullptr;
    }

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * w_conv = w->type == GGML_TYPE_F16 ? w : ggml_cast(ctx, w, GGML_TYPE_F16);
    ggml_tensor * y = ggml_conv_1d_dw(ctx, w_conv, x_pad, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return ggml_cont(ctx, y);
}
