#include "conv1d.h"

#include "ggml_ops.h"
#include "../runtime/tensor_utils.h"

// Conv weights are loaded directly from the GGUF context. Quantized weight
// types can't be reshaped (their row size is fixed by the block format), so
// cast to F32 before any reshape/im2col path. F16 weights are kept as F16 to
// preserve the fast im2col-F16 path that ggml_conv_1d uses.
static ggml_tensor * codec_conv1d_prepare_w(ggml_context * ctx, ggml_tensor * w) {
    if (w == nullptr) {
        return nullptr;
    }
    if (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16) {
        return w;
    }
    return ggml_cast(ctx, w, GGML_TYPE_F32);
}

static ggml_tensor * codec_conv1d_pointwise_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w) {

    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }
    if (x->ne[1] != w->ne[1] || w->ne[0] != 1) {
        return nullptr;
    }

    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x));       // [c_in, t]
    ggml_tensor * w_ic = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);  // [c_in, c_out]
    ggml_tensor * y_ct = ggml_mul_mat(ctx, w_ic, x_ct);                // [c_out, t]
    if (y_ct == nullptr) {
        return nullptr;
    }

    return ggml_cont(ctx, ggml_transpose(ctx, y_ct));                  // [t, c_out]
}

static ggml_tensor * codec_conv1d_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    if (w->ne[0] == 1 && stride == 1 && dilation == 1 && padding == 0) {
        return codec_conv1d_pointwise_impl(ctx, x, w);
    }

    const ggml_type im2col_type = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im2col = ggml_im2col(ctx, w, x, stride, 0, padding, 0, dilation, 0, false, im2col_type);
    if (im2col == nullptr) {
        return nullptr;
    }

    ggml_tensor * lhs = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    ggml_tensor * rhs = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor * y = codec_mul_mat(ctx, lhs, rhs);
    if (y == nullptr) {
        return nullptr;
    }

    return ggml_reshape_3d(ctx, y, im2col->ne[1], w->ne[2], im2col->ne[2]);
}

static ggml_tensor * codec_conv1d_depthwise_impl(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    int32_t stride,
    int32_t padding,
    int32_t dilation) {

    if (ctx == nullptr || x == nullptr || w == nullptr || stride <= 0 || dilation <= 0 || padding < 0) {
        return nullptr;
    }

    // Stride 1, unbatched: y = sum_k x[t + k*d] * w[k] as K broadcast multiply-adds
    // over shifted views — F32 throughout, no K-fold im2col copy.
    if (stride == 1 && x->ne[2] <= 1 && x->ne[3] <= 1 && w->ne[1] == 1 && w->ne[2] == x->ne[1]) {
        const int64_t K = w->ne[0], C = x->ne[1];
        ggml_tensor * xp = padding > 0 ? codec_op_pad_1d(ctx, x, padding, padding) : x;
        if (!ggml_is_contiguous(xp)) xp = ggml_cont(ctx, xp);
        const int64_t t_out = xp->ne[0] - (int64_t) dilation * (K - 1);
        if (t_out <= 0) return nullptr;
        ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, w), K, C)));  // [C, K]
        ggml_tensor * y = nullptr;
        for (int64_t k = 0; k < K; ++k) {
            ggml_tensor * xk = ggml_view_2d(ctx, xp, t_out, C, xp->nb[1], (size_t) (k * dilation) * xp->nb[0]);
            ggml_tensor * wk = ggml_view_2d(ctx, wt, 1, C, wt->nb[0], (size_t) k * wt->nb[1]);              // [1, C]
            ggml_tensor * term = ggml_mul(ctx, xk, wk);
            y = y == nullptr ? term : ggml_add(ctx, y, term);
        }
        return ggml_reshape_3d(ctx, y, t_out, C, 1);
    }

    ggml_tensor * x4 = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
    const ggml_type im2col_type = w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * im2col = ggml_im2col(ctx, w, x4, stride, 0, padding, 0, dilation, 0, false, im2col_type);
    if (im2col == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = codec_mul_mat(ctx, im2col, w);
    if (y == nullptr) {
        return nullptr;
    }

    return ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);
}

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

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    ggml_tensor * y = codec_conv1d_impl(ctx, x, w, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    y = ggml_cont(ctx, y);
    // Squeeze the trailing batch dim when input was 2D so callers don't
    // need to manually reshape after every non-pointwise conv (the im2col
    // path always returns ne=(t_out, c_out, 1)).
    if (x->ne[2] <= 1 && y->ne[2] == 1 && y->ne[3] == 1) {
        y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    }
    return y;
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

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    ggml_tensor * y = codec_conv1d_depthwise_impl(ctx, x, w, stride, padding, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
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

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = codec_conv1d_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return ggml_cont(ctx, y);
}

ggml_tensor * codec_conv1d_causal_replicate(
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

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d_replicate(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = codec_conv1d_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
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

    w = codec_conv1d_prepare_w(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    const int32_t kernel = (int32_t) w->ne[0];
    const int32_t kernel_eff = (kernel - 1) * dilation + 1;
    const int32_t pad_left = kernel_eff - stride;
    const int32_t t_in = (int32_t) x->ne[0];
    const int32_t extra_pad = t_in > 0 ? (((t_in + stride - 1) / stride) * stride - t_in) : 0;
    ggml_tensor * x_pad = codec_op_pad_1d(ctx, x, pad_left, extra_pad);
    if (x_pad == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = codec_conv1d_depthwise_impl(ctx, x_pad, w, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    return ggml_cont(ctx, y);
}
