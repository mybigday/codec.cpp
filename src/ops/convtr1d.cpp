#include "convtr1d.h"

#include "ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <string>
#include <vector>

// Zero-pad ne[1] by `left` / `right` rows.  Metal's PAD only pads at the end, so
// the left side is a concatenated zero constant (cf. codec_op_pad_1d).
static ggml_tensor * codec_convtr1d_pad_rows(ggml_context * ctx, ggml_tensor * x, int64_t left, int64_t right) {
    if (left > 0) {
        const int64_t ne[4] = {x->ne[0], left, x->ne[2], x->ne[3]};
        const std::string name = "lut.zeros.f32." + std::to_string(ne[0]) + "x" + std::to_string(ne[1]) + "x" +
                                 std::to_string(ne[2]) + "x" + std::to_string(ne[3]);
        std::vector<float> zeros((size_t) (ne[0] * ne[1] * ne[2] * ne[3]), 0.0f);
        ggml_tensor * z = codec_graph_const(ctx, name.c_str(), GGML_TYPE_F32, 4, ne, zeros.data());
        x = z != nullptr ? ggml_concat(ctx, z, x, 1) : ggml_pad_ext(ctx, x, 0, 0, (int) left, 0, 0, 0, 0, 0);
    }
    return right > 0 ? ggml_pad(ctx, x, 0, (int) right, 0, 0) : x;
}

// ConvTranspose1d as one GEMM plus K/stride shifted adds (dilation 1, K a
// multiple of stride).  ggml's CPU CONV_TRANSPOSE_1D runs one dot product per
// output sample with no blocking; this is the same arithmetic in F32 (bit-level
// equal up to summation order) at GEMM speed.
//   cols[k + K*co, t] = sum_ci w[k, co, ci] * x[t, ci]
//   y[t*s + k, co]   += cols[k + K*co, t];  with k = j*s + p, phase block j
//   lands j frames later.
static ggml_tensor * codec_convtr1d_gemm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int32_t stride) {
    const int64_t K = w->ne[0], C_out = w->ne[1], C_in = w->ne[2], T = x->ne[0];
    const int64_t J = K / stride;
    w = codec_graph_cast_f32(ctx, w);
    ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, w, K * C_out, C_in)));  // [C_in, K*C_out]
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, T, C_in)));          // [C_in, T]
    ggml_tensor * cols = ggml_mul_mat(ctx, wt, xt);                                                    // [K*C_out, T]
    ggml_tensor * y = nullptr;
    for (int64_t j = 0; j < J; ++j) {
        ggml_tensor * blk = ggml_view_3d(ctx, cols, stride, C_out, T, cols->nb[0] * K, cols->nb[1],
                                         cols->nb[0] * j * stride);                                    // [s, C_out, T]
        blk = ggml_cont(ctx, ggml_permute(ctx, blk, 0, 2, 1, 3));                                      // [s, T, C_out]
        blk = codec_convtr1d_pad_rows(ctx, blk, j, J - 1 - j);                                         // [s, T+J-1, C_out]
        y = y == nullptr ? blk : ggml_add(ctx, y, blk);
    }
    return ggml_reshape_2d(ctx, y, stride * (T + J - 1), C_out);
}

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

    // ggml_conv_transpose_1d expects F32/F16 weights; quantized weights would
    // need an explicit dequant. Conv kernels are typically too small to be
    // eligible for Q8_0/K-quants, but cast defensively.
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) {
        w = ggml_cast(ctx, w, GGML_TYPE_F32);
    }
    b = codec_graph_cast_f32(ctx, b);

    const bool gemm = dilation == 1 && w->ne[0] % stride == 0 && x->ne[1] == w->ne[2] && x->ne[2] <= 1 && x->ne[3] <= 1;
    ggml_tensor * y = gemm ? codec_convtr1d_gemm(ctx, x, w, stride) : ggml_conv_transpose_1d(ctx, w, x, stride, 0, dilation);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, 1, y->ne[1]);
        y = ggml_add(ctx, y, b2);
    }
    if (padding > 0) {
        y = codec_op_crop_1d(ctx, y, padding, padding);
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
