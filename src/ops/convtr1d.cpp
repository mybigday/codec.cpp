#include "convtr1d.h"

#include "ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cstdio>
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

// ConvTranspose1d as one GEMM plus ceil(K/stride) shifted adds (dilation 1).
// ggml's CPU CONV_TRANSPOSE_1D runs one dot product per output sample with no
// blocking; this is the same arithmetic (bit-level equal up to summation order,
// at the kernel's stored precision) at GEMM speed.
//   cols[k + K*co, t] = sum_ci w[k, co, ci] * x[t, ci]
//   y[t*s + k, co]   += cols[k + K*co, t];  with k = j*s + p, phase block j
//   lands j frames later.
static ggml_tensor * codec_convtr1d_gemm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int32_t stride) {
    const int64_t K0 = w->ne[0];
    // A kernel that is not a multiple of the stride is treated as one with zero
    // taps appended: its last phase block is padded (on the F32 GEMM output, so
    // the kernel itself stays untouched); they only add samples past the true
    // output end, cropped below.
    const int64_t K = (K0 + stride - 1) / stride * stride;
    const int64_t C_out = w->ne[1], C_in = w->ne[2], T = x->ne[0];
    const int64_t J = K / stride;
    ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, w, K0 * C_out, C_in)));  // [C_in, K0*C_out]
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, T, C_in)));           // [C_in, T]
    ggml_tensor * cols = ggml_mul_mat(ctx, wt, xt);                                                     // [K0*C_out, T]
    ggml_tensor * y = nullptr;
    for (int64_t j = 0; j < J; ++j) {
        const int64_t len = std::min<int64_t>(stride, K0 - j * stride);
        ggml_tensor * blk = ggml_view_3d(ctx, cols, len, C_out, T, cols->nb[0] * K0, cols->nb[1],
                                         cols->nb[0] * j * stride);                                    // [len, C_out, T]
        blk = ggml_cont(ctx, ggml_permute(ctx, blk, 0, 2, 1, 3));                                      // [len, T, C_out]
        if (len < stride) blk = ggml_pad(ctx, blk, (int) (stride - len), 0, 0, 0);                     // [s, T, C_out]
        blk = codec_convtr1d_pad_rows(ctx, blk, j, J - 1 - j);                                         // [s, T+J-1, C_out]
        y = y == nullptr ? blk : ggml_add(ctx, y, blk);
    }
    y = ggml_reshape_2d(ctx, y, stride * (T + J - 1), C_out);
    return K == K0 ? y : codec_op_crop_1d(ctx, y, 0, (int32_t) (K - K0));
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

    w = codec_graph_stored_weight(ctx, w);
    b = codec_graph_cast_f32(ctx, b);

    // The kernel runs at its stored type.  The GEMM path transposes it, which a
    // block-quantized tensor (rows fixed along K) cannot do, and ggml's
    // CONV_TRANSPOSE_1D takes F32/F16 kernels only.
    const bool gemm = dilation == 1 && x->ne[1] == w->ne[2] && x->ne[2] <= 1 && x->ne[3] <= 1;
    const bool float_w = w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_BF16;
    if (!float_w || (!gemm && w->type == GGML_TYPE_BF16)) {
        std::fprintf(stderr, "codec_convtr1d: kernel '%s' of type %s is not supported%s\n", w->name,
                     ggml_type_name(w->type), gemm ? "" : " (dilated/batched ConvTranspose1d needs F32/F16)");
        return nullptr;
    }
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
