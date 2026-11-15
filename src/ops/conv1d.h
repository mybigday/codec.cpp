#ifndef CODEC_OPS_CONV1D_H
#define CODEC_OPS_CONV1D_H

#include "../codec_internal.h"

// Conv kernels are (K, C_in, C_out) or flattened (K*C_in, 1, C_out) (same bytes;
// the layout block-quantized kernels need), at any stored type.  Kernel
// size of either form, given the input channel count (0 if they don't fit).
int64_t codec_conv1d_kernel_size(const ggml_tensor * w, int64_t c_in);

ggml_tensor * codec_conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding);

ggml_tensor * codec_conv1d_depthwise(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding);

ggml_tensor * codec_conv1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

ggml_tensor * codec_conv1d_causal_replicate(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

ggml_tensor * codec_conv1d_depthwise_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

// Left padding of a causal conv: (kernel - 1) * dilation + 1 - stride.
int32_t codec_conv1d_causal_pad_left(const ggml_tensor * w, int64_t c_in, int32_t stride, int32_t dilation, bool depthwise);

// Causal conv whose left padding is `prev` ([pad_left, C_in], the cached tail
// of the previous input) instead of zeros. The right padding to a stride
// multiple is the batch op's (zeros, or the last value when `replicate`).
// *x_cat = concat(prev, x), for the next tail.
ggml_tensor * codec_conv1d_causal_ctx(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * prev,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    bool depthwise,
    ggml_tensor ** x_cat,
    bool replicate = false);

#endif
