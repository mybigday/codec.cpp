#ifndef CODEC_OPS_CONVTR1D_H
#define CODEC_OPS_CONVTR1D_H

#include "../codec_internal.h"

// ConvTranspose1d kernels are (K, C_out, C_in) or C_in-major (C_in, C_out, K), the
// layout block-quantized kernels need.  Kernel size of either, given C_in.
int64_t codec_convtr1d_kernel_size(const ggml_tensor * w, int64_t c_in);

ggml_tensor * codec_convtr1d(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t padding,
    int32_t dilation);

ggml_tensor * codec_convtr1d_causal(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation);

// Overlap a causal ConvTranspose1d carries to the next call: kernel - stride.
int32_t codec_convtr1d_causal_overlap(const ggml_tensor * w, int64_t c_in, int32_t stride);

// Causal ConvTranspose1d that adds `prev` ([overlap, C_out], the part of the
// previous call's last frames that falls on this call's first samples, before
// the bias) and returns T * stride samples. *next = the overlap this call
// leaves for the next one (before the bias).
ggml_tensor * codec_convtr1d_causal_ctx(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * prev,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    ggml_tensor ** next);

#endif
