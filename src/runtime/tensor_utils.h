#ifndef CODEC_RUNTIME_TENSOR_UTILS_H
#define CODEC_RUNTIME_TENSOR_UTILS_H

#include "../codec_internal.h"

bool codec_runtime_write_tensor(ggml_tensor * t, const void * data, size_t n_bytes, std::string * error);
bool codec_runtime_read_tensor(ggml_tensor * t, void * data, size_t n_bytes, std::string * error);
bool codec_runtime_read_tensor_i32_2d_tq(ggml_tensor * t, std::vector<int32_t> * out, std::string * error);
bool codec_runtime_copy_tensor_f32_exact_from(ggml_tensor * src, const std::string & src_name, ggml_tensor * dst, std::string * error);
bool codec_runtime_copy_tensor_f32_exact(codec_context * ctx, const std::string & src_name, ggml_tensor * dst, std::string * error);
bool codec_runtime_copy_tensor_f32_exact_or_zeros(codec_context * ctx, const std::string & src_name, ggml_tensor * dst, std::string * error);

#endif
