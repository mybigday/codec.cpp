#include "rvq.h"

static void codec_rvq_argmin_rows_map_custom1(
    ggml_tensor * dst,
    const ggml_tensor * src,
    int ith,
    int nth,
    void * userdata) {

    (void) userdata;

    if (dst == nullptr || src == nullptr || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return;
    }
    if (src->ne[0] <= 0 || src->ne[1] <= 0) {
        return;
    }

    const int64_t cols = src->ne[1] * src->ne[2] * src->ne[3];
    const int64_t start = (cols * ith) / nth;
    const int64_t end = (cols * (ith + 1)) / nth;

    for (int64_t col = start; col < end; ++col) {
        int64_t rem = col;
        const int64_t i1 = rem % src->ne[1];
        rem /= src->ne[1];
        const int64_t i2 = rem % src->ne[2];
        rem /= src->ne[2];
        const int64_t i3 = rem;

        const size_t col_offset = (size_t) i1 * src->nb[1] + (size_t) i2 * src->nb[2] + (size_t) i3 * src->nb[3];
        const float * src_col = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src->data) + col_offset);
        float * dst_col = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + col_offset);

        float best = src_col[0];
        int32_t best_idx = 0;
        for (int32_t i = 1; i < (int32_t) src->ne[0]; ++i) {
            const float v = src_col[(size_t) i];
            if (v < best) {
                best = v;
                best_idx = i;
            }
        }

        dst_col[0] = (float) best_idx;
    }
}

ggml_tensor * codec_rvq_argmin_map_custom1(
    ggml_context * ctx_eval,
    ggml_tensor * distances_ct) {

    if (ctx_eval == nullptr || distances_ct == nullptr || distances_ct->type != GGML_TYPE_F32 || distances_ct->ne[0] <= 0 || distances_ct->ne[1] <= 0) {
        return nullptr;
    }

    ggml_tensor * dist = ggml_cont(ctx_eval, distances_ct);
    ggml_tensor * argmin_full = ggml_map_custom1(ctx_eval, dist, codec_rvq_argmin_rows_map_custom1, GGML_N_TASKS_MAX, nullptr);
    if (argmin_full == nullptr) {
        return nullptr;
    }

    ggml_tensor * argmin_row = ggml_view_2d(ctx_eval, argmin_full, 1, dist->ne[1], argmin_full->nb[1], 0);
    argmin_row = ggml_cont(ctx_eval, argmin_row);
    argmin_row = ggml_reshape_1d(ctx_eval, argmin_row, dist->ne[1]);
    return ggml_cast(ctx_eval, argmin_row, GGML_TYPE_I32);
}

bool codec_rvq_build_layer_ggml(
    ggml_context * ctx_eval,
    ggml_tensor * residual_ct,
    ggml_tensor * codebook_dc,
    codec_rvq_layer_result_ggml * out) {

    if (ctx_eval == nullptr || residual_ct == nullptr || codebook_dc == nullptr || out == nullptr) {
        return false;
    }
    if (residual_ct->type != GGML_TYPE_F32 || codebook_dc->type != GGML_TYPE_F32) {
        return false;
    }
    if (residual_ct->ne[0] <= 0 || residual_ct->ne[1] <= 0 || codebook_dc->ne[0] != residual_ct->ne[0] || codebook_dc->ne[1] <= 0) {
        return false;
    }

    // dist(c, t) = ||residual(:,t)||^2 + ||codebook(:,c)||^2 - 2 * dot(codebook(:,c), residual(:,t))
    ggml_tensor * residual_sq = ggml_sqr(ctx_eval, residual_ct);
    ggml_tensor * codebook_sq = ggml_sqr(ctx_eval, codebook_dc);
    ggml_tensor * residual_norm = ggml_sum_rows(ctx_eval, residual_sq); // [1, t]
    ggml_tensor * codebook_norm = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, ggml_sum_rows(ctx_eval, codebook_sq))); // [cbs, 1]

    ggml_tensor * dot = ggml_mul_mat(ctx_eval, codebook_dc, residual_ct); // [cbs, t]
    ggml_tensor * dist = ggml_add(
        ctx_eval,
        ggml_repeat(ctx_eval, codebook_norm, dot),
        ggml_repeat(ctx_eval, residual_norm, dot));
    dist = ggml_sub(ctx_eval, dist, ggml_scale(ctx_eval, dot, 2.0f));

    ggml_tensor * indices = codec_rvq_argmin_map_custom1(ctx_eval, dist);
    if (indices == nullptr) {
        return false;
    }

    ggml_tensor * quantized = ggml_get_rows(ctx_eval, codebook_dc, indices); // [d, t]
    if (quantized == nullptr) {
        return false;
    }

    out->indices = indices;
    out->residual = ggml_sub(ctx_eval, residual_ct, quantized);
    return out->residual != nullptr;
}
