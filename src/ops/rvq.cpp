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

static void codec_rvq_select_indices_map_custom2(
    ggml_tensor * dst,
    const ggml_tensor * residual_ct,
    const ggml_tensor * codebook_dc,
    int ith,
    int nth,
    void * userdata) {

    (void) userdata;

    if (dst == nullptr || residual_ct == nullptr || codebook_dc == nullptr) {
        return;
    }
    if (dst->type != GGML_TYPE_F32 || residual_ct->type != GGML_TYPE_F32 || codebook_dc->type != GGML_TYPE_F32) {
        return;
    }
    if (residual_ct->ne[0] <= 0 || residual_ct->ne[1] <= 0 || codebook_dc->ne[0] != residual_ct->ne[0] || codebook_dc->ne[1] <= 0) {
        return;
    }

    const int64_t dim = residual_ct->ne[0];
    const int64_t frames = residual_ct->ne[1];
    const int64_t codebook_size = codebook_dc->ne[1];
    const int64_t start = (frames * ith) / nth;
    const int64_t end = (frames * (ith + 1)) / nth;

    for (int64_t t = start; t < end; ++t) {
        const float * residual = reinterpret_cast<const float *>(reinterpret_cast<const char *>(residual_ct->data) + (size_t) t * residual_ct->nb[1]);
        float best_dist = 0.0f;
        int32_t best_idx = 0;

        for (int32_t cb = 0; cb < (int32_t) codebook_size; ++cb) {
            const float * code = reinterpret_cast<const float *>(reinterpret_cast<const char *>(codebook_dc->data) + (size_t) cb * codebook_dc->nb[1]);

            float dist = 0.0f;
            for (int32_t i = 0; i < (int32_t) dim; ++i) {
                const float diff = residual[(size_t) i] - code[(size_t) i];
                dist += diff * diff;
            }

            if (cb == 0 || dist < best_dist) {
                best_dist = dist;
                best_idx = cb;
            }
        }

        float * dst_col = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + (size_t) t * dst->nb[1]);
        dst_col[0] = (float) best_idx;
    }
}

static void codec_rvq_residual_update_map_custom3(
    ggml_tensor * dst,
    const ggml_tensor * residual_ct,
    const ggml_tensor * codebook_dc,
    const ggml_tensor * indices_t,
    int ith,
    int nth,
    void * userdata) {

    (void) userdata;

    if (dst == nullptr || residual_ct == nullptr || codebook_dc == nullptr || indices_t == nullptr) {
        return;
    }
    if (dst->type != GGML_TYPE_F32 || residual_ct->type != GGML_TYPE_F32 || codebook_dc->type != GGML_TYPE_F32 || indices_t->type != GGML_TYPE_I32) {
        return;
    }
    if (residual_ct->ne[0] <= 0 || residual_ct->ne[1] <= 0 || codebook_dc->ne[0] != residual_ct->ne[0] || codebook_dc->ne[1] <= 0 || indices_t->ne[0] != residual_ct->ne[1]) {
        return;
    }

    const int64_t dim = residual_ct->ne[0];
    const int64_t frames = residual_ct->ne[1];
    const int64_t codebook_size = codebook_dc->ne[1];
    const int64_t start = (frames * ith) / nth;
    const int64_t end = (frames * (ith + 1)) / nth;

    for (int64_t t = start; t < end; ++t) {
        const int32_t idx = *reinterpret_cast<const int32_t *>(reinterpret_cast<const char *>(indices_t->data) + (size_t) t * indices_t->nb[0]);
        if (idx < 0 || idx >= codebook_size) {
            continue;
        }

        const float * residual = reinterpret_cast<const float *>(reinterpret_cast<const char *>(residual_ct->data) + (size_t) t * residual_ct->nb[1]);
        const float * code = reinterpret_cast<const float *>(reinterpret_cast<const char *>(codebook_dc->data) + (size_t) idx * codebook_dc->nb[1]);
        float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + (size_t) t * dst->nb[1]);

        for (int32_t i = 0; i < (int32_t) dim; ++i) {
            out[(size_t) i] = residual[(size_t) i] - code[(size_t) i];
        }
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

ggml_tensor * codec_rvq_select_indices_ggml(
    ggml_context * ctx_eval,
    ggml_tensor * residual_ct,
    ggml_tensor * codebook_dc) {

    if (ctx_eval == nullptr || residual_ct == nullptr || codebook_dc == nullptr) {
        return nullptr;
    }
    if (residual_ct->type != GGML_TYPE_F32 || codebook_dc->type != GGML_TYPE_F32) {
        return nullptr;
    }
    if (residual_ct->ne[0] <= 0 || residual_ct->ne[1] <= 0 || codebook_dc->ne[0] != residual_ct->ne[0] || codebook_dc->ne[1] <= 0) {
        return nullptr;
    }

    ggml_tensor * residual = ggml_cont(ctx_eval, residual_ct);
    ggml_tensor * codebook = ggml_cont(ctx_eval, codebook_dc);
    ggml_tensor * argmin_full = ggml_map_custom2(ctx_eval, residual, codebook, codec_rvq_select_indices_map_custom2, GGML_N_TASKS_MAX, nullptr);
    if (argmin_full == nullptr) {
        return nullptr;
    }

    ggml_tensor * argmin_row = ggml_view_2d(ctx_eval, argmin_full, 1, residual->ne[1], argmin_full->nb[1], 0);
    argmin_row = ggml_cont(ctx_eval, argmin_row);
    argmin_row = ggml_reshape_1d(ctx_eval, argmin_row, residual->ne[1]);
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

    ggml_tensor * indices = codec_rvq_select_indices_ggml(ctx_eval, residual_ct, codebook_dc);
    if (indices == nullptr) {
        return false;
    }

    ggml_tensor * residual = ggml_map_custom3(ctx_eval, residual_ct, codebook_dc, indices, codec_rvq_residual_update_map_custom3, GGML_N_TASKS_MAX, nullptr);
    if (residual == nullptr) {
        return false;
    }

    out->indices = indices;
    out->residual = residual;
    return true;
}
