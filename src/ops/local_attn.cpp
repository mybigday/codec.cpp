#include "local_attn.h"

#include <algorithm>
#include <cmath>
#include <vector>

static void codec_local_attn_map_custom3(
    ggml_tensor * dst,
    const ggml_tensor * q,
    const ggml_tensor * k,
    const ggml_tensor * v,
    int ith,
    int nth,
    void * userdata) {

    const codec_local_attn_params * p = static_cast<const codec_local_attn_params *>(userdata);
    if (dst == nullptr || q == nullptr || k == nullptr || v == nullptr || p == nullptr) {
        return;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return;
    }
    if (q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0) {
        return;
    }

    const int32_t head_dim = std::max(1, p->head_dim);
    const int32_t heads = std::max(1, p->heads);
    const int32_t attn_span = std::max(2, p->window);
    const int32_t window = std::max(1, attn_span / 2);
    const int32_t max_dist = std::max(1, p->max_dist);
    const float scale = 1.0f / std::sqrt((float) head_dim);

    const int64_t t = q->ne[1];
    const int64_t h = q->ne[2];

    const int64_t cols = t * h;
    const int64_t start = (cols * ith) / nth;
    const int64_t end = (cols * (ith + 1)) / nth;

    for (int64_t col = start; col < end; ++col) {
        const int64_t ti = col % t;
        const int64_t hi = col / t;

        if (hi >= heads) {
            continue;
        }

        const float * q_ptr = reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(q->data) + (size_t) hi * q->nb[2] + (size_t) ti * q->nb[1]);
        float * out_ptr = reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst->data) + (size_t) hi * dst->nb[2] + (size_t) ti * dst->nb[1]);

        const int64_t block = ti / window;
        const int64_t i_in = ti % window;
        const int64_t key_start = block * window - window;
        const int64_t span = 2 * (int64_t) window;
        std::vector<float> scores((size_t) span, -INFINITY);
        float max_score = -INFINITY;

        for (int64_t idx = 0; idx < span; ++idx) {
            const int64_t kj = key_start + idx;
            if (kj < 0 || kj >= t || kj > ti) {
                continue;
            }
            const float * k_ptr = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(k->data) + (size_t) hi * k->nb[2] + (size_t) kj * k->nb[1]);
            float dot = 0.0f;
            for (int32_t di = 0; di < head_dim; ++di) {
                dot += q_ptr[(size_t) di] * k_ptr[(size_t) di];
            }
            dot *= scale;
            const int64_t rel = std::llabs((int64_t) window + i_in - idx);
            if (p->bias != nullptr && rel < max_dist) {
                dot += p->bias[(size_t) hi * (size_t) max_dist + (size_t) rel];
            }
            scores[(size_t) idx] = dot;
            if (dot > max_score) {
                max_score = dot;
            }
        }

        if (!std::isfinite(max_score)) {
            std::fill(out_ptr, out_ptr + head_dim, 0.0f);
            continue;
        }

        float denom = 0.0f;
        for (int64_t idx = 0; idx < span; ++idx) {
            if (!std::isfinite(scores[(size_t) idx])) {
                scores[(size_t) idx] = 0.0f;
                continue;
            }
            float e = std::exp(scores[(size_t) idx] - max_score);
            scores[(size_t) idx] = e;
            denom += e;
        }
        denom = denom > 0.0f ? denom : 1.0f;

        std::fill(out_ptr, out_ptr + head_dim, 0.0f);
        for (int64_t idx = 0; idx < span; ++idx) {
            if (scores[(size_t) idx] <= 0.0f) {
                continue;
            }
            const int64_t kj = key_start + idx;
            const float w = scores[(size_t) idx] / denom;
            const float * v_ptr = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(v->data) + (size_t) hi * v->nb[2] + (size_t) kj * v->nb[1]);
            for (int32_t di = 0; di < head_dim; ++di) {
                out_ptr[(size_t) di] += w * v_ptr[(size_t) di];
            }
        }
    }
}

ggml_tensor * codec_op_local_attn(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const codec_local_attn_params * params) {

    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr || params == nullptr) {
        return nullptr;
    }
    ggml_tensor * q = q_dth->type == GGML_TYPE_F32 ? q_dth : ggml_cast(ctx, q_dth, GGML_TYPE_F32);
    ggml_tensor * k = k_dth->type == GGML_TYPE_F32 ? k_dth : ggml_cast(ctx, k_dth, GGML_TYPE_F32);
    ggml_tensor * v = v_dth->type == GGML_TYPE_F32 ? v_dth : ggml_cast(ctx, v_dth, GGML_TYPE_F32);
    ggml_tensor * out = ggml_map_custom3(ctx, q, k, v, codec_local_attn_map_custom3, GGML_N_TASKS_MAX, (void *) params);
    return out ? ggml_cont(ctx, out) : nullptr;
}
