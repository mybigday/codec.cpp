#include "pool1d.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>

struct pool1d_params {
    int32_t kernel = 1;
    int32_t pad = 0;
    bool is_max = false;
};

static void codec_pool1d_map_custom1(
    ggml_tensor * dst,
    const ggml_tensor * src,
    int ith,
    int nth,
    void * userdata) {

    pool1d_params * p = static_cast<pool1d_params *>(userdata);
    if (dst == nullptr || src == nullptr || p == nullptr) {
        return;
    }
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return;
    }
    if (src->ne[0] <= 0 || src->ne[1] <= 0) {
        return;
    }

    const int64_t t = src->ne[0];
    const int64_t c = src->ne[1];
    const int64_t cols = c * src->ne[2] * src->ne[3];
    const int64_t start = (cols * ith) / nth;
    const int64_t end = (cols * (ith + 1)) / nth;

    const int32_t kernel = std::max(1, p->kernel);
    const int32_t pad = std::max(0, p->pad);

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

        for (int64_t ti = 0; ti < t; ++ti) {
            const int64_t left = ti - pad;
            const int64_t right = left + kernel;
            float acc = p->is_max ? -INFINITY : 0.0f;
            int32_t count = 0;

            for (int64_t si = left; si < right; ++si) {
                float v = 0.0f;
                if (si >= 0 && si < t) {
                    v = src_col[(size_t) si];
                }
                if (p->is_max) {
                    if (v > acc) {
                        acc = v;
                    }
                } else {
                    acc += v;
                    count += 1;
                }
            }

            if (!p->is_max) {
                acc = count > 0 ? acc / (float) count : 0.0f;
            }
            dst_col[(size_t) ti] = acc;
        }
    }
}

static ggml_tensor * codec_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad,
    bool is_max) {

    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }
    // ggml_map_custom* keeps userdata pointer for graph lifetime.
    // Use a static registry so userdata remains valid after this function returns.
    static std::mutex s_params_mu;
    static std::map<uint64_t, pool1d_params> s_params;

    const int32_t k = std::max(1, kernel);
    const int32_t p = std::max(0, pad);
    const uint64_t key = (uint64_t) (uint32_t) k |
                         ((uint64_t) (uint32_t) p << 32) |
                         ((uint64_t) (is_max ? 1 : 0) << 63);

    pool1d_params * params = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_params_mu);
        auto it = s_params.find(key);
        if (it == s_params.end()) {
            pool1d_params v;
            v.kernel = k;
            v.pad = p;
            v.is_max = is_max;
            it = s_params.emplace(key, v).first;
        }
        params = &it->second;
    }

    ggml_tensor * x_f32 = x->type == GGML_TYPE_F32 ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    ggml_tensor * out = ggml_map_custom1(ctx, x_f32, codec_pool1d_map_custom1, GGML_N_TASKS_MAX, params);
    return out ? ggml_cont(ctx, out) : nullptr;
}

ggml_tensor * codec_op_max_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad) {

    return codec_pool1d(ctx, x, kernel, pad, true);
}

ggml_tensor * codec_op_avg_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad) {

    return codec_pool1d(ctx, x, kernel, pad, false);
}
