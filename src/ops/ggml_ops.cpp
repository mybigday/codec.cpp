#include "ggml_ops.h"

#include <algorithm>
#include <cfloat>

ggml_tensor * codec_op_unary(ggml_context * ctx, ggml_tensor * x, codec_unary_op op) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    switch (op) {
        case CODEC_UNARY_SIGMOID: {
            ggml_tensor * x_half = ggml_scale(ctx, x, 0.5f);
            ggml_tensor * th = ggml_tanh(ctx, x_half);
            return ggml_scale_bias(ctx, th, 0.5f, 0.5f);
        }
        case CODEC_UNARY_ELU:
            return ggml_elu(ctx, x);
        case CODEC_UNARY_SILU:
            return ggml_silu(ctx, x);
        case CODEC_UNARY_GELU_ERF:
            return ggml_gelu_erf(ctx, x);
        default:
            return nullptr;
    }
}

ggml_tensor * codec_op_layer_norm(ggml_context * ctx, ggml_tensor * x, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = ggml_norm(ctx, x, eps);
    if (gamma != nullptr) {
        ggml_tensor * g2 = ggml_reshape_2d(ctx, gamma, 1, x->ne[1]);
        y = ggml_mul(ctx, y, ggml_repeat(ctx, g2, y));
    }
    if (beta != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, beta, 1, x->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return y;
}

ggml_tensor * codec_op_group_norm(ggml_context * ctx, ggml_tensor * x, int32_t n_groups, float eps, ggml_tensor * gamma, ggml_tensor * beta) {
    if (ctx == nullptr || x == nullptr || n_groups <= 0 || x->ne[1] % n_groups != 0) {
        return nullptr;
    }

    const int64_t cpg = x->ne[1] / n_groups;
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, x->ne[0], cpg, n_groups);
    ggml_tensor * y = ggml_reshape_2d(ctx, ggml_group_norm(ctx, x3, n_groups, eps), x->ne[0], x->ne[1]);

    if (gamma != nullptr) {
        ggml_tensor * g2 = ggml_reshape_2d(ctx, gamma, 1, x->ne[1]);
        y = ggml_mul(ctx, y, ggml_repeat(ctx, g2, y));
    }
    if (beta != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, beta, 1, x->ne[1]);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return y;
}

ggml_tensor * codec_op_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    if (ctx == nullptr || x == nullptr || w == nullptr) {
        return nullptr;
    }

    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx, b, w->ne[1], 1);
        y = ggml_add(ctx, y, ggml_repeat(ctx, b2, y));
    }
    return y;
}

ggml_tensor * codec_op_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps) {
    if (ctx == nullptr || x == nullptr || alpha == nullptr) {
        return nullptr;
    }

    ggml_tensor * alpha_2d = ggml_reshape_2d(ctx, alpha, 1, x->ne[1]);
    ggml_tensor * alpha_rep = ggml_repeat(ctx, alpha_2d, x);
    ggml_tensor * alpha_clamped = ggml_clamp(ctx, alpha_rep, eps, FLT_MAX);
    ggml_tensor * ax = ggml_mul(ctx, alpha_clamped, x);
    ggml_tensor * s = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    ggml_tensor * frac = ggml_div(ctx, s2, alpha_clamped);
    return ggml_add(ctx, x, frac);
}

ggml_tensor * codec_op_pad_1d(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    return ggml_pad_ext(ctx, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
}

ggml_tensor * codec_op_pad_1d_replicate(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right) {
    if (ctx == nullptr || x == nullptr || pad_left < 0 || pad_right < 0) {
        return nullptr;
    }
    if (pad_left == 0 && pad_right == 0) {
        return x;
    }

    ggml_tensor * out = x;
    if (pad_left > 0) {
        ggml_tensor * left = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], 0);
        ggml_tensor * left_target = ggml_new_tensor_2d(ctx, x->type, pad_left, x->ne[1]);
        ggml_tensor * left_rep = ggml_repeat(ctx, left, left_target);
        out = ggml_concat(ctx, left_rep, out, 0);
    }

    if (pad_right > 0) {
        const size_t offset = (size_t) (x->ne[0] - 1) * (size_t) x->nb[0];
        ggml_tensor * right = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], offset);
        ggml_tensor * right_target = ggml_new_tensor_2d(ctx, x->type, pad_right, x->ne[1]);
        ggml_tensor * right_rep = ggml_repeat(ctx, right, right_target);
        out = ggml_concat(ctx, out, right_rep, 0);
    }

    return ggml_cont(ctx, out);
}

ggml_tensor * codec_op_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t crop_left, int32_t crop_right) {
    if (ctx == nullptr || x == nullptr || crop_left < 0 || crop_right < 0) {
        return nullptr;
    }
    const int64_t out_t = x->ne[0] - crop_left - crop_right;
    if (out_t <= 0) {
        return nullptr;
    }
    ggml_tensor * view = ggml_view_2d(ctx, x, out_t, x->ne[1], x->nb[1], (size_t) crop_left * sizeof(float));
    return ggml_cont(ctx, view);
}

ggml_tensor * codec_op_causal_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t target_t) {
    if (ctx == nullptr || x == nullptr || target_t <= 0 || x->ne[0] < target_t) {
        return nullptr;
    }
    return codec_op_crop_1d(ctx, x, 0, (int32_t)x->ne[0] - target_t);
}

ggml_tensor * codec_op_channel_scale(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale) {
    if (ctx == nullptr || x == nullptr || scale == nullptr) {
        return nullptr;
    }
    ggml_tensor * s2 = ggml_reshape_2d(ctx, scale, x->ne[0], 1);
    return ggml_mul(ctx, x, ggml_repeat(ctx, s2, x));
}

ggml_tensor * codec_op_tokens_to_features(ggml_context * ctx, ggml_tensor * tokens, int32_t out_channels) {
    if (ctx == nullptr || tokens == nullptr || out_channels <= 0) {
        return nullptr;
    }

    ggml_tensor * x = ggml_scale(ctx, tokens, 1.0f / 1024.0f);
    ggml_tensor * base = nullptr;
    if (x->ne[1] <= 1) {
        base = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], 0));
    } else {
        // Aggregate all quantizers instead of only using q=0.
        ggml_tensor * x_qt = ggml_cont(ctx, ggml_transpose(ctx, x)); // [q, t]
        ggml_tensor * mean_1t = ggml_scale(ctx, ggml_sum_rows(ctx, x_qt), 1.0f / (float) x_qt->ne[0]); // [1, t]
        base = ggml_cont(ctx, ggml_transpose(ctx, mean_1t));         // [t, 1]
    }
    if (out_channels == 1) {
        return base;
    }
    return ggml_repeat(ctx, base, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], out_channels));
}
