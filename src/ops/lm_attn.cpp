#include "lm_attn.h"

#include <cmath>

ggml_tensor * codec_op_lm_attn_ctx_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const codec_lm_attn_params * params) {

    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr) {
        return nullptr;
    }
    if (q_dth->ne[0] != k_dth->ne[0] || q_dth->ne[0] != v_dth->ne[0] ||
        q_dth->ne[1] != k_dth->ne[1] || q_dth->ne[1] != v_dth->ne[1] ||
        q_dth->ne[2] != k_dth->ne[2] || q_dth->ne[2] != v_dth->ne[2]) {
        return nullptr;
    }

    const int32_t head_dim = (int32_t) q_dth->ne[0];
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max(1, head_dim)));
    const bool causal = params != nullptr && params->causal;

    ggml_tensor * k_cont = ggml_cont(ctx, k_dth);
    ggml_tensor * attn_scores = ggml_mul_mat(ctx, k_cont, q_dth); // [t, t, h]
    if (attn_scores == nullptr) {
        return nullptr;
    }

    attn_scores = ggml_scale_inplace(ctx, attn_scores, scale);
    if (causal) {
        attn_scores = ggml_diag_mask_inf_inplace(ctx, attn_scores, 0);
    }

    ggml_tensor * attn_probs = ggml_soft_max(ctx, attn_scores);
    if (attn_probs == nullptr) {
        return nullptr;
    }

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_tdh, attn_probs); // [d, t, h]
}

ggml_tensor * codec_op_rel_shift_espnet(ggml_context * ctx, ggml_tensor * x) {
    if (ctx == nullptr || x == nullptr) return nullptr;
    const int64_t two_t1 = x->ne[0];
    const int64_t t = x->ne[1];
    const int64_t h = x->ne[2];
    if (two_t1 != 2 * t - 1) return nullptr;

    // 1. Pad zeros on the LEFT of ne[0] → (2t, t, h).
    ggml_tensor * zp = ggml_new_tensor_3d(ctx, x->type, 1, t, h);
    zp = ggml_scale(ctx, zp, 0.0f);
    ggml_tensor * x_pad = ggml_concat(ctx, zp, x, /*dim=*/0);
    if (x_pad == nullptr) return nullptr;

    // 2. Reshape (2t, t, h) → (t, 2t, h) (same memory, different view).
    ggml_tensor * x_pad_cont = ggml_cont(ctx, x_pad);
    ggml_tensor * x_view = ggml_reshape_3d(ctx, x_pad_cont, t, 2 * t, h);

    // 3. Drop first row of ne[1] → (t, 2t-1, h).
    ggml_tensor * x_drop = ggml_view_3d(
        ctx, x_view,
        /*ne0=*/t, /*ne1=*/2 * t - 1, /*ne2=*/h,
        /*nb1=*/x_view->nb[1],
        /*nb2=*/x_view->nb[2],
        /*offset=*/x_view->nb[1]);
    x_drop = ggml_cont(ctx, x_drop);

    // 4. Reshape back to (2t-1, t, h), then take first t entries of ne[0].
    ggml_tensor * x_back = ggml_reshape_3d(ctx, x_drop, 2 * t - 1, t, h);
    ggml_tensor * x_out = ggml_view_3d(
        ctx, x_back,
        /*ne0=*/t, /*ne1=*/t, /*ne2=*/h,
        /*nb1=*/x_back->nb[1],
        /*nb2=*/x_back->nb[2],
        /*offset=*/0);
    return ggml_cont(ctx, x_out);
}

ggml_tensor * codec_op_lm_attn_rel_pos_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    ggml_tensor * p_dth,
    ggml_tensor * pos_bias_u,
    ggml_tensor * pos_bias_v,
    const codec_lm_attn_params * params) {
    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr ||
        p_dth == nullptr || pos_bias_u == nullptr || pos_bias_v == nullptr) {
        return nullptr;
    }
    const int64_t head_dim = q_dth->ne[0];
    const int64_t t = q_dth->ne[1];
    const int64_t h = q_dth->ne[2];
    if (k_dth->ne[0] != head_dim || v_dth->ne[0] != head_dim || p_dth->ne[0] != head_dim ||
        k_dth->ne[1] != t || v_dth->ne[1] != t || p_dth->ne[1] != 2 * t - 1 ||
        k_dth->ne[2] != h || v_dth->ne[2] != h || p_dth->ne[2] != h) {
        return nullptr;
    }
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max<int64_t>(1, head_dim)));

    // Add per-head bias broadcast over t: q + bias_u, q + bias_v.
    auto add_bias = [&](ggml_tensor * q, ggml_tensor * bias_dh) -> ggml_tensor * {
        ggml_tensor * b3 = ggml_reshape_3d(ctx, bias_dh, head_dim, 1, h);
        return ggml_add(ctx, q, ggml_repeat(ctx, b3, q));
    };
    ggml_tensor * q_u = add_bias(q_dth, pos_bias_u);
    ggml_tensor * q_v = add_bias(q_dth, pos_bias_v);
    if (q_u == nullptr || q_v == nullptr) return nullptr;

    // matrix_ac = q_u · k.T  → (t, t, h).
    ggml_tensor * mat_ac = ggml_mul_mat(ctx, ggml_cont(ctx, k_dth), q_u);
    // matrix_bd = q_v · p.T  → (2t-1, t, h), then rel-shift to (t, t, h).
    ggml_tensor * mat_bd = ggml_mul_mat(ctx, ggml_cont(ctx, p_dth), q_v);
    if (mat_ac == nullptr || mat_bd == nullptr) return nullptr;
    mat_bd = codec_op_rel_shift_espnet(ctx, mat_bd);
    if (mat_bd == nullptr) return nullptr;

    ggml_tensor * scores = ggml_add(ctx, mat_ac, mat_bd);
    scores = ggml_scale(ctx, scores, scale);
    ggml_tensor * attn_w = ggml_soft_max(ctx, scores);

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_tdh, attn_w);
}

ggml_tensor * codec_op_lm_attn_rel_key_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    ggml_tensor * dist_emb_dn,
    ggml_tensor * bucket_idx_1d,
    const codec_lm_attn_params * params) {
    if (ctx == nullptr || q_dth == nullptr || k_dth == nullptr || v_dth == nullptr ||
        dist_emb_dn == nullptr || bucket_idx_1d == nullptr) {
        return nullptr;
    }
    const int64_t head_dim = q_dth->ne[0];
    const int64_t t = q_dth->ne[1];
    const int64_t h = q_dth->ne[2];
    if (k_dth->ne[0] != head_dim || v_dth->ne[0] != head_dim ||
        k_dth->ne[1] != t || v_dth->ne[1] != t ||
        k_dth->ne[2] != h || v_dth->ne[2] != h ||
        dist_emb_dn->ne[0] != head_dim ||
        bucket_idx_1d->ne[0] != t * t) {
        return nullptr;
    }
    const float scale = (params != nullptr && params->scale > 0.0f)
        ? params->scale
        : (1.0f / std::sqrt((float) std::max<int64_t>(1, head_dim)));

    // Standard attention scores: [t_k, t_q, h].
    ggml_tensor * ac = ggml_mul_mat(ctx, ggml_cont(ctx, k_dth), q_dth);
    if (ac == nullptr) return nullptr;

    // Gather E[d, t_k * t_q] = D[d, bucket(t_k, t_q)]. The bucket index is laid
    // out row-major with t_k inner, t_q outer, so reshape to (t_k, t_q).
    ggml_tensor * E_flat = ggml_get_rows(ctx, dist_emb_dn, bucket_idx_1d);  // [d, t*t]
    if (E_flat == nullptr) return nullptr;
    ggml_tensor * E_3d = ggml_reshape_3d(ctx, E_flat, head_dim, t, t);      // [d, t_k, t_q]

    // Permute Q to per-t_q batch: [d, t_q, h] -> [d, h, t_q].
    ggml_tensor * q_dh_tq = ggml_cont(ctx, ggml_permute(ctx, q_dth, 0, 2, 1, 3));  // [d, h, t_q]

    // Per-t_q batched mul_mat: result ne=(t_k, h, t_q). mul_mat contracts d.
    ggml_tensor * rel = ggml_mul_mat(ctx, E_3d, q_dh_tq);                          // [t_k, h, t_q]
    if (rel == nullptr) return nullptr;

    // Permute to [t_k, t_q, h] to align with `ac`.
    rel = ggml_cont(ctx, ggml_permute(ctx, rel, 0, 2, 1, 3));

    ggml_tensor * scores = ggml_add(ctx, ac, rel);
    scores = ggml_scale(ctx, scores, scale);
    ggml_tensor * probs = ggml_soft_max(ctx, scores);

    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));
    return ggml_mul_mat(ctx, v_tdh, probs);
}

