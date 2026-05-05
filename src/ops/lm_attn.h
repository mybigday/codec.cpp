#pragma once

#include "../codec_internal.h"

struct codec_lm_attn_params {
    float scale = 0.0f;   // if <= 0, use 1 / sqrt(head_dim)
    bool causal = false;
};

// q_dth, k_dth, v_dth are [head_dim, t, n_heads]
// returns context tensor [head_dim, t, n_heads]
ggml_tensor * codec_op_lm_attn_ctx_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    const codec_lm_attn_params * params);

// Espnet rel-shift trick. Input has ne[0]=2t-1, ne[1]=t, ne[2]=heads — the
// matrix-BD shape from a Conformer self-attention with relative positional
// encoding (matrix_ac shape (t, t, h) ≠ matrix_bd shape (2t-1, t, h)).
// Returns ne[0]=t, ne[1]=t, ne[2]=heads via the standard zero-pad / view-shift
// permutation described in https://arxiv.org/abs/1901.02860 §B.
ggml_tensor * codec_op_rel_shift_espnet(ggml_context * ctx, ggml_tensor * x);

// Conformer self-attention with Espnet relative positional encoding.
// Inputs:
//   q_dth, k_dth, v_dth: [head_dim, t, n_heads]
//   p_dth (linear_pos(pos_emb) reshaped to per-head): [head_dim, 2t-1, n_heads]
//   pos_bias_u, pos_bias_v: [head_dim, n_heads]  (broadcast over t)
// Returns context tensor [head_dim, t, n_heads].
ggml_tensor * codec_op_lm_attn_rel_pos_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    ggml_tensor * p_dth,
    ggml_tensor * pos_bias_u,
    ggml_tensor * pos_bias_v,
    const codec_lm_attn_params * params);

// Shaw-style relative-key self-attention (used by Wav2Vec2-Bert /
// SeamlessM4T conformer). Adds Q · D[bucket(j-i)] / sqrt(d) on top of the
// standard q@k.T / sqrt(d) attention scores, where D is a learned distance
// embedding shared across heads.
//   q_dth, k_dth, v_dth: [head_dim, t, n_heads]
//   dist_emb_dn:         [head_dim, n_buckets]  (n_buckets = left_max+right_max+1)
//   bucket_idx_1d:       int32 [t*t]  with row-major layout (t_k inner, t_q outer)
//                        and value `clamp(t_k - t_q, -left_max, right_max) + left_max`.
// Returns context tensor [head_dim, t, n_heads].
ggml_tensor * codec_op_lm_attn_rel_key_dth(
    ggml_context * ctx,
    ggml_tensor * q_dth,
    ggml_tensor * k_dth,
    ggml_tensor * v_dth,
    ggml_tensor * dist_emb_dn,
    ggml_tensor * bucket_idx_1d,
    const codec_lm_attn_params * params);

