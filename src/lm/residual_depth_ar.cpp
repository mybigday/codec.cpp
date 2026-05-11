#include "lm_internal.h"

#include "../runtime/graph.h"
#include "../runtime/graph_exec.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// codec_lm kind: residual_depth_ar  (CSM, Qwen3-TTS, Moshi, LFM2-Audio)
//
// CSM is the reference implementation:
//   - Backbone (Llama-3.2-1B) runs in llama.cpp; the caller hands its
//     last-position hidden state to codec_lm_step_begin.
//   - codec_lm holds the audio embedding tables, c0 head, per-cb depth
//     heads, an `in_proj` (backbone_hidden -> depth_hidden), and a
//     small Llama-style depth decoder transformer (4 layers @ 1024
//     hidden for CSM-100M).
//   - Per backbone step:
//       1. c0_logits = c0_head @ h_in
//       2. caller samples c0
//       3. depth decoder runs over the prefix
//          [in_proj(h_in), in_proj(audio_embd_0[c0])]
//          → c1_logits via depth_heads[0]
//       4. caller samples c1
//       5. for k = 2..N-1, append in_proj(audio_embd_{k-1}[c_{k-1}]),
//          rerun, → ck_logits via depth_heads[k-1]
//
// First implementation uses prefix-recompute: each step k>=1 rebuilds
// the depth decoder graph with the full prefix and re-runs from scratch.
// O(N²) total compute for N codebooks, but for CSM (4 layers @ 1024
// hidden, max 32 positions) it's tractable and avoids KV-cache plumbing
// in the first cut.  KV-cache becomes a perf optimisation later.
//
// Llama3-style RoPE scaling is handled via the `lm.depth.rope_freq_factors`
// tensor the converter precomputes — the runtime feeds it to
// `ggml_rope_ext` as `freq_factors`, no in-graph piecewise math.
// =====================================================================

namespace {

struct rda_layer_w {
    ggml_tensor * attn_norm;
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * o;
    ggml_tensor * ffn_norm;
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_up;
    ggml_tensor * ffn_down;
    ggml_tensor * q_norm;   // optional (Qwen3 family)
    ggml_tensor * k_norm;   // optional
};

struct rda_impl {
    int32_t n_codebook       = 0;
    int32_t hidden_dim       = 0;   // backbone
    int32_t audio_embed_dim  = 0;   // = hidden_dim for CSM
    int32_t depth_hidden     = 0;
    int32_t depth_layers     = 0;
    int32_t depth_n_heads    = 0;
    int32_t depth_n_kv_heads = 0;
    int32_t depth_head_dim   = 0;
    int32_t depth_inter      = 0;
    int32_t depth_max_pos    = 0;
    float   depth_rope_theta = 0.0f;
    float   depth_rms_eps    = 0.0f;
    bool    has_in_proj      = false;
    bool    has_qk_norm      = false;
    bool    has_output_norm  = true;   // most models have one; Moshi doesn't
    bool    use_rope         = true;   // most models apply RoPE; Moshi doesn't
    bool    is_flexible      = false;  // false = shared (CSM/Qwen3-TTS),
                                       // true = flexible (Moshi: per-pos weights)
    bool    c0_is_text       = false;  // c0_input_modality="text" (Moshi)
    int32_t depth_text_vocab = 0;      // text vocab for the c0_is_text input

    // Tensor handles (live in codec->weights).
    //
    // Shared mode (is_flexible=false):
    //   - `audio_embds[0..N-1]`: 2D embedding tables, one per codebook;
    //     used for both compose_audio_embd (sum across cb) AND depth-
    //     position embed lookup.
    //   - `c0_head`: separate 2D head for cb 0.
    //   - `depth_heads[0..N-2]`: 2D heads for cb 1..N-1.
    //   - `layers[*]`: 2D per-layer weights (rda_layer_w).
    //
    // Flexible mode (is_flexible=true):
    //   - `text_embd`: 2D embedding table for depth pos 0 (text vocab).
    //   - `audio_embds[0..N-2]`: 2D audio embed tables (N-1 of them),
    //     used at depth pos 1..N-1.  No compose_audio_embd path —
    //     caller handles backbone-side composition.
    //   - `flex_heads`: single 3D tensor (N, V_audio, H_d); per-position
    //     head at depth position p uses `flex_heads[p]`.
    //   - `flex_layers[*]`: 3D per-layer weights (rda_flex_layer_w).
    std::vector<ggml_tensor *> audio_embds;   // shared: [N]; flexible: [N-1]
    ggml_tensor * text_embd         = nullptr; // flexible only
    ggml_tensor * c0_head           = nullptr; // shared only
    std::vector<ggml_tensor *> depth_heads;    // shared only [N-1]
    ggml_tensor * flex_heads        = nullptr; // flexible only (3D)
    ggml_tensor * in_proj           = nullptr; // shared (2D) or flexible (3D)
    ggml_tensor * in_proj_bias      = nullptr; // shared only (Qwen3-TTS bias)
    ggml_tensor * depth_output_norm = nullptr; // shared only (Moshi omits)
    ggml_tensor * rope_freq_factors = nullptr; // shared only (llama3 scaling)
    std::vector<rda_layer_w> layers;           // shared mode [depth_layers]
    std::vector<rda_layer_w> flex_layers;      // flexible mode [depth_layers]
                                                // (same struct, 3D tensors)

    // Lazily-allocated codec_context for `compose_audio_embd` graphs;
    // kept separate from per-state ctx so concurrent step + compose
    // don't fight over an eval arena.
    codec_context * compose_ctx = nullptr;
};

struct rda_state {
    // Buffers backing the step machine:
    std::vector<float> h_in_buf;                    // [hidden_dim]; written at step_begin
    std::vector<std::vector<float>> logits_buf;     // [n_codebook]; per-cb scratch
};

// ---------------------------------------------------------------------
// graph helpers — Llama-style block (RMSNorm + GQA + RoPE + SwiGLU)
// ---------------------------------------------------------------------

// RMSNorm with weighted gain.  ggml_rms_norm divides by sqrt(mean(x^2)),
// then we scale by `gamma` channel-wise.  Input/output are (hidden, T).
ggml_tensor * rda_rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gamma, float eps) {
    ggml_tensor * normed = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, normed, gamma);
}

// One Llama transformer layer.  `x_ht` shape (depth_hidden, T) where
// depth_hidden = n_heads * head_dim (for q) or n_kv_heads * head_dim (for kv).
// Returns the residual-updated hidden state (same shape).
//
// `t_pos` is a (T,) int32 tensor with RoPE positions for this layer.
// `freq_factors` is an optional (head_dim/2,) f32 tensor of llama3 scaling.
ggml_tensor * rda_depth_layer(
    ggml_context * ctx,
    ggml_tensor * x_ht,
    const rda_layer_w & w,
    ggml_tensor * t_pos,
    ggml_tensor * freq_factors,
    int32_t head_dim,
    int32_t n_heads,
    int32_t n_kv_heads,
    float   rope_theta,
    float   rms_eps,
    bool    has_qk_norm) {

    const int64_t T = x_ht->ne[1];

    // ── Attention ──────────────────────────────────────────────────
    ggml_tensor * h = rda_rms_norm(ctx, x_ht, w.attn_norm, rms_eps);

    // Project Q/K/V — w.* are (in, out) so mul_mat(w, h) → (out, T).
    ggml_tensor * q = ggml_mul_mat(ctx, w.q, h);
    ggml_tensor * k = ggml_mul_mat(ctx, w.k, h);
    ggml_tensor * v = ggml_mul_mat(ctx, w.v, h);

    // Reshape into (head_dim, n_heads, T).
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads,    T);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, T);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads, T);

    if (has_qk_norm && w.q_norm != nullptr && w.k_norm != nullptr) {
        // Per-head RMSNorm on q/k (Qwen3 family); not used for CSM.
        q = rda_rms_norm(ctx, q, w.q_norm, rms_eps);
        k = rda_rms_norm(ctx, k, w.k_norm, rms_eps);
    }

    // RoPE on q and k along the head_dim axis.  Mode is NEOX — HF Llama
    // (and CSM, which inherits Llama's RoPE) uses the half-rotation
    // variant (`rotate_half(x) = cat(-x[D/2:], x[:D/2])`), which is what
    // ggml calls NEOX.  GGML_ROPE_TYPE_NORMAL is interleaved pairs
    // (GPT-J style) and produces a different rotation pattern.
    const int32_t rope_n_dims = head_dim;     // rotate the full head dim
    const int32_t rope_mode   = GGML_ROPE_TYPE_NEOX;
    const int32_t n_ctx_orig  = 2048;          // unused when no YaRN extension
    const float   freq_scale  = 1.0f;
    const float   ext_factor  = 0.0f;
    const float   attn_factor = 1.0f;
    const float   beta_fast   = 32.0f;
    const float   beta_slow   = 1.0f;

    q = ggml_rope_ext(ctx, q, t_pos, freq_factors, rope_n_dims, rope_mode,
                      n_ctx_orig, rope_theta, freq_scale, ext_factor,
                      attn_factor, beta_fast, beta_slow);
    k = ggml_rope_ext(ctx, k, t_pos, freq_factors, rope_n_dims, rope_mode,
                      n_ctx_orig, rope_theta, freq_scale, ext_factor,
                      attn_factor, beta_fast, beta_slow);

    // Permute for attention.  ggml's `ggml_mul_mat(a, b)` contracts on
    // ne[0]; we want q @ k^T so q_perm and k_perm both expose head_dim
    // on ne[0] and time on ne[1].  Result is (T_k, T_q, n_heads).
    //
    // GQA: ggml_mul_mat broadcasts when a and b have matching ne[2..]
    // ratios; with q ne[2]=n_heads and k ne[2]=n_kv_heads where
    // n_heads = group_size * n_kv_heads, ggml repeats k along the
    // group axis automatically (see existing usage in llama.cpp /
    // chatterbox graphs).
    ggml_tensor * q_p = ggml_permute(ctx, q, 0, 2, 1, 3);  // (head_dim, T, n_heads)
    ggml_tensor * k_p = ggml_permute(ctx, k, 0, 2, 1, 3);  // (head_dim, T, n_kv_heads)

    q_p = ggml_cont(ctx, q_p);
    k_p = ggml_cont(ctx, k_p);

    // Attention scores: (T_k, T_q, n_heads).  Scaled by 1/sqrt(head_dim).
    ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);
    const float scale = 1.0f / std::sqrt((float) head_dim);
    scores = ggml_scale(ctx, scores, scale);

    // Causal mask.  ggml_diag_mask_inf shifts the mask so query position
    // q can attend to key positions 0..q.  n_past=0 since both q and k
    // share the same starting position in the prefix-recompute regime.
    scores = ggml_diag_mask_inf(ctx, scores, /*n_past=*/0);
    scores = ggml_soft_max(ctx, scores);

    // V layout for the attn @ v multiply.  v originally is
    // (head_dim, n_kv_heads, T); we want (T, head_dim, n_kv_heads) so
    // that mul_mat(v_p, scores) contracts on T_k and produces
    // (head_dim, T_q, n_heads) — ggml broadcasts n_kv_heads → n_heads
    // on ne[2] when a.ne[2] divides b.ne[2].
    //
    // ggml_permute(a, ax0, ax1, ax2, ax3) sends input axis i to output
    // axis ax_i (i.e. result.ne[ax_i] = a.ne[i]).  To map
    // input axes (0=head_dim, 1=n_kv_heads, 2=T) onto output axes
    // (0=T, 1=head_dim, 2=n_kv_heads), the args are (1, 2, 0, 3).
    ggml_tensor * v_p = ggml_permute(ctx, v, 1, 2, 0, 3);
    v_p = ggml_cont(ctx, v_p);

    ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);
    // attn: (head_dim, T_q, n_heads)
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3);  // (head_dim, n_heads, T_q)
    attn = ggml_cont(ctx, attn);
    attn = ggml_reshape_2d(ctx, attn, head_dim * n_heads, T);

    // Output projection + residual.
    ggml_tensor * o = ggml_mul_mat(ctx, w.o, attn);
    x_ht = ggml_add(ctx, x_ht, o);

    // ── FFN (SwiGLU) ───────────────────────────────────────────────
    h = rda_rms_norm(ctx, x_ht, w.ffn_norm, rms_eps);
    ggml_tensor * gate = ggml_mul_mat(ctx, w.ffn_gate, h);
    ggml_tensor * up   = ggml_mul_mat(ctx, w.ffn_up,   h);
    ggml_tensor * mlp  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * down = ggml_mul_mat(ctx, w.ffn_down, mlp);
    x_ht = ggml_add(ctx, x_ht, down);

    return x_ht;
}

// ---------------------------------------------------------------------
// graph builders
// ---------------------------------------------------------------------

// step_begin's c0 head graph:
//   input:  t_h_in (hidden_dim,)
//   output: c0_logits (vocab_0,)  via c0_head @ t_h_in.
struct rda_c0_build {
    rda_impl * impl;
};

bool rda_build_c0(ggml_context * ctx_eval, void * ud, ggml_tensor ** out) {
    auto * b = static_cast<rda_c0_build *>(ud);
    if (!ctx_eval || !b || !b->impl || !out) return false;

    ggml_tensor * t_h = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, b->impl->hidden_dim);
    ggml_set_name(t_h, "lm.c0.h_in");

    ggml_tensor * head = codec_graph_cast_f32(ctx_eval, b->impl->c0_head);
    ggml_tensor * logits = ggml_mul_mat(ctx_eval, head, t_h);
    ggml_set_name(logits, "lm.c0.logits");
    *out = logits;
    return true;
}

// Depth-step graph: takes a prefix of `T` already-projected (depth_hidden,
// T) embeddings, runs the L Llama layers, applies head[k-1] to the
// last-position hidden, returns ck_logits.
struct rda_depth_build {
    rda_impl * impl;
    int32_t    T;            // prefix length (>= 2)
    int32_t    head_idx;     // depth_heads index = T-2 (since we predict cb at pos T-1's head)
};

bool rda_build_depth_step(ggml_context * ctx_eval, void * ud, ggml_tensor ** out) {
    auto * b = static_cast<rda_depth_build *>(ud);
    if (!ctx_eval || !b || !b->impl || !out) return false;
    rda_impl * impl = b->impl;

    // Inputs:
    //   t_x: (depth_hidden, T) — already projected through in_proj
    //        by the runtime before write.
    //   t_pos: (T,) i32 RoPE positions (0..T-1).
    ggml_tensor * t_x   = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32,
                                             impl->depth_hidden, b->T);
    ggml_set_name(t_x, "lm.depth.x");
    ggml_tensor * t_pos = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, b->T);
    ggml_set_name(t_pos, "lm.depth.pos");

    ggml_tensor * freqs = nullptr;
    if (impl->rope_freq_factors != nullptr) {
        freqs = codec_graph_cast_f32(ctx_eval, impl->rope_freq_factors);
    }

    ggml_tensor * x = t_x;
    for (int32_t l = 0; l < impl->depth_layers; ++l) {
        rda_layer_w wl = impl->layers[(size_t) l];
        // Cast each weight to F32 in the eval ctx.  codec_graph_cast_f32
        // is a no-op for already-F32 tensors and emits a `cast` node for
        // F16/quantized weights — exactly what we want.
        rda_layer_w wl_f32;
        wl_f32.attn_norm = codec_graph_cast_f32(ctx_eval, wl.attn_norm);
        wl_f32.q         = codec_graph_cast_f32(ctx_eval, wl.q);
        wl_f32.k         = codec_graph_cast_f32(ctx_eval, wl.k);
        wl_f32.v         = codec_graph_cast_f32(ctx_eval, wl.v);
        wl_f32.o         = codec_graph_cast_f32(ctx_eval, wl.o);
        wl_f32.ffn_norm  = codec_graph_cast_f32(ctx_eval, wl.ffn_norm);
        wl_f32.ffn_gate  = codec_graph_cast_f32(ctx_eval, wl.ffn_gate);
        wl_f32.ffn_up    = codec_graph_cast_f32(ctx_eval, wl.ffn_up);
        wl_f32.ffn_down  = codec_graph_cast_f32(ctx_eval, wl.ffn_down);
        wl_f32.q_norm    = (impl->has_qk_norm && wl.q_norm) ? codec_graph_cast_f32(ctx_eval, wl.q_norm) : nullptr;
        wl_f32.k_norm    = (impl->has_qk_norm && wl.k_norm) ? codec_graph_cast_f32(ctx_eval, wl.k_norm) : nullptr;

        x = rda_depth_layer(ctx_eval, x, wl_f32, t_pos, freqs,
                            impl->depth_head_dim, impl->depth_n_heads,
                            impl->depth_n_kv_heads, impl->depth_rope_theta,
                            impl->depth_rms_eps, impl->has_qk_norm);
    }

    // Final norm.
    ggml_tensor * out_norm = codec_graph_cast_f32(ctx_eval, impl->depth_output_norm);
    x = rda_rms_norm(ctx_eval, x, out_norm, impl->depth_rms_eps);

    // Take last position only, apply head[head_idx].
    ggml_tensor * h_last = ggml_view_1d(
        ctx_eval, x,
        impl->depth_hidden,
        (size_t)(b->T - 1) * impl->depth_hidden * sizeof(float));
    h_last = ggml_cont(ctx_eval, h_last);

    ggml_tensor * head_w = codec_graph_cast_f32(
        ctx_eval, impl->depth_heads[(size_t) b->head_idx]);
    ggml_tensor * logits = ggml_mul_mat(ctx_eval, head_w, h_last);
    ggml_set_name(logits, "lm.depth.ck_logits");
    *out = logits;
    return true;
}

// Flexible (Moshi) depth-step graph.  Inputs:
//   t_x_prefix: (depth_hidden, T) — token embeddings, one per depth
//               position.  Host-side builds them by indexing
//               `text_embd` at pos 0 and `audio_embd_{p-1}` at pos p>=1.
//   t_h_in:     (hidden_dim,)     — backbone hidden, shared across all
//               positions (Moshi adds in_proj[p] @ h_in to each pos).
//
// The graph:
//   - Computes per-position in_proj projection via batched mul_mat over
//     a sliced 3D weight (slice [0..T-1] of (in, out, N_max)).
//   - Adds projected h_in to the prefix.
//   - Runs L flexible-weight transformer layers (per-position q/k/v/o
//     and fc1/fc2 via batched mul_mat; shared RMSNorms; standard causal
//     attention; no RoPE; no output norm).
//   - Applies the per-position lm_head slice [T-1] to the last
//     position's hidden state, yielding (audio_vocab,) logits for cb T-1.
struct rda_flex_step_build {
    rda_impl * impl;
    int32_t    T;
};

// Slice the first T positions of a 3D weight tensor with HF shape
// (N, out, in) — ggml ne = (in, out, N).  Returns a view of (in, out, T).
static inline ggml_tensor * rda_flex_weight_slice(
        ggml_context * ctx, ggml_tensor * w_3d, int32_t T) {
    return ggml_view_3d(
        ctx, w_3d,
        w_3d->ne[0], w_3d->ne[1], (int64_t) T,
        w_3d->nb[1], w_3d->nb[2],
        /*offset=*/0);
}

// Apply per-position linear: `out[:, p] = w[p] @ x[:, p]` for p in 0..T-1.
// w has HF shape (N, out_dim, in_dim), x has (in_dim, T).
// Returns (out_dim, T) as a 2D tensor.
//
// ggml_mul_mat(a, b) requires `b.ne[2] % a.ne[2] == 0` for the batch
// dim (only `a` may broadcast); so we put the input `x` as `a` and the
// weight as `b`.  Then result ne = (M=1, N=out_dim, T), which reshapes
// contiguously to (out_dim, T).  Math:
//   result[0, n, t] = sum_k x[k, t] * w_slice[k, n, t]
//                   = sum_k x[k, t] * w_HF[t, n, k]
//                   = (w_HF[t] @ x[:, t])[n]
static inline ggml_tensor * rda_flex_apply_linear(
        ggml_context * ctx, ggml_tensor * w_3d, ggml_tensor * x_2d,
        int32_t out_dim, int32_t T) {
    ggml_tensor * w_f32   = codec_graph_cast_f32(ctx, w_3d);
    ggml_tensor * w_slice = rda_flex_weight_slice(ctx, w_f32, T);
    const int64_t in_dim  = x_2d->ne[0];
    ggml_tensor * x_3d    = ggml_reshape_3d(ctx, x_2d, in_dim, 1, (int64_t) T);
    ggml_tensor * y_3d    = ggml_mul_mat(ctx, x_3d, w_slice);
    return ggml_reshape_2d(ctx, y_3d, (int64_t) out_dim, (int64_t) T);
}

bool rda_build_depth_step_flexible(ggml_context * ctx_eval, void * ud, ggml_tensor ** out) {
    auto * b = static_cast<rda_flex_step_build *>(ud);
    if (!ctx_eval || !b || !b->impl || !out) return false;
    rda_impl * impl = b->impl;
    const int32_t T = b->T;
    if (T < 1 || T > impl->n_codebook) return false;

    // --- inputs ----------------------------------------------------------
    ggml_tensor * t_x = ggml_new_tensor_2d(
        ctx_eval, GGML_TYPE_F32, impl->depth_hidden, T);
    ggml_set_name(t_x, "lm.flex.x");
    ggml_tensor * t_h_in = ggml_new_tensor_1d(
        ctx_eval, GGML_TYPE_F32, impl->hidden_dim);
    ggml_set_name(t_h_in, "lm.flex.h_in");

    // --- in_proj per position: y[:, p] = in_proj[p] @ h_in ---------------
    // in_proj has HF shape (N, depth_hidden, hidden_dim) -> ggml
    // ne=(hidden_dim, depth_hidden, N).  Slice [0..T-1].
    //
    // h_in is constant across all depth positions, so we use mul_mat's
    // batch broadcast on `a` (a.ne[2]=1 broadcasts to b.ne[2]=T).  This
    // requires putting h_in as `a` and the weight as `b` — the same
    // operand order as `rda_flex_apply_linear` uses.
    ggml_tensor * in_proj_f32 = codec_graph_cast_f32(ctx_eval, impl->in_proj);
    ggml_tensor * in_proj_sl  = rda_flex_weight_slice(ctx_eval, in_proj_f32, T);
    ggml_tensor * h_in_3d   = ggml_reshape_3d(
        ctx_eval, t_h_in, impl->hidden_dim, 1, 1);
    // mul_mat(a=h_in_3d, b=in_proj_sl): a.ne=(H_b, 1, 1) broadcasts to
    // batch T from b.  Output (1, depth_hidden, T) -> reshape (H_d, T).
    ggml_tensor * proj_h_3d = ggml_mul_mat(ctx_eval, h_in_3d, in_proj_sl);
    ggml_tensor * proj_h    = ggml_reshape_2d(
        ctx_eval, proj_h_3d, impl->depth_hidden, T);

    // Residual stream starts at the token embeddings plus projected h_in.
    ggml_tensor * x = ggml_add(ctx_eval, t_x, proj_h);

    // --- flexible transformer layers -------------------------------------
    const float rms_eps = impl->depth_rms_eps;
    for (int32_t l = 0; l < impl->depth_layers; ++l) {
        rda_layer_w wl = impl->flex_layers[(size_t) l];

        // --- attention ---
        ggml_tensor * attn_norm_f32 = codec_graph_cast_f32(ctx_eval, wl.attn_norm);
        ggml_tensor * h_attn = rda_rms_norm(ctx_eval, x, attn_norm_f32, rms_eps);

        ggml_tensor * q_2d = rda_flex_apply_linear(
            ctx_eval, wl.q, h_attn, impl->depth_n_heads * impl->depth_head_dim, T);
        ggml_tensor * k_2d = rda_flex_apply_linear(
            ctx_eval, wl.k, h_attn, impl->depth_n_kv_heads * impl->depth_head_dim, T);
        ggml_tensor * v_2d = rda_flex_apply_linear(
            ctx_eval, wl.v, h_attn, impl->depth_n_kv_heads * impl->depth_head_dim, T);

        // (out, T) -> (head_dim, n_heads, T)
        ggml_tensor * q = ggml_reshape_3d(
            ctx_eval, q_2d, impl->depth_head_dim, impl->depth_n_heads, T);
        ggml_tensor * k = ggml_reshape_3d(
            ctx_eval, k_2d, impl->depth_head_dim, impl->depth_n_kv_heads, T);
        ggml_tensor * v = ggml_reshape_3d(
            ctx_eval, v_2d, impl->depth_head_dim, impl->depth_n_kv_heads, T);

        // No RoPE in flexible (Moshi sets use_rope=False on MoshiDecoderLayer).

        // Attention: q_p = (head_dim, T, n_heads), k_p = (head_dim, T, n_kv_heads).
        ggml_tensor * q_p = ggml_permute(ctx_eval, q, 0, 2, 1, 3);
        ggml_tensor * k_p = ggml_permute(ctx_eval, k, 0, 2, 1, 3);
        q_p = ggml_cont(ctx_eval, q_p);
        k_p = ggml_cont(ctx_eval, k_p);

        ggml_tensor * scores = ggml_mul_mat(ctx_eval, k_p, q_p);
        scores = ggml_scale(ctx_eval, scores,
                            1.0f / std::sqrt((float) impl->depth_head_dim));
        scores = ggml_diag_mask_inf(ctx_eval, scores, /*n_past=*/0);
        scores = ggml_soft_max(ctx_eval, scores);

        ggml_tensor * v_p = ggml_permute(ctx_eval, v, 1, 2, 0, 3);
        v_p = ggml_cont(ctx_eval, v_p);
        ggml_tensor * attn = ggml_mul_mat(ctx_eval, v_p, scores);
        attn = ggml_permute(ctx_eval, attn, 0, 2, 1, 3);
        attn = ggml_cont(ctx_eval, attn);
        attn = ggml_reshape_2d(
            ctx_eval, attn, impl->depth_head_dim * impl->depth_n_heads, T);

        ggml_tensor * o = rda_flex_apply_linear(
            ctx_eval, wl.o, attn, impl->depth_hidden, T);
        x = ggml_add(ctx_eval, x, o);

        // --- FFN (SwiGLU) ---
        ggml_tensor * ffn_norm_f32 = codec_graph_cast_f32(ctx_eval, wl.ffn_norm);
        ggml_tensor * h_ffn = rda_rms_norm(ctx_eval, x, ffn_norm_f32, rms_eps);

        ggml_tensor * gate = rda_flex_apply_linear(
            ctx_eval, wl.ffn_gate, h_ffn, impl->depth_inter, T);
        ggml_tensor * up   = rda_flex_apply_linear(
            ctx_eval, wl.ffn_up,   h_ffn, impl->depth_inter, T);
        ggml_tensor * act  = ggml_mul(ctx_eval, ggml_silu(ctx_eval, gate), up);
        ggml_tensor * down = rda_flex_apply_linear(
            ctx_eval, wl.ffn_down, act, impl->depth_hidden, T);
        x = ggml_add(ctx_eval, x, down);
    }

    // No final norm — Moshi applies lm_heads directly to the last
    // layer's residual stream.
    if (impl->has_output_norm && impl->depth_output_norm != nullptr) {
        ggml_tensor * onorm_f32 = codec_graph_cast_f32(ctx_eval, impl->depth_output_norm);
        x = rda_rms_norm(ctx_eval, x, onorm_f32, rms_eps);
    }

    // --- lm_heads slice [T-1] at the last position -----------------------
    // flex_heads HF shape (N, audio_vocab, depth_hidden) -> ggml
    // ne=(depth_hidden, audio_vocab, N).  We need slice index T-1.
    ggml_tensor * heads_f32 = codec_graph_cast_f32(ctx_eval, impl->flex_heads);
    ggml_tensor * head_slice = ggml_view_2d(
        ctx_eval, heads_f32,
        heads_f32->ne[0], heads_f32->ne[1],
        heads_f32->nb[1],
        (size_t) (T - 1) * heads_f32->nb[2]);

    // x at last position (depth_hidden,)
    ggml_tensor * x_last = ggml_view_1d(
        ctx_eval, x, impl->depth_hidden,
        (size_t) (T - 1) * impl->depth_hidden * sizeof(float));
    x_last = ggml_cont(ctx_eval, x_last);

    ggml_tensor * logits = ggml_mul_mat(ctx_eval, head_slice, x_last);
    ggml_set_name(logits, "lm.flex.ck_logits");
    *out = logits;
    return true;
}

// compose_audio_embd graph: get_rows on each cb's table, sum.  Same
// pattern as parallel_heads_delay's compose graph.
struct rda_compose_build {
    rda_impl * impl;
};

bool rda_build_compose(ggml_context * ctx_eval, void * ud, ggml_tensor ** out) {
    auto * b = static_cast<rda_compose_build *>(ud);
    if (!ctx_eval || !b || !b->impl || !out) return false;
    rda_impl * impl = b->impl;

    ggml_tensor * t_codes = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, impl->n_codebook);
    ggml_set_name(t_codes, "lm.compose.codes");

    ggml_tensor * acc = nullptr;
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        ggml_tensor * embd = impl->audio_embds[(size_t) i];
        ggml_tensor * idx_view = ggml_view_1d(
            ctx_eval, t_codes, /*ne0=*/1, (size_t) i * sizeof(int32_t));
        ggml_tensor * row = ggml_get_rows(ctx_eval, embd, idx_view);
        row = codec_graph_cast_f32(ctx_eval, row);
        acc = (acc == nullptr) ? row : ggml_add(ctx_eval, acc, row);
    }
    if (acc == nullptr) return false;
    ggml_set_name(acc, "lm.compose.out");
    *out = acc;
    return true;
}

// ---------------------------------------------------------------------
// init / free
// ---------------------------------------------------------------------

static ggml_tensor * find_required(codec_lm * lm, const char * name) {
    ggml_tensor * t = ggml_get_tensor(lm->codec->weights, name);
    if (t == nullptr) {
        lm->last_error = std::string("missing tensor: ") + name;
    }
    return t;
}

bool init(codec_lm * lm) {
    if (!lm || !lm->codec || !lm->codec->weights || !lm->codec->gguf) return false;
    gguf_context * gf = lm->codec->gguf;

    rda_impl * impl = new (std::nothrow) rda_impl();
    if (!impl) { lm->last_error = "out of memory"; return false; }

    impl->n_codebook       = lm->info.n_codebook;
    impl->hidden_dim       = lm->info.hidden_dim;
    impl->audio_embed_dim  = lm->info.audio_embed_dim;
    impl->depth_layers     = codec_read_i32_kv(gf, "codec.lm.residual.depth_layers",     0);
    impl->depth_hidden     = codec_read_i32_kv(gf, "codec.lm.residual.depth_hidden",     0);
    impl->depth_n_heads    = codec_read_i32_kv(gf, "codec.lm.residual.depth_n_heads",    0);
    impl->depth_n_kv_heads = codec_read_i32_kv(gf, "codec.lm.residual.depth_n_kv_heads", 0);
    impl->depth_head_dim   = codec_read_i32_kv(gf, "codec.lm.residual.depth_head_dim",   0);
    impl->depth_inter      = codec_read_i32_kv(gf, "codec.lm.residual.depth_intermediate", 0);
    impl->depth_max_pos    = codec_read_i32_kv(gf, "codec.lm.residual.depth_max_position", 0);
    impl->depth_rope_theta = codec_read_f32_kv(gf, "codec.lm.residual.depth_rope_theta", 10000.0f);
    impl->depth_rms_eps    = codec_read_f32_kv(gf, "codec.lm.residual.depth_rms_norm_eps", 1e-5f);
    impl->has_in_proj      = codec_read_bool_kv(gf, "codec.lm.residual.depth_has_in_proj", false);
    impl->has_qk_norm      = codec_read_bool_kv(gf, "codec.lm.residual.depth_has_qk_norm", false);
    impl->has_output_norm  = codec_read_bool_kv(gf, "codec.lm.residual.depth_has_output_norm", true);
    impl->use_rope         = codec_read_bool_kv(gf, "codec.lm.residual.depth_use_rope", true);
    impl->depth_text_vocab = codec_read_i32_kv (gf, "codec.lm.residual.depth_text_vocab", 0);

    // weight_layout dispatch.  "shared" (CSM, Qwen3-TTS): one transformer
    // reused at every depth position.  "flexible" (Moshi): N transformer
    // weight tensors per layer, gathered by depth position.
    {
        std::string wl = codec_lm_read_string_kv(
            lm->codec, "codec.lm.residual.weight_layout");
        impl->is_flexible = (wl == "flexible");
    }
    {
        std::string m = codec_lm_read_string_kv(
            lm->codec, "codec.lm.residual.c0_input_modality");
        impl->c0_is_text = (m == "text");
    }

    if (impl->depth_layers <= 0 || impl->depth_hidden <= 0 ||
        impl->depth_n_heads <= 0 || impl->depth_head_dim <= 0 ||
        impl->depth_inter <= 0) {
        lm->last_error = "invalid residual_depth_ar metadata: zero/negative dims";
        delete impl;
        return false;
    }

    char buf[80];

    if (!impl->is_flexible) {
        // ---- shared mode (CSM, Qwen3-TTS) ------------------------------
        // Audio embedding tables (per-cb).
        impl->audio_embds.resize((size_t) impl->n_codebook, nullptr);
        for (int32_t i = 0; i < impl->n_codebook; ++i) {
            std::snprintf(buf, sizeof(buf), "lm.audio_embd_%d.weight", i);
            impl->audio_embds[(size_t) i] = find_required(lm, buf);
            if (impl->audio_embds[(size_t) i] == nullptr) { delete impl; return false; }
        }

        impl->c0_head = find_required(lm, "lm.c0_head.weight");
        if (!impl->c0_head) { delete impl; return false; }

        // Depth heads (n_codebook - 1).
        impl->depth_heads.resize((size_t) (impl->n_codebook - 1), nullptr);
        for (int32_t i = 0; i < impl->n_codebook - 1; ++i) {
            std::snprintf(buf, sizeof(buf), "lm.depth.heads_%d.weight", i);
            impl->depth_heads[(size_t) i] = find_required(lm, buf);
            if (impl->depth_heads[(size_t) i] == nullptr) { delete impl; return false; }
        }
    } else {
        // ---- flexible mode (Moshi) -------------------------------------
        // Depth pos 0 input table.  Moshi: text vocab.
        if (impl->c0_is_text) {
            impl->text_embd = find_required(lm, "lm.depth.text_embd.weight");
            if (!impl->text_embd) { delete impl; return false; }
        }

        // Depth pos 1..N-1 input tables — N-1 audio embed tables.
        // (The last codebook is never an input.)
        impl->audio_embds.resize((size_t) (impl->n_codebook - 1), nullptr);
        for (int32_t i = 0; i < impl->n_codebook - 1; ++i) {
            std::snprintf(buf, sizeof(buf), "lm.depth.audio_embd_%d.weight", i);
            impl->audio_embds[(size_t) i] = find_required(lm, buf);
            if (impl->audio_embds[(size_t) i] == nullptr) { delete impl; return false; }
        }

        // Single 3D heads tensor; slice[p] applied at depth position p.
        impl->flex_heads = find_required(lm, "lm.depth.heads.weight");
        if (!impl->flex_heads) { delete impl; return false; }
    }

    if (impl->has_in_proj) {
        impl->in_proj = find_required(lm, "lm.depth.in_proj.weight");
        if (!impl->in_proj) { delete impl; return false; }
        // Optional bias (Qwen3-TTS's `small_to_mtp_projection` has bias=True;
        // CSM's `inputs_embeds_projector` doesn't).  Look it up but don't
        // require it.  Not used in flexible mode (Moshi's input_projections
        // has no bias either).
        if (!impl->is_flexible) {
            impl->in_proj_bias = ggml_get_tensor(lm->codec->weights, "lm.depth.in_proj.bias");
        }
    } else {
        if (impl->is_flexible) {
            lm->last_error = "flexible weight_layout requires depth_has_in_proj=true";
            delete impl; return false;
        }
        // Identity in_proj (e.g. Qwen3-TTS-0.6B-Base where talker hidden
        // equals depth_hidden = 1024).  Requires hidden_dim == depth_hidden
        // AND audio_embed_dim == depth_hidden so the raw vectors land in
        // the depth-decoder input space directly.
        if (impl->hidden_dim != impl->depth_hidden ||
            impl->audio_embed_dim != impl->depth_hidden) {
            char buf2[160];
            std::snprintf(buf2, sizeof(buf2),
                "depth_has_in_proj=false requires hidden_dim==depth_hidden=="
                "audio_embed_dim, got hidden=%d depth_hidden=%d audio_embed=%d",
                impl->hidden_dim, impl->depth_hidden, impl->audio_embed_dim);
            lm->last_error = buf2;
            delete impl; return false;
        }
    }

    if (impl->has_output_norm) {
        impl->depth_output_norm = find_required(lm, "lm.depth.output_norm.weight");
        if (!impl->depth_output_norm) { delete impl; return false; }
    }

    // Optional llama3 RoPE freq factors (shared-mode only).
    if (!impl->is_flexible) {
        impl->rope_freq_factors = ggml_get_tensor(lm->codec->weights, "lm.depth.rope_freq_factors");
    }

    // Per-layer weights — same tensor names, different shapes (2D vs 3D)
    // between shared / flexible.
    std::vector<rda_layer_w> & layers_out = impl->is_flexible ? impl->flex_layers : impl->layers;
    layers_out.resize((size_t) impl->depth_layers);
    for (int32_t l = 0; l < impl->depth_layers; ++l) {
        rda_layer_w & w = layers_out[(size_t) l];
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.attn_norm.weight", l);
        w.attn_norm = find_required(lm, buf); if (!w.attn_norm) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.q.weight", l);
        w.q = find_required(lm, buf); if (!w.q) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.k.weight", l);
        w.k = find_required(lm, buf); if (!w.k) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.v.weight", l);
        w.v = find_required(lm, buf); if (!w.v) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.o.weight", l);
        w.o = find_required(lm, buf); if (!w.o) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.ffn_norm.weight", l);
        w.ffn_norm = find_required(lm, buf); if (!w.ffn_norm) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.ffn_gate.weight", l);
        w.ffn_gate = find_required(lm, buf); if (!w.ffn_gate) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.ffn_up.weight", l);
        w.ffn_up = find_required(lm, buf); if (!w.ffn_up) { delete impl; return false; }
        std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.ffn_down.weight", l);
        w.ffn_down = find_required(lm, buf); if (!w.ffn_down) { delete impl; return false; }
        if (impl->has_qk_norm) {
            std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.q_norm.weight", l);
            w.q_norm = find_required(lm, buf); if (!w.q_norm) { delete impl; return false; }
            std::snprintf(buf, sizeof(buf), "lm.depth.blk_%d.k_norm.weight", l);
            w.k_norm = find_required(lm, buf); if (!w.k_norm) { delete impl; return false; }
        } else {
            w.q_norm = nullptr;
            w.k_norm = nullptr;
        }
    }

    lm->impl = impl;
    return true;
}

void free_lm(codec_lm * lm) {
    if (!lm || !lm->impl) return;
    rda_impl * impl = static_cast<rda_impl *>(lm->impl);
    if (impl->compose_ctx != nullptr) {
        codec_runtime_free(impl->compose_ctx);
        delete impl->compose_ctx;
    }
    delete impl;
    lm->impl = nullptr;
}

bool state_init(codec_lm_state * st) {
    if (!st || !st->lm || !st->lm->impl) return false;
    rda_impl * impl = static_cast<rda_impl *>(st->lm->impl);

    rda_state * sst = new (std::nothrow) rda_state();
    if (!sst) return false;
    sst->h_in_buf.resize((size_t) impl->hidden_dim);
    sst->logits_buf.resize((size_t) impl->n_codebook);
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        sst->logits_buf[(size_t) i].resize((size_t) st->lm->info.codebook_sizes[i]);
    }
    st->impl = sst;
    return true;
}

void state_free(codec_lm_state * st) {
    if (!st || !st->impl) return;
    delete static_cast<rda_state *>(st->impl);
    st->impl = nullptr;
}

void state_reset(codec_lm_state * /*st*/) {
    // No KV cache to clear in the prefix-recompute regime; logits buffers
    // are overwritten by step_begin/step_logits anyway.
}

// ---------------------------------------------------------------------
// step machine
// ---------------------------------------------------------------------

// Run the c0 head over `h_in` and copy logits into the state's scratch.
enum codec_status run_c0_head(codec_lm_state * st, const float * h_in) {
    rda_impl * impl = static_cast<rda_impl *>(st->lm->impl);
    rda_state * sst = static_cast<rda_state *>(st->impl);

    rda_c0_build build = { impl };
    codec_graph_eval_guard guard(st->ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = (int32_t) CODEC_GRAPH_LM_RDA_C0_HEAD;
    key.n_in = impl->hidden_dim;

    if (!codec_graph_cache_get_or_build(
            st->ctx, key, rda_build_c0, &build, sizeof(build), &entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_h = codec_graph_get_tensor(st->ctx, entry, "lm.c0.h_in");
    ggml_tensor * t_lg = codec_graph_get_tensor(st->ctx, entry, "lm.c0.logits");
    if (!t_h || !t_lg) {
        st->last_error = "c0 graph missing tensors"; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_h, h_in, (size_t) impl->hidden_dim * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_threads = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, n_threads, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t v0 = st->lm->info.codebook_sizes[0];
    if (!codec_runtime_read_tensor(t_lg, sst->logits_buf[0].data(),
                                   (size_t) v0 * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}

// Compose the depth decoder's prefix embeddings (length T = current_k+1)
// and run the depth-step graph to produce ck_logits where ck = current_k.
// `current_k` is the codebook index whose logits we want — must be >= 1.
enum codec_status run_depth_step(codec_lm_state * st, int32_t current_k) {
    rda_impl * impl = static_cast<rda_impl *>(st->lm->impl);
    rda_state * sst = static_cast<rda_state *>(st->impl);
    if (current_k < 1 || current_k >= impl->n_codebook) {
        st->last_error = "depth step called with out-of-range cb"; return CODEC_STATUS_INVALID_STATE;
    }
    const int32_t T = current_k + 1;   // [h, c0, c1, ..., c_{k-1}] → T = k+1 positions.

    // Build the prefix on the host side: h_proj (in_proj @ h_in), then
    // for each previously-pushed code j, in_proj @ audio_embd_j[code_j].
    // This is a small CPU-side compute (one matmul per prefix entry against
    // a 2048-row table → 1024-vector); doing it inline keeps the graph
    // small.  We store the result as a flat (depth_hidden, T) buffer
    // ready to copy into the graph's input tensor.
    std::vector<float> x_prefix((size_t) impl->depth_hidden * (size_t) T, 0.0f);

    // Helper: project an arbitrary [hidden_dim] (or [audio_embed_dim]) vector
    // through in_proj into [depth_hidden].  When in_proj is absent
    // (`has_in_proj=false`, e.g. Qwen3-TTS-0.6B where talker hidden ==
    // depth_hidden), this is a plain memcpy — init() guarantees the dims
    // match.  When in_proj has a bias (Qwen3-TTS's `small_to_mtp_projection`),
    // it's added after the matmul.
    auto project = [&](const float * src, float * dst) -> enum codec_status {
        if (impl->in_proj == nullptr) {
            // Identity projection — caller's input is already in the
            // depth_hidden space (audio_embed_dim == depth_hidden checked
            // at init()).
            std::memcpy(dst, src, (size_t) impl->depth_hidden * sizeof(float));
            return CODEC_STATUS_SUCCESS;
        }
        // Cache key uses CODEC_GRAPH_LM_RDA_IN_PROJ + hidden_dim as
        // discriminators; same graph regardless of layer or step.
        struct in_proj_build { rda_impl * impl; };
        in_proj_build b = { impl };
        codec_graph_eval_guard guard(st->ctx);
        std::string err;
        codec_graph_cache_entry * entry = nullptr;
        codec_graph_cache_key key = {};
        key.kind = (int32_t) CODEC_GRAPH_LM_RDA_IN_PROJ;
        key.n_in = impl->hidden_dim;
        auto build_fn = [](ggml_context * ctx_e, void * ud, ggml_tensor ** out) -> bool {
            auto * bb = static_cast<in_proj_build *>(ud);
            ggml_tensor * t_in = ggml_new_tensor_1d(ctx_e, GGML_TYPE_F32, bb->impl->hidden_dim);
            ggml_set_name(t_in, "lm.in_proj.in");
            ggml_tensor * w = codec_graph_cast_f32(ctx_e, bb->impl->in_proj);
            ggml_tensor * y = ggml_mul_mat(ctx_e, w, t_in);
            if (bb->impl->in_proj_bias != nullptr) {
                ggml_tensor * bv = codec_graph_cast_f32(ctx_e, bb->impl->in_proj_bias);
                y = ggml_add(ctx_e, y, bv);
            }
            ggml_set_name(y, "lm.in_proj.out");
            *out = y;
            return true;
        };
        if (!codec_graph_cache_get_or_build(
                st->ctx, key, build_fn, &b, sizeof(b), &entry, &err)) {
            st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
        }
        ggml_tensor * t_in  = codec_graph_get_tensor(st->ctx, entry, "lm.in_proj.in");
        ggml_tensor * t_out = codec_graph_get_tensor(st->ctx, entry, "lm.in_proj.out");
        if (!codec_graph_prepare_io(st->ctx, entry, &err)) {
            st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
        }
        if (!codec_runtime_write_tensor(t_in, src,
                                        (size_t) impl->hidden_dim * sizeof(float), &err)) {
            st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
        }
        const int32_t nt = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
        if (!codec_graph_compute(st->ctx, entry, nt, &err)) {
            st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
        }
        if (!codec_runtime_read_tensor(t_out, dst,
                                       (size_t) impl->depth_hidden * sizeof(float), &err)) {
            st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
        }
        return CODEC_STATUS_SUCCESS;
    };

    // Position 0: in_proj(h_in_buf).
    enum codec_status rc = project(sst->h_in_buf.data(), x_prefix.data());
    if (rc != CODEC_STATUS_SUCCESS) return rc;

    // Positions 1..T-1: in_proj(audio_embd_{j}[codes[j]]) for j = 0..T-2.
    // We need each audio_embd row as a CPU-side float buffer.
    std::vector<float> embd_row((size_t) impl->audio_embed_dim);
    for (int32_t j = 0; j < T - 1; ++j) {
        ggml_tensor * embd_t = impl->audio_embds[(size_t) j];
        const int32_t code   = st->codes_buf[(size_t) j];
        if (code < 0 || code >= st->lm->info.codebook_sizes[j]) {
            st->last_error = "depth: code out of range"; return CODEC_STATUS_INVALID_STATE;
        }
        const size_t row_bytes = (size_t) impl->audio_embed_dim * sizeof(float);
        // ggml_backend_tensor_get reads a slice; row j is at offset
        // code * audio_embed_dim * dtype_size.  But the tensor may be F16
        // — use codec_tensor_data_f32 for F32-host tensors, otherwise
        // convert via codec_tensor_as_vec_f32 (slower; full-table dequant).
        if (embd_t->type == GGML_TYPE_F32 &&
            (embd_t->buffer == nullptr || ggml_backend_buffer_is_host(embd_t->buffer))) {
            const float * data = static_cast<const float *>(ggml_get_data(embd_t));
            std::memcpy(embd_row.data(),
                        data + (size_t) code * impl->audio_embed_dim,
                        row_bytes);
        } else {
            // Convert the entire table once via the helper (thread_local
            // cached).  For F16 weights this is quick; for quantized
            // weights it dequants fully — fine for a small per-cb table.
            std::vector<float> all;
            if (!codec_tensor_as_vec_f32(embd_t, &all)) {
                st->last_error = "audio embd dequant failed";
                return CODEC_STATUS_INTERNAL_ERROR;
            }
            std::memcpy(embd_row.data(),
                        all.data() + (size_t) code * impl->audio_embed_dim,
                        row_bytes);
        }
        rc = project(embd_row.data(),
                     x_prefix.data() + (size_t)(j + 1) * impl->depth_hidden);
        if (rc != CODEC_STATUS_SUCCESS) return rc;
    }

    // Now run the depth step graph.
    rda_depth_build build = { impl, T, current_k - 1 };
    codec_graph_eval_guard guard(st->ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind     = (int32_t) CODEC_GRAPH_LM_RDA_DEPTH_STEP;
    key.n_frames = T;
    key.n_q      = current_k - 1;   // pin per-head index — each head has its own weight
    if (!codec_graph_cache_get_or_build(
            st->ctx, key, rda_build_depth_step, &build, sizeof(build), &entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_x   = codec_graph_get_tensor(st->ctx, entry, "lm.depth.x");
    ggml_tensor * t_pos = codec_graph_get_tensor(st->ctx, entry, "lm.depth.pos");
    ggml_tensor * t_lg  = codec_graph_get_tensor(st->ctx, entry, "lm.depth.ck_logits");
    if (!t_x || !t_pos || !t_lg) {
        st->last_error = "depth graph missing tensors"; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_x, x_prefix.data(),
                                    x_prefix.size() * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<int32_t> positions((size_t) T);
    for (int32_t i = 0; i < T; ++i) positions[(size_t) i] = i;
    if (!codec_runtime_write_tensor(t_pos, positions.data(),
                                    positions.size() * sizeof(int32_t), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t nt = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, nt, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t vk = st->lm->info.codebook_sizes[current_k];
    if (!codec_runtime_read_tensor(t_lg, sst->logits_buf[(size_t) current_k].data(),
                                   (size_t) vk * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}

// Flexible (Moshi) depth step.  Builds a (depth_hidden, T) token-
// embedding prefix on the host, runs the flexible graph (which adds
// per-position in_proj(h_in) and runs L flexible-weight layers + the
// per-position head slice), and writes the ck logits into the state.
//
// `current_k` is the cb index whose logits we want (0..N-1).  Prefix
// length is T = current_k + 1.  At depth pos 0 the input embedding is
// `text_embd[text_token_context]` (the caller must have stashed
// text_token_context via `codec_lm_state_set_text_context`).  At depth
// pos p>=1 the input is `audio_embds[p-1][codes_buf[p-1]]`.
enum codec_status run_depth_step_flexible(codec_lm_state * st, int32_t current_k) {
    rda_impl  * impl = static_cast<rda_impl  *>(st->lm->impl);
    rda_state * sst  = static_cast<rda_state *>(st->impl);
    if (current_k < 0 || current_k >= impl->n_codebook) {
        st->last_error = "flex depth step: cb index out of range";
        return CODEC_STATUS_INVALID_STATE;
    }
    const int32_t T = current_k + 1;

    if (impl->c0_is_text && st->text_token_context < 0) {
        st->last_error =
            "flex depth step: c0_input_modality=text but no text context "
            "set; call codec_lm_state_set_text_context before step_begin";
        return CODEC_STATUS_INVALID_STATE;
    }
    if (impl->c0_is_text && impl->depth_text_vocab > 0 &&
        st->text_token_context >= impl->depth_text_vocab + 1) {
        // text_embd table has `vocab + 1` rows (Moshi pads with one row);
        // accept anything in [0, vocab].  Negative was checked above.
        st->last_error = "flex depth step: text_token_context out of range";
        return CODEC_STATUS_INVALID_STATE;
    }

    // ---- Build the (depth_hidden, T) prefix host-side ------------------
    std::vector<float> x_prefix((size_t) impl->depth_hidden * (size_t) T, 0.0f);

    auto copy_embd_row = [&](ggml_tensor * tbl, int32_t row_id, float * dst,
                             const char * tag) -> bool {
        if (tbl == nullptr) {
            st->last_error = std::string("flex depth step: missing embed table for ") + tag;
            return false;
        }
        const int64_t n_rows  = tbl->ne[1];
        if (row_id < 0 || (int64_t) row_id >= n_rows) {
            st->last_error = std::string("flex depth step: row id out of range for ") + tag;
            return false;
        }
        const size_t row_bytes = (size_t) impl->depth_hidden * sizeof(float);
        if (tbl->type == GGML_TYPE_F32 &&
            (tbl->buffer == nullptr || ggml_backend_buffer_is_host(tbl->buffer))) {
            const float * data = static_cast<const float *>(ggml_get_data(tbl));
            std::memcpy(dst, data + (size_t) row_id * impl->depth_hidden, row_bytes);
        } else {
            std::vector<float> all;
            if (!codec_tensor_as_vec_f32(tbl, &all)) {
                st->last_error = std::string("flex depth step: dequant failed for ") + tag;
                return false;
            }
            std::memcpy(dst, all.data() + (size_t) row_id * impl->depth_hidden, row_bytes);
        }
        return true;
    };

    // Position 0: token embedding for the c0_input_modality input.
    if (impl->c0_is_text) {
        if (!copy_embd_row(impl->text_embd, st->text_token_context,
                            x_prefix.data(), "text_embd@pos0")) {
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    } else {
        // Flexible + audio c0: pos 0 input is the c0 code itself (unused
        // by Moshi; included for completeness).  This requires N audio
        // embed tables when c0_is_text=false.
        if (impl->audio_embds.empty() || impl->audio_embds[0] == nullptr) {
            st->last_error =
                "flex depth step: c0_input_modality=audio not supported "
                "(no model exercises this path yet)";
            return CODEC_STATUS_INVALID_STATE;
        }
        if (!copy_embd_row(impl->audio_embds[0], st->codes_buf[0],
                            x_prefix.data(), "audio_embd@pos0")) {
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    // Positions 1..T-1: audio_embd_{p-1}[codes_buf[p-1]].
    for (int32_t p = 1; p < T; ++p) {
        const int32_t embd_idx = p - 1;
        if ((size_t) embd_idx >= impl->audio_embds.size() ||
            impl->audio_embds[(size_t) embd_idx] == nullptr) {
            st->last_error =
                "flex depth step: missing audio_embd for prefix position";
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        if (!copy_embd_row(impl->audio_embds[(size_t) embd_idx],
                            st->codes_buf[(size_t) embd_idx],
                            x_prefix.data() + (size_t) p * impl->depth_hidden,
                            "audio_embd")) {
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    // ---- Run the flexible depth graph ----------------------------------
    rda_flex_step_build build = { impl, T };
    codec_graph_eval_guard guard(st->ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind     = (int32_t) CODEC_GRAPH_LM_RDA_FLEX_STEP;
    key.n_frames = T;

    if (!codec_graph_cache_get_or_build(
            st->ctx, key, rda_build_depth_step_flexible,
            &build, sizeof(build), &entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_x    = codec_graph_get_tensor(st->ctx, entry, "lm.flex.x");
    ggml_tensor * t_h_in = codec_graph_get_tensor(st->ctx, entry, "lm.flex.h_in");
    ggml_tensor * t_lg   = codec_graph_get_tensor(st->ctx, entry, "lm.flex.ck_logits");
    if (!t_x || !t_h_in || !t_lg) {
        st->last_error = "flex depth graph missing tensors";
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_x, x_prefix.data(),
                                    x_prefix.size() * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_h_in, sst->h_in_buf.data(),
                                    (size_t) impl->hidden_dim * sizeof(float),
                                    &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t nt = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, nt, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t vk = st->lm->info.codebook_sizes[current_k];
    if (!codec_runtime_read_tensor(t_lg, sst->logits_buf[(size_t) current_k].data(),
                                   (size_t) vk * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}

enum codec_status step_begin(codec_lm_state * st, const float * h_in) {
    if (!st || !h_in) return CODEC_STATUS_INVALID_ARG;
    rda_impl * impl = static_cast<rda_impl *>(st->lm->impl);
    rda_state * sst = static_cast<rda_state *>(st->impl);
    std::memcpy(sst->h_in_buf.data(), h_in, (size_t) impl->hidden_dim * sizeof(float));
    if (impl->is_flexible) {
        // Flexible models compute c0 inside the depth decoder (no separate
        // c0_head).  Defer the actual graph until step_logits(0), which
        // builds a T=1 prefix and runs the flexible graph.
        return CODEC_STATUS_SUCCESS;
    }
    return run_c0_head(st, h_in);
}

bool step_pending(const codec_lm_state * st) {
    return st != nullptr && st->next_cb < st->lm->info.n_codebook;
}

const float * step_logits(codec_lm_state * st, int32_t * out_cb_idx, int32_t * out_n) {
    if (!st || !st->impl) return nullptr;
    rda_impl  * impl = static_cast<rda_impl  *>(st->lm->impl);
    rda_state * sst  = static_cast<rda_state *>(st->impl);
    const int32_t k = st->next_cb;
    if (k >= st->lm->info.n_codebook) return nullptr;

    if (impl->is_flexible) {
        // All N codebooks come from the depth decoder in flexible mode
        // (Moshi has no separate c0_head; lm_heads[0] predicts c0).
        if (run_depth_step_flexible(st, k) != CODEC_STATUS_SUCCESS) {
            return nullptr;
        }
    } else if (k >= 1) {
        // Shared mode: c0 was computed at step_begin via c0_head;
        // c1..c_{N-1} come from the shared depth decoder.
        if (run_depth_step(st, k) != CODEC_STATUS_SUCCESS) {
            return nullptr;
        }
    }
    if (out_cb_idx) *out_cb_idx = k;
    if (out_n)      *out_n      = st->lm->info.codebook_sizes[k];
    return sst->logits_buf[(size_t) k].data();
}

enum codec_status step_push_code(codec_lm_state * /*st*/, int32_t /*code*/) {
    // Generic dispatch records code into st->codes_buf[k]; nothing
    // kind-specific here in the prefix-recompute regime.  The next
    // step_logits call rebuilds the depth prefix from those codes.
    return CODEC_STATUS_SUCCESS;
}

enum codec_status step_finish(codec_lm_state * st, int32_t * out_codes) {
    if (!st || !out_codes) return CODEC_STATUS_INVALID_ARG;
    std::memcpy(out_codes, st->codes_buf.data(),
                (size_t) st->lm->info.n_codebook * sizeof(int32_t));
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// audio embd
// ---------------------------------------------------------------------

const float * audio_embd(codec_lm * lm, int32_t cb_idx, int32_t code) {
    rda_impl * impl = static_cast<rda_impl *>(lm->impl);
    if (!impl) return nullptr;
    // Vector size depends on mode: shared = N, flexible = N-1 (the
    // last codebook is never an input under Moshi's flexible layout).
    if (cb_idx < 0 || (size_t) cb_idx >= impl->audio_embds.size()) return nullptr;
    ggml_tensor * t = impl->audio_embds[(size_t) cb_idx];
    if (t == nullptr) return nullptr;
    const float * data = codec_tensor_data_f32(t);
    if (!data) return nullptr;
    return data + (size_t) code * impl->audio_embed_dim;
}

enum codec_status compose_audio_embd(codec_lm * lm, const int32_t * codes, float * out_embd) {
    if (!lm || !lm->impl || !codes || !out_embd) return CODEC_STATUS_INVALID_ARG;
    rda_impl * impl = static_cast<rda_impl *>(lm->impl);
    if (impl->is_flexible) {
        // Flexible models (Moshi) compose the next-step backbone input
        // out-of-band: the backbone has its own dual-stream embedding
        // tables (model + user audio + text) and the caller owns the
        // sum.  codec_lm doesn't have the user-side audio tables here.
        lm->last_error =
            "compose_audio_embd: not supported in flexible weight_layout "
            "(caller composes backbone input directly)";
        return CODEC_STATUS_INVALID_STATE;
    }
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        if (codes[i] < 0 || codes[i] >= lm->info.codebook_sizes[i]) {
            return CODEC_STATUS_INVALID_ARG;
        }
    }
    if (impl->compose_ctx == nullptr) {
        codec_context * cctx = new (std::nothrow) codec_context();
        if (!cctx) return CODEC_STATUS_INTERNAL_ERROR;
        cctx->model   = lm->codec;
        cctx->backend = lm->codec->backend;
        cctx->params  = codec_context_default_params();
        std::string err;
        if (!codec_runtime_init(cctx, &err)) {
            delete cctx; lm->last_error = err;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        impl->compose_ctx = cctx;
    }

    rda_compose_build build = { impl };
    codec_graph_eval_guard guard(impl->compose_ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = (int32_t) CODEC_GRAPH_LM_RDA_COMPOSE;
    key.n_q  = impl->n_codebook;
    key.n_in = impl->audio_embed_dim;
    if (!codec_graph_cache_get_or_build(
            impl->compose_ctx, key, rda_build_compose, &build, sizeof(build),
            &entry, &err)) {
        lm->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_codes = codec_graph_get_tensor(impl->compose_ctx, entry, "lm.compose.codes");
    ggml_tensor * t_out   = codec_graph_get_tensor(impl->compose_ctx, entry, "lm.compose.out");
    if (!codec_graph_prepare_io(impl->compose_ctx, entry, &err)) {
        lm->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_codes, codes,
                                    (size_t) impl->n_codebook * sizeof(int32_t), &err)) {
        lm->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t nt = lm->codec->n_threads > 0 ? lm->codec->n_threads : 1;
    if (!codec_graph_compute(impl->compose_ctx, entry, nt, &err)) {
        lm->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_read_tensor(t_out, out_embd,
                                   (size_t) impl->audio_embed_dim * sizeof(float), &err)) {
        lm->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}

}  // namespace

const codec_lm_kind_vtable codec_lm_vtable_residual_depth_ar = {
    /*.kind               =*/ CODEC_LM_KIND_RESIDUAL_DEPTH_AR,
    /*.name               =*/ "residual_depth_ar",
    /*.init               =*/ init,
    /*.free               =*/ free_lm,
    /*.state_init         =*/ state_init,
    /*.state_free         =*/ state_free,
    /*.state_reset        =*/ state_reset,
    /*.step_begin         =*/ step_begin,
    /*.step_pending       =*/ step_pending,
    /*.step_logits        =*/ step_logits,
    /*.step_push_code     =*/ step_push_code,
    /*.step_finish        =*/ step_finish,
    /*.audio_embd         =*/ audio_embd,
    /*.compose_audio_embd =*/ compose_audio_embd,
};
