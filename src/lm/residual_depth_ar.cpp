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

    // Tensor handles (live in codec->weights).
    std::vector<ggml_tensor *> audio_embds;   // [n_codebook]
    ggml_tensor * c0_head           = nullptr;
    std::vector<ggml_tensor *> depth_heads;   // [n_codebook-1]
    ggml_tensor * in_proj           = nullptr; // optional
    ggml_tensor * depth_output_norm = nullptr;
    ggml_tensor * rope_freq_factors = nullptr; // optional (llama3 scaling)
    std::vector<rda_layer_w> layers;          // [depth_layers]

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

    // RoPE on q and k along the head_dim axis.  Mode is NORMAL (Llama
    // uses interleaved cos/sin pairs, which is the GGML "normal" mode
    // — NEOX is the half-rotation variant).
    const int32_t rope_n_dims = head_dim;     // rotate the full head dim
    const int32_t rope_mode   = GGML_ROPE_TYPE_NORMAL;
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

    if (impl->depth_layers <= 0 || impl->depth_hidden <= 0 ||
        impl->depth_n_heads <= 0 || impl->depth_head_dim <= 0 ||
        impl->depth_inter <= 0) {
        lm->last_error = "invalid residual_depth_ar metadata: zero/negative dims";
        delete impl;
        return false;
    }

    // Audio embedding tables (per-cb).
    impl->audio_embds.resize((size_t) impl->n_codebook, nullptr);
    char buf[80];
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

    if (impl->has_in_proj) {
        impl->in_proj = find_required(lm, "lm.depth.in_proj.weight");
        if (!impl->in_proj) { delete impl; return false; }
    }

    impl->depth_output_norm = find_required(lm, "lm.depth.output_norm.weight");
    if (!impl->depth_output_norm) { delete impl; return false; }

    // Optional llama3 RoPE freq factors.
    impl->rope_freq_factors = ggml_get_tensor(lm->codec->weights, "lm.depth.rope_freq_factors");

    // Per-layer weights.
    impl->layers.resize((size_t) impl->depth_layers);
    for (int32_t l = 0; l < impl->depth_layers; ++l) {
        rda_layer_w & w = impl->layers[(size_t) l];
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

    // Helper: project an arbitrary [hidden_dim] vector through in_proj.
    // We do this via a dedicated single-position graph each call (cheap;
    // the graph's small matmul is cached).
    auto project = [&](const float * src, float * dst) -> enum codec_status {
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

enum codec_status step_begin(codec_lm_state * st, const float * h_in) {
    if (!st || !h_in) return CODEC_STATUS_INVALID_ARG;
    rda_impl * impl = static_cast<rda_impl *>(st->lm->impl);
    rda_state * sst = static_cast<rda_state *>(st->impl);
    std::memcpy(sst->h_in_buf.data(), h_in, (size_t) impl->hidden_dim * sizeof(float));
    return run_c0_head(st, h_in);
}

bool step_pending(const codec_lm_state * st) {
    return st != nullptr && st->next_cb < st->lm->info.n_codebook;
}

const float * step_logits(codec_lm_state * st, int32_t * out_cb_idx, int32_t * out_n) {
    if (!st || !st->impl) return nullptr;
    rda_state * sst = static_cast<rda_state *>(st->impl);
    const int32_t k = st->next_cb;
    if (k >= st->lm->info.n_codebook) return nullptr;

    if (k >= 1) {
        // Lazily run the depth step that produces ck_logits.  Earlier
        // codes already pushed are in st->codes_buf (the generic dispatch
        // records them on push_code).
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
    if (cb_idx < 0 || cb_idx >= impl->n_codebook) return nullptr;
    ggml_tensor * t = impl->audio_embds[(size_t) cb_idx];
    const float * data = codec_tensor_data_f32(t);
    if (!data) return nullptr;
    return data + (size_t) code * impl->audio_embed_dim;
}

enum codec_status compose_audio_embd(codec_lm * lm, const int32_t * codes, float * out_embd) {
    if (!lm || !lm->impl || !codes || !out_embd) return CODEC_STATUS_INVALID_ARG;
    rda_impl * impl = static_cast<rda_impl *>(lm->impl);
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
