#include "lm_internal.h"

#include "../runtime/graph.h"
#include "../runtime/graph_exec.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/tensor_utils.h"
#include "../ops/ggml_ops.h"
#include "../ops/lm_attn.h"

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
    bool    has_output_norm  = true;   // most models have one; Moshi / LFM2 don't
    bool    use_rope         = true;   // most models apply RoPE; Moshi doesn't
    bool    rope_interleaved = false;  // true = NORMAL mode (GPT-J pair layout,
                                       // LFM2-Audio).  false = NEOX (Llama
                                       // rotate_half; CSM/Qwen3-TTS).
    bool    c0_is_text       = false;  // c0_input_modality="text" (Moshi)
    bool    c0_is_none       = false;  // c0_input_modality="none" (LFM2: pos 0
                                       // input is zero, plus in_proj(h_in))
    bool    depth_emits_c0   = false;  // Moshi / LFM2: all N cb come from depth;
                                       // c0_head absent.
    bool    in_proj_per_pos  = false;  // LFM2: in_proj is 3D (N, H_d, H_b);
                                       // each depth position uses its own slice
                                       // added to the embed.  Auto-detected
                                       // at init from `in_proj->ne[2] > 1`
                                       // when not set by metadata.
    bool    in_proj_has_bias = false;  // LFM2: per-pos bias of shape (H_d, N).
    bool    has_pre_head_norm = false; // LFM2: RMSNorm applied to depth_h_last
                                       // before each head.  Per-cb weight.
    int32_t depth_text_vocab = 0;      // text vocab for the c0_is_text input

    // Tensor handles (live in codec->weights).  One uniform layout —
    // the per-pos linear inside `codec_op_lm_per_pos_linear` handles
    // both 2D shared weights and 3D per-pos weights transparently, so
    // `layers` carries either shape.  Sizing differences between models:
    //
    //   - `audio_embds`: N entries when depth pos 1..N-1 cover all
    //     codebooks (CSM/Qwen3-TTS/LFM2), N-1 entries when only c1..
    //     c_{N-1} are inputs (Moshi).  Index `j` is the embed for
    //     depth position `j+1`, i.e. the table that embeds c_j.
    //   - `depth_heads`: per-cb 2D heads.  N entries when depth emits
    //     c0 (LFM2), N-1 entries otherwise (CSM/Qwen3-TTS).  Empty
    //     when a single 3D `flex_heads` covers all positions (Moshi).
    //   - `flex_heads`: alternative single 3D `(V, depth_hidden, N)`
    //     storage; if set, head at position p uses slice `[p]` and
    //     `depth_heads` should be empty.
    std::vector<ggml_tensor *> audio_embds;
    ggml_tensor * text_embd         = nullptr; // c0_is_text only (Moshi)
    ggml_tensor * c0_head           = nullptr; // when !depth_emits_c0
    std::vector<ggml_tensor *> depth_heads;    // per-cb 2D heads
    std::vector<ggml_tensor *> heads_pre_norm; // per-cb pre-head RMSNorm
                                               // (has_pre_head_norm only)
    ggml_tensor * flex_heads        = nullptr; // single 3D heads (Moshi)
    ggml_tensor * in_proj           = nullptr; // 2D (shared) or 3D (per-pos)
    ggml_tensor * in_proj_bias      = nullptr; // 1D (Qwen3-TTS) or 2D (LFM2)
    ggml_tensor * depth_output_norm = nullptr; // present except Moshi/LFM2
    ggml_tensor * rope_freq_factors = nullptr; // llama3 RoPE scaling
    std::vector<rda_layer_w> layers;           // per-layer weights (2D or 3D)

    // Backbone-side compose embedding for the next AR step (LFM2).
    // When set, `compose_audio_embd` does a fused-table lookup
    // `sum_i compose_table[c_i + i * compose_codebook_stride]` and
    // returns a `compose_audio_embed_dim`-wide vector.  When null, the
    // legacy per-cb path (sum over `audio_embds[i][c_i]`) is used.
    ggml_tensor * compose_audio_embd_fused = nullptr;
    int32_t       compose_audio_embed_dim  = 0;
    int32_t       compose_codebook_stride  = 0;

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

// Thin shim over the shared Llama-style depth block in src/ops/lm_attn.
// Pulls the per-layer weights out of `rda_layer_w`, masks q/k-norm
// to nullptr when `has_qk_norm=false`, and delegates everything else.
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
    bool    has_qk_norm,
    int32_t rope_mode = GGML_ROPE_TYPE_NEOX,
    bool    use_rope  = true) {
    ggml_tensor * q_norm = (has_qk_norm && w.q_norm) ? w.q_norm : nullptr;
    ggml_tensor * k_norm = (has_qk_norm && w.k_norm) ? w.k_norm : nullptr;
    return codec_op_lm_llama_depth_block(
        ctx, x_ht,
        w.attn_norm, w.q, w.k, w.v, w.o,
        q_norm, k_norm,
        t_pos, freq_factors,
        w.ffn_norm, w.ffn_gate, w.ffn_up, w.ffn_down,
        head_dim, n_heads, n_kv_heads,
        rope_theta, rms_eps,
        rope_mode, use_rope);
}

// Host-side helper: copy one row of an embedding table into a float
// buffer.  Both the flex (Moshi) and lfm2 prefix builders need this to
// fill the (depth_hidden, T) input embeddings.  F32 host-buffer tables
// memcpy directly; F16 / quantized tables go through
// `codec_tensor_as_vec_f32` which dequants the whole table.
//
// Returns true on success.  On failure, `error_out` (if non-null) gets
// a descriptive message including the caller-supplied `tag`.
static bool rda_copy_embd_row(
        const ggml_tensor * tbl, int32_t row_id, int32_t row_dim,
        float * dst, const char * tag, std::string * error_out) {
    if (tbl == nullptr) {
        if (error_out) *error_out = std::string("missing embed table for ") + tag;
        return false;
    }
    const int64_t n_rows = tbl->ne[1];
    if (row_id < 0 || (int64_t) row_id >= n_rows) {
        if (error_out) *error_out = std::string("row id out of range for ") + tag;
        return false;
    }
    const size_t row_bytes = (size_t) row_dim * sizeof(float);
    if (tbl->type == GGML_TYPE_F32 &&
        (tbl->buffer == nullptr || ggml_backend_buffer_is_host(tbl->buffer))) {
        const float * data = static_cast<const float *>(ggml_get_data(tbl));
        std::memcpy(dst, data + (size_t) row_id * row_dim, row_bytes);
    } else {
        std::vector<float> all;
        if (!codec_tensor_as_vec_f32(tbl, &all)) {
            if (error_out) *error_out = std::string("dequant failed for ") + tag;
            return false;
        }
        std::memcpy(dst, all.data() + (size_t) row_id * row_dim, row_bytes);
    }
    return true;
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

// Unified depth-step graph builder.  ONE graph handles all variants by
// branching on metadata; per-model differences are expressed via flags
// in `rda_impl`, not via separate code paths.
//
// Two compose modes — discriminated by whether the per-pos in_proj is
// applied to the prefix row or added on top of it:
//
//   `in_proj_per_pos = false` (CSM / Qwen3-TTS):
//     Prefix row width = audio_embed_dim (typically = hidden_dim).
//     Pos 0  : raw h_in (host-built memcpy)
//     Pos p>=1: raw audio_embd row in audio_embed_dim-space
//     Graph applies a SINGLE shared 2D in_proj (or identity) to each
//     row, producing `(depth_hidden, T)` ready for the layers.
//
//   `in_proj_per_pos = true` (Moshi / LFM2-Audio):
//     Prefix row width = audio_embed_dim (= depth_hidden).
//     Pos 0  : text_embd[text_tok] (c0_is_text), or zero (c0_is_none).
//     Pos p>=1: raw audio_embd row in depth_hidden-space
//     Graph applies per-pos `in_proj[p] @ h_in (+ bias[p])` and ADDS
//     it to the prefix.
//
// After compose, the same loop runs the `depth_layers` Llama-style
// blocks (RMSNorm + GQA + optional QK-norm + optional RoPE + SwiGLU
// FFN), optionally a final RMSNorm, then picks the head at the last
// position (`head_idx`) and applies an optional per-cb pre-head
// RMSNorm before the logits matmul.  Head storage picks one of:
//   - 3D `lm.depth.heads.weight` slice (Moshi) — `flex_heads != nullptr`
//   - 2D `lm.depth.heads_{i}.weight`             — `depth_heads[head_idx]`
struct rda_depth_build {
    rda_impl * impl;
    int32_t    T;          // prefix length (= current_k + 1)
    int32_t    head_idx;   // which head/pre-norm slot to apply at pos T-1
};

bool rda_build_depth_step(ggml_context * ctx_eval, void * ud, ggml_tensor ** out) {
    auto * b = static_cast<rda_depth_build *>(ud);
    if (!ctx_eval || !b || !b->impl || !out) return false;
    rda_impl * impl = b->impl;
    const int32_t T = b->T;
    if (T < 1 || T > impl->n_codebook) return false;

    // ---- Inputs ---------------------------------------------------------
    // t_x is always declared (the embed-row prefix).
    // t_h_in only exists when in_proj is per-pos and reads h_in.
    // t_pos only exists when use_rope.
    ggml_tensor * t_x = ggml_new_tensor_2d(
        ctx_eval, GGML_TYPE_F32, impl->audio_embed_dim, T);
    ggml_set_name(t_x, "lm.depth.x");

    ggml_tensor * t_h_in = nullptr;
    if (impl->in_proj_per_pos) {
        t_h_in = ggml_new_tensor_1d(
            ctx_eval, GGML_TYPE_F32, impl->hidden_dim);
        ggml_set_name(t_h_in, "lm.depth.h_in");
    }

    ggml_tensor * t_pos = nullptr;
    if (impl->use_rope) {
        t_pos = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, T);
        ggml_set_name(t_pos, "lm.depth.pos");
    }

    // ---- Compose: get x into (depth_hidden, T) space -------------------
    ggml_tensor * x;
    if (!impl->in_proj_per_pos) {
        // CSM / Qwen3-TTS: prefix is in hidden_dim space, apply shared
        // 2D in_proj (or skip when has_in_proj=false → Identity, in
        // which case audio_embed_dim must equal depth_hidden).
        if (impl->has_in_proj && impl->in_proj != nullptr) {
            x = codec_op_lm_per_pos_linear(
                ctx_eval, impl->in_proj, t_x, impl->depth_hidden, T);
            if (impl->in_proj_bias != nullptr) {
                // 1D bias broadcast across all positions (Qwen3-TTS-1.7B).
                ggml_tensor * bias_f32 = codec_graph_cast_f32(ctx_eval, impl->in_proj_bias);
                x = ggml_add(ctx_eval, x, bias_f32);
            }
        } else {
            x = t_x;
        }
    } else {
        // Moshi / LFM2: prefix already in depth_hidden space; add
        // per-pos `in_proj[p] @ h_in (+ bias[p])`.
        x = t_x;
        if (impl->in_proj != nullptr) {
            ggml_tensor * w_f32 = codec_graph_cast_f32(ctx_eval, impl->in_proj);
            // Slice first T positions of the 3D weight.  Inlined
            // (rather than via codec_op_lm_per_pos_linear) because the
            // input here is a single h_in vector broadcast across
            // batch, not a per-pos prefix; we want mul_mat with the
            // input as the LHS broadcaster.
            ggml_tensor * w_sl = ggml_view_3d(
                ctx_eval, w_f32,
                w_f32->ne[0], w_f32->ne[1], (int64_t) T,
                w_f32->nb[1], w_f32->nb[2], 0);
            ggml_tensor * h_3d   = ggml_reshape_3d(
                ctx_eval, t_h_in, impl->hidden_dim, 1, 1);
            ggml_tensor * proj_3d = ggml_mul_mat(ctx_eval, h_3d, w_sl);
            ggml_tensor * proj    = ggml_reshape_2d(
                ctx_eval, proj_3d, impl->depth_hidden, T);

            if (impl->in_proj_bias != nullptr) {
                ggml_tensor * bias_f32 = codec_graph_cast_f32(ctx_eval, impl->in_proj_bias);
                if (bias_f32->ne[1] > 1) {
                    // 2D `(depth_hidden, N)` per-pos bias (LFM2).
                    ggml_tensor * bias_sl = ggml_view_2d(
                        ctx_eval, bias_f32,
                        bias_f32->ne[0], (int64_t) T,
                        bias_f32->nb[1], 0);
                    proj = ggml_add(ctx_eval, proj, bias_sl);
                } else {
                    // 1D bias broadcast across all positions.
                    proj = ggml_add(ctx_eval, proj, bias_f32);
                }
            }
            x = ggml_add(ctx_eval, x, proj);
        }
    }

    // ---- Transformer layers --------------------------------------------
    ggml_tensor * freqs = (impl->use_rope && impl->rope_freq_factors)
        ? codec_graph_cast_f32(ctx_eval, impl->rope_freq_factors)
        : nullptr;
    const int32_t rope_mode = impl->rope_interleaved
        ? GGML_ROPE_TYPE_NORMAL : GGML_ROPE_TYPE_NEOX;
    std::vector<rda_layer_w> & layers = impl->layers;

    for (int32_t l = 0; l < impl->depth_layers; ++l) {
        x = rda_depth_layer(
            ctx_eval, x, layers[(size_t) l], t_pos, freqs,
            impl->depth_head_dim, impl->depth_n_heads, impl->depth_n_kv_heads,
            impl->depth_rope_theta, impl->depth_rms_eps,
            impl->has_qk_norm, rope_mode, impl->use_rope);
    }

    // ---- Optional output norm ------------------------------------------
    if (impl->has_output_norm && impl->depth_output_norm != nullptr) {
        ggml_tensor * onorm = codec_graph_cast_f32(ctx_eval, impl->depth_output_norm);
        x = codec_op_rms_norm_ct(ctx_eval, x, impl->depth_rms_eps, onorm);
    }

    // ---- Per-cb head at the last position ------------------------------
    const int32_t head_idx = b->head_idx;
    ggml_tensor * x_last = ggml_view_1d(
        ctx_eval, x, impl->depth_hidden,
        (size_t) (T - 1) * impl->depth_hidden * sizeof(float));
    x_last = ggml_cont(ctx_eval, x_last);

    if (impl->has_pre_head_norm) {
        if ((size_t) head_idx >= impl->heads_pre_norm.size() ||
            impl->heads_pre_norm[(size_t) head_idx] == nullptr) {
            return false;
        }
        ggml_tensor * pn = codec_graph_cast_f32(
            ctx_eval, impl->heads_pre_norm[(size_t) head_idx]);
        x_last = codec_op_rms_norm_ct(ctx_eval, x_last, impl->depth_rms_eps, pn);
    }

    ggml_tensor * head_w;
    if (impl->flex_heads != nullptr) {
        // Single 3D heads tensor (Moshi): slice [head_idx] as a 2D view.
        ggml_tensor * heads_f32 = codec_graph_cast_f32(ctx_eval, impl->flex_heads);
        head_w = ggml_view_2d(
            ctx_eval, heads_f32,
            heads_f32->ne[0], heads_f32->ne[1],
            heads_f32->nb[1],
            (size_t) head_idx * heads_f32->nb[2]);
    } else {
        if ((size_t) head_idx >= impl->depth_heads.size() ||
            impl->depth_heads[(size_t) head_idx] == nullptr) {
            return false;
        }
        head_w = codec_graph_cast_f32(ctx_eval, impl->depth_heads[(size_t) head_idx]);
    }

    ggml_tensor * logits = ggml_mul_mat(ctx_eval, head_w, x_last);
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
    impl->has_output_norm  = codec_read_bool_kv(gf, "codec.lm.residual.depth_has_output_norm", true);
    impl->use_rope         = codec_read_bool_kv(gf, "codec.lm.residual.depth_use_rope", true);
    impl->depth_text_vocab = codec_read_i32_kv (gf, "codec.lm.residual.depth_text_vocab", 0);
    impl->in_proj_per_pos  = codec_read_bool_kv(gf, "codec.lm.residual.depth_in_proj_per_pos", false);
    impl->in_proj_has_bias = codec_read_bool_kv(gf, "codec.lm.residual.depth_in_proj_has_bias", false);
    impl->has_pre_head_norm = codec_read_bool_kv(gf, "codec.lm.residual.depth_has_pre_head_norm", false);
    impl->depth_emits_c0   = codec_read_bool_kv(gf, "codec.lm.residual.depth_emits_c0", false);
    impl->rope_interleaved = codec_read_bool_kv(gf, "codec.lm.residual.depth_rope_interleaved", false);

    {
        std::string m = codec_lm_read_string_kv(
            lm->codec, "codec.lm.residual.c0_input_modality");
        impl->c0_is_text = (m == "text");
        impl->c0_is_none = (m == "none");
    }

    if (impl->depth_layers <= 0 || impl->depth_hidden <= 0 ||
        impl->depth_n_heads <= 0 || impl->depth_head_dim <= 0 ||
        impl->depth_inter <= 0) {
        lm->last_error = "invalid residual_depth_ar metadata: zero/negative dims";
        delete impl;
        return false;
    }

    char buf[80];

    // Audio embedding tables.  Two naming conventions in flight:
    //   - CSM / Qwen3-TTS: `lm.audio_embd_{i}.weight`
    //   - Moshi / LFM2-Audio (in-depth namespace): `lm.depth.audio_embd_{i}.weight`
    // Load whichever is present.  Vector size is the maximum index for
    // which a tensor exists, capped at N — Moshi has N-1 entries (last
    // cb never feeds back); CSM/Qwen3-TTS/LFM2 have N.
    impl->audio_embds.clear();
    impl->audio_embds.reserve((size_t) impl->n_codebook);
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        std::snprintf(buf, sizeof(buf), "lm.depth.audio_embd_%d.weight", i);
        ggml_tensor * t = ggml_get_tensor(lm->codec->weights, buf);
        if (t == nullptr) {
            std::snprintf(buf, sizeof(buf), "lm.audio_embd_%d.weight", i);
            t = ggml_get_tensor(lm->codec->weights, buf);
        }
        if (t == nullptr) break;  // first missing index ends the vector
        impl->audio_embds.push_back(t);
    }
    if (impl->audio_embds.empty()) {
        lm->last_error = "missing lm.audio_embd_0.weight / lm.depth.audio_embd_0.weight";
        delete impl; return false;
    }

    // Optional text_embd (Moshi: pos 0 when c0_is_text).
    if (impl->c0_is_text) {
        impl->text_embd = find_required(lm, "lm.depth.text_embd.weight");
        if (!impl->text_embd) { delete impl; return false; }
    }

    // c0 source: either a backbone-side c0_head OR depth-internal head[0].
    impl->c0_head = ggml_get_tensor(lm->codec->weights, "lm.c0_head.weight");

    // Depth heads: prefer per-cb 2D entries; fall back to the single 3D
    // `lm.depth.heads.weight` (Moshi).
    {
        ggml_tensor * single = ggml_get_tensor(lm->codec->weights, "lm.depth.heads.weight");
        if (single != nullptr && single->ne[2] > 1) {
            impl->flex_heads = single;
        } else {
            for (int32_t i = 0; i < impl->n_codebook; ++i) {
                std::snprintf(buf, sizeof(buf), "lm.depth.heads_%d.weight", i);
                ggml_tensor * t = ggml_get_tensor(lm->codec->weights, buf);
                if (t == nullptr) break;
                impl->depth_heads.push_back(t);
            }
        }
        if (impl->flex_heads == nullptr && impl->depth_heads.empty()) {
            lm->last_error = "no depth heads found "
                "(expected lm.depth.heads.weight or lm.depth.heads_{i}.weight)";
            delete impl; return false;
        }
    }

    // Optional per-cb pre-head RMSNorm (LFM2-Audio).
    if (impl->has_pre_head_norm) {
        impl->heads_pre_norm.reserve(impl->depth_heads.size());
        for (size_t i = 0; i < impl->depth_heads.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "lm.depth.heads_%zu_norm.weight", i);
            ggml_tensor * t = find_required(lm, buf);
            if (t == nullptr) { delete impl; return false; }
            impl->heads_pre_norm.push_back(t);
        }
    }

    if (impl->has_in_proj) {
        impl->in_proj = find_required(lm, "lm.depth.in_proj.weight");
        if (!impl->in_proj) { delete impl; return false; }
        // Bias is optional; CSM doesn't have one, Qwen3-TTS / LFM2 do.
        impl->in_proj_bias = ggml_get_tensor(lm->codec->weights, "lm.depth.in_proj.bias");
    } else {
        // Identity in_proj — only viable when prefix row dim already matches
        // depth_hidden AND the runtime won't try to add a per-pos h_in
        // contribution.  Qwen3-TTS-0.6B is the canonical example.
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

    // Optional llama3 RoPE freq factors.
    impl->rope_freq_factors = ggml_get_tensor(lm->codec->weights, "lm.depth.rope_freq_factors");

    // Optional backbone-side compose table (LFM2-Audio's
    // `audio_embedding.embedding`).  When present, compose_audio_embd
    // uses a fused single-table sum with codebook offsets instead of
    // the per-cb depth-side audio_embds.
    impl->compose_audio_embd_fused = ggml_get_tensor(
        lm->codec->weights, "lm.compose.audio_embd.weight");
    if (impl->compose_audio_embd_fused != nullptr) {
        impl->compose_audio_embed_dim = codec_read_i32_kv(
            gf, "codec.lm.compose.audio_embed_dim", 0);
        impl->compose_codebook_stride = codec_read_i32_kv(
            gf, "codec.lm.compose.codebook_stride", 0);
        if (impl->compose_audio_embed_dim <= 0 || impl->compose_codebook_stride <= 0) {
            lm->last_error =
                "lm.compose.audio_embd.weight present but compose metadata "
                "(audio_embed_dim / codebook_stride) is missing or zero";
            delete impl; return false;
        }
    }

    // Per-layer weights — same tensor names regardless of 2D / 3D shape.
    std::vector<rda_layer_w> & layers_out = impl->layers;
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

    // ---- Auto-detection backfill --------------------------------------
    // Older converters (esp. Moshi pre-unification) don't write the
    // metadata flags the unified runtime uses to dispatch.  Derive the
    // missing ones from tensor shapes so legacy GGUFs keep working.
    if (impl->in_proj != nullptr && impl->in_proj->ne[2] > 1) {
        // 3D weight => per-position in_proj.
        impl->in_proj_per_pos = true;
    }
    if (impl->c0_head == nullptr) {
        // No backbone-side c0 head => depth emits c0 too.
        impl->depth_emits_c0 = true;
    }
    if (impl->in_proj_bias != nullptr) {
        impl->in_proj_has_bias = true;
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
// Unified depth-step driver.  Builds the host-side (audio_embed_dim, T)
// prefix according to the compose mode + c0_input_modality, then runs
// the single `rda_build_depth_step` graph.
//
// In shared / `in_proj_per_pos=false` mode (CSM / Qwen3-TTS):
//   pos 0   = h_in_buf (`hidden_dim` floats, since audio_embed_dim ==
//             hidden_dim for this mode).
//   pos p>=1 = audio_embd_{p-1}[c_{p-1}].
//
// In per-pos / `in_proj_per_pos=true` mode (Moshi / LFM2-Audio):
//   pos 0   = text_embd[text_token_context] (c0_is_text), or zero
//             (c0_is_none), or audio_embds[0][...] (c0_is_audio — no
//             current model exercises this).
//   pos p>=1 = audio_embd_{p-1}[c_{p-1}].
//
// In both modes the graph picks up the row width from
// `impl->audio_embed_dim` and the per-pos `head_idx` selects which
// pre-head norm / 2D head / 3D head slice to apply at the last position.
enum codec_status run_depth_step(codec_lm_state * st, int32_t current_k) {
    rda_impl  * impl = static_cast<rda_impl  *>(st->lm->impl);
    rda_state * sst  = static_cast<rda_state *>(st->impl);
    if (current_k < 0 || current_k >= impl->n_codebook) {
        st->last_error = "depth step: cb index out of range";
        return CODEC_STATUS_INVALID_STATE;
    }
    if (!impl->depth_emits_c0 && current_k < 1) {
        // Models with a separate c0_head shouldn't enter the depth path
        // for k=0; the dispatcher in step_logits enforces this.
        st->last_error = "depth step: k=0 invalid when depth_emits_c0=false";
        return CODEC_STATUS_INVALID_STATE;
    }
    const int32_t T = current_k + 1;
    const int32_t row_dim = impl->audio_embed_dim;

    // ---- Build (row_dim, T) prefix host-side ----------------------------
    std::vector<float> prefix((size_t) row_dim * (size_t) T, 0.0f);

    if (!impl->in_proj_per_pos) {
        // CSM / Qwen3-TTS: pos 0 input is the raw backbone hidden, sitting
        // in hidden_dim-space.  Init guarantees audio_embed_dim ==
        // hidden_dim for this mode.
        std::memcpy(prefix.data(), sst->h_in_buf.data(),
                    (size_t) impl->hidden_dim * sizeof(float));
    } else if (impl->c0_is_text) {
        // Moshi: pos 0 = text_embd[text_token_context].
        if (st->text_token_context < 0) {
            st->last_error =
                "depth step: c0_input_modality=text but no text context set; "
                "call codec_lm_state_set_text_context before step_begin";
            return CODEC_STATUS_INVALID_STATE;
        }
        std::string err;
        if (!rda_copy_embd_row(impl->text_embd, st->text_token_context,
                               row_dim, prefix.data(), "text_embd@pos0", &err)) {
            st->last_error = "depth step: " + err;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    } else if (impl->c0_is_none) {
        // LFM2-Audio: pos 0 left as zero (already zeroed at init).
    } else {
        // Hypothetical "audio c0" in flex mode — no current model uses this.
        st->last_error =
            "depth step: c0_input_modality=audio + in_proj_per_pos not supported";
        return CODEC_STATUS_INVALID_STATE;
    }

    // Pos p>=1: audio_embd_{p-1}[c_{p-1}], rows in row_dim-space.
    for (int32_t p = 1; p < T; ++p) {
        const int32_t embd_idx = p - 1;
        if ((size_t) embd_idx >= impl->audio_embds.size() ||
            impl->audio_embds[(size_t) embd_idx] == nullptr) {
            st->last_error = "depth step: missing audio_embd for prefix pos";
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        std::string err;
        if (!rda_copy_embd_row(impl->audio_embds[(size_t) embd_idx],
                               st->codes_buf[(size_t) embd_idx],
                               row_dim,
                               prefix.data() + (size_t) p * row_dim,
                               "audio_embd", &err)) {
            st->last_error = "depth step: " + err;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    // ---- Run the unified depth graph -----------------------------------
    // head_idx maps cb -> head slot: with depth_emits_c0 the depth path
    // owns all N heads (idx = k); otherwise c0 lives in c0_head and the
    // depth heads cover c1..c_{N-1} (idx = k - 1).
    const int32_t head_idx = impl->depth_emits_c0 ? current_k : (current_k - 1);

    rda_depth_build build = { impl, T, head_idx };
    codec_graph_eval_guard guard(st->ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind     = (int32_t) CODEC_GRAPH_LM_RDA_DEPTH_STEP;
    key.n_frames = T;
    key.n_q      = head_idx;

    if (!codec_graph_cache_get_or_build(
            st->ctx, key, rda_build_depth_step, &build, sizeof(build), &entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_x    = codec_graph_get_tensor(st->ctx, entry, "lm.depth.x");
    ggml_tensor * t_h_in = impl->in_proj_per_pos
        ? codec_graph_get_tensor(st->ctx, entry, "lm.depth.h_in") : nullptr;
    ggml_tensor * t_pos  = impl->use_rope
        ? codec_graph_get_tensor(st->ctx, entry, "lm.depth.pos") : nullptr;
    ggml_tensor * t_lg   = codec_graph_get_tensor(st->ctx, entry, "lm.depth.ck_logits");
    if (!t_x || !t_lg ||
        (impl->in_proj_per_pos && !t_h_in) ||
        (impl->use_rope && !t_pos)) {
        st->last_error = "depth graph missing tensors";
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_x, prefix.data(),
                                    prefix.size() * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (impl->in_proj_per_pos &&
        !codec_runtime_write_tensor(t_h_in, sst->h_in_buf.data(),
                                    (size_t) impl->hidden_dim * sizeof(float),
                                    &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (impl->use_rope) {
        std::vector<int32_t> positions((size_t) T);
        for (int32_t i = 0; i < T; ++i) positions[(size_t) i] = i;
        if (!codec_runtime_write_tensor(t_pos, positions.data(),
                                        positions.size() * sizeof(int32_t), &err)) {
            st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
        }
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
    if (impl->depth_emits_c0) {
        // Models without a separate c0_head defer all logits to the
        // depth decoder.  Just stash h_in; step_logits(k=0) will run
        // the first depth step (T=1).
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

    // Two cases:
    //  - `depth_emits_c0 = true` (Moshi / LFM2-Audio): every cb including
    //    k=0 comes from the unified depth graph.
    //  - `depth_emits_c0 = false` (CSM / Qwen3-TTS): c0 was already
    //    computed at step_begin via a separate c0_head; c1..c_{N-1}
    //    flow through the same depth graph with a different `head_idx`.
    if (impl->depth_emits_c0 || k >= 1) {
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
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        if (codes[i] < 0 || codes[i] >= lm->info.codebook_sizes[i]) {
            return CODEC_STATUS_INVALID_ARG;
        }
    }

    // LFM2-Audio fused-table path: sum of N rows from a single
    // backbone-side embedding table at offsets `c_i + i * stride`.
    // Output dim = backbone hidden (different from depth-side
    // audio_embed_dim), exposed via compose_audio_embed_dim.
    if (impl->compose_audio_embd_fused != nullptr) {
        const int32_t out_dim = impl->compose_audio_embed_dim;
        const int32_t stride  = impl->compose_codebook_stride;
        ggml_tensor * tbl     = impl->compose_audio_embd_fused;
        // Dequant the whole table to a thread-local F32 buffer once
        // per call.  For F32 / host buffers this is a fast pass-through
        // via codec_tensor_data_f32; for F16 / quantised it dequants
        // fully (one-shot per call — the table is 16392 × 2048 = 128 MB
        // F32 max, but for our F16-quantised LFM2 GGUF it's the same
        // amount we'd materialise per-row anyway).
        const float * data = codec_tensor_data_f32(tbl);
        std::vector<float> dequant;
        if (data == nullptr) {
            if (!codec_tensor_as_vec_f32(tbl, &dequant)) {
                lm->last_error = "compose_audio_embd: fused-table dequant failed";
                return CODEC_STATUS_INTERNAL_ERROR;
            }
            data = dequant.data();
        }
        std::fill(out_embd, out_embd + out_dim, 0.0f);
        for (int32_t i = 0; i < impl->n_codebook; ++i) {
            const int64_t row = (int64_t) codes[i] + (int64_t) i * stride;
            if (row < 0 || row >= tbl->ne[1]) {
                lm->last_error = "compose_audio_embd: cb code + offset out of range";
                return CODEC_STATUS_INVALID_ARG;
            }
            const float * src = data + row * out_dim;
            for (int32_t j = 0; j < out_dim; ++j) {
                out_embd[j] += src[j];
            }
        }
        return CODEC_STATUS_SUCCESS;
    }

    if (impl->in_proj_per_pos) {
        // Moshi: still no compose support here — caller composes via
        // backbone's dual-stream tables.
        lm->last_error =
            "compose_audio_embd: not supported for in_proj_per_pos models "
            "without lm.compose.audio_embd (caller composes via backbone)";
        return CODEC_STATUS_INVALID_STATE;
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
