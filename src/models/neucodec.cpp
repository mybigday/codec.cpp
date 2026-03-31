#include "neucodec.h"

#include "../ops/conv1d.h"
#include "../ops/ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/local_attn.h"
#include "../ops/rope.h"
#include "../ops/pool1d.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

static std::string codec_neu_name_tok() { return "neucodec.decode.tok"; }
static std::string codec_neu_name_codebook() { return "neucodec.decode.codebook"; }
static std::string codec_neu_name_quant_w() { return "neucodec.decode.quant.project_out.w"; }
static std::string codec_neu_name_quant_b() { return "neucodec.decode.quant.project_out.b"; }
static std::string codec_neu_name_fc_post_w() { return "neucodec.decode.fc_post_a.w"; }
static std::string codec_neu_name_fc_post_b() { return "neucodec.decode.fc_post_a.b"; }
static std::string codec_neu_name_embed_w() { return "neucodec.decode.embed.w"; }
static std::string codec_neu_name_embed_b() { return "neucodec.decode.embed.b"; }
static std::string codec_neu_name_final_ln_w() { return "neucodec.decode.final_ln.w"; }
static std::string codec_neu_name_final_ln_b() { return "neucodec.decode.final_ln.b"; }
static std::string codec_neu_name_head_w() { return "neucodec.decode.head.out.w"; }
static std::string codec_neu_name_head_b() { return "neucodec.decode.head.out.b"; }
static std::string codec_neu_name_istft_window() { return "neucodec.decode.istft.window"; }

static std::string codec_neu_name_prior(int32_t li, const char * suffix) {
    return "neucodec.decode.prior." + std::to_string(li) + "." + suffix;
}

static std::string codec_neu_name_post(int32_t li, const char * suffix) {
    return "neucodec.decode.post." + std::to_string(li) + "." + suffix;
}

static std::string codec_neu_name_transformer(int32_t li, const char * suffix) {
    return "neucodec.decode.transformer." + std::to_string(li) + "." + suffix;
}

static std::string codec_neu_encode_name(const std::string & name) {
    if (name.rfind("neucodec.encode.", 0) != 0) {
        return name;
    }
    uint64_t h = 1469598103934665603ull;
    for (char c : name) {
        h ^= (uint8_t) c;
        h *= 1099511628211ull;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "nce.%016llx", (unsigned long long) h);
    return std::string(buf);
}

static void codec_neu_set_enc_name(ggml_tensor * tensor, const std::string & name) {
    if (tensor == nullptr) {
        return;
    }
    const std::string short_name = codec_neu_encode_name(name);
    ggml_set_name(tensor, short_name.c_str());
}

static ggml_tensor * codec_neu_get_tensor(codec_model * model, const std::string & name) {
    if (model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    const std::string short_name = codec_neu_encode_name(name);
    return ggml_get_tensor(model->weights, short_name.c_str());
}

static float codec_neu_silu(float x) {
    // Match torch.nn.functional.silu behavior with numerically stable sigmoid.
    // This avoids large-|x| overflow drift when precomputing dynamic position bias.
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return x / (1.0f + e);
    }
    const float e = std::exp(x);
    return x * e / (1.0f + e);
}

static bool codec_neu_build_dynamic_pos_bias(
    codec_model * model,
    const std::string & prefix,
    int32_t max_dist,
    std::vector<float> * out,
    std::string * err) {

    if (model == nullptr || out == nullptr || max_dist <= 0) {
        if (err) *err = "invalid dynamic_pos_bias args";
        return false;
    }

    const std::string w0 = prefix + ".mlp.0.weight";
    const std::string b0 = prefix + ".mlp.0.bias";
    const std::string w1 = prefix + ".mlp.2.weight";
    const std::string b1 = prefix + ".mlp.2.bias";
    const std::string w2 = prefix + ".mlp.4.weight";
    const std::string b2 = prefix + ".mlp.4.bias";

    ggml_tensor * tw0 = codec_neu_get_tensor(model, w0);
    ggml_tensor * tb0 = codec_neu_get_tensor(model, b0);
    ggml_tensor * tw1 = codec_neu_get_tensor(model, w1);
    ggml_tensor * tb1 = codec_neu_get_tensor(model, b1);
    ggml_tensor * tw2 = codec_neu_get_tensor(model, w2);
    ggml_tensor * tb2 = codec_neu_get_tensor(model, b2);
    if (tw0 == nullptr || tb0 == nullptr || tw1 == nullptr || tb1 == nullptr || tw2 == nullptr || tb2 == nullptr) {
        if (err) *err = "missing dynamic_pos_bias tensors for " + prefix;
        return false;
    }

    std::vector<float> w0v, b0v, w1v, b1v, w2v, b2v;
    if (!codec_tensor_as_vec_f32(tw0, &w0v) ||
        !codec_tensor_as_vec_f32(tb0, &b0v) ||
        !codec_tensor_as_vec_f32(tw1, &w1v) ||
        !codec_tensor_as_vec_f32(tb1, &b1v) ||
        !codec_tensor_as_vec_f32(tw2, &w2v) ||
        !codec_tensor_as_vec_f32(tb2, &b2v)) {
        if (err) *err = "failed reading dynamic_pos_bias tensors";
        return false;
    }

    const int32_t dim = (int32_t) b0v.size();
    const int32_t heads = (int32_t) b2v.size();
    if ((int32_t) w0v.size() != dim * 1 || (int32_t) w1v.size() != dim * dim || (int32_t) w2v.size() != heads * dim) {
        if (err) *err = "unexpected dynamic_pos_bias tensor shapes";
        return false;
    }

    out->assign((size_t) heads * (size_t) max_dist, 0.0f);
    std::vector<float> y0((size_t) dim, 0.0f);
    std::vector<float> y1((size_t) dim, 0.0f);
    std::vector<float> y2((size_t) heads, 0.0f);

    for (int32_t d = 0; d < max_dist; ++d) {
        const float x = (float) d;
        for (int32_t i = 0; i < dim; ++i) {
            float v = w0v[(size_t) i] * x + b0v[(size_t) i];
            y0[(size_t) i] = codec_neu_silu(v);
        }

        for (int32_t j = 0; j < dim; ++j) {
            float acc = b1v[(size_t) j];
            const float * w_row = &w1v[(size_t) j * (size_t) dim];
            for (int32_t i = 0; i < dim; ++i) {
                acc += w_row[(size_t) i] * y0[(size_t) i];
            }
            y1[(size_t) j] = codec_neu_silu(acc);
        }

        for (int32_t h = 0; h < heads; ++h) {
            float acc = b2v[(size_t) h];
            const float * w_row = &w2v[(size_t) h * (size_t) dim];
            for (int32_t i = 0; i < dim; ++i) {
                acc += w_row[(size_t) i] * y1[(size_t) i];
            }
            y2[(size_t) h] = acc;
        }

        for (int32_t h = 0; h < heads; ++h) {
            (*out)[(size_t) h * (size_t) max_dist + (size_t) d] = y2[(size_t) h];
        }
    }

    return true;
}

static void codec_neu_read_i32_array(
    struct gguf_context * gf,
    const char * key,
    int32_t * dst,
    int32_t dst_n) {

    if (gf == nullptr || key == nullptr || dst == nullptr || dst_n <= 0) {
        return;
    }
    const int key_id = gguf_find_key(gf, key);
    if (key_id < 0 || gguf_get_kv_type(gf, key_id) != GGUF_TYPE_ARRAY) {
        return;
    }
    const enum gguf_type arr_t = gguf_get_arr_type(gf, key_id);
    const size_t n = gguf_get_arr_n(gf, key_id);
    const size_t n_copy = std::min(n, (size_t) dst_n);
    if (arr_t == GGUF_TYPE_UINT32) {
        const uint32_t * src = static_cast<const uint32_t *>(gguf_get_arr_data(gf, key_id));
        if (src != nullptr) {
            for (size_t i = 0; i < n_copy; ++i) {
                dst[i] = (int32_t) src[i];
            }
        }
    } else if (arr_t == GGUF_TYPE_INT32) {
        const int32_t * src = static_cast<const int32_t *>(gguf_get_arr_data(gf, key_id));
        if (src != nullptr) {
            for (size_t i = 0; i < n_copy; ++i) {
                dst[i] = src[i];
            }
        }
    }
}

static ggml_tensor * codec_neu_layer_norm_ct(
    ggml_context * ctx_eval,
    ggml_tensor * x_ct,
    ggml_tensor * gamma,
    ggml_tensor * beta) {

    if (ctx_eval == nullptr || x_ct == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }
    ggml_tensor * y = ggml_norm(ctx_eval, x_ct, 1e-6f);
    ggml_tensor * g2 = ggml_reshape_2d(ctx_eval, gamma, x_ct->ne[0], 1);
    ggml_tensor * b2 = ggml_reshape_2d(ctx_eval, beta, x_ct->ne[0], 1);
    y = ggml_mul(ctx_eval, y, ggml_repeat(ctx_eval, g2, y));
    y = ggml_add(ctx_eval, y, ggml_repeat(ctx_eval, b2, y));
    return y;
}

static ggml_tensor * codec_neu_layer_norm_ct_eps(
    ggml_context * ctx_eval,
    ggml_tensor * x_ct,
    ggml_tensor * gamma,
    ggml_tensor * beta,
    float eps) {

    if (ctx_eval == nullptr || x_ct == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }
    ggml_tensor * y = ggml_norm(ctx_eval, x_ct, eps);
    ggml_tensor * g2 = ggml_reshape_2d(ctx_eval, gamma, x_ct->ne[0], 1);
    ggml_tensor * b2 = ggml_reshape_2d(ctx_eval, beta, x_ct->ne[0], 1);
    y = ggml_mul(ctx_eval, y, ggml_repeat(ctx_eval, g2, y));
    y = ggml_add(ctx_eval, y, ggml_repeat(ctx_eval, b2, y));
    return y;
}

static ggml_tensor * codec_neu_layer_norm_tc(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * gamma,
    ggml_tensor * beta) {

    if (ctx_eval == nullptr || x_tc == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }
    ggml_tensor * x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_tc));
    ggml_tensor * y_ct = codec_neu_layer_norm_ct(ctx_eval, x_ct, gamma, beta);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return ggml_cont(ctx_eval, ggml_transpose(ctx_eval, y_ct));
}

static ggml_tensor * codec_neu_layer_norm_tc_eps(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * gamma,
    ggml_tensor * beta,
    float eps) {

    if (ctx_eval == nullptr || x_tc == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }
    ggml_tensor * x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_tc));
    ggml_tensor * y_ct = codec_neu_layer_norm_ct_eps(ctx_eval, x_ct, gamma, beta, eps);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return ggml_cont(ctx_eval, ggml_transpose(ctx_eval, y_ct));
}

static ggml_tensor * codec_neu_linear_tc(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * w,
    ggml_tensor * b) {

    if (ctx_eval == nullptr || x_tc == nullptr || w == nullptr) {
        return nullptr;
    }
    ggml_tensor * x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_tc));
    ggml_tensor * y_ct = codec_op_linear(ctx_eval, x_ct, w, b);
    if (y_ct == nullptr) {
        return nullptr;
    }
    return ggml_cont(ctx_eval, ggml_transpose(ctx_eval, y_ct));
}

static ggml_tensor * codec_neu_grn_tc(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * gamma,
    ggml_tensor * beta,
    float eps) {

    if (ctx_eval == nullptr || x_tc == nullptr || gamma == nullptr || beta == nullptr) {
        return nullptr;
    }
    (void) eps;
    // distill-neucodec GRN is applied in channels_last format with T=1 reduction axes,
    // which simplifies to x + gamma * x + beta.
    ggml_tensor * g2 = ggml_reshape_2d(ctx_eval, gamma, 1, x_tc->ne[1]);
    ggml_tensor * b2 = ggml_reshape_2d(ctx_eval, beta, 1, x_tc->ne[1]);
    ggml_tensor * y = ggml_mul(ctx_eval, x_tc, ggml_repeat(ctx_eval, g2, x_tc));
    y = ggml_add(ctx_eval, y, ggml_repeat(ctx_eval, b2, y));
    y = ggml_add(ctx_eval, y, x_tc);
    return y;
}

static ggml_tensor * codec_neu_snake_tc(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * alpha,
    float eps) {

    if (ctx_eval == nullptr || x_tc == nullptr || alpha == nullptr) {
        return nullptr;
    }
    ggml_tensor * alpha_2d = ggml_reshape_2d(ctx_eval, alpha, 1, x_tc->ne[1]);
    ggml_tensor * alpha_rep = ggml_repeat(ctx_eval, alpha_2d, x_tc);
    ggml_tensor * alpha_eps = ggml_scale_bias(ctx_eval, alpha_rep, 1.0f, eps);
    ggml_tensor * ax = ggml_mul(ctx_eval, alpha_rep, x_tc);
    ggml_tensor * s = ggml_sin(ctx_eval, ax);
    ggml_tensor * s2 = ggml_mul(ctx_eval, s, s);
    ggml_tensor * frac = ggml_div(ctx_eval, s2, alpha_eps);
    return ggml_add(ctx_eval, x_tc, frac);
}

static ggml_tensor * codec_neu_conv1d_grouped(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * w,
    ggml_tensor * b,
    int32_t stride,
    int32_t dilation,
    int32_t padding,
    int32_t groups) {

    if (ctx_eval == nullptr || x_tc == nullptr || w == nullptr || groups <= 0) {
        return nullptr;
    }
    const int32_t in_channels = (int32_t) x_tc->ne[1];
    const int32_t out_channels = (int32_t) w->ne[2];
    if (in_channels % groups != 0 || out_channels % groups != 0) {
        return nullptr;
    }
    const int32_t in_g = in_channels / groups;
    const int32_t out_g = out_channels / groups;

    ggml_tensor * out = nullptr;
    for (int32_t g = 0; g < groups; ++g) {
        const size_t x_off = (size_t) g * (size_t) in_g * x_tc->nb[1];
        ggml_tensor * x_g = ggml_view_2d(ctx_eval, x_tc, (int32_t) x_tc->ne[0], in_g, x_tc->nb[1], x_off);

        const size_t w_off = (size_t) g * (size_t) out_g * w->nb[2];
        ggml_tensor * w_g = ggml_view_3d(ctx_eval, w, (int32_t) w->ne[0], in_g, out_g, w->nb[1], w->nb[2], w_off);
        ggml_tensor * b_g = nullptr;
        if (b != nullptr) {
            const size_t b_off = (size_t) g * (size_t) out_g * b->nb[0];
            b_g = ggml_view_1d(ctx_eval, b, out_g, b_off);
        }

        ggml_tensor * y_g = codec_conv1d(ctx_eval, x_g, w_g, b_g, stride, dilation, padding);
        if (y_g == nullptr) {
            return nullptr;
        }
        out = out == nullptr ? y_g : ggml_concat(ctx_eval, out, y_g, 1);
    }
    return ggml_cont(ctx_eval, out);
}

static ggml_tensor * codec_neu_rms_norm_ct(
    ggml_context * ctx_eval,
    ggml_tensor * x_ct,
    ggml_tensor * gamma) {

    if (ctx_eval == nullptr || x_ct == nullptr || gamma == nullptr) {
        return nullptr;
    }
    ggml_tensor * y = ggml_rms_norm(ctx_eval, x_ct, 1e-6f);
    ggml_tensor * g2 = ggml_reshape_2d(ctx_eval, gamma, x_ct->ne[0], 1);
    y = ggml_mul(ctx_eval, y, ggml_repeat(ctx_eval, g2, y));
    return y;
}

static ggml_tensor * codec_neu_resnet_block(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * n1_w,
    ggml_tensor * n1_b,
    ggml_tensor * c1_w,
    ggml_tensor * c1_b,
    ggml_tensor * n2_w,
    ggml_tensor * n2_b,
    ggml_tensor * c2_w,
    ggml_tensor * c2_b) {

    if (ctx_eval == nullptr || x_tc == nullptr) {
        return nullptr;
    }

    ggml_tensor * h = codec_op_group_norm(ctx_eval, x_tc, 32, 1e-6f, n1_w, n1_b);
    if (h == nullptr) {
        return nullptr;
    }
    h = ggml_silu(ctx_eval, h);
    h = codec_conv1d(ctx_eval, h, c1_w, c1_b, 1, 1, 1);
    if (h == nullptr) {
        return nullptr;
    }
    h = codec_op_group_norm(ctx_eval, h, 32, 1e-6f, n2_w, n2_b);
    if (h == nullptr) {
        return nullptr;
    }
    h = ggml_silu(ctx_eval, h);
    h = codec_conv1d(ctx_eval, h, c2_w, c2_b, 1, 1, 1);
    if (h == nullptr) {
        return nullptr;
    }

    return ggml_add(ctx_eval, x_tc, h);
}

static ggml_tensor * codec_neu_transformer_block(
    ggml_context * ctx_eval,
    ggml_tensor * x_ct,
    ggml_tensor * att_norm_w,
    ggml_tensor * ffn_norm_w,
    ggml_tensor * att_c_attn_w,
    ggml_tensor * att_c_proj_w,
    ggml_tensor * mlp_fc1_w,
    ggml_tensor * mlp_fc2_w,
    int32_t head_dim,
    int32_t n_heads,
    float rope_theta) {

    if (ctx_eval == nullptr || x_ct == nullptr) {
        return nullptr;
    }
    const int32_t hidden_dim = (int32_t) x_ct->ne[0];
    if (hidden_dim != head_dim * n_heads) {
        return nullptr;
    }

    ggml_tensor * h = codec_neu_rms_norm_ct(ctx_eval, x_ct, att_norm_w);
    if (h == nullptr) {
        return nullptr;
    }

    ggml_tensor * qkv = ggml_mul_mat(ctx_eval, att_c_attn_w, h); // [3*hidden, t]
    if (qkv == nullptr) {
        return nullptr;
    }

    const int64_t t = qkv->ne[1];
    ggml_tensor * q = ggml_cont(ctx_eval, ggml_view_2d(ctx_eval, qkv, hidden_dim, t, qkv->nb[1], 0));
    ggml_tensor * k = ggml_cont(ctx_eval, ggml_view_2d(ctx_eval, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0]));
    ggml_tensor * v = ggml_cont(ctx_eval, ggml_view_2d(ctx_eval, qkv, hidden_dim, t, qkv->nb[1], (size_t) hidden_dim * qkv->nb[0] * 2));

    ggml_tensor * q_dht = ggml_reshape_3d(ctx_eval, q, head_dim, n_heads, t); // [d, h, t]
    ggml_tensor * k_dht = ggml_reshape_3d(ctx_eval, k, head_dim, n_heads, t); // [d, h, t]
    ggml_tensor * v_dth = ggml_permute(ctx_eval, ggml_reshape_3d(ctx_eval, v, head_dim, n_heads, t), 0, 2, 1, 3); // [d, t, h]

    ggml_tensor * q_rope_dht = codec_op_rope(ctx_eval, q_dht, head_dim, rope_theta, 1.0f);
    ggml_tensor * k_rope_dht = codec_op_rope(ctx_eval, k_dht, head_dim, rope_theta, 1.0f);
    ggml_tensor * q_rope = q_rope_dht ? ggml_cont(ctx_eval, ggml_permute(ctx_eval, q_rope_dht, 0, 2, 1, 3)) : nullptr; // [d, t, h]
    ggml_tensor * k_rope = k_rope_dht ? ggml_cont(ctx_eval, ggml_permute(ctx_eval, k_rope_dht, 0, 2, 1, 3)) : nullptr; // [d, t, h]
    if (q_rope == nullptr || k_rope == nullptr) {
        return nullptr;
    }

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx_eval, q_rope, k_rope, v_dth, &attn_p); // [d, t, h]
    if (attn_ctx == nullptr) {
        return nullptr;
    }
    ggml_tensor * attn_ct = ggml_reshape_2d(
        ctx_eval,
        ggml_cont(ctx_eval, ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
        hidden_dim,
        t);

    ggml_tensor * attn_proj = ggml_mul_mat(ctx_eval, att_c_proj_w, attn_ct); // [hidden, t]
    if (attn_proj == nullptr) {
        return nullptr;
    }
    x_ct = ggml_add(ctx_eval, x_ct, attn_proj);

    ggml_tensor * m = codec_neu_rms_norm_ct(ctx_eval, x_ct, ffn_norm_w);
    if (m == nullptr) {
        return nullptr;
    }
    ggml_tensor * ff = ggml_mul_mat(ctx_eval, mlp_fc1_w, m);
    ff = ggml_silu(ctx_eval, ff);
    ff = ggml_mul_mat(ctx_eval, mlp_fc2_w, ff);
    if (ff == nullptr) {
        return nullptr;
    }
    x_ct = ggml_add(ctx_eval, x_ct, ff);
    return x_ct;
}

static ggml_tensor * codec_neu_attention_full_tc(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * w_q,
    ggml_tensor * b_q,
    ggml_tensor * w_k,
    ggml_tensor * b_k,
    ggml_tensor * w_v,
    ggml_tensor * b_v,
    ggml_tensor * w_o,
    ggml_tensor * b_o,
    int32_t head_dim,
    int32_t n_heads) {

    if (ctx_eval == nullptr || x_tc == nullptr || w_q == nullptr || w_k == nullptr || w_v == nullptr || w_o == nullptr) {
        return nullptr;
    }

    ggml_tensor * x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_tc)); // [c, t]
    ggml_tensor * q_ct = codec_op_linear(ctx_eval, x_ct, w_q, b_q);
    ggml_tensor * k_ct = codec_op_linear(ctx_eval, x_ct, w_k, b_k);
    ggml_tensor * v_ct = codec_op_linear(ctx_eval, x_ct, w_v, b_v);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr) {
        return nullptr;
    }

    const int64_t t = q_ct->ne[1];
    ggml_tensor * q_dht = ggml_reshape_3d(ctx_eval, q_ct, head_dim, n_heads, t); // [d,h,t]
    ggml_tensor * k_dht = ggml_reshape_3d(ctx_eval, k_ct, head_dim, n_heads, t);
    ggml_tensor * v_dth = ggml_permute(ctx_eval, ggml_reshape_3d(ctx_eval, v_ct, head_dim, n_heads, t), 0, 2, 1, 3); // [d,t,h]

    ggml_tensor * q_dth = ggml_cont(ctx_eval, ggml_permute(ctx_eval, q_dht, 0, 2, 1, 3)); // [d,t,h]
    ggml_tensor * k_dth = ggml_cont(ctx_eval, ggml_permute(ctx_eval, k_dht, 0, 2, 1, 3)); // [d,t,h]
    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx_eval, q_dth, k_dth, v_dth, &attn_p); // [d,t,h]
    if (attn_ctx == nullptr) {
        return nullptr;
    }
    ggml_tensor * attn_ct = ggml_reshape_2d(
        ctx_eval,
        ggml_cont(ctx_eval, ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
        head_dim * n_heads,
        t);

    ggml_tensor * out_ct = codec_op_linear(ctx_eval, attn_ct, w_o, b_o);
    if (out_ct == nullptr) {
        return nullptr;
    }
    return ggml_cont(ctx_eval, ggml_transpose(ctx_eval, out_ct)); // [t,c]
}

static ggml_tensor * codec_neu_geglu_tc(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    int32_t inner_dim) {

    if (ctx_eval == nullptr || x_tc == nullptr || inner_dim <= 0) {
        return nullptr;
    }
    const int32_t t = (int32_t) x_tc->ne[0];
    const int32_t c = (int32_t) x_tc->ne[1];
    if (c != inner_dim * 2) {
        return nullptr;
    }

    ggml_tensor * x1 = ggml_view_2d(ctx_eval, x_tc, t, inner_dim, x_tc->nb[1], 0);
    ggml_tensor * x2 = ggml_view_2d(ctx_eval, x_tc, t, inner_dim, x_tc->nb[1], (size_t) inner_dim * x_tc->nb[1]);
    x1 = ggml_cont(ctx_eval, x1);
    x2 = ggml_cont(ctx_eval, x2);
    ggml_tensor * x2_gelu = ggml_gelu_erf(ctx_eval, x2);
    return ggml_mul(ctx_eval, x1, x2_gelu);
}

static ggml_tensor * codec_neu_local_mha_tc(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b,
    ggml_tensor * w_qkv,
    ggml_tensor * w_out,
    int32_t heads,
    int32_t head_dim,
    const codec_local_attn_params * attn_params,
    float ln_eps) {

    if (ctx_eval == nullptr || x_tc == nullptr || w_qkv == nullptr || w_out == nullptr || attn_params == nullptr) {
        return nullptr;
    }
    ggml_tensor * h_tc = x_tc;
    if (ln_w != nullptr && ln_b != nullptr) {
        h_tc = codec_neu_layer_norm_tc_eps(ctx_eval, h_tc, ln_w, ln_b, ln_eps);
        if (h_tc == nullptr) {
            return nullptr;
        }
    }

    ggml_tensor * qkv_tc = codec_neu_linear_tc(ctx_eval, h_tc, w_qkv, nullptr); // [t, 3*dim]
    if (qkv_tc == nullptr) {
        return nullptr;
    }

    const int32_t t = (int32_t) qkv_tc->ne[0];
    const int32_t qkv_out = (int32_t) qkv_tc->ne[1];
    if (qkv_out % 3 != 0) {
        return nullptr;
    }
    const int32_t inner = qkv_out / 3;
    if (inner != heads * head_dim) {
        return nullptr;
    }
    ggml_tensor * q_tc = ggml_view_2d(ctx_eval, qkv_tc, t, inner, qkv_tc->nb[1], 0);
    ggml_tensor * k_tc = ggml_view_2d(ctx_eval, qkv_tc, t, inner, qkv_tc->nb[1], (size_t) inner * qkv_tc->nb[1]);
    ggml_tensor * v_tc = ggml_view_2d(ctx_eval, qkv_tc, t, inner, qkv_tc->nb[1], (size_t) inner * qkv_tc->nb[1] * 2);
    q_tc = ggml_cont(ctx_eval, q_tc);
    k_tc = ggml_cont(ctx_eval, k_tc);
    v_tc = ggml_cont(ctx_eval, v_tc);

    ggml_tensor * q_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, q_tc));
    ggml_tensor * k_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, k_tc));
    ggml_tensor * v_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, v_tc));
    ggml_tensor * q_dht = ggml_reshape_3d(ctx_eval, q_ct, head_dim, heads, t);
    ggml_tensor * k_dht = ggml_reshape_3d(ctx_eval, k_ct, head_dim, heads, t);
    ggml_tensor * v_dht = ggml_reshape_3d(ctx_eval, v_ct, head_dim, heads, t);
    ggml_tensor * q_dth = ggml_cont(ctx_eval, ggml_permute(ctx_eval, q_dht, 0, 2, 1, 3));
    ggml_tensor * k_dth = ggml_cont(ctx_eval, ggml_permute(ctx_eval, k_dht, 0, 2, 1, 3));
    ggml_tensor * v_dth = ggml_cont(ctx_eval, ggml_permute(ctx_eval, v_dht, 0, 2, 1, 3));

    ggml_tensor * attn_dth = codec_op_local_attn(ctx_eval, q_dth, k_dth, v_dth, attn_params);
    if (attn_dth == nullptr) {
        return nullptr;
    }

    ggml_tensor * attn_dht = ggml_cont(ctx_eval, ggml_permute(ctx_eval, attn_dth, 0, 2, 1, 3));
    ggml_tensor * attn_ct = ggml_reshape_2d(ctx_eval, attn_dht, inner, t);
    ggml_tensor * out_tc = codec_neu_linear_tc(ctx_eval, ggml_cont(ctx_eval, ggml_transpose(ctx_eval, attn_ct)), w_out, nullptr);
    return out_tc;
}

static bool codec_neu_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    const std::vector<float> * window,
    std::vector<float> * out_pcm,
    std::string * err) {

    if (out_pcm == nullptr || out_dim <= 0 || n_frames <= 0 || hop <= 0 || (out_dim % 2) != 0) {
        if (err != nullptr) {
            *err = "invalid NeuCodec ISTFT arguments";
        }
        return false;
    }
    const int32_t n_bins = out_dim / 2;
    const int32_t n_fft = 2 * (n_bins - 1);
    const float pi = 3.14159265358979323846f;
    if (n_fft <= 0) {
        if (err != nullptr) {
            *err = "invalid NeuCodec head output dimension";
        }
        return false;
    }
    const int32_t pad = (n_fft - hop) / 2;
    const int32_t out_size = (n_frames - 1) * hop + n_fft;

    std::vector<float> win((size_t) n_fft, 0.0f);
    if (window != nullptr && (int32_t) window->size() == n_fft) {
        win = *window;
    } else {
        for (int32_t n = 0; n < n_fft; ++n) {
            win[(size_t) n] = 0.5f - 0.5f * std::cos(2.0f * pi * (float) n / (float) (n_fft - 1));
        }
    }

    std::vector<float> y((size_t) out_size, 0.0f);
    std::vector<float> env((size_t) out_size, 0.0f);
    std::vector<float> frame((size_t) n_fft, 0.0f);

    for (int32_t ti = 0; ti < n_frames; ++ti) {
        for (int32_t n = 0; n < n_fft; ++n) {
            float sum = 0.0f;
            float mag0 = std::exp(head[(size_t) 0 + (size_t) out_dim * (size_t) ti]);
            if (mag0 > 1e2f) {
                mag0 = 1e2f;
            }
            const float re0 = mag0 * std::cos(head[(size_t) n_bins + (size_t) out_dim * (size_t) ti]);
            sum += re0;
            float magn = std::exp(head[(size_t) (n_bins - 1) + (size_t) out_dim * (size_t) ti]);
            if (magn > 1e2f) {
                magn = 1e2f;
            }
            const float ren = magn * std::cos(head[(size_t) (2 * n_bins - 1) + (size_t) out_dim * (size_t) ti]);
            sum += ren * ((n & 1) ? -1.0f : 1.0f);
            for (int32_t k = 1; k < n_bins - 1; ++k) {
                float mag = std::exp(head[(size_t) k + (size_t) out_dim * (size_t) ti]);
                if (mag > 1e2f) {
                    mag = 1e2f;
                }
                const float ph = head[(size_t) (n_bins + k) + (size_t) out_dim * (size_t) ti];
                const float re = mag * std::cos(ph);
                const float im = mag * std::sin(ph);
                const float ang = 2.0f * pi * (float) k * (float) n / (float) n_fft;
                sum += 2.0f * (re * std::cos(ang) - im * std::sin(ang));
            }
            frame[(size_t) n] = (sum / (float) n_fft) * win[(size_t) n];
        }
        const int32_t off = ti * hop;
        for (int32_t n = 0; n < n_fft; ++n) {
            y[(size_t) (off + n)] += frame[(size_t) n];
            env[(size_t) (off + n)] += win[(size_t) n] * win[(size_t) n];
        }
    }

    const int32_t out_begin = std::max(0, pad);
    const int32_t out_end = std::max(out_begin, out_size - pad);
    out_pcm->assign((size_t) (out_end - out_begin), 0.0f);
    for (int32_t i = out_begin; i < out_end; ++i) {
        const float den = env[(size_t) i] > 1e-11f ? env[(size_t) i] : 1.0f;
        (*out_pcm)[(size_t) (i - out_begin)] = y[(size_t) i] / den;
    }
    return true;
}

struct neucodec_decode_build {
    int32_t t = 0;
    int32_t q = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    int32_t vq_dim = 0;
    int32_t hidden_dim = 0;
    int32_t num_layers = 0;
    int32_t num_heads = 0;
    int32_t head_dim = 0;
    int32_t head_out_dim = 0;
    float rope_theta = 10000.0f;
};

static bool codec_neu_build_decode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    neucodec_decode_build * p = static_cast<neucodec_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr) {
        return false;
    }
    if (p->t <= 0 || p->q <= 0 || p->codebook_dim <= 0 || p->codebook_size <= 0 ||
        p->vq_dim <= 0 || p->hidden_dim <= 0 || p->num_layers <= 0 || p->num_heads <= 0 || p->head_dim <= 0 ||
        p->head_out_dim <= 0) {
        return false;
    }

    ggml_tensor * t_tok = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_I32, p->t, p->q);
    ggml_set_name(t_tok, codec_neu_name_tok().c_str());

    ggml_tensor * t_codebook = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->codebook_dim, p->codebook_size);
    ggml_set_name(t_codebook, codec_neu_name_codebook().c_str());

    ggml_tensor * t_idx = ggml_view_1d(ctx_eval, t_tok, p->t, 0);
    ggml_tensor * t_q = ggml_get_rows(ctx_eval, t_codebook, t_idx); // [codebook_dim, t]

    ggml_tensor * t_qp_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->codebook_dim, p->vq_dim);
    ggml_set_name(t_qp_w, codec_neu_name_quant_w().c_str());
    ggml_tensor * t_qp_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->vq_dim);
    ggml_set_name(t_qp_b, codec_neu_name_quant_b().c_str());

    ggml_tensor * x_ct = codec_op_linear(ctx_eval, t_q, t_qp_w, t_qp_b); // [vq_dim, t]
    if (x_ct == nullptr) {
        return false;
    }

    ggml_tensor * t_fc_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->vq_dim, p->hidden_dim);
    ggml_set_name(t_fc_w, codec_neu_name_fc_post_w().c_str());
    ggml_tensor * t_fc_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
    ggml_set_name(t_fc_b, codec_neu_name_fc_post_b().c_str());
    x_ct = codec_op_linear(ctx_eval, x_ct, t_fc_w, t_fc_b); // [hidden_dim, t]
    if (x_ct == nullptr) {
        return false;
    }

    ggml_tensor * x_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_ct)); // [t, hidden]

    ggml_tensor * t_embed_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 7, p->hidden_dim, p->hidden_dim);
    ggml_set_name(t_embed_w, codec_neu_name_embed_w().c_str());
    ggml_tensor * t_embed_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
    ggml_set_name(t_embed_b, codec_neu_name_embed_b().c_str());
    x_tc = codec_conv1d(ctx_eval, x_tc, t_embed_w, t_embed_b, 1, 1, 3);
    if (x_tc == nullptr) {
        return false;
    }

    for (int32_t li = 0; li < 2; ++li) {
        ggml_tensor * n1_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n1_w, codec_neu_name_prior(li, "norm1.w").c_str());
        ggml_tensor * n1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n1_b, codec_neu_name_prior(li, "norm1.b").c_str());
        ggml_tensor * c1_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, p->hidden_dim, p->hidden_dim);
        ggml_set_name(c1_w, codec_neu_name_prior(li, "conv1.w").c_str());
        ggml_tensor * c1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(c1_b, codec_neu_name_prior(li, "conv1.b").c_str());
        ggml_tensor * n2_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n2_w, codec_neu_name_prior(li, "norm2.w").c_str());
        ggml_tensor * n2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n2_b, codec_neu_name_prior(li, "norm2.b").c_str());
        ggml_tensor * c2_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, p->hidden_dim, p->hidden_dim);
        ggml_set_name(c2_w, codec_neu_name_prior(li, "conv2.w").c_str());
        ggml_tensor * c2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(c2_b, codec_neu_name_prior(li, "conv2.b").c_str());

        x_tc = codec_neu_resnet_block(ctx_eval, x_tc, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
        if (x_tc == nullptr) {
            return false;
        }
    }

    x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_tc)); // [hidden, t]

    for (int32_t li = 0; li < p->num_layers; ++li) {
        ggml_tensor * attn_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(attn_w, codec_neu_name_transformer(li, "att_norm.w").c_str());
        ggml_tensor * ffn_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(ffn_w, codec_neu_name_transformer(li, "ffn_norm.w").c_str());
        ggml_tensor * c_attn_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hidden_dim, 3 * p->hidden_dim);
        ggml_set_name(c_attn_w, codec_neu_name_transformer(li, "att.c_attn.w").c_str());
        ggml_tensor * c_proj_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hidden_dim, p->hidden_dim);
        ggml_set_name(c_proj_w, codec_neu_name_transformer(li, "att.c_proj.w").c_str());
        ggml_tensor * fc1_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hidden_dim, 4 * p->hidden_dim);
        ggml_set_name(fc1_w, codec_neu_name_transformer(li, "mlp.fc1.w").c_str());
        ggml_tensor * fc2_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, 4 * p->hidden_dim, p->hidden_dim);
        ggml_set_name(fc2_w, codec_neu_name_transformer(li, "mlp.fc2.w").c_str());

        x_ct = codec_neu_transformer_block(
            ctx_eval,
            x_ct,
            attn_w,
            ffn_w,
            c_attn_w,
            c_proj_w,
            fc1_w,
            fc2_w,
            p->head_dim,
            p->num_heads,
            p->rope_theta);
        if (x_ct == nullptr) {
            return false;
        }
    }

    x_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_ct)); // [t, hidden]

    for (int32_t li = 0; li < 2; ++li) {
        ggml_tensor * n1_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n1_w, codec_neu_name_post(li, "norm1.w").c_str());
        ggml_tensor * n1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n1_b, codec_neu_name_post(li, "norm1.b").c_str());
        ggml_tensor * c1_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, p->hidden_dim, p->hidden_dim);
        ggml_set_name(c1_w, codec_neu_name_post(li, "conv1.w").c_str());
        ggml_tensor * c1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(c1_b, codec_neu_name_post(li, "conv1.b").c_str());
        ggml_tensor * n2_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n2_w, codec_neu_name_post(li, "norm2.w").c_str());
        ggml_tensor * n2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(n2_b, codec_neu_name_post(li, "norm2.b").c_str());
        ggml_tensor * c2_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, p->hidden_dim, p->hidden_dim);
        ggml_set_name(c2_w, codec_neu_name_post(li, "conv2.w").c_str());
        ggml_tensor * c2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
        ggml_set_name(c2_b, codec_neu_name_post(li, "conv2.b").c_str());

        x_tc = codec_neu_resnet_block(ctx_eval, x_tc, n1_w, n1_b, c1_w, c1_b, n2_w, n2_b, c2_w, c2_b);
        if (x_tc == nullptr) {
            return false;
        }
    }

    x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_tc)); // [hidden, t]

    ggml_tensor * t_fln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
    ggml_set_name(t_fln_w, codec_neu_name_final_ln_w().c_str());
    ggml_tensor * t_fln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hidden_dim);
    ggml_set_name(t_fln_b, codec_neu_name_final_ln_b().c_str());
    x_ct = codec_neu_layer_norm_ct(ctx_eval, x_ct, t_fln_w, t_fln_b);
    if (x_ct == nullptr) {
        return false;
    }

    ggml_tensor * t_head_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hidden_dim, p->head_out_dim);
    ggml_set_name(t_head_w, codec_neu_name_head_w().c_str());
    ggml_tensor * t_head_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->head_out_dim);
    ggml_set_name(t_head_b, codec_neu_name_head_b().c_str());
    ggml_tensor * t_out = codec_op_linear(ctx_eval, x_ct, t_head_w, t_head_b); // [out_dim, t]
    if (t_out == nullptr) {
        return false;
    }
    ggml_set_name(t_out, "neucodec.decode.head.out");
    *out = t_out;
    return true;
}

static bool codec_neu_copy_linear_weight_to_2d(
    codec_context * ctx,
    const std::string & src_name,
    ggml_tensor * dst,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || dst == nullptr) {
        if (err != nullptr) {
            *err = "invalid NeuCodec linear copy arguments";
        }
        return false;
    }
    ggml_tensor * src = codec_neu_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing NeuCodec tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> v;
    if (!codec_tensor_as_vec_f32(src, &v)) {
        if (err != nullptr) {
            *err = "failed reading NeuCodec tensor: " + src_name;
        }
        return false;
    }
    if ((int32_t) codec_ne(src, 0) != (int32_t) dst->ne[0] || (int32_t) codec_ne(src, 1) != (int32_t) dst->ne[1]) {
        if (err != nullptr) {
            *err = "unexpected NeuCodec linear shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, v.data(), v.size() * sizeof(float), err);
}

static bool codec_neu_copy_conv1d_weight_to_3d(
    codec_context * ctx,
    const std::string & src_name,
    ggml_tensor * dst,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || dst == nullptr) {
        if (err != nullptr) {
            *err = "invalid NeuCodec conv copy arguments";
        }
        return false;
    }
    ggml_tensor * src = codec_neu_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing NeuCodec tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> v;
    if (!codec_tensor_as_vec_f32(src, &v)) {
        if (err != nullptr) {
            *err = "failed reading NeuCodec tensor: " + src_name;
        }
        return false;
    }
    if ((int32_t) codec_ne(src, 0) != (int32_t) dst->ne[0] ||
        (int32_t) codec_ne(src, 1) != (int32_t) dst->ne[1] ||
        (int32_t) codec_ne(src, 2) != (int32_t) dst->ne[2]) {
        if (err != nullptr) {
            *err = "unexpected NeuCodec conv1d shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, v.data(), v.size() * sizeof(float), err);
}

static bool codec_neu_copy_bias_1d(
    codec_context * ctx,
    const std::string & src_name,
    ggml_tensor * dst,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || dst == nullptr) {
        if (err != nullptr) {
            *err = "invalid NeuCodec bias copy arguments";
        }
        return false;
    }
    ggml_tensor * src = codec_neu_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing NeuCodec tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> v;
    if (!codec_tensor_as_vec_f32(src, &v) || (int32_t) v.size() != (int32_t) dst->ne[0]) {
        if (err != nullptr) {
            *err = "invalid NeuCodec bias tensor: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, v.data(), v.size() * sizeof(float), err);
}

static bool codec_neu_init_decode_build(
    codec_context * ctx,
    const codec_neucodec * neu,
    int32_t t,
    int32_t q,
    neucodec_decode_build * build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || neu == nullptr || build == nullptr || t <= 0 || q <= 0) {
        if (err != nullptr) {
            *err = "invalid NeuCodec decode build arguments";
        }
        return false;
    }

    if (neu->hidden_dim != neu->num_heads * neu->head_dim) {
        if (err != nullptr) {
            *err = "NeuCodec head_dim * num_heads mismatch";
        }
        return false;
    }

    build->t = t;
    build->q = q;
    build->codebook_dim = neu->codebook_dim;
    build->codebook_size = neu->codebook_size;
    build->vq_dim = neu->vq_dim;
    build->hidden_dim = neu->hidden_dim;
    build->num_layers = neu->num_layers;
    build->num_heads = neu->num_heads;
    build->head_dim = neu->head_dim;
    build->head_out_dim = neu->n_fft + 2;
    build->rope_theta = neu->rope_theta;

    if (build->head_out_dim <= 0) {
        if (err != nullptr) {
            *err = "invalid NeuCodec head output dimension";
        }
        return false;
    }
    return true;
}

struct neucodec_encode_build {
    int32_t n_in = 0;
    int32_t n_in_sem = 0;
    int32_t n_q = 1;
    int32_t codebook_dim = 8;
    int32_t codebook_size = 65536;
    int32_t encoder_type = 0;
    int32_t hubert_hidden = 768;
    int32_t hubert_heads = 12;
    int32_t hubert_intermediate = 3072;
    int32_t hubert_layers = 2;
    int32_t hubert_pos_k = 128;
    int32_t hubert_pos_groups = 16;
    float hubert_ln_eps = 1e-5f;
    int32_t hubert_feat_layers = 7;
    int32_t hubert_conv_dim[7] = { 512, 512, 512, 512, 512, 512, 512 };
    int32_t hubert_conv_kernel[7] = { 10, 3, 3, 3, 3, 2, 2 };
    int32_t hubert_conv_stride[7] = { 5, 2, 2, 2, 2, 2, 2 };
    int32_t local_window = 300;
    int32_t local_down_window = 1500;
    const codec_local_attn_params * down_attn = nullptr;
    const codec_local_attn_params * local_attn = nullptr;
};

static ggml_tensor * codec_neu_build_distill_first_block(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc) {

    if (ctx_eval == nullptr || x_tc == nullptr) {
        return nullptr;
    }

    const int32_t pool_kernels[5] = { 1, 5, 11, 21, 45 };
    ggml_tensor * concat = nullptr;
    for (int32_t i = 0; i < 5; ++i) {
        const int32_t k = pool_kernels[i];
        ggml_tensor * x_abs = ggml_abs(ctx_eval, x_tc);
        ggml_tensor * x_max = codec_op_max_pool1d(ctx_eval, x_abs, k, k / 2);
        ggml_tensor * x_avg = codec_op_avg_pool1d(ctx_eval, x_max, k, k / 2);

        ggml_tensor * w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 7, 1, 4);
        codec_neu_set_enc_name(w, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.blocks." + std::to_string(i) + ".1.weight");
        ggml_tensor * b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 4);
        codec_neu_set_enc_name(b, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.blocks." + std::to_string(i) + ".1.bias");
        ggml_tensor * y = codec_conv1d(ctx_eval, x_avg, w, b, 1, 1, 3);
        if (y == nullptr) {
            return nullptr;
        }
        concat = concat == nullptr ? y : ggml_concat(ctx_eval, concat, y, 1);
    }

    ggml_tensor * w1 = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 1, 20, 80);
    codec_neu_set_enc_name(w1, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.weight");
    ggml_tensor * b1 = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 80);
    codec_neu_set_enc_name(b1, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.bias");
    ggml_tensor * h = codec_conv1d(ctx_eval, concat, w1, b1, 1, 1, 0);
    if (h == nullptr) {
        return nullptr;
    }
    h = ggml_gelu_erf(ctx_eval, h);

    ggml_tensor * x_cat = ggml_concat(ctx_eval, h, x_tc, 1);
    ggml_tensor * w2 = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 1, 81, 32);
    codec_neu_set_enc_name(w2, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.weight");
    ggml_tensor * b2 = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 32);
    codec_neu_set_enc_name(b2, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.bias");
    ggml_tensor * y = codec_conv1d(ctx_eval, x_cat, w2, b2, 1, 1, 0);
    return y;
}

static ggml_tensor * codec_neu_build_distill_base_unit(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    const std::string & prefix) {

    if (ctx_eval == nullptr || x_tc == nullptr) {
        return nullptr;
    }
    const int32_t dim = (int32_t) x_tc->ne[1];
    ggml_tensor * dw_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 7, 1, dim);
    codec_neu_set_enc_name(dw_w, prefix + ".dw_conv.weight");
    ggml_tensor * dw_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, dim);
    codec_neu_set_enc_name(dw_b, prefix + ".dw_conv.bias");
    ggml_tensor * h = codec_conv1d_depthwise(ctx_eval, x_tc, dw_w, dw_b, 1, 1, 3);
    if (h == nullptr) {
        return nullptr;
    }

    ggml_tensor * pw1_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, dim, 4 * dim);
    codec_neu_set_enc_name(pw1_w, prefix + ".pw_conv1.weight");
    ggml_tensor * pw1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 4 * dim);
    codec_neu_set_enc_name(pw1_b, prefix + ".pw_conv1.bias");
    h = codec_neu_linear_tc(ctx_eval, h, pw1_w, pw1_b);
    if (h == nullptr) {
        return nullptr;
    }

    ggml_tensor * act_a = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 4 * dim);
    codec_neu_set_enc_name(act_a, prefix + ".act.alpha");
    h = codec_neu_snake_tc(ctx_eval, h, act_a, 1.1920929e-7f);
    if (h == nullptr) {
        return nullptr;
    }

    ggml_tensor * grn_g = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 4 * dim);
    codec_neu_set_enc_name(grn_g, prefix + ".grn.gamma");
    ggml_tensor * grn_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 4 * dim);
    codec_neu_set_enc_name(grn_b, prefix + ".grn.beta");
    h = codec_neu_grn_tc(ctx_eval, h, grn_g, grn_b, 1.1920929e-7f);
    if (h == nullptr) {
        return nullptr;
    }

    ggml_tensor * pw2_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, 4 * dim, dim);
    codec_neu_set_enc_name(pw2_w, prefix + ".pw_conv2.weight");
    ggml_tensor * pw2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, dim);
    codec_neu_set_enc_name(pw2_b, prefix + ".pw_conv2.bias");
    h = codec_neu_linear_tc(ctx_eval, h, pw2_w, pw2_b);
    if (h == nullptr) {
        return nullptr;
    }

    return ggml_add(ctx_eval, x_tc, h);
}

static ggml_tensor * codec_neu_build_distill_local_trans(
    ggml_context * ctx_eval,
    ggml_tensor * x_tc,
    const std::string & prefix,
    int32_t depth,
    int32_t heads,
    int32_t head_dim,
    const codec_local_attn_params * attn_params) {

    if (ctx_eval == nullptr || x_tc == nullptr || depth <= 0 || attn_params == nullptr) {
        return nullptr;
    }

    const int32_t dim = (int32_t) x_tc->ne[1];
    const int32_t attn_inner = heads * head_dim;
    const int32_t inner_dim = (int32_t) (dim * 4 * 2 / 3);

    for (int32_t li = 0; li < depth; ++li) {
        // LocalMHA
        ggml_tensor * ln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, dim);
        codec_neu_set_enc_name(ln_w, prefix + ".layers." + std::to_string(li) + ".0.norm.weight");
        ggml_tensor * ln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, dim);
        codec_neu_set_enc_name(ln_b, prefix + ".layers." + std::to_string(li) + ".0.norm.bias");
        ggml_tensor * w_qkv = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, dim, 3 * attn_inner);
        codec_neu_set_enc_name(w_qkv, prefix + ".layers." + std::to_string(li) + ".0.to_qkv.weight");
        ggml_tensor * w_out = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, attn_inner, dim);
        codec_neu_set_enc_name(w_out, prefix + ".layers." + std::to_string(li) + ".0.to_out.weight");

        ggml_tensor * attn = codec_neu_local_mha_tc(ctx_eval, x_tc, ln_w, ln_b, w_qkv, w_out, heads, head_dim, attn_params, 1e-5f);
        if (attn == nullptr) {
            return nullptr;
        }
        x_tc = ggml_add(ctx_eval, x_tc, attn);

        // FeedForward
        ggml_tensor * ff_ln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, dim);
        codec_neu_set_enc_name(ff_ln_w, prefix + ".layers." + std::to_string(li) + ".1.0.weight");
        ggml_tensor * ff_ln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, dim);
        codec_neu_set_enc_name(ff_ln_b, prefix + ".layers." + std::to_string(li) + ".1.0.bias");
        ggml_tensor * ff_w1 = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, dim, inner_dim * 2);
        codec_neu_set_enc_name(ff_w1, prefix + ".layers." + std::to_string(li) + ".1.1.weight");
        ggml_tensor * ff_w2 = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, inner_dim, dim);
        codec_neu_set_enc_name(ff_w2, prefix + ".layers." + std::to_string(li) + ".1.4.weight");

        ggml_tensor * ff = codec_neu_layer_norm_tc_eps(ctx_eval, x_tc, ff_ln_w, ff_ln_b, 1e-5f);
        if (ff == nullptr) {
            return nullptr;
        }
        ff = codec_neu_linear_tc(ctx_eval, ff, ff_w1, nullptr);
        if (ff == nullptr) {
            return nullptr;
        }
        ff = codec_neu_geglu_tc(ctx_eval, ff, inner_dim);
        if (ff == nullptr) {
            return nullptr;
        }
        ff = codec_neu_linear_tc(ctx_eval, ff, ff_w2, nullptr);
        if (ff == nullptr) {
            return nullptr;
        }
        x_tc = ggml_add(ctx_eval, x_tc, ff);
    }

    return x_tc;
}

static bool codec_neu_build_encode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    neucodec_encode_build * p = static_cast<neucodec_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr) {
        return false;
    }
    if (p->n_in <= 0 || p->n_in_sem <= 0) {
        return false;
    }
    if (p->encoder_type != 1) {
        return false;
    }

    ggml_tensor * t_pcm = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->n_in, 1);
    codec_neu_set_enc_name(t_pcm, "neucodec.encode.pcm");
    ggml_tensor * t_sem = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->n_in_sem, 1);
    codec_neu_set_enc_name(t_sem, "neucodec.encode.sem");

    // Distill acoustic encoder
    ggml_tensor * x = codec_neu_build_distill_first_block(ctx_eval, t_pcm);
    if (x == nullptr) {
        return false;
    }

    // stage 0
    x = codec_neu_build_distill_base_unit(ctx_eval, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.1.0.module");
    if (x == nullptr) return false;
    ggml_tensor * d0_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 4, 32, 64);
    codec_neu_set_enc_name(d0_w, "neucodec.encode.distill.codec_encoder.encoder.blocks.2.0.weight");
    ggml_tensor * d0_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 64);
    codec_neu_set_enc_name(d0_b, "neucodec.encode.distill.codec_encoder.encoder.blocks.2.0.bias");
    x = codec_conv1d(ctx_eval, x, d0_w, d0_b, 4, 1, 0);
    if (x == nullptr) return false;

    // stage 1
    x = codec_neu_build_distill_base_unit(ctx_eval, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.3.0.module");
    if (x == nullptr) return false;
    ggml_tensor * d1_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 4, 64, 128);
    codec_neu_set_enc_name(d1_w, "neucodec.encode.distill.codec_encoder.encoder.blocks.4.0.weight");
    ggml_tensor * d1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 128);
    codec_neu_set_enc_name(d1_b, "neucodec.encode.distill.codec_encoder.encoder.blocks.4.0.bias");
    x = codec_conv1d(ctx_eval, x, d1_w, d1_b, 4, 1, 0);
    if (x == nullptr) return false;

    // stage 2
    x = codec_neu_build_distill_base_unit(ctx_eval, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.5.0.module");
    if (x == nullptr) return false;
    ggml_tensor * d2_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 4, 128, 256);
    codec_neu_set_enc_name(d2_w, "neucodec.encode.distill.codec_encoder.encoder.blocks.6.0.weight");
    ggml_tensor * d2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 256);
    codec_neu_set_enc_name(d2_b, "neucodec.encode.distill.codec_encoder.encoder.blocks.6.0.bias");
    x = codec_conv1d(ctx_eval, x, d2_w, d2_b, 4, 1, 0);
    if (x == nullptr) return false;

    // final stage (2 blocks)
    x = codec_neu_build_distill_base_unit(ctx_eval, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.7.0.module");
    if (x == nullptr) return false;
    x = codec_neu_build_distill_base_unit(ctx_eval, x, "neucodec.encode.distill.codec_encoder.encoder.blocks.7.1.module");
    if (x == nullptr) return false;
    ggml_tensor * d3_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, 256, 512);
    codec_neu_set_enc_name(d3_w, "neucodec.encode.distill.codec_encoder.encoder.blocks.8.weight");
    ggml_tensor * d3_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 512);
    codec_neu_set_enc_name(d3_b, "neucodec.encode.distill.codec_encoder.encoder.blocks.8.bias");
    x = codec_conv1d(ctx_eval, x, d3_w, d3_b, 1, 1, 1);
    if (x == nullptr) return false;
    codec_neu_set_enc_name(x, "neucodec.encode.debug.acoustic_tc");

    // en_encoder down_trans
    x = codec_neu_build_distill_local_trans(
        ctx_eval,
        x,
        "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.trans",
        2,
        6,
        512 / 4,
        p->down_attn);
    if (x == nullptr) {
        return false;
    }
    ggml_tensor * down_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 5, 512, 512);
    codec_neu_set_enc_name(down_w, "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.weight");
    ggml_tensor * down_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 512);
    codec_neu_set_enc_name(down_b, "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.bias");
    x = codec_conv1d(ctx_eval, x, down_w, down_b, 5, 1, 0);
    if (x == nullptr) return false;
    codec_neu_set_enc_name(x, "neucodec.encode.debug.down_tc");

    // en_encoder local_trans
    x = codec_neu_build_distill_local_trans(
        ctx_eval,
        x,
        "neucodec.encode.distill.codec_encoder.en_encoder.local_trans",
        3,
        6,
        512 / 4,
        p->local_attn);
    if (x == nullptr) {
        return false;
    }
    codec_neu_set_enc_name(x, "neucodec.encode.debug.local_tc");

    // fc_sq_prior (512->768)
    ggml_tensor * t_fc_sq_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, 512, 768);
    codec_neu_set_enc_name(t_fc_sq_w, "neucodec.encode.fc_sq_prior.w");
    ggml_tensor * t_fc_sq_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 768);
    codec_neu_set_enc_name(t_fc_sq_b, "neucodec.encode.fc_sq_prior.b");
    ggml_tensor * fsq_tc = codec_neu_linear_tc(ctx_eval, x, t_fc_sq_w, t_fc_sq_b);
    if (fsq_tc == nullptr) return false;
    codec_neu_set_enc_name(fsq_tc, "neucodec.encode.debug.fsq_tc");

    // HuBERT semantic model
    ggml_tensor * sem = t_sem;
    for (int32_t li = 0; li < p->hubert_feat_layers; ++li) {
        ggml_tensor * w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, p->hubert_conv_kernel[li], li == 0 ? 1 : p->hubert_conv_dim[li - 1], p->hubert_conv_dim[li]);
        codec_neu_set_enc_name(w, "neucodec.encode.hubert.feat.conv." + std::to_string(li) + ".w");
        sem = codec_conv1d(ctx_eval, sem, w, nullptr, p->hubert_conv_stride[li], 1, 0);
        if (sem == nullptr) return false;
        if (li == 0) {
            ggml_tensor * gn_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_conv_dim[li]);
            codec_neu_set_enc_name(gn_w, "neucodec.encode.hubert.feat.conv.0.gn.w");
            ggml_tensor * gn_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_conv_dim[li]);
            codec_neu_set_enc_name(gn_b, "neucodec.encode.hubert.feat.conv.0.gn.b");
            sem = codec_op_group_norm(ctx_eval, sem, p->hubert_conv_dim[li], p->hubert_ln_eps, gn_w, gn_b);
        }
        sem = ggml_gelu_erf(ctx_eval, sem);
    }

    ggml_tensor * feat_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hubert_conv_dim[p->hubert_feat_layers - 1], p->hubert_hidden);
    codec_neu_set_enc_name(feat_w, "neucodec.encode.hubert.feature_projection.w");
    ggml_tensor * feat_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
    codec_neu_set_enc_name(feat_b, "neucodec.encode.hubert.feature_projection.b");
    ggml_tensor * hs = codec_neu_linear_tc(ctx_eval, sem, feat_w, feat_b);
    if (hs == nullptr) return false;

    // positional conv
    ggml_tensor * pos_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, p->hubert_pos_k, p->hubert_hidden / p->hubert_pos_groups, p->hubert_hidden);
    codec_neu_set_enc_name(pos_w, "neucodec.encode.hubert.encoder.pos_conv.w");
    ggml_tensor * pos_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
    codec_neu_set_enc_name(pos_b, "neucodec.encode.hubert.encoder.pos_conv.b");
    ggml_tensor * pos = codec_neu_conv1d_grouped(ctx_eval, hs, pos_w, pos_b, 1, 1, p->hubert_pos_k / 2, p->hubert_pos_groups);
    if (pos == nullptr) return false;
    if ((p->hubert_pos_k % 2) == 0) {
        pos = codec_op_crop_1d(ctx_eval, pos, 0, 1);
    }
    pos = ggml_gelu_erf(ctx_eval, pos);
    hs = ggml_add(ctx_eval, hs, pos);

    // encoder layer norm
    ggml_tensor * enc_ln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
    codec_neu_set_enc_name(enc_ln_w, "neucodec.encode.hubert.encoder.layer_norm.w");
    ggml_tensor * enc_ln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
    codec_neu_set_enc_name(enc_ln_b, "neucodec.encode.hubert.encoder.layer_norm.b");
    hs = codec_neu_layer_norm_tc_eps(ctx_eval, hs, enc_ln_w, enc_ln_b, p->hubert_ln_eps);
    if (hs == nullptr) return false;

    const int32_t h_head_dim = p->hubert_hidden / p->hubert_heads;
    for (int32_t li = 0; li < p->hubert_layers; ++li) {
        ggml_tensor * res = hs;
        ggml_tensor * q_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden, p->hubert_hidden);
        codec_neu_set_enc_name(q_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.q.w");
        ggml_tensor * q_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(q_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.q.b");
        ggml_tensor * k_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden, p->hubert_hidden);
        codec_neu_set_enc_name(k_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.k.w");
        ggml_tensor * k_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(k_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.k.b");
        ggml_tensor * v_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden, p->hubert_hidden);
        codec_neu_set_enc_name(v_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.v.w");
        ggml_tensor * v_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(v_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.v.b");
        ggml_tensor * o_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden, p->hubert_hidden);
        codec_neu_set_enc_name(o_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.o.w");
        ggml_tensor * o_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(o_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".att.o.b");

        ggml_tensor * attn = codec_neu_attention_full_tc(ctx_eval, hs, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, h_head_dim, p->hubert_heads);
        if (attn == nullptr) return false;
        hs = ggml_add(ctx_eval, res, attn);

        ggml_tensor * ln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(ln_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ln.w");
        ggml_tensor * ln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(ln_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ln.b");
        hs = codec_neu_layer_norm_tc_eps(ctx_eval, hs, ln_w, ln_b, p->hubert_ln_eps);
        if (hs == nullptr) return false;

        ggml_tensor * ff1_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden, p->hubert_intermediate);
        codec_neu_set_enc_name(ff1_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ffn.fc1.w");
        ggml_tensor * ff1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_intermediate);
        codec_neu_set_enc_name(ff1_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ffn.fc1.b");
        ggml_tensor * ff2_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->hubert_intermediate, p->hubert_hidden);
        codec_neu_set_enc_name(ff2_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ffn.fc2.w");
        ggml_tensor * ff2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(ff2_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ffn.fc2.b");
        ggml_tensor * ff = codec_neu_linear_tc(ctx_eval, hs, ff1_w, ff1_b);
        if (ff == nullptr) return false;
        ff = ggml_gelu_erf(ctx_eval, ff);
        ff = codec_neu_linear_tc(ctx_eval, ff, ff2_w, ff2_b);
        if (ff == nullptr) return false;
        hs = ggml_add(ctx_eval, hs, ff);

        ggml_tensor * ffn_ln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(ffn_ln_w, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ffn_ln.w");
        ggml_tensor * ffn_ln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->hubert_hidden);
        codec_neu_set_enc_name(ffn_ln_b, "neucodec.encode.hubert.encoder.layers." + std::to_string(li) + ".ffn_ln.b");
        hs = codec_neu_layer_norm_tc_eps(ctx_eval, hs, ffn_ln_w, ffn_ln_b, p->hubert_ln_eps);
        if (hs == nullptr) return false;
    }
    codec_neu_set_enc_name(hs, "neucodec.encode.debug.hubert_hs");

    // Semantic encoder conv stack
    ggml_tensor * sem_init_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, p->hubert_hidden, 1024);
    codec_neu_set_enc_name(sem_init_w, "neucodec.encode.semantic_encoder.initial_conv.w");
    ggml_tensor * sem1_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, 1024, 1024);
    codec_neu_set_enc_name(sem1_w, "neucodec.encode.semantic_encoder.residual.1.w");
    ggml_tensor * sem1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 1024);
    codec_neu_set_enc_name(sem1_b, "neucodec.encode.semantic_encoder.residual.1.b");
    ggml_tensor * sem2_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, 1024, 1024);
    codec_neu_set_enc_name(sem2_w, "neucodec.encode.semantic_encoder.residual.3.w");
    ggml_tensor * sem2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 1024);
    codec_neu_set_enc_name(sem2_b, "neucodec.encode.semantic_encoder.residual.3.b");
    ggml_tensor * sem_out_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 3, 1024, 768);
    codec_neu_set_enc_name(sem_out_w, "neucodec.encode.semantic_encoder.final_conv.w");

    ggml_tensor * sem_tc = codec_conv1d(ctx_eval, hs, sem_init_w, nullptr, 1, 1, 1);
    if (sem_tc == nullptr) return false;
    codec_neu_set_enc_name(sem_tc, "neucodec.encode.debug.sem_init_tc");
    sem_tc = ggml_relu(ctx_eval, sem_tc);
    codec_neu_set_enc_name(sem_tc, "neucodec.encode.debug.sem_relu1_tc");
    // SemanticEncoder uses inplace ReLU in residual path.
    ggml_tensor * sem_res = sem_tc;
    sem_tc = codec_conv1d(ctx_eval, sem_tc, sem1_w, sem1_b, 1, 1, 1);
    codec_neu_set_enc_name(sem_tc, "neucodec.encode.debug.sem_conv1_tc");
    sem_tc = ggml_relu(ctx_eval, sem_tc);
    codec_neu_set_enc_name(sem_tc, "neucodec.encode.debug.sem_relu2_tc");
    sem_tc = codec_conv1d(ctx_eval, sem_tc, sem2_w, sem2_b, 1, 1, 1);
    codec_neu_set_enc_name(sem_tc, "neucodec.encode.debug.sem_conv2_tc");
    sem_tc = ggml_add(ctx_eval, sem_tc, sem_res);
    codec_neu_set_enc_name(sem_tc, "neucodec.encode.debug.sem_res_tc");
    sem_tc = codec_conv1d(ctx_eval, sem_tc, sem_out_w, nullptr, 1, 1, 1);
    if (sem_tc == nullptr) return false;
    codec_neu_set_enc_name(sem_tc, "neucodec.encode.debug.sem_tc");

    // match lengths
    if (sem_tc->ne[0] != fsq_tc->ne[0]) {
        const int32_t min_t = (int32_t) std::min(sem_tc->ne[0], fsq_tc->ne[0]);
        sem_tc = codec_op_crop_1d(ctx_eval, sem_tc, 0, (int32_t) sem_tc->ne[0] - min_t);
        fsq_tc = codec_op_crop_1d(ctx_eval, fsq_tc, 0, (int32_t) fsq_tc->ne[0] - min_t);
    }

    ggml_tensor * concat = ggml_concat(ctx_eval, sem_tc, fsq_tc, 1);
    ggml_tensor * fc_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, 1536, 2048);
    codec_neu_set_enc_name(fc_w, "neucodec.encode.fc_prior.w");
    ggml_tensor * fc_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 2048);
    codec_neu_set_enc_name(fc_b, "neucodec.encode.fc_prior.b");
    ggml_tensor * prior_tc = codec_neu_linear_tc(ctx_eval, concat, fc_w, fc_b);
    if (prior_tc == nullptr) return false;
    codec_neu_set_enc_name(prior_tc, "neucodec.encode.debug.prior_tc");

    // FSQ project_in
    ggml_tensor * proj_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, 2048, p->codebook_dim);
    codec_neu_set_enc_name(proj_w, "neucodec.encode.quant.project_in.w");
    ggml_tensor * proj_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->codebook_dim);
    codec_neu_set_enc_name(proj_b, "neucodec.encode.quant.project_in.b");
    ggml_tensor * z_tc = codec_neu_linear_tc(ctx_eval, prior_tc, proj_w, proj_b);
    if (z_tc == nullptr) return false;
    codec_neu_set_enc_name(z_tc, "neucodec.encode.debug.z_tc");

    // FSQ bound + quantize (vector-quantize-pytorch FSQ)
    const float eps = 1e-3f;
    const float half_l = (3.0f * (1.0f + eps)) / 2.0f;
    const float offset = 0.5f;
    const float shift = std::atanh(offset / half_l);
    const float half_width = 2.0f;

    // ResidualFSQ initializes residual with bound(x), then FSQ quantize()
    // applies bound() again before rounding.
    ggml_tensor * z_bound = ggml_tanh(ctx_eval, ggml_scale_bias(ctx_eval, z_tc, 1.0f, shift));
    z_bound = ggml_scale_bias(ctx_eval, z_bound, half_l, -offset);
    z_bound = ggml_tanh(ctx_eval, ggml_scale_bias(ctx_eval, z_bound, 1.0f, shift));
    z_bound = ggml_scale_bias(ctx_eval, z_bound, half_l, -offset);
    codec_neu_set_enc_name(z_bound, "neucodec.encode.debug.z_bound");
    ggml_tensor * zq = ggml_scale(ctx_eval, ggml_round(ctx_eval, z_bound), 1.0f / half_width);
    codec_neu_set_enc_name(zq, "neucodec.encode.debug.zq");

    // indices
    ggml_tensor * basis = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->codebook_dim);
    codec_neu_set_enc_name(basis, "neucodec.encode.fsq.basis");
    ggml_tensor * z_scaled = ggml_scale_bias(ctx_eval, zq, half_width, half_width);
    ggml_tensor * basis_2d = ggml_reshape_2d(ctx_eval, basis, 1, p->codebook_dim);
    ggml_tensor * basis_rep = ggml_repeat(ctx_eval, basis_2d, z_scaled);
    ggml_tensor * z_mul = ggml_mul(ctx_eval, z_scaled, basis_rep);
    ggml_tensor * z_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, z_mul));
    ggml_tensor * idx_sum = ggml_sum_rows(ctx_eval, z_ct);
    ggml_tensor * idx_1d = ggml_reshape_1d(ctx_eval, idx_sum, z_mul->ne[0]);
    ggml_tensor * idx_2d = ggml_reshape_2d(ctx_eval, idx_1d, (int32_t) z_mul->ne[0], 1);
    ggml_tensor * out_t = ggml_cont(ctx_eval, idx_2d);
    codec_neu_set_enc_name(out_t, "neucodec.encode.out");
    *out = out_t;
    return true;
}

static bool codec_neu_write_decode_weights(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    const neucodec_decode_build & build,
    std::string * err) {

    if (ctx == nullptr || entry == nullptr) {
        if (err != nullptr) {
            *err = "invalid NeuCodec write arguments";
        }
        return false;
    }

    auto graph = [&](const std::string & name) -> ggml_tensor * {
        const std::string short_name = codec_neu_encode_name(name);
        return codec_graph_get_tensor(ctx, entry, short_name.c_str());
    };

    if (!codec_neu_copy_linear_weight_to_2d(ctx, codec_neu_name_quant_w(), graph(codec_neu_name_quant_w()), err) ||
        !codec_neu_copy_bias_1d(ctx, codec_neu_name_quant_b(), graph(codec_neu_name_quant_b()), err) ||
        !codec_neu_copy_linear_weight_to_2d(ctx, codec_neu_name_fc_post_w(), graph(codec_neu_name_fc_post_w()), err) ||
        !codec_neu_copy_bias_1d(ctx, codec_neu_name_fc_post_b(), graph(codec_neu_name_fc_post_b()), err) ||
        !codec_neu_copy_conv1d_weight_to_3d(ctx, codec_neu_name_embed_w(), graph(codec_neu_name_embed_w()), err) ||
        !codec_neu_copy_bias_1d(ctx, codec_neu_name_embed_b(), graph(codec_neu_name_embed_b()), err)) {
        return false;
    }

    for (int32_t li = 0; li < 2; ++li) {
        if (!codec_neu_copy_bias_1d(ctx, codec_neu_name_prior(li, "norm1.w"), graph(codec_neu_name_prior(li, "norm1.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_prior(li, "norm1.b"), graph(codec_neu_name_prior(li, "norm1.b")), err) ||
            !codec_neu_copy_conv1d_weight_to_3d(ctx, codec_neu_name_prior(li, "conv1.w"), graph(codec_neu_name_prior(li, "conv1.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_prior(li, "conv1.b"), graph(codec_neu_name_prior(li, "conv1.b")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_prior(li, "norm2.w"), graph(codec_neu_name_prior(li, "norm2.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_prior(li, "norm2.b"), graph(codec_neu_name_prior(li, "norm2.b")), err) ||
            !codec_neu_copy_conv1d_weight_to_3d(ctx, codec_neu_name_prior(li, "conv2.w"), graph(codec_neu_name_prior(li, "conv2.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_prior(li, "conv2.b"), graph(codec_neu_name_prior(li, "conv2.b")), err)) {
            return false;
        }
    }

    for (int32_t li = 0; li < 2; ++li) {
        if (!codec_neu_copy_bias_1d(ctx, codec_neu_name_post(li, "norm1.w"), graph(codec_neu_name_post(li, "norm1.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_post(li, "norm1.b"), graph(codec_neu_name_post(li, "norm1.b")), err) ||
            !codec_neu_copy_conv1d_weight_to_3d(ctx, codec_neu_name_post(li, "conv1.w"), graph(codec_neu_name_post(li, "conv1.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_post(li, "conv1.b"), graph(codec_neu_name_post(li, "conv1.b")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_post(li, "norm2.w"), graph(codec_neu_name_post(li, "norm2.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_post(li, "norm2.b"), graph(codec_neu_name_post(li, "norm2.b")), err) ||
            !codec_neu_copy_conv1d_weight_to_3d(ctx, codec_neu_name_post(li, "conv2.w"), graph(codec_neu_name_post(li, "conv2.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_post(li, "conv2.b"), graph(codec_neu_name_post(li, "conv2.b")), err)) {
            return false;
        }
    }

    for (int32_t li = 0; li < build.num_layers; ++li) {
        if (!codec_neu_copy_bias_1d(ctx, codec_neu_name_transformer(li, "att_norm.w"), graph(codec_neu_name_transformer(li, "att_norm.w")), err) ||
            !codec_neu_copy_bias_1d(ctx, codec_neu_name_transformer(li, "ffn_norm.w"), graph(codec_neu_name_transformer(li, "ffn_norm.w")), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, codec_neu_name_transformer(li, "att.c_attn.w"), graph(codec_neu_name_transformer(li, "att.c_attn.w")), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, codec_neu_name_transformer(li, "att.c_proj.w"), graph(codec_neu_name_transformer(li, "att.c_proj.w")), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, codec_neu_name_transformer(li, "mlp.fc1.w"), graph(codec_neu_name_transformer(li, "mlp.fc1.w")), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, codec_neu_name_transformer(li, "mlp.fc2.w"), graph(codec_neu_name_transformer(li, "mlp.fc2.w")), err)) {
            return false;
        }
    }

    if (!codec_neu_copy_bias_1d(ctx, codec_neu_name_final_ln_w(), graph(codec_neu_name_final_ln_w()), err) ||
        !codec_neu_copy_bias_1d(ctx, codec_neu_name_final_ln_b(), graph(codec_neu_name_final_ln_b()), err) ||
        !codec_neu_copy_linear_weight_to_2d(ctx, codec_neu_name_head_w(), graph(codec_neu_name_head_w()), err) ||
        !codec_neu_copy_bias_1d(ctx, codec_neu_name_head_b(), graph(codec_neu_name_head_b()), err)) {
        return false;
    }

    return true;
}

static bool codec_neu_init_encode_build(
    const codec_neucodec * neu,
    int32_t n_in,
    int32_t n_in_sem,
    neucodec_encode_build * build,
    std::string * err) {

    if (neu == nullptr || build == nullptr || n_in <= 0 || n_in_sem <= 0) {
        if (err) *err = "invalid NeuCodec encode build arguments";
        return false;
    }
    build->n_in = n_in;
    build->n_in_sem = n_in_sem;
    build->n_q = neu->n_q;
    build->codebook_dim = neu->codebook_dim;
    build->codebook_size = neu->codebook_size;
    build->encoder_type = neu->encoder_type;
    build->hubert_hidden = neu->hubert_hidden;
    build->hubert_heads = neu->hubert_heads;
    build->hubert_intermediate = neu->hubert_intermediate;
    build->hubert_layers = neu->hubert_layers;
    build->hubert_pos_k = neu->hubert_pos_k;
    build->hubert_pos_groups = neu->hubert_pos_groups;
    build->hubert_ln_eps = neu->hubert_ln_eps;
    build->hubert_feat_layers = neu->hubert_feat_layers;
    for (int i = 0; i < neu->hubert_feat_layers; ++i) {
        build->hubert_conv_dim[i] = neu->hubert_conv_dim[i];
        build->hubert_conv_kernel[i] = neu->hubert_conv_kernel[i];
        build->hubert_conv_stride[i] = neu->hubert_conv_stride[i];
    }
    build->local_window = 300;
    build->local_down_window = 1500;
    build->down_attn = &neu->distill_attn_down;
    build->local_attn = &neu->distill_attn_local;
    return true;
}

static bool codec_neu_write_encode_weights(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    const neucodec_encode_build & build,
    std::string * err) {

    if (ctx == nullptr || entry == nullptr) {
        if (err) *err = "invalid NeuCodec encode write arguments";
        return false;
    }

    auto graph = [&](const std::string & name) -> ggml_tensor * {
        const std::string short_name = codec_neu_encode_name(name);
        return codec_graph_get_tensor(ctx, entry, short_name.c_str());
    };

    // First block
    for (int32_t i = 0; i < 5; ++i) {
        const std::string base = "neucodec.encode.distill.codec_encoder.encoder.blocks.0.blocks." + std::to_string(i) + ".1";
        if (!codec_neu_copy_conv1d_weight_to_3d(ctx, base + ".weight", graph(base + ".weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".bias", graph(base + ".bias"), err)) {
            return false;
        }
    }
    if (!codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.weight",
                                            graph("neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.weight"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.bias",
                                graph("neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_1.bias"), err) ||
        !codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.weight",
                                            graph("neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.weight"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.bias",
                                graph("neucodec.encode.distill.codec_encoder.encoder.blocks.0.conv_2.bias"), err)) {
        return false;
    }

    // Base units
    const int32_t base_blocks[][2] = { {1, 0}, {3, 0}, {5, 0}, {7, 0}, {7, 1} };
    for (size_t bi = 0; bi < 5; ++bi) {
        const std::string prefix = "neucodec.encode.distill.codec_encoder.encoder.blocks." +
            std::to_string(base_blocks[bi][0]) + "." + std::to_string(base_blocks[bi][1]) + ".module";
        if (!codec_neu_copy_conv1d_weight_to_3d(ctx, prefix + ".dw_conv.weight", graph(prefix + ".dw_conv.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, prefix + ".dw_conv.bias", graph(prefix + ".dw_conv.bias"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, prefix + ".pw_conv1.weight", graph(prefix + ".pw_conv1.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, prefix + ".pw_conv1.bias", graph(prefix + ".pw_conv1.bias"), err) ||
            !codec_neu_copy_bias_1d(ctx, prefix + ".act.alpha", graph(prefix + ".act.alpha"), err) ||
            !codec_neu_copy_bias_1d(ctx, prefix + ".grn.gamma", graph(prefix + ".grn.gamma"), err) ||
            !codec_neu_copy_bias_1d(ctx, prefix + ".grn.beta", graph(prefix + ".grn.beta"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, prefix + ".pw_conv2.weight", graph(prefix + ".pw_conv2.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, prefix + ".pw_conv2.bias", graph(prefix + ".pw_conv2.bias"), err)) {
            return false;
        }
    }

    // Down layers + final conv
    const int32_t down_blocks[] = { 2, 4, 6, 8 };
    for (int32_t i = 0; i < 4; ++i) {
        const std::string base = "neucodec.encode.distill.codec_encoder.encoder.blocks." + std::to_string(down_blocks[i]) + ".0";
        if (down_blocks[i] == 8) {
            if (!codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.distill.codec_encoder.encoder.blocks.8.weight",
                                                    graph("neucodec.encode.distill.codec_encoder.encoder.blocks.8.weight"), err) ||
                !codec_neu_copy_bias_1d(ctx, "neucodec.encode.distill.codec_encoder.encoder.blocks.8.bias",
                                        graph("neucodec.encode.distill.codec_encoder.encoder.blocks.8.bias"), err)) {
                return false;
            }
        } else {
            if (!codec_neu_copy_conv1d_weight_to_3d(ctx, base + ".weight", graph(base + ".weight"), err) ||
                !codec_neu_copy_bias_1d(ctx, base + ".bias", graph(base + ".bias"), err)) {
                return false;
            }
        }
    }

    // en_encoder local trans weights
    const std::string down_trans = "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.trans";
    for (int32_t li = 0; li < 2; ++li) {
        const std::string base = down_trans + ".layers." + std::to_string(li);
        if (!codec_neu_copy_bias_1d(ctx, base + ".0.norm.weight", graph(base + ".0.norm.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".0.norm.bias", graph(base + ".0.norm.bias"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".0.to_qkv.weight", graph(base + ".0.to_qkv.weight"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".0.to_out.weight", graph(base + ".0.to_out.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".1.0.weight", graph(base + ".1.0.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".1.0.bias", graph(base + ".1.0.bias"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".1.1.weight", graph(base + ".1.1.weight"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".1.4.weight", graph(base + ".1.4.weight"), err)) {
            return false;
        }
    }
    if (!codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.weight",
                                            graph("neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.weight"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.bias",
                                graph("neucodec.encode.distill.codec_encoder.en_encoder.down_trans.down_layer.bias"), err)) {
        return false;
    }

    const std::string local_trans = "neucodec.encode.distill.codec_encoder.en_encoder.local_trans";
    for (int32_t li = 0; li < 3; ++li) {
        const std::string base = local_trans + ".layers." + std::to_string(li);
        if (!codec_neu_copy_bias_1d(ctx, base + ".0.norm.weight", graph(base + ".0.norm.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".0.norm.bias", graph(base + ".0.norm.bias"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".0.to_qkv.weight", graph(base + ".0.to_qkv.weight"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".0.to_out.weight", graph(base + ".0.to_out.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".1.0.weight", graph(base + ".1.0.weight"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".1.0.bias", graph(base + ".1.0.bias"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".1.1.weight", graph(base + ".1.1.weight"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".1.4.weight", graph(base + ".1.4.weight"), err)) {
            return false;
        }
    }

    // fc layers
    if (!codec_neu_copy_linear_weight_to_2d(ctx, "neucodec.encode.fc_sq_prior.w", graph("neucodec.encode.fc_sq_prior.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.fc_sq_prior.b", graph("neucodec.encode.fc_sq_prior.b"), err) ||
        !codec_neu_copy_linear_weight_to_2d(ctx, "neucodec.encode.fc_prior.w", graph("neucodec.encode.fc_prior.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.fc_prior.b", graph("neucodec.encode.fc_prior.b"), err) ||
        !codec_neu_copy_linear_weight_to_2d(ctx, "neucodec.encode.quant.project_in.w", graph("neucodec.encode.quant.project_in.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.quant.project_in.b", graph("neucodec.encode.quant.project_in.b"), err)) {
        return false;
    }

    // semantic encoder weights
    if (!codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.semantic_encoder.initial_conv.w",
                                            graph("neucodec.encode.semantic_encoder.initial_conv.w"), err) ||
        !codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.semantic_encoder.residual.1.w",
                                            graph("neucodec.encode.semantic_encoder.residual.1.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.semantic_encoder.residual.1.b",
                                graph("neucodec.encode.semantic_encoder.residual.1.b"), err) ||
        !codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.semantic_encoder.residual.3.w",
                                            graph("neucodec.encode.semantic_encoder.residual.3.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.semantic_encoder.residual.3.b",
                                graph("neucodec.encode.semantic_encoder.residual.3.b"), err) ||
        !codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.semantic_encoder.final_conv.w",
                                            graph("neucodec.encode.semantic_encoder.final_conv.w"), err)) {
        return false;
    }

    // HuBERT weights
    for (int32_t li = 0; li < build.hubert_feat_layers; ++li) {
        const std::string name = "neucodec.encode.hubert.feat.conv." + std::to_string(li) + ".w";
        if (!codec_neu_copy_conv1d_weight_to_3d(ctx, name, graph(name), err)) {
            return false;
        }
    }
    if (!codec_neu_copy_bias_1d(ctx, "neucodec.encode.hubert.feat.conv.0.gn.w", graph("neucodec.encode.hubert.feat.conv.0.gn.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.hubert.feat.conv.0.gn.b", graph("neucodec.encode.hubert.feat.conv.0.gn.b"), err) ||
        !codec_neu_copy_linear_weight_to_2d(ctx, "neucodec.encode.hubert.feature_projection.w", graph("neucodec.encode.hubert.feature_projection.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.hubert.feature_projection.b", graph("neucodec.encode.hubert.feature_projection.b"), err) ||
        !codec_neu_copy_conv1d_weight_to_3d(ctx, "neucodec.encode.hubert.encoder.pos_conv.w", graph("neucodec.encode.hubert.encoder.pos_conv.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.hubert.encoder.pos_conv.b", graph("neucodec.encode.hubert.encoder.pos_conv.b"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.hubert.encoder.layer_norm.w", graph("neucodec.encode.hubert.encoder.layer_norm.w"), err) ||
        !codec_neu_copy_bias_1d(ctx, "neucodec.encode.hubert.encoder.layer_norm.b", graph("neucodec.encode.hubert.encoder.layer_norm.b"), err)) {
        return false;
    }

    for (int32_t li = 0; li < build.hubert_layers; ++li) {
        const std::string base = "neucodec.encode.hubert.encoder.layers." + std::to_string(li);
        if (!codec_neu_copy_linear_weight_to_2d(ctx, base + ".att.q.w", graph(base + ".att.q.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".att.q.b", graph(base + ".att.q.b"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".att.k.w", graph(base + ".att.k.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".att.k.b", graph(base + ".att.k.b"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".att.v.w", graph(base + ".att.v.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".att.v.b", graph(base + ".att.v.b"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".att.o.w", graph(base + ".att.o.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".att.o.b", graph(base + ".att.o.b"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".ln.w", graph(base + ".ln.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".ln.b", graph(base + ".ln.b"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".ffn.fc1.w", graph(base + ".ffn.fc1.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".ffn.fc1.b", graph(base + ".ffn.fc1.b"), err) ||
            !codec_neu_copy_linear_weight_to_2d(ctx, base + ".ffn.fc2.w", graph(base + ".ffn.fc2.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".ffn.fc2.b", graph(base + ".ffn.fc2.b"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".ffn_ln.w", graph(base + ".ffn_ln.w"), err) ||
            !codec_neu_copy_bias_1d(ctx, base + ".ffn_ln.b", graph(base + ".ffn_ln.b"), err)) {
            return false;
        }
    }

    return true;
}

static enum codec_status codec_neu_decode_graph(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid NeuCodec token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    const int32_t hop = std::max(1, neu.hop_size);
    const size_t mem = 64 * 1024 * 1024 + (size_t) t * (size_t) q * sizeof(float) * 64;
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    neucodec_decode_build build = {};
    if (!codec_neu_init_decode_build(ctx, &neu, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_graph_cache_entry * entry = nullptr;
    err.clear();
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_NEUCODEC_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/0 },
            mem,
            codec_neu_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, codec_neu_name_tok().c_str());
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "neucodec.decode.head.out");
    ggml_tensor * t_codebook = codec_graph_get_tensor(ctx, entry, codec_neu_name_codebook().c_str());
    if (t_tok == nullptr || t_out == nullptr || t_codebook == nullptr) {
        codec_context_set_error(ctx, "cached NeuCodec decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<int32_t> tok_i32((size_t) t * (size_t) q, 0);
    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < q; ++qi) {
            int32_t tok = tokens->data[(size_t) ti * (size_t) tokens->n_q + (size_t) qi];
            tok = std::max(0, std::min(build.codebook_size - 1, tok));
            tok_i32[(size_t) qi * (size_t) t + (size_t) ti] = tok;
        }
    }

    ggml_tensor * cb_src = codec_neu_get_tensor(ctx->model, codec_neu_name_codebook());
    if (cb_src == nullptr) {
        codec_context_set_error(ctx, "missing NeuCodec codebook tensor");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<float> cb;
    if (!codec_tensor_as_vec_f32(cb_src, &cb)) {
        codec_context_set_error(ctx, "failed reading NeuCodec codebook tensor");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t ncb0 = (int32_t) codec_ne(cb_src, 0);
    const int32_t ncb1 = (int32_t) codec_ne(cb_src, 1);
    if (ncb0 != build.codebook_dim || ncb1 != build.codebook_size) {
        codec_context_set_error(ctx, "unexpected NeuCodec codebook shape");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_codebook, cb.data(), cb.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_neu_write_decode_weights(ctx, entry, build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_tok, tok_i32.data(), tok_i32.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> head((size_t) build.head_out_dim * (size_t) t, 0.0f);
    if (!codec_runtime_read_tensor(t_out, head.data(), head.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> window;
    ggml_tensor * w_tensor = codec_neu_get_tensor(ctx->model, codec_neu_name_istft_window());
    if (w_tensor != nullptr) {
        codec_tensor_as_vec_f32(w_tensor, &window);
    }

    std::vector<float> pcm_v;
    if (!codec_neu_istft_from_head(head, build.head_out_dim, t, hop, w_tensor != nullptr ? &window : nullptr, &pcm_v, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    float * pcm = static_cast<float *>(std::malloc(pcm_v.size() * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(pcm, pcm_v.data(), pcm_v.size() * sizeof(float));

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = pcm;
    out_pcm->n_samples = (int32_t) pcm_v.size();
    out_pcm->sample_rate = neu.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_neucodec_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_neucodec & neu = *static_cast<codec_neucodec *>(model->impl);
    neu.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", neu.sample_rate);
    neu.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", neu.encode_sample_rate);
    neu.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", neu.hop_size);
    neu.n_fft = codec_read_i32_kv(model->gguf, "codec.n_fft", neu.n_fft);
    neu.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", neu.n_q);
    neu.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", neu.codebook_size);
    neu.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", neu.codebook_dim);
    neu.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", neu.latent_dim);
    neu.hidden_dim = codec_read_i32_kv(model->gguf, "neucodec.hidden_dim", neu.hidden_dim);
    neu.vq_dim = codec_read_i32_kv(model->gguf, "neucodec.vq_dim", neu.vq_dim);
    neu.num_layers = codec_read_i32_kv(model->gguf, "neucodec.num_layers", neu.num_layers);
    neu.num_heads = codec_read_i32_kv(model->gguf, "neucodec.num_heads", neu.num_heads);
    neu.head_dim = codec_read_i32_kv(model->gguf, "neucodec.head_dim", neu.head_dim);
    neu.rope_theta = codec_read_f32_kv(model->gguf, "neucodec.rope_theta", neu.rope_theta);
    neu.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", neu.has_encoder);
    neu.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", neu.has_decoder);
    neu.encoder_type = (model->arch == CODEC_ARCH_DISTILL_NEUCODEC) ? 1 : 0;
    const int key = gguf_find_key(model->gguf, "neucodec.encoder_type");
    if (key >= 0 && gguf_get_kv_type(model->gguf, key) == GGUF_TYPE_STRING) {
        const char * val = gguf_get_val_str(model->gguf, key);
        if (val != nullptr && std::strcmp(val, "distill") == 0) {
            neu.encoder_type = 1;
        } else {
            neu.encoder_type = 0;
        }
    }
    if (model->arch == CODEC_ARCH_DISTILL_NEUCODEC && neu.encoder_type != 1) {
        return CODEC_STATUS_INVALID_STATE;
    }

    if (neu.encoder_type == 1) {
        neu.hubert_hidden = codec_read_i32_kv(model->gguf, "neucodec.hubert.hidden_size", neu.hubert_hidden);
        neu.hubert_heads = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_heads", neu.hubert_heads);
        neu.hubert_intermediate = codec_read_i32_kv(model->gguf, "neucodec.hubert.intermediate_size", neu.hubert_intermediate);
        neu.hubert_layers = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_layers", neu.hubert_layers);
        neu.hubert_pos_k = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_conv_pos_embeddings", neu.hubert_pos_k);
        neu.hubert_pos_groups = codec_read_i32_kv(model->gguf, "neucodec.hubert.num_conv_pos_embedding_groups", neu.hubert_pos_groups);
        neu.hubert_ln_eps = codec_read_f32_kv(model->gguf, "neucodec.hubert.layer_norm_eps", neu.hubert_ln_eps);
        codec_neu_read_i32_array(model->gguf, "neucodec.hubert.conv_dim", neu.hubert_conv_dim, neu.hubert_feat_layers);
        codec_neu_read_i32_array(model->gguf, "neucodec.hubert.conv_kernel", neu.hubert_conv_kernel, neu.hubert_feat_layers);
        codec_neu_read_i32_array(model->gguf, "neucodec.hubert.conv_stride", neu.hubert_conv_stride, neu.hubert_feat_layers);
    }

    model->sample_rate = neu.sample_rate;
    model->encode_sample_rate = neu.encode_sample_rate;
    model->has_encoder = neu.has_encoder;
    model->has_decoder = neu.has_decoder;
    model->hop_size = neu.hop_size;
    model->n_q = neu.n_q;
    model->codebook_size = neu.codebook_size;
    model->latent_dim = neu.latent_dim;
    model->n_fft = neu.n_fft;
    model->win_length = neu.n_fft;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_neucodec_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (!neu.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, neu.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "NeuCodec decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    return codec_neu_decode_graph(ctx, tokens, use_n_q, out_pcm);
}

static enum codec_status codec_neu_encode_graph(
    codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (pcm.empty()) {
        codec_context_set_error(ctx, "invalid NeuCodec PCM input");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (neu.encoder_type != 1) {
        codec_context_set_error(ctx, "NeuCodec encoder_type not supported (only distill implemented)");
        return CODEC_STATUS_NOT_SUPPORTED;
    }

    const int32_t n_in = (int32_t) pcm.size();
    const int32_t pad_for_wav = 320 - (n_in % 320);
    const int32_t n_in_pad = n_in + pad_for_wav;
    const int32_t n_in_sem = n_in_pad + 320;

    std::vector<float> pcm_pad((size_t) n_in_pad, 0.0f);
    std::memcpy(pcm_pad.data(), pcm.data(), pcm.size() * sizeof(float));
    std::vector<float> sem_pad((size_t) n_in_sem, 0.0f);
    std::memcpy(sem_pad.data() + 160, pcm_pad.data(), pcm_pad.size() * sizeof(float));

    const size_t mem = 512 * 1024 * 1024;
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    neucodec_encode_build build = {};
    if (!codec_neu_init_encode_build(&neu, n_in_pad, n_in_sem, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_NEUCODEC_ENCODE, /*n_frames=*/0, /*n_q=*/build.n_q, /*hop=*/0, /*n_in=*/n_in_pad, /*latent_dim=*/0 },
            mem,
            codec_neu_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.pcm").c_str());
    ggml_tensor * t_sem = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.sem").c_str());
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.out").c_str());
    ggml_tensor * t_basis = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.fsq.basis").c_str());
    ggml_tensor * t_zq = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name("neucodec.encode.debug.zq").c_str());
    if (t_pcm == nullptr || t_sem == nullptr || t_out == nullptr || t_basis == nullptr || t_zq == nullptr) {
        codec_context_set_error(ctx, "cached NeuCodec encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_write_tensor(t_pcm, pcm_pad.data(), pcm_pad.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_sem, sem_pad.data(), sem_pad.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const float basis_vals[8] = { 1, 4, 16, 64, 256, 1024, 4096, 16384 };
    if (!codec_runtime_write_tensor(t_basis, basis_vals, sizeof(basis_vals), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_neu_write_encode_weights(ctx, entry, build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_compute(ctx, entry, ctx->model->n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (std::getenv("CODEC_DEBUG_NEUCODEC_ENC") != nullptr) {
        auto dump_stats = [&](const char * name) {
            ggml_tensor * t = codec_graph_get_tensor(ctx, entry, codec_neu_encode_name(name).c_str());
            if (t == nullptr || t->type != GGML_TYPE_F32) {
                std::fprintf(stderr, "neucodec debug: missing/non-f32 %s\n", name);
                return;
            }
            int64_t n = std::max<int64_t>(1, t->ne[0]);
            n *= std::max<int64_t>(1, t->ne[1]);
            n *= std::max<int64_t>(1, t->ne[2]);
            n *= std::max<int64_t>(1, t->ne[3]);
            std::vector<float> v((size_t) n, 0.0f);
            std::string derr;
            if (!codec_runtime_read_tensor(t, v.data(), v.size() * sizeof(float), &derr)) {
                std::fprintf(stderr, "neucodec debug: read failed %s: %s\n", name, derr.c_str());
                return;
            }
            float mn = v.empty() ? 0.0f : v[0];
            float mx = v.empty() ? 0.0f : v[0];
            int64_t nan_n = 0;
            for (float x : v) {
                if (!std::isfinite(x)) {
                    ++nan_n;
                    continue;
                }
                mn = std::min(mn, x);
                mx = std::max(mx, x);
            }
            std::fprintf(stderr, "neucodec debug: %s n=%lld min=%f max=%f nan=%lld\n",
                name, (long long) n, mn, mx, (long long) nan_n);
            std::fprintf(stderr, "neucodec debug: %s first=", name);
            for (int i = 0; i < std::min<int64_t>(8, n); ++i) {
                std::fprintf(stderr, "%s%f", i == 0 ? "" : ",", v[(size_t) i]);
            }
            std::fprintf(stderr, "\n");

            const char * dump_dir = std::getenv("CODEC_DEBUG_NEUCODEC_DUMP_DIR");
            if (dump_dir != nullptr && dump_dir[0] != '\0') {
                std::string fname(name);
                for (char & c : fname) {
                    if (!(std::isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-')) {
                        c = '_';
                    }
                }
                const std::string path = std::string(dump_dir) + "/" + fname + ".f32bin";
                FILE * fp = std::fopen(path.c_str(), "wb");
                if (fp != nullptr) {
                    std::fwrite(v.data(), sizeof(float), v.size(), fp);
                    std::fclose(fp);
                }
            }
        };
        dump_stats("neucodec.encode.debug.prior_tc");
        dump_stats("neucodec.encode.debug.fsq_tc");
        dump_stats("neucodec.encode.debug.sem_tc");
        dump_stats("neucodec.encode.debug.hubert_hs");
        dump_stats("neucodec.encode.debug.acoustic_tc");
        dump_stats("neucodec.encode.debug.down_tc");
        dump_stats("neucodec.encode.debug.local_tc");
        dump_stats("neucodec.encode.debug.sem_init_tc");
        dump_stats("neucodec.encode.debug.sem_relu1_tc");
        dump_stats("neucodec.encode.debug.sem_conv1_tc");
        dump_stats("neucodec.encode.debug.sem_relu2_tc");
        dump_stats("neucodec.encode.debug.sem_conv2_tc");
        dump_stats("neucodec.encode.debug.sem_res_tc");
        dump_stats("neucodec.encode.debug.z_tc");
        dump_stats("neucodec.encode.debug.z_bound");
        dump_stats("neucodec.encode.debug.zq");
    }

    if (t_out->type != GGML_TYPE_F32 || t_out->ne[1] != 1) {
        codec_context_set_error(ctx, "unexpected NeuCodec token tensor shape/type");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_frames = (int32_t) t_out->ne[0];
    std::vector<float> out_f((size_t) n_frames, 0.0f);
    if (!codec_runtime_read_tensor(t_out, out_f.data(), out_f.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<int32_t> tok((size_t) n_frames, 0);
    for (int32_t ti = 0; ti < n_frames; ++ti) {
        float v = out_f[(size_t) ti];
        if (!std::isfinite(v)) v = 0.0f;
        int32_t idx = (int32_t) std::lrintf(v);
        tok[(size_t) ti] = std::max(0, std::min(neu.codebook_size - 1, idx));
    }

    int32_t * data = static_cast<int32_t *>(std::malloc(tok.size() * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, tok.data(), tok.size() * sizeof(int32_t));
    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = n_frames;
    out_tokens->n_frames = n_frames;
    out_tokens->n_q = 1;
    out_tokens->codebook_size = neu.codebook_size;
    out_tokens->sample_rate = neu.encode_sample_rate;
    out_tokens->hop_size = neu.hop_size;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_neucodec_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    codec_neucodec & neu = *static_cast<codec_neucodec *>(ctx->model->impl);
    if (!neu.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    (void) out_latent;
    if (params.n_q != 0 && params.n_q != 1) {
        codec_context_set_error(ctx, "NeuCodec encode n_q must be 0 or 1");
        return CODEC_STATUS_INVALID_ARG;
    }

    if (neu.encoder_type == 1 && !neu.distill_bias_ready) {
        std::string err;
        const int32_t down_max_dist = 3000;
        const int32_t local_max_dist = 600;
        if (!codec_neu_build_dynamic_pos_bias(ctx->model, "neucodec.encode.distill.codec_encoder.en_encoder.down_trans.trans.dynamic_pos_bias",
                                              down_max_dist, &neu.distill_bias_down, &err) ||
            !codec_neu_build_dynamic_pos_bias(ctx->model, "neucodec.encode.distill.codec_encoder.en_encoder.local_trans.dynamic_pos_bias",
                                              local_max_dist, &neu.distill_bias_local, &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        neu.distill_attn_down.bias = neu.distill_bias_down.data();
        neu.distill_attn_down.heads = 6;
        neu.distill_attn_down.head_dim = 128;
        // local_attention uses look_backward=1, so effective causal context is 2 * window_size.
        neu.distill_attn_down.window = 3000;
        neu.distill_attn_down.max_dist = down_max_dist;

        neu.distill_attn_local.bias = neu.distill_bias_local.data();
        neu.distill_attn_local.heads = 6;
        neu.distill_attn_local.head_dim = 128;
        neu.distill_attn_local.window = 600;
        neu.distill_attn_local.max_dist = local_max_dist;

        neu.distill_bias_ready = true;
    }

    return codec_neu_encode_graph(ctx, pcm, out_tokens);
}

static void * codec_neu_create_impl() {
    return new (std::nothrow) codec_neucodec();
}

static void codec_neu_destroy_impl(void * ptr) {
    delete static_cast<codec_neucodec *>(ptr);
}

static enum codec_status codec_neu_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    return codec_neucodec_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_neu_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    return codec_neucodec_encode(ctx, pcm, out_tokens, out_latent, params);
}

const struct codec_model_vtable * codec_neucodec_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_NEUCODEC,
        "neucodec",
        codec_neu_create_impl,
        codec_neu_destroy_impl,
        codec_neucodec_init,
        codec_neu_encode_wrap,
        codec_neu_decode_wrap,
    };
    return &vtable;
}

const struct codec_model_vtable * codec_distill_neucodec_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_DISTILL_NEUCODEC,
        "distill_neucodec",
        codec_neu_create_impl,
        codec_neu_destroy_impl,
        codec_neucodec_init,
        codec_neu_encode_wrap,
        codec_neu_decode_wrap,
    };
    return &vtable;
}
