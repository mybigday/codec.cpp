#include "wavtokenizer.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

enum codec_status codec_wavtokenizer_init(struct codec_model * model) {
    codec_wavtokenizer_large & wt = model->wavtokenizer_large;

    wt.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    wt.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 320);
    wt.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", true);
    wt.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);
    wt.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", codec_infer_n_q_from_tensor_names(model));

    wt.vq_embed = ggml_get_tensor(model->weights, "vq.vq.layers.0._codebook.embed");
    if (wt.vq_embed != nullptr) {
        wt.codebook_dim = (int32_t) wt.vq_embed->ne[0];
        wt.codebook_size = (int32_t) wt.vq_embed->ne[1];
    }
    if (wt.codebook_size <= 0) {
        wt.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 1024);
    }

    model->sample_rate = wt.sample_rate;
    model->has_encoder = wt.has_encoder;
    model->has_decoder = wt.has_decoder;
    model->hop_size = wt.hop_size;
    model->n_q = wt.n_q;
    model->codebook_size = wt.codebook_size;
    model->latent_dim = wt.codebook_dim > 0 ? wt.codebook_dim : 1;

    static const char * const keys_n_fft[] = { "codec.n_fft", "codec.stft.n_fft" };
    static const char * const keys_win_length[] = { "codec.win_length", "codec.stft.win_length" };
    static const char * const keys_n_mels[] = { "codec.n_mels", "codec.mel.n_mels" };

    model->n_fft = codec_read_i32_kv_any(model->gguf, keys_n_fft, 2, -1);
    model->win_length = codec_read_i32_kv_any(model->gguf, keys_win_length, 2, -1);
    model->n_mels = codec_read_i32_kv_any(model->gguf, keys_n_mels, 2, -1);

    return CODEC_STATUS_SUCCESS;
}

struct wt_decode_build {
    int32_t t;
    int32_t q;
    int32_t hop;
    int32_t codebook_dim;
    int32_t codebook_size;
    int32_t backbone_dim;
    int32_t backbone_intermediate;
    int32_t n_convnext;
    int32_t head_out_dim;
    int32_t use_adanorm;
};

static std::string codec_wt_decode_codebook_tensor_name(int32_t qi) {
    return "wt.decode.vq.q" + std::to_string(qi) + ".codebook";
}

static ggml_tensor * codec_wt_sum_codebook_features(
    ggml_context * ctx_eval,
    ggml_tensor * t_tok,
    int32_t t,
    int32_t q,
    int32_t codebook_dim,
    int32_t codebook_size) {

    ggml_tensor * sum = nullptr;
    for (int32_t qi = 0; qi < q; ++qi) {
        ggml_tensor * t_codebook = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, codebook_dim, codebook_size);
        ggml_set_name(t_codebook, codec_wt_decode_codebook_tensor_name(qi).c_str());
        ggml_tensor * t_idx = ggml_view_1d(ctx_eval, t_tok, t, (size_t) qi * t_tok->nb[1]);
        ggml_tensor * t_q = ggml_get_rows(ctx_eval, t_codebook, t_idx); // [codebook_dim, t]
        sum = (sum == nullptr) ? t_q : ggml_add(ctx_eval, sum, t_q);
    }
    return sum;
}

static ggml_tensor * codec_wt_layer_norm_ct(
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

static std::string codec_wt_decode_embed_w_name() { return "wt.decode.bb.embed.w"; }
static std::string codec_wt_decode_embed_b_name() { return "wt.decode.bb.embed.b"; }
static std::string codec_wt_decode_norm_w_name() { return "wt.decode.bb.norm.w"; }
static std::string codec_wt_decode_norm_b_name() { return "wt.decode.bb.norm.b"; }
static std::string codec_wt_decode_final_ln_w_name() { return "wt.decode.bb.final_ln.w"; }
static std::string codec_wt_decode_final_ln_b_name() { return "wt.decode.bb.final_ln.b"; }
static std::string codec_wt_decode_head_w_name() { return "wt.decode.head.out.w"; }
static std::string codec_wt_decode_head_b_name() { return "wt.decode.head.out.b"; }

static std::string codec_wt_decode_blk_dw_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".dw.w"; }
static std::string codec_wt_decode_blk_dw_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".dw.b"; }
static std::string codec_wt_decode_blk_ln_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".ln.w"; }
static std::string codec_wt_decode_blk_ln_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".ln.b"; }
static std::string codec_wt_decode_blk_pw1_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw1.w"; }
static std::string codec_wt_decode_blk_pw1_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw1.b"; }
static std::string codec_wt_decode_blk_pw2_w_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw2.w"; }
static std::string codec_wt_decode_blk_pw2_b_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".pw2.b"; }
static std::string codec_wt_decode_blk_gamma_name(int32_t li) { return "wt.decode.bb.l" + std::to_string(li) + ".gamma"; }

static bool codec_wt_build_decode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    wt_decode_build * p = static_cast<wt_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->t <= 0 || p->q <= 0 || p->codebook_dim <= 0 || p->codebook_size <= 1 ||
        p->backbone_dim <= 0 || p->backbone_intermediate <= 0 || p->n_convnext <= 0 || p->head_out_dim <= 0) {
        return false;
    }

    ggml_tensor * t_tok = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_I32, p->t, p->q);
    ggml_set_name(t_tok, "wt.decode.tok");

    ggml_tensor * t_feat_ct = codec_wt_sum_codebook_features(ctx_eval, t_tok, p->t, p->q, p->codebook_dim, p->codebook_size);
    if (t_feat_ct == nullptr) {
        return false;
    }
    ggml_tensor * x = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, t_feat_ct)); // [t, c]

    ggml_tensor * t_embed_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 7, p->codebook_dim, p->backbone_dim);
    ggml_set_name(t_embed_w, codec_wt_decode_embed_w_name().c_str());
    ggml_tensor * t_embed_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
    ggml_set_name(t_embed_b, codec_wt_decode_embed_b_name().c_str());
    x = codec_conv1d(ctx_eval, x, t_embed_w, t_embed_b, 1, 1, 3);
    if (x == nullptr) {
        return false;
    }

    ggml_tensor * t_inln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
    ggml_set_name(t_inln_w, codec_wt_decode_norm_w_name().c_str());
    ggml_tensor * t_inln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
    ggml_set_name(t_inln_b, codec_wt_decode_norm_b_name().c_str());

    ggml_tensor * x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x)); // [c, t]
    x_ct = codec_wt_layer_norm_ct(ctx_eval, x_ct, t_inln_w, t_inln_b);
    if (x_ct == nullptr) {
        return false;
    }

    for (int32_t li = 0; li < p->n_convnext; ++li) {
        ggml_tensor * res_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_ct));

        ggml_tensor * t_dw_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 7, p->backbone_dim, 1);
        ggml_set_name(t_dw_w, codec_wt_decode_blk_dw_w_name(li).c_str());
        ggml_tensor * t_dw_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
        ggml_set_name(t_dw_b, codec_wt_decode_blk_dw_b_name(li).c_str());
        ggml_tensor * x_dw = codec_conv1d_depthwise(ctx_eval, res_tc, t_dw_w, t_dw_b, 1, 1, 3);
        if (x_dw == nullptr) {
            return false;
        }

        ggml_tensor * t_lnw = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
        ggml_set_name(t_lnw, codec_wt_decode_blk_ln_w_name(li).c_str());
        ggml_tensor * t_lnb = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
        ggml_set_name(t_lnb, codec_wt_decode_blk_ln_b_name(li).c_str());
        ggml_tensor * x_blk_ct = codec_wt_layer_norm_ct(ctx_eval, ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_dw)), t_lnw, t_lnb);
        if (x_blk_ct == nullptr) {
            return false;
        }

        ggml_tensor * t_pw1_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->backbone_dim, p->backbone_intermediate);
        ggml_set_name(t_pw1_w, codec_wt_decode_blk_pw1_w_name(li).c_str());
        ggml_tensor * t_pw1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_intermediate);
        ggml_set_name(t_pw1_b, codec_wt_decode_blk_pw1_b_name(li).c_str());
        ggml_tensor * x_pw = codec_op_linear(ctx_eval, x_blk_ct, t_pw1_w, t_pw1_b);
        if (x_pw == nullptr) {
            return false;
        }
        x_pw = ggml_gelu_erf(ctx_eval, x_pw);

        ggml_tensor * t_pw2_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->backbone_intermediate, p->backbone_dim);
        ggml_set_name(t_pw2_w, codec_wt_decode_blk_pw2_w_name(li).c_str());
        ggml_tensor * t_pw2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
        ggml_set_name(t_pw2_b, codec_wt_decode_blk_pw2_b_name(li).c_str());
        ggml_tensor * x_pw2 = codec_op_linear(ctx_eval, x_pw, t_pw2_w, t_pw2_b);
        if (x_pw2 == nullptr) {
            return false;
        }

        ggml_tensor * t_gamma = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
        ggml_set_name(t_gamma, codec_wt_decode_blk_gamma_name(li).c_str());
        x_pw2 = codec_op_channel_scale(ctx_eval, x_pw2, t_gamma);
        if (x_pw2 == nullptr) {
            return false;
        }
        x_ct = ggml_add(ctx_eval, ggml_cont(ctx_eval, ggml_transpose(ctx_eval, res_tc)), x_pw2);
    }

    ggml_tensor * t_fln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
    ggml_set_name(t_fln_w, codec_wt_decode_final_ln_w_name().c_str());
    ggml_tensor * t_fln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->backbone_dim);
    ggml_set_name(t_fln_b, codec_wt_decode_final_ln_b_name().c_str());
    x_ct = codec_wt_layer_norm_ct(ctx_eval, x_ct, t_fln_w, t_fln_b);
    if (x_ct == nullptr) {
        return false;
    }

    ggml_tensor * t_head_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->backbone_dim, p->head_out_dim);
    ggml_set_name(t_head_w, codec_wt_decode_head_w_name().c_str());
    ggml_tensor * t_head_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->head_out_dim);
    ggml_set_name(t_head_b, codec_wt_decode_head_b_name().c_str());
    ggml_tensor * t_head = codec_op_linear(ctx_eval, x_ct, t_head_w, t_head_b); // [out_dim, t]
    if (t_head == nullptr) {
        return false;
    }

    ggml_tensor * t_out = ggml_cont(ctx_eval, t_head);
    ggml_set_name(t_out, "wt.decode.head.out");
    *out = t_out;
    return true;
}

struct wt_encode_build {
    int32_t n_in;
    int32_t hop;
};

static bool codec_wt_load_kernel_f16(
    struct codec_context * ctx,
    const char * name,
    int32_t hop,
    std::vector<ggml_fp16_t> * kernel_f16,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || kernel_f16 == nullptr || name == nullptr || hop <= 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer kernel load arguments";
        }
        return false;
    }

    ggml_tensor * t_kernel = ggml_get_tensor(ctx->model->weights, name);
    if (t_kernel == nullptr) {
        if (err != nullptr) {
            *err = std::string("missing model tensor: ") + name;
        }
        return false;
    }
    if (t_kernel->type != GGML_TYPE_F16 || ggml_nelements(t_kernel) != hop) {
        if (err != nullptr) {
            *err = std::string("invalid model kernel tensor: ") + name;
        }
        return false;
    }

    kernel_f16->assign((size_t) hop, 0);
    return codec_runtime_read_tensor(t_kernel, kernel_f16->data(), kernel_f16->size() * sizeof(ggml_fp16_t), err);
}

static bool codec_wt_build_encode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    wt_encode_build * p = static_cast<wt_encode_build *>(user_data);
    ggml_tensor * t_pcm = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->n_in, 1);
    ggml_set_name(t_pcm, "wt.encode.pcm");

    ggml_tensor * t_kernel = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F16, p->hop, 1, 1);
    ggml_set_name(t_kernel, "wt.encode.kernel");

    ggml_tensor * t_feat = codec_conv1d(ctx_eval, t_pcm, t_kernel, nullptr, p->hop, 1, 0);
    ggml_tensor * t_alpha = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 1);
    ggml_set_name(t_alpha, "wt.encode.alpha");

    ggml_tensor * t_snake = ggml_cont(ctx_eval, codec_op_snake(ctx_eval, t_feat, t_alpha, 1e-4f));
    ggml_set_name(t_snake, "wt.encode.out");

    *out = t_snake;
    return true;
}

static ggml_tensor * codec_wt_get_tensor(codec_model * model, const std::string & name) {
    if (model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    return ggml_get_tensor(model->weights, name.c_str());
}

static bool codec_wt_copy_conv1d_weight_to_3d(
    codec_context * ctx,
    const std::string & src_name,
    ggml_tensor * dst,
    std::string * err) {

    ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    const int32_t dk = (int32_t) codec_ne(dst, 0);
    const int32_t din = (int32_t) codec_ne(dst, 1);
    const int32_t dout = (int32_t) codec_ne(dst, 2);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    const int32_t n2 = (int32_t) codec_ne(src, 2);
    std::vector<float> dst_v((size_t) dk * (size_t) din * (size_t) dout, 0.0f);

    if (n0 == dk && n1 == din && n2 == dout) {
        dst_v = src_v;
    } else if (n0 == dout && n1 == din && n2 == dk) {
        for (int32_t k = 0; k < dk; ++k) {
            for (int32_t i = 0; i < din; ++i) {
                for (int32_t o = 0; o < dout; ++o) {
                    const size_t src_idx = (size_t) o + (size_t) dout * ((size_t) i + (size_t) din * (size_t) k);
                    const size_t dst_idx = (size_t) k + (size_t) dk * ((size_t) i + (size_t) din * (size_t) o);
                    dst_v[dst_idx] = src_v[src_idx];
                }
            }
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer conv1d shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, dst_v.data(), dst_v.size() * sizeof(float), err);
}

static bool codec_wt_copy_linear_weight_to_2d(
    codec_context * ctx,
    const std::string & src_name,
    ggml_tensor * dst,
    std::string * err) {

    ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    const int32_t din = (int32_t) codec_ne(dst, 0);
    const int32_t dout = (int32_t) codec_ne(dst, 1);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    std::vector<float> dst_v((size_t) din * (size_t) dout, 0.0f);
    if (n0 == din && n1 == dout) {
        dst_v = src_v;
    } else if (n0 == dout && n1 == din) {
        for (int32_t i = 0; i < din; ++i) {
            for (int32_t o = 0; o < dout; ++o) {
                dst_v[(size_t) i + (size_t) din * (size_t) o] = src_v[(size_t) o + (size_t) dout * (size_t) i];
            }
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer linear shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, dst_v.data(), dst_v.size() * sizeof(float), err);
}

static bool codec_wt_copy_bias_1d(codec_context * ctx, const std::string & src_name, ggml_tensor * dst, std::string * err) {
    ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> v;
    if (!codec_tensor_as_vec_f32(src, &v) || (int32_t) v.size() != (int32_t) codec_ne(dst, 0)) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer bias tensor: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, v.data(), v.size() * sizeof(float), err);
}

static bool codec_wt_copy_embedding_row0(codec_context * ctx, const std::string & src_name, ggml_tensor * dst, std::string * err) {
    ggml_tensor * src = codec_wt_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading WavTokenizer tensor: " + src_name;
        }
        return false;
    }
    const int32_t d = (int32_t) codec_ne(dst, 0);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    std::vector<float> out((size_t) d, 0.0f);
    if (n0 == d && n1 >= 1) {
        for (int32_t i = 0; i < d; ++i) {
            out[(size_t) i] = src_v[(size_t) i];
        }
    } else if (n1 == d && n0 >= 1) {
        for (int32_t i = 0; i < d; ++i) {
            out[(size_t) i] = src_v[(size_t) 0 + (size_t) n0 * (size_t) i];
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer embedding shape: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, out.data(), out.size() * sizeof(float), err);
}

static bool codec_wt_init_decode_build(codec_context * ctx, int32_t t, int32_t q, wt_decode_build * build, std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || t <= 0 || q <= 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer decode build arguments";
        }
        return false;
    }
    const codec_wavtokenizer_large & wt = ctx->model->wavtokenizer_large;
    build->t = t;
    build->q = q;
    build->hop = std::max(1, wt.hop_size);
    build->codebook_dim = std::max(1, wt.codebook_dim);
    build->codebook_size = std::max(2, wt.codebook_size);

    ggml_tensor * embed_w = codec_wt_get_tensor(ctx->model, "dec.backbone.embed.weight");
    ggml_tensor * embed_b = codec_wt_get_tensor(ctx->model, "dec.backbone.embed.bias");
    if (embed_w == nullptr || embed_b == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer backbone embed tensors";
        }
        return false;
    }
    const int32_t eb0 = (int32_t) codec_ne(embed_w, 0);
    const int32_t eb1 = (int32_t) codec_ne(embed_w, 1);
    const int32_t eb2 = (int32_t) codec_ne(embed_w, 2);
    const int32_t bo = (int32_t) codec_ne(embed_b, 0);
    if (eb2 == bo) {
        build->backbone_dim = eb2;
        build->codebook_dim = eb1;
    } else if (eb0 == bo) {
        build->backbone_dim = eb0;
        build->codebook_dim = eb1;
    } else {
        if (err != nullptr) {
            *err = "unexpected WavTokenizer embed shape";
        }
        return false;
    }

    build->n_convnext = 0;
    for (int32_t li = 0; li < 64; ++li) {
        if (codec_wt_get_tensor(ctx->model, "dec.backbone.convnext." + std::to_string(li) + ".dwconv.weight") == nullptr) {
            break;
        }
        build->n_convnext = li + 1;
    }
    if (build->n_convnext <= 0) {
        if (err != nullptr) {
            *err = "no WavTokenizer convnext layers found";
        }
        return false;
    }

    ggml_tensor * pw1 = codec_wt_get_tensor(ctx->model, "dec.backbone.convnext.0.pwconv1.weight");
    if (pw1 == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer pwconv1 tensor";
        }
        return false;
    }
    build->backbone_intermediate = std::max((int32_t) codec_ne(pw1, 0), (int32_t) codec_ne(pw1, 1));

    ggml_tensor * head_b = codec_wt_get_tensor(ctx->model, "dec.head.out.bias");
    if (head_b == nullptr) {
        if (err != nullptr) {
            *err = "missing WavTokenizer head bias tensor";
        }
        return false;
    }
    build->head_out_dim = (int32_t) codec_ne(head_b, 0);
    build->use_adanorm = codec_wt_get_tensor(ctx->model, "dec.backbone.norm.scale.weight") != nullptr ? 1 : 0;
    return true;
}

static bool codec_wt_write_decode_weights(codec_context * ctx, codec_graph_cache_entry * entry, const wt_decode_build & build, std::string * err) {
    auto graph = [&](const std::string & n) { return codec_graph_get_tensor(ctx, entry, n.c_str()); };

    if (!codec_wt_copy_conv1d_weight_to_3d(ctx, "dec.backbone.embed.weight", graph(codec_wt_decode_embed_w_name()), err) ||
        !codec_wt_copy_bias_1d(ctx, "dec.backbone.embed.bias", graph(codec_wt_decode_embed_b_name()), err)) {
        return false;
    }

    if (build.use_adanorm) {
        if (!codec_wt_copy_embedding_row0(ctx, "dec.backbone.norm.scale.weight", graph(codec_wt_decode_norm_w_name()), err) ||
            !codec_wt_copy_embedding_row0(ctx, "dec.backbone.norm.shift.weight", graph(codec_wt_decode_norm_b_name()), err)) {
            return false;
        }
    } else {
        if (!codec_wt_copy_bias_1d(ctx, "dec.backbone.norm.weight", graph(codec_wt_decode_norm_w_name()), err) ||
            !codec_wt_copy_bias_1d(ctx, "dec.backbone.norm.bias", graph(codec_wt_decode_norm_b_name()), err)) {
            return false;
        }
    }

    for (int32_t li = 0; li < build.n_convnext; ++li) {
        const std::string p = "dec.backbone.convnext." + std::to_string(li) + ".";
        if (!codec_wt_copy_conv1d_weight_to_3d(ctx, p + "dwconv.weight", graph(codec_wt_decode_blk_dw_w_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "dwconv.bias", graph(codec_wt_decode_blk_dw_b_name(li)), err) ||
            !codec_wt_copy_linear_weight_to_2d(ctx, p + "pwconv1.weight", graph(codec_wt_decode_blk_pw1_w_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "pwconv1.bias", graph(codec_wt_decode_blk_pw1_b_name(li)), err) ||
            !codec_wt_copy_linear_weight_to_2d(ctx, p + "pwconv2.weight", graph(codec_wt_decode_blk_pw2_w_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "pwconv2.bias", graph(codec_wt_decode_blk_pw2_b_name(li)), err) ||
            !codec_wt_copy_bias_1d(ctx, p + "gamma", graph(codec_wt_decode_blk_gamma_name(li)), err)) {
            return false;
        }
        if (build.use_adanorm) {
            if (!codec_wt_copy_embedding_row0(ctx, p + "norm.scale.weight", graph(codec_wt_decode_blk_ln_w_name(li)), err) ||
                !codec_wt_copy_embedding_row0(ctx, p + "norm.shift.weight", graph(codec_wt_decode_blk_ln_b_name(li)), err)) {
                return false;
            }
        } else {
            if (!codec_wt_copy_bias_1d(ctx, p + "norm.weight", graph(codec_wt_decode_blk_ln_w_name(li)), err) ||
                !codec_wt_copy_bias_1d(ctx, p + "norm.bias", graph(codec_wt_decode_blk_ln_b_name(li)), err)) {
                return false;
            }
        }
    }

    if (!codec_wt_copy_bias_1d(ctx, "dec.backbone.final_layer_norm.weight", graph(codec_wt_decode_final_ln_w_name()), err) ||
        !codec_wt_copy_bias_1d(ctx, "dec.backbone.final_layer_norm.bias", graph(codec_wt_decode_final_ln_b_name()), err) ||
        !codec_wt_copy_linear_weight_to_2d(ctx, "dec.head.out.weight", graph(codec_wt_decode_head_w_name()), err) ||
        !codec_wt_copy_bias_1d(ctx, "dec.head.out.bias", graph(codec_wt_decode_head_b_name()), err)) {
        return false;
    }
    return true;
}

static bool codec_wt_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    std::vector<float> * out_pcm,
    std::string * err) {

    if (out_pcm == nullptr || out_dim <= 0 || n_frames <= 0 || hop <= 0 || (out_dim % 2) != 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer ISTFT arguments";
        }
        return false;
    }
    const int32_t n_bins = out_dim / 2;
    const int32_t n_fft = 2 * (n_bins - 1);
    const float pi = 3.14159265358979323846f;
    if (n_fft <= 0) {
        if (err != nullptr) {
            *err = "invalid WavTokenizer head output dimension";
        }
        return false;
    }
    const int32_t pad = (n_fft - hop) / 2;
    const int32_t out_size = (n_frames - 1) * hop + n_fft;
    std::vector<float> window((size_t) n_fft, 0.0f);
    for (int32_t n = 0; n < n_fft; ++n) {
        window[(size_t) n] = 0.5f - 0.5f * std::cos(2.0f * pi * (float) n / (float) (n_fft - 1));
    }
    std::vector<float> y((size_t) out_size, 0.0f);
    std::vector<float> env((size_t) out_size, 0.0f);
    std::vector<float> frame((size_t) n_fft, 0.0f);

    for (int32_t ti = 0; ti < n_frames; ++ti) {
        for (int32_t n = 0; n < n_fft; ++n) {
            float sum = 0.0f;
            const float re0 = std::exp(std::min(100.0f, head[(size_t) 0 + (size_t) out_dim * (size_t) ti])) * std::cos(head[(size_t) n_bins + (size_t) out_dim * (size_t) ti]);
            sum += re0;
            const float ren = std::exp(std::min(100.0f, head[(size_t) (n_bins - 1) + (size_t) out_dim * (size_t) ti])) * std::cos(head[(size_t) (2 * n_bins - 1) + (size_t) out_dim * (size_t) ti]);
            sum += ren * ((n & 1) ? -1.0f : 1.0f);
            for (int32_t k = 1; k < n_bins - 1; ++k) {
                const float mag = std::exp(std::min(100.0f, head[(size_t) k + (size_t) out_dim * (size_t) ti]));
                const float ph = head[(size_t) (n_bins + k) + (size_t) out_dim * (size_t) ti];
                const float re = mag * std::cos(ph);
                const float im = mag * std::sin(ph);
                const float ang = 2.0f * pi * (float) k * (float) n / (float) n_fft;
                sum += 2.0f * (re * std::cos(ang) - im * std::sin(ang));
            }
            frame[(size_t) n] = (sum / (float) n_fft) * window[(size_t) n];
        }
        const int32_t off = ti * hop;
        for (int32_t n = 0; n < n_fft; ++n) {
            y[(size_t) (off + n)] += frame[(size_t) n];
            env[(size_t) (off + n)] += window[(size_t) n] * window[(size_t) n];
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

static enum codec_status codec_wt_decode_graph(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm) {

    codec_wavtokenizer_large & wt = ctx->model->wavtokenizer_large;
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid WavTokenizer token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    const int32_t hop = std::max(1, wt.hop_size);
    const size_t mem = 32 * 1024 * 1024 + (size_t) t * (size_t) q * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    wt_decode_build build = {};
    if (!codec_wt_init_decode_build(ctx, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_graph_cache_entry * entry = nullptr;
    err.clear();
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_WT_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/0 },
            mem,
            codec_wt_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "wt.decode.tok");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "wt.decode.head.out");
    if (t_tok == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached WavTokenizer decode graph is invalid");
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
    for (int32_t qi = 0; qi < q; ++qi) {
        ggml_tensor * t_codebook = codec_graph_get_tensor(ctx, entry, codec_wt_decode_codebook_tensor_name(qi).c_str());
        if (t_codebook == nullptr) {
            codec_context_set_error(ctx, "cached WavTokenizer decode graph is missing codebook tensors");
            return CODEC_STATUS_INTERNAL_ERROR;
        }

        const std::string n0 = "vq.vq.layers." + std::to_string(qi) + "._codebook.embed";
        const std::string n1 = "vq.vq.layers." + std::to_string(qi) + ".codebook.embed";
        ggml_tensor * src = ggml_get_tensor(ctx->model->weights, n0.c_str());
        if (src == nullptr) {
            src = ggml_get_tensor(ctx->model->weights, n1.c_str());
        }
        if (src == nullptr) {
            codec_context_set_error(ctx, "missing WavTokenizer codebook tensor");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        std::vector<float> cb;
        if (!codec_tensor_as_vec_f32(src, &cb)) {
            codec_context_set_error(ctx, "failed reading WavTokenizer codebook tensor");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        const int32_t ncb0 = (int32_t) codec_ne(src, 0);
        const int32_t ncb1 = (int32_t) codec_ne(src, 1);
        std::vector<float> cb_dst((size_t) build.codebook_dim * (size_t) build.codebook_size, 0.0f);
        if (ncb0 == build.codebook_dim && ncb1 == build.codebook_size) {
            cb_dst = cb;
        } else if (ncb0 == build.codebook_size && ncb1 == build.codebook_dim) {
            for (int32_t i = 0; i < build.codebook_dim; ++i) {
                for (int32_t j = 0; j < build.codebook_size; ++j) {
                    cb_dst[(size_t) i + (size_t) build.codebook_dim * (size_t) j] =
                        cb[(size_t) j + (size_t) build.codebook_size * (size_t) i];
                }
            }
        } else {
            codec_context_set_error(ctx, "unexpected WavTokenizer codebook shape");
            return CODEC_STATUS_INTERNAL_ERROR;
        }

        if (!codec_runtime_write_tensor(t_codebook, cb_dst.data(), cb_dst.size() * sizeof(float), &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    if (!codec_wt_write_decode_weights(ctx, entry, build, &err)) {
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
    std::vector<float> pcm_v;
    if (!codec_wt_istft_from_head(head, build.head_out_dim, t, hop, &pcm_v, &err)) {
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
    out_pcm->sample_rate = wt.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_wavtokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_wavtokenizer_large & wt = ctx->model->wavtokenizer_large;
    if (!wt.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, wt.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "WavTokenizer decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    return codec_wt_decode_graph(ctx, tokens, use_n_q, out_pcm);
}

enum codec_status codec_wavtokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params) {

    codec_wavtokenizer_large & wt = ctx->model->wavtokenizer_large;
    if (!wt.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (pcm.empty()) {
        codec_context_set_error(ctx, "empty pcm");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t model_n_q = std::max(1, wt.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "WavTokenizer encode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t hop = std::max(1, params.hop_size > 0 ? params.hop_size : wt.hop_size);
    const int32_t n_in = (int32_t) pcm.size();

    const size_t mem = 32 * 1024 * 1024 + (size_t) n_in * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);
    wt_encode_build build = { n_in, hop };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_WT_ENCODE, /*n_frames=*/0, /*n_q=*/0, /*hop=*/hop, /*n_in=*/n_in, /*latent_dim=*/0 },
            mem,
            codec_wt_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "wt.encode.pcm");
    ggml_tensor * t_alpha = codec_graph_get_tensor(ctx, entry, "wt.encode.alpha");
    ggml_tensor * t_kernel = codec_graph_get_tensor(ctx, entry, "wt.encode.kernel");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "wt.encode.out");
    if (t_pcm == nullptr || t_alpha == nullptr || t_kernel == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached WavTokenizer encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<ggml_fp16_t> kernel_f16;
    if (!codec_wt_load_kernel_f16(ctx, "wt.encode.kernel", hop, &kernel_f16, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const float alpha = 0.5f;
    if (!codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_kernel, kernel_f16.data(), kernel_f16.size() * sizeof(ggml_fp16_t), &err) ||
        !codec_runtime_write_tensor(t_alpha, &alpha, sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_frames = (int32_t) t_out->ne[0];
    std::vector<float> feat((size_t) n_frames, 0.0f);
    if (!codec_runtime_read_tensor(t_out, feat.data(), feat.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) n_frames * (size_t) use_n_q * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t codebook_size = std::max(2, wt.codebook_size);
    for (int32_t t = 0; t < n_frames; ++t) {
        const float x = std::max(-1.0f, std::min(1.0f, feat[(size_t) t]));
        const int32_t tok = std::max(0, std::min(codebook_size - 1, (int32_t) std::llround((x * 0.5f + 0.5f) * (codebook_size - 1))));
        for (int32_t q = 0; q < use_n_q; ++q) {
            data[(size_t) t * (size_t) use_n_q + (size_t) q] = tok;
        }
    }

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = n_frames * use_n_q;
    out_tokens->n_frames = n_frames;
    out_tokens->n_q = use_n_q;
    out_tokens->codebook_size = codebook_size;
    out_tokens->sample_rate = wt.sample_rate;
    out_tokens->hop_size = hop;

    return CODEC_STATUS_SUCCESS;
}
