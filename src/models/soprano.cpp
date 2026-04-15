#include "soprano.h"

#include "../ops/conv1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

static std::string codec_sop_name_embed_w() { return "sop.decode.embed.w"; }
static std::string codec_sop_name_embed_b() { return "sop.decode.embed.b"; }
static std::string codec_sop_name_norm_w() { return "sop.decode.norm.w"; }
static std::string codec_sop_name_norm_b() { return "sop.decode.norm.b"; }
static std::string codec_sop_name_fln_w() { return "sop.decode.fln.w"; }
static std::string codec_sop_name_fln_b() { return "sop.decode.fln.b"; }
static std::string codec_sop_name_head_w() { return "sop.decode.head.out.w"; }
static std::string codec_sop_name_head_b() { return "sop.decode.head.out.b"; }
static std::string codec_sop_name_istft_window() { return "sop.decode.istft.window"; }

static std::string codec_sop_name_cnx_dw_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".dw.w"; }
static std::string codec_sop_name_cnx_dw_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".dw.b"; }
static std::string codec_sop_name_cnx_ln_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".ln.w"; }
static std::string codec_sop_name_cnx_ln_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".ln.b"; }
static std::string codec_sop_name_cnx_pw1_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw1.w"; }
static std::string codec_sop_name_cnx_pw1_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw1.b"; }
static std::string codec_sop_name_cnx_pw2_w(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw2.w"; }
static std::string codec_sop_name_cnx_pw2_b(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".pw2.b"; }
static std::string codec_sop_name_cnx_gamma(int32_t li) { return "sop.decode.cnx." + std::to_string(li) + ".gamma"; }

static ggml_tensor * codec_sop_get_tensor(codec_model * model, const std::string & name) {
    if (model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    return ggml_get_tensor(model->weights, name.c_str());
}

struct sop_decode_build {
    int32_t t = 0;
    int32_t in_ch = 0;
    int32_t dim = 0;
    int32_t intermediate = 0;
    int32_t n_layers = 0;
    int32_t dw_kernel = 0;
    int32_t head_out_dim = 0;
};

static bool codec_sop_build_decode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    sop_decode_build * p = static_cast<sop_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr) {
        return false;
    }
    if (p->t <= 0 || p->in_ch <= 0 || p->dim <= 0 || p->intermediate <= 0 || p->n_layers <= 0 || p->dw_kernel <= 0 || p->head_out_dim <= 0) {
        return false;
    }

    ggml_tensor * t_in = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->t, p->in_ch);
    ggml_set_name(t_in, "sop.decode.in");

    ggml_tensor * t_emb_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, 1, p->in_ch, p->dim);
    ggml_set_name(t_emb_w, codec_sop_name_embed_w().c_str());
    ggml_tensor * t_emb_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
    ggml_set_name(t_emb_b, codec_sop_name_embed_b().c_str());
    ggml_tensor * x = codec_conv1d(ctx_eval, t_in, t_emb_w, t_emb_b, 1, 1, 0); // [t, dim]
    if (x == nullptr) {
        return false;
    }

    ggml_tensor * t_norm_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
    ggml_set_name(t_norm_w, codec_sop_name_norm_w().c_str());
    ggml_tensor * t_norm_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
    ggml_set_name(t_norm_b, codec_sop_name_norm_b().c_str());

    ggml_tensor * x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x)); // [c, t]
    ggml_set_name(x_ct, "sop.stage.embed.ct");
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f, t_norm_w, t_norm_b);
    if (x_ct == nullptr) {
        return false;
    }
    ggml_set_name(x_ct, "sop.stage.norm.ct");

    const int32_t pad = p->dw_kernel / 2;
    for (int32_t li = 0; li < p->n_layers; ++li) {
        ggml_tensor * res_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_ct)); // [t, c]

        ggml_tensor * t_dw_w = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, p->dw_kernel, 1, p->dim);
        ggml_set_name(t_dw_w, codec_sop_name_cnx_dw_w(li).c_str());
        ggml_tensor * t_dw_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
        ggml_set_name(t_dw_b, codec_sop_name_cnx_dw_b(li).c_str());
        ggml_tensor * x_dw = codec_conv1d_depthwise(ctx_eval, res_tc, t_dw_w, t_dw_b, 1, 1, pad);
        if (x_dw == nullptr) {
            return false;
        }

        ggml_tensor * t_ln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
        ggml_set_name(t_ln_w, codec_sop_name_cnx_ln_w(li).c_str());
        ggml_tensor * t_ln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
        ggml_set_name(t_ln_b, codec_sop_name_cnx_ln_b(li).c_str());
        ggml_tensor * x_blk_ct = codec_op_layer_norm_ct(
            ctx_eval,
            ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_dw)),
            1e-6f,
            t_ln_w,
            t_ln_b);
        if (x_blk_ct == nullptr) {
            return false;
        }

        ggml_tensor * t_pw1_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->dim, p->intermediate);
        ggml_set_name(t_pw1_w, codec_sop_name_cnx_pw1_w(li).c_str());
        ggml_tensor * t_pw1_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->intermediate);
        ggml_set_name(t_pw1_b, codec_sop_name_cnx_pw1_b(li).c_str());
        ggml_tensor * x_pw = codec_op_linear(ctx_eval, x_blk_ct, t_pw1_w, t_pw1_b);
        if (x_pw == nullptr) {
            return false;
        }
        x_pw = ggml_gelu_erf(ctx_eval, x_pw);

        ggml_tensor * t_pw2_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->intermediate, p->dim);
        ggml_set_name(t_pw2_w, codec_sop_name_cnx_pw2_w(li).c_str());
        ggml_tensor * t_pw2_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
        ggml_set_name(t_pw2_b, codec_sop_name_cnx_pw2_b(li).c_str());
        ggml_tensor * x_pw2 = codec_op_linear(ctx_eval, x_pw, t_pw2_w, t_pw2_b);
        if (x_pw2 == nullptr) {
            return false;
        }

        ggml_tensor * t_gamma = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
        ggml_set_name(t_gamma, codec_sop_name_cnx_gamma(li).c_str());
        x_pw2 = codec_op_channel_scale(ctx_eval, x_pw2, t_gamma);
        if (x_pw2 == nullptr) {
            return false;
        }
        x_ct = ggml_add(ctx_eval, ggml_cont(ctx_eval, ggml_transpose(ctx_eval, res_tc)), x_pw2);
        ggml_set_name(x_ct, ("sop.stage.block." + std::to_string(li) + ".ct").c_str());
    }

    ggml_tensor * t_fln_w = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
    ggml_set_name(t_fln_w, codec_sop_name_fln_w().c_str());
    ggml_tensor * t_fln_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->dim);
    ggml_set_name(t_fln_b, codec_sop_name_fln_b().c_str());
    x_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-6f, t_fln_w, t_fln_b);
    if (x_ct == nullptr) {
        return false;
    }
    ggml_set_name(x_ct, "sop.stage.final.ct");

    ggml_tensor * t_head_w = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->dim, p->head_out_dim);
    ggml_set_name(t_head_w, codec_sop_name_head_w().c_str());
    ggml_tensor * t_head_b = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, p->head_out_dim);
    ggml_set_name(t_head_b, codec_sop_name_head_b().c_str());
    ggml_tensor * t_head = codec_op_linear(ctx_eval, x_ct, t_head_w, t_head_b); // [out_dim, t]
    if (t_head == nullptr) {
        return false;
    }
    ggml_tensor * t_out = ggml_cont(ctx_eval, t_head);
    ggml_set_name(t_out, "sop.decode.head.out");
    *out = t_out;
    return true;
}

static bool codec_sop_write_decode_weights(codec_context * ctx, codec_graph_cache_entry * entry, const sop_decode_build & build, std::string * err) {
    auto graph = [&](const std::string & n) { return codec_graph_get_tensor(ctx, entry, n.c_str()); };

    if (!codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_embed_w(), graph(codec_sop_name_embed_w()), err) ||
        !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_embed_b(), graph(codec_sop_name_embed_b()), err) ||
        !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_norm_w(), graph(codec_sop_name_norm_w()), err) ||
        !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_norm_b(), graph(codec_sop_name_norm_b()), err)) {
        return false;
    }

    for (int32_t li = 0; li < build.n_layers; ++li) {
        if (!codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_dw_w(li), graph(codec_sop_name_cnx_dw_w(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_dw_b(li), graph(codec_sop_name_cnx_dw_b(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_ln_w(li), graph(codec_sop_name_cnx_ln_w(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_ln_b(li), graph(codec_sop_name_cnx_ln_b(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_pw1_w(li), graph(codec_sop_name_cnx_pw1_w(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_pw1_b(li), graph(codec_sop_name_cnx_pw1_b(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_pw2_w(li), graph(codec_sop_name_cnx_pw2_w(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_pw2_b(li), graph(codec_sop_name_cnx_pw2_b(li)), err) ||
            !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_cnx_gamma(li), graph(codec_sop_name_cnx_gamma(li)), err)) {
            return false;
        }
    }

    if (!codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_fln_w(), graph(codec_sop_name_fln_w()), err) ||
        !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_fln_b(), graph(codec_sop_name_fln_b()), err) ||
        !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_head_w(), graph(codec_sop_name_head_w()), err) ||
        !codec_runtime_copy_tensor_f32_exact(ctx, codec_sop_name_head_b(), graph(codec_sop_name_head_b()), err)) {
        return false;
    }
    return true;
}

static bool codec_sop_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    const std::vector<float> * window,
    std::vector<float> * out_pcm,
    std::string * err) {

    if (out_pcm == nullptr || out_dim <= 0 || n_frames <= 0 || hop <= 0 || (out_dim % 2) != 0) {
        if (err != nullptr) {
            *err = "invalid Soprano ISTFT arguments";
        }
        return false;
    }
    const int32_t n_bins = out_dim / 2;
    const int32_t n_fft = 2 * (n_bins - 1);
    const float pi = 3.14159265358979323846f;
    if (n_fft <= 0) {
        if (err != nullptr) {
            *err = "invalid Soprano head output dimension";
        }
        return false;
    }

    std::vector<float> win;
    if (window != nullptr && (int32_t)window->size() == n_fft) {
        win = *window;
    } else {
        win.assign((size_t)n_fft, 0.0f);
        for (int32_t n = 0; n < n_fft; ++n) {
            win[(size_t)n] = 0.5f - 0.5f * std::cos(2.0f * pi * (float)n / (float)(n_fft - 1));
        }
    }

    const int32_t pad = n_fft / 2;
    const int32_t out_size = (n_frames - 1) * hop + n_fft;
    std::vector<float> y((size_t) out_size, 0.0f);
    std::vector<float> env((size_t) out_size, 0.0f);
    std::vector<float> frame((size_t) n_fft, 0.0f);

    for (int32_t ti = 0; ti < n_frames; ++ti) {
        for (int32_t n = 0; n < n_fft; ++n) {
            float sum = 0.0f;

            // DC and Nyquist bins are zeroed in the original implementation.
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

enum codec_status codec_soprano_init(struct codec_model * model) {
    if (model == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_soprano & sop = *static_cast<codec_soprano *>(model->impl);

    sop.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", sop.sample_rate);
    sop.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", sop.hop_size);
    sop.n_fft = codec_read_i32_kv(model->gguf, "codec.n_fft", sop.n_fft);
    sop.win_length = codec_read_i32_kv(model->gguf, "codec.win_length", sop.win_length);
    sop.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", sop.latent_dim);
    sop.decoder_dim = codec_read_i32_kv(model->gguf, "soprano.decoder_dim", sop.decoder_dim);
    sop.intermediate_dim = codec_read_i32_kv(model->gguf, "soprano.intermediate_dim", sop.intermediate_dim);
    sop.num_layers = codec_read_i32_kv(model->gguf, "soprano.num_layers", sop.num_layers);
    sop.upscale = codec_read_i32_kv(model->gguf, "soprano.upscale", sop.upscale);
    sop.dw_kernel = codec_read_i32_kv(model->gguf, "soprano.dw_kernel", sop.dw_kernel);
    sop.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", false);
    sop.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    model->sample_rate = sop.sample_rate;
    model->hop_size = sop.hop_size;
    model->n_fft = sop.n_fft;
    model->win_length = sop.win_length;
    model->latent_dim = sop.latent_dim;
    model->has_encoder = sop.has_encoder;
    model->has_decoder = sop.has_decoder;
    model->n_q = 0;
    model->codebook_size = 0;

    return CODEC_STATUS_SUCCESS;
}

static bool codec_sop_init_decode_build(codec_context * ctx, int32_t t, sop_decode_build * build, std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || t <= 0) {
        if (err != nullptr) {
            *err = "invalid Soprano decode build arguments";
        }
        return false;
    }
    const codec_soprano & sop = *static_cast<const codec_soprano *>(ctx->model->impl);
    build->t = t;
    build->in_ch = std::max(1, sop.latent_dim);
    build->dim = std::max(1, sop.decoder_dim);
    build->intermediate = std::max(1, sop.intermediate_dim);
    build->n_layers = std::max(1, sop.num_layers);
    build->dw_kernel = std::max(1, sop.dw_kernel);

    ggml_tensor * head_b = codec_sop_get_tensor(ctx->model, codec_sop_name_head_b());
    if (head_b == nullptr) {
        if (err != nullptr) {
            *err = "missing Soprano head bias tensor";
        }
        return false;
    }
    build->head_out_dim = (int32_t) codec_ne(head_b, 0);
    return true;
}

enum codec_status codec_soprano_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    (void) tokens;
    (void) params;
    if (ctx == nullptr || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_context_set_error(ctx, "Soprano decoder does not accept token inputs; use decode_latent");
    return CODEC_STATUS_NOT_SUPPORTED;
}

enum codec_status codec_soprano_decode_latent(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || out_pcm == nullptr || quantized_representation == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_soprano & sop = *static_cast<codec_soprano *>(ctx->model->impl);
    if (!sop.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (latent_dim <= 0 || n_frames <= 0) {
        codec_context_set_error(ctx, "invalid Soprano latent shape");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (latent_dim != sop.latent_dim) {
        codec_context_set_error(ctx, "Soprano latent_dim mismatch");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t upscale = std::max(1, sop.upscale);
    const int32_t t_up = upscale * (n_frames - 1) + 1;

    std::vector<float> up((size_t) t_up * (size_t) latent_dim, 0.0f);
    for (int32_t c = 0; c < latent_dim; ++c) {
        for (int32_t ti = 0; ti < t_up; ++ti) {
            const int32_t base = std::min(n_frames - 1, ti / upscale);
            const int32_t next = std::min(n_frames - 1, base + 1);
            const float frac = (float)(ti - base * upscale) / (float) upscale;
            const float v0 = quantized_representation[(size_t) base * (size_t) latent_dim + (size_t) c];
            const float v1 = quantized_representation[(size_t) next * (size_t) latent_dim + (size_t) c];
            // ggml tensors are column-major: ne0 is contiguous.
            up[(size_t) c * (size_t) t_up + (size_t) ti] = v0 + (v1 - v0) * frac;
        }
    }

    const int32_t hop = std::max(1, sop.hop_size);
    const size_t mem = 32 * 1024 * 1024 + (size_t) t_up * (size_t) latent_dim * sizeof(float) * 8;
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;
    sop_decode_build build = {};
    if (!codec_sop_init_decode_build(ctx, t_up, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_SOPRANO_DECODE, /*n_frames=*/t_up, /*n_q=*/0, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/latent_dim },
            mem,
            codec_sop_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_in = codec_graph_get_tensor(ctx, entry, "sop.decode.in");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "sop.decode.head.out");
    if (t_in == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached Soprano decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_sop_write_decode_weights(ctx, entry, build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_in, up.data(), up.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = params.n_threads > 0 ? params.n_threads : std::max(1, ctx->model->n_threads);
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> head((size_t) build.head_out_dim * (size_t) t_up, 0.0f);
    if (!codec_runtime_read_tensor(t_out, head.data(), head.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> window;
    ggml_tensor * w_tensor = codec_sop_get_tensor(ctx->model, codec_sop_name_istft_window());
    if (w_tensor != nullptr && w_tensor->ne[1] == 1 && w_tensor->ne[2] == 1) {
        if (!codec_tensor_as_vec_f32(w_tensor, &window)) {
            window.clear();
        }
    }

    std::vector<float> pcm_v;
    const std::vector<float> * win_ptr = window.empty() ? nullptr : &window;
    if (!codec_sop_istft_from_head(head, build.head_out_dim, t_up, hop, win_ptr, &pcm_v, &err)) {
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
    out_pcm->sample_rate = sop.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

static void * codec_sop_create_impl() {
    return new (std::nothrow) codec_soprano();
}

static void codec_sop_destroy_impl(void * impl) {
    codec_soprano * sop = static_cast<codec_soprano *>(impl);
    delete sop;
}

static enum codec_status codec_sop_init_wrap(struct codec_model * model) {
    return codec_soprano_init(model);
}

static enum codec_status codec_sop_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_soprano_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_sop_decode_latent_wrap(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_soprano_decode_latent(ctx, quantized_representation, latent_dim, n_frames, out_pcm, params);
}

const struct codec_model_vtable * codec_soprano_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_SOPRANO,
        "Soprano",
        codec_sop_create_impl,
        codec_sop_destroy_impl,
        codec_sop_init_wrap,
        nullptr,
        codec_sop_decode_wrap,
        codec_sop_decode_latent_wrap,
    };
    return &vtable;
}
