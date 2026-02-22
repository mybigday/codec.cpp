#include "wavtokenizer.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
};

static bool codec_wt_build_decode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    wt_decode_build * p = static_cast<wt_decode_build *>(user_data);
    ggml_tensor * t_tok = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->t, p->q);
    ggml_set_name(t_tok, "wt.decode.tok");

    ggml_tensor * t_feat = codec_op_tokens_to_features(ctx_eval, t_tok, 1);
    ggml_tensor * t_alpha = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 1);
    ggml_set_name(t_alpha, "wt.decode.alpha");

    ggml_tensor * t_snake = codec_op_snake(ctx_eval, t_feat, t_alpha, 1e-4f);
    ggml_tensor * t_kernel = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F16, p->hop, 1, 1);
    ggml_set_name(t_kernel, "wt.decode.kernel");

    ggml_tensor * t_pcm = codec_convtr1d(ctx_eval, t_snake, t_kernel, nullptr, p->hop, 0, 1);
    ggml_tensor * t_out = ggml_cont(ctx_eval, ggml_tanh(ctx_eval, t_pcm));
    ggml_set_name(t_out, "wt.decode.out");

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

    wt_decode_build build = { t, q, hop };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
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
    ggml_tensor * t_alpha = codec_graph_get_tensor(ctx, entry, "wt.decode.alpha");
    ggml_tensor * t_kernel = codec_graph_get_tensor(ctx, entry, "wt.decode.kernel");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "wt.decode.out");
    if (t_tok == nullptr || t_alpha == nullptr || t_kernel == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached WavTokenizer decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> tok_f32((size_t) t * (size_t) q, 0.0f);
    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < q; ++qi) {
            tok_f32[(size_t) qi * (size_t) t + (size_t) ti] = (float) tokens->data[(size_t) ti * (size_t) tokens->n_q + (size_t) qi];
        }
    }
    std::vector<ggml_fp16_t> kernel_f16;
    if (!codec_wt_load_kernel_f16(ctx, "wt.decode.kernel", hop, &kernel_f16, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const float alpha = 0.5f;

    if (!codec_runtime_write_tensor(t_tok, tok_f32.data(), tok_f32.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_alpha, &alpha, sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_kernel, kernel_f16.data(), kernel_f16.size() * sizeof(ggml_fp16_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_samples = (int32_t) t_out->ne[0];
    float * pcm = static_cast<float *>(std::malloc((size_t) n_samples * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_read_tensor(t_out, pcm, (size_t) n_samples * sizeof(float), &err)) {
        std::free(pcm);
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = pcm;
    out_pcm->n_samples = n_samples;
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
