#include "dac.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

enum codec_status codec_dac_init(struct codec_model * model) {
    codec_dac & dac = model->dac;

    dac.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    dac.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 512);
    dac.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", 4);
    dac.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 1024);
    dac.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", 1024);
    dac.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", 8);
    dac.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", true);
    dac.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    model->sample_rate = dac.sample_rate;
    model->has_encoder = dac.has_encoder;
    model->has_decoder = dac.has_decoder;
    model->hop_size = dac.hop_size;
    model->n_q = dac.n_q;
    model->codebook_size = dac.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = dac.latent_dim;

    return CODEC_STATUS_SUCCESS;
}

struct dac_decode_build {
    int32_t t;
    int32_t q;
    int32_t hop;
};

static bool codec_dac_build_decode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    dac_decode_build * p = static_cast<dac_decode_build *>(user_data);
    ggml_tensor * t_tok = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->t, p->q);
    ggml_set_name(t_tok, "dac.decode.tok");

    ggml_tensor * t_feat = codec_op_tokens_to_features(ctx_eval, t_tok, 1);
    ggml_tensor * t_kernel = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, p->hop, 1, 1);
    ggml_set_name(t_kernel, "dac.decode.kernel");

    ggml_tensor * t_pcm = codec_convtr1d(ctx_eval, t_feat, t_kernel, nullptr, p->hop, 0, 1);
    ggml_tensor * t_out = ggml_cont(ctx_eval, ggml_tanh(ctx_eval, t_pcm));
    ggml_set_name(t_out, "dac.decode.out");

    *out = t_out;
    return true;
}

struct dac_decode_latent_build {
    int32_t n_frames;
    int32_t latent_dim;
    int32_t hop;
};

static bool codec_dac_build_decode_latent(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    dac_decode_latent_build * p = static_cast<dac_decode_latent_build *>(user_data);
    ggml_tensor * t_lat = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->n_frames, p->latent_dim);
    ggml_set_name(t_lat, "dac.decode_latent.lat");

    ggml_tensor * t_ch0 = ggml_view_2d(ctx_eval, t_lat, p->n_frames, 1, t_lat->nb[1], 0);
    ggml_tensor * t_kernel = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, p->hop, 1, 1);
    ggml_set_name(t_kernel, "dac.decode_latent.kernel");

    ggml_tensor * t_pcm = codec_convtr1d(ctx_eval, t_ch0, t_kernel, nullptr, p->hop, 0, 1);
    ggml_tensor * t_out = ggml_cont(ctx_eval, ggml_tanh(ctx_eval, t_pcm));
    ggml_set_name(t_out, "dac.decode_latent.out");

    *out = t_out;
    return true;
}

struct dac_encode_build {
    int32_t n_in;
    int32_t hop;
};

static bool codec_dac_build_encode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    dac_encode_build * p = static_cast<dac_encode_build *>(user_data);
    ggml_tensor * t_pcm = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->n_in, 1);
    ggml_set_name(t_pcm, "dac.encode.pcm");

    ggml_tensor * t_kernel = ggml_new_tensor_3d(ctx_eval, GGML_TYPE_F32, p->hop, 1, 1);
    ggml_set_name(t_kernel, "dac.encode.kernel");

    ggml_tensor * t_feat = ggml_cont(ctx_eval, codec_conv1d(ctx_eval, t_pcm, t_kernel, nullptr, p->hop, 1, 0));
    ggml_set_name(t_feat, "dac.encode.out");

    *out = t_feat;
    return true;
}

static enum codec_status codec_dac_decode_tokens_graph(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm,
    int32_t hop_size,
    int32_t sample_rate) {

    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid DAC token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    const size_t mem = 32 * 1024 * 1024 + (size_t) t * (size_t) q * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);

    dac_decode_build build = { t, q, hop_size };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_DAC_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop_size, /*n_in=*/0, /*latent_dim=*/0 },
            mem,
            codec_dac_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "dac.decode.tok");
    ggml_tensor * t_kernel = codec_graph_get_tensor(ctx, entry, "dac.decode.kernel");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "dac.decode.out");
    if (t_tok == nullptr || t_kernel == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached DAC decode graph is invalid");
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
    std::vector<float> kernel((size_t) hop_size, 1.0f / (float) std::max(1, hop_size));

    if (!codec_runtime_write_tensor(t_tok, tok_f32.data(), tok_f32.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_kernel, kernel.data(), kernel.size() * sizeof(float), &err)) {
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
    out_pcm->sample_rate = sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_dac_decode_latent(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    (void) params;

    codec_dac & dac = ctx->model->dac;
    if (!dac.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (quantized_representation == nullptr || latent_dim <= 0 || n_frames <= 0) {
        codec_context_set_error(ctx, "invalid DAC latent input");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t hop = std::max(1, dac.hop_size);
    const size_t mem = 32 * 1024 * 1024 + (size_t) n_frames * (size_t) latent_dim * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);

    dac_decode_latent_build build = { n_frames, latent_dim, hop };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_DAC_DECODE_LATENT, /*n_frames=*/n_frames, /*n_q=*/0, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/latent_dim },
            mem,
            codec_dac_build_decode_latent,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_lat = codec_graph_get_tensor(ctx, entry, "dac.decode_latent.lat");
    ggml_tensor * t_kernel = codec_graph_get_tensor(ctx, entry, "dac.decode_latent.kernel");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "dac.decode_latent.out");
    if (t_lat == nullptr || t_kernel == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached DAC latent decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> kernel((size_t) hop, 1.0f / (float) hop);
    if (!codec_runtime_write_tensor(t_lat, quantized_representation, (size_t) n_frames * (size_t) latent_dim * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_kernel, kernel.data(), kernel.size() * sizeof(float), &err)) {
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
    out_pcm->sample_rate = dac.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_dac_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_dac & dac = ctx->model->dac;
    if (!dac.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, dac.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "DAC decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    return codec_dac_decode_tokens_graph(
        ctx,
        tokens,
        use_n_q,
        out_pcm,
        std::max(1, dac.hop_size),
        dac.sample_rate);
}

enum codec_status codec_dac_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    codec_dac & dac = ctx->model->dac;
    if (!dac.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (pcm.empty()) {
        codec_context_set_error(ctx, "empty pcm");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t model_n_q = std::max(1, dac.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "DAC encode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t hop = std::max(1, params.hop_size > 0 ? params.hop_size : dac.hop_size);
    const int32_t n_in = (int32_t) pcm.size();

    const size_t mem = 32 * 1024 * 1024 + (size_t) n_in * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);
    dac_encode_build build = { n_in, hop };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_DAC_ENCODE, /*n_frames=*/0, /*n_q=*/0, /*hop=*/hop, /*n_in=*/n_in, /*latent_dim=*/0 },
            mem,
            codec_dac_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "dac.encode.pcm");
    ggml_tensor * t_kernel = codec_graph_get_tensor(ctx, entry, "dac.encode.kernel");
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "dac.encode.out");
    if (t_pcm == nullptr || t_kernel == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached DAC encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> kernel((size_t) hop, 1.0f / (float) hop);
    if (!codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_kernel, kernel.data(), kernel.size() * sizeof(float), &err)) {
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

    const int32_t codebook_size = std::max(2, dac.codebook_size);
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
    out_tokens->sample_rate = dac.sample_rate;
    out_tokens->hop_size = hop;

    if (out_latent != nullptr) {
        float * latent = static_cast<float *>(std::malloc((size_t) n_frames * sizeof(float)));
        if (latent == nullptr) {
            codec_token_buffer_free(out_tokens);
            codec_context_set_error(ctx, "failed to allocate latent output");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(latent, feat.data(), feat.size() * sizeof(float));
        codec_latent_buffer_reset(out_latent);
        out_latent->data = latent;
        out_latent->latent_dim = 1;
        out_latent->n_frames = n_frames;
        out_latent->sample_rate = dac.sample_rate;
        out_latent->hop_size = hop;
    }

    return CODEC_STATUS_SUCCESS;
}
