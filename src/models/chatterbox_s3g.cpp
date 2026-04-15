#include "chatterbox_s3g.h"

#include <new>

enum codec_status codec_chatterbox_s3g_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_chatterbox_s3g & s3g = *static_cast<codec_chatterbox_s3g *>(model->impl);
    s3g.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", s3g.sample_rate);
    s3g.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", s3g.hop_size);
    s3g.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", s3g.n_q);
    s3g.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", s3g.codebook_size);
    s3g.meanflow = codec_read_bool_kv(model->gguf, "chatterbox_s3g.meanflow", s3g.meanflow);
    s3g.has_builtin_conditioning = codec_read_bool_kv(
        model->gguf, "chatterbox_s3g.has_builtin_conditioning", s3g.has_builtin_conditioning);
    s3g.builtin_prompt_token_len = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.prompt_token_len", s3g.builtin_prompt_token_len);
    s3g.builtin_prompt_feat_frames = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.prompt_feat_frames", s3g.builtin_prompt_feat_frames);
    s3g.builtin_prompt_feat_dim = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.prompt_feat_dim", s3g.builtin_prompt_feat_dim);
    s3g.builtin_embedding_dim = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.embedding_dim", s3g.builtin_embedding_dim);
    codec_read_i32_array_kv_vec(model->gguf, "chatterbox_s3g.cond.prompt_token", &s3g.builtin_prompt_token);
    s3g.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", s3g.has_encoder);
    s3g.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", s3g.has_decoder);

    model->sample_rate = s3g.sample_rate;
    model->encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", 0);
    model->has_encoder = s3g.has_encoder;
    model->has_decoder = s3g.has_decoder;
    model->hop_size = s3g.hop_size;
    model->n_q = s3g.n_q;
    model->codebook_size = s3g.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = -1;

    if (s3g.n_q != 1 || s3g.codebook_size != 6561 || s3g.sample_rate <= 0 || s3g.hop_size <= 0) {
        return CODEC_STATUS_INVALID_ARG;
    }

    if (s3g.has_builtin_conditioning) {
        if (s3g.builtin_prompt_token_len <= 0 ||
            s3g.builtin_prompt_feat_frames <= 0 ||
            s3g.builtin_prompt_feat_dim <= 0 ||
            s3g.builtin_embedding_dim <= 0 ||
            s3g.builtin_prompt_token.empty() ||
            s3g.builtin_prompt_token_len > (int32_t) s3g.builtin_prompt_token.size()) {
            return CODEC_STATUS_INVALID_ARG;
        }

        ggml_tensor * prompt_feat = ggml_get_tensor(model->weights, "s3g.cond.prompt_feat");
        ggml_tensor * embedding = ggml_get_tensor(model->weights, "s3g.cond.embedding");
        if (prompt_feat == nullptr || embedding == nullptr) {
            return CODEC_STATUS_INVALID_ARG;
        }
        if (prompt_feat->type != GGML_TYPE_F32 || embedding->type != GGML_TYPE_F32) {
            return CODEC_STATUS_INVALID_ARG;
        }
        if (prompt_feat->ne[0] != s3g.builtin_prompt_feat_dim ||
            prompt_feat->ne[1] != s3g.builtin_prompt_feat_frames ||
            prompt_feat->ne[2] != 1 ||
            embedding->ne[0] != s3g.builtin_embedding_dim ||
            embedding->ne[1] != 1) {
            return CODEC_STATUS_INVALID_ARG;
        }
    }

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_chatterbox_s3g_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * /*tokens*/,
    struct codec_pcm_buffer * /*out_pcm*/,
    struct codec_decode_params /*params*/) {
    const codec_chatterbox_s3g * s3g =
        ctx != nullptr && ctx->model != nullptr ? static_cast<const codec_chatterbox_s3g *>(ctx->model->impl) : nullptr;
    if (s3g != nullptr && s3g->has_builtin_conditioning) {
        codec_context_set_error(ctx, "Chatterbox-S3G decode is not implemented yet (builtin fixed-voice conditioning is loaded)");
    } else {
        codec_context_set_error(ctx, "Chatterbox-S3G decode is not implemented yet (no builtin conditioning in model)");
    }
    return CODEC_STATUS_NOT_SUPPORTED;
}

static void * codec_chatterbox_s3g_create_impl() {
    return new (std::nothrow) codec_chatterbox_s3g();
}

static void codec_chatterbox_s3g_destroy_impl(void * ptr) {
    delete static_cast<codec_chatterbox_s3g *>(ptr);
}

const struct codec_model_vtable * codec_chatterbox_s3g_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_CHATTERBOX_S3G,
        "Chatterbox-S3G",
        codec_chatterbox_s3g_create_impl,
        codec_chatterbox_s3g_destroy_impl,
        codec_chatterbox_s3g_init,
        nullptr,
        codec_chatterbox_s3g_decode,
        nullptr,
    };
    return &vtable;
}
