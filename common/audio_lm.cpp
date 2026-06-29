#include "codec_common.h"

#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace codec_common {

// =====================================================================
// Internal context.  Holds the codec_model + codec_context + codec_lm
// handles plus the per-sequence state buffers.
// =====================================================================
struct audio_lm_context {
    codec_model    * model      = nullptr;
    codec_context  * codec_ctx  = nullptr;
    codec_lm       * lm         = nullptr;
    codec_lm_state * state      = nullptr;

    // Cached capabilities (read once at init).
    uint32_t modality_mask = 0;
    int32_t  n_cb          = 0;
    int32_t  hidden        = 0;
    bool     has_spk_enc   = false;

    // Per-sequence buffers (cleared on reset).
    //
    // `codes` is laid out (T, n_cb) interleaved: codes[t*n_cb + q] = code
    // for codebook q at frame t.  Matches codec_token_buffer convention.
    std::vector<int32_t> codes;
    int32_t              codes_n_frames = 0;

    // The vector returned by `audio_lm_get_next_embed`.  Owned by ctx;
    // valid until the next observe_token / reset / free.
    std::vector<float> next_embed_buf;
    int32_t            next_embed_dim = 0;

    mutable std::string last_error;
};

// ─────────────────────────────────────────────────────────────────────
// Modality detection
// ─────────────────────────────────────────────────────────────────────
//
// Prefer explicit `codec.lm.modality.*` GGUF keys when present (set by
// the converter).  When absent (legacy GGUFs), infer from what the
// codec / codec_lm side actually supports:
//
//   * codec has decoder              → OUTPUT_AUDIO
//   * codec has encoder OR has_spk_enc → INPUT_AUDIO
//   * codec_lm present               → INPUT_TEXT (TTS-style models
//                                                   consume a text
//                                                   prompt)
//   * OUTPUT_TEXT  → only when explicitly declared (no robust signal
//                    to infer it from existing GGUFs).
//
// This keeps the API working today for the GGUFs we already shipped;
// new converters should write the explicit keys so the heuristic isn't
// needed.

static uint32_t read_modality_or_infer(audio_lm_context * ctx) {
    uint32_t mask = 0;

    const codec_gguf_metadata * meta = codec_model_metadata(ctx->model);
    bool saw_explicit = false;

    if (meta != nullptr) {
        for (size_t i = 0; i < meta->n_items; ++i) {
            const char * key = meta->items[i].key;
            const char * val = meta->items[i].value;
            if (key == nullptr || val == nullptr) continue;
            // Match "true" loosely — codec_gguf_metadata serialises
            // bools as the strings "true" / "false".
            const bool on = (std::strcmp(val, "true") == 0 ||
                             std::strcmp(val, "1")    == 0);
            if      (std::strcmp(key, "codec.lm.modality.input_text"  ) == 0) { if (on) mask |= INPUT_TEXT;   saw_explicit = true; }
            else if (std::strcmp(key, "codec.lm.modality.input_audio" ) == 0) { if (on) mask |= INPUT_AUDIO;  saw_explicit = true; }
            else if (std::strcmp(key, "codec.lm.modality.output_text" ) == 0) { if (on) mask |= OUTPUT_TEXT;  saw_explicit = true; }
            else if (std::strcmp(key, "codec.lm.modality.output_audio") == 0) { if (on) mask |= OUTPUT_AUDIO; saw_explicit = true; }
        }
    }

    if (!saw_explicit) {
        if (codec_model_has_decoder(ctx->model)) mask |= OUTPUT_AUDIO;
        // INPUT_AUDIO heuristic: the LM consumes ref / prompt audio.
        // `codec_model_has_encoder` is too generous — bidirectional
        // codecs (Mimi, Qwen3-TTS-Tokenizer, XY-Tokenizer) expose an
        // encoder even when the LM is zero-shot (CSM).  Use the LM's
        // own speaker_encoder section as the only positive signal;
        // models that consume ref audio without an explicit speaker
        // encoder (MOSS-TTSD's interleaved speech-tokenizer prompt)
        // should set `codec.lm.modality.input_audio=true` at convert
        // time to opt in.
        if (ctx->has_spk_enc) mask |= INPUT_AUDIO;
        if (ctx->lm != nullptr) mask |= INPUT_TEXT;
    }

    return mask;
}

// =====================================================================
// Lifecycle
// =====================================================================

audio_lm_context * audio_lm_init(const audio_lm_params & p, std::string * err) {
    if (p.codec_path.empty()) {
        if (err) *err = "audio_lm_init: codec_path is empty";
        return nullptr;
    }
    auto * ctx = new (std::nothrow) audio_lm_context();
    if (ctx == nullptr) {
        if (err) *err = "audio_lm_init: out of memory";
        return nullptr;
    }

    auto mp = codec_model_default_params();
    mp.use_gpu   = p.use_gpu;
    if (p.n_threads > 0) mp.n_threads = p.n_threads;
    ctx->model = codec_model_load_from_file(p.codec_path.c_str(), mp);
    if (ctx->model == nullptr) {
        if (err) *err = "audio_lm_init: codec_model_load_from_file failed for " + p.codec_path;
        delete ctx;
        return nullptr;
    }

    auto cp = codec_context_default_params();
    ctx->codec_ctx = codec_init_from_model(ctx->model, cp);
    if (ctx->codec_ctx == nullptr) {
        if (err) *err = "audio_lm_init: codec_init_from_model failed";
        codec_model_free(ctx->model);
        delete ctx;
        return nullptr;
    }

    // LM adaptor is OPTIONAL.  codec-only GGUFs (e.g. wavtokenizer.gguf,
    // dac.gguf) still work for decode_audio paths — the AR-side hooks
    // just return NOT_SUPPORTED at observe / build time.
    ctx->lm = codec_lm_create(ctx->model);
    if (ctx->lm != nullptr) {
        ctx->state = codec_lm_state_new(ctx->lm);
        const codec_lm_info * info = codec_lm_get_info(ctx->lm);
        if (info != nullptr) {
            ctx->n_cb   = info->n_codebook;
            ctx->hidden = info->hidden_dim;
        }
        ctx->has_spk_enc = (codec_lm_speaker_get_info(ctx->lm) != nullptr);
    }

    ctx->modality_mask = read_modality_or_infer(ctx);
    return ctx;
}

void audio_lm_free(audio_lm_context * ctx) {
    if (ctx == nullptr) return;
    if (ctx->state)     codec_lm_state_free(ctx->state);
    if (ctx->lm)        codec_lm_free(ctx->lm);
    if (ctx->codec_ctx) codec_free(ctx->codec_ctx);
    if (ctx->model)     codec_model_free(ctx->model);
    delete ctx;
}

void audio_lm_reset(audio_lm_context * ctx) {
    if (ctx == nullptr) return;
    ctx->codes.clear();
    ctx->codes_n_frames = 0;
    ctx->next_embed_buf.clear();
    ctx->next_embed_dim = 0;
    if (ctx->state) codec_lm_state_reset(ctx->state);
    ctx->last_error.clear();
}

// =====================================================================
// Capability queries
// =====================================================================

uint32_t audio_lm_modality(const audio_lm_context * ctx) {
    return ctx ? ctx->modality_mask : 0u;
}

bool audio_lm_has_speaker_enc(const audio_lm_context * ctx) {
    return ctx ? ctx->has_spk_enc : false;
}

int32_t audio_lm_n_codebook(const audio_lm_context * ctx) {
    return ctx ? ctx->n_cb : 0;
}

int32_t audio_lm_hidden_dim(const audio_lm_context * ctx) {
    return ctx ? ctx->hidden : 0;
}

const char * audio_lm_last_error(const audio_lm_context * ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

// =====================================================================
// Prompt build + per-step observe + next-embed lookup
//
// Skeletons.  Reference impls per inference Type land in roadmap steps
// 2–5.  Returning a clear NOT_SUPPORTED-style error keeps the lifecycle
// + decode_audio paths usable today for codec-only smoke tests and
// makes it obvious when a host hits an un-wired path.
// =====================================================================

bool audio_lm_build_prompt(audio_lm_context * ctx,
                            const audio_lm_input  & /*in*/,
                            audio_lm_prompt       * /*out*/) {
    if (ctx) ctx->last_error =
        "audio_lm_build_prompt: not yet implemented for this model "
        "(reference impls land in roadmap steps 2–5)";
    return false;
}

observe_action audio_lm_observe_token(
        audio_lm_context * ctx,
        codec_common_token /*tok*/,
        const float *      /*last_hidden*/,
        int32_t            /*hidden_dim*/) {
    if (ctx) ctx->last_error =
        "audio_lm_observe_token: not yet implemented for this model "
        "(reference impls land in roadmap steps 3–5)";
    return OBSERVE_STOP;
}

const float * audio_lm_get_next_embed(const audio_lm_context * ctx,
                                       int32_t * out_dim) {
    if (out_dim) *out_dim = ctx ? ctx->next_embed_dim : 0;
    return (ctx && !ctx->next_embed_buf.empty()) ? ctx->next_embed_buf.data() : nullptr;
}

// =====================================================================
// End of sequence — codes → PCM
// =====================================================================

bool audio_lm_decode_audio(audio_lm_context * ctx, audio_lm_audio_output * out) {
    if (ctx == nullptr || out == nullptr) return false;
    if (ctx->codes_n_frames <= 0 || ctx->n_cb <= 0) {
        ctx->last_error = "audio_lm_decode_audio: no codes accumulated";
        return false;
    }
    if (!(ctx->modality_mask & OUTPUT_AUDIO)) {
        ctx->last_error = "audio_lm_decode_audio: model has no OUTPUT_AUDIO modality";
        return false;
    }
    if (!codec_model_has_decoder(ctx->model)) {
        ctx->last_error = "audio_lm_decode_audio: codec has no decoder";
        return false;
    }

    codec_token_buffer tokens = {};
    tokens.data         = ctx->codes.data();
    tokens.n_tokens     = (int32_t) ctx->codes.size();
    tokens.n_frames     = ctx->codes_n_frames;
    tokens.n_q          = ctx->n_cb;
    tokens.codebook_size = codec_model_codebook_size(ctx->model);
    tokens.sample_rate   = codec_model_sample_rate(ctx->model);
    tokens.hop_size      = codec_model_hop_size(ctx->model);

    codec_pcm_buffer pcm = {};
    auto dp = codec_decode_default_params();
    const enum codec_status rc =
        codec_decode(ctx->codec_ctx, &tokens, &pcm, dp);
    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_get_last_error(ctx->codec_ctx);
        ctx->last_error = std::string("audio_lm_decode_audio: codec_decode failed (")
                           + (raw ? raw : "?") + ")";
        return false;
    }

    out->pcm.assign(pcm.data, pcm.data + (size_t) pcm.n_samples * (size_t) pcm.n_channels);
    out->sample_rate = pcm.sample_rate;
    out->n_channels  = pcm.n_channels;
    codec_pcm_buffer_free(&pcm);
    return true;
}

}  // namespace codec_common
