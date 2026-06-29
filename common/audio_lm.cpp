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
// Prompt build
//
// For step 2 the implemented branch is the speaker-encode path: when
// the model has a speaker_encoder section AND the caller supplied either
// `ref_pcm` or `speaker_emb` (+ `ref_speech_tokens` if needed), run
// `codec_lm_speaker_encode` and populate `embeds_prefix` with the
// resulting (n_rows × hidden_dim) matrix.
//
// `tokens` is left empty — text tokenization runs in the host (which
// owns the llama_model + tokenizer), and codec_common doesn't depend on
// llama.cpp.  Once Type A reference models join, build_prompt may also
// emit a few seed tokens (e.g. audio_start markers) directly.
//
// Sampler hints + start/stop tokens are zero for now; they'll get
// populated from `codec.lm.sampling.*` GGUF keys in a follow-up — for
// step 2 the host fills them in itself.
//
// Returns true even when no speaker encoding happens (Type A / zero-
// shot models) — empty `tokens` + empty `embeds_prefix` is a valid
// "host does everything text-side" prompt.
// =====================================================================

bool audio_lm_build_prompt(audio_lm_context * ctx,
                            const audio_lm_input  & in,
                            audio_lm_prompt       * out) {
    if (ctx == nullptr || out == nullptr) return false;
    out->tokens.clear();
    out->embeds_prefix.clear();
    out->embeds_prefix_rows   = 0;
    out->embeds_prefix_hidden = 0;
    out->embeds_uncond.clear();
    out->default_temperature        = 0.0f;
    out->default_top_p              = 0.0f;
    out->default_min_p              = 0.0f;
    out->default_repetition_penalty = 0.0f;
    out->default_cfg_weight         = 0.0f;
    out->start_token                = -1;
    out->stop_token                 = -1;

    if (!ctx->has_spk_enc) {
        // Type A / zero-shot — no speaker prefix to compute.  Caller
        // still gets a successful (empty) prompt; the host AR loop runs
        // entirely on its own tokenized text input.
        return true;
    }

    // ────────────────────────────────────────────────────────────────
    // Speaker-encode branch.  Two flavours depending on what the caller
    // has on hand: (a) raw `ref_pcm` → run the full speaker encoder
    // (VE + cond_enc for Chatterbox, ECAPA-TDNN for Qwen3-TTS);
    // (b) cached `speaker_emb` → skip the front-end via `_from_embedding`.
    // ────────────────────────────────────────────────────────────────
    const codec_lm_speaker_info * si = codec_lm_speaker_get_info(ctx->lm);
    if (si == nullptr || si->n_rows <= 0 || si->hidden_dim <= 0) {
        ctx->last_error = "audio_lm_build_prompt: speaker_info missing or invalid";
        return false;
    }
    const int32_t need_elems = si->n_rows * si->hidden_dim;
    out->embeds_prefix.resize((size_t) need_elems);

    const bool has_pcm = (in.ref_pcm != nullptr && in.ref_n_samples > 0);
    const bool has_emb = (in.speaker_emb != nullptr && in.speaker_emb_dim > 0);

    if (!has_pcm && !has_emb) {
        // Caller declared neither — but the model has a speaker encoder.
        // Allow the no-speaker path (build_prompt returns success with
        // empty prefix) only when the model doesn't strictly require ref
        // audio; otherwise complain.
        if (si->needs_ref_pcm || si->needs_ref_speech_tokens) {
            ctx->last_error =
                "audio_lm_build_prompt: model requires ref_pcm / ref_speech_tokens "
                "but neither ref_pcm nor speaker_emb was provided";
            out->embeds_prefix.clear();
            return false;
        }
        out->embeds_prefix.clear();
        return true;
    }

    codec_audio audio = {};
    if (has_pcm) {
        audio.data        = in.ref_pcm;
        audio.n_samples   = in.ref_n_samples;
        audio.sample_rate = in.ref_sample_rate > 0
                            ? in.ref_sample_rate
                            : si->ref_sample_rate;
        audio.n_channels  = 1;
        audio.pcm_type    = CODEC_PCM_TYPE_F32;
    }

    // ref_speech_tokens are model-specific (Chatterbox needs them; the
    // Qwen3-TTS ECAPA path doesn't).  For now they come from `in.extra`
    // when the model needs them: key `"ref_speech_tokens_csv"` =
    // comma-separated int32 ids.  This stays a private convention until
    // the input schema doc lands; tts-cli doesn't drive it yet.
    std::vector<int32_t> ref_codes;
    const int32_t * ref_codes_ptr = nullptr;
    int32_t         ref_codes_n   = 0;
    if (si->needs_ref_speech_tokens) {
        auto it = in.extra.find("ref_speech_tokens_csv");
        if (it != in.extra.end()) {
            const std::string & s = it->second;
            size_t i = 0;
            while (i < s.size()) {
                size_t j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                if (j > i) {
                    try { ref_codes.push_back(std::stoi(s.substr(i, j - i))); }
                    catch (...) { /* skip malformed */ }
                }
                i = j + 1;
            }
        }
        if (!ref_codes.empty()) {
            ref_codes_ptr = ref_codes.data();
            ref_codes_n   = (int32_t) ref_codes.size();
        }
    }

    const enum codec_status rc = has_pcm
        ? codec_lm_speaker_encode(
            ctx->lm, &audio, ref_codes_ptr, ref_codes_n, in.emotion,
            out->embeds_prefix.data(), need_elems)
        : codec_lm_speaker_encode_from_embedding(
            ctx->lm, in.speaker_emb, in.speaker_emb_dim,
            ref_codes_ptr, ref_codes_n, in.emotion,
            out->embeds_prefix.data(), need_elems);

    if (rc != CODEC_STATUS_SUCCESS) {
        const char * raw = codec_lm_get_last_error(ctx->lm);
        ctx->last_error = std::string("audio_lm_build_prompt: speaker_encode failed (")
                          + (raw && *raw ? raw : "?") + ")";
        out->embeds_prefix.clear();
        return false;
    }
    out->embeds_prefix_rows   = si->n_rows;
    out->embeds_prefix_hidden = si->hidden_dim;
    return true;
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
// External codes push (offline / debug)
//
// Appends to the same `ctx->codes` accumulator `observe_token` will
// populate during AR.  Format mirrors codec_token_buffer: (T, n_q)
// interleaved row-major.
// =====================================================================

bool audio_lm_push_codes(audio_lm_context * ctx,
                          const int32_t   * codes,
                          int32_t           n_frames,
                          int32_t           n_q) {
    if (ctx == nullptr || codes == nullptr) return false;
    if (n_frames <= 0 || n_q <= 0) {
        ctx->last_error = "audio_lm_push_codes: non-positive n_frames / n_q";
        return false;
    }
    // codec_lm absent → we can still accumulate; decode_audio works on
    // codec_model only, n_q comes from the caller.  Just don't enforce
    // a model-side n_codebook check in that case.
    if (ctx->n_cb > 0 && n_q != ctx->n_cb) {
        ctx->last_error = "audio_lm_push_codes: n_q doesn't match model n_codebook";
        return false;
    }
    if (ctx->codes_n_frames == 0) {
        ctx->n_cb = n_q;   // remember it so decode_audio can lay out the buffer
    }
    const size_t prev = ctx->codes.size();
    ctx->codes.resize(prev + (size_t) n_frames * (size_t) n_q);
    std::memcpy(ctx->codes.data() + prev, codes,
                (size_t) n_frames * (size_t) n_q * sizeof(int32_t));
    ctx->codes_n_frames += n_frames;
    return true;
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
