// C API over codec_common's audio-LM layer (include/codec_audio_lm.h).
// Thin forwarding: every call maps to the codec_common:: function of the
// same name; this file only converts types and owns the returned strings.

#include "codec_audio_lm.h"
#include "codec_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

using namespace codec_common;

struct codec_audio_lm {
    audio_lm_context * ctx  = nullptr;
    bool               bare = false;
    audio_lm_prompt                     prompt;
    audio_lm_prompt_info                info;
    std::vector<audio_lm_speaker_entry> speakers;
};

static bool fill_pcm(const audio_lm_audio_output & in, codec_pcm_buffer * out) {
    if (out == nullptr) return false;
    out->data = static_cast<float *>(std::malloc(std::max<size_t>(1, in.pcm.size()) * sizeof(float)));
    if (out->data == nullptr) return false;
    if (!in.pcm.empty()) std::memcpy(out->data, in.pcm.data(), in.pcm.size() * sizeof(float));
    out->n_channels  = in.n_channels;
    out->n_samples   = in.n_channels > 0 ? (int32_t) (in.pcm.size() / (size_t) in.n_channels) : 0;
    out->sample_rate = in.sample_rate;
    return true;
}

static int32_t copy_str(const std::string & s, char * buf, int32_t buf_size) {
    if (buf != nullptr && buf_size > 0) {
        const size_t n = std::min(s.size(), (size_t) buf_size - 1);
        std::memcpy(buf, s.data(), n);
        buf[n] = '\0';
    }
    return (int32_t) s.size();
}

extern "C" {

struct codec_audio_lm_params codec_audio_lm_default_params(void) {
    codec_audio_lm_params p{};
    return p;
}

struct codec_audio_lm * codec_audio_lm_init(const struct codec_audio_lm_params * params, char * err, size_t err_size) {
    std::string e;
    if (params == nullptr || params->codec_path == nullptr) {
        e = "codec_audio_lm_init: codec_path is required";
    } else {
        audio_lm_params p;
        p.codec_path = params->codec_path;
        p.use_gpu    = params->use_gpu;
        p.n_threads  = params->n_threads;
        if (params->kv_overrides != nullptr && params->n_kv_overrides > 0) {
            p.kv_overrides.assign(params->kv_overrides, params->kv_overrides + params->n_kv_overrides);
        }
        if (audio_lm_context * ctx = audio_lm_init(p, &e)) {
            auto * w = new (std::nothrow) codec_audio_lm();
            if (w != nullptr) { w->ctx = ctx; return w; }
            audio_lm_free(ctx);
            e = "codec_audio_lm_init: out of memory";
        }
    }
    if (err != nullptr && err_size > 0) std::snprintf(err, err_size, "%s", e.c_str());
    return nullptr;
}

struct codec_audio_lm * codec_audio_lm_alloc_bare(void) {
    audio_lm_context * ctx = audio_lm_alloc_bare();
    if (ctx == nullptr) return nullptr;
    auto * w = new (std::nothrow) codec_audio_lm();
    if (w == nullptr) { audio_lm_free_bare(ctx); return nullptr; }
    w->ctx = ctx;
    w->bare = true;
    return w;
}

void codec_audio_lm_free(struct codec_audio_lm * w) {
    if (w == nullptr) return;
    if (w->bare) audio_lm_free_bare(w->ctx);
    else         audio_lm_free(w->ctx);
    delete w;
}

void codec_audio_lm_reset(struct codec_audio_lm * w) { if (w) audio_lm_reset(w->ctx); }

bool codec_audio_lm_parse_kv_override(const char * spec, struct codec_model_kv_override * out) {
    return parse_kv_override(spec, out);
}

uint32_t codec_audio_lm_modality(const struct codec_audio_lm * w)        { return w ? audio_lm_modality(w->ctx) : 0; }
bool     codec_audio_lm_has_speaker_enc(const struct codec_audio_lm * w) { return w && audio_lm_has_speaker_enc(w->ctx); }
int32_t  codec_audio_lm_n_codebook(const struct codec_audio_lm * w)      { return w ? audio_lm_n_codebook(w->ctx) : 0; }
int32_t  codec_audio_lm_hidden_dim(const struct codec_audio_lm * w)      { return w ? audio_lm_hidden_dim(w->ctx) : 0; }
bool     codec_audio_lm_is_continuous(const struct codec_audio_lm * w)   { return w && audio_lm_is_continuous(w->ctx); }

const char * codec_audio_lm_last_error(const struct codec_audio_lm * w) {
    if (w == nullptr) return "null codec_audio_lm";
    const char * e = audio_lm_last_error(w->ctx);
    return e != nullptr ? e : "";
}

const char * codec_audio_lm_meta_str(const struct codec_audio_lm * w, const char * key) {
    return w ? audio_lm_meta_str(w->ctx, key) : nullptr;
}

struct codec_lm * codec_audio_lm_get_lm(struct codec_audio_lm * w) { return w ? audio_lm_get_lm(w->ctx) : nullptr; }

bool codec_audio_lm_codec_has_speakers(const struct codec_audio_lm * w) {
    return w && audio_lm_codec_has_speakers(w->ctx);
}
bool codec_audio_lm_codec_can_speaker_from_audio(const struct codec_audio_lm * w) {
    return w && audio_lm_codec_can_speaker_from_audio(w->ctx);
}
bool codec_audio_lm_set_decode_speaker(struct codec_audio_lm * w, const char * name) {
    return w && audio_lm_set_decode_speaker(w->ctx, name);
}
bool codec_audio_lm_load_decode_speaker(struct codec_audio_lm * w, const char * path, const char * name) {
    return w && audio_lm_load_decode_speaker(w->ctx, path, name);
}
bool codec_audio_lm_decode_speaker_from_audio(struct codec_audio_lm * w, const float * pcm, int32_t n_samples,
                                              int32_t sample_rate, const char * ref_text) {
    return w && audio_lm_decode_speaker_from_audio(w->ctx, pcm, n_samples, sample_rate, ref_text);
}
bool codec_audio_lm_save_decode_speaker(struct codec_audio_lm * w, const char * path, const char * name) {
    return w && audio_lm_save_decode_speaker(w->ctx, path, name);
}
const struct codec_speaker * codec_audio_lm_decode_speaker(const struct codec_audio_lm * w) {
    return w ? audio_lm_decode_speaker(w->ctx) : nullptr;
}

int32_t codec_audio_lm_list_codec_speakers(struct codec_audio_lm * w) {
    if (w == nullptr) return -1;
    w->speakers = audio_lm_list_codec_speakers(w->ctx);
    return (int32_t) w->speakers.size();
}
int32_t codec_audio_lm_list_speaker_file(struct codec_audio_lm * w, const char * path) {
    if (w == nullptr) return -1;
    w->speakers.clear();
    if (!audio_lm_list_speaker_file(w->ctx, path, &w->speakers)) return -1;
    return (int32_t) w->speakers.size();
}
bool codec_audio_lm_listed_speaker(const struct codec_audio_lm * w, int32_t i,
                                   const char ** name, const char ** ref_text, bool * is_default) {
    if (w == nullptr || i < 0 || i >= (int32_t) w->speakers.size()) return false;
    const auto & s = w->speakers[(size_t) i];
    if (name)       *name       = s.name.c_str();
    if (ref_text)   *ref_text   = s.ref_text.c_str();
    if (is_default) *is_default = s.is_default;
    return true;
}

struct codec_audio_lm_input codec_audio_lm_default_input(void) {
    codec_audio_lm_input in{};
    return in;
}

bool codec_audio_lm_build_prompt(struct codec_audio_lm * w, const struct codec_audio_lm_input * in,
                                 struct codec_audio_lm_prompt * out) {
    if (w == nullptr || in == nullptr || out == nullptr) return false;
    audio_lm_input x;
    x.ref_pcm         = in->ref_pcm;
    x.ref_n_samples   = in->ref_n_samples;
    x.ref_sample_rate = in->ref_sample_rate;
    x.speaker_emb     = in->speaker_emb;
    x.speaker_emb_dim = in->speaker_emb_dim;
    x.emotion         = in->emotion;
    if (in->ref_text) x.ref_text = in->ref_text;
    if (in->text)     x.text     = in->text;
    for (int32_t i = 0; i < in->n_extra; ++i) {
        if (in->extra_keys[i] && in->extra_values[i]) x.extra[in->extra_keys[i]] = in->extra_values[i];
    }
    w->prompt = audio_lm_prompt();
    if (!audio_lm_build_prompt(w->ctx, x, &w->prompt)) return false;
    const audio_lm_prompt & p = w->prompt;
    out->tokens                     = p.tokens.empty() ? nullptr : p.tokens.data();
    out->n_tokens                   = (int32_t) p.tokens.size();
    out->embeds_prefix              = p.embeds_prefix.empty() ? nullptr : p.embeds_prefix.data();
    out->embeds_prefix_rows         = p.embeds_prefix_rows;
    out->embeds_prefix_hidden       = p.embeds_prefix_hidden;
    out->embeds_uncond              = p.embeds_uncond.empty() ? nullptr : p.embeds_uncond.data();
    out->default_temperature        = p.default_temperature;
    out->default_top_p              = p.default_top_p;
    out->default_min_p              = p.default_min_p;
    out->default_repetition_penalty = p.default_repetition_penalty;
    out->default_cfg_weight         = p.default_cfg_weight;
    out->start_token                = p.start_token;
    out->stop_token                 = p.stop_token;
    return true;
}

bool codec_audio_lm_get_prompt_info(struct codec_audio_lm * w, struct codec_audio_lm_prompt_info * out) {
    if (w == nullptr || out == nullptr) return false;
    w->info = audio_lm_prompt_info();
    if (!audio_lm_get_prompt_info(w->ctx, &w->info)) return false;
    const audio_lm_prompt_info & i = w->info;
    out->model_kind                 = (enum codec_audio_lm_kind) i.model_kind;
    out->host_arch                  = i.host_arch.c_str();
    out->prompt_prefix              = i.prompt_prefix.c_str();
    out->prompt_suffix              = i.prompt_suffix.c_str();
    out->add_bos                    = i.add_bos;
    out->parse_special              = i.parse_special;
    out->cb0_from_backbone          = i.cb0_from_backbone;
    out->audio_codebook_offset      = i.audio_codebook_offset;
    out->is_continuous              = i.is_continuous;
    out->n_codebook                 = i.n_codebook;
    out->hidden_dim                 = i.hidden_dim;
    out->eos_code_c0                = i.eos_code_c0;
    out->eos_min_step               = i.eos_min_step;
    out->default_temperature        = i.default_temperature;
    out->default_top_p              = i.default_top_p;
    out->default_top_k              = i.default_top_k;
    out->streaming_interleave       = i.streaming_interleave;
    out->text_externally_added      = i.text_externally_added;
    out->prefill_text_len           = i.prefill_text_len;
    out->text_pad_id                = i.text_pad_id;
    out->audio_pad_code             = i.audio_pad_code;
    out->bos_code_c0                = i.bos_code_c0;
    out->default_repetition_penalty = i.default_repetition_penalty;
    out->repetition_window          = i.repetition_window;
    out->sequential_text_audio      = i.sequential_text_audio;
    out->audio_start_id             = i.audio_start_id;
    out->text_end_id                = i.text_end_id;
    out->max_text_tokens            = i.max_text_tokens;
    out->cb0_speech_range_start     = i.cb0_speech_range_start;
    out->cb0_speech_range_end       = i.cb0_speech_range_end;
    out->audio_tok_offset           = i.audio_tok_offset;
    out->audio_tok_count            = i.audio_tok_count;
    out->audio_tok_eos              = i.audio_tok_eos;
    return true;
}

int32_t codec_audio_lm_outetts_build_grammar(enum codec_audio_lm_outetts_version version, const char * text,
                                             char * buf, int32_t buf_size) {
    if (version < CODEC_AUDIO_LM_OUTETTS_V1 || version > CODEC_AUDIO_LM_OUTETTS_V3) return -1;
    return copy_str(outetts_build_grammar((outetts_version) version, text ? text : ""), buf, buf_size);
}

int32_t codec_audio_lm_tts_auto_grammar(struct codec_audio_lm * w, const char * text, char * buf, int32_t buf_size) {
    if (w == nullptr) return -1;
    audio_lm_prompt_info pi;
    if (!audio_lm_get_prompt_info(w->ctx, &pi)) return -1;
    return copy_str(tts_auto_grammar(pi, text ? text : ""), buf, buf_size);
}

void codec_audio_lm_set_audio_token_range(struct codec_audio_lm * w, int32_t offset, int32_t count, int32_t eos_id) {
    if (w) audio_lm_set_audio_token_range(w->ctx, offset, count, eos_id);
}
void codec_audio_lm_set_audio_token_ranges(struct codec_audio_lm * w, const int32_t * offsets, const int32_t * counts,
                                           int32_t n_q, int32_t frame_sentinel_id, int32_t eos_id) {
    if (w) audio_lm_set_audio_token_ranges(w->ctx, offsets, counts, n_q, frame_sentinel_id, eos_id);
}
void codec_audio_lm_get_audio_token_range(const struct codec_audio_lm * w, int32_t * offset, int32_t * count,
                                          int32_t * eos_id) {
    int32_t o = -1, c = 0, e = -1;
    if (w) audio_lm_get_audio_token_range(w->ctx, &o, &c, &e);
    if (offset) *offset = o;
    if (count)  *count  = c;
    if (eos_id) *eos_id = e;
}

int32_t codec_audio_lm_codes_n_frames(const struct codec_audio_lm * w) { return w ? audio_lm_codes_n_frames(w->ctx) : 0; }

const int32_t * codec_audio_lm_codes(const struct codec_audio_lm * w, int32_t * n_q, int32_t * n_frames) {
    int32_t q = 0, t = 0;
    const int32_t * p = w ? audio_lm_debug_codes(w->ctx, &q, &t) : nullptr;
    if (n_q)      *n_q      = q;
    if (n_frames) *n_frames = t;
    return p;
}

void codec_audio_lm_get_lm_eos(const struct codec_audio_lm * w, int32_t * eos_code_c0, int32_t * eos_min_step) {
    int32_t c = -1, s = 0;
    if (w) audio_lm_get_lm_eos(w->ctx, &c, &s);
    if (eos_code_c0)  *eos_code_c0  = c;
    if (eos_min_step) *eos_min_step = s;
}

void codec_audio_lm_set_uses_embed_override(struct codec_audio_lm * w, bool enabled, int32_t start_step) {
    if (w) audio_lm_set_uses_embed_override(w->ctx, enabled, start_step);
}
bool codec_audio_lm_get_uses_embed_override(const struct codec_audio_lm * w) {
    return w && audio_lm_get_uses_embed_override(w->ctx);
}

enum codec_audio_lm_observe codec_audio_lm_observe_token(struct codec_audio_lm * w, int32_t token,
                                                         const float * last_hidden, int32_t hidden_dim) {
    if (w == nullptr) return CODEC_AUDIO_LM_OBSERVE_STOP;
    return (enum codec_audio_lm_observe) audio_lm_observe_token(w->ctx, token, last_hidden, hidden_dim);
}

const float * codec_audio_lm_get_next_embed(const struct codec_audio_lm * w, int32_t * dim) {
    if (w == nullptr) { if (dim) *dim = 0; return nullptr; }
    return audio_lm_get_next_embed(w->ctx, dim);
}

enum codec_audio_lm_observe codec_audio_lm_observe_hidden(struct codec_audio_lm * w, const float * hidden,
                                                          int32_t hidden_dim, const float * noise) {
    if (w == nullptr) return CODEC_AUDIO_LM_OBSERVE_STOP;
    return (enum codec_audio_lm_observe) audio_lm_observe_hidden(w->ctx, hidden, hidden_dim, noise);
}

void codec_audio_lm_set_continuous_params(struct codec_audio_lm * w, float cfg_value, int32_t n_timesteps,
                                          int32_t min_len) {
    if (w) audio_lm_set_continuous_params(w->ctx, cfg_value, n_timesteps, min_len);
}

bool codec_audio_lm_text_prefill(struct codec_audio_lm * w, const float * hiddens, int32_t n_pos, int32_t hidden_dim) {
    return w && audio_lm_text_prefill(w->ctx, hiddens, n_pos, hidden_dim);
}

enum codec_audio_lm_observe codec_audio_lm_observe_codes(struct codec_audio_lm * w, const int32_t * codes,
                                                         int32_t n_codes, const float * last_hidden,
                                                         int32_t hidden_dim) {
    if (w == nullptr) return CODEC_AUDIO_LM_OBSERVE_STOP;
    return (enum codec_audio_lm_observe) audio_lm_observe_codes(w->ctx, codes, n_codes, last_hidden, hidden_dim);
}

bool codec_audio_lm_push_codes(struct codec_audio_lm * w, const int32_t * codes, int32_t n_frames, int32_t n_q) {
    return w && audio_lm_push_codes(w->ctx, codes, n_frames, n_q);
}

bool codec_audio_lm_step_set_text_context(struct codec_audio_lm * w, int32_t text_token) {
    return w && audio_lm_step_set_text_context(w->ctx, text_token);
}
bool codec_audio_lm_step_begin(struct codec_audio_lm * w, const float * last_hidden, int32_t hidden_dim) {
    return w && audio_lm_step_begin(w->ctx, last_hidden, hidden_dim);
}
const float * codec_audio_lm_step_logits(struct codec_audio_lm * w, int32_t * cb_idx, int32_t * n) {
    return w ? audio_lm_step_logits(w->ctx, cb_idx, n) : nullptr;
}
bool codec_audio_lm_step_push_code(struct codec_audio_lm * w, int32_t code) {
    return w && audio_lm_step_push_code(w->ctx, code);
}
bool codec_audio_lm_step_finish(struct codec_audio_lm * w, int32_t * codes, int32_t n_codes) {
    return w && audio_lm_step_finish(w->ctx, codes, n_codes);
}

bool codec_audio_lm_prompt_needs_composed_embd(const struct codec_audio_lm * w) {
    return w && audio_lm_prompt_needs_composed_embd(w->ctx);
}
bool codec_audio_lm_compose_prompt_embd(struct codec_audio_lm * w, int32_t text_token, float * out, int32_t out_dim) {
    return w && audio_lm_compose_prompt_embd(w->ctx, text_token, out, out_dim);
}
bool codec_audio_lm_compose_audio_codes_embd(struct codec_audio_lm * w, const int32_t * codes, int32_t n_codes,
                                             float * out, int32_t out_dim) {
    return w && audio_lm_compose_audio_codes_embd(w->ctx, codes, n_codes, out, out_dim);
}

bool codec_audio_lm_talker_has_projection(const struct codec_audio_lm * w) {
    return w && audio_lm_talker_has_projection(w->ctx);
}
bool codec_audio_lm_build_talker_prefix(struct codec_audio_lm * w, const int32_t * role_tokens, int32_t n_role,
                                        const int32_t * text_tokens, int32_t n_text, const float * xvector,
                                        int32_t xvec_dim, float * out_embds, int32_t out_cap_rows,
                                        int32_t * out_n_rows, int32_t * out_text_consumed) {
    return w && audio_lm_build_talker_prefix(w->ctx, role_tokens, n_role, text_tokens, n_text, xvector, xvec_dim,
                                             out_embds, out_cap_rows, out_n_rows, out_text_consumed);
}
bool codec_audio_lm_talker_trailing_text_embd(struct codec_audio_lm * w, const int32_t * text_tokens, int32_t n_text,
                                              int32_t trailing_idx, float * out, int32_t out_dim) {
    return w && audio_lm_talker_trailing_text_embd(w->ctx, text_tokens, n_text, trailing_idx, out, out_dim);
}

bool codec_audio_lm_decode_audio(struct codec_audio_lm * w, struct codec_pcm_buffer * out) {
    if (w == nullptr) return false;
    audio_lm_audio_output pcm;
    return audio_lm_decode_audio(w->ctx, &pcm) && fill_pcm(pcm, out);
}

bool codec_audio_lm_decode_latents(struct codec_audio_lm * w, const float * latents, int32_t n_frames, int32_t dim,
                                   struct codec_pcm_buffer * out) {
    if (w == nullptr) return false;
    audio_lm_audio_output pcm;
    return audio_lm_decode_latents(w->ctx, latents, n_frames, dim, &pcm) && fill_pcm(pcm, out);
}

}  // extern "C"
