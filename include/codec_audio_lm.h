#ifndef CODEC_AUDIO_LM_H
#define CODEC_AUDIO_LM_H

#include "codec.h"
#include "codec_lm.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// codec_audio_lm — C API over codec.cpp's audio-LM host layer
// (common/codec_common.h, namespace codec_common).  Same functions, same
// semantics; see that header and docs/codec_common_api.md.  C++ hosts can
// use either; this one is for C and FFI callers (python/codec_cpp).
//
// Conventions:
//   * One `codec_audio_lm` wraps one codec_common::audio_lm_context.
//   * Strings and arrays the library returns (prompt, prompt info, speaker
//     lists, last error) are owned by the context and stay valid until the
//     next call that refills them (or codec_audio_lm_free).
//   * PCM comes back in a `codec_pcm_buffer`; free it with
//     codec_pcm_buffer_free.
// =====================================================================

struct codec_audio_lm;

// Bitmask of codec_audio_lm_modality().
enum codec_audio_lm_modality_flag {
    CODEC_AUDIO_LM_INPUT_TEXT   = 1u << 0,
    CODEC_AUDIO_LM_INPUT_AUDIO  = 1u << 1,
    CODEC_AUDIO_LM_OUTPUT_TEXT  = 1u << 2,
    CODEC_AUDIO_LM_OUTPUT_AUDIO = 1u << 3,
};

// Verdict of the observe_* calls: how to feed the next backbone step.
enum codec_audio_lm_observe {
    CODEC_AUDIO_LM_OBSERVE_PASSTHROUGH    = 0,  // text token: token batch
    CODEC_AUDIO_LM_OBSERVE_CONSUMED       = 1,  // audio token: token batch
    CODEC_AUDIO_LM_OBSERVE_CONSUMED_EMBED = 2,  // feed codec_audio_lm_get_next_embed
    CODEC_AUDIO_LM_OBSERVE_STOP           = 3,  // end of audio
};

enum codec_audio_lm_kind {
    CODEC_AUDIO_LM_KIND_UNKNOWN             = 0,
    CODEC_AUDIO_LM_KIND_RESIDUAL_DEPTH_AR   = 1,  // CSM, Qwen3-TTS, MOSS-TTS-Realtime
    CODEC_AUDIO_LM_KIND_PARALLEL_HEADS_DELAY = 2, // MOSS-TTSD
    CODEC_AUDIO_LM_KIND_CONTINUOUS_CFM      = 3,  // BlueMagpie
    CODEC_AUDIO_LM_KIND_TOKEN_SINGLE_CB     = 4,  // OuteTTS-style token LMs
};

// ── Lifecycle ────────────────────────────────────────────────────────

struct codec_audio_lm_params {
    const char * codec_path;                              // codec / codec_lm GGUF
    bool         use_gpu;
    int32_t      n_threads;                               // 0 = codec.cpp default
    const struct codec_model_kv_override * kv_overrides;  // may be NULL
    int32_t      n_kv_overrides;                          // no terminator needed
};

struct codec_audio_lm_params codec_audio_lm_default_params(void);

// NULL on failure; the reason goes to `err` (may be NULL) as a C string.
struct codec_audio_lm * codec_audio_lm_init(const struct codec_audio_lm_params * params,
                                            char * err, size_t err_size);
void codec_audio_lm_free (struct codec_audio_lm * ctx);
void codec_audio_lm_reset(struct codec_audio_lm * ctx);

// A context with no model: only the code accumulator and the token-range
// state work (tests).  Free with codec_audio_lm_free.
struct codec_audio_lm * codec_audio_lm_alloc_bare(void);

// "KEY=TYPE:VALUE" (TYPE = int | float | bool | str), as llama.cpp's --override-kv.
bool codec_audio_lm_parse_kv_override(const char * spec, struct codec_model_kv_override * out);

// ── Capabilities ─────────────────────────────────────────────────────

uint32_t     codec_audio_lm_modality       (const struct codec_audio_lm * ctx);
bool         codec_audio_lm_has_speaker_enc(const struct codec_audio_lm * ctx);
int32_t      codec_audio_lm_n_codebook     (const struct codec_audio_lm * ctx);
int32_t      codec_audio_lm_hidden_dim     (const struct codec_audio_lm * ctx);
bool         codec_audio_lm_is_continuous  (const struct codec_audio_lm * ctx);
const char * codec_audio_lm_last_error     (const struct codec_audio_lm * ctx);
// Raw codec GGUF metadata value (stringified), or NULL.
const char * codec_audio_lm_meta_str       (const struct codec_audio_lm * ctx, const char * key);
// Borrowed codec_lm handle (NULL when the GGUF has none): codec_lm.h helpers.
struct codec_lm * codec_audio_lm_get_lm    (struct codec_audio_lm * ctx);

// ── Decoder speaker ──────────────────────────────────────────────────

bool codec_audio_lm_codec_has_speakers          (const struct codec_audio_lm * ctx);
bool codec_audio_lm_codec_can_speaker_from_audio(const struct codec_audio_lm * ctx);
// name NULL = the model's default speaker.
bool codec_audio_lm_set_decode_speaker          (struct codec_audio_lm * ctx, const char * name);
// name NULL = the file's default speaker.
bool codec_audio_lm_load_decode_speaker         (struct codec_audio_lm * ctx, const char * path, const char * name);
// ref_text (may be NULL) is the clip's transcript, kept on the speaker.
bool codec_audio_lm_decode_speaker_from_audio   (struct codec_audio_lm * ctx, const float * pcm,
                                                 int32_t n_samples, int32_t sample_rate, const char * ref_text);
bool codec_audio_lm_save_decode_speaker         (struct codec_audio_lm * ctx, const char * path, const char * name);
// The speaker decode uses (NULL when the codec has no speakers).
const struct codec_speaker * codec_audio_lm_decode_speaker(const struct codec_audio_lm * ctx);

// Speaker listings fill a per-context list read with codec_audio_lm_listed_speaker.
// Return the number of entries, or -1 on error.
int32_t codec_audio_lm_list_codec_speakers(struct codec_audio_lm * ctx);
int32_t codec_audio_lm_list_speaker_file  (struct codec_audio_lm * ctx, const char * path);
// Entry i of the last listing.  Any out-pointer may be NULL.
bool    codec_audio_lm_listed_speaker     (const struct codec_audio_lm * ctx, int32_t i,
                                           const char ** name, const char ** ref_text, bool * is_default);

// ── Prompt build ─────────────────────────────────────────────────────

struct codec_audio_lm_input {
    const float * ref_pcm;          // reference audio (may be NULL)
    int32_t       ref_n_samples;
    int32_t       ref_sample_rate;
    const float * speaker_emb;      // precomputed speaker embedding (may be NULL)
    int32_t       speaker_emb_dim;
    const float * emotion;          // [0, 1]; NULL = model default
    const char *  ref_text;         // transcript of ref_pcm (may be NULL)
    const char *  text;             // text to synthesize (may be NULL)
    const char * const * extra_keys;   // model-specific knobs (may be NULL)
    const char * const * extra_values;
    int32_t       n_extra;
};

struct codec_audio_lm_input codec_audio_lm_default_input(void);

// Owned by the context; valid until the next build_prompt / free.
struct codec_audio_lm_prompt {
    const int32_t * tokens;
    int32_t         n_tokens;
    const float *   embeds_prefix;         // rows x hidden, row-major (may be NULL)
    int32_t         embeds_prefix_rows;
    int32_t         embeds_prefix_hidden;
    const float *   embeds_uncond;         // CFG branch, same shape (may be NULL)
    float           default_temperature;
    float           default_top_p;
    float           default_min_p;
    float           default_repetition_penalty;
    float           default_cfg_weight;    // 0 = no CFG
    int32_t         start_token;           // -1 = none
    int32_t         stop_token;            // -1 = vocab EOS
};

bool codec_audio_lm_build_prompt(struct codec_audio_lm * ctx, const struct codec_audio_lm_input * in,
                                 struct codec_audio_lm_prompt * out);

// Per-model prompt descriptor (codec_common::audio_lm_prompt_info).
// Strings are owned by the context; valid until the next call / free.
struct codec_audio_lm_prompt_info {
    enum codec_audio_lm_kind model_kind;
    const char * host_arch;
    const char * prompt_prefix;
    const char * prompt_suffix;
    bool    add_bos;
    bool    parse_special;
    bool    cb0_from_backbone;
    int32_t audio_codebook_offset;
    bool    is_continuous;
    int32_t n_codebook;
    int32_t hidden_dim;
    int32_t eos_code_c0;
    int32_t eos_min_step;
    float   default_temperature;
    float   default_top_p;
    int32_t default_top_k;
    // streaming text<->audio interleave (MOSS-TTS-Realtime)
    bool    streaming_interleave;
    bool    text_externally_added;
    int32_t prefill_text_len;
    int32_t text_pad_id;
    int32_t audio_pad_code;
    int32_t bos_code_c0;
    float   default_repetition_penalty;
    int32_t repetition_window;
    // sequential text -> audio (LFM2-Audio)
    bool    sequential_text_audio;
    int32_t audio_start_id;
    int32_t text_end_id;
    int32_t max_text_tokens;
    // merged-cb0 speech sub-range (MOSS-TTSD)
    int32_t cb0_speech_range_start;
    int32_t cb0_speech_range_end;
    // token-based single-codebook audio range
    int32_t audio_tok_offset;
    int32_t audio_tok_count;
    int32_t audio_tok_eos;
};

bool codec_audio_lm_get_prompt_info(struct codec_audio_lm * ctx, struct codec_audio_lm_prompt_info * out);

// GBNF for the backbone sampler that this model's metadata derives for
// `text` ("" when the model has none), written snprintf-style: returns the
// full length (excluding the NUL); `buf` may be NULL / too small (then it is
// truncated).  -1 on error.
int32_t codec_audio_lm_grammar(struct codec_audio_lm * ctx, const char * text, char * buf, int32_t buf_size);

// ── Token-range config (token-based models) ──────────────────────────

void codec_audio_lm_set_audio_token_range (struct codec_audio_lm * ctx, int32_t offset, int32_t count, int32_t eos_id);
void codec_audio_lm_set_audio_token_ranges(struct codec_audio_lm * ctx, const int32_t * offsets, const int32_t * counts,
                                           int32_t n_q, int32_t frame_sentinel_id, int32_t eos_id);
void codec_audio_lm_get_audio_token_range (const struct codec_audio_lm * ctx, int32_t * offset, int32_t * count, int32_t * eos_id);

int32_t         codec_audio_lm_codes_n_frames(const struct codec_audio_lm * ctx);
// Accumulated codes, (n_frames, n_q) frame-major; NULL when none.
const int32_t * codec_audio_lm_codes         (const struct codec_audio_lm * ctx, int32_t * n_q, int32_t * n_frames);
void            codec_audio_lm_get_lm_eos    (const struct codec_audio_lm * ctx, int32_t * eos_code_c0, int32_t * eos_min_step);

void codec_audio_lm_set_uses_embed_override(struct codec_audio_lm * ctx, bool enabled, int32_t start_step);
bool codec_audio_lm_get_uses_embed_override(const struct codec_audio_lm * ctx);

// ── Per-step hooks ───────────────────────────────────────────────────

enum codec_audio_lm_observe codec_audio_lm_observe_token(struct codec_audio_lm * ctx, int32_t token,
                                                         const float * last_hidden, int32_t hidden_dim);
// Valid right after OBSERVE_CONSUMED_EMBED, until the next observe / reset.
const float * codec_audio_lm_get_next_embed(const struct codec_audio_lm * ctx, int32_t * dim);

// Continuous-latent models.  noise: [patch_size * latent_dim] or NULL to sample.
enum codec_audio_lm_observe codec_audio_lm_observe_hidden(struct codec_audio_lm * ctx, const float * hidden,
                                                          int32_t hidden_dim, const float * noise);
// min_len < 0 keeps the model default.
void codec_audio_lm_set_continuous_params(struct codec_audio_lm * ctx, float cfg_value, int32_t n_timesteps, int32_t min_len);
bool codec_audio_lm_text_prefill(struct codec_audio_lm * ctx, const float * hiddens, int32_t n_pos, int32_t hidden_dim);

enum codec_audio_lm_observe codec_audio_lm_observe_codes(struct codec_audio_lm * ctx, const int32_t * codes,
                                                         int32_t n_codes, const float * last_hidden, int32_t hidden_dim);
bool codec_audio_lm_push_codes(struct codec_audio_lm * ctx, const int32_t * codes, int32_t n_frames, int32_t n_q);

// Codebook step machine (residual depth-AR / parallel heads).
bool          codec_audio_lm_step_set_text_context(struct codec_audio_lm * ctx, int32_t text_token);
bool          codec_audio_lm_step_begin    (struct codec_audio_lm * ctx, const float * last_hidden, int32_t hidden_dim);
const float * codec_audio_lm_step_logits   (struct codec_audio_lm * ctx, int32_t * cb_idx, int32_t * n);
bool          codec_audio_lm_step_push_code(struct codec_audio_lm * ctx, int32_t code);
bool          codec_audio_lm_step_finish   (struct codec_audio_lm * ctx, int32_t * codes, int32_t n_codes);

// ── Prompt / step embeddings ─────────────────────────────────────────

bool codec_audio_lm_prompt_needs_composed_embd(const struct codec_audio_lm * ctx);
bool codec_audio_lm_compose_prompt_embd     (struct codec_audio_lm * ctx, int32_t text_token, float * out, int32_t out_dim);
bool codec_audio_lm_compose_audio_codes_embd(struct codec_audio_lm * ctx, const int32_t * codes, int32_t n_codes,
                                             float * out, int32_t out_dim);

// Text-lane models: the codec side projects the text tokens itself and the
// backbone input is the sum of a text lane and an audio / control lane.  The
// prompt is a prefix of embedding rows built from the header tokens, the
// text tokens and an optional speaker embedding (codec_audio_lm_build_prompt
// speaker_emb); the text tokens the prefix did not consume are fed one per
// AR step, added to codec_audio_lm_get_next_embed.
bool codec_audio_lm_has_text_lane(const struct codec_audio_lm * ctx);
// Writes out_n_rows rows (row-major, hidden floats each) into out_embds
// (capacity out_cap_rows) and how many text tokens they consumed.
bool codec_audio_lm_build_prefix_embd(struct codec_audio_lm * ctx, const int32_t * header_tokens, int32_t n_header,
                                      const int32_t * text_tokens, int32_t n_text, const float * speaker_emb,
                                      int32_t speaker_emb_dim, float * out_embds, int32_t out_cap_rows,
                                      int32_t * out_n_rows, int32_t * out_text_consumed);
// Text-lane row for AR step `step` (0 = the first step after the prefix);
// past the end of the text it is the end-of-text row.
bool codec_audio_lm_step_text_embd(struct codec_audio_lm * ctx, const int32_t * text_tokens, int32_t n_text,
                                   int32_t step, float * out, int32_t out_dim);

// ── End of sequence ──────────────────────────────────────────────────

// Accumulated codes / latents -> PCM.  Free `out` with codec_pcm_buffer_free.
bool codec_audio_lm_decode_audio  (struct codec_audio_lm * ctx, struct codec_pcm_buffer * out);
// Host-collected latents in the codec's own layout (Soprano: [n_frames, dim]).
bool codec_audio_lm_decode_latents(struct codec_audio_lm * ctx, const float * latents, int32_t n_frames,
                                   int32_t dim, struct codec_pcm_buffer * out);

#ifdef __cplusplus
}
#endif

#endif  // CODEC_AUDIO_LM_H
