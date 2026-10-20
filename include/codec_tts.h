#ifndef CODEC_TTS_H
#define CODEC_TTS_H

#include "codec.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// codec_tts — text to speech in one call: the reference host loop of
// tts-cli / tts-server (common/tts_runner.h) behind a C ABI.  A session
// loads a codec GGUF and (for every model but Pocket-TTS) a llama.cpp
// backbone GGUF once, then synthesizes many requests.  The backbone runs in
// the llama.cpp bundled with codec.cpp (with its architecture / tokenizer
// patches), so this lives in its own library, libcodec_tts, next to
// libcodec.  Runs on one session are not thread-safe; serialize them.
// =====================================================================

struct codec_tts;

struct codec_tts_params {
    const char * codec_path;                               // codec / codec_lm GGUF (required)
    const char * backbone_path;                            // llama.cpp GGUF; NULL for Pocket-TTS
    const struct codec_model_kv_override * kv_overrides;   // codec metadata overrides (may be NULL)
    int32_t      n_kv_overrides;
    int32_t      n_threads;                                // 0 = default
    bool         use_gpu;
    int32_t      max_frames;                               // 0 = model default
    float        cfg_weight;                               // Chatterbox T3 CFG; < 0 = model default
};

// Per-request options.  NULL strings and "unset" sentinels keep the model's
// defaults (from its GGUF metadata).
struct codec_tts_request {
    const char * text;            // required
    const char * voice;           // builtin speaker name (codec_tts_voices)
    const char * speaker_file;    // speaker GGUF (codec_speaker_save / save_speaker)
    const char * ref_audio_path;  // WAV to clone the voice from
    const char * ref_text;        // transcript of ref_audio
    const char * save_speaker;    // write the decoder speaker in use here
    const char * instruct;        // CosyVoice instruct text
    const char * preset;          // sampling preset (KittenTTS)
    const char * grammar;         // GBNF for the backbone sampler (NULL = model auto-grammar)
    uint32_t     seed;
    float        temperature;          // < 0 = unset
    float        top_p;                // < 0 = unset
    int32_t      top_k;                // < 0 = unset
    float        min_p;                // < 0 = unset
    float        repetition_penalty;   // < 0 = unset
    float        cfg;                  // continuous CFM / FlowLM CFG; < 0 = unset
    int32_t      timesteps;            // continuous CFM / FlowLM steps; < 0 = unset
    int32_t      min_len;              // < 0 = model default
};

struct codec_tts_result {
    int32_t n_frames;             // audio frames generated
    char    stop_reason[32];      // "eos", "eos_code_c0", "max_frames", ...
    double  total_ms;
    double  decode_ms;
};

struct codec_tts_params  codec_tts_default_params(void);
struct codec_tts_request codec_tts_default_request(void);

// NULL on failure; the reason goes to `err` (may be NULL).
struct codec_tts * codec_tts_load(const struct codec_tts_params * params, char * err, size_t err_size);
void               codec_tts_free(struct codec_tts * tts);

// Synthesize one request into `out_pcm` (free with codec_pcm_buffer_free);
// `out_result` may be NULL.  On failure returns false; codec_tts_last_error
// says why.
bool codec_tts_run(struct codec_tts * tts, const struct codec_tts_request * req,
                   struct codec_pcm_buffer * out_pcm, struct codec_tts_result * out_result);
const char * codec_tts_last_error(const struct codec_tts * tts);

// Voices `voice` can name: the codec's builtin speakers, or the speakers in
// `speaker_file` (may be NULL).  Fills a per-session list read with
// codec_tts_voice; returns its size, or -1.
int32_t      codec_tts_voices       (struct codec_tts * tts, const char * speaker_file);
const char * codec_tts_voice        (const struct codec_tts * tts, int32_t i);
const char * codec_tts_default_voice(const struct codec_tts * tts);
// CosyVoice instruct models take `instruct`.
bool         codec_tts_takes_instruct(const struct codec_tts * tts);

#ifdef __cplusplus
}
#endif

#endif  // CODEC_TTS_H
