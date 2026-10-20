#ifndef CODEC_AUDIO_LM_FAMILY_H
#define CODEC_AUDIO_LM_FAMILY_H

// Internal: the model families behind include/codec_audio_lm.h.  Each family
// turns a request into a completion description (prompt chunks, sampler,
// constraints, host flags) and interprets every AR step; the public C API
// (audio_lm_c.cpp) only forwards.  Nothing here calls llama.cpp.

#include "codec_audio_lm.h"
#include "codec_common.h"

#include <memory>
#include <string>
#include <vector>

namespace codec_common {

// The host's backbone vocabulary (codec_audio_lm_vocab) with C++ helpers.
struct alm_vocab {
    const codec_audio_lm_vocab * v = nullptr;

    bool    valid() const { return v != nullptr && v->tokenize != nullptr && v->token_to_piece != nullptr; }
    int32_t n_tokens() const { return v ? v->n_tokens : 0; }
    int32_t eos() const { return v ? v->eos : -1; }
    bool    is_eog(int32_t t) const { return v && (v->is_eog ? v->is_eog(v->user_data, t) : t == v->eos); }
    std::vector<int32_t> tokenize(const std::string & text, bool add_special, bool parse_special) const;
    std::string          piece(int32_t token) const;
    // Id of the token whose piece is exactly `text`, or -1.  Linear scan.
    int32_t              find_piece(const std::string & text) const;
};

// A prompt chunk the family owns (codec_audio_lm_chunk points into it).
struct alm_chunk {
    codec_audio_lm_chunk_type type = CODEC_AUDIO_LM_CHUNK_TEXT;
    std::string          text;
    bool                 add_special   = false;
    bool                 parse_special = false;
    std::vector<int32_t> tokens;
    std::vector<float>   embd;   // rows x n_embd
    int32_t              n_rows = 0;

    static alm_chunk of_text(std::string t, bool add_special, bool parse_special);
    static alm_chunk of_tokens(std::vector<int32_t> t);
    static alm_chunk of_embd(std::vector<float> e, int32_t n_rows);
};

struct alm_completion {
    codec_audio_lm_flow flow = CODEC_AUDIO_LM_FLOW_TOKENS;
    int32_t n_embd = 0;
    std::vector<std::vector<alm_chunk>> seqs;   // one chunk list per sequence
    codec_audio_lm_sampler sampler{};
    std::vector<int32_t> penalty_exempt;
    std::string          samplers;   // order; "" = llama.cpp default
    std::string grammar;
    std::vector<codec_audio_lm_token_range> allowed;
    bool    needs_embeddings    = false;
    bool    needs_prompt_hidden = false;
    bool    sample_first        = true;
    bool    streamable          = false;
    int32_t max_frames          = 0;
    int32_t min_frames          = 0;
    int32_t stop_token          = -1;

    alm_completion();   // sampler "off" values
};

struct alm_tts_request {
    std::string text;
    std::string instruct;
    std::string preset;
};

struct alm_tts_voice {
    std::string        voice;
    std::string        speaker_file;
    std::vector<float> ref_pcm;
    int32_t            ref_sample_rate = 0;
    std::string        ref_text;
    std::string        save_speaker;
};

// One model family.  Lifetime: owned by the codec_audio_lm context; `alm`
// (codec + codec_lm) outlives it.
class alm_family {
public:
    explicit alm_family(audio_lm_context * alm) : alm_(alm) {}
    virtual ~alm_family() = default;

    virtual const char * name() const = 0;
    virtual void caps(codec_audio_lm_caps * out) const;
    virtual bool backbone_requirements(const alm_vocab & vocab, int32_t max_frames,
                                       codec_audio_lm_backbone_req * out);
    // Default: the decoder speaker (speaker file > ref audio > voice > default).
    virtual bool set_voice(const alm_tts_voice & v);
    // Segmentation (codec_audio_lm_tts_split) and joining (tts_join).
    // Default: tts_split_text at max_bytes (0 = whole text); no edge handling.
    virtual std::vector<std::string> split(const std::string & text, int32_t max_bytes);
    virtual int32_t join(std::vector<float> & pcm, int32_t sample_rate, int32_t i, int32_t n);
    virtual bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) = 0;
    virtual bool prefill(const float * hiddens, int32_t n_pos);
    virtual bool observe(const int32_t * tokens, const float * hiddens, codec_audio_lm_verdict * v) = 0;
    virtual const float * next_embd(int32_t seq) const;
    // Set by observe() on STOP; "max_frames" until then.
    const std::string & stop_reason() const { return stop_reason_; }
    virtual bool generate_self_contained();
    // Default: audio_lm_decode_audio over the accumulated codes / latents.
    virtual bool decode(audio_lm_audio_output * out);

    const std::string & error() const { return error_; }

protected:
    bool fail(const std::string & e) { error_ = e; return false; }
    bool fail_alm(const char * what);   // e + audio_lm_last_error

    audio_lm_context * alm_;
    std::string        error_;
    alm_tts_voice      voice_;
    std::string        stop_reason_ = "max_frames";
};

// Wrap an audio_lm context the caller keeps owning (tts_runner while its
// families move behind the public API).  nullptr when no family fits.
struct codec_audio_lm * alm_wrap_borrowed(audio_lm_context * alm, std::string * err);

// The family for a loaded codec GGUF, or nullptr (err set) when the codec is
// not the audio side of a supported model.
std::unique_ptr<alm_family> alm_make_family(audio_lm_context * alm, std::string * err);

// Families (audio_lm_family_*.cpp).
std::unique_ptr<alm_family> alm_family_soprano(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_neutts(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_outetts(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_kitten(audio_lm_context * alm);
bool alm_is_kitten(audio_lm_context * alm);

}  // namespace codec_common

#endif  // CODEC_AUDIO_LM_FAMILY_H
