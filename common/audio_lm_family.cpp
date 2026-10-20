#include "audio_lm_family.h"
#include "tts_runner.h"

#include <cstring>

namespace codec_common {

// ── alm_vocab ────────────────────────────────────────────────────────

std::vector<int32_t> alm_vocab::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    if (!valid()) return {};
    std::vector<int32_t> out(text.size() + 16);
    int32_t n = v->tokenize(v->user_data, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                            add_special, parse_special);
    if (n < 0) {
        out.resize((size_t) -n);
        n = v->tokenize(v->user_data, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                        add_special, parse_special);
    }
    out.resize((size_t) std::max(0, n));
    return out;
}

std::string alm_vocab::piece(int32_t token) const {
    if (!valid()) return {};
    char buf[128];
    int32_t n = v->token_to_piece(v->user_data, token, buf, (int32_t) sizeof(buf));
    if (n >= 0) return std::string(buf, (size_t) std::min<int32_t>(n, (int32_t) sizeof(buf)));
    std::string big((size_t) -n, '\0');
    n = v->token_to_piece(v->user_data, token, big.data(), (int32_t) big.size());
    big.resize((size_t) std::max(0, n));
    return big;
}

int32_t alm_vocab::find_piece(const std::string & text) const {
    for (int32_t id = 0; id < n_tokens(); ++id) {
        if (piece(id) == text) return id;
    }
    return -1;
}

// ── chunks / completion ─────────────────────────────────────────────

alm_chunk alm_chunk::of_text(std::string t, bool add_special, bool parse_special) {
    alm_chunk c;
    c.type = CODEC_AUDIO_LM_CHUNK_TEXT;
    c.text = std::move(t);
    c.add_special = add_special;
    c.parse_special = parse_special;
    return c;
}

alm_chunk alm_chunk::of_tokens(std::vector<int32_t> t) {
    alm_chunk c;
    c.type = CODEC_AUDIO_LM_CHUNK_TOKENS;
    c.tokens = std::move(t);
    return c;
}

alm_chunk alm_chunk::of_embd(std::vector<float> e, int32_t n_rows) {
    alm_chunk c;
    c.type = CODEC_AUDIO_LM_CHUNK_EMBD;
    c.embd = std::move(e);
    c.n_rows = n_rows;
    return c;
}

alm_completion::alm_completion() {
    sampler.temp = 0.0f;
    sampler.top_k = 0;
    sampler.top_p = 1.0f;
    sampler.min_p = 0.0f;
    sampler.penalty_last_n = 0;
    sampler.penalty_repeat = 1.0f;
    sampler.penalty_freq = 0.0f;
    sampler.penalty_present = 0.0f;
    sampler.penalize_prompt = false;
    sampler.dry_multiplier = 0.0f;
    sampler.dry_base = 1.75f;
    sampler.dry_allowed_length = 2;
    sampler.dry_penalty_last_n = -1;
}

// ── alm_family defaults ─────────────────────────────────────────────

bool alm_family::fail_alm(const char * what) {
    const char * e = audio_lm_last_error(alm_);
    return fail(std::string(what) + (e && *e ? std::string(": ") + e : std::string()));
}

void alm_family::caps(codec_audio_lm_caps * out) const {
    *out = codec_audio_lm_caps{};
    out->modality       = audio_lm_modality(alm_);
    out->needs_backbone = true;
    out->takes_ref_audio = audio_lm_codec_can_speaker_from_audio(alm_);
}

bool alm_family::backbone_requirements(const alm_vocab &, int32_t max_frames, codec_audio_lm_backbone_req * out) {
    out->n_ctx      = std::max(4096, (max_frames > 0 ? max_frames : 512) + 512);
    out->n_ubatch   = 0;
    out->n_seq_max  = 1;
    out->embeddings = true;
    out->flash_attn = true;
    return true;
}

bool alm_family::set_voice(const alm_tts_voice & v) {
    voice_ = v;
    audio_lm_set_decode_speaker(alm_, nullptr);
    if (!v.speaker_file.empty()) {
        if (!audio_lm_load_decode_speaker(alm_, v.speaker_file.c_str(), v.voice.empty() ? nullptr : v.voice.c_str())) {
            return fail_alm("speaker file");
        }
        codec_log(CODEC_LOG_LEVEL_INFO, "speaker: loaded %s%s%s\n", v.speaker_file.c_str(),
                  v.voice.empty() ? "" : " : ", v.voice.c_str());
    } else if (!v.ref_pcm.empty() && audio_lm_codec_can_speaker_from_audio(alm_)) {
        if (!audio_lm_decode_speaker_from_audio(alm_, v.ref_pcm.data(), (int32_t) v.ref_pcm.size(), v.ref_sample_rate,
                                                v.ref_text.empty() ? nullptr : v.ref_text.c_str())) {
            return fail_alm("speaker from reference audio");
        }
        codec_log(CODEC_LOG_LEVEL_INFO, "speaker: derived from reference audio (%zu samples @ %d Hz)\n",
                  v.ref_pcm.size(), v.ref_sample_rate);
    } else if (!v.voice.empty()) {
        if (!audio_lm_set_decode_speaker(alm_, v.voice.c_str())) return fail_alm("voice");
        codec_log(CODEC_LOG_LEVEL_INFO, "speaker: voice %s\n", v.voice.c_str());
    }
    if (v.speaker_file.empty() && !v.ref_pcm.empty() &&
        audio_lm_codec_has_speakers(alm_) && !audio_lm_codec_can_speaker_from_audio(alm_)) {
        codec_log(CODEC_LOG_LEVEL_WARN,
                  "speaker: this codec GGUF cannot derive a speaker from reference audio; keeping the %s voice\n",
                  v.voice.empty() ? "default" : v.voice.c_str());
    }
    if (!v.save_speaker.empty()) {
        if (!audio_lm_save_decode_speaker(alm_, v.save_speaker.c_str(), "saved")) return fail_alm("save speaker");
        codec_log(CODEC_LOG_LEVEL_INFO, "speaker: saved to %s\n", v.save_speaker.c_str());
    }
    return true;
}

std::vector<std::string> alm_family::split(const std::string & text, int32_t max_bytes) {
    return max_bytes > 0 ? tts_split_text(text, max_bytes) : std::vector<std::string>{text};
}

int32_t alm_family::join(std::vector<float> &, int32_t, int32_t, int32_t) { return 0; }

bool alm_family::prefill(const float *, int32_t) { return fail("this model takes no prompt hidden states"); }

const float * alm_family::next_embd(int32_t) const { return nullptr; }

bool alm_family::generate_self_contained() { return fail("this model needs a backbone"); }

bool alm_family::decode(audio_lm_audio_output * out) {
    return audio_lm_decode_audio(alm_, out) || fail_alm("decode");
}

// ── detection ────────────────────────────────────────────────────────

std::unique_ptr<alm_family> alm_make_family(audio_lm_context * alm, std::string * err) {
    const char * arch = audio_lm_meta_str(alm, "general.architecture");
    const std::string a = arch ? arch : "";
    if (a == "soprano") return alm_family_soprano(alm);
    if (alm_is_kitten(alm)) return alm_family_kitten(alm);
    if (audio_lm_get_lm(alm) == nullptr) {
        // Codec-only GGUFs whose model writes codec tokens with its own vocab.
        if (a == "neucodec" || a == "distill_neucodec") return alm_family_neutts(alm);
        const codec_speaker * spk = audio_lm_decode_speaker(alm);
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        if (spk != nullptr && codec_speaker_get_slot(spk, "audio_prompt", &ty, ne, &data)) return alm_family_outetts(alm);
    }
    if (err) *err = std::string("no audio-LM recipe for this codec GGUF (architecture ") + (arch ? arch : "?") + ")";
    return nullptr;
}

}  // namespace codec_common
