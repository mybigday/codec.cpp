// Pocket-TTS FlowLM: a self-contained continuous-latent AR model whose
// transformer, text LUT, LSD flow head and EOS head all live in the codec
// GGUF (codec.lm.kind = "flow_lm"), so there is no backbone: the completion
// is SELF_CONTAINED and codec_audio_lm_generate_self_contained runs it.
// Reference: pocket_tts/models/tts_model.py.

#include "audio_lm_family.h"
#include "lm/lm_models.h"

#include <cmath>
#include <cstring>
#include <random>

namespace codec_common {
namespace {

// prepare_text_prompt: strip, collapse double spaces, uppercase the first
// letter, ensure trailing punctuation.  Returns the guessed frames_after_eos
// (3 if <= 4 words else 1).
int flow_prepare_text(std::string & text) {
    const size_t b = text.find_first_not_of(" \t\r\n");
    const size_t e = text.find_last_not_of(" \t\r\n");
    text = (b == std::string::npos) ? "" : text.substr(b, e - b + 1);
    for (char & c : text) if (c == '\n' || c == '\r') c = ' ';
    for (size_t i = text.find("  "); i != std::string::npos; i = text.find("  ")) text.replace(i, 2, " ");
    if (text.empty()) return 3;
    int words = 0;
    bool in_w = false;
    for (char c : text) { if (c == ' ') in_w = false; else if (!in_w) { in_w = true; ++words; } }
    const int frames_after_eos_guess = (words <= 4) ? 3 : 1;
    if (text[0] >= 'a' && text[0] <= 'z') text[0] = (char) (text[0] - 'a' + 'A');
    const char last = text.back();
    const bool alnum = (last >= 'a' && last <= 'z') || (last >= 'A' && last <= 'Z') || (last >= '0' && last <= '9');
    if (alnum) text += '.';
    return frames_after_eos_guess;
}

class flow_family : public alm_family {
public:
    flow_family(audio_lm_context * alm, codec_lm * lm, const codec_lm_flow_info * fi) : alm_family(alm), lm_(lm), fi_(fi) {}
    ~flow_family() override { if (st_) codec_lm_state_free(st_); }
    const char * name() const override { return "flow"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->needs_backbone  = false;
        out->takes_ref_audio = true;
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : 24000;
    }

    // The voice conditions the LM (reference latents), not a decoder speaker.
    bool set_voice(const alm_tts_voice & v) override {
        voice_ = v;
        return true;
    }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab &, alm_completion * out) override {
        text_ = req.text;
        fae_guess_ = flow_prepare_text(text_) + 2;   // the reference adds 2
        seed_ = req.seed_or_default();
        min_len_ = req.min_frames >= 0 ? req.min_frames : 0;
        max_frames_ = req.max_frames;
        stop_reason_ = "max_frames";
        latents_.clear();
        n_frames_ = 0;
        *out = alm_completion();
        out->flow = CODEC_AUDIO_LM_FLOW_SELF_CONTAINED;
        out->max_frames = 0;   // estimated from the text in generate_self_contained
        return true;
    }

    int32_t decodable_frames() const override { return n_frames_; }

    bool observe(const int32_t *, const float *, codec_audio_lm_verdict *) override {
        return fail("this model runs self-contained");
    }

    bool generate_self_contained() override {
        const int32_t ldim = fi_->ldim, d_model = fi_->d_model;
        std::vector<int32_t> ids(512);
        int32_t n_tok = 0;
        if (codec_lm_flow_tokenize(lm_, text_.c_str(), ids.data(), (int32_t) ids.size(), &n_tok) != CODEC_STATUS_SUCCESS ||
            n_tok <= 0) {
            return fail(std::string("tokenize: ") + codec_lm_get_last_error(lm_));
        }
        ids.resize((size_t) n_tok);
        codec_log(CODEC_LOG_LEVEL_INFO, "flow_lm: text=\"%s\" -> %d tokens; d_model=%d ldim=%d\n", text_.c_str(), n_tok, d_model, ldim);

        // Voice conditioning: the reference's latents through the speaker projection.
        std::vector<float> voice_rows;
        int32_t n_voice = 0;
        if (!voice_.ref_pcm.empty()) {
            codec_audio au;
            au.data = voice_.ref_pcm.data();
            au.n_samples = (int32_t) voice_.ref_pcm.size();
            au.sample_rate = voice_.ref_sample_rate;
            au.n_channels = 1;
            au.pcm_type = CODEC_PCM_TYPE_F32;
            codec_token_buffer tb;
            std::memset(&tb, 0, sizeof(tb));
            codec_latent_buffer lb;
            std::memset(&lb, 0, sizeof(lb));
            codec_context * cctx = audio_lm_get_codec_context(alm_);
            if (codec_encode_latent(cctx, &au, &tb, &lb, codec_encode_default_params()) != CODEC_STATUS_SUCCESS) {
                return fail(std::string("reference audio encode: ") + codec_get_last_error(cctx));
            }
            n_voice = lb.n_frames;
            voice_rows.resize((size_t) n_voice * d_model);
            const bool ok = codec_lm_flow_speaker_rows(lm_, lb.data, n_voice, voice_rows.data(), n_voice) == CODEC_STATUS_SUCCESS;
            codec_latent_buffer_free(&lb);
            if (!ok) return fail(std::string("speaker rows: ") + codec_lm_get_last_error(lm_));
            codec_log(CODEC_LOG_LEVEL_INFO, "flow_lm: voice conditioning -> %d rows\n", n_voice);
        }

        if (st_) codec_lm_state_free(st_);
        st_ = codec_lm_state_new(lm_);
        if (!st_) return fail("state allocation failed");
        if (codec_lm_flow_prefill(st_, ids.data(), n_tok, n_voice > 0 ? voice_rows.data() : nullptr, n_voice) != CODEC_STATUS_SUCCESS) {
            return fail(std::string("prefill: ") + codec_lm_state_get_last_error(st_));
        }

        // frames_after_eos: fixed from the GGUF if >= 0, else the word-count guess.
        const int32_t frames_after_eos = fi_->frames_after_eos >= 0 ? fi_->frames_after_eos : fae_guess_;
        // _estimate_max_gen_len: frame_rate 12.5, ~3 tokens/s + 2 s padding.
        int32_t max_gen = max_frames_ > 0 ? max_frames_ : (int32_t) std::ceil(((double) n_tok / 3.0 + 2.0) * 12.5);
        if (max_gen < 8) max_gen = 8;

        // CFM init noise ~ N(0, temperature): Box-Muller on a uniform mt19937
        // stream (std::normal_distribution's polar method gives poor early
        // draws; the FlowLM is very sensitive to the first frames' noise).
        std::seed_seq seq{ seed_, 0x9E3779B9u, 0x243F6A88u, 0xB7E15162u };
        std::mt19937 rng(seq);
        const float noise_std = std::sqrt(fi_->temperature);
        auto gauss = [&]() -> float {
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            float u1 = u(rng), u2 = u(rng);
            if (u1 < 1e-12f) u1 = 1e-12f;
            return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2) * noise_std;
        };
        std::vector<float> noise((size_t) ldim), lat((size_t) ldim, 0.0f), den((size_t) ldim);
        int32_t eos_step = -1;
        for (int32_t step = 0; step < max_gen; ++step) {
            for (int32_t d = 0; d < ldim; ++d) noise[(size_t) d] = gauss();
            int32_t is_eos = 0;
            float eos_logit = 0.0f;
            if (codec_lm_flow_step(st_, noise.data(), lat.data(), &eos_logit, &is_eos) != CODEC_STATUS_SUCCESS) {
                return fail("step " + std::to_string(step) + ": " + codec_lm_state_get_last_error(st_));
            }
            // min_frames ignores the EOS head for the first frames (it can fire on
            // frame 0/1 for some prompts + noise streams).
            if (is_eos && eos_step < 0 && step >= min_len_) eos_step = step;
            if (eos_step >= 0 && step >= eos_step + frames_after_eos) { stop_reason_ = "eos_head"; break; }
            codec_lm_flow_denorm_latent(lm_, lat.data(), den.data());
            latents_.insert(latents_.end(), den.begin(), den.end());
            ++n_frames_;
        }
        codec_log(CODEC_LOG_LEVEL_INFO, "flow_lm: AR done: %d frames, eos_step=%d, stop=%s\n", n_frames_, eos_step, stop_reason_.c_str());
        return n_frames_ > 0 || fail("no frames generated");
    }

    bool decode(audio_lm_audio_output * out) override {
        const int32_t ldim = fi_->ldim, n = n_frames_;
        if (n == 0) return fail("no frames generated");
        // Frame-major [t * ldim + d] -> channel-major [d * n + t].
        std::vector<float> cm((size_t) ldim * n);
        for (int32_t t = 0; t < n; ++t) {
            for (int32_t d = 0; d < ldim; ++d) cm[(size_t) d * n + t] = latents_[(size_t) t * ldim + d];
        }
        codec_context * cctx = audio_lm_get_codec_context(alm_);
        codec_pcm_buffer pcm;
        std::memset(&pcm, 0, sizeof(pcm));
        if (codec_decode_quantized_representation(cctx, cm.data(), ldim, n, &pcm, codec_decode_default_params()) != CODEC_STATUS_SUCCESS) {
            return fail(std::string("decode: ") + codec_get_last_error(cctx));
        }
        out->pcm.assign(pcm.data, pcm.data + pcm.n_samples);
        out->sample_rate = pcm.sample_rate;
        out->n_channels  = 1;
        codec_pcm_buffer_free(&pcm);
        return true;
    }

private:
    codec_lm * lm_;
    const codec_lm_flow_info * fi_;
    codec_lm_state * st_ = nullptr;
    std::string text_;
    int32_t fae_guess_ = 3, min_len_ = 0, max_frames_ = 0, n_frames_ = 0;
    uint32_t seed_ = 0;
    std::vector<float> latents_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_flow(audio_lm_context * alm, codec_lm * lm) {
    const codec_lm_flow_info * fi = codec_lm_flow_get_info(lm);
    return fi ? std::make_unique<flow_family>(alm, lm, fi) : nullptr;
}

}  // namespace codec_common
