// MOSS-TTS-Nano (codec_lm residual_depth_ar with a GPT-2 depth decoder):
// the global GPT-2 transformer is the backbone (llama.cpp starcoder2, its
// token_embd = wte), the codec_lm adaptor owns the local transformer, the
// audio embeddings and the 16 codebook heads.  Reference:
// modeling_moss_tts_nano.py (build_inference_input_ids, _iter_generation_events)
// and prompting.py.
//
// Backbone sequence (voice clone; without a voice "None" stands in for the
// reference section):
//   user_prefix <|audio_start|> [reference rows] <|audio_end|> after_reference
//   text assistant_prefix <|audio_start|>
// Template spans are token ids baked by the converter; the text is normalized
// (nmt_nfkc) and tokenized with the backbone vocab.  A reference row is the
// composed embedding of [user_slot, codes...].  Per frame the depth decoder
// reads the backbone hidden: codebook 0 (the text slot) is the assistant slot
// or the audio end, which stops the generation before any audio code is
// drawn; otherwise the 16 codes follow and the next backbone row is the
// composed [assistant_slot, codes...].

#include "audio_lm_family.h"
#include "lm/lm_models.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

namespace codec_common {
namespace {

class moss_nano_family : public alm_family {
public:
    moss_nano_family(audio_lm_context * alm, codec_lm * lm, const codec_lm_moss_nano_info * mi)
        : alm_family(alm), lm_(lm), mi_(mi) {}
    const char * name() const override { return "moss_tts_nano"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : 48000;
    }

    // voice_clone splits text above 50 tokens into sentence / clause chunks
    // (_split_text_into_best_sentences); in bytes that is about 50 CJK
    // characters' worth (kana are about one token each), and English packs well
    // below 50 tokens at the same size.  Chunks are joined with a 0.24 s pause
    // (_concat_voice_clone_waveform_chunks, no trimming or fades).
    split_params segment_split() const override { return {split_mode::clause, 150, 0}; }
    join_params  segment_join() const override { return {false, 0, 240}; }

    // Prompt: <__codec:user_prefix__><__codec:audio_start__><__voice__><__codec:audio_end__>
    //         <__codec:after_reference__><__codec:text__>{normalized text}
    //         <__codec:assistant_prefix__><__codec:audio_start__>
    // (no voice: <__codec:no_reference__> in place of the reference section).
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        std::string text;
        if (req.text_known && !normalize(req.text, &text)) return false;
        const bool voice = ref_codes(req.voice, nullptr, nullptr);
        out->prompt = alm_codec_marker("user_prefix") +
                      (voice ? alm_codec_marker("audio_start") + CODEC_AUDIO_LM_MARKER_VOICE + alm_codec_marker("audio_end")
                             : alm_codec_marker("no_reference")) +
                      alm_codec_marker("after_reference") + alm_codec_marker("text") + text +
                      alm_codec_marker("assistant_prefix") + alm_codec_marker("audio_start");
        alm_completion & d = out->desc;
        d.flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        d.n_embd = mi_->hidden_dim;
        // One random stream for the text slot and the codebooks; the reference
        // applies no repetition penalty (audio_repetition_penalty = 1).
        d.head_groups      = {alm_head_group{false, {}, {}}};
        d.needs_embeddings = true;
        d.sample_first     = false;
        d.streamable       = true;
        d.max_frames       = req.max_frames > 0 ? req.max_frames : 300;   // inference(max_new_frames)
        d.min_frames       = std::max(0, req.min_frames);
        return true;
    }

    bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        hidden_ = mi_->hidden_dim;
        n_cb_   = audio_lm_n_codebook(alm_);
        if (n_cb_ != mi_->n_codebook) return fail("codebook count mismatch");
        frames_ = 0;
        min_frames_ = std::max(0, req.min_frames);
        set_seed(CODEC_AUDIO_LM_DEFAULT_SEED);   // random unless the host sets one (codec_audio_lm_set_seed)
        stop_reason_ = "max_frames";
        audio_lm_set_uses_embed_override(alm_, true, 1);

        std::vector<alm_chunk> chunks;
        std::vector<int32_t> toks;
        auto ids = [&](const int32_t * p, int32_t n) { toks.insert(toks.end(), p, p + n); };
        auto flush = [&]() {
            if (!toks.empty()) chunks.push_back(alm_chunk::of_tokens(std::move(toks)));
            toks.clear();
        };
        if (!prompt.codec("user_prefix")) return fail("the prompt starts with <__codec:user_prefix__>");
        ids(mi_->user_prefix, mi_->n_user_prefix);
        if (prompt.codec("audio_start")) {
            toks.push_back(mi_->audio_start);
            if (!prompt.voice()) return fail("<__codec:audio_start__> in the user turn must precede <__voice__>");
            const int32_t * codes = nullptr;
            int32_t n_frames = 0;
            if (!ref_codes(req.voice, &codes, &n_frames)) return fail("<__voice__> needs a voice with reference codes");
            flush();
            std::vector<float> rows((size_t) n_frames * hidden_);
            std::vector<int32_t> row((size_t) n_cb_);
            for (int32_t t = 0; t < n_frames; ++t) {
                row[0] = mi_->user_slot;
                std::memcpy(row.data() + 1, codes + (size_t) t * (n_cb_ - 1), (size_t) (n_cb_ - 1) * sizeof(int32_t));
                if (!audio_lm_compose_audio_codes_embd(alm_, row.data(), n_cb_, rows.data() + (size_t) t * hidden_, hidden_)) {
                    return fail_alm("reference rows");
                }
            }
            chunks.push_back(alm_chunk::of_embd(std::move(rows), n_frames));
            if (!prompt.codec("audio_end")) return fail("<__voice__> must be followed by <__codec:audio_end__>");
            toks.push_back(mi_->audio_end);
        } else if (prompt.codec("no_reference")) {
            ids(mi_->no_reference, mi_->n_no_reference);
        } else {
            return fail("the user turn needs a reference section or <__codec:no_reference__>");
        }
        if (!prompt.codec("after_reference")) return fail("missing <__codec:after_reference__>");
        ids(mi_->after_reference, mi_->n_after_reference);
        if (!prompt.codec("text")) return fail("missing <__codec:text__>");
        const std::vector<int32_t> text = vocab.tokenize(prompt.text(), false, false);
        if (text.empty()) return fail("empty text after tokenize");
        toks.insert(toks.end(), text.begin(), text.end());
        if (!prompt.codec("assistant_prefix") || !prompt.codec("audio_start") || !prompt.done()) {
            return fail("the prompt ends with <__codec:assistant_prefix__><__codec:audio_start__>");
        }
        ids(mi_->assistant_prefix, mi_->n_assistant_prefix);
        toks.push_back(mi_->audio_start);
        flush();
        codec_log(CODEC_LOG_LEVEL_INFO, "moss-tts-nano: %zu text tokens, prompt %zu chunks\n", text.size(), chunks.size());
        out->seqs = {std::move(chunks)};
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        if (!audio_lm_step_begin(alm_, hiddens, hidden_)) return fail_alm("step");
        hidden_frame_.assign(hiddens, hiddens + hidden_);
        cb_ = 0;
        return next_head(v);
    }

    const float * head_logits() const override { return logits_.data(); }

    bool head_token(int32_t id, codec_audio_lm_verdict * v) override {
        if (id < 0 || id >= (int32_t) logits_.size()) return fail("head token out of range");
        if (cb_ >= n_cb_) return fail("no codebook is waiting for a token");
        if (cb_ == 0 && id == mi_->audio_end) {
            // The reference breaks here: no audio codes for this frame.
            stop_reason_ = "audio_end";
            codec_log(CODEC_LOG_LEVEL_INFO, "AR loop done: %d frames, stop=audio_end\n", frames_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1, 0, 0};
            return true;
        }
        if (!audio_lm_step_push_code(alm_, id)) return fail_alm("step");
        ++cb_;
        return next_head(v);
    }

    const float * next_embd(int32_t) const override { return audio_lm_get_next_embed(alm_, nullptr); }

    void set_seed(uint32_t seed) override { rng_.seed(alm_resolve_seed(seed)); }

private:
    bool next_head(codec_audio_lm_verdict * v) {
        if (cb_ < n_cb_) {
            int32_t idx = 0, n = 0;
            const float * lg = audio_lm_step_logits(alm_, &idx, &n);
            if (lg == nullptr) return fail_alm("step logits");
            logits_.assign(lg, lg + n);
            if (cb_ == 0) decide_text_slot();
            *v = {CODEC_AUDIO_LM_SAMPLE_HEAD, false, true, false, -1, 0, n};
            return true;
        }
        std::vector<int32_t> codes((size_t) n_cb_, 0);
        if (!audio_lm_step_finish(alm_, codes.data(), n_cb_)) return fail_alm("step");
        // Accumulates the frame and composes the next backbone row (the sum of
        // the 17 tables: wte[assistant_slot] + the codes' audio embeddings).
        if (audio_lm_observe_codes(alm_, codes.data(), n_cb_, hidden_frame_.data(), hidden_) != OBSERVE_CONSUMED_EMBED) {
            return fail_alm("frame");
        }
        ++frames_;
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, true, -1, 0, 0};
        return true;
    }

    // The text slot is the stop decision: the reference samples it between the
    // assistant slot and the audio end at its own temperature (1.5, top_p 1),
    // apart from the codebooks' settings (1.7 / top_k 25 / top_p 0.8) the host
    // applies to every head; under those, the end could not be drawn below
    // p = 0.2 and generations ran on.  So the model draws it here, like a stop
    // head, with its own seeded stream, and hands the host a one-hot head.
    // codec.lm.moss_tts_nano.text_temperature <= 0: argmax (greedy).
    void decide_text_slot() {
        const int32_t n = (int32_t) logits_.size(), end = mi_->audio_end, slot = mi_->assistant_slot;
        if (end < 0 || end >= n || slot < 0 || slot >= n) return;
        bool stop;
        if (frames_ < min_frames_) {
            stop = false;   // min_frames not reached
        } else {
            const char * tv = audio_lm_meta_str(alm_, "codec.lm.moss_tts_nano.text_temperature");
            const float temp = tv ? std::strtof(tv, nullptr) : 1.5f;
            const float d = logits_[(size_t) end] - logits_[(size_t) slot];
            if (temp <= 0.0f) {
                stop = d > 0.0f;
            } else {
                const double p_end = 1.0 / (1.0 + std::exp(-(double) d / temp));
                stop = std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < p_end;
            }
        }
        for (int32_t i = 0; i < n; ++i) {
            if (i != (stop ? end : slot)) logits_[(size_t) i] = -INFINITY;
        }
    }

    // The voice's reference codes: I32 [n_q, T] with n_q = n_codebook - 1.
    bool ref_codes(const codec_speaker * spk, const int32_t ** codes, int32_t * n_frames) const {
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        if (spk == nullptr || !codec_speaker_get_slot(spk, "ref_codes", &ty, ne, &data)) return false;
        if (ty != CODEC_SPEAKER_SLOT_I32 || ne[0] != mi_->n_codebook - 1 || ne[1] < 1) return false;
        if (codes) *codes = static_cast<const int32_t *>(data);
        if (n_frames) *n_frames = (int32_t) ne[1];
        return true;
    }

    bool normalize(const std::string & text, std::string * out) {
        const int32_t n = codec_lm_moss_nano_normalize_text(lm_, text.c_str(), nullptr, 0);
        if (n < 0) return fail("text normalization");
        out->assign((size_t) n + 1, '\0');
        codec_lm_moss_nano_normalize_text(lm_, text.c_str(), out->data(), n + 1);
        out->resize((size_t) n);
        return true;
    }

    codec_lm * lm_;
    const codec_lm_moss_nano_info * mi_;
    std::vector<float> hidden_frame_, logits_;
    std::mt19937 rng_;
    int32_t hidden_ = 0, n_cb_ = 0, cb_ = 0, frames_ = 0, min_frames_ = 0;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_moss_nano(audio_lm_context * alm, struct codec_lm * lm) {
    const codec_lm_moss_nano_info * mi = codec_lm_moss_nano_get_info(lm);
    if (mi == nullptr) return nullptr;
    return std::make_unique<moss_nano_family>(alm, lm, mi);
}

}  // namespace codec_common
