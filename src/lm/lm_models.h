#ifndef CODEC_LM_MODELS_H
#define CODEC_LM_MODELS_H

// Internal: model-specific codec_lm helpers the codec_audio_lm families build
// on (common/audio_lm_family_*.cpp).  Not part of the public API: hosts reach
// these models through include/codec_audio_lm.h.

#include "codec_lm.h"

#ifdef __cplusplus
extern "C" {
#endif

// ─── Chatterbox T3 host-orchestration helpers ───────────────────────
// T3 is an embd-driven Llama backbone: the host owns the llama.cpp
// decode loop; these helpers supply the T3-specific pieces (tokenizer,
// prompt embeds, per-step speech embeds) that live on the codec.cpp
// side.  All return CODEC_STATUS_NOT_SUPPORTED when the loaded model is
// not a Chatterbox T3 adaptor (no `codec.lm.chatterbox.*` metadata).

// Static config surfaced from `codec.lm.chatterbox.*` metadata.
struct codec_lm_chatterbox_info {
    int32_t hidden_dim;             // 1024
    int32_t text_vocab_size;        // 704 (en) / 2454 (mtl)
    int32_t speech_vocab_size;      // 8194
    int32_t start_text_token;       // 255
    int32_t stop_text_token;        // 0
    int32_t start_speech_token;     // 6561
    int32_t stop_speech_token;      // 6562
    int32_t cond_rows;              // cond_enc output rows (34)
    int32_t has_tokenizer;          // 1 if BPE tokenizer baked into GGUF
    int32_t has_builtin_conds;      // 1 if builtin speaker conditioning baked
    int32_t is_multilingual;        // 1 for the 2454-vocab variants
    // Turbo (GPT-2 T3, `codec.lm.chatterbox.variant = "turbo"`) — fields
    // appended so older callers keep their layout.
    int32_t is_turbo;               // 1 for Chatterbox Turbo
    int32_t backbone_tokenizer;     // 1: tokenize text with the backbone vocab
                                    //    (after codec_lm_chatterbox_normalize_text)
    int32_t silence_token;          // S3Gen silence code appended after AR (-1 = none)
    int32_t n_trailing_silence;     // how many silence codes to append (Turbo: 3)
    // Reference sampling defaults (`codec.lm.chatterbox.sampling.*`); a
    // value < 0 means "not specified" and the host picks its own default.
    float   temperature;
    int32_t top_k;
    float   top_p;
    float   min_p;
    float   repetition_penalty;
    float   cfg_weight;
};

// Returns the chatterbox info, or NULL if the model is not a T3 adaptor.
const struct codec_lm_chatterbox_info *
codec_lm_chatterbox_get_info(struct codec_lm * lm);

// Tokenize `text` with the baked EnTokenizer BPE (punc_norm applied
// internally, mirroring ChatterboxTTS.generate).  Writes token ids into
// `out_ids` (capacity `cap`); sets `*n_out` to the count.  Does NOT add
// the start/stop text tokens — the host wraps those.  Returns
// CODEC_STATUS_NOT_SUPPORTED if no tokenizer is baked.
enum codec_status codec_lm_chatterbox_tokenize(
    struct codec_lm * lm,
    const char *      text,
    int32_t *         out_ids,
    int32_t           cap,
    int32_t *         n_out);

// Apply the variant's punc_norm text cleanup (tts.py / tts_turbo.py) into
// `out` (capacity `cap` bytes incl. NUL).  Hosts with backbone_tokenizer=1
// call this, then tokenize with the backbone vocab.
enum codec_status codec_lm_chatterbox_normalize_text(
    struct codec_lm * lm,
    const char *      text,
    char *            out,
    int32_t           cap);

// Build the full backbone-input embed prefix for T3 inference:
//   [ cond_emb (cond_rows) | text_emb+text_pos_emb (n_text_wrapped) | BOS ]
// Turbo: [ cond_emb (1 + n_cond_tokens) | text_emb (n_text) | BOS ] — no
// start/stop text tokens, no learned PEs, one BOS row, never a CFG lane.
// The caller passes RAW text ids (without start/stop text tokens); this
// helper prepends start_text_token, appends stop_text_token, adds
// text_pos_emb, then appends the speech BOS
// (start_speech_token @ speech_pos_emb[0]).
//
// When `cfg_weight > 0` two rows are produced (cond, then uncond): the
// uncond row is identical except its text-embedding content is zeroed
// (text_pos_emb preserved), matching T3's `text_emb[1].zero_()`.  The
// output is laid out row-major as `[n_rows_total × hidden]` where
// n_rows_total = n_seq * (cond_rows + n_text_wrapped + 1) and n_seq is
// 2 when cfg_weight>0 else 1.  `*out_seq_len` = per-sequence row count.
//
// Conditioning source: if `speaker_emb`!=NULL it is used (with
// `ref_speech_tokens`/`emotion`); otherwise the builtin conds baked
// into the GGUF are used (requires has_builtin_conds).
// Conditioning source precedence:
//   1. `ref_pcm` (mono F32, `ref_sample_rate`) → run the voice encoder
//      (VE) + cond_enc to derive the speaker prefix from reference audio.
//      The cond-prompt speech tokens come from `ref_speech_tokens` if
//      given, else the builtin prompt tokens.
//   2. else `speaker_emb` (256-d) → cond_enc from a cached embedding.
//   3. else the builtin conds baked into the GGUF (has_builtin_conds).
enum codec_status codec_lm_chatterbox_build_prompt(
    struct codec_lm * lm,
    const int32_t *   text_ids,
    int32_t           n_text,
    float             cfg_weight,
    const float *     speaker_emb,          // NULL → builtin/ref
    int32_t           speaker_emb_dim,
    const int32_t *   ref_speech_tokens,    // NULL → builtin
    int32_t           n_ref_speech_tokens,
    const float *     emotion,              // NULL → builtin/default
    const float *     ref_pcm,              // NULL → no ref audio
    int32_t           ref_n_samples,
    int32_t           ref_sample_rate,
    float *           out_embeds,
    int32_t           out_cap_rows,         // capacity in rows
    int32_t *         out_seq_len,          // per-sequence rows
    int32_t *         out_n_seq);           // 1 or 2 (CFG)

// Compose the next backbone-input speech embed for AR step `pos`:
//   speech_emb[code] + speech_pos_emb[pos].
// `pos` is the speech-position index: BOS is 0, the first generated
// token is 1, etc.  When CFG is active the host feeds the same embed to
// both lanes.  Writes `hidden` floats to `out`.
enum codec_status codec_lm_chatterbox_compose_speech_embd(
    struct codec_lm * lm,
    int32_t           code,
    int32_t           pos,
    float *           out,
    int32_t           out_cap);

// ─── CosyVoice host-orchestration helpers ───────────────────────────
// CosyVoice (v1 300M / BreezyVoice, CosyVoice2, CosyVoice3) runs its AR
// backbone in llama.cpp (Qwen2, or the `cosyvoice` arch for v1).  The speech
// head + speech embedding are the generic parallel_heads_delay pair
// (n_codebook = 1); these helpers supply the CosyVoice prompt rows.  All
// return CODEC_STATUS_NOT_SUPPORTED (or NULL) when the model is not a
// CosyVoice adaptor (no `codec.lm.cosyvoice.*` metadata).
//
// Prompt (llm.py inference):
//   v1:    [sos] [spk row + text-encoder rows (codec_lm_cosyvoice_text_rows)] [task_id] [speech rows]
//   v2/v3: [sos] [backbone token embeds of prompt_text ++ text]               [task_id] [speech rows]
// The host samples from the speech-head logits and stops on any id in
// [stop_begin, stop_end).
struct codec_lm_cosyvoice_info {
    int32_t version;                 // 1, 2, 3
    int32_t hidden_dim;              // backbone hidden (1024 v1, 896 v2/v3)
    int32_t speech_token_size;       // 4096 / 6561 (codec tokens are [0, speech_token_size))
    int32_t n_vocab;                 // speech-head size (4097 / 6564 / 6761)
    int32_t special_in_speech_table; // 1: sos / task_id are speech-table rows (v3)
    int32_t sos;
    int32_t task_id;
    int32_t stop_begin;              // ids in [stop_begin, stop_end) end generation
    int32_t stop_end;
    int32_t ignore_eos_id;           // masked to -inf while step < min_len (reference quirk)
    int32_t text_in_backbone;        // 1: tokenize + embed text with the backbone vocab
    int32_t has_text_encoder;        // 1: v1 text side via codec_lm_cosyvoice_text_rows
    int32_t ras_sampling;            // 1: repetition-aware sampling, 0: plain top-k
    int32_t top_k;
    float   top_p;
    int32_t win_size;
    float   tau_r;
    float   min_token_text_ratio;    // min_len = text_len * ratio (target text only)
    float   max_token_text_ratio;
    int32_t n_silent_tokens;         // v3: runs > max_silent_run of these are dropped before the flow
    const int32_t * silent_tokens;
    int32_t max_silent_run;
    const char * prompt_prefix;      // v3 system prompt ("" when none)
    int32_t bopomofo_frontend;       // BreezyVoice: text expects 注音 annotations (g2pw, external)
};

const struct codec_lm_cosyvoice_info * codec_lm_cosyvoice_get_info(struct codec_lm * lm);

// which = 0: sos row, 1: task_id row.  Writes hidden_dim floats.
enum codec_status codec_lm_cosyvoice_special_embd(
    struct codec_lm * lm, int32_t which, float * out, int32_t out_cap);

// Speech-embedding rows for `codes` (prompt speech tokens, or the token
// sampled at each step), row-major [n × hidden_dim].
enum codec_status codec_lm_cosyvoice_speech_embd(
    struct codec_lm * lm, const int32_t * codes, int32_t n, float * out, int32_t out_cap_rows);

// v1 text side: [spk_aff(F.normalize(xvec))] ++ text_encoder_affine(
// text_encoder(text_embedding[ids])) as row-major [rows × hidden_dim].
// `xvec` (192-d CAMPPlus embedding) is optional; `text_ids` must be the
// whole prompt_text ++ text sequence (the encoder is bidirectional).
enum codec_status codec_lm_cosyvoice_text_rows(
    struct codec_lm * lm, const int32_t * text_ids, int32_t n_text,
    const float * xvec, int32_t xvec_dim,
    float * out, int32_t out_cap_rows, int32_t * out_rows);

// ─── Pocket-TTS FlowLM host-orchestration helpers (CODEC_LM_KIND_FLOW_LM) ───
// FlowLM is self-contained: the AR transformer, text LUT, LSD flow head and EOS
// head all live in the codec GGUF, so there is no llama.cpp backbone.  The host
// drives generation entirely through these helpers.  All return
// CODEC_STATUS_NOT_SUPPORTED when the model is not a FlowLM adaptor.

// Static config surfaced from `codec.lm.*` metadata.
struct codec_lm_flow_info {
    int32_t d_model;                 // AR transformer hidden (1024)
    int32_t ldim;                    // continuous latent dim (32)
    int32_t n_txt_bins;              // SentencePiece vocab (4000)
    int32_t insert_bos_before_voice; // 1 if a learned BOS row precedes voice rows
    int32_t frames_after_eos;        // -1 = derive from word count (1-3), else fixed
    float   temperature;             // LSD init-noise variance (0.7)
    float   eos_threshold;           // EOS fires when out_eos logit > this (-4.0)
    int32_t lsd_decode_steps;        // LSD Euler steps (1)
    int32_t has_tokenizer;           // 1 if a SentencePiece model is baked in
};

// Returns the FlowLM info, or NULL if the model is not a FlowLM adaptor.
const struct codec_lm_flow_info * codec_lm_flow_get_info(struct codec_lm * lm);

// Tokenize `text` with the baked SentencePiece unigram model (identity
// normalizer, add_dummy_prefix, byte fallback).  Writes ids into `out_ids`
// (capacity `cap`); sets `*n_out`.  Does NOT prepend/append BOS/EOS.  Returns
// NOT_SUPPORTED if no tokenizer is baked.
enum codec_status codec_lm_flow_tokenize(
    struct codec_lm * lm,
    const char *      text,
    int32_t *         out_ids,
    int32_t           cap,
    int32_t *         n_out);

// Project a Mimi voice-conditioning latent `mu` [ldim × n_voice] (channel-major,
// mu[d*n_voice + t]) through `speaker_proj` into `out` [n_voice × d_model] rows
// (row-major, out[t*d_model + c]).  Used to build the voice-cloning rows for
// codec_lm_flow_prefill.  Returns NOT_SUPPORTED if the model has no speaker_proj.
enum codec_status codec_lm_flow_speaker_rows(
    struct codec_lm * lm,
    const float *     mu,
    int32_t           n_voice,
    float *           out,
    int32_t           out_cap_rows);

// Prefill the AR transformer KV cache over the prompt prefix:
//   [ text LUT embeds (n_tok) | (bos_before_voice) | voice rows (n_voice) ]
// `voice_rows` is [n_voice × d_model] row-major or NULL (text-only / default
// voice).  Resets any prior generation state.  After this the state is primed
// for codec_lm_flow_step.
enum codec_status codec_lm_flow_prefill(
    struct codec_lm_state * st,
    const int32_t *         token_ids,
    int32_t                 n_tok,
    const float *           voice_rows,   // NULL = no voice conditioning
    int32_t                 n_voice);

// Advance one AR frame.  Runs the transformer step over the KV cache, computes
// the EOS logit, samples LSD init noise (or uses `noise` [ldim] when non-NULL
// for deterministic / parity runs), LSD-decodes the next latent, appends its
// input embedding to the KV cache, and writes:
//   out_latent   : [ldim] the generated latent (pre-denormalization).
//   out_eos_logit: the raw out_eos scalar (optional; may be NULL).
//   out_is_eos   : 1 if out_eos_logit > eos_threshold, else 0 (optional).
// Feed successive frames until the EOS + frames_after_eos policy stops (host).
enum codec_status codec_lm_flow_step(
    struct codec_lm_state * st,
    const float *           noise,          // [ldim] or NULL to sample
    float *                 out_latent,     // [ldim]
    float *                 out_eos_logit,  // scalar, optional
    int32_t *               out_is_eos);    // optional

// Denormalize a generated latent for Mimi decode: out = latent * emb_std +
// emb_mean, elementwise over ldim.  (The FlowLM predicts normalized latents;
// Mimi consumes the denormalized ones.)
enum codec_status codec_lm_flow_denorm_latent(
    struct codec_lm * lm,
    const float *     latent,     // [ldim]
    float *           out);       // [ldim]

#ifdef __cplusplus
}
#endif

#endif  // CODEC_LM_MODELS_H
