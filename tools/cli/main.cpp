// tts-cli — thin reference driver over codec_common + the TTS runner.
//
// A demo of wiring codec.cpp (the codec_common audio-LM API) + llama.cpp
// (the isolated backbone).  Everything reusable lives in the common layer:
//
//   * per-model prompt/step bookkeeping + codes→PCM decode → common/audio_lm
//     (the codec_common per-step hook API; llama.rn owns its own loop)
//   * the full reference host AR loop (backbone load, tokenize/prefill,
//     every per-model flow, sampling, CFG, streaming, EOS, decode) →
//     common/tts_runner (codec_tts_runner, built with CODEC_TTS_BACKBONE)
//
// This file is just: parse args → load → { info | voices | synthesize } →
// write WAV.  tools/server/tts-server links codec_tts_runner the same way.
//
//   tts-cli info       --model X.gguf
//        Lifecycle + capability queries (modality, n_codebook, hidden_dim,
//        has_speaker_enc) via the plain codec_common API.
//
//   tts-cli synthesize --model X.gguf [--backbone LLAMA.gguf] --text "..."
//                      --output OUT.wav [--ref-audio REF.wav] [knobs]
//        Full TTS — delegates to codec_common::tts_runner_synthesize.

#include "codec_common.h"
#include "tts_runner.h"
#include "utils/wav_io.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ── Optional espeak-ng phonemization (CODEC_TTS_ESPEAK) ──────────────────────
//
// When built with -DCODEC_TTS_ESPEAK=ON, --text and --ref-text are phonemized
// via espeak-ng (IPA, with stress, language en-us) before being handed to the
// runner.  This matches the NeuTTS reference phonemizer params (neutts-facts
// §Step 3: EspeakBackend language='en-us', preserve_punctuation=True,
// with_stress=True).
//
// NOTE: this path is OPT-IN and NOT exercised by the e2e smoke test.  Task 6
// pre-phonemizes in Python with the phonemizer library; the tts-cli espeak-OFF
// (default) path passes text through unchanged, which is what the e2e uses.
//
// Requires: libespeak-ng-dev (provides <espeak-ng/speak_lib.h>).
// On this machine (Ubuntu 6.17, 2026-08-06) libespeak-ng-dev is NOT installed;
// espeak-ON builds will fail with "header not found" until the package is added.
// The runtime binary (/usr/bin/espeak-ng 1.51) is present but its headers are not.
#ifdef CODEC_TTS_ESPEAK
#include <espeak-ng/speak_lib.h>
#include <sstream>
#endif

namespace {

// ── espeak-ng phonemization helper ───────────────────────────────────────────
#ifdef CODEC_TTS_ESPEAK
// Phonemize a single UTF-8 text string via espeak-ng (IPA output, with stress).
// Matches the NeuTTS Python reference (neutts-facts §Step 3):
//   EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
//
// espeak-ng C API:
//   espeak_Initialize(AUDIO_OUTPUT_SYNCH_PLAYBACK, ...)  — no audio device
//   espeak_SetVoiceByName("en-us")
//   espeak_TextToPhonemes(&text_ptr, espeakCHARS_UTF8, espeakPHONEMES_IPA)
//     — advances text_ptr past each clause, returns IPA phoneme string for
//       that clause (NULL at end-of-input).
//
// Returns the IPA string on success, or the original text on failure (with a
// warning to stderr so the espeak-OFF behavior is preserved).
static std::string espeak_phonemize(const std::string & text) {
    if (text.empty()) return text;

    // AUDIO_OUTPUT_SYNCH_PLAYBACK suppresses audio device opening (phoneme-only
    // mode uses synchronous output; the buffer size 0 means use the default).
    int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCH_PLAYBACK, /*buflength=*/0,
                               /*path=*/nullptr, /*options=*/0);
    if (sr < 0) {
        std::fprintf(stderr, "[espeak] espeak_Initialize failed (sr=%d); "
                             "passing text unchanged\n", sr);
        return text;
    }

    if (espeak_SetVoiceByName("en-us") != EE_OK) {
        std::fprintf(stderr, "[espeak] SetVoiceByName('en-us') failed; "
                             "passing text unchanged\n");
        espeak_Terminate();
        return text;
    }

    // espeak_TextToPhonemes walks the text clause by clause.
    // text_ptr is advanced by the library; we keep a local copy for the loop.
    const char * text_ptr = text.c_str();
    std::string ipa;
    bool first = true;
    while (text_ptr && *text_ptr) {
        // espeakPHONEMES_IPA → IPA with stress markers embedded
        // espeakCHARS_UTF8  → input is UTF-8
        const char * clause = espeak_TextToPhonemes(
            reinterpret_cast<const void **>(&text_ptr),
            espeakCHARS_UTF8,
            espeakPHONEMES_IPA);
        if (!clause) break;
        if (!first) ipa += ' ';
        ipa += clause;
        first = false;
    }

    espeak_Terminate();

    if (ipa.empty()) {
        std::fprintf(stderr, "[espeak] phonemization returned empty; "
                             "passing text unchanged\n");
        return text;
    }

    // Mirror NeuTTS _to_phones: split on whitespace, rejoin with single spaces.
    // This collapses the trailing space that espeak-ng inserts before punctuation.
    std::istringstream ss(ipa);
    std::string tok, out;
    while (ss >> tok) { if (!out.empty()) out += ' '; out += tok; }
    return out.empty() ? text : out;
}
#endif  // CODEC_TTS_ESPEAK

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s <subcommand> --model PATH [options]\n"
        "\n"
        "subcommands:\n"
        "  info       --model PATH\n"
        "  voices     --model CODEC.gguf [--speaker-file PATH.gguf]\n"
        "             List the codec's builtin speakers (or those in a speaker\n"
        "             file); * marks the default.\n"
        "  synthesize --model CODEC.gguf [--backbone LLAMA.gguf] --text STRING\n"
        "             --output PATH.wav [--ref-audio PATH.wav] [--ref-text STR]\n"
        "             [--voice NAME] [--speaker-file PATH.gguf] [--save-speaker PATH.gguf]\n"
        "             [--instruct STR] [--max-frames N] [--min-len N] [--segment-bytes N]\n"
        "             [--seed N] [--temp F] [--top-k N] [--top-p F] [--min-p F]\n"
        "             [--repeat-last-n N] [--repeat-penalty F] [--presence-penalty F]\n"
        "             [--frequency-penalty F] [--dry-multiplier F] [--dry-base F]\n"
        "             [--dry-allowed-length N] [--dry-penalty-last-n N] [--samplers LIST]\n"
        "             [--cfg-weight F] [--cfg F] [--timesteps N] [--n-threads N]\n"
        "             [--grammar FILE|STRING]\n"
        "             Sampling flags are llama.cpp's, with its defaults; settings left\n"
        "             unset come from the backbone GGUF (general.sampling.*), the\n"
        "             model card's recommendation.  --cfg-weight (backbone pair),\n"
        "             --cfg / --timesteps (flow / CFM decoder) default to the codec\n"
        "             GGUF's.\n"
        "             --grammar attaches a GBNF constraint to the backbone\n"
        "             sampler (models with a metadata-derived auto-grammar use\n"
        "             it by default; --grammar overrides).\n"
        "             Full host AR loop: a llama.cpp backbone (--backbone) drives\n"
        "             the codec_common per-step hooks end-to-end.  Flow is chosen\n"
        "             from the model's GGUF metadata.  --temp 0 = greedy, --seed -1 = random.\n"
        "             Decoder speaker: --speaker-file > --ref-audio (when the codec\n"
        "             bundles speaker encoders) > --voice (builtin) > model default;\n"
        "             with --speaker-file, --voice picks a speaker inside the file.\n"
        "             --save-speaker writes the one in use for reuse.\n"
        "             --segment-bytes N synthesizes long text a segment of at most\n"
        "             N UTF-8 bytes at a time (sentence / paragraph breaks; 0 = off).\n"
        "\n"
        "  all subcommands: [--override-kv KEY=int|float|bool|str:VALUE] (repeatable)\n"
        "             overrides codec GGUF metadata at load, e.g.\n"
        "             --override-kv chatterbox_s3g.flow.n_timesteps=int:4\n",
        prog);
}

bool parse_i32(const char * s, int32_t * out) {
    if (!s || !out) return false;
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    *out = (int32_t) v;
    return true;
}

bool parse_f32(const char * s, float * out) {
    if (!s || !out) return false;
    char * end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0') return false;
    *out = (float) v;
    return true;
}

// --grammar accepts either a path to a .gbnf FILE or a literal GBNF STRING.
// Heuristic: if the arg names an existing readable file, load its contents;
// otherwise treat the arg itself as the grammar text.  (A literal GBNF always
// contains "::=", so a bare filename that happens not to exist just passes
// through as-is and the grammar parser reports the error downstream.)
std::string load_grammar_arg(const char * v) {
    if (!v || !*v) return "";
    if (FILE * f = std::fopen(v, "rb")) {
        std::string s;
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (n > 0) { s.resize((size_t) n); size_t rd = std::fread(&s[0], 1, (size_t) n, f); s.resize(rd); }
        std::fclose(f);
        return s;
    }
    return v;
}

void print_modality_mask(uint32_t mask) {
    if (mask == 0u) { std::printf("(none)"); return; }
    bool first = true;
    auto emit = [&](const char * n) { std::printf("%s%s", first ? "" : " | ", n); first = false; };
    if (mask & codec_common::INPUT_TEXT  ) emit("INPUT_TEXT");
    if (mask & codec_common::INPUT_AUDIO ) emit("INPUT_AUDIO");
    if (mask & codec_common::OUTPUT_TEXT ) emit("OUTPUT_TEXT");
    if (mask & codec_common::OUTPUT_AUDIO) emit("OUTPUT_AUDIO");
}

// ─── Command-line args ────────────────────────────────────────────
struct args {
    std::string sub;
    std::string model;
    std::string text;
    std::string ref_audio;
    std::string ref_text;
    std::string voice;
    std::string speaker_file;
    std::string save_speaker;
    std::string instruct;
    std::string output;
    std::string backbone;

    int32_t  n_threads = 0;
    bool     use_gpu   = false;
    int32_t  max_frames = 0;
    int32_t  segment_bytes = 0;
    int32_t  min_len   = -1;

    // llama.cpp sampling; `sampling_set` marks the flags given (the backbone
    // GGUF's general.sampling.* fills the rest).
    codec_common::tts_sampling sampling;
    std::string samplers;
    uint32_t    sampling_set = 0;

    float   cfg_weight = -1.0f;  // decoder settings; < 0 = the codec GGUF's
    float   cfg        = -1.0f;
    int32_t timesteps  = -1;

    std::string grammar;   // GBNF for the backbone sampler (FILE contents or literal)

    std::vector<codec_model_kv_override> kv_overrides;  // --override-kv (codec GGUF)
};

const codec_common::tts_sampling_option * find_sampling_flag(const std::string & a) {
    for (const auto & o : codec_common::tts_sampling_options()) if (a == o.flag) return &o;
    return nullptr;
}

bool parse_args(int argc, char ** argv, args * out) {
    if (argc < 2) return false;
    out->sub = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&]() -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); return nullptr; }
            return argv[i + 1];
        };
        if      (a == "--model")      { const char * v = need(); if (!v) return false; out->model = v; i++; }
        else if (a == "--text")       { const char * v = need(); if (!v) return false; out->text = v;  i++; }
        else if (a == "--ref-audio")  { const char * v = need(); if (!v) return false; out->ref_audio = v; i++; }
        else if (a == "--ref-text")   { const char * v = need(); if (!v) return false; out->ref_text = v; i++; }
        else if (a == "--voice")      { const char * v = need(); if (!v) return false; out->voice = v; i++; }
        else if (a == "--speaker-file") { const char * v = need(); if (!v) return false; out->speaker_file = v; i++; }
        else if (a == "--save-speaker") { const char * v = need(); if (!v) return false; out->save_speaker = v; i++; }
        else if (a == "--instruct")   { const char * v = need(); if (!v) return false; out->instruct = v; i++; }
        else if (a == "--output")     { const char * v = need(); if (!v) return false; out->output = v; i++; }
        else if (a == "--backbone")   { const char * v = need(); if (!v) return false; out->backbone = v; i++; }
        else if (a == "--n-threads")  { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->n_threads)) return false; i++; }
        else if (a == "--gpu")        { out->use_gpu = true; }
        else if (a == "--max-frames") { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->max_frames)) return false; i++; }
        else if (a == "--segment-bytes") { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->segment_bytes)) return false; i++; }
        else if (a == "--seed")       { const char * v = need(); if (!v) return false; int32_t s; if (!parse_i32(v, &s)) return false; out->sampling.seed = (uint32_t) s; i++; }
        else if (a == "--min-len")    { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->min_len)) return false; i++; }
        else if (a == "--cfg-weight") { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->cfg_weight)) return false; i++; }
        else if (a == "--cfg")        { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->cfg)) return false; i++; }
        else if (a == "--timesteps")  { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->timesteps)) return false; i++; }
        else if (a == "--samplers")   { const char * v = need(); if (!v) return false; out->samplers = v; out->sampling_set |= codec_common::TTS_SAMPLING_SAMPLERS; i++; }
        else if (const codec_common::tts_sampling_option * so = find_sampling_flag(a)) {
            const char * v = need(); if (!v) return false;
            if (so->f32 ? !parse_f32(v, &(out->sampling.*(so->f32))) : !parse_i32(v, &(out->sampling.*(so->i32)))) return false;
            out->sampling_set |= so->bit; i++;
        }
        else if (a == "--override-kv"){
            const char * v = need(); if (!v) return false;
            codec_model_kv_override ov;
            if (!codec_common::parse_kv_override(v, &ov)) {
                std::fprintf(stderr, "invalid --override-kv '%s' (want KEY=int|float|bool|str:VALUE)\n", v);
                return false;
            }
            out->kv_overrides.push_back(ov); i++;
        }
        else if (a == "--grammar")    { const char * v = need(); if (!v) return false; out->grammar = load_grammar_arg(v); i++; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return false; }
    }
    return !out->model.empty();
}

// ─── Subcommand: info ────────────────────────────────────────────
int cmd_info(const args & a) {
    codec_common::audio_lm_params p;
    p.codec_path = a.model;
    p.kv_overrides = a.kv_overrides;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;

    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    std::printf("path:            %s\n",  a.model.c_str());
    std::printf("modality:        0x%x  ", codec_common::audio_lm_modality(ctx));
    print_modality_mask(codec_common::audio_lm_modality(ctx));
    std::printf("\n");
    std::printf("n_codebook:      %d\n",  codec_common::audio_lm_n_codebook(ctx));
    std::printf("hidden_dim:      %d\n",  codec_common::audio_lm_hidden_dim(ctx));
    std::printf("has_speaker_enc: %s\n",  codec_common::audio_lm_has_speaker_enc(ctx) ? "true" : "false");

    codec_common::audio_lm_free(ctx);
    return 0;
}

// ─── Subcommand: voices ──────────────────────────────────────────
int cmd_voices(const args & a) {
    codec_common::audio_lm_params p;
    p.codec_path   = a.model;
    p.kv_overrides = a.kv_overrides;
    p.use_gpu      = false;
    p.n_threads    = a.n_threads;

    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    std::vector<codec_common::audio_lm_speaker_entry> list;
    const char * source = a.model.c_str();
    if (!a.speaker_file.empty()) {
        source = a.speaker_file.c_str();
        if (!codec_common::audio_lm_list_speaker_file(ctx, source, &list)) {
            std::fprintf(stderr, "%s\n", codec_common::audio_lm_last_error(ctx));
            codec_common::audio_lm_free(ctx);
            return 6;
        }
    } else {
        list = codec_common::audio_lm_list_codec_speakers(ctx);
    }
    std::printf("%s: %zu speaker(s)", source, list.size());
    if (a.speaker_file.empty()) {
        std::printf(", from audio: %s",
                    codec_common::audio_lm_codec_can_speaker_from_audio(ctx) ? "yes" : "no");
    }
    std::printf("\n");
    for (const auto & e : list) {
        std::printf("  %s%s", e.name.c_str(), e.is_default ? " *" : "");
        if (!e.ref_text.empty()) {
            // Keep one line per speaker: cut long transcripts on a UTF-8 boundary.
            std::string rt = e.ref_text;
            if (rt.size() > 60) {
                size_t cut = 60;
                while (cut > 0 && (static_cast<unsigned char>(rt[cut]) & 0xC0) == 0x80) --cut;
                rt = rt.substr(0, cut) + "...";
            }
            std::printf("  \"%s\"", rt.c_str());
        }
        std::printf("\n");
    }
    codec_common::audio_lm_free(ctx);
    return 0;
}

// ─── Subcommand: synthesize ──────────────────────────────────────
// Thin driver over the reference runner.  With the backbone linked in
// (TTS_CLI_HAVE_BACKBONE) the full flow set is available; without it, only
// the self-contained (no-backbone) FlowLM path runs, else a clear message.
codec_common::tts_runner_params make_runner_params(const args & a) {
    codec_common::tts_runner_params rp;
    rp.codec_path    = a.model;
    rp.kv_overrides  = a.kv_overrides;
    rp.backbone_path = a.backbone;
    rp.text          = a.text;
    rp.ref_audio_path = a.ref_audio;
    rp.ref_text       = a.ref_text;
    rp.voice          = a.voice;
    rp.speaker_file   = a.speaker_file;
    rp.save_speaker   = a.save_speaker;
    rp.instruct       = a.instruct;
    rp.n_threads     = a.n_threads;
    rp.use_gpu       = a.use_gpu;
    rp.max_frames    = a.max_frames;
    rp.min_frames    = a.min_len;
    rp.segment_bytes = a.segment_bytes;
    rp.sampling          = a.sampling;
    rp.samplers          = a.samplers;
    rp.sampling_user_set = a.sampling_set;
    rp.lm_guidance   = a.cfg_weight;
    rp.flow_guidance = a.cfg;
    rp.flow_steps    = a.timesteps;
    rp.grammar = a.grammar;
    return rp;
}

int write_result_wav(const args & a, const codec_common::tts_runner_result & r) {
    const int32_t nch = r.n_channels > 0 ? r.n_channels : 1;
    const int32_t nsamp = (int32_t) (r.pcm.size() / (size_t) nch);
    std::string wav_err;
    if (!codec_util_write_wav_pcm16(a.output.c_str(), r.pcm.data(), nsamp,
                                       r.sample_rate, &wav_err, nch)) {
        std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
        return 9;
    }
    const double secs = (double) nsamp / (double) (r.sample_rate > 0 ? r.sample_rate : 1);
    std::printf("wrote %s: %d samples @ %d Hz (%.2fs, %d ch)\n",
                a.output.c_str(), nsamp, r.sample_rate, secs, nch);
    std::printf("timing: load %.0f ms, speaker %.0f ms, generate %.0f ms, decode %.0f ms, total %.0f ms, frames %d\n",
                r.load_ms - r.speaker_ms, r.speaker_ms, r.total_ms - r.load_ms - r.decode_ms, r.decode_ms,
                r.total_ms, r.n_frames);
    return 0;
}

int cmd_synthesize(args a) {  // note: args by value so we can mutate text/ref_text in-place
    if (a.output.empty()) {
        std::fprintf(stderr, "synthesize requires --output PATH.wav\n");
        return 1;
    }

#ifdef CODEC_TTS_ESPEAK
    // Phonemize --text and --ref-text before building rp (NeuTTS phonemes path).
    // NOTE: this path is opt-in (CODEC_TTS_ESPEAK) and NOT exercised by the e2e
    // smoke test.  The e2e pre-phonemizes in Python; espeak-OFF (default) passes
    // text through unchanged.
    if (!a.text.empty()) {
        const std::string phones = espeak_phonemize(a.text);
        std::printf("[espeak] text    → '%s'\n", phones.c_str());
        a.text = phones;
    }
    if (!a.ref_text.empty()) {
        const std::string phones = espeak_phonemize(a.ref_text);
        std::printf("[espeak] ref_text → '%s'\n", phones.c_str());
        a.ref_text = phones;
    }
#endif  // CODEC_TTS_ESPEAK

    const codec_common::tts_runner_params rp = make_runner_params(a);
    codec_common::tts_runner_result r;

#ifdef TTS_CLI_HAVE_BACKBONE
    // Full runner: tries the self-contained FlowLM first, then backbone flows.
    if (!codec_common::tts_runner_synthesize(rp, &r)) {
        std::fprintf(stderr, "%s\n", r.error.c_str());
        return 6;
    }
#else
    // No backbone linked — only the self-contained (no-backbone) FlowLM path
    // is available.  A 0 return means "not a FlowLM"; print the same message
    // as before and bail.
    int handled = codec_common::tts_runner_synthesize_selfcontained(rp, &r);
    if (!handled) {
        std::fprintf(stderr,
            "synthesize: tts-cli was built without the llama backbone "
            "(CODEC_TTS_BACKBONE=OFF).  Reconfigure with the backbone to enable.\n");
        return 1;
    }
    if (!r.error.empty()) {
        std::fprintf(stderr, "%s\n", r.error.c_str());
        return 6;
    }
#endif

    return write_result_wav(a, r);
}

}  // namespace

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, &a)) { print_usage(argv[0]); return 1; }

    if (a.sub == "info")       return cmd_info(a);
    if (a.sub == "voices")     return cmd_voices(a);
    if (a.sub == "synthesize") return cmd_synthesize(a);

    std::fprintf(stderr, "unknown subcommand: %s\n", a.sub.c_str());
    print_usage(argv[0]);
    return 1;
}
