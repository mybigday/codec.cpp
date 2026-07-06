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
// This file is just: parse args → load → { info | decode | synthesize } →
// write WAV.  A future server example links codec_tts_runner the same way.
//
//   tts-cli info       --model X.gguf
//        Lifecycle + capability queries (modality, n_codebook, hidden_dim,
//        has_speaker_enc) via the plain codec_common API.
//
//   tts-cli decode     --model X.gguf --codes codes.npy --output OUT.wav
//        Reads a (T, n_cb) int32 .npy of codes, runs codec_decode through
//        codec_common, writes mono PCM16 WAV.
//
//   tts-cli synthesize --model X.gguf [--backbone LLAMA.gguf] --text "..."
//                      --output OUT.wav [--ref-audio REF.wav] [knobs]
//        Full TTS — delegates to codec_common::tts_runner_synthesize.

#include "codec_common.h"
#include "tts_runner.h"
#include "utils/wav_io.h"
#include "utils/npy_io.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s <subcommand> --model PATH [options]\n"
        "\n"
        "subcommands:\n"
        "  info       --model PATH\n"
        "  decode     --model PATH --codes PATH.npy --output PATH.wav [--n-threads N]\n"
        "  synthesize --model CODEC.gguf [--backbone LLAMA.gguf] --text STRING\n"
        "             --output PATH.wav [--ref-audio PATH.wav]\n"
        "             [--max-frames N] [--seed N] [--temp F] [--top-p F] [--top-k N]\n"
        "             [--cfg F] [--timesteps N] [--min-len N]\n"
        "             [--cfg-weight F] [--min-p F] [--rep-penalty F] [--n-threads N]\n"
        "             Full host AR loop: a llama.cpp backbone (--backbone) drives\n"
        "             the codec_common per-step hooks end-to-end.  Flow is chosen\n"
        "             from the model's GGUF metadata.  Self-contained models\n"
        "             (Pocket-TTS FlowLM) need no backbone.  --temp 0 = greedy.\n",
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
    std::string codes;
    std::string text;
    std::string ref_audio;
    std::string output;
    std::string backbone;

    int32_t  n_threads = 0;
    bool     use_gpu   = false;
    int32_t  max_frames = 0;
    uint32_t seed       = 0xC0DEC1AB;

    // Sampler / flow knobs — has_* gates whether the value was supplied.
    bool has_temp = false;   float   temp  = 0.0f;
    bool has_top_p = false;  float   top_p = 0.0f;
    bool has_top_k = false;  int32_t top_k = 0;

    float   cfg       = 2.0f;    // continuous CFM / FlowLM
    int32_t timesteps = 10;
    int32_t min_len   = -1;

    bool has_cfg_weight = false;  float cfg_weight = 0.0f;   // Chatterbox T3
    bool has_min_p = false;       float min_p = 0.0f;
    bool has_rep_penalty = false; float repetition_penalty = 0.0f;
};

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
        else if (a == "--codes")      { const char * v = need(); if (!v) return false; out->codes = v; i++; }
        else if (a == "--text")       { const char * v = need(); if (!v) return false; out->text = v;  i++; }
        else if (a == "--ref-audio")  { const char * v = need(); if (!v) return false; out->ref_audio = v; i++; }
        else if (a == "--output")     { const char * v = need(); if (!v) return false; out->output = v; i++; }
        else if (a == "--backbone")   { const char * v = need(); if (!v) return false; out->backbone = v; i++; }
        else if (a == "--n-threads")  { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->n_threads)) return false; i++; }
        else if (a == "--gpu")        { out->use_gpu = true; }
        else if (a == "--max-frames") { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->max_frames)) return false; i++; }
        else if (a == "--seed")       { const char * v = need(); if (!v) return false; int32_t s; if (!parse_i32(v, &s)) return false; out->seed = (uint32_t) s; i++; }
        else if (a == "--temp")       { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->temp)) return false; out->has_temp = true; i++; }
        else if (a == "--top-p")      { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->top_p)) return false; out->has_top_p = true; i++; }
        else if (a == "--top-k")      { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->top_k)) return false; out->has_top_k = true; i++; }
        else if (a == "--cfg")        { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->cfg)) return false; i++; }
        else if (a == "--timesteps")  { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->timesteps)) return false; i++; }
        else if (a == "--min-len")    { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->min_len)) return false; i++; }
        else if (a == "--cfg-weight") { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->cfg_weight)) return false; out->has_cfg_weight = true; i++; }
        else if (a == "--min-p")      { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->min_p)) return false; out->has_min_p = true; i++; }
        else if (a == "--rep-penalty"){ const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->repetition_penalty)) return false; out->has_rep_penalty = true; i++; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return false; }
    }
    return !out->model.empty();
}

// ─── Subcommand: info ────────────────────────────────────────────
int cmd_info(const args & a) {
    codec_common::audio_lm_params p;
    p.codec_path = a.model;
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

// ─── Subcommand: decode ──────────────────────────────────────────
// Offline decode of pre-sampled codes through the codec_common pipeline.
int cmd_decode(const args & a) {
    if (a.codes.empty() || a.output.empty()) {
        std::fprintf(stderr, "decode requires --codes and --output\n");
        return 1;
    }
    codec_common::audio_lm_params p;
    p.codec_path = a.model;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;
    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    std::vector<int32_t> npy_codes;
    int32_t n_frames = 0, n_q = 0;
    std::string npy_err;
    if (!codec_example_load_npy_i32_2d_tq(a.codes.c_str(), &npy_codes, &n_q, &n_frames, &npy_err)) {
        std::fprintf(stderr, "failed to load %s: %s\n", a.codes.c_str(), npy_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 3;
    }

    if (!codec_common::audio_lm_push_codes(ctx, npy_codes.data(), n_frames, n_q)) {
        std::fprintf(stderr, "audio_lm_push_codes failed: %s\n", codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 4;
    }

    codec_common::audio_lm_audio_output pcm;
    if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
        std::fprintf(stderr, "audio_lm_decode_audio failed: %s\n", codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 5;
    }

    std::string wav_err;
    if (!codec_example_write_wav_pcm16(
            a.output.c_str(), pcm.pcm.data(),
            (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels),
            pcm.sample_rate, &wav_err, pcm.n_channels)) {
        std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 6;
    }
    std::printf("wrote %s: %d samples @ %d Hz (%d ch)\n",
                a.output.c_str(),
                (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels),
                pcm.sample_rate, pcm.n_channels);
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
    rp.backbone_path = a.backbone;
    rp.text          = a.text;
    rp.ref_audio_path = a.ref_audio;
    rp.n_threads     = a.n_threads;
    rp.use_gpu       = a.use_gpu;
    rp.seed          = a.seed;
    rp.max_frames    = a.max_frames;
    rp.has_temp = a.has_temp;   rp.temp  = a.temp;
    rp.has_top_p = a.has_top_p; rp.top_p = a.top_p;
    rp.has_top_k = a.has_top_k; rp.top_k = a.top_k;
    rp.cfg = a.cfg; rp.timesteps = a.timesteps; rp.min_len = a.min_len;
    rp.has_cfg_weight = a.has_cfg_weight; rp.cfg_weight = a.cfg_weight;
    rp.has_min_p = a.has_min_p; rp.min_p = a.min_p;
    rp.has_rep_penalty = a.has_rep_penalty; rp.repetition_penalty = a.repetition_penalty;
    return rp;
}

int write_result_wav(const args & a, const codec_common::tts_runner_result & r) {
    const int32_t nch = r.n_channels > 0 ? r.n_channels : 1;
    const int32_t nsamp = (int32_t) (r.pcm.size() / (size_t) nch);
    std::string wav_err;
    if (!codec_example_write_wav_pcm16(a.output.c_str(), r.pcm.data(), nsamp,
                                       r.sample_rate, &wav_err, nch)) {
        std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
        return 9;
    }
    const double secs = (double) nsamp / (double) (r.sample_rate > 0 ? r.sample_rate : 1);
    std::printf("wrote %s: %d samples @ %d Hz (%.2fs, %d ch)\n",
                a.output.c_str(), nsamp, r.sample_rate, secs, nch);
    return 0;
}

int cmd_synthesize(const args & a) {
    if (a.output.empty()) {
        std::fprintf(stderr, "synthesize requires --output PATH.wav\n");
        return 1;
    }
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
    if (a.sub == "decode")     return cmd_decode(a);
    if (a.sub == "synthesize") return cmd_synthesize(a);

    std::fprintf(stderr, "unknown subcommand: %s\n", a.sub.c_str());
    print_usage(argv[0]);
    return 1;
}
