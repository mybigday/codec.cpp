// tts-cli — reference CLI driving the codec_common audio-LM API.
//
// Mirrors codec-cli's subcommand layout.  Each subcommand exercises one
// slice of the codec_common surface; new subcommands light up as the
// roadmap (docs/codec_common_api.md §"Landing roadmap") advances.
//
//   tts-cli info     --model X.gguf
//        Lifecycle + capability queries (modality, n_codebook,
//        hidden_dim, has_speaker_enc).  Available from step 1.
//
//   tts-cli decode   --model X.gguf --codes codes.npy --output OUT.wav
//        Reads a (T, n_cb) int32 .npy of codes, runs codec_decode via
//        codec_common, writes mono PCM16 WAV.  Available from step 1.
//
//   tts-cli synthesize --model X.gguf --text "..." --output OUT.wav
//                      [--ref-audio REF.wav] [--emotion FLOAT]
//        Full TTS — needs build_prompt + per-step observe + decode.
//        Reports the unwired piece per inference Type until the
//        corresponding roadmap step lands.

#include "codec_common.h"
#include "utils/wav_io.h"
#include "utils/npy_io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
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
        "  synthesize --model PATH --text STRING --output PATH.wav\n"
        "             [--ref-audio PATH.wav] [--emotion FLOAT] [--n-threads N]\n",
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

// ─── Common args struct ───────────────────────────────────────────
struct args {
    std::string sub;
    std::string model;
    std::string codes;
    std::string text;
    std::string ref_audio;
    std::string output;
    std::optional<float> emotion;
    int32_t n_threads = 0;
    bool    use_gpu   = false;
};

bool parse_args(int argc, char ** argv, args * out) {
    if (argc < 2) return false;
    out->sub = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](int32_t skip = 1) -> const char * {
            if (i + skip >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                return nullptr;
            }
            return argv[i + skip];
        };
        if (a == "--model")     { const char * v = need(); if (!v) return false; out->model = v; i++; }
        else if (a == "--codes"){ const char * v = need(); if (!v) return false; out->codes = v; i++; }
        else if (a == "--text") { const char * v = need(); if (!v) return false; out->text = v;  i++; }
        else if (a == "--ref-audio") { const char * v = need(); if (!v) return false; out->ref_audio = v; i++; }
        else if (a == "--output"){ const char * v = need(); if (!v) return false; out->output = v; i++; }
        else if (a == "--emotion"){ const char * v = need(); if (!v) return false; float f; if (!parse_f32(v, &f)) return false; out->emotion = f; i++; }
        else if (a == "--n-threads"){ const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->n_threads)) return false; i++; }
        else if (a == "--gpu") { out->use_gpu = true; }
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
    if (!ctx) {
        std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str());
        return 2;
    }

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
// Offline decode of pre-sampled codes through the codec_common pipeline:
// load → push_codes → decode_audio.  The same code path observe_token
// (step 3+) will run through, just with the AR loop swapped for a file.
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
        std::fprintf(stderr, "audio_lm_push_codes failed: %s\n",
                     codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 4;
    }

    codec_common::audio_lm_audio_output pcm;
    if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
        std::fprintf(stderr, "audio_lm_decode_audio failed: %s\n",
                     codec_common::audio_lm_last_error(ctx));
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
// Step 2 deliverable: build_prompt runs the speaker-encode pipeline
// when --ref-audio is given (or implicitly via speaker_emb in a
// future invocation).  AR loop + decode_audio require host llama.cpp
// integration (step 3+); we print prompt structure as proof the
// speaker-encode side reaches `embeds_prefix` correctly.
int cmd_synthesize(const args & a) {
    codec_common::audio_lm_params p;
    p.codec_path = a.model;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;
    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    codec_common::audio_lm_input in;
    in.text = a.text;
    if (a.emotion) in.emotion = &a.emotion.value();

    // Optional ref audio: WAV load + auto-resample to the encoder's
    // declared sample rate is the host's job in a real integration;
    // tts-cli reads the WAV as-is and refuses if the SR doesn't match
    // (forces the user to bring resampled audio for now — a simple
    // resampler can land alongside Type C/D once it's broadly useful).
    std::vector<float> ref_pcm;
    int32_t ref_sr = 0;
    if (!a.ref_audio.empty()) {
        codec_example_wav_data w;
        std::string werr;
        if (!codec_example_load_wav_pcm16(a.ref_audio.c_str(), &w, &werr)) {
            std::fprintf(stderr, "failed to load %s: %s\n", a.ref_audio.c_str(), werr.c_str());
            codec_common::audio_lm_free(ctx);
            return 3;
        }
        // Downmix to mono + convert PCM16 → F32.
        const int32_t nch = w.n_channels > 0 ? w.n_channels : 1;
        const int32_t nframes = (int32_t) (w.pcm_i16.size() / (size_t) nch);
        ref_pcm.assign((size_t) nframes, 0.0f);
        for (int32_t i = 0; i < nframes; ++i) {
            float acc = 0.0f;
            for (int32_t c = 0; c < nch; ++c) {
                acc += w.pcm_i16[(size_t) i * (size_t) nch + (size_t) c] / 32768.0f;
            }
            ref_pcm[(size_t) i] = acc / (float) nch;
        }
        ref_sr  = w.sample_rate;
        in.ref_pcm         = ref_pcm.data();
        in.ref_n_samples   = (int32_t) ref_pcm.size();
        in.ref_sample_rate = ref_sr;
    }

    codec_common::audio_lm_prompt prompt;
    if (!codec_common::audio_lm_build_prompt(ctx, in, &prompt)) {
        std::fprintf(stderr, "build_prompt failed: %s\n",
                     codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 4;
    }

    std::printf("build_prompt OK\n");
    std::printf("  tokens.size        = %zu\n", prompt.tokens.size());
    std::printf("  embeds_prefix_rows = %d\n",  prompt.embeds_prefix_rows);
    std::printf("  embeds_prefix_hidd = %d\n",  prompt.embeds_prefix_hidden);
    std::printf("  embeds_prefix.size = %zu (= rows×hidden)\n", prompt.embeds_prefix.size());

    // Host-side llama.cpp AR loop + observe_token + decode_audio land
    // in roadmap step 3+.  Bail cleanly here so the user knows where
    // tts-cli sits relative to the doc.
    std::fprintf(stderr,
        "\nsynthesize: AR loop not yet wired (roadmap step 3+).  "
        "Output file %s NOT written.\n",
        a.output.empty() ? "<unspecified>" : a.output.c_str());
    codec_common::audio_lm_free(ctx);
    return 0;
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
