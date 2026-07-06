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

#ifdef TTS_CLI_HAVE_BACKBONE
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
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
        "  synthesize --model CODEC.gguf --backbone LLAMA.gguf --text STRING\n"
        "             --output PATH.wav [--ref-audio PATH.wav] [--emotion FLOAT]\n"
        "             [--max-frames N] [--seed N] [--temp F] [--top-p F] [--top-k N]\n"
        "             [--cfg F] [--timesteps N] [--min-len N] [--n-threads N]\n"
        "             Full host AR loop: a llama.cpp backbone (--backbone) drives\n"
        "             the codec_common per-step hooks end-to-end.  Flow is chosen\n"
        "             from the model's GGUF metadata (host_arch / codec.lm.kind):\n"
        "             continuous CFM (BlueMagpie), residual depth-AR (CSM,\n"
        "             Qwen3-TTS), or parallel-heads delay (MOSS-TTSD).  Sampling\n"
        "             defaults to the model's training-time knobs; --temp 0 = greedy.\n",
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

    // trace / simulate-typeA / simulate-typeB
    std::string tokens_csv;
    int32_t     audio_offset = -1;
    int32_t     audio_count  = 0;
    int32_t     audio_eos    = -1;
    bool        audio_offset_set = false;
    int32_t     start_step   = 1;   // Type B AR-step start
    bool        probe_first  = false;
    bool        use_embed_override = false;

    // simulate-continuous (BlueMagpie / continuous-latent_cfm)
    std::string hidden_npy;   // (T, hidden_dim) F32
    std::string noise_npy;    // (T, patch_size*latent_dim) F32, optional
    float       cont_cfg     = 2.0f;
    int32_t     cont_timesteps = 10;
    int32_t     cont_min_len = -1;   // -1 = model default (GGUF codec.lm.min_len / 2)

    // synthesize (host AR loop)
    std::string backbone;            // llama.cpp backbone gguf
    int32_t     max_frames = 0;      // 0 → per-model default
    uint32_t    seed       = 0xC0DEC1AB;
    std::optional<float>   temp;     // sampler overrides (unset → model default)
    std::optional<float>   top_p;
    std::optional<int32_t> top_k;

    // Chatterbox T3 synthesize knobs (mirror ChatterboxTTS.generate).
    std::optional<float>   cfg_weight;         // unset → 0.5
    std::optional<float>   min_p;              // unset → 0.05
    std::optional<float>   repetition_penalty; // unset → 1.2
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
        else if (a == "--tokens"){ const char * v = need(); if (!v) return false; out->tokens_csv = v; i++; }
        else if (a == "--audio-offset"){ const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->audio_offset)) return false; out->audio_offset_set = true; i++; }
        else if (a == "--audio-count") { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->audio_count))  return false; i++; }
        else if (a == "--audio-eos")   { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->audio_eos))    return false; i++; }
        else if (a == "--start-step")  { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->start_step))   return false; i++; }
        else if (a == "--probe-first") { out->probe_first = true; }
        else if (a == "--use-embed-override") { out->use_embed_override = true; }
        else if (a == "--hidden-npy") { const char * v = need(); if (!v) return false; out->hidden_npy = v; i++; }
        else if (a == "--noise-npy")  { const char * v = need(); if (!v) return false; out->noise_npy  = v; i++; }
        else if (a == "--cfg")        { const char * v = need(); if (!v) return false; if (!parse_f32(v, &out->cont_cfg)) return false; i++; }
        else if (a == "--timesteps")  { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->cont_timesteps)) return false; i++; }
        else if (a == "--min-len")    { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->cont_min_len)) return false; i++; }
        else if (a == "--backbone")   { const char * v = need(); if (!v) return false; out->backbone = v; i++; }
        else if (a == "--max-frames") { const char * v = need(); if (!v) return false; if (!parse_i32(v, &out->max_frames)) return false; i++; }
        else if (a == "--seed")       { const char * v = need(); if (!v) return false; int32_t s; if (!parse_i32(v, &s)) return false; out->seed = (uint32_t) s; i++; }
        else if (a == "--temp")       { const char * v = need(); if (!v) return false; float f; if (!parse_f32(v, &f)) return false; out->temp = f; i++; }
        else if (a == "--top-p")      { const char * v = need(); if (!v) return false; float f; if (!parse_f32(v, &f)) return false; out->top_p = f; i++; }
        else if (a == "--top-k")      { const char * v = need(); if (!v) return false; int32_t k; if (!parse_i32(v, &k)) return false; out->top_k = k; i++; }
        else if (a == "--cfg-weight") { const char * v = need(); if (!v) return false; float f; if (!parse_f32(v, &f)) return false; out->cfg_weight = f; i++; }
        else if (a == "--min-p")      { const char * v = need(); if (!v) return false; float f; if (!parse_f32(v, &f)) return false; out->min_p = f; i++; }
        else if (a == "--rep-penalty"){ const char * v = need(); if (!v) return false; float f; if (!parse_f32(v, &f)) return false; out->repetition_penalty = f; i++; }
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
// Full host AR loop: a llama.cpp backbone (`--backbone`) drives the
// codec_common per-step hooks end-to-end.  Three flows keyed on the
// model's `audio_lm_get_prompt_info` (host_arch / kind metadata):
//   * continuous CFM (BlueMagpie / barbet) — text_prefill + observe_hidden
//   * residual depth-AR (CSM / Qwen3-TTS / MOSS-TTS-Realtime) — codebook
//     step machine + observe_codes
//   * parallel-heads delay (MOSS-TTSD) — cb0 from backbone lm_head +
//     set_text_context + step machine + observe_codes
#ifndef TTS_CLI_HAVE_BACKBONE
int cmd_synthesize(const args & a) {
    (void) a;
    std::fprintf(stderr,
        "synthesize: tts-cli was built without the llama backbone "
        "(CODEC_TTS_BACKBONE=OFF).  Reconfigure with the backbone to enable.\n");
    return 1;
}
#else

namespace {

// ── Backbone text-embedding table reader ──────────────────────────────
// MOSS-TTS-Realtime composes each backbone-step input as
//   text_embd[text_token] + compose_audio_embd(prev_frame_codes)
// where the audio part lives in the codec_lm but the TEXT embedding table
// (`token_embd.weight`, [hidden, V_text]) lives in the backbone GGUF.
// llama.cpp exposes no raw-embedding API, so we mmap the backbone GGUF a
// second time and dequant embedding rows on demand via ggml type traits
// (handles bf16 / f16 / quantised transparently).
struct TextEmbdTable {
    gguf_context * gg   = nullptr;
    ggml_context * meta = nullptr;   // holds tensor metadata (no_alloc)
    const uint8_t * base = nullptr;  // mmapped tensor-data region
    std::vector<uint8_t> blob;       // owns the file bytes
    int64_t hidden = 0;
    int64_t vocab  = 0;
    ggml_type type = GGML_TYPE_F32;
    size_t row_bytes = 0;
    ggml_to_float_t to_float = nullptr;

    bool load(const char * path, int32_t want_hidden, std::string & err) {
        struct ggml_init_params ip = { /*mem_size*/ 0, /*mem_buffer*/ nullptr,
                                       /*no_alloc*/ true };
        gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta };
        gg = gguf_init_from_file(path, gp);
        if (!gg) { err = "gguf_init_from_file failed"; return false; }
        const int64_t tid = gguf_find_tensor(gg, "token_embd.weight");
        if (tid < 0) { err = "token_embd.weight not found in backbone"; return false; }
        ggml_tensor * t = ggml_get_tensor(meta, "token_embd.weight");
        if (!t) { err = "token_embd metadata lookup failed"; return false; }
        hidden = t->ne[0];
        vocab  = t->ne[1];
        type   = t->type;
        if ((int32_t) hidden != want_hidden) {
            err = "token_embd hidden mismatch"; return false;
        }
        const ggml_type_traits * tr = ggml_get_type_traits(type);
        to_float = tr ? tr->to_float : nullptr;
        // For a quantised type to_float works on a whole row (k = hidden,
        // which must be a multiple of block size for legal types).
        if (!to_float) { err = "no to_float for token_embd type"; return false; }
        row_bytes = ggml_row_size(type, hidden);

        // Read the whole file into `blob`, then point `base` at the tensor
        // data region (data_offset within the file).
        FILE * f = std::fopen(path, "rb");
        if (!f) { err = "fopen backbone failed"; return false; }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        blob.resize((size_t) sz);
        size_t rd = std::fread(blob.data(), 1, (size_t) sz, f);
        std::fclose(f);
        if (rd != (size_t) sz) { err = "backbone read short"; return false; }
        const size_t data_off = gguf_get_data_offset(gg);
        const size_t t_off     = gguf_get_tensor_offset(gg, tid);
        base = blob.data() + data_off + t_off;
        return true;
    }

    // Dequant embedding row `token` into `out` (hidden floats).
    bool row(int32_t token, float * out) const {
        if (token < 0 || token >= (int32_t) vocab || !base || !to_float) return false;
        const void * src = base + (size_t) token * row_bytes;
        to_float(src, out, hidden);
        return true;
    }

    ~TextEmbdTable() {
        if (gg)   gguf_free(gg);
        if (meta) ggml_free(meta);
    }
};

// Minimal xorshift64* sampler mirroring rn-tts's sample_codec_logits:
// temp<=0 → greedy argmax; else softmax(temp) → top-k → top-p → sample.
struct Sampler {
    std::mt19937_64 rng;
    explicit Sampler(uint32_t seed) : rng(seed ? seed : 0xC0DEC1ABull) {}

    // Apply CTRL-style repetition penalty in place over `history` codes
    // (mutates a caller-owned logits copy).  Mirrors
    // streaming_mossttsrealtime.py apply_repetition_penalty.
    static void apply_rep_penalty(float * logits, int32_t n,
                                  const int32_t * history, int32_t n_hist,
                                  float penalty) {
        if (penalty == 1.0f || n_hist <= 0) return;
        for (int32_t i = 0; i < n_hist; ++i) {
            const int32_t tok = history[i];
            if (tok < 0 || tok >= n) continue;
            float v = logits[tok];
            logits[tok] = (v < 0.0f) ? v * penalty : v / penalty;
        }
    }

    int32_t sample(const float * logits, int32_t n, float temp, float top_p, int32_t top_k) {
        if (n <= 0) return 0;
        if (temp <= 0.0f) {
            int32_t best = 0; float bv = logits[0];
            for (int32_t i = 1; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
            return best;
        }
        std::vector<std::pair<float,int32_t>> p(n);
        float mx = logits[0];
        for (int32_t i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
        double sum = 0.0;
        for (int32_t i = 0; i < n; ++i) { double e = std::exp((logits[i]-mx)/temp); p[i] = {(float)e, i}; sum += e; }
        for (auto & q : p) q.first = (float)(q.first / sum);
        std::sort(p.begin(), p.end(), [](auto&x,auto&y){ return x.first > y.first; });
        int32_t keep = n;
        if (top_k > 0 && top_k < keep) keep = top_k;
        if (top_p > 0.0f && top_p < 1.0f) {
            double c = 0.0; int32_t kk = 0;
            for (; kk < keep; ++kk) { c += p[kk].first; if (c >= top_p) { kk++; break; } }
            keep = std::max(1, kk);
        }
        double norm = 0.0;
        for (int32_t i = 0; i < keep; ++i) norm += p[i].first;
        std::uniform_real_distribution<double> u(0.0, norm);
        double r = u(rng), acc = 0.0;
        for (int32_t i = 0; i < keep; ++i) { acc += p[i].first; if (r <= acc) return p[i].second; }
        return p[keep-1].second;
    }
};

// Load ref audio (mono F32) into `in` if given.  Returns false on error.
bool load_ref_audio(const args & a, codec_common::audio_lm_input & in, std::vector<float> & ref_pcm) {
    if (a.ref_audio.empty()) return true;
    codec_example_wav_data w;
    std::string werr;
    if (!codec_example_load_wav_pcm16(a.ref_audio.c_str(), &w, &werr)) {
        std::fprintf(stderr, "failed to load %s: %s\n", a.ref_audio.c_str(), werr.c_str());
        return false;
    }
    const int32_t nch = w.n_channels > 0 ? w.n_channels : 1;
    const int32_t nframes = (int32_t) (w.pcm_i16.size() / (size_t) nch);
    ref_pcm.assign((size_t) nframes, 0.0f);
    for (int32_t i = 0; i < nframes; ++i) {
        float acc = 0.0f;
        for (int32_t c = 0; c < nch; ++c) acc += w.pcm_i16[(size_t) i * nch + c] / 32768.0f;
        ref_pcm[(size_t) i] = acc / (float) nch;
    }
    in.ref_pcm = ref_pcm.data();
    in.ref_n_samples = (int32_t) ref_pcm.size();
    in.ref_sample_rate = w.sample_rate;
    return true;
}

// Replace all occurrences of `from` with `to` in `s`.
static std::string replace_all_str(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Tokenize prefix + text + suffix through the backbone vocab.
// Tokenize a raw string with explicit bos/special controls.
std::vector<llama_token> tokenize_str(const llama_vocab * vocab,
                                      const std::string & s,
                                      bool add_bos, bool parse_special) {
    int32_t cap = (int32_t) s.size() + 8;
    std::vector<llama_token> toks(cap);
    int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                               toks.data(), cap, add_bos, parse_special);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                               toks.data(), (int32_t) toks.size(), add_bos, parse_special); }
    toks.resize(std::max(0, n));
    return toks;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab,
                                         const codec_common::audio_lm_prompt_info & pi,
                                         const std::string & text_in) {
    std::string text = text_in;
    // MOSS-TTSD dialogue tags: the processor maps [S1]/[S2] → <speaker1>/
    // <speaker2> before tokenizing (see processing_moss_ttsd prepare_sample).
    if (pi.model_kind == codec_common::audio_lm_prompt_info::KIND_PARALLEL_HEADS_DELAY) {
        text = replace_all_str(text, "[S1]", "<speaker1>");
        text = replace_all_str(text, "[S2]", "<speaker2>");
    }
    const std::string full = pi.prompt_prefix + text + pi.prompt_suffix;
    int32_t cap = (int32_t) full.size() + 8;
    std::vector<llama_token> toks(cap);
    int32_t n = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(),
                               toks.data(), cap, pi.add_bos, pi.parse_special);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(),
                               toks.data(), (int32_t) toks.size(), pi.add_bos, pi.parse_special); }
    toks.resize(std::max(0, n));
    return toks;
}

// Decode a token batch, requesting per-position embeddings (logits at
// every position when `all_pos` else only last).  Returns pointer to the
// last position's hidden after decode (owned by ctx).
bool decode_tokens(llama_context * lctx, const std::vector<llama_token> & toks,
                   int32_t n_past, bool all_pos) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = n_past + (int32_t) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (all_pos || i == toks.size() - 1) ? 1 : 0;
    }
    b.n_tokens = (int32_t) toks.size();
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Decode a single embedding vector (inputs_embeds path).
bool decode_embed(llama_context * lctx, const float * embd, int32_t dim, int32_t n_past) {
    llama_batch b = llama_batch_init(1, dim, 1);
    std::memcpy(b.embd, embd, (size_t) dim * sizeof(float));
    b.token = nullptr;
    b.pos[0] = n_past;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;
    b.n_tokens = 1;
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Flow 1 — continuous CFM (BlueMagpie).  Logits-all prefill → text_prefill
// → per-step observe_hidden ↔ feedback embed inject.
bool run_continuous(codec_common::audio_lm_context * ctx, llama_context * lctx,
                    const std::vector<llama_token> & toks, int32_t hidden,
                    int32_t max_frames, const args & a,
                    int32_t * out_frames, const char ** out_stop) {
    codec_common::audio_lm_set_continuous_params(ctx, a.cont_cfg, a.cont_timesteps, a.cont_min_len);
    if (!decode_tokens(lctx, toks, 0, /*all_pos=*/true)) {
        std::fprintf(stderr, "prefill decode failed\n"); return false;
    }
    const int32_t np = (int32_t) toks.size();
    std::vector<float> hid((size_t) np * hidden);
    for (int32_t i = 0; i < np; ++i) {
        const float * h = llama_get_embeddings_ith(lctx, i);
        if (!h) { std::fprintf(stderr, "no embeddings at pos %d\n", i); return false; }
        std::memcpy(hid.data() + (size_t) i * hidden, h, (size_t) hidden * sizeof(float));
    }
    int32_t n_past = np;
    if (!codec_common::audio_lm_text_prefill(ctx, hid.data(), np, hidden)) return false;

    std::vector<float> cur(hid.end() - hidden, hid.end());
    for (int32_t step = 0; step < max_frames; ++step) {
        auto act = codec_common::audio_lm_observe_hidden(ctx, cur.data(), hidden, nullptr);
        if (act == codec_common::OBSERVE_STOP) {
            const char * e = codec_common::audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "stop_head"; break;
        }
        (*out_frames)++;
        int32_t dim = 0;
        const float * fb = codec_common::audio_lm_get_next_embed(ctx, &dim);
        if (!fb || dim != hidden) return false;
        std::vector<float> fbc(fb, fb + dim);
        if (!decode_embed(lctx, fbc.data(), dim, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// Decode a contiguous block of `n` inputs_embeds rows (single sequence),
// flagging only the last for logits.  Rows are `n * dim` floats.
bool decode_embed_block(llama_context * lctx, const float * embds, int32_t dim,
                        int32_t n, int32_t n_past) {
    if (n <= 0) return true;
    llama_batch b = llama_batch_init(n, dim, 1);
    std::memcpy(b.embd, embds, (size_t) n * dim * sizeof(float));
    b.token = nullptr;
    for (int32_t i = 0; i < n; ++i) {
        b.pos[i] = n_past + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (i == n - 1) ? 1 : 0;
    }
    b.n_tokens = n;
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Flow 3-streaming — MOSS-TTS-Realtime.  The reference is a STREAMING model
// (streaming_mossttsrealtime.py): text is interleaved one token per audio
// frame rather than fed all at prefill.  Per backbone step the input embed
// is  text_embd[text_token] + compose_audio_embd(prev_frame_codes)  where
// the text table lives in the backbone (TextEmbdTable) and the audio sum in
// the codec_lm.  The spoken text goes in the ASSISTANT turn.
//
// Phases (prefill_text_len = 12 by default):
//   1. Prefill the system+user+assistant-open context (text lane = token,
//      audio lanes = audio_pad → zero embed), then the first
//      min(12, |text|) spoken-text rows (audio lanes = audio_pad; the LAST
//      prefill row's cb0 lane = audio BOS).  → 1 audio frame.
//   2. Stream: one spoken-text token per step, summed with the previous
//      frame's audio compose.  → 1 frame/step.
//   3. Drain: text lane = text_pad once text is exhausted, until cb0 == EOS
//      or max_frames.
bool run_realtime_streaming(codec_common::audio_lm_context * ctx,
                            llama_context * lctx, const llama_vocab * vocab,
                            const codec_common::audio_lm_prompt_info & pi,
                            const TextEmbdTable & tetab,
                            const std::string & payload_text,
                            int32_t hidden, int32_t n_cb, int32_t max_frames,
                            Sampler & sampler, float temp, float top_p,
                            int32_t top_k, float rep_penalty, int32_t rep_window,
                            int32_t * out_frames, const char ** out_stop) {
    codec_common::audio_lm_set_uses_embed_override(ctx, true, 1);

    // ── Tokenize context (system + empty user + assistant opener) and the
    //    spoken text (assistant lane, streamed). ────────────────────────
    std::vector<llama_token> ctx_toks =
        tokenize_str(vocab, pi.prompt_prefix + pi.prompt_suffix,
                     pi.add_bos, pi.parse_special);
    std::vector<llama_token> text_toks =
        tokenize_str(vocab, payload_text, /*add_bos=*/false, /*parse_special=*/false);
    if (ctx_toks.empty() || text_toks.empty()) {
        std::fprintf(stderr, "realtime: empty context or text tokens\n");
        return false;
    }

    const int32_t audio_pad = pi.audio_pad_code;
    const int32_t bos_c0    = pi.bos_code_c0;
    const int32_t text_pad  = pi.text_pad_id;
    const int32_t prefill_n = std::min<int32_t>(pi.prefill_text_len,
                                                (int32_t) text_toks.size());

    // Compose one row: text_embd[text_tok] + compose_audio_embd(codes).
    std::vector<int32_t> pad_codes((size_t) n_cb, audio_pad);
    auto compose_row = [&](int32_t text_tok, const int32_t * codes,
                           float * dst) -> bool {
        if (!tetab.row(text_tok, dst)) {
            std::fprintf(stderr, "realtime: text_embd row %d failed\n", text_tok);
            return false;
        }
        std::vector<float> aud((size_t) hidden, 0.0f);
        if (!codec_common::audio_lm_compose_audio_codes_embd(
                ctx, codes, n_cb, aud.data(), hidden)) {
            std::fprintf(stderr, "realtime: compose_audio failed: %s\n",
                         codec_common::audio_lm_last_error(ctx));
            return false;
        }
        for (int32_t i = 0; i < hidden; ++i) dst[i] += aud[i];
        return true;
    };

    // ── Build the prefill block: context rows + prefill_n spoken-text rows.
    const int32_t n_rows = (int32_t) ctx_toks.size() + prefill_n;
    std::vector<float> block((size_t) n_rows * hidden);
    int32_t r = 0;
    for (size_t i = 0; i < ctx_toks.size(); ++i, ++r) {
        // context: audio lanes are all pad → zero embed contribution.
        if (!compose_row(ctx_toks[i], pad_codes.data(),
                         block.data() + (size_t) r * hidden)) return false;
    }
    for (int32_t i = 0; i < prefill_n; ++i, ++r) {
        // last prefill row opens the audio channel: cb0 lane = BOS.
        std::vector<int32_t> codes = pad_codes;
        if (i == prefill_n - 1) codes[0] = bos_c0;
        if (!compose_row(text_toks[(size_t) i], codes.data(),
                         block.data() + (size_t) r * hidden)) return false;
    }
    if (!decode_embed_block(lctx, block.data(), hidden, n_rows, 0)) {
        std::fprintf(stderr, "realtime: prefill decode failed\n");
        return false;
    }
    int32_t n_past = n_rows;

    std::vector<float> cur(hidden);
    {
        const float * h0 = llama_get_embeddings_ith(lctx, -1);
        if (!h0) return false;
        std::memcpy(cur.data(), h0, (size_t) hidden * sizeof(float));
    }

    // Per-codebook history for the CTRL repetition penalty (last frames).
    std::vector<std::vector<int32_t>> hist((size_t) n_cb);

    // text_idx = next spoken-text token to feed (prefill consumed [0,prefill_n)).
    int32_t text_idx = prefill_n;
    std::vector<int32_t> codes(n_cb);
    for (int32_t step = 0; step < max_frames; ++step) {
        // ── Depth-decode the frame from the current backbone hidden. ─────
        if (!codec_common::audio_lm_step_begin(ctx, cur.data(), hidden)) return false;
        for (int32_t cb = 0; cb < n_cb; ++cb) {
            int32_t idx = 0, nlog = 0;
            const float * lg = codec_common::audio_lm_step_logits(ctx, &idx, &nlog);
            if (!lg) return false;
            int32_t code;
            if (rep_penalty == 1.0f) {
                code = sampler.sample(lg, nlog, temp, top_p, top_k);
            } else {
                // CTRL-style penalty per codebook over its own last-window
                // history, on a logits copy (mirrors the reference).
                std::vector<float> lc(lg, lg + nlog);
                const std::vector<int32_t> & h = hist[(size_t) cb];
                const int32_t hn = (int32_t) h.size();
                const int32_t w  = (rep_window > 0 && rep_window < hn) ? rep_window : hn;
                Sampler::apply_rep_penalty(lc.data(), nlog,
                                           h.data() + (hn - w), w, rep_penalty);
                code = sampler.sample(lc.data(), nlog, temp, top_p, top_k);
            }
            if (!codec_common::audio_lm_step_push_code(ctx, code)) return false;
        }
        if (!codec_common::audio_lm_step_finish(ctx, codes.data(), n_cb)) return false;

        // ── Accumulate + EOS check + compose the prev-frame audio embed. ──
        auto act = codec_common::audio_lm_observe_codes(ctx, codes.data(), n_cb,
                                                        cur.data(), hidden);
        if (act == codec_common::OBSERVE_STOP) {
            const char * e = codec_common::audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "eos_code_c0";
            break;
        }
        (*out_frames)++;
        for (int32_t cb = 0; cb < n_cb; ++cb) hist[(size_t) cb].push_back(codes[cb]);

        // ── Next backbone input = text_embd[next_text] + audio(prev frame).
        int32_t text_tok = (text_idx < (int32_t) text_toks.size())
                         ? text_toks[(size_t) text_idx] : text_pad;
        ++text_idx;
        std::vector<float> row(hidden);
        if (!compose_row(text_tok, codes.data(), row.data())) return false;
        if (!decode_embed(lctx, row.data(), hidden, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// Flow 5 — LFM2-Audio sequential text→audio TTS.  Prefill the ChatML prompt
// (token path), free-run TEXT modality on the backbone's tied-embedding
// lm_head until <|audio_start|>, then switch to AUDIO_OUT: depth-decode
// 8-codebook Mimi frames, feed each frame back through
// compose_audio_codes_embd, and stop on cb0 == EOAudio or <|im_end|>.
// Mirrors liquid_audio.LFM2AudioModel.generate_sequential.
bool run_lfm2_sequential(codec_common::audio_lm_context * ctx, llama_context * lctx,
                         const llama_vocab * vocab,
                         const codec_common::audio_lm_prompt_info & pi,
                         const TextEmbdTable & tetab,
                         const std::vector<llama_token> & toks, int32_t hidden,
                         int32_t n_cb, int32_t max_frames, Sampler & sampler,
                         float temp, float top_p, int32_t top_k,
                         int32_t * out_frames, const char ** out_stop) {
    codec_common::audio_lm_set_uses_embed_override(ctx, true, 1);

    // ── Prefill the ChatML prompt via the raw token path. ────────────────
    if (!decode_tokens(lctx, toks, 0, /*all_pos=*/false)) {
        std::fprintf(stderr, "lfm2: prefill decode failed\n");
        return false;
    }
    int32_t n_past = (int32_t) toks.size();
    const int32_t n_vocab = (int32_t) tetab.vocab;

    // Text logits from the current backbone hidden.  llama.cpp does NOT build
    // the lm_head when embeddings are enabled (lfm2.cpp: `if
    // (!cparams.embeddings)`), and we need embeddings for the depth decoder.
    // The reference samples text via `linear(last_hidden, embed_tokens.weight)`
    // — since LFM2 ties the output head to the input embedding, we reproduce
    // it as hidden · token_embd[row] over the full vocab (the returned
    // embeddings are already post-final-norm, matching HF's last_hidden_state).
    std::vector<float> tlog((size_t) n_vocab);
    std::vector<float> erow((size_t) hidden);
    auto text_logits = [&](const float * h) -> const float * {
        for (int32_t v = 0; v < n_vocab; ++v) {
            if (!tetab.row(v, erow.data())) { tlog[v] = -1e30f; continue; }
            double acc = 0.0;
            for (int32_t i = 0; i < hidden; ++i) acc += (double) h[i] * (double) erow[i];
            tlog[v] = (float) acc;
        }
        return tlog.data();
    };

    // ── Phase 1: TEXT warmup.  Sample text tokens from the tied lm_head
    //    until the model emits <|audio_start|>.  A cap guards runaway text —
    //    with the "Perform TTS." system prompt the model emits <|audio_start|>
    //    as its very first generated token, so this loop runs once. ──
    (void) vocab;
    for (int32_t t = 0; t < pi.max_text_tokens; ++t) {
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) { std::fprintf(stderr, "lfm2: no hidden for text logits\n"); return false; }
        const float * bl = text_logits(h);
        int32_t tok = sampler.sample(bl, n_vocab, temp, top_p, top_k);
        if (tok == pi.audio_start_id) break;
        if (tok == pi.text_end_id)    { *out_stop = "text_end"; return true; }
        std::vector<llama_token> one(1, (llama_token) tok);
        if (!decode_tokens(lctx, one, n_past++, /*all_pos=*/false)) {
            std::fprintf(stderr, "lfm2: text step decode failed\n");
            return false;
        }
    }
    {
        // Feed <|audio_start|> so the first audio frame is conditioned on it
        // (reference embeds audio_start before switching to AUDIO_OUT).
        std::vector<llama_token> as(1, (llama_token) pi.audio_start_id);
        if (!decode_tokens(lctx, as, n_past++, /*all_pos=*/false)) return false;
    }

    // ── Phase 2: AUDIO_OUT.  One depth-decoded N-codebook frame per step. ──
    std::vector<float> cur(hidden);
    const float * h0 = llama_get_embeddings_ith(lctx, -1);
    if (!h0) return false;
    std::memcpy(cur.data(), h0, (size_t) hidden * sizeof(float));

    std::vector<int32_t> codes(n_cb);
    for (int32_t step = 0; step < max_frames; ++step) {
        if (!codec_common::audio_lm_step_begin(ctx, cur.data(), hidden)) return false;
        for (int32_t cb = 0; cb < n_cb; ++cb) {
            int32_t idx = 0, nlog = 0;
            const float * lg = codec_common::audio_lm_step_logits(ctx, &idx, &nlog);
            if (!lg) return false;
            int32_t code = sampler.sample(lg, nlog, temp, top_p, top_k);
            if (!codec_common::audio_lm_step_push_code(ctx, code)) return false;
        }
        if (!codec_common::audio_lm_step_finish(ctx, codes.data(), n_cb)) return false;

        // EOAudio: cb0 == eos_code_c0 (2048).  observe_codes handles the
        // accumulate + EOS check; STOP means the frame was the terminator.
        auto act = codec_common::audio_lm_observe_codes(ctx, codes.data(), n_cb,
                                                        cur.data(), hidden);
        if (act == codec_common::OBSERVE_STOP) {
            const char * e = codec_common::audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "eos_code_c0";
            break;
        }
        (*out_frames)++;

        // Next backbone input = compose_audio_codes_embd(frame) — the
        // reference's audio_embedding(codes + codebook_offsets).sum(0).
        std::vector<float> row(hidden, 0.0f);
        if (!codec_common::audio_lm_compose_audio_codes_embd(
                ctx, codes.data(), n_cb, row.data(), hidden)) {
            std::fprintf(stderr, "lfm2: compose_audio failed: %s\n",
                         codec_common::audio_lm_last_error(ctx));
            return false;
        }
        if (!decode_embed(lctx, row.data(), hidden, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// Flow 2/3 — codebook AR.  Prefill → per-step { [cb0 from backbone] +
// step machine cb0..N-1 } → observe_codes (accumulate + EOS + compose) →
// inject next embed.
bool run_codebook_ar(codec_common::audio_lm_context * ctx, llama_context * lctx,
                     const llama_vocab * vocab,
                     const codec_common::audio_lm_prompt_info & pi,
                     const std::vector<llama_token> & toks, int32_t hidden, int32_t n_cb,
                     int32_t max_frames, Sampler & sampler,
                     float temp, float top_p, int32_t top_k,
                     const std::vector<float> & speaker_prefix,
                     const std::string & payload_text,
                     int32_t * out_frames, const char ** out_stop) {
    codec_common::audio_lm_set_uses_embed_override(ctx, true, 1);
    int32_t n_past = 0;

    // ── Qwen3-TTS talker: faithful additive dual-lane prompt ───────────
    // When the model carries the talker text_projection + control tags, the
    // prompt is not raw ChatML tokens: it's a composed inputs_embeds prefix
    // (projected text lane + codec control-tag lane, with the ECAPA x-vector
    // inserted between the think-tags and pad/bos), and the trailing text is
    // injected per-step summed with the audio next-embed (delay-pattern).
    std::vector<llama_token> talker_text;   // payload text tokens (for trailing)
    int32_t talker_trailing = 0;            // index into trailing text stream
    const bool talker = codec_common::audio_lm_talker_has_projection(ctx);
    if (talker) {
        // Role header = "<|im_start|>assistant\n" (projected text, no codec).
        std::vector<llama_token> role =
            tokenize_str(vocab, "<|im_start|>assistant\n", /*add_bos=*/false,
                         /*parse_special=*/true);
        talker_text = tokenize_str(vocab, payload_text,
                                   /*add_bos=*/false, /*parse_special=*/false);
        if (talker_text.empty()) { std::fprintf(stderr, "talker: empty text\n"); return false; }

        const int32_t cap_rows = (int32_t) role.size() + 6 + 4;
        std::vector<float> prefix((size_t) cap_rows * hidden);
        int32_t n_rows = 0, consumed = 0;
        const float * xv = (!speaker_prefix.empty() &&
                            (int32_t) speaker_prefix.size() == hidden)
                         ? speaker_prefix.data() : nullptr;
        if (!codec_common::audio_lm_build_talker_prefix(
                ctx, role.data(), (int32_t) role.size(),
                talker_text.data(), (int32_t) talker_text.size(),
                xv, xv ? hidden : 0,
                prefix.data(), cap_rows, &n_rows, &consumed)) {
            std::fprintf(stderr, "build_talker_prefix failed: %s\n",
                         codec_common::audio_lm_last_error(ctx));
            return false;
        }
        talker_trailing = 0;   // trailing token 0 → text[1]
        llama_batch b = llama_batch_init(n_rows, hidden, 1);
        std::memcpy(b.embd, prefix.data(), (size_t) n_rows * hidden * sizeof(float));
        b.token = nullptr; b.n_tokens = n_rows;
        for (int32_t i = 0; i < n_rows; ++i) {
            b.pos[i] = n_past + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (i == n_rows - 1) ? 1 : 0;
        }
        int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "talker prefill decode failed\n"); return false; }
        n_past += n_rows;
    } else
    // Speaker x-vector prefix (non-talker fallback): prepend as row 0.
    if (!speaker_prefix.empty() && (int32_t) speaker_prefix.size() == hidden) {
        std::vector<float> pfx(speaker_prefix);
        if (!decode_embed(lctx, pfx.data(), hidden, n_past)) {
            std::fprintf(stderr, "speaker prefix decode failed\n"); return false;
        }
        n_past += 1;
    }
    if (!talker) {
    // Merged-cb0 models (MOSS-TTSD) need the multi-modal prompt embedding:
    // each position is sum(embed_cb0[text_tok], embed_cb1..7[speech_pad]).
    // Feed those via the inputs_embeds path instead of the raw token path so
    // the backbone sees the same prefill the HF processor produces.
    if (codec_common::audio_lm_prompt_needs_composed_embd(ctx)) {
        std::vector<float> prompt_embd((size_t) toks.size() * hidden);
        for (size_t i = 0; i < toks.size(); ++i) {
            if (!codec_common::audio_lm_compose_prompt_embd(
                    ctx, toks[i], prompt_embd.data() + i * hidden, hidden)) {
                std::fprintf(stderr, "compose_prompt_embd failed: %s\n",
                             codec_common::audio_lm_last_error(ctx));
                return false;
            }
        }
        llama_batch b = llama_batch_init((int32_t) toks.size(), hidden, 1);
        std::memcpy(b.embd, prompt_embd.data(), prompt_embd.size() * sizeof(float));
        b.token = nullptr;
        b.n_tokens = (int32_t) toks.size();
        for (size_t i = 0; i < toks.size(); ++i) {
            b.pos[i] = n_past + (int32_t) i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == toks.size() - 1) ? 1 : 0;
        }
        int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "prefill (composed) decode failed\n"); return false; }
    } else if (!decode_tokens(lctx, toks, n_past, /*all_pos=*/false)) {
        std::fprintf(stderr, "prefill decode failed\n"); return false;
    }
    n_past += (int32_t) toks.size();
    }  // end if (!talker)

    std::vector<float> cur(hidden);
    const float * h0 = llama_get_embeddings_ith(lctx, -1);
    if (!h0) return false;
    std::memcpy(cur.data(), h0, (size_t) hidden * sizeof(float));

    std::vector<int32_t> codes(n_cb);
    for (int32_t step = 0; step < max_frames; ++step) {
        if (pi.cb0_from_backbone) {
            const float * bl = llama_get_logits_ith(lctx, -1);
            if (!bl) return false;
            int32_t nv = llama_vocab_n_tokens(vocab);
            int32_t c0 = sampler.sample(bl, nv, temp, top_p, top_k);
            if (!codec_common::audio_lm_step_set_text_context(ctx, c0)) return false;
            codes[0] = c0;
        }
        if (!codec_common::audio_lm_step_begin(ctx, cur.data(), hidden)) return false;
        for (int32_t cb = 0; cb < n_cb; ++cb) {
            int32_t idx = 0, nlog = 0;
            const float * lg = codec_common::audio_lm_step_logits(ctx, &idx, &nlog);
            if (!lg) return false;
            int32_t code = (pi.cb0_from_backbone && cb == 0)
                         ? codes[0]
                         : sampler.sample(lg, nlog, temp, top_p, top_k);
            if (!codec_common::audio_lm_step_push_code(ctx, code)) return false;
        }
        if (!codec_common::audio_lm_step_finish(ctx, codes.data(), n_cb)) return false;

        auto act = codec_common::audio_lm_observe_codes(ctx, codes.data(), n_cb, cur.data(), hidden);
        if (act == codec_common::OBSERVE_STOP) {
            const char * e = codec_common::audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "eos_code_c0"; break;
        }
        (*out_frames)++;
        int32_t dim = 0;
        const float * nb = codec_common::audio_lm_get_next_embed(ctx, &dim);
        if (!nb || dim != hidden) return false;
        std::vector<float> nbc(nb, nb + dim);
        // Qwen3-TTS talker: the next backbone input is the audio next-embed
        // SUMMED with the projected trailing text token (delay-pattern text
        // injection, mirrors the reference trailing_text_hidden add).
        if (talker) {
            std::vector<float> tt(hidden);
            if (!codec_common::audio_lm_talker_trailing_text_embd(
                    ctx, talker_text.data(), (int32_t) talker_text.size(),
                    talker_trailing, tt.data(), hidden)) {
                std::fprintf(stderr, "talker trailing text failed: %s\n",
                             codec_common::audio_lm_last_error(ctx));
                return false;
            }
            for (int32_t i = 0; i < hidden; ++i) nbc[i] += tt[i];
            ++talker_trailing;
        }
        if (!decode_embed(lctx, nbc.data(), dim, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// T3-faithful per-step sampler over the CFG-combined speech logits.
// Mirrors t3.py inference: repetition_penalty → /temperature → min_p →
// top_p → multinomial.  `generated` is the running list of already-
// sampled speech ids (incl BOS) used for the repetition penalty.
int32_t sample_t3(std::mt19937_64 & rng, std::vector<float> logits,
                  const std::vector<int32_t> & generated,
                  float temperature, float min_p, float top_p,
                  float repetition_penalty) {
    const int32_t n = (int32_t) logits.size();
    if (n <= 0) return 0;

    // temperature<=0 → greedy argmax (parity/debug path).
    if (temperature <= 0.0f) {
        int32_t best = 0; for (int32_t i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }

    // 1. Repetition penalty (HF RepetitionPenaltyLogitsProcessor).
    if (repetition_penalty != 1.0f) {
        for (int32_t id : generated) {
            if (id < 0 || id >= n) continue;
            float & l = logits[id];
            l = (l > 0.0f) ? (l / repetition_penalty) : (l * repetition_penalty);
        }
    }
    // 2. Temperature.
    if (temperature > 0.0f && temperature != 1.0f) {
        for (float & l : logits) l /= temperature;
    }
    // Softmax → probs.
    float mx = logits[0];
    for (int32_t i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
    std::vector<float> probs(n);
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) { double e = std::exp((double) logits[i] - mx); probs[i] = (float) e; sum += e; }
    for (float & p : probs) p = (float) (p / sum);

    // 3. min_p filter: keep tokens with prob >= min_p * max_prob.
    if (min_p > 0.0f) {
        float pmax = 0.0f;
        for (float p : probs) pmax = std::max(pmax, p);
        const float thresh = min_p * pmax;
        for (int32_t i = 0; i < n; ++i) if (probs[i] < thresh) probs[i] = 0.0f;
    }
    // 4. top_p (nucleus): sort desc, keep smallest set with cumsum >= top_p.
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int32_t> idx(n);
        for (int32_t i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int32_t x, int32_t y){ return probs[x] > probs[y]; });
        double c = 0.0; bool cut = false;
        for (int32_t r = 0; r < n; ++r) {
            if (cut) { probs[idx[r]] = 0.0f; continue; }
            c += probs[idx[r]];
            if (c >= top_p) cut = true;   // keep this one, drop the rest
        }
    }
    // Renormalise + multinomial sample.
    double z = 0.0; for (float p : probs) z += p;
    if (z <= 0.0) {  // degenerate → argmax of original logits
        int32_t best = 0; for (int32_t i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }
    std::uniform_real_distribution<double> u(0.0, z);
    double r = u(rng), acc = 0.0;
    for (int32_t i = 0; i < n; ++i) { acc += probs[i]; if (r <= acc) return i; }
    return n - 1;
}

// Decode a batch of `n_seq` embed rows (one per CFG lane) at a single
// position, requesting logits at the last (only) row of each lane.
// Rows are laid out [seq0_row | seq1_row]; each lane is its own llama seq.
bool decode_embed_batch(llama_context * lctx, const float * embds, int32_t dim,
                        int32_t n_seq, int32_t pos) {
    llama_batch b = llama_batch_init(n_seq, dim, 1);
    std::memcpy(b.embd, embds, (size_t) n_seq * dim * sizeof(float));
    b.token = nullptr;
    b.n_tokens = n_seq;
    for (int32_t s = 0; s < n_seq; ++s) {
        b.pos[s] = pos;
        b.n_seq_id[s] = 1;
        b.seq_id[s][0] = s;
        b.logits[s] = 1;
    }
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Flow 4 — Chatterbox T3.  Prompt = [cond_enc | text_emb+pos | BOS],
// batched over 1 (no CFG) or 2 (CFG) llama sequences; per-step CFG-blended
// speech_head logits → sample → speech_emb[code]+speech_pos_emb[step+1].
bool run_chatterbox(codec_common::audio_lm_context * ctx, llama_context * lctx,
                    codec_lm * lm, const codec_lm_chatterbox_info * ci,
                    const args & a, int32_t hidden, int32_t max_frames,
                    std::vector<int32_t> * out_codes,
                    int32_t * out_frames, const char ** out_stop) {
    const float cfg_weight = a.cfg_weight ? *a.cfg_weight : 0.5f;
    const float temperature = a.temp ? *a.temp : 0.8f;
    const float top_p = a.top_p ? *a.top_p : 1.0f;
    const float min_p = a.min_p ? *a.min_p : 0.05f;
    const float rep_pen = a.repetition_penalty ? *a.repetition_penalty : 1.2f;

    // Tokenize the text with the baked EnTokenizer.
    std::vector<int32_t> text_ids(a.text.size() + 64);
    int32_t n_text = 0;
    if (codec_lm_chatterbox_tokenize(lm, a.text.c_str(), text_ids.data(),
                                     (int32_t) text_ids.size(), &n_text) != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "chatterbox tokenize failed: %s\n", codec_lm_get_last_error(lm));
        return false;
    }
    text_ids.resize(n_text);
    std::printf("chatterbox: %d text tokens, cfg_weight=%.2f temp=%.2f min_p=%.2f top_p=%.2f rep=%.2f\n",
                n_text, cfg_weight, temperature, min_p, top_p, rep_pen);

    // Ref-audio conditioning: when --ref-audio is given, run VE+cond_enc
    // from the waveform (cond-prompt speech tokens fall back to builtin).
    // Otherwise use the builtin baked conds.
    std::vector<float> ref_pcm;
    const float * ref_pcm_ptr = nullptr;
    int32_t ref_n = 0, ref_sr = 0;
    if (!a.ref_audio.empty()) {
        codec_common::audio_lm_input in;
        in.text = a.text;
        if (!load_ref_audio(a, in, ref_pcm)) return false;
        if (in.ref_pcm != nullptr) {
            // The Chatterbox VE expects 16 kHz mono; linearly resample.
            const int32_t target_sr = 16000;
            if (in.ref_sample_rate != target_sr && in.ref_sample_rate > 0) {
                const int32_t n_in = (int32_t) ref_pcm.size();
                const int64_t n_out = (int64_t) n_in * target_sr / in.ref_sample_rate;
                std::vector<float> rs((size_t) n_out);
                for (int64_t i = 0; i < n_out; ++i) {
                    double src = (double) i * in.ref_sample_rate / target_sr;
                    int64_t i0 = (int64_t) src;
                    double f = src - i0;
                    float a0 = ref_pcm[(size_t) std::min<int64_t>(i0, n_in - 1)];
                    float a1 = ref_pcm[(size_t) std::min<int64_t>(i0 + 1, n_in - 1)];
                    rs[(size_t) i] = (float) (a0 * (1.0 - f) + a1 * f);
                }
                ref_pcm.swap(rs);
            }
            ref_pcm_ptr = ref_pcm.data();
            ref_n = (int32_t) ref_pcm.size();
            ref_sr = 16000;
            std::printf("chatterbox: using ref audio %s (%d samples @ 16000 Hz after resample)\n",
                        a.ref_audio.c_str(), ref_n);
        }
    }

    const int32_t cond_rows = ci->cond_rows;
    const int32_t seq_len_cap = cond_rows + (n_text + 2) + 2;
    const int32_t n_seq_cap = (cfg_weight > 0.0f) ? 2 : 1;
    std::vector<float> prompt((size_t) seq_len_cap * n_seq_cap * hidden);
    int32_t seq_len = 0, n_seq = 0;
    if (codec_lm_chatterbox_build_prompt(
            lm, text_ids.data(), n_text, cfg_weight,
            nullptr, 0, nullptr, 0, nullptr,
            ref_pcm_ptr, ref_n, ref_sr,
            prompt.data(), seq_len_cap * n_seq_cap, &seq_len, &n_seq) != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "chatterbox build_prompt failed: %s\n", codec_lm_get_last_error(lm));
        return false;
    }
    std::printf("chatterbox: prompt seq_len=%d n_seq=%d (%d rows total)\n",
                seq_len, n_seq, seq_len * n_seq);

    // Prefill: feed each lane's rows as its own llama sequence.  Logits
    // requested only at the last row of each lane.
    {
        const int32_t total = seq_len * n_seq;
        llama_batch b = llama_batch_init(total, hidden, 1);
        b.token = nullptr;
        b.n_tokens = total;
        int32_t bi = 0;
        for (int32_t s = 0; s < n_seq; ++s) {
            for (int32_t r = 0; r < seq_len; ++r) {
                std::memcpy(b.embd + (size_t) bi * hidden,
                            prompt.data() + ((size_t) s * seq_len + r) * hidden,
                            (size_t) hidden * sizeof(float));
                b.pos[bi] = r;
                b.n_seq_id[bi] = 1;
                b.seq_id[bi][0] = s;
                b.logits[bi] = (r == seq_len - 1) ? 1 : 0;
                ++bi;
            }
        }
        int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "chatterbox prefill decode failed\n"); return false; }
    }

    const int32_t V = ci->speech_vocab_size;
    std::mt19937_64 rng(a.seed ? a.seed : 0xC0DEC1ABull);
    std::vector<int32_t> generated;       // sampled ids incl BOS (for rep penalty)
    generated.push_back(ci->start_speech_token);
    int32_t n_past = seq_len;             // both lanes share the same length

    // The backbone's own lm_head is a vocab=8 placeholder — the real speech
    // logits come from applying the T3 speech_head (lm.heads_0) to the
    // backbone hidden.  codec_common's step machine does exactly that.
    // Lane batch indices for the last row: lane s's last-row output is the
    // (s+1)-th flagged output → read via llama_get_embeddings_ith at the
    // last row's batch position.
    auto lane_hidden = [&](int32_t lane) -> const float * {
        // Last row of lane `lane`: batch index lane*seq_len + (seq_len-1)
        // on prefill; on per-step decode each lane contributes one row so
        // the index is just `lane` there.  We pass the absolute output
        // index; llama maps it via output_ids.
        return llama_get_embeddings_ith(lctx, -(n_seq - lane));
    };
    // On prefill both lanes' last rows are the only flagged outputs, in
    // lane order → lane 0 = ith(-2 or -1), lane 1 = ith(-1).  After a
    // per-step decode of n_seq rows (all flagged) the same holds.
    auto speech_logits = [&](const float * h, std::vector<float> * out) -> bool {
        if (!codec_common::audio_lm_step_begin(ctx, h, hidden)) return false;
        int32_t cb = 0, nlog = 0;
        const float * lg = codec_common::audio_lm_step_logits(ctx, &cb, &nlog);
        if (!lg || nlog <= 0) return false;
        out->assign(lg, lg + nlog);
        // Close the step without accumulating (we manage codes ourselves).
        codec_common::audio_lm_step_push_code(ctx, 0);
        int32_t dummy = 0;
        codec_common::audio_lm_step_finish(ctx, &dummy, 1);
        return true;
    };

    for (int32_t step = 0; step < max_frames; ++step) {
        const float * hc = lane_hidden(0);
        const float * hu = (n_seq == 2) ? lane_hidden(1) : nullptr;
        if (!hc) { std::fprintf(stderr, "no hidden at step %d\n", step); return false; }
        std::vector<float> cond, uncond;
        if (!speech_logits(hc, &cond)) {
            std::fprintf(stderr, "speech_head (cond) failed: %s\n", codec_common::audio_lm_last_error(ctx));
            return false;
        }
        if (hu && !speech_logits(hu, &uncond)) {
            std::fprintf(stderr, "speech_head (uncond) failed\n"); return false;
        }
        const int32_t VV = (int32_t) cond.size();
        std::vector<float> logits(VV);
        for (int32_t i = 0; i < VV; ++i)
            logits[i] = hu ? (cond[i] + cfg_weight * (cond[i] - uncond[i])) : cond[i];
        int32_t code = sample_t3(rng, std::move(logits), generated,
                                 temperature, min_p, top_p, rep_pen);
        (void) V;
        generated.push_back(code);
        if (code == ci->stop_speech_token) { *out_stop = "eos_code_c0"; break; }
        // Real speech codes are < start_speech_token (6561); ignore control
        // codes in the output stream (drop_invalid_tokens + <6561 filter).
        if (code < ci->start_speech_token) out_codes->push_back(code);
        (*out_frames)++;

        // Next backbone input: speech_emb[code] + speech_pos_emb[step+1],
        // fed to both lanes.
        std::vector<float> nb(hidden);
        if (codec_lm_chatterbox_compose_speech_embd(lm, code, step + 1, nb.data(), hidden)
                != CODEC_STATUS_SUCCESS) {
            std::fprintf(stderr, "compose_speech_embd failed: %s\n", codec_lm_get_last_error(lm));
            return false;
        }
        std::vector<float> row((size_t) n_seq * hidden);
        for (int32_t s = 0; s < n_seq; ++s)
            std::memcpy(row.data() + (size_t) s * hidden, nb.data(), (size_t) hidden * sizeof(float));
        if (!decode_embed_batch(lctx, row.data(), hidden, n_seq, n_past)) {
            std::fprintf(stderr, "chatterbox step decode failed\n"); return false;
        }
        ++n_past;
    }
    return true;
}

}  // namespace

int cmd_synthesize(const args & a) {
    if (a.backbone.empty()) {
        std::fprintf(stderr, "synthesize requires --backbone PATH.gguf (llama.cpp model)\n");
        return 1;
    }
    if (a.output.empty()) {
        std::fprintf(stderr, "synthesize requires --output PATH.wav\n");
        return 1;
    }

    codec_common::audio_lm_params p;
    p.codec_path = a.model;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;
    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    codec_common::audio_lm_prompt_info pi;
    if (!codec_common::audio_lm_get_prompt_info(ctx, &pi)) {
        std::fprintf(stderr, "get_prompt_info failed: %s\n", codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 3;
    }
    const int32_t hidden = codec_common::audio_lm_hidden_dim(ctx);
    const int32_t n_cb   = codec_common::audio_lm_n_codebook(ctx);
    std::printf("model: arch=%s kind=%d n_cb=%d hidden=%d cb0_backbone=%d audio_offset=%d eos_c0=%d\n",
                pi.host_arch.c_str(), (int) pi.model_kind, n_cb, hidden,
                (int) pi.cb0_from_backbone, pi.audio_codebook_offset, pi.eos_code_c0);

    // codec_common's decode_audio now applies the model-specific codes→PCM
    // transform (cb0 text/control slice via audio_codebook_offset + delay-
    // pattern un-shift) for parallel-heads-delay + control-cb0 models, so
    // no family is refused here anymore.

    // ── Moshi: formally out of scope for one-shot synthesize ────────────
    // Moshi (host_arch=llama, residual_depth_ar, c0_input_modality=text, NO
    // eos_code_c0) is a full-duplex dialogue model: its backbone is Helium
    // (unsupported by the pinned llama.cpp submodule — there is no `helium`
    // arch), and its audio stream has no cb0 EOS to stop a one-shot loop on.
    // Its one-shot-TTS sibling is a different model (kyutai/dsm).  Refuse
    // with a concrete message rather than running a stop-less loop against an
    // unloadable backbone.  (CSM is also host_arch=llama but has eos_code_c0
    // and c0_input_modality=audio, so it is NOT caught here.)  See
    // docs/codec_common_api.md §"Moshi: why not in synthesize".
    if (pi.host_arch == "llama" &&
        pi.model_kind == codec_common::audio_lm_prompt_info::KIND_RESIDUAL_DEPTH_AR &&
        pi.eos_code_c0 < 0 && !pi.cb0_from_backbone) {
        std::fprintf(stderr,
            "synthesize: this looks like a Moshi codec_lm (full-duplex dialogue, "
            "no audio EOS, Helium backbone).  Moshi is not supported by "
            "`synthesize`: the pinned llama.cpp has no Helium arch, moshiko.gguf "
            "ships no backbone, and the duplex protocol has no one-shot stop "
            "condition.  Its one-shot-TTS sibling is kyutai/dsm.  See "
            "docs/codec_common_api.md.  The codec_lm side is still validated by "
            "tests/e2e/moshi_lm_smoke.py.\n");
        codec_common::audio_lm_free(ctx);
        return 10;
    }

    // ── Backbone init ──────────────────────────────────────────────
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * lmodel = llama_model_load_from_file(a.backbone.c_str(), mp);
    if (!lmodel) {
        std::fprintf(stderr, "llama_model_load_from_file failed: %s\n", a.backbone.c_str());
        codec_common::audio_lm_free(ctx); llama_backend_free();
        return 4;
    }
    const int32_t n_embd = llama_model_n_embd(lmodel);
    if (n_embd != hidden) {
        std::fprintf(stderr, "backbone n_embd=%d != codec hidden=%d — wrong backbone?\n", n_embd, hidden);
        llama_model_free(lmodel); codec_common::audio_lm_free(ctx); llama_backend_free();
        return 4;
    }
    // Chatterbox T3 detection: the codec_lm carries a `codec.lm.chatterbox.*`
    // section.  Its flow (embd prompt + CFG dual-sequence + own tokenizer)
    // diverges from the generic token-prompt flows, so route it separately.
    codec_lm * lm_handle = codec_common::audio_lm_get_lm(ctx);
    const codec_lm_chatterbox_info * cbx =
        lm_handle ? codec_lm_chatterbox_get_info(lm_handle) : nullptr;
    const bool is_chatterbox = (cbx != nullptr);
    const float cbx_cfg = a.cfg_weight ? *a.cfg_weight : 0.5f;
    const int32_t cbx_n_seq = (is_chatterbox && cbx_cfg > 0.0f) ? 2 : 1;

    const int32_t max_frames = a.max_frames > 0 ? a.max_frames : 512;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t) std::max(4096, max_frames + 512);
    cp.n_batch      = cp.n_ctx;
    cp.n_ubatch     = cp.n_ctx;
    cp.n_seq_max    = (uint32_t) cbx_n_seq;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (a.n_threads > 0) { cp.n_threads = a.n_threads; cp.n_threads_batch = a.n_threads; }
    llama_context * lctx = llama_init_from_model(lmodel, cp);
    if (!lctx) {
        std::fprintf(stderr, "llama_init_from_model failed\n");
        llama_model_free(lmodel); codec_common::audio_lm_free(ctx); llama_backend_free();
        return 4;
    }
    const llama_vocab * vocab = llama_model_get_vocab(lmodel);

    // ── Chatterbox T3 flow (Flow 4) ───────────────────────────────────
    if (is_chatterbox) {
        std::vector<int32_t> speech_codes;
        int32_t n_frames = 0;
        const char * stop_reason = "max_frames";
        bool ok = run_chatterbox(ctx, lctx, lm_handle, cbx, a, hidden, max_frames,
                                 &speech_codes, &n_frames, &stop_reason);
        llama_free(lctx);
        llama_model_free(lmodel);
        llama_backend_free();
        if (!ok) {
            std::fprintf(stderr, "chatterbox AR failed: %s\n", codec_common::audio_lm_last_error(ctx));
            codec_common::audio_lm_free(ctx);
            return 6;
        }
        std::printf("chatterbox AR done: %d frames, %zu speech codes, stop=%s\n",
                    n_frames, speech_codes.size(), stop_reason);
        if (speech_codes.empty()) {
            std::fprintf(stderr, "no speech codes generated\n");
            codec_common::audio_lm_free(ctx);
            return 7;
        }
        // The per-step speech_head calls ran through the step machine,
        // which accumulated placeholder codes into the context.  Clear
        // that before pushing the real speech codes for decode.
        codec_common::audio_lm_reset(ctx);
        // Push codes (n_q=1) and decode through Chatterbox-S3G.
        if (!codec_common::audio_lm_push_codes(ctx, speech_codes.data(),
                                               (int32_t) speech_codes.size(), 1)) {
            std::fprintf(stderr, "push_codes failed: %s\n", codec_common::audio_lm_last_error(ctx));
            codec_common::audio_lm_free(ctx);
            return 8;
        }
        codec_common::audio_lm_audio_output pcm;
        if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
            std::fprintf(stderr, "decode_audio failed: %s\n", codec_common::audio_lm_last_error(ctx));
            codec_common::audio_lm_free(ctx);
            return 8;
        }
        std::string wav_err;
        const int32_t nsamp = (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels);
        if (!codec_example_write_wav_pcm16(a.output.c_str(), pcm.pcm.data(), nsamp,
                                           pcm.sample_rate, &wav_err, pcm.n_channels)) {
            std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
            codec_common::audio_lm_free(ctx);
            return 9;
        }
        const double secs = (double) nsamp / (double) (pcm.sample_rate > 0 ? pcm.sample_rate : 1);
        std::printf("wrote %s: %d samples @ %d Hz (%.2fs, %d ch)\n",
                    a.output.c_str(), nsamp, pcm.sample_rate, secs, pcm.n_channels);
        codec_common::audio_lm_free(ctx);
        return 0;
    }

    // ── Prompt tokenize + prefill ─────────────────────────────────
    std::vector<llama_token> toks = tokenize_prompt(vocab, pi, a.text);
    if (toks.empty()) {
        std::fprintf(stderr, "empty prompt after tokenize\n");
        llama_free(lctx); llama_model_free(lmodel); codec_common::audio_lm_free(ctx); llama_backend_free();
        return 5;
    }
    std::printf("prompt: \"%s%s%s\" → %zu tokens\n",
                pi.prompt_prefix.c_str(), a.text.c_str(), pi.prompt_suffix.c_str(), toks.size());

    // ── Speaker conditioning (voice clone) ────────────────────────────
    // When --ref-audio is supplied and the model has a speaker encoder
    // (Qwen3-TTS ECAPA-TDNN), build_prompt runs the encoder to produce a
    // 1×hidden x-vector; run_codebook_ar prepends it as inputs_embeds row 0.
    std::vector<float> speaker_prefix;
    {
        codec_common::audio_lm_input in;
        in.text = a.text;
        std::vector<float> ref_pcm;
        if (!load_ref_audio(a, in, ref_pcm)) {
            llama_free(lctx); llama_model_free(lmodel); codec_common::audio_lm_free(ctx); llama_backend_free();
            return 5;
        }
        if (codec_common::audio_lm_has_speaker_enc(ctx) && in.ref_pcm != nullptr) {
            codec_common::audio_lm_prompt sp;
            if (!codec_common::audio_lm_build_prompt(ctx, in, &sp)) {
                std::fprintf(stderr, "build_prompt (speaker) failed: %s\n",
                             codec_common::audio_lm_last_error(ctx));
                llama_free(lctx); llama_model_free(lmodel); codec_common::audio_lm_free(ctx); llama_backend_free();
                return 5;
            }
            if (!sp.embeds_prefix.empty() && sp.embeds_prefix_hidden == hidden) {
                // Use only the first row as the conditioning x-vector.
                speaker_prefix.assign(sp.embeds_prefix.begin(),
                                      sp.embeds_prefix.begin() + hidden);
                std::printf("speaker: x-vector prefix rows=%d hidden=%d (from %s)\n",
                            sp.embeds_prefix_rows, sp.embeds_prefix_hidden, a.ref_audio.c_str());
            }
        } else if (!a.ref_audio.empty() && !codec_common::audio_lm_has_speaker_enc(ctx)) {
            std::printf("note: --ref-audio given but model has no speaker encoder; ignoring\n");
        }
    }

    const float temp  = a.temp  ? *a.temp  : pi.default_temperature;
    const float top_p = a.top_p ? *a.top_p : pi.default_top_p;
    const int32_t top_k = a.top_k ? *a.top_k : pi.default_top_k;
    Sampler sampler(a.seed);

    const char * stop_reason = "max_frames";
    int32_t n_frames = 0;
    bool    ar_ok    = false;

    // ── Run the appropriate flow.  Each returns via ar_ok; cleanup of
    //    the llama handles happens once, after. ─────────────────────
    if (pi.is_continuous) {
        // Flow 1: continuous CFM (BlueMagpie / barbet).
        ar_ok = run_continuous(ctx, lctx, toks, hidden, max_frames, a,
                               &n_frames, &stop_reason);
    } else if (pi.sequential_text_audio) {
        // Flow 5: LFM2-Audio sequential text→audio TTS.  Load the backbone's
        // tied token_embd table so the text warmup phase can compute lm_head
        // logits (llama.cpp omits the output head when embeddings are on).
        TextEmbdTable tetab;
        std::string terr;
        if (!tetab.load(a.backbone.c_str(), hidden, terr)) {
            std::fprintf(stderr, "lfm2: text_embd load failed: %s\n", terr.c_str());
            llama_free(lctx); llama_model_free(lmodel); llama_backend_free();
            codec_common::audio_lm_free(ctx);
            return 6;
        }
        ar_ok = run_lfm2_sequential(ctx, lctx, vocab, pi, tetab, toks, hidden,
                                    n_cb, max_frames, sampler, temp, top_p,
                                    top_k, &n_frames, &stop_reason);
    } else if (pi.streaming_interleave) {
        // Flow 3-streaming: MOSS-TTS-Realtime.  Load the backbone text
        // embedding table (composed on top of the codec audio embed) and
        // run the incremental text↔audio interleave.
        TextEmbdTable tetab;
        std::string terr;
        if (!tetab.load(a.backbone.c_str(), hidden, terr)) {
            std::fprintf(stderr, "realtime: text_embd load failed: %s\n", terr.c_str());
            llama_free(lctx); llama_model_free(lmodel); llama_backend_free();
            codec_common::audio_lm_free(ctx);
            return 6;
        }
        const float rep_pen = a.repetition_penalty ? *a.repetition_penalty
                                                   : pi.default_repetition_penalty;
        ar_ok = run_realtime_streaming(ctx, lctx, vocab, pi, tetab, a.text,
                                       hidden, n_cb, max_frames, sampler,
                                       temp, top_p, top_k, rep_pen,
                                       pi.repetition_window,
                                       &n_frames, &stop_reason);
    } else {
        // Flow 2/3: codebook AR (residual depth-AR + parallel-heads delay).
        ar_ok = run_codebook_ar(ctx, lctx, vocab, pi, toks, hidden, n_cb,
                                max_frames, sampler, temp, top_p, top_k,
                                speaker_prefix, a.text, &n_frames, &stop_reason);
    }

    llama_free(lctx);
    llama_model_free(lmodel);
    llama_backend_free();

    if (!ar_ok) {
        std::fprintf(stderr, "AR loop failed: %s\n", codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 6;
    }
    std::printf("AR loop done: %d frames, stop=%s\n", n_frames, stop_reason);
    if (n_frames == 0) {
        std::fprintf(stderr, "no audio frames generated\n");
        codec_common::audio_lm_free(ctx);
        return 7;
    }

    codec_common::audio_lm_audio_output pcm;
    if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
        std::fprintf(stderr, "decode_audio failed: %s\n", codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 8;
    }
    {
        std::string wav_err;
        const int32_t nsamp = (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels);
        if (!codec_example_write_wav_pcm16(a.output.c_str(), pcm.pcm.data(), nsamp,
                                           pcm.sample_rate, &wav_err, pcm.n_channels)) {
            std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
            codec_common::audio_lm_free(ctx);
            return 9;
        }
        const double secs = (double) nsamp / (double) (pcm.sample_rate > 0 ? pcm.sample_rate : 1);
        std::printf("wrote %s: %d samples @ %d Hz (%.2fs, %d ch)\n",
                    a.output.c_str(), nsamp, pcm.sample_rate, secs, pcm.n_channels);
    }
    codec_common::audio_lm_free(ctx);
    return 0;
}
#endif  // TTS_CLI_HAVE_BACKBONE

}  // namespace

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, &a)) { print_usage(argv[0]); return 1; }

    if (a.sub == "info")                return cmd_info(a);
    if (a.sub == "decode")              return cmd_decode(a);
    if (a.sub == "synthesize")          return cmd_synthesize(a);

    std::fprintf(stderr, "unknown subcommand: %s\n", a.sub.c_str());
    print_usage(argv[0]);
    return 1;
}
