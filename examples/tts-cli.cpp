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
        "             defaults to the model's training-time knobs; --temp 0 = greedy.\n"
        "  trace      --model PATH --tokens \"id1,id2,...\"\n"
        "             [--audio-offset N --audio-count K [--audio-eos M]]\n"
        "             dispatch the given token stream through observe_token;\n"
        "             prints one verdict per token.  When --audio-offset is\n"
        "             omitted the model's GGUF metadata determines the range.\n"
        "  simulate-typeA --model PATH --codes PATH.npy --output PATH.wav\n"
        "             treat each code as `code + 10000` and dispatch through\n"
        "             observe_token (Type A path) → decode_audio.  Validates\n"
        "             the observe→decode data flow end-to-end without needing\n"
        "             a real Type A LM checkpoint.\n"
        "  simulate-typeB --model PATH --codes PATH.npy --output PATH.wav\n"
        "             [--start-step N] [--probe-first]\n"
        "             same as simulate-typeA but with the Type B embed-override\n"
        "             flag set.  Each consume should return CONSUMED_EMBED and\n"
        "             populate get_next_embed with `audio_embd[code] +\n"
        "             pos_embd[step]`.  Used with chatterbox.gguf to verify the\n"
        "             compose_next_embd path.\n"
        "  simulate-multicb --model PATH --codes PATH.npy --output PATH.wav\n"
        "             [--use-embed-override [--start-step N]]\n"
        "             Multi-codebook frame observe (Type C / Type D).  Loads a\n"
        "             (T, n_q) .npy and dispatches one frame per observe_codes\n"
        "             call.  With --use-embed-override, each consume composes\n"
        "             the next backbone-input embed (validates Type C path on\n"
        "             residual_depth_ar models).  decode_audio at the end\n"
        "             checks the accumulator matches a direct codec_decode.\n"
        "  simulate-continuous --model PATH --hidden-npy PATH.npy --output PATH.wav\n"
        "             [--noise-npy PATH.npy] [--cfg FLOAT] [--timesteps N] [--min-len N]\n"
        "             Continuous-latent observe (BlueMagpie / continuous_latent_cfm).\n"
        "             Loads a (T, hidden_dim) F32 .npy of backbone hidden states,\n"
        "             dispatches one step per audio_lm_observe_hidden call.  Each\n"
        "             step runs tslm_adapter + FSQ + RALM + LocDiT CFM internally;\n"
        "             OBSERVE_STOP from the stop head breaks the loop.  Output WAV\n"
        "             via audio_lm_decode_audio (AudioVAE).  Pass --noise-npy of\n"
        "             shape (T, patch_size*latent_dim) for deterministic output.\n",
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

// Minimal xorshift64* sampler mirroring rn-tts's sample_codec_logits:
// temp<=0 → greedy argmax; else softmax(temp) → top-k → top-p → sample.
struct Sampler {
    std::mt19937_64 rng;
    explicit Sampler(uint32_t seed) : rng(seed ? seed : 0xC0DEC1ABull) {}

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
                     int32_t * out_frames, const char ** out_stop) {
    codec_common::audio_lm_set_uses_embed_override(ctx, true, 1);
    int32_t n_past = 0;
    // Speaker x-vector prefix (Qwen3-TTS voice clone): the ECAPA-TDNN
    // x-vector is prepended as inputs_embeds row 0 before the text prompt,
    // conditioning the talker on the reference voice.  Decode it first so
    // the text tokens attend to it via the KV cache.
    if (!speaker_prefix.empty() && (int32_t) speaker_prefix.size() == hidden) {
        std::vector<float> pfx(speaker_prefix);
        if (!decode_embed(lctx, pfx.data(), hidden, n_past)) {
            std::fprintf(stderr, "speaker prefix decode failed\n"); return false;
        }
        n_past += 1;
    }
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
        if (!decode_embed(lctx, nbc.data(), dim, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
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
    const int32_t max_frames = a.max_frames > 0 ? a.max_frames : 512;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t) std::max(4096, max_frames + 256);
    cp.n_batch      = cp.n_ctx;
    cp.n_ubatch     = cp.n_ctx;
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
    } else {
        // Flow 2/3: codebook AR (residual depth-AR + parallel-heads delay).
        ar_ok = run_codebook_ar(ctx, lctx, vocab, pi, toks, hidden, n_cb,
                                max_frames, sampler, temp, top_p, top_k,
                                speaker_prefix, &n_frames, &stop_reason);
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

// ─── Subcommand: trace ───────────────────────────────────────────
// Dispatch a comma-separated token stream through observe_token and
// print one verdict line per token.  Used to hand-validate the Type A
// classification logic with model-side metadata or manual overrides.
int cmd_trace(const args & a) {
    if (a.tokens_csv.empty()) {
        std::fprintf(stderr, "trace requires --tokens \"id1,id2,...\"\n");
        return 1;
    }
    codec_common::audio_lm_params p;
    p.codec_path = a.model;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;
    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    if (a.audio_offset_set) {
        codec_common::audio_lm_set_audio_token_range(
            ctx, a.audio_offset, a.audio_count, a.audio_eos);
    }
    int32_t off=0, cnt=0, eos=0;
    codec_common::audio_lm_get_audio_token_range(ctx, &off, &cnt, &eos);
    std::printf("audio_token: offset=%d count=%d eos_id=%d\n", off, cnt, eos);

    // Parse the CSV stream.
    std::vector<int32_t> toks;
    {
        const std::string & s = a.tokens_csv;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) {
                try { toks.push_back(std::stoi(s.substr(i, j - i))); }
                catch (...) { std::fprintf(stderr, "skipped malformed token: %s\n", s.substr(i, j-i).c_str()); }
            }
            i = j + 1;
        }
    }

    auto name_of = [](codec_common::observe_action a) {
        switch (a) {
            case codec_common::OBSERVE_PASSTHROUGH:    return "PASSTHROUGH";
            case codec_common::OBSERVE_CONSUMED:       return "CONSUMED";
            case codec_common::OBSERVE_CONSUMED_EMBED: return "CONSUMED_EMBED";
            case codec_common::OBSERVE_STOP:           return "STOP";
        }
        return "?";
    };

    for (int32_t tok : toks) {
        auto act = codec_common::audio_lm_observe_token(ctx, tok, nullptr, 0);
        std::printf("  tok=%-6d → %s\n", tok, name_of(act));
        if (act == codec_common::OBSERVE_STOP) break;
    }
    codec_common::audio_lm_free(ctx);
    return 0;
}

// ─── Subcommand: simulate-typeA ──────────────────────────────────
// Take a real codes .npy (must be n_q=1; treat each value as
// `code + 10000` so it looks like a Type A LM-vocab token), dispatch
// each through observe_token, then call decode_audio.  Validates the
// observe → accumulator → decode_audio data flow end-to-end without
// needing a real Type A LM checkpoint.
int cmd_simulate_typeA(const args & a) {
    if (a.codes.empty() || a.output.empty()) {
        std::fprintf(stderr, "simulate-typeA requires --codes and --output\n");
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
    if (n_q != 1) {
        std::fprintf(stderr, "simulate-typeA expects n_q=1 codes, got n_q=%d\n", n_q);
        codec_common::audio_lm_free(ctx);
        return 4;
    }

    const int32_t OFFSET = 10000;
    const int32_t COUNT  = 65536;   // larger than any reasonable codebook
    const int32_t EOS    = 99999;
    codec_common::audio_lm_set_audio_token_range(ctx, OFFSET, COUNT, EOS);

    int32_t n_consumed = 0;
    for (int32_t i = 0; i < n_frames; ++i) {
        const int32_t fake_tok = OFFSET + npy_codes[(size_t) i];
        auto act = codec_common::audio_lm_observe_token(ctx, fake_tok, nullptr, 0);
        if (act != codec_common::OBSERVE_CONSUMED) {
            std::fprintf(stderr,
                "simulate-typeA: expected CONSUMED, got action=%d at frame %d (tok=%d)\n",
                (int) act, i, fake_tok);
            codec_common::audio_lm_free(ctx);
            return 5;
        }
        n_consumed++;
    }
    auto eos_act = codec_common::audio_lm_observe_token(ctx, EOS, nullptr, 0);
    if (eos_act != codec_common::OBSERVE_STOP) {
        std::fprintf(stderr, "simulate-typeA: expected STOP on EOS, got %d\n", (int) eos_act);
        codec_common::audio_lm_free(ctx);
        return 6;
    }
    std::printf("simulate-typeA: %d tokens CONSUMED + STOP on EOS\n", n_consumed);

    codec_common::audio_lm_audio_output pcm;
    if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
        std::fprintf(stderr, "decode_audio failed: %s\n",
                     codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 7;
    }

    std::string wav_err;
    if (!codec_example_write_wav_pcm16(
            a.output.c_str(), pcm.pcm.data(),
            (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels),
            pcm.sample_rate, &wav_err, pcm.n_channels)) {
        std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 8;
    }
    std::printf("wrote %s: %zu samples @ %d Hz\n",
                a.output.c_str(), pcm.pcm.size() / (size_t) pcm.n_channels, pcm.sample_rate);
    codec_common::audio_lm_free(ctx);
    return 0;
}

// ─── Subcommand: simulate-typeB ──────────────────────────────────
// Like simulate-typeA but with the embed-override flag set: every
// consumed token should return CONSUMED_EMBED with `get_next_embed`
// populated by `audio_embd[code] + pos_embd[step]`.  Exercises the
// codec_lm_compose_next_embd code path end-to-end.  The same
// accumulator still backs decode_audio so we also get a WAV out at
// the end (matches a direct codec_decode of the input codes).
int cmd_simulate_typeB(const args & a) {
    if (a.codes.empty() || a.output.empty()) {
        std::fprintf(stderr, "simulate-typeB requires --codes and --output\n");
        return 1;
    }
    codec_common::audio_lm_params p;
    p.codec_path = a.model;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;
    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    if (codec_common::audio_lm_n_codebook(ctx) != 1) {
        std::fprintf(stderr,
            "simulate-typeB expects a single-codebook model (n_q=1), got %d\n",
            codec_common::audio_lm_n_codebook(ctx));
        codec_common::audio_lm_free(ctx);
        return 2;
    }
    const int32_t hidden = codec_common::audio_lm_hidden_dim(ctx);
    if (hidden <= 0) {
        std::fprintf(stderr, "simulate-typeB: model reports hidden_dim=0\n");
        codec_common::audio_lm_free(ctx);
        return 2;
    }

    std::vector<int32_t> npy_codes;
    int32_t n_frames = 0, n_q = 0;
    std::string npy_err;
    if (!codec_example_load_npy_i32_2d_tq(a.codes.c_str(), &npy_codes, &n_q, &n_frames, &npy_err)) {
        std::fprintf(stderr, "failed to load %s: %s\n", a.codes.c_str(), npy_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 3;
    }
    if (n_q != 1) {
        std::fprintf(stderr, "simulate-typeB expects n_q=1 codes, got n_q=%d\n", n_q);
        codec_common::audio_lm_free(ctx);
        return 4;
    }

    // Use a synthetic audio range that wraps the model's actual codebook.
    // For chatterbox.gguf this would naturally be offset=0, count=6561,
    // eos_id=6562 — but we keep the same `code + 10000` convention as
    // simulate-typeA so the test runs without any model-specific knowledge.
    const int32_t OFFSET = 10000;
    const int32_t COUNT  = 65536;
    const int32_t EOS    = 99999;
    codec_common::audio_lm_set_audio_token_range(ctx, OFFSET, COUNT, EOS);
    codec_common::audio_lm_set_uses_embed_override(ctx, true, a.start_step);

    int32_t n_consumed_embed = 0;
    float last_embed_l2 = 0.0f;
    for (int32_t i = 0; i < n_frames; ++i) {
        const int32_t fake_tok = OFFSET + npy_codes[(size_t) i];
        auto act = codec_common::audio_lm_observe_token(ctx, fake_tok, nullptr, 0);
        if (act != codec_common::OBSERVE_CONSUMED_EMBED) {
            std::fprintf(stderr,
                "simulate-typeB: expected CONSUMED_EMBED, got action=%d at frame %d (tok=%d): %s\n",
                (int) act, i, fake_tok,
                codec_common::audio_lm_last_error(ctx));
            codec_common::audio_lm_free(ctx);
            return 5;
        }
        int32_t dim = 0;
        const float * eb = codec_common::audio_lm_get_next_embed(ctx, &dim);
        if (eb == nullptr || dim != hidden) {
            std::fprintf(stderr,
                "simulate-typeB: get_next_embed empty/wrong dim (dim=%d, hidden=%d) at frame %d\n",
                dim, hidden, i);
            codec_common::audio_lm_free(ctx);
            return 6;
        }
        if (i == n_frames - 1) {
            double sq = 0.0;
            for (int32_t k = 0; k < dim; ++k) sq += (double) eb[k] * (double) eb[k];
            last_embed_l2 = (float) std::sqrt(sq);
        }
        if (i == 0 && a.probe_first) {
            // Dump enough state for an external script to verify the
            // composition.  Code value + step value pin which table rows
            // were summed; the first 8 floats + L2 are the comparison
            // points.
            std::printf("probe-first: code=%d step=%d ||v||=", npy_codes[0], a.start_step);
            double sq = 0.0;
            for (int32_t k = 0; k < dim; ++k) sq += (double) eb[k] * (double) eb[k];
            std::printf("%.6f  v[:8]=", std::sqrt(sq));
            for (int32_t k = 0; k < 8 && k < dim; ++k) std::printf(" %.6f", eb[k]);
            std::printf("\n");
        }
        n_consumed_embed++;
    }
    auto eos_act = codec_common::audio_lm_observe_token(ctx, EOS, nullptr, 0);
    if (eos_act != codec_common::OBSERVE_STOP) {
        std::fprintf(stderr, "simulate-typeB: expected STOP on EOS, got %d\n", (int) eos_act);
        codec_common::audio_lm_free(ctx);
        return 7;
    }
    std::printf("simulate-typeB: %d tokens CONSUMED_EMBED + STOP on EOS (hidden=%d, last embed ||v||=%.4f, end step=%d)\n",
                n_consumed_embed, hidden, last_embed_l2, a.start_step + n_consumed_embed);

    codec_common::audio_lm_audio_output pcm;
    if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
        std::fprintf(stderr, "decode_audio failed: %s\n",
                     codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 8;
    }
    std::string wav_err;
    if (!codec_example_write_wav_pcm16(
            a.output.c_str(), pcm.pcm.data(),
            (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels),
            pcm.sample_rate, &wav_err, pcm.n_channels)) {
        std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 9;
    }
    std::printf("wrote %s: %zu samples @ %d Hz\n",
                a.output.c_str(), pcm.pcm.size() / (size_t) pcm.n_channels, pcm.sample_rate);
    codec_common::audio_lm_free(ctx);
    return 0;
}

// ─── Subcommand: simulate-multicb ────────────────────────────────
// Multi-codebook frame observe (Type C / Type D).  Read a (T, n_q)
// .npy of codes, hand each frame as one observe_codes call, then
// decode_audio to a WAV.  When --use-embed-override is set, also runs
// compose_next_embd on every frame (Type C path — verifies
// residual_depth_ar's compose works through the new public API).
//
// Bit-exact decode_audio output vs `codec-cli decode` on the same
// codes proves the accumulator's (T, n_q) interleaved layout is
// correct for arbitrary n_q.
int cmd_simulate_multicb(const args & a) {
    if (a.codes.empty()) {
        std::fprintf(stderr, "simulate-multicb requires --codes\n");
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
    std::printf("loaded codes: n_frames=%d n_q=%d\n", n_frames, n_q);

    const codec_common::observe_action want =
        a.use_embed_override ? codec_common::OBSERVE_CONSUMED_EMBED
                              : codec_common::OBSERVE_CONSUMED;
    if (a.use_embed_override) {
        codec_common::audio_lm_set_uses_embed_override(ctx, true, a.start_step);
    }

    int32_t n_consumed = 0;
    for (int32_t i = 0; i < n_frames; ++i) {
        const int32_t * frame = npy_codes.data() + (size_t) i * (size_t) n_q;
        auto act = codec_common::audio_lm_observe_codes(
            ctx, frame, n_q, /*last_hidden=*/nullptr, /*hidden_dim=*/0);
        if (act != want) {
            std::fprintf(stderr,
                "simulate-multicb: expected %s, got action=%d at frame %d: %s\n",
                a.use_embed_override ? "CONSUMED_EMBED" : "CONSUMED",
                (int) act, i, codec_common::audio_lm_last_error(ctx));
            codec_common::audio_lm_free(ctx);
            return 5;
        }
        if (a.use_embed_override) {
            int32_t dim = 0;
            const float * eb = codec_common::audio_lm_get_next_embed(ctx, &dim);
            if (eb == nullptr || dim != codec_common::audio_lm_hidden_dim(ctx)) {
                std::fprintf(stderr,
                    "simulate-multicb: get_next_embed null/wrong dim at frame %d\n", i);
                codec_common::audio_lm_free(ctx);
                return 6;
            }
        }
        n_consumed++;
    }
    std::printf("simulate-multicb: %d frames %s (n_q=%d)\n",
                n_consumed,
                a.use_embed_override ? "CONSUMED_EMBED" : "CONSUMED",
                n_q);

    if (a.output.empty()) {
        // codec_lm-only models (CSM, MOSS-TTSD LM-only) have no
        // decoder in this GGUF — skip the decode_audio step so we can
        // still exercise observe_codes + compose_next_embd against
        // them.  Useful for verifying the embed-override path on
        // residual_depth_ar (CSM) without needing a paired Mimi codec.
        std::printf("simulate-multicb: --output omitted — skipping decode_audio\n");
        codec_common::audio_lm_free(ctx);
        return 0;
    }

    codec_common::audio_lm_audio_output pcm;
    if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
        std::fprintf(stderr, "decode_audio failed: %s\n",
                     codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 8;
    }
    std::string wav_err;
    if (!codec_example_write_wav_pcm16(
            a.output.c_str(), pcm.pcm.data(),
            (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels),
            pcm.sample_rate, &wav_err, pcm.n_channels)) {
        std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 9;
    }
    std::printf("wrote %s: %zu samples @ %d Hz\n",
                a.output.c_str(), pcm.pcm.size() / (size_t) pcm.n_channels, pcm.sample_rate);
    codec_common::audio_lm_free(ctx);
    return 0;
}

// ─── Subcommand: simulate-continuous ───────────────────────────────
// Continuous-latent observe (BlueMagpie / VoxCPM2 family).  The host
// owns the backbone (Barbet llama.cpp variant) and drives one
// observe_hidden call per AR step with the latest hidden state.
// codec_common runs tslm_adapter + FSQ + RALM + LocDiT CFM internally
// per step, accumulates the produced latent patch into the codec_lm
// state, and reports OBSERVE_STOP when the stop head fires.
// decode_audio at the end runs AudioVAE on the accumulated patches.
int cmd_simulate_continuous(const args & a) {
    if (a.hidden_npy.empty() || a.output.empty()) {
        std::fprintf(stderr, "simulate-continuous requires --hidden-npy and --output\n");
        return 1;
    }
    codec_common::audio_lm_params p;
    p.codec_path = a.model;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;
    std::string err;
    auto * ctx = codec_common::audio_lm_init(p, &err);
    if (!ctx) { std::fprintf(stderr, "audio_lm_init failed: %s\n", err.c_str()); return 2; }

    if (!codec_common::audio_lm_is_continuous(ctx)) {
        std::fprintf(stderr, "simulate-continuous: model is not a continuous-latent kind\n");
        codec_common::audio_lm_free(ctx);
        return 2;
    }
    const int32_t hidden = codec_common::audio_lm_hidden_dim(ctx);
    if (hidden <= 0) {
        std::fprintf(stderr, "simulate-continuous: hidden_dim=0\n");
        codec_common::audio_lm_free(ctx);
        return 2;
    }
    codec_common::audio_lm_set_continuous_params(ctx, a.cont_cfg, a.cont_timesteps, a.cont_min_len);

    std::vector<float> hidden_vec;
    int32_t n_steps = 0, h_cols = 0;
    std::string npy_err;
    if (!codec_example_load_npy_f32_2d(a.hidden_npy.c_str(), &hidden_vec, &n_steps, &h_cols, &npy_err)) {
        std::fprintf(stderr, "failed to load %s: %s\n", a.hidden_npy.c_str(), npy_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 3;
    }
    if (h_cols != hidden) {
        std::fprintf(stderr,
            "simulate-continuous: hidden NPY column dim %d != model hidden_dim %d\n",
            h_cols, hidden);
        codec_common::audio_lm_free(ctx);
        return 4;
    }

    std::vector<float> noise_vec;
    int32_t noise_steps = 0, noise_cols = 0;
    const float * noise_ptr = nullptr;
    if (!a.noise_npy.empty()) {
        if (!codec_example_load_npy_f32_2d(a.noise_npy.c_str(), &noise_vec,
                                            &noise_steps, &noise_cols, &npy_err)) {
            std::fprintf(stderr, "failed to load %s: %s\n", a.noise_npy.c_str(), npy_err.c_str());
            codec_common::audio_lm_free(ctx);
            return 3;
        }
        if (noise_steps != n_steps) {
            std::fprintf(stderr,
                "simulate-continuous: noise has %d steps but hidden has %d\n",
                noise_steps, n_steps);
            codec_common::audio_lm_free(ctx);
            return 4;
        }
        noise_ptr = noise_vec.data();
    }

    std::printf("simulate-continuous: hidden=(%d, %d) cfg=%.3f timesteps=%d noise=%s\n",
                n_steps, hidden, a.cont_cfg, a.cont_timesteps,
                noise_ptr ? "fixed" : "sampled");

    int32_t n_consumed = 0;
    bool stopped = false;
    for (int32_t i = 0; i < n_steps; ++i) {
        const float * h = hidden_vec.data() + (size_t) i * (size_t) hidden;
        const float * z = noise_ptr ? noise_ptr + (size_t) i * (size_t) noise_cols : nullptr;
        auto act = codec_common::audio_lm_observe_hidden(ctx, h, hidden, z);
        if (act == codec_common::OBSERVE_STOP) {
            const char * raw_err = codec_common::audio_lm_last_error(ctx);
            if (raw_err && *raw_err) {
                std::fprintf(stderr,
                    "simulate-continuous: OBSERVE_STOP at step %d (error path): %s\n",
                    i, raw_err);
                codec_common::audio_lm_free(ctx);
                return 5;
            }
            std::printf("simulate-continuous: stop head fired at step %d\n", i);
            stopped = true;
            break;
        }
        if (act != codec_common::OBSERVE_CONSUMED_EMBED) {
            std::fprintf(stderr,
                "simulate-continuous: expected CONSUMED_EMBED, got %d at step %d: %s\n",
                (int) act, i, codec_common::audio_lm_last_error(ctx));
            codec_common::audio_lm_free(ctx);
            return 6;
        }
        n_consumed++;
    }
    if (!stopped) {
        std::printf("simulate-continuous: %d steps CONSUMED_EMBED (exhausted hidden npy)\n",
                    n_consumed);
    }

    codec_common::audio_lm_audio_output pcm;
    if (!codec_common::audio_lm_decode_audio(ctx, &pcm)) {
        std::fprintf(stderr, "decode_audio failed: %s\n",
                     codec_common::audio_lm_last_error(ctx));
        codec_common::audio_lm_free(ctx);
        return 7;
    }
    std::string wav_err;
    if (!codec_example_write_wav_pcm16(
            a.output.c_str(), pcm.pcm.data(),
            (int32_t) (pcm.pcm.size() / (size_t) pcm.n_channels),
            pcm.sample_rate, &wav_err, pcm.n_channels)) {
        std::fprintf(stderr, "failed to write %s: %s\n", a.output.c_str(), wav_err.c_str());
        codec_common::audio_lm_free(ctx);
        return 8;
    }
    std::printf("wrote %s: %zu samples @ %d Hz\n",
                a.output.c_str(), pcm.pcm.size() / (size_t) pcm.n_channels, pcm.sample_rate);
    codec_common::audio_lm_free(ctx);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, &a)) { print_usage(argv[0]); return 1; }

    if (a.sub == "info")                return cmd_info(a);
    if (a.sub == "decode")              return cmd_decode(a);
    if (a.sub == "synthesize")          return cmd_synthesize(a);
    if (a.sub == "trace")               return cmd_trace(a);
    if (a.sub == "simulate-typeA")      return cmd_simulate_typeA(a);
    if (a.sub == "simulate-typeB")      return cmd_simulate_typeB(a);
    if (a.sub == "simulate-multicb")    return cmd_simulate_multicb(a);
    if (a.sub == "simulate-continuous") return cmd_simulate_continuous(a);

    std::fprintf(stderr, "unknown subcommand: %s\n", a.sub.c_str());
    print_usage(argv[0]);
    return 1;
}
