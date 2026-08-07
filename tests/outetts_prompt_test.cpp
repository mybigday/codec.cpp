// outetts_prompt_test.cpp — TDD token-oracle test for build_outetts_v3_prompt.
//
// Gate: OUTETTS_GGUF env var must point to an OuteTTS-1.0 backbone gguf
// (e.g. models/hf/outetts_1_0/OuteTTS-1.0-0.6B-Q8_0.gguf).
// SKIP (exit 0) if unset or file missing.
//
// Oracle (0.6B vocab, from outetts10-facts.md §Step 1 + §Step 2):
//   Synthetic speaker: 1 word "hi", duration=0.10, c1={0}, c2={1},
//   features: energy=5, spectral_centroid=10, pitch=20.
//   Target text: "test"
//
// Expected verbatim prefill (hardcoded independently from facts §Step 1):
//   "<|im_start|>\n"
//   "<|text_start|>hi. test<|text_end|>\n"     ← SPEAKER_TEXT. TARGET_TEXT
//   "<|audio_start|>\n"
//   "<|word_start|>hi<|features|><|t_0.10|><|energy_5|><|spectral_centroid_10|>"
//   "<|pitch_20|><|code|><|c1_0|><|c2_1|><|word_end|>\n"
//   "<|word_start|>"
// Plus the trailing code-token ids (c1_0 → 151669, c2_1 → 152695) already
// embedded inside the speaker word-block in the prefill string above — but the
// code tokens are emitted as raw llama_token ids after the text-tokenization
// step, so we verify them at the right position in the returned vector.
//
// Non-tautological design: the expected sequence is built from a HARDCODED
// string (not by calling build_outetts_v3_prompt itself), so a wrong template
// in the production code causes a mismatch.

// Include llama.h FIRST so tts_runner.h sees LLAMA_H and skips its fallback typedef.
#include <llama.h>
#include "tts_runner.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <string>
#include <vector>

// ─── constants from outetts10-facts.md §Step 2 (0.6B) ────────────────────────
static const int32_t OUTETTS_C1_BASE = 151669;  // <|c1_0|>
static const int32_t OUTETTS_C2_BASE = 152694;  // <|c2_0|>

static bool file_exists(const char * path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static int g_fail = 0;

#define CHECK(cond, msg)                                                           \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);             \
            g_fail = 1;                                                            \
        }                                                                          \
    } while (0)

#define CHECK_EQ_I32(actual, expected)                                             \
    do {                                                                           \
        int32_t a_ = (int32_t)(actual);                                            \
        int32_t e_ = (int32_t)(expected);                                          \
        if (a_ != e_) {                                                            \
            std::printf("FAIL %s:%d: got %d want %d\n",                           \
                        __FILE__, __LINE__, a_, e_);                               \
            g_fail = 1;                                                            \
        }                                                                          \
    } while (0)

// ─── helpers ─────────────────────────────────────────────────────────────────

static std::vector<llama_token> tokenize_for_test(const llama_vocab * vocab,
                                                   const std::string & s,
                                                   bool parse_special) {
    int32_t cap = (int32_t) s.size() + 8;
    std::vector<llama_token> toks((size_t) cap);
    int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                               toks.data(), cap, /*add_bos=*/false, parse_special);
    if (n < 0) {
        toks.resize((size_t) -n);
        n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                           toks.data(), (int32_t) toks.size(),
                           /*add_bos=*/false, parse_special);
    }
    toks.resize((size_t) std::max(0, n));
    return toks;
}

// ─── oracle test ─────────────────────────────────────────────────────────────
//
// Synthetic speaker: word="hi", duration=0.10 s (→ round(0.10*75)=7 frames),
// c1={0,0,0,0,0,0,0}, c2={1,1,1,1,1,1,1},
// features: energy=5, spectral_centroid=10, pitch=20.
// Target text: "test"
//
// The INDEPENDENT hardcoded expected string (outetts10-facts.md §Step 1):
//
//   <|im_start|>\n
//   <|text_start|>hi. test<|text_end|>\n
//   <|audio_start|>\n
//   <|word_start|>hi<|features|><|t_0.10|><|energy_5|><|spectral_centroid_10|><|pitch_20|>
//   <|code|><|c1_0|><|c2_1|><|c1_0|><|c2_1|><|c1_0|><|c2_1|><|c1_0|><|c2_1|>
//   <|c1_0|><|c2_1|><|c1_0|><|c2_1|><|c1_0|><|c2_1|><|word_end|>\n
//   <|word_start|>
//
// Note: the speaker text and target text are merged as "SPEAKER_TEXT. TARGET_TEXT"
// per the facts §Step 1 format; the speaker word-block is serialized before the
// trailing <|word_start|> that primes the first generated word.
//
// The c1/c2 code tokens within the word-block are emitted as raw llama_token ids
// (c1_base + code, c2_base + code) rather than via parse_special — the string
// uses the "<|cN_X|>" piece form which llama_tokenize(parse_special=true) resolves
// to the same ids.  We verify the actual ids at the code-token positions.

static void test_build_outetts_v3_prompt(const llama_vocab * vocab) {
    // ── 1. Build synthetic speaker ────────────────────────────────────────────
    codec_common::OutettsSpeaker spk;
    {
        codec_common::OutettsSpeakerWord w;
        w.word     = "hi";
        w.duration = 0.10f;
        // 7 frames: c1 all 0, c2 all 1
        w.c1 = {0, 0, 0, 0, 0, 0, 0};
        w.c2 = {1, 1, 1, 1, 1, 1, 1};
        w.energy             = 5;
        w.spectral_centroid  = 10;
        w.pitch              = 20;
        spk.words.push_back(w);
        spk.text = "hi";
    }

    // ── 2. Call the function under test ──────────────────────────────────────
    std::vector<llama_token> got =
        codec_common::build_outetts_v3_prompt(vocab, "test", spk);

    CHECK(!got.empty(), "build_outetts_v3_prompt returned empty vector");

    // ── 3. Build the INDEPENDENT oracle from the hardcoded template ──────────
    // This string is derived from outetts10-facts.md §Step 1 WITHOUT reusing
    // the production code.  If build_outetts_v3_prompt uses a wrong separator
    // (e.g. ", " instead of ". ") or wrong feature order, this check catches it.
    //
    // Speaker text is "hi" (one word, no trailing period in the word itself).
    // Per facts §Step 1, merged text = "SPEAKER_TEXT. TARGET_TEXT".
    // The duration 0.10 → "<|t_0.10|>".  7 frames of c1_0 / c2_1.

    const std::string oracle_str =
        "<|im_start|>\n"
        "<|text_start|>hi. test<|text_end|>\n"
        "<|audio_start|>\n"
        "<|word_start|>hi<|features|><|t_0.10|>"
        "<|energy_5|><|spectral_centroid_10|><|pitch_20|>"
        "<|code|>"
        "<|c1_0|><|c2_1|>"
        "<|c1_0|><|c2_1|>"
        "<|c1_0|><|c2_1|>"
        "<|c1_0|><|c2_1|>"
        "<|c1_0|><|c2_1|>"
        "<|c1_0|><|c2_1|>"
        "<|c1_0|><|c2_1|>"
        "<|word_end|>\n"
        "<|word_start|>";

    std::vector<llama_token> expected =
        tokenize_for_test(vocab, oracle_str, /*parse_special=*/true);

    CHECK(!expected.empty(), "oracle tokenize returned empty — check GGUF path");

    // ── 4. Compare ────────────────────────────────────────────────────────────
    bool size_ok = (got.size() == expected.size());
    CHECK(size_ok, "build_outetts_v3_prompt: token count mismatch vs oracle");

    if (size_ok) {
        bool match = true;
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != expected[i]) {
                std::printf("FAIL build_outetts_v3_prompt: token[%zu] got %d want %d\n",
                            i, got[i], expected[i]);
                match = false;
                g_fail = 1;
            }
        }
        if (match) {
            std::printf("[outetts_prompt_test] build_outetts_v3_prompt oracle PASS"
                        " (%zu tokens)\n", got.size());
        }
    } else {
        std::printf("FAIL build_outetts_v3_prompt: got %zu tokens, expected %zu\n",
                    got.size(), expected.size());
        // Print a few tokens for diagnostic
        size_t lim = std::min(got.size(), expected.size());
        for (size_t i = 0; i < lim; ++i) {
            if (got[i] != expected[i]) {
                std::printf("  first diff at [%zu]: got %d expected %d\n",
                            i, got[i], expected[i]);
                break;
            }
        }
        g_fail = 1;
    }

    // ── 5. Spot-check raw code-token ids ─────────────────────────────────────
    // The c1_0 token must equal OUTETTS_C1_BASE+0 = 151669.
    // The c2_1 token must equal OUTETTS_C2_BASE+1 = 152695.
    // These should appear in the returned vector.  We verify by checking that
    // the oracle tokenization of "<|c1_0|>" and "<|c2_1|>" yields those ids.
    {
        auto v = tokenize_for_test(vocab, "<|c1_0|>", /*parse_special=*/true);
        if (v.size() == 1) {
            CHECK_EQ_I32(v[0], OUTETTS_C1_BASE);
        } else {
            std::printf("FAIL: <|c1_0|> did not tokenize to a single token (got %zu tokens)\n",
                        v.size());
            g_fail = 1;
        }
    }
    {
        auto v = tokenize_for_test(vocab, "<|c2_1|>", /*parse_special=*/true);
        if (v.size() == 1) {
            CHECK_EQ_I32(v[0], OUTETTS_C2_BASE + 1);
        } else {
            std::printf("FAIL: <|c2_1|> did not tokenize to a single token (got %zu tokens)\n",
                        v.size());
            g_fail = 1;
        }
    }

    // ── 6. Guard: out-of-range code must return empty ─────────────────────────
    {
        codec_common::OutettsSpeaker bad_spk;
        codec_common::OutettsSpeakerWord bw;
        bw.word = "x"; bw.duration = 0.05f;
        bw.c1 = {1024};  // 1024 is out of range (valid: 0..1023)
        bw.c2 = {0};
        bw.energy = 0; bw.spectral_centroid = 0; bw.pitch = 0;
        bad_spk.words.push_back(bw);
        bad_spk.text = "x";
        auto bad = codec_common::build_outetts_v3_prompt(vocab, "test", bad_spk);
        CHECK(bad.empty(), "build_outetts_v3_prompt: code=1024 (out-of-range) must return empty");
    }

    // ── 7. Guard: empty text must return empty ────────────────────────────────
    {
        codec_common::OutettsSpeaker spk2 = spk;
        auto empty_text = codec_common::build_outetts_v3_prompt(vocab, "", spk2);
        CHECK(empty_text.empty(), "build_outetts_v3_prompt: empty text must return empty");
    }
}

// ─── load_outetts_speaker test ───────────────────────────────────────────────
// Round-trips the bundled default speaker through load_outetts_speaker.
// Checks: non-empty words, word[0] text="The", word[0] c1[0]=720, c2[0]=658,
// features energy=10/sc=15/pitch=45 (from the worked example in facts §Step 1).
static void test_load_outetts_speaker() {
    // Try to find the bundled default speaker asset.
    const char * candidates[] = {
        "assets/speakers/en-female-1-neutral.json",
        "docs/superpowers/notes/outetts10-en-female-1-neutral.json",
        nullptr
    };
    const char * path = nullptr;
    for (int i = 0; candidates[i]; ++i) {
        if (file_exists(candidates[i])) { path = candidates[i]; break; }
    }
    if (!path) {
        std::printf("[outetts_prompt_test] SKIP load_outetts_speaker:"
                    " no speaker JSON found in assets/ or docs/\n");
        return;
    }

    codec_common::OutettsSpeaker spk;
    bool ok = codec_common::load_outetts_speaker(path, &spk);
    CHECK(ok, "load_outetts_speaker returned false");
    if (!ok) return;

    CHECK(!spk.words.empty(), "speaker words must be non-empty");
    CHECK(!spk.text.empty(),  "speaker text must be non-empty");

    if (!spk.words.empty()) {
        const auto & w0 = spk.words[0];
        CHECK(w0.word == "The", "word[0].word must be 'The'");
        CHECK(!w0.c1.empty(), "word[0].c1 must be non-empty");
        CHECK(!w0.c2.empty(), "word[0].c2 must be non-empty");
        if (!w0.c1.empty()) CHECK_EQ_I32(w0.c1[0], 720);
        if (!w0.c2.empty()) CHECK_EQ_I32(w0.c2[0], 658);
        CHECK_EQ_I32(w0.energy,            10);
        CHECK_EQ_I32(w0.spectral_centroid, 15);
        CHECK_EQ_I32(w0.pitch,             45);
    }
    std::printf("[outetts_prompt_test] load_outetts_speaker PASS (%zu words, text[0..20]=\"%.20s\")\n",
                spk.words.size(), spk.text.c_str());
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    const char * gguf = std::getenv("OUTETTS_GGUF");
    if (!gguf || !file_exists(gguf)) {
        std::printf("SKIP outetts_prompt_test: OUTETTS_GGUF not set or file missing\n");
        return 0;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only   = true;
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(gguf, mparams);
    if (!model) {
        std::printf("FAIL: could not load backbone from %s\n", gguf);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    test_build_outetts_v3_prompt(vocab);
    test_load_outetts_speaker();

    llama_model_free(model);

    if (g_fail) {
        std::printf("outetts_prompt_test: FAILURES\n");
        return 1;
    }
    std::printf("outetts_prompt_test: all passed\n");
    return 0;
}
