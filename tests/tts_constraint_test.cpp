// Standalone assert test for the model-less TTS constraint logic.
// Returns 0 on success, non-zero (and prints) on the first failure.
#include "codec_common.h"
#include "tts_runner.h"
#include <cmath>
#include <cstdio>
#include <string>

using codec_common::outetts_version;
using codec_common::outetts_build_grammar;
using codec_common::build_flat_audio_mask;

static int g_fail = 0;
#define CHECK_EQ(actual, expected)                                            \
    do {                                                                      \
        const std::string a_ = (actual);                                      \
        const std::string e_ = (expected);                                    \
        if (a_ != e_) {                                                       \
            std::printf("FAIL %s:%d\n--- got ---\n%s\n--- want ---\n%s\n",    \
                        __FILE__, __LINE__, a_.c_str(), e_.c_str());          \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

static const char * kWavTokCode =
    "CODE       ::= \"<|\" ( [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | [1-3] [0-9] [0-9] [0-9] | "
    "\"4\" ( \"0\" [0-8] [0-9] | \"0\" \"9\" [0-5] ) ) \"|>\"\n";

static void test_v2_two_ascii_words() {
    const std::string g = outetts_build_grammar(outetts_version::V2, "hello world");
    const std::string want =
        std::string("root       ::= NL? \"hello\" audioBlock \"world\" audioBlock audioEnd NL? EOS?\n"
        "audioBlock ::= TIME CODE{1,750} space NL?\n"
        "EOS        ::= <|im_end|>\n"
        "audioEnd   ::= <|audio_end|>\n"
        "space      ::= <|space|>\n"
        "NL         ::= [\\n]\n"
        "TIME       ::= \"<|t_\" DECIMAL \"|>\"\n") + kWavTokCode +
        "DECIMAL    ::= [0-9] \".\" [0-9] [0-9] | \"10.00\"\n";
    CHECK_EQ(g, want);
}

static void test_v1_code_start_end() {
    const std::string g = outetts_build_grammar(outetts_version::V1, "hi");
    const std::string want =
        std::string("root       ::= NL? \"hi\" audioBlock audioEnd NL? EOS?\n"
        "audioBlock ::= TIME codeStart CODE{1,750} codeEnd NL?\n"
        "EOS        ::= <|im_end|>\n"
        "audioEnd   ::= <|audio_end|>\n"
        "codeStart  ::= <|code_start|>\n"
        "codeEnd    ::= <|code_end|>\n"
        "NL         ::= [\\n]\n"
        "TIME       ::= \"<|t_\" DECIMAL \"|>\"\n") + kWavTokCode +
        "DECIMAL    ::= [0-9] \".\" [0-9] [0-9] | \"10.00\"\n";
    CHECK_EQ(g, want);
}

static void test_v3_uses_word_alternation_and_features() {
    const std::string g = outetts_build_grammar(outetts_version::V3, "hi there");
    const std::string want =
        "root       ::= leadWord wordBlock* audioEnd NL EOS?\n"
        "leadWord ::= WORD audioBlock\n"
        "wordBlock ::= wordStart WORD audioBlock\n"
        "audioBlock ::= codeBlock wordEnd NL?\n"
        "codeBlock ::= features TIME energy spectralCentroid pitch CODE CODES{1,750}\n"
        "TEXT ::= [A-Za-z0-9.,!?]+\n"
        "EOS      ::= <|im_end|>\n"
        "audioEnd   ::= <|audio_end|>\n"
        "wordStart ::= <|word_start|>\n"
        "wordEnd ::= <|word_end|>\n"
        "features ::= <|features|>\n"
        "energy ::= \"<|energy_\" DIGITS \"|>\"\n"
        "spectralCentroid ::= \"<|spectral_centroid_\" DIGITS \"|>\"\n"
        "pitch ::= \"<|pitch_\" DIGITS \"|>\"\n"
        "WORD       ::= \"hi\" | \"there\"\n"
        "NL         ::= [\\n]\n"
        "TIME       ::= \"<|t_\" DECIMAL \"|>\"\n"
        "CODE       ::= <|code|>\n"
        "CODES      ::= CODE1 CODE2\n"
        "CODE1      ::= \"<|c1_\" ( [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | \"1\" ( \"0\" [0-1] [0-9] | \"0\" \"2\" [0-3] ) ) \"|>\"\n"
        "CODE2      ::= \"<|c2_\" ( [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | \"1\" ( \"0\" [0-1] [0-9] | \"0\" \"2\" [0-3] ) ) \"|>\"\n"
        "DIGITS     ::= [0-9] | [1-9] [0-9] | [1-9] [0-9] [0-9] | \"1\" ( \"0\" \"0\" [0-0] )\n"
        "DECIMAL    ::= [0-9] \".\" [0-9] [0-9] | \"10.00\"\n";
    CHECK_EQ(g, want);
}

static void test_escaping_and_non_ascii() {
    // ASCII word containing a quote → escaped (safe deviation from reference).
    const std::string q = outetts_build_grammar(outetts_version::V3, "sa\"y");
    // WORD line holds the single escaped literal.
    if (q.find("WORD       ::= \"sa\\\"y\"\n") == std::string::npos) {
        std::printf("FAIL escaping: %s\n", q.c_str()); g_fail = 1;
    }
    // Non-ASCII (CJK) word → [unique-codepoints]+ preserving first-seen order.
    // "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xBD\xA0" == U+4F60 U+597D U+4F60 (你好你)
    const std::string c = outetts_build_grammar(outetts_version::V3,
                                                "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xBD\xA0");
    if (c.find("WORD       ::= [\xE4\xBD\xA0\xE5\xA5\xBD]+\n") == std::string::npos) {
        std::printf("FAIL non-ascii: %s\n", c.c_str()); g_fail = 1;
    }
}

static void test_flat_audio_mask() {
    // vocab 10, audio ids [3,7), eos = 9 → allowed {3,4,5,6,9}; mask the rest.
    auto m = build_flat_audio_mask(/*offset=*/3, /*count=*/4, /*eos=*/9, /*n_vocab=*/10);
    bool masked[10] = {false,false,false,false,false,false,false,false,false,false};
    for (const auto & lb : m) {
        if (lb.token < 0 || lb.token >= 10) {
            std::printf("FAIL mask oob %d\n", lb.token); g_fail = 1;
        } else {
            masked[lb.token] = true;
            if (lb.bias != -INFINITY) { std::printf("FAIL mask bias\n"); g_fail = 1; }
        }
    }
    // allowed = {3,4,5,6,9}; expected masked = {0,1,2,7,8}.
    const bool exp[10] = {true,true,true,false,false,false,false,true,true,false};
    for (int i = 0; i < 10; ++i) if (masked[i] != exp[i]) {
        std::printf("FAIL mask id %d: got %d want %d\n", i, masked[i], exp[i]); g_fail = 1;
    }
}

static void test_flat_audio_mask_edges() {
    // Case 1: eos inside range — build_flat_audio_mask(3, 4, 5, 10)
    // audio range [3,7), eos=5 already in range.
    // allowed = {3,4,5,6}; masked = {0,1,2,7,8,9}.
    // eos token 5 must NOT appear in the returned vector (it is allowed, not masked).
    {
        auto m = build_flat_audio_mask(/*offset=*/3, /*count=*/4, /*eos=*/5, /*n_vocab=*/10);
        bool masked[10] = {};
        for (const auto & lb : m) {
            if (lb.token < 0 || lb.token >= 10) {
                std::printf("FAIL edge1 mask oob %d\n", lb.token); g_fail = 1;
            } else {
                if (lb.token == 5) {
                    std::printf("FAIL edge1: eos=5 (inside range) appears in mask\n"); g_fail = 1;
                }
                masked[lb.token] = true;
                if (lb.bias != -INFINITY) { std::printf("FAIL edge1 mask bias\n"); g_fail = 1; }
            }
        }
        const bool exp[10] = {true,true,true,false,false,false,false,true,true,true};
        for (int i = 0; i < 10; ++i) if (masked[i] != exp[i]) {
            std::printf("FAIL edge1 mask id %d: got %d want %d\n", i, masked[i], exp[i]); g_fail = 1;
        }
    }
    // Case 2: negative offset → returns empty vector.
    {
        auto m = build_flat_audio_mask(/*offset=*/-1, /*count=*/4, /*eos=*/9, /*n_vocab=*/10);
        if (!m.empty()) {
            std::printf("FAIL edge2: negative offset should return empty, got %zu entries\n", m.size());
            g_fail = 1;
        }
    }
}

int main() {
    test_v2_two_ascii_words();
    test_v1_code_start_end();
    test_v3_uses_word_alternation_and_features();
    test_escaping_and_non_ascii();
    test_flat_audio_mask();
    test_flat_audio_mask_edges();
    if (g_fail) { std::printf("tts_constraint_test: FAILURES\n"); return 1; }
    std::printf("tts_constraint_test: all passed\n");
    return 0;
}
