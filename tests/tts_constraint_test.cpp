// Standalone assert test for the model-less TTS constraint logic.
// Returns 0 on success, non-zero (and prints) on the first failure.
#include "codec_common.h"
#include <cstdio>
#include <string>

using codec_common::outetts_version;
using codec_common::outetts_build_grammar;

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

static void test_v2_two_ascii_words() {
    const std::string g = outetts_build_grammar(outetts_version::V2, "hello world");
    const std::string want =
        "root       ::= NL? \"hello\" audioBlock \"world\" audioEnd NL EOS?\n"
        "audioBlock ::= TIME CODE* space NL?\n"
        "TEXT ::= [A-Za-z0-9 .,?!]+\n"
        "EOS      ::= \"<|im_end|>\"\n"
        "emotionStart ::= \"<|emotion_start|>\"\n"
        "emotionEnd ::= \"<|emotion_end|>\"\n"
        "audioEnd   ::= \"<|audio_end|>\"\n"
        "space      ::= \"<|space|>\"\n"
        "WORD       ::= \"hello\" | \"world\"\n"
        "NL         ::= [\\n]\n"
        "TIME  ::= \"<|t_\" DECIMAL \"|>\"\n"
        "CODE    ::= \"<|\" DIGITS \"|>\"\n"
        "DIGITS     ::= [0-9]+\n"
        "DECIMAL    ::= [0-9]+ \".\" [0-9]+\n"
        "punch ::= \"<|\" [a-z_]+ \"|>\"\n";
    CHECK_EQ(g, want);
}

static void test_v3_uses_word_alternation_and_features() {
    const std::string g = outetts_build_grammar(outetts_version::V3, "hi there");
    const std::string want =
        "root       ::= leadWord wordBlock* audioEnd NL EOS?\n"
        "leadWord ::= WORD audioBlock\n"
        "wordBlock ::= wordStart WORD audioBlock\n"
        "audioBlock ::= codeBlock wordEnd NL?\n"
        "codeBlock ::= features TIME energy spectralCentroid pitch CODE CODES*\n"
        "TEXT ::= [A-Za-z0-9.,!?]+\n"
        "EOS      ::= \"<|im_end|>\"\n"
        "audioEnd   ::= \"<|audio_end|>\"\n"
        "wordStart ::= \"<|word_start|>\"\n"
        "wordEnd ::= \"<|word_end|>\"\n"
        "features ::= \"<|features|>\"\n"
        "energy ::= \"<|energy_\" DIGITS \"|>\"\n"
        "spectralCentroid ::= \"<|spectral_centroid_\" DIGITS \"|>\"\n"
        "pitch ::= \"<|pitch_\" DIGITS \"|>\"\n"
        "WORD       ::= \"hi\" | \"there\"\n"
        "NL         ::= [\\n]\n"
        "TIME       ::= \"<|t_\" DECIMAL \"|>\"\n"
        "CODE       ::= \"<|code|>\"\n"
        "CODES      ::= CODE1 CODE2\n"
        "CODE1      ::= \"<|c1_\" DIGITS \"|>\"\n"
        "CODE2      ::= \"<|c2_\" DIGITS \"|>\"\n"
        "DIGITS     ::= [0-9]+\n"
        "DECIMAL    ::= [0-9]+ \".\" [0-9]+\n";
    CHECK_EQ(g, want);
}

static void test_escaping_and_non_ascii() {
    // ASCII word containing a quote → escaped (safe deviation from reference).
    const std::string q = outetts_build_grammar(outetts_version::V2, "sa\"y");
    // WORD line holds the single escaped literal.
    if (q.find("WORD       ::= \"sa\\\"y\"\n") == std::string::npos) {
        std::printf("FAIL escaping: %s\n", q.c_str()); g_fail = 1;
    }
    // Non-ASCII (CJK) word → [unique-codepoints]+ preserving first-seen order.
    // "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xBD\xA0" == U+4F60 U+597D U+4F60 (你好你)
    const std::string c = outetts_build_grammar(outetts_version::V2,
                                                "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xBD\xA0");
    if (c.find("WORD       ::= [\xE4\xBD\xA0\xE5\xA5\xBD]+\n") == std::string::npos) {
        std::printf("FAIL non-ascii: %s\n", c.c_str()); g_fail = 1;
    }
}

int main() {
    test_v2_two_ascii_words();
    test_v3_uses_word_alternation_and_features();
    test_escaping_and_non_ascii();
    if (g_fail) { std::printf("tts_constraint_test: FAILURES\n"); return 1; }
    std::printf("tts_constraint_test: all passed\n");
    return 0;
}
