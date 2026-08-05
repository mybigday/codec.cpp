// neutts_detect_test.cpp — TDD test for detect_neutts + neutts_audio_token_range.
//
// Gate: NEUTTS_GGUF env var must point to a NeuTTS backbone gguf.
// SKIP (exit 0) if unset or file missing.
//
// Oracle (neutts-nano, Task 0 Step 2):
//   detect_neutts          == true
//   neutts_audio_token_range -> offset=128262, count=65536, eos_id=128261
//
// Negative gate: NEUTTS_NEG_GGUF (default: models/moss_audio_nano/moss_audio_nano.gguf)
//   detect_neutts          == false

// Include llama.h FIRST so tts_runner.h sees LLAMA_H defined and skips its
// replicated llama_logit_bias typedef (which would otherwise conflict).
#include <llama.h>
#include "tts_runner.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

static bool file_exists(const char * path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static int g_fail = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);          \
            g_fail = 1;                                                         \
        }                                                                       \
    } while (0)

#define CHECK_EQ_I32(actual, expected)                                          \
    do {                                                                        \
        int32_t a_ = (actual);                                                  \
        int32_t e_ = (expected);                                                \
        if (a_ != e_) {                                                         \
            std::printf("FAIL %s:%d: got %d want %d\n",                        \
                        __FILE__, __LINE__, a_, e_);                            \
            g_fail = 1;                                                         \
        }                                                                       \
    } while (0)

int main() {
    // ── Positive test ────────────────────────────────────────────────────────
    const char * neutts_gguf = std::getenv("NEUTTS_GGUF");
    if (!neutts_gguf || !file_exists(neutts_gguf)) {
        std::printf("SKIP neutts_detect_test: NEUTTS_GGUF not set or file missing\n");
        return 0;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(neutts_gguf, mparams);
    if (!model) {
        std::printf("FAIL: could not load backbone from %s\n", neutts_gguf);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // detect_neutts must return true for NeuTTS backbone
    CHECK(codec_common::detect_neutts(vocab), "detect_neutts should be true for NeuTTS backbone");

    // neutts_audio_token_range must return correct oracle values
    int32_t offset = -1, count = -1, eos_id = -1;
    bool ok = codec_common::neutts_audio_token_range(vocab, &offset, &count, &eos_id);
    CHECK(ok, "neutts_audio_token_range should return true");

    std::printf("[neutts_detect_test] offset=%d  count=%d  eos_id=%d\n",
                offset, count, eos_id);

    CHECK_EQ_I32(offset, 128262);
    CHECK_EQ_I32(count,  65536);
    CHECK_EQ_I32(eos_id, 128261);

    llama_model_free(model);

    // ── Negative test (optional) ─────────────────────────────────────────────
    // Default fixture: a non-NeuTTS LLM backbone that llama.cpp can load.
    // (moss_audio_nano.gguf has arch 'moss_audio_tokenizer' which is unknown
    // to llama.cpp — it cannot be loaded at all, even with vocab_only=false.
    // Use CSM's llama backbone or any other known-arch backbone instead.)
    const char * neg_gguf = std::getenv("NEUTTS_NEG_GGUF");
    const char * default_neg = "models/csm/llama_backbone.gguf";
    if (!neg_gguf) neg_gguf = default_neg;

    if (!file_exists(neg_gguf)) {
        std::printf("[neutts_detect_test] SKIP negative (no neg gguf): %s not found\n", neg_gguf);
    } else {
        // Load WITHOUT vocab_only so that models requiring full init also work.
        llama_model_params neg_mparams = llama_model_default_params();
        neg_mparams.vocab_only = false;
        neg_mparams.n_gpu_layers = 0;
        llama_model * neg_model = llama_model_load_from_file(neg_gguf, neg_mparams);
        if (!neg_model) {
            // Present but unloadable — this is a broken fixture, not a valid skip.
            std::fprintf(stderr,
                "[neutts_detect_test] WARN: negative fixture present but failed to load: %s\n"
                "  (If this is a codec/non-LLM model, it cannot be used as a neg fixture;\n"
                "   set NEUTTS_NEG_GGUF to a loadable LLM backbone gguf instead.)\n",
                neg_gguf);
            return 1;
        }
        const llama_vocab * neg_vocab = llama_model_get_vocab(neg_model);
        CHECK(!codec_common::detect_neutts(neg_vocab),
              "detect_neutts should be false for non-NeuTTS backbone");
        std::printf("[neutts_detect_test] negative case: detect_neutts=false PASS\n");
        llama_model_free(neg_model);
    }

    if (g_fail) {
        std::printf("neutts_detect_test: FAILURES\n");
        return 1;
    }
    std::printf("neutts_detect_test: all passed\n");
    return 0;
}
