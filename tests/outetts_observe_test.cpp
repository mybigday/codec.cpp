// Model-less unit test for the multi-codebook Type A observe_token accumulator.
// Tests audio_lm_set_audio_token_ranges + audio_lm_observe_token routing.
// No GGUF, no model — only codec_common routing logic.
// Returns 0 on success, non-zero on first failure.

#include "codec_common.h"
#include <cstdio>
#include <cstring>

using namespace codec_common;

static int g_fail = 0;

#define CHECK(cond)                                                           \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

#define CHECK_EQ_INT(actual, expected)                                        \
    do {                                                                      \
        int a_ = (int)(actual);                                               \
        int e_ = (int)(expected);                                             \
        if (a_ != e_) {                                                       \
            std::printf("FAIL %s:%d: got %d, want %d\n",                     \
                        __FILE__, __LINE__, a_, e_);                          \
            g_fail = 1;                                                       \
        }                                                                     \
    } while (0)

// ─── Test: 2-codebook synthetic stream ──────────────────────────────────────
//
// Config: c1=[1000,1004), c2=[2000,2004), sentinel=1500, eos=999.
// Stream:  1500  1000  2001  1500  1003  2002  999
//          sent  c1=0  c2=1  sent  c1=3  c2=2  eos
// Expected:
//   codes (layout [q*n_frames + t]):
//     cb0: frame0=0, frame1=3   → indices 0,1
//     cb1: frame0=1, frame1=2   → indices 2,3
//   codes_n_frames = 2
//   sentinels → OBSERVE_PASSTHROUGH
//   eos       → OBSERVE_STOP
static void test_two_cb_synthetic_stream() {
    // Construct a bare accumulator context via the new multi-cb range API.
    // audio_lm_set_audio_token_ranges requires a valid ctx pointer.
    // We construct a minimal audio_lm_context via the internal test helper:
    // we can reach it because audio_lm_context is in the same TU only when
    // linked, so we use the public C API instead.
    //
    // Strategy: use audio_lm_alloc_bare() if available; otherwise we must
    // either expose one or use the helper extraction approach.
    // Per the brief we call audio_lm_set_audio_token_ranges + observe_token
    // + audio_lm_codes_n_frames on a bare context.

    audio_lm_context * ctx = audio_lm_alloc_bare();
    if (!ctx) {
        std::printf("FAIL: audio_lm_alloc_bare() returned null\n");
        g_fail = 1;
        return;
    }

    // Configure 2 codebooks: offsets={1000,2000}, counts={4,4}, sentinel=1500, eos=999
    const int32_t offsets[2] = { 1000, 2000 };
    const int32_t counts[2]  = {    4,    4 };
    audio_lm_set_audio_token_ranges(ctx, offsets, counts, /*n_q=*/2,
                                    /*frame_sentinel_id=*/1500, /*eos_id=*/999);

    // Feed: [sentinel, c1=0, c2=1, sentinel, c1=3, c2=2, eos]
    // tok 1500 (sentinel) → PASSTHROUGH
    observe_action a0 = audio_lm_observe_token(ctx, 1500, nullptr, 0);
    CHECK_EQ_INT(a0, OBSERVE_PASSTHROUGH);

    // tok 1000 (c1=0, cb cursor 0) → CONSUMED, not yet a full frame
    observe_action a1 = audio_lm_observe_token(ctx, 1000, nullptr, 0);
    CHECK_EQ_INT(a1, OBSERVE_CONSUMED);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 0);  // frame not complete yet

    // tok 2001 (c2=1, cb cursor 1 → last cb → frame complete)
    observe_action a2 = audio_lm_observe_token(ctx, 2001, nullptr, 0);
    CHECK_EQ_INT(a2, OBSERVE_CONSUMED);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 1);  // frame 0 complete

    // tok 1500 (sentinel) → PASSTHROUGH, resets cursor
    observe_action a3 = audio_lm_observe_token(ctx, 1500, nullptr, 0);
    CHECK_EQ_INT(a3, OBSERVE_PASSTHROUGH);

    // tok 1003 (c1=3, cb cursor 0)
    observe_action a4 = audio_lm_observe_token(ctx, 1003, nullptr, 0);
    CHECK_EQ_INT(a4, OBSERVE_CONSUMED);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 1);  // still 1 frame

    // tok 2002 (c2=2, cb cursor 1 → last cb → frame 1 complete)
    observe_action a5 = audio_lm_observe_token(ctx, 2002, nullptr, 0);
    CHECK_EQ_INT(a5, OBSERVE_CONSUMED);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 2);  // 2 frames done

    // tok 999 (eos) → STOP
    observe_action a6 = audio_lm_observe_token(ctx, 999, nullptr, 0);
    CHECK_EQ_INT(a6, OBSERVE_STOP);

    // Validate accumulated codes in frame-major (T, n_cb) layout:
    //   codes[t*n_cb + q].  Matches codec_token_buffer convention.
    //   frame0: cb0=0, cb1=1  →  codes[0]=0, codes[1]=1
    //   frame1: cb0=3, cb1=2  →  codes[2]=3, codes[3]=2
    int32_t n_q = 0, n_frames = 0;
    const int32_t * codes = audio_lm_debug_codes(ctx, &n_q, &n_frames);
    CHECK(codes != nullptr);
    CHECK_EQ_INT(n_q,     2);
    CHECK_EQ_INT(n_frames, 2);
    if (codes) {
        // frame0, cb0 = codes[0*2 + 0] = codes[0] = 0
        CHECK_EQ_INT(codes[0], 0);
        // frame0, cb1 = codes[0*2 + 1] = codes[1] = 1
        CHECK_EQ_INT(codes[1], 1);
        // frame1, cb0 = codes[1*2 + 0] = codes[2] = 3
        CHECK_EQ_INT(codes[2], 3);
        // frame1, cb1 = codes[1*2 + 1] = codes[3] = 2
        CHECK_EQ_INT(codes[3], 2);
    }

    audio_lm_free_bare(ctx);
}

// ─── Test: non-audio token → PASSTHROUGH ─────────────────────────────────────
static void test_non_audio_passthrough() {
    audio_lm_context * ctx = audio_lm_alloc_bare();
    if (!ctx) { std::printf("FAIL: alloc_bare null\n"); g_fail = 1; return; }

    const int32_t offsets[1] = { 100 };
    const int32_t counts[1]  = {  50 };
    audio_lm_set_audio_token_ranges(ctx, offsets, counts, /*n_q=*/1,
                                    /*frame_sentinel_id=*/-1, /*eos_id=*/42);

    // A text/BOS token far outside the range
    observe_action a = audio_lm_observe_token(ctx, 5, nullptr, 0);
    CHECK_EQ_INT(a, OBSERVE_PASSTHROUGH);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 0);

    audio_lm_free_bare(ctx);
}

// ─── Test: single-cb fallback (n_q=1, no sentinel) ───────────────────────────
// Verifies NeuTTS-style (n_q=1) still works through the new path.
static void test_single_cb_no_sentinel() {
    audio_lm_context * ctx = audio_lm_alloc_bare();
    if (!ctx) { std::printf("FAIL: alloc_bare null\n"); g_fail = 1; return; }

    const int32_t offsets[1] = { 500 };
    const int32_t counts[1]  = { 100 };
    audio_lm_set_audio_token_ranges(ctx, offsets, counts, /*n_q=*/1,
                                    /*frame_sentinel_id=*/-1, /*eos_id=*/999);

    // Feed 3 audio tokens
    CHECK_EQ_INT(audio_lm_observe_token(ctx, 501, nullptr, 0), OBSERVE_CONSUMED);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 1);
    CHECK_EQ_INT(audio_lm_observe_token(ctx, 502, nullptr, 0), OBSERVE_CONSUMED);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 2);
    CHECK_EQ_INT(audio_lm_observe_token(ctx, 503, nullptr, 0), OBSERVE_CONSUMED);
    CHECK_EQ_INT(audio_lm_codes_n_frames(ctx), 3);

    // EOS
    CHECK_EQ_INT(audio_lm_observe_token(ctx, 999, nullptr, 0), OBSERVE_STOP);

    int32_t n_q = 0, n_frames = 0;
    const int32_t * codes = audio_lm_debug_codes(ctx, &n_q, &n_frames);
    CHECK(codes != nullptr);
    CHECK_EQ_INT(n_q,     1);
    CHECK_EQ_INT(n_frames, 3);
    if (codes) {
        CHECK_EQ_INT(codes[0], 1);   // 501 - 500
        CHECK_EQ_INT(codes[1], 2);   // 502 - 500
        CHECK_EQ_INT(codes[2], 3);   // 503 - 500
    }

    audio_lm_free_bare(ctx);
}

int main() {
    test_two_cb_synthetic_stream();
    test_non_audio_passthrough();
    test_single_cb_no_sentinel();

    if (g_fail) {
        std::printf("outetts_observe_test: FAILURES\n");
        return 1;
    }
    std::printf("outetts_observe_test: all passed\n");
    return 0;
}
