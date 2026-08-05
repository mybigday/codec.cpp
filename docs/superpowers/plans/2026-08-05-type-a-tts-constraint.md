# Type A TTS Constraint (auto-grammar + logit-bias) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically constrain the TTS backbone sampler for Type A token-based models — a prompt-dependent GBNF for structured models (OuteTTS V2/V3) and an ID logit-bias mask for flat direct-audio models — leaving MOSS-TTSD's grammar unchanged.

**Architecture:** Pure `(version, text) → GBNF` string builders live in codec_common (no llama dependency, reusable + testable without a model). The backbone-linked `tts_runner` owns a `tts_constraint {grammar, logit_bias}`, detects the OuteTTS family/version from backbone metadata, materializes the flat logit-bias mask (`llama_logit_bias`), and wires either into the existing `common_sampler` (`sp.grammar` / `sp.logit_bias`).

**Tech Stack:** C++17, ggml, llama.cpp (`common_sampler`, `common_params_sampling`), CMake. Tests: one standalone C++ binary + a thin Python subprocess wrapper (repo has no C++ unit framework).

## Global Constraints

- `common/audio_lm.cpp` (codec_common) MUST NOT include `llama.h` or reference `llama_*` types — it is built WITHOUT the backbone. Anything needing `llama_logit_bias` / `llama_vocab` lives in `common/tts_runner.cpp`.
- `cmake --build build -j` is the source of truth; inline clangd diagnostics are unreliable (no `compile_commands.json`).
- The `tts-cli` public surface stays `info`/`decode`/`synthesize` only — do NOT add debug subcommands.
- Port the OuteTTS grammar templates VERBATIM from the reference (`docs/superpowers/specs/2026-08-05-type-a-tts-constraint-design.md`), including intentional quirks (V2 hard-codes word order; V3 uses `WORD` alternation). Do NOT "tighten" V3.
- Memory-literal casts: any byte size uses `(size_t)` casts (e.g. `(size_t) 8 * 1024 * 1024 * 1024`).
- New backbone-only code compiles only under `-DCODEC_TTS_BACKBONE=ON`.

---

### Task 1: Pure OuteTTS grammar builders + test harness

**Files:**
- Modify: `common/codec_common.h` (add `outetts_version` enum + `outetts_build_grammar` decl, near `tts_auto_grammar` at :498)
- Modify: `common/audio_lm.cpp` (add `word_to_grammar` + `outetts_build_grammar`, near `tts_auto_grammar` at :1164)
- Create: `tests/tts_constraint_test.cpp` (assert-based C++ test, returns non-zero on failure)
- Create: `tests/e2e/tts_constraint_smoke.py` (subprocess wrapper for the existing e2e pattern)
- Modify: `CMakeLists.txt` (add `tts-constraint-test` executable under `CODEC_TTS_BACKBONE`, mirroring `tts-cli`'s link libraries at :173)

**Interfaces:**
- Produces: `enum class codec_common::outetts_version { V2, V3 };`
- Produces: `std::string codec_common::outetts_build_grammar(outetts_version v, const std::string & text);`

- [ ] **Step 1: Declare the API in `common/codec_common.h`**

Immediately above the `tts_auto_grammar` declaration (currently `common/codec_common.h:498`), add:

```cpp
// OuteTTS interface version — selects the grammar template family.
enum class outetts_version { V2, V3 };

// Build the prompt-dependent GBNF for an OuteTTS backbone. Pure string:
// splits `text` on ASCII whitespace into words, converts each to a GBNF
// terminal (ASCII → quoted literal with " and \ escaped; non-ASCII →
// [unique-codepoints]+), and substitutes them into the V2 or V3 template.
// No llama dependency — callable from hook hosts (rn-tts) and tests without
// a backbone. Faithful port of BricksDisplay/OuteTTS-Speaker-Creator app.py.
std::string outetts_build_grammar(outetts_version v, const std::string & text);
```

- [ ] **Step 2: Write the failing test `tests/tts_constraint_test.cpp`**

```cpp
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
```

- [ ] **Step 3: Wire the CMake target**

In `CMakeLists.txt`, inside the `if(CODEC_TTS_BACKBONE)` block that defines `tts-cli` (near :173), after the `tts-cli` executable add:

```cmake
    add_executable(tts-constraint-test tests/tts_constraint_test.cpp)
    target_link_libraries(tts-constraint-test PRIVATE $<TARGET_PROPERTY:tts-cli,LINK_LIBRARIES>)
    target_include_directories(tts-constraint-test PRIVATE $<TARGET_PROPERTY:tts-cli,INCLUDE_DIRECTORIES>)
```

(If `$<TARGET_PROPERTY:...>` proves awkward, copy `tts-cli`'s explicit `target_link_libraries` / `target_include_directories` lines verbatim for `tts-constraint-test`.)

- [ ] **Step 4: Build and verify the test FAILS to link/compile (function undefined)**

Run:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCODEC_TTS_BACKBONE=ON
cmake --build build --target tts-constraint-test -j
```
Expected: FAIL — unresolved reference to `codec_common::outetts_build_grammar`.

- [ ] **Step 5: Implement `word_to_grammar` + `outetts_build_grammar` in `common/audio_lm.cpp`**

Add just above `tts_auto_grammar` (currently `common/audio_lm.cpp:1164`), inside the existing namespace/TU (add `#include <vector>` / `<string>` if not present):

```cpp
// Split on ASCII whitespace (space/tab/newline/CR), dropping empties.
static std::vector<std::string> oute_split_ws(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                         c == '\f' || c == '\v');
        if (ws) { if (!cur.empty()) { out.push_back(cur); cur.clear(); } }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// One word → a GBNF terminal, mirroring the reference word_to_grammar:
//   all-ASCII  → "word"  (with " and \ escaped — safe hardening)
//   otherwise  → [unique-codepoints]+  (first-seen order, UTF-8 bytes re-emitted)
static std::string word_to_grammar(const std::string & word) {
    bool ascii = true;
    for (unsigned char c : word) if (c >= 128) { ascii = false; break; }
    if (ascii) {
        std::string esc;
        for (char c : word) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
        return "\"" + esc + "\"";
    }
    // Decode UTF-8 into codepoint byte-spans; dedupe by codepoint, keep order.
    std::string cls;
    std::vector<std::string> seen;   // unique codepoints as their UTF-8 bytes
    size_t i = 0;
    while (i < word.size()) {
        unsigned char b = (unsigned char) word[i];
        size_t len = 1;
        if      ((b & 0x80) == 0x00) len = 1;
        else if ((b & 0xE0) == 0xC0) len = 2;
        else if ((b & 0xF0) == 0xE0) len = 3;
        else if ((b & 0xF8) == 0xF0) len = 4;
        if (i + len > word.size()) len = 1;         // malformed tail — byte-wise
        std::string cp = word.substr(i, len);
        bool dup = false;
        for (const auto & s : seen) if (s == cp) { dup = true; break; }
        if (!dup) { seen.push_back(cp); cls += cp; }
        i += len;
    }
    return "[" + cls + "]+";
}

std::string outetts_build_grammar(outetts_version v, const std::string & text) {
    std::vector<std::string> wg;
    for (const auto & w : oute_split_ws(text)) wg.push_back(word_to_grammar(w));

    auto join = [](const std::vector<std::string> & xs, const std::string & sep) {
        std::string o;
        for (size_t k = 0; k < xs.size(); ++k) { if (k) o += sep; o += xs[k]; }
        return o;
    };
    const std::string word_alt = join(wg, " | ");

    std::string g;
    if (v == outetts_version::V2) {
        g += "root       ::= NL? " + join(wg, " audioBlock ") + " audioEnd NL EOS?\n";
        g += "audioBlock ::= TIME CODE* space NL?\n";
        g += "TEXT ::= [A-Za-z0-9 .,?!]+\n";
        g += "EOS      ::= \"<|im_end|>\"\n";
        g += "emotionStart ::= \"<|emotion_start|>\"\n";
        g += "emotionEnd ::= \"<|emotion_end|>\"\n";
        g += "audioEnd   ::= \"<|audio_end|>\"\n";
        g += "space      ::= \"<|space|>\"\n";
        g += "WORD       ::= " + word_alt + "\n";
        g += "NL         ::= [\\n]\n";
        g += "TIME  ::= \"<|t_\" DECIMAL \"|>\"\n";
        g += "CODE    ::= \"<|\" DIGITS \"|>\"\n";
        g += "DIGITS     ::= [0-9]+\n";
        g += "DECIMAL    ::= [0-9]+ \".\" [0-9]+\n";
        g += "punch ::= \"<|\" [a-z_]+ \"|>\"\n";
        return g;
    }
    // V3
    g += "root       ::= leadWord wordBlock* audioEnd NL EOS?\n";
    g += "leadWord ::= WORD audioBlock\n";
    g += "wordBlock ::= wordStart WORD audioBlock\n";
    g += "audioBlock ::= codeBlock wordEnd NL?\n";
    g += "codeBlock ::= features TIME energy spectralCentroid pitch CODE CODES*\n";
    g += "TEXT ::= [A-Za-z0-9.,!?]+\n";
    g += "EOS      ::= \"<|im_end|>\"\n";
    g += "audioEnd   ::= \"<|audio_end|>\"\n";
    g += "wordStart ::= \"<|word_start|>\"\n";
    g += "wordEnd ::= \"<|word_end|>\"\n";
    g += "features ::= \"<|features|>\"\n";
    g += "energy ::= \"<|energy_\" DIGITS \"|>\"\n";
    g += "spectralCentroid ::= \"<|spectral_centroid_\" DIGITS \"|>\"\n";
    g += "pitch ::= \"<|pitch_\" DIGITS \"|>\"\n";
    g += "WORD       ::= " + word_alt + "\n";
    g += "NL         ::= [\\n]\n";
    g += "TIME       ::= \"<|t_\" DECIMAL \"|>\"\n";
    g += "CODE       ::= \"<|code|>\"\n";
    g += "CODES      ::= CODE1 CODE2\n";
    g += "CODE1      ::= \"<|c1_\" DIGITS \"|>\"\n";
    g += "CODE2      ::= \"<|c2_\" DIGITS \"|>\"\n";
    g += "DIGITS     ::= [0-9]+\n";
    g += "DECIMAL    ::= [0-9]+ \".\" [0-9]+\n";
    return g;
}
```

- [ ] **Step 6: Build and run the test — expect PASS**

Run:
```bash
cmake --build build --target tts-constraint-test -j && ./build/tts-constraint-test
```
Expected: `tts_constraint_test: all passed`, exit 0.

- [ ] **Step 7: Add the Python e2e wrapper `tests/e2e/tts_constraint_smoke.py`**

```python
#!/usr/bin/env python3
"""E2E wrapper: runs the model-less C++ constraint test binary.

Run from repo root:  .venv/bin/python tests/e2e/tts_constraint_smoke.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "build" / "tts-constraint-test"

def main() -> int:
    if not BIN.exists():
        print(f"not built: {BIN}; run "
              f"cmake --build build --target tts-constraint-test")
        return 1
    proc = subprocess.run([str(BIN)], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run the wrapper**

Run: `.venv/bin/python tests/e2e/tts_constraint_smoke.py`
Expected: prints `tts_constraint_test: all passed`, exit 0.

- [ ] **Step 9: Commit**

```bash
git add common/codec_common.h common/audio_lm.cpp tests/tts_constraint_test.cpp \
        tests/e2e/tts_constraint_smoke.py CMakeLists.txt
git commit -m "feat(tts): pure OuteTTS V2/V3 GBNF builders + model-less constraint test"
```

---

### Task 2: Flat logit-bias mask builder (tts_runner)

**Files:**
- Modify: `common/tts_runner.h` (declare `build_flat_audio_mask`)
- Modify: `common/tts_runner.cpp` (implement it; include the struct decl above `SamplerChain`)
- Modify: `tests/tts_constraint_test.cpp` (add membership assertions)

**Interfaces:**
- Produces: `struct tts_constraint { std::string grammar; std::vector<llama_logit_bias> logit_bias; };`
- Produces: `std::vector<llama_logit_bias> build_flat_audio_mask(int32_t offset, int32_t count, int32_t eos_id, int32_t n_vocab);` — every id NOT in `[offset, offset+count) ∪ {eos_id≥0}` gets bias `-INFINITY`; allowed ids are omitted (unbiased). Returns empty if `offset < 0`.

- [ ] **Step 1: Declare in `common/tts_runner.h`**

Add near the top (after the existing includes; `tts_runner.h` already pulls llama types transitively via the backbone build — if not, add `#include "llama.h"`):

```cpp
// A resolved backbone-sampler constraint. Exactly one field is populated for
// a constrained model; both empty = unconstrained. Grammar drives structured
// models (OuteTTS / MOSS-TTSD); logit_bias masks flat direct-audio Type A.
struct tts_constraint {
    std::string                   grammar;
    std::vector<llama_logit_bias> logit_bias;
};

// Mask every token OUTSIDE [offset, offset+count) ∪ {eos_id} to -INFINITY.
// eos_id < 0 = no sentinel. offset < 0 → returns {} (Type A disabled).
std::vector<llama_logit_bias> build_flat_audio_mask(int32_t offset, int32_t count,
                                                    int32_t eos_id, int32_t n_vocab);
```

- [ ] **Step 2: Add failing membership test to `tests/tts_constraint_test.cpp`**

Add `#include "tts_runner.h"` and `#include <cmath>` at the top of the test file, then this test + call it from `main`:

```cpp
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
```

- [ ] **Step 3: Build — expect link failure (undefined `build_flat_audio_mask`)**

Run: `cmake --build build --target tts-constraint-test -j`
Expected: FAIL — unresolved `build_flat_audio_mask`.

- [ ] **Step 4: Implement in `common/tts_runner.cpp`**

Place the `tts_constraint` struct + this function above `SamplerChain` (near `common/tts_runner.cpp:140`), after the includes:

```cpp
std::vector<llama_logit_bias> build_flat_audio_mask(int32_t offset, int32_t count,
                                                    int32_t eos_id, int32_t n_vocab) {
    std::vector<llama_logit_bias> out;
    if (offset < 0 || count <= 0 || n_vocab <= 0) return out;
    const int32_t lo = offset;
    const int32_t hi = offset + count;                 // exclusive
    out.reserve((size_t) n_vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        const bool allowed = (id >= lo && id < hi) || (eos_id >= 0 && id == eos_id);
        if (!allowed) out.push_back(llama_logit_bias{ id, -INFINITY });
    }
    return out;
}
```

- [ ] **Step 5: Build and run — expect PASS**

Run: `cmake --build build --target tts-constraint-test -j && ./build/tts-constraint-test`
Expected: `all passed`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add common/tts_runner.h common/tts_runner.cpp tests/tts_constraint_test.cpp
git commit -m "feat(tts): flat audio-token logit-bias mask builder + test"
```

---

### Task 3: Surface the Type A id-range in `audio_lm_prompt_info`

**Files:**
- Modify: `common/codec_common.h` (add 3 fields to `struct audio_lm_prompt_info` + `KIND_TOKEN_SINGLE_CB`)
- Modify: `common/audio_lm.cpp` (populate them in `audio_lm_get_prompt_info`, from the ctx fields)

**Interfaces:**
- Produces: `audio_lm_prompt_info::KIND_TOKEN_SINGLE_CB`, and fields `int32_t audio_tok_offset/audio_tok_count/audio_tok_eos` mirroring the private ctx fields (`common/audio_lm.cpp:90-92`).

- [ ] **Step 1: Add the enum value + fields in `common/codec_common.h`**

In `struct audio_lm_prompt_info` (`common/codec_common.h:383`): add `KIND_TOKEN_SINGLE_CB` to the `enum kind` (after `KIND_CONTINUOUS_CFM`), and add near the other id fields:

```cpp
    // Type A (token-based single-cb) audio-token range, mirrored from the
    // codec_lm context so backbone-side code (tts_runner) can build a flat
    // logit-bias mask without reaching into the private context.
    // audio_tok_offset < 0 → not a Type A model.
    int32_t audio_tok_offset = -1;
    int32_t audio_tok_count  = 0;
    int32_t audio_tok_eos    = -1;
```

- [ ] **Step 2: Populate them in `audio_lm_get_prompt_info` (`common/audio_lm.cpp:907`)**

Inside `audio_lm_get_prompt_info`, after the `pi` struct is otherwise filled and before returning, copy from the ctx (fields at `common/audio_lm.cpp:90-92`) and set the kind when in range:

```cpp
    // Mirror the Type A audio-token range into pi (see codec_common.h).
    out->audio_tok_offset = ctx->audio_tok_offset;
    out->audio_tok_count  = ctx->audio_tok_count;
    out->audio_tok_eos    = ctx->audio_tok_eos;
    if (ctx->audio_tok_offset >= 0 &&
        out->model_kind == audio_lm_prompt_info::KIND_UNKNOWN) {
        out->model_kind = audio_lm_prompt_info::KIND_TOKEN_SINGLE_CB;
    }
```

(Match the actual local/param name used by `audio_lm_get_prompt_info` — confirm whether it writes into a pointer `out` or a reference; use that name.)

- [ ] **Step 3: Build the whole project (no test yet — plumbing consumed in Task 5)**

Run: `cmake --build build -j`
Expected: builds clean (both with and without `CODEC_TTS_BACKBONE`).

- [ ] **Step 4: Commit**

```bash
git add common/codec_common.h common/audio_lm.cpp
git commit -m "feat(tts): mirror Type A audio-token range into audio_lm_prompt_info"
```

---

### Task 4: OuteTTS family/version detection from backbone metadata

**Files:**
- Modify: `common/tts_runner.cpp` (add `detect_outetts_version`, near the model-load region at :1099-1135)

**Interfaces:**
- Produces: `bool detect_outetts_version(const llama_model * m, const llama_vocab * vocab, codec_common::outetts_version * out);` — returns true + sets `*out` when the backbone is identified as OuteTTS; false otherwise. Detection is by vocab signature tokens (corroborated by `general.name` when present).

**Note on testability:** detection needs a real backbone vocab, so it is NOT covered by the model-less test — its e2e validation is DEFERRED to when an OuteTTS backbone gguf exists. Implement conservatively (fail closed → false → no auto-grammar) and log the decision.

- [ ] **Step 1: Implement `detect_outetts_version` in `common/tts_runner.cpp`**

Add above `run_synthesize` / the resolve site (before `common/tts_runner.cpp:1099`):

```cpp
// Identify an OuteTTS backbone from vocab signature tokens. V3 carries
// per-word feature/paired-code markers; V2 carries <|space|> + bare code
// tokens and lacks the V3 markers. Fail closed (return false) on anything
// ambiguous so non-OuteTTS models are never mis-constrained.
static bool detect_outetts_version(const llama_model * m, const llama_vocab * vocab,
                                   codec_common::outetts_version * out) {
    if (!vocab || !out) return false;
    auto has = [&](const char * piece) -> bool {
        // token id for an exact special-token piece; -1 if absent.
        return llama_vocab_get_token(vocab, piece) >= 0;   // see NOTE below
    };
    const bool v3 = has("<|word_start|>") || has("<|features|>") || has("<|c1_0|>");
    const bool v2 = has("<|space|>") && has("<|audio_end|>") && !v3;
    // Corroborate with general.name when available (non-fatal).
    char name[256] = {0};
    llama_model_meta_val_str(m, "general.name", name, sizeof(name));
    if (v3) { *out = codec_common::outetts_version::V3; return true; }
    if (v2) { *out = codec_common::outetts_version::V2; return true; }
    (void) name;
    return false;
}
```

NOTE for the implementer: verify the exact llama.cpp vocab-lookup call available in the pinned tree. Candidates: a direct `llama_vocab_get_token(vocab, "<|...|>")`, or tokenizing the single special piece with `parse_special=true` and checking it yields exactly one id. Pick whichever exists (grep `llama.h` for `llama_vocab_` token lookups); keep the `has()` lambda as the single point that depends on it.

- [ ] **Step 2: Build**

Run: `cmake --build build --target tts-cli -j`
Expected: builds clean.

- [ ] **Step 3: Commit**

```bash
git add common/tts_runner.cpp
git commit -m "feat(tts): detect OuteTTS V2/V3 from backbone vocab signature tokens"
```

---

### Task 5: Resolve + apply the constraint; init validation; call-site swap

**Files:**
- Modify: `common/tts_runner.cpp` — `resolve_constraint`; `BackboneSampler::build` (+ logit_bias); `run_codebook_ar` signature + call to `build`; init validation after vocab load (:1135); replace the grammar resolution at :1251-1252 and the `run_codebook_ar` call at :1295.

**Interfaces:**
- Consumes: `outetts_build_grammar` (Task 1), `build_flat_audio_mask` + `tts_constraint` (Task 2), `pi.audio_tok_*` (Task 3), `detect_outetts_version` (Task 4), `BackboneSampler::build` (`common/tts_runner.cpp:147`), `run_codebook_ar` (`:707`,`:1295`).
- Produces: `tts_constraint resolve_constraint(const llama_model*, const llama_vocab*, const audio_lm_prompt_info&, const std::string & user_grammar, const std::string & text, int32_t n_vocab);`

- [ ] **Step 1: Add `logit_bias` to `BackboneSampler::build`**

In `BackboneSampler::build` (`common/tts_runner.cpp:147`), add a parameter and set it on `sp`. New signature + the two added lines:

```cpp
    bool build(const llama_model * model, uint32_t seed, float temp,
               int32_t top_k, float top_p, float min_p, float rep_penalty,
               int32_t rep_last_n, const std::string & grammar,
               const std::vector<llama_logit_bias> & logit_bias,   // NEW
               std::string * err) {
        // ... existing sp setup ...
        if (!grammar.empty()) {
            sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
        }
        if (!logit_bias.empty()) {                                  // NEW
            sp.logit_bias = logit_bias;                             // NEW
        }
        // ... existing common_sampler_init ...
    }
```

- [ ] **Step 2: Thread the constraint through `run_codebook_ar`**

Change `run_codebook_ar` (`common/tts_runner.cpp:707`) to accept `const tts_constraint & constraint` in place of the current `const std::string & grammar` parameter. At its internal `bbsmpl.build(...)` call (`:815`), pass both fields:

```cpp
        if (!bbsmpl.build(lmodel, seed, temp, top_k, top_p, /*min_p=*/0.0f,
                          /*rep_penalty=*/1.0f, /*rep_last_n=*/0,
                          constraint.grammar, constraint.logit_bias, &berr)) {
```

- [ ] **Step 3: Implement `resolve_constraint`**

Add above the call site (before `common/tts_runner.cpp:1251`):

```cpp
static tts_constraint resolve_constraint(const llama_model * lmodel,
                                         const llama_vocab * vocab,
                                         const audio_lm_prompt_info & pi,
                                         const std::string & user_grammar,
                                         const std::string & text,
                                         int32_t n_vocab) {
    tts_constraint c;
    // 1) explicit user grammar wins.
    if (!user_grammar.empty()) { c.grammar = user_grammar; return c; }
    // 2) structured backbone family (OuteTTS) → prompt-dependent grammar.
    codec_common::outetts_version ov;
    if (detect_outetts_version(lmodel, vocab, &ov)) {
        c.grammar = codec_common::outetts_build_grammar(ov, text);
        return c;
    }
    // 3a) codec-metadata grammar (MOSS-TTSD; unchanged).
    c.grammar = tts_auto_grammar(pi, text);
    if (!c.grammar.empty()) return c;
    // 3b) flat Type A → logit-bias mask.
    if (pi.audio_tok_offset >= 0) {
        c.logit_bias = build_flat_audio_mask(pi.audio_tok_offset, pi.audio_tok_count,
                                             pi.audio_tok_eos, n_vocab);
    }
    return c;
}
```

- [ ] **Step 4: Replace the resolution block + `run_codebook_ar` call**

Replace `common/tts_runner.cpp:1251-1252` (the `std::string grammar = ... tts_auto_grammar(...)`) with:

```cpp
    const int32_t n_vocab_bb = llama_vocab_n_tokens(vocab);
    const tts_constraint constraint =
        resolve_constraint(lmodel, vocab, pi, a.grammar, a.text, n_vocab_bb);
    if (!constraint.grammar.empty()) {
        std::printf("grammar: %s (%zu bytes)\n",
                    !a.grammar.empty() ? "user-supplied" : "auto (model-derived)",
                    constraint.grammar.size());
    } else if (!constraint.logit_bias.empty()) {
        std::printf("logit-bias: flat audio mask (%zu tokens masked)\n",
                    constraint.logit_bias.size());
    }
```

Then update the `run_codebook_ar(...)` call (`:1295`) to pass `constraint` instead of `grammar`. (The continuous / sequential / streaming branches do not take a grammar and are unchanged.)

- [ ] **Step 5: Add init-time validation after vocab load**

Right after `const llama_vocab * vocab = llama_model_get_vocab(lmodel);` (`common/tts_runner.cpp:1135`), add:

```cpp
    // Type A sanity: a mis-counted codec.audio_token.offset (added-token order
    // drift in the converter) produces in-range-but-wrong ids → silent garbage.
    // Surface it: bound-check the range and log the range's edge pieces.
    {
        int32_t a_off = -1, a_cnt = 0, a_eos = -1;
        audio_lm_get_audio_token_range(ctx, &a_off, &a_cnt, &a_eos);
        if (a_off >= 0) {
            const int32_t nv = llama_vocab_n_tokens(vocab);
            if (a_off + a_cnt > nv) {
                std::printf("WARN: audio-token range [%d,%d) exceeds n_vocab=%d "
                            "(bad codec.audio_token.offset/count?)\n",
                            a_off, a_off + a_cnt, nv);
            }
            char pf[128] = {0}, pl[128] = {0};
            llama_token_to_piece(vocab, a_off,             pf, sizeof(pf), 0, true);
            llama_token_to_piece(vocab, a_off + a_cnt - 1, pl, sizeof(pl), 0, true);
            std::printf("audio-token range [%d,%d) eos=%d; first=<%s> last=<%s>\n",
                        a_off, a_off + a_cnt, a_eos, pf, pl);
        }
    }
```

(Confirm the pinned `llama_token_to_piece` arity via `llama.h`; adjust the trailing `lstrip/special` args to match.)

- [ ] **Step 6: Build everything**

Run: `cmake --build build -j`
Expected: builds clean with `-DCODEC_TTS_BACKBONE=ON`.

- [ ] **Step 7: Regression — MOSS-TTSD grammar path unchanged**

Run the existing synthesize smoke (drives `tts-cli synthesize`, exercises MOSS-TTSD which still gets its auto-grammar via `tts_auto_grammar` inside `resolve_constraint`):
```bash
.venv/bin/python tests/e2e/ttscli_synthesize_smoke.py
```
Expected: the MOSS-TTSD case still passes (same CER as before); overall pass count unchanged from baseline. Confirm the log now prints `grammar: auto (model-derived)` for MOSS-TTSD (behavior identical to before this change).

- [ ] **Step 8: Re-run the model-less constraint smoke (no regression)**

Run: `.venv/bin/python tests/e2e/tts_constraint_smoke.py`
Expected: `all passed`.

- [ ] **Step 9: Commit**

```bash
git add common/tts_runner.cpp
git commit -m "feat(tts): auto-resolve Type A backbone constraint (OuteTTS grammar / flat mask) + init validation"
```

---

## Self-Review notes (author)

- **Spec coverage:** abstraction (T2), resolution order (T5), location split (T1 codec_common builders / T2,T4,T5 tts_runner), pi fields (T3), OuteTTS V2/V3 verbatim (T1), flat logit-bias (T2), detection (T4), init validation (T5), SamplerChain wiring (T5), testing (T1/T2 binary + wrapper; MOSS regression T5). MOSS-TTSD unchanged output guarded by T5 Step 7.
- **Deferred (documented, needs assets):** detection e2e (T4), OuteTTS end-to-end audio, true GBNF parse+token-match. Not silently dropped — called out in T4 and the spec.
- **Type consistency:** `outetts_version`, `outetts_build_grammar`, `tts_constraint`, `build_flat_audio_mask`, `resolve_constraint`, `detect_outetts_version`, `audio_tok_offset/count/eos` used consistently across tasks.
- **Implementer verification points (pinned-tree API arities):** vocab special-token lookup in T4 Step 1; `llama_token_to_piece` args in T5 Step 5; `audio_lm_get_prompt_info` out-param name in T3 Step 2. Each is localized and flagged.
