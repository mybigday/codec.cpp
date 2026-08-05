# NeuTTS-Air / Nano Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tts-cli synthesize` produce round-trip-correct speech (Text→TTS→Whisper→Text) for NeuTTS-Air and NeuTTS-Nano, validated against the BricksDisplay HF ggufs.

**Architecture:** Flat Type A (single-codebook NeuCodec tokens `<|speech_N|>`). Backbones convert as plain qwen2/llama (no new converter, no gguf metadata); `tts_runner` detects NeuTTS from the backbone vocab, derives the audio-token range from the vocab, runs a new `run_token_single_cb` loop (reusing `observe_token`/`decode_audio`), and assembles the ICL prompt (NeuCodec-encode ref audio → `<|speech_N|>`). espeak-ng phonemization lives only in tts-cli behind a CMake option.

**Tech Stack:** C++17, ggml, llama.cpp backbone, NeuCodec codec, CMake; espeak-ng (optional, tts-cli only); Python e2e (transformers-whisper, phonemizer).

## Global Constraints

- **No NeuTTS metadata on the codec gguf; no large params in any gguf.** NeuTTS specifics are derived from the backbone vocab at runtime (via `audio_lm_set_audio_token_range`).
- `common/audio_lm.cpp` / `common/codec_common.h` (codec_common) MUST NOT include `llama.h` / `llama_*` — backbone-dependent code lives in `common/tts_runner.cpp`.
- espeak-ng may be referenced ONLY from `examples/tts-cli.cpp`, guarded by CMake option `CODEC_TTS_ESPEAK` (default OFF). Core `libcodec`/`codec_common` never links espeak.
- Port prompt format VERBATIM from official `neuphonic/neutts-air` (`neutts/neutts.py::_ggml_prompt`), as pinned by Task 0 in `docs/superpowers/notes/neutts-facts.md` §Step 5: prompt `user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>` + ref `<|speech_{i}|>` codes; stop `<|SPEECH_GENERATION_END|>`; code token `<|speech_{i}|>`. (Task 0 correction: `<|SPEECH_GENERATION_START|>` sits between `assistant:` and the ref codes — the earlier draft omitted it.) `ref_text`/`input_text` are phonemized separately then joined with a single space.
- `cmake --build build -j` is the source of truth; ignore clangd diagnostics (no compile_commands.json).
- Reference asset must be CC0/public-domain (NOT neuphonic `samples/*.wav`).
- Memory-size literals use `(size_t)` casts.

---

### Task 0: Investigation — pin the numeric contract from official source (NO code)

**Deliverable:** `docs/superpowers/notes/neutts-facts.md` capturing the exact facts the build depends on. This task writes NO product code; it de-risks every later task.

**Files:**
- Create: `docs/superpowers/notes/neutts-facts.md`

- [ ] **Step 1: Confirm the NeuCodec variant + that codec.cpp matches it**

Download `neuphonic/neutts-air` config + tokenizer, and identify the codec repo it uses (`neuphonic/emilia-yodas-english-neucodec` per README). Compare its config (sample_rate, hop, codebook size, n_q) against codec.cpp's NeuCodec (`src/models/neucodec.*`, and the local `neucodec.gguf`). Record: codebook size (→ `count`), sample rate, whether codec.cpp's `codec_encode`/`codec_decode` for NeuCodec produce/consume the SAME integer code space. If the variant differs, note exactly how.

- [ ] **Step 2: Extract the audio-token IDs from the backbone vocab**

From the NeuTTS-Air tokenizer (`tokenizer.json`/`added_tokens`), record the token id of `<|speech_0|>`, the highest `<|speech_N|>`, whether the block is contiguous, and the id of `<|SPEECH_GENERATION_END|>`, `<|TEXT_PROMPT_START|>`, `<|TEXT_PROMPT_END|>`. These become `offset`/`count`/`eos_id` + detection signatures. Repeat for `neuphonic/neutts-nano`.

- [ ] **Step 3: Confirm phonemization details**

From `neutts/neutts.py` + `neutts/phonemizers.py` (github neuphonic/neutts-air): record whether Air/Nano use phoneme mode (espeak-ng) or BPE mode, the exact phonemizer call (language, backbone args, whether punctuation/stress is kept), and how `ref_text` + `input_text` are joined. Record one worked example: raw "Hello world." → the exact phoneme string the reference produces (run `phonemizer` locally with espeak-ng).

- [ ] **Step 4: Confirm the reference-encode path**

From `neutts/neutts.py::encode_reference`: record how ref audio → codes (NeuCodec encode params, sample rate expected, mono). Confirm codec.cpp's `codec_encode` on a NeuCodec context produces the matching integer codes for the same input (spot-check one clip).

- [ ] **Step 5: Commit the notes**

```bash
git add docs/superpowers/notes/neutts-facts.md
git commit -m "docs(neutts): pin numeric contract (codec variant, token ids, phonemes, ref-encode) from official source"
```

**GATE:** if Step 1 or 4 shows a NeuCodec mismatch that codec.cpp can't reproduce, STOP and escalate — the integration is blocked on codec parity, not on this plan.

---

### Task 1: `--ref-text` CLI flag + espeak CMake option (tts-cli only)

**Files:**
- Modify: `examples/tts-cli.cpp` (arg parse near the `--ref-audio` handler ~line 153; usage text ~line 48)
- Modify: `CMakeLists.txt` (add `option(CODEC_TTS_ESPEAK ...)` + conditional link/define on the `tts-cli` target near :173)

**Interfaces:**
- Produces: tts-cli passes `--ref-text` value to `runner_params.ref_text` (new field, Task 3 consumes) and, when `CODEC_TTS_ESPEAK` and the model is NeuTTS, phonemizes `--text`/`--ref-text` before handing to the runner.

- [ ] **Step 1: Add the `--ref-text` arg + runner field**

In `examples/tts-cli.cpp`, add `std::string ref_text;` next to `ref_audio` (~line 116), an arg handler mirroring `--ref-audio`:
```cpp
else if (a == "--ref-text") { const char * v = need(); if (!v) return false; out->ref_text = v; i++; }
```
and wire `rp.ref_text = a.ref_text;` next to `rp.ref_audio_path` (~line 261). Add `[--ref-text STR]` to the usage string. (Add `std::string ref_text;` to the runner params struct in `common/tts_runner.h` — Task 3 uses it to fill `audio_lm_input.ref_text`.)

- [ ] **Step 2: Add the CMake option (default OFF)**

In `CMakeLists.txt`, inside the `tts-cli` block:
```cmake
option(CODEC_TTS_ESPEAK "Link espeak-ng into tts-cli for NeuTTS phonemization" OFF)
if(CODEC_TTS_ESPEAK)
    find_library(ESPEAK_NG_LIB NAMES espeak-ng REQUIRED)
    target_compile_definitions(tts-cli PRIVATE CODEC_TTS_ESPEAK=1)
    target_link_libraries(tts-cli PRIVATE ${ESPEAK_NG_LIB})
endif()
```

- [ ] **Step 3: Build both ways**

Run: `cmake -S . -B build -DCODEC_TTS_BACKBONE=ON && cmake --build build --target tts-cli -j` (espeak OFF — default), then `cmake -S . -B build -DCODEC_TTS_ESPEAK=ON && cmake --build build --target tts-cli -j` only if libespeak-ng is installed.
Expected: builds clean with espeak OFF (the default path this plan targets for CI/e2e).

- [ ] **Step 4: Commit**

```bash
git add examples/tts-cli.cpp common/tts_runner.h CMakeLists.txt
git commit -m "feat(tts-cli): --ref-text flag + optional CODEC_TTS_ESPEAK phonemization hook"
```

(NOTE: the actual espeak phonemization call in tts-cli is added in Task 5, once the runner path exists. This task only lands the flag + build wiring so later tasks can consume `ref_text`.)

---

### Task 2: NeuTTS detection + audio-token range derivation (`tts_runner`)

**Files:**
- Modify: `common/tts_runner.cpp` (add helpers near the existing `detect_outetts_version`, ~line 1050-1099)

**Interfaces:**
- Consumes: `vocab_has_piece(vocab, piece)` and the vocab-scan pattern from the grammar branch's `detect_outetts_version`.
- Produces: `bool detect_neutts(const llama_vocab * vocab)` and `bool neutts_audio_token_range(const llama_vocab * vocab, int32_t * offset, int32_t * count, int32_t * eos_id)` — the latter finds `<|speech_0|>` (offset), the contiguous `<|speech_N|>` count, and `<|SPEECH_GENERATION_END|>` (eos). Values from Task 0 Step 2 are the test oracle.

- [ ] **Step 1: Write the failing test** (extend `tests/tts_constraint_test.cpp` — it already links the backbone + can load a vocab via a small helper, OR add a dedicated `tests/neutts_detect_test.cpp` if the vocab is only available with a model). Since detection needs a real vocab, gate this behind the presence of a local NeuTTS backbone gguf; when absent, the test prints SKIP and returns 0. Assert: with the NeuTTS-Air vocab, `detect_neutts` is true and `neutts_audio_token_range` returns the exact offset/count/eos recorded in Task 0 Step 2; with a non-NeuTTS vocab (e.g. the MOSS backbone), `detect_neutts` is false.

```cpp
// tests/neutts_detect_test.cpp (backbone-only). Uses NEUTTS_GGUF env var; SKIP if unset/missing.
```
(Full harness code: load the backbone via `llama_model_load_from_file`, get vocab, call the two functions, compare to the Task-0 constants. Mirror the loading in `common/tts_runner.cpp:1099`.)

- [ ] **Step 2: Run — expect FAIL (undefined `detect_neutts`)**

Run: `cmake --build build --target neutts-detect-test -j` → link error.

- [ ] **Step 3: Implement**

In `common/tts_runner.cpp`, using the existing `vocab_has_piece` helper:
```cpp
static bool detect_neutts(const llama_vocab * vocab) {
    return vocab_has_piece(vocab, "<|SPEECH_GENERATION_END|>")
        && vocab_has_piece(vocab, "<|TEXT_PROMPT_START|>")
        && vocab_has_piece(vocab, "<|speech_0|>");
}
// Find id of a piece by linear scan of llama_vocab_get_text (as vocab_has_piece does), returning -1 if absent.
static int32_t vocab_piece_id(const llama_vocab * vocab, const char * piece);
static bool neutts_audio_token_range(const llama_vocab * vocab, int32_t * offset, int32_t * count, int32_t * eos_id) {
    const int32_t off = vocab_piece_id(vocab, "<|speech_0|>");
    const int32_t eos = vocab_piece_id(vocab, "<|SPEECH_GENERATION_END|>");
    if (off < 0 || eos < 0) return false;
    // count contiguous <|speech_N|> from N=0 upward (buffer the piece string)
    int32_t n = 0; char buf[32];
    for (;;) { std::snprintf(buf, sizeof(buf), "<|speech_%d|>", n);
               if (vocab_piece_id(vocab, buf) != off + n) break; ++n; }
    *offset = off; *count = n; *eos_id = eos; return true;
}
```
(If Task 0 shows the block is NOT contiguous in id-space, replace the contiguity assumption with the codebook-size `count` from Task 0 and drop the `== off+n` check — record which in the notes.)

- [ ] **Step 4: Run — expect PASS** (with `NEUTTS_GGUF` set to a local NeuTTS-Air backbone gguf; else SKIP).

- [ ] **Step 5: Commit**

```bash
git add common/tts_runner.cpp tests/neutts_detect_test.cpp CMakeLists.txt
git commit -m "feat(tts): detect NeuTTS + derive audio-token range from backbone vocab"
```

---

### Task 3: ICL reference encoding + NeuTTS prompt assembly

**Files:**
- Modify: `common/tts_runner.cpp` (new `build_neutts_prompt(...)` producing the token vector; uses the codec to encode ref audio)
- Possibly modify: `common/codec_common.h` / `common/audio_lm.cpp` if a small `audio_lm_encode_ref_codes(ctx, ref_pcm, n, sr, out_codes)` helper is cleaner than reaching the codec directly (decide from Task 0 Step 4).

**Interfaces:**
- Consumes: `runner_params.ref_text` (Task 1), `ref_pcm` load (existing `load_ref_audio`), NeuCodec encode (`codec_encode` / an `audio_lm_encode_ref_codes` helper), `detect_neutts`/`neutts_audio_token_range` (Task 2).
- Produces: `std::vector<llama_token> build_neutts_prompt(const llama_vocab*, const std::string & ref_text_phonemes, const std::string & input_phonemes, const std::vector<int32_t> & ref_codes)` — the prefill token sequence: template text tokens + `<|speech_{c}|>` tokens for each ref code, per the verbatim format.

- [ ] **Step 1: Write the failing test** — a token-sequence oracle. With the NeuTTS-Air vocab and a tiny synthetic `ref_codes={0,1,2}`, assert `build_neutts_prompt` yields exactly: tokenized `user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>` followed by the ids of `<|speech_0|><|speech_1|><|speech_2|>` (Task 0 §Step 5: note the `<|SPEECH_GENERATION_START|>` marker before the codes, and the single space joining ref/input phonemes). Compare against the ids Task 0 recorded. (SKIP if no NeuTTS gguf.)

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement `build_neutts_prompt`** — assemble the exact template string (verbatim), `llama_tokenize` it (parse_special=true so the `<|...|>` resolve), then append `vocab_piece_id("<|speech_{c}|>")` for each ref code (equivalently `offset + c`). Guard: every ref code `< count`.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Wire ref encoding** — in the runner, when `detect_neutts`: load `ref_pcm` (existing), NeuCodec-encode → `ref_codes` (per Task 0 Step 4), phonemize already done upstream (Task 5 / pre-phonemized), call `build_neutts_prompt`. Require `ref_pcm` + `ref_text` non-empty; error clearly otherwise.

- [ ] **Step 6: Commit**

```bash
git add common/tts_runner.cpp common/codec_common.h common/audio_lm.cpp tests/neutts_detect_test.cpp
git commit -m "feat(tts): NeuTTS ICL ref-code encoding + verbatim prompt assembly"
```

---

### Task 4: `run_token_single_cb` decode loop + dispatch

**Files:**
- Modify: `common/tts_runner.cpp` (new `run_token_single_cb(...)`; dispatch branch near :1449; range set via `audio_lm_set_audio_token_range`)

**Interfaces:**
- Consumes: `BackboneSampler` (grammar branch), `build_flat_audio_mask` (optional), `audio_lm_observe_token`, `audio_lm_decode_audio`, `build_neutts_prompt` (Task 3), `neutts_audio_token_range` (Task 2).
- Produces: `bool run_token_single_cb(audio_lm_context*, llama_context*, const llama_model*, const llama_vocab*, const audio_lm_prompt_info&, const std::vector<llama_token>& prompt_toks, int32_t max_frames, uint32_t seed, float temp, float top_p, int32_t top_k, int32_t* n_frames, const char** stop_reason)`.

- [ ] **Step 1: Implement the loop** — prefill `prompt_toks`; then per step: `tok = bbsmpl.sample(lctx, -1)`; `act = audio_lm_observe_token(ctx, tok, nullptr, 0)`; on `OBSERVE_STOP` set `stop_reason="eos"` and break; on `OBSERVE_CONSUMED` feed `tok` back (standard token path, `llama_decode` a 1-token batch); cap at `max_frames`. After the loop, the accumulated codes are inside `ctx`; the caller runs `audio_lm_decode_audio`. Optionally build the flat mask from the derived range and pass to `bbsmpl.build`.

- [ ] **Step 2: Add the dispatch branch** — before the `run_codebook_ar` else at `common/tts_runner.cpp:1453`, add:
```cpp
} else if (pi.model_kind == audio_lm_prompt_info::KIND_TOKEN_SINGLE_CB) {
    ar_ok = run_token_single_cb(ctx, lctx, lmodel, vocab, pi, neutts_prompt_toks,
                                max_frames, seed, temp, top_p, top_k, &n_frames, &stop_reason);
```
Set `audio_lm_set_audio_token_range(ctx, off, count, eos)` (from Task 2) before dispatch so `observe_token` recognizes the codes, and ensure `resolve_constraint`/`pi.model_kind` reflect Type A when `detect_neutts`.

- [ ] **Step 3: Build** — `cmake --build build -j` green.

- [ ] **Step 4: Commit**

```bash
git add common/tts_runner.cpp
git commit -m "feat(tts): run_token_single_cb Type A decode loop + KIND_TOKEN_SINGLE_CB dispatch"
```

---

### Task 5: espeak phonemization in tts-cli + reference asset

**Files:**
- Modify: `examples/tts-cli.cpp` (phonemize `--text`/`--ref-text` when `CODEC_TTS_ESPEAK` + NeuTTS)
- Create: `assets/refs/<name>.wav` (CC0) + `assets/refs/<name>.txt` (transcript)

- [ ] **Step 1: Source a CC0 reference clip** — a short (3–8 s) public-domain English speech clip (e.g. LibriVox excerpt) resampled to the NeuCodec rate; write its exact transcript to the `.txt`. Record provenance/license in a header comment or `assets/refs/README.md`.

- [ ] **Step 2: Add the espeak call (guarded)** — behind `#ifdef CODEC_TTS_ESPEAK`, phonemize `a.text`/`a.ref_text` via espeak-ng (match Task 0 Step 3 params) before building `rp`. Without the define, pass through unchanged.

- [ ] **Step 3: Build (espeak OFF) + commit**

```bash
git add examples/tts-cli.cpp assets/refs/
git commit -m "feat(tts-cli): espeak-ng phonemization (opt-in) + CC0 reference asset"
```

---

### Task 6: Convert backbones, e2e round-trip, publish

**Files:**
- Modify: `tests/e2e/ttscli_synthesize_smoke.py` (add `neutts_air`, `neutts_nano` cases; pre-phonemize `--text`/`--ref-text` in python)

- [ ] **Step 1: Convert backbones** — `neuphonic/neutts-air` (qwen2) + `neuphonic/neutts-nano` (llama) → gguf via the existing convert path; verify the `<|speech_N|>` block survives in the gguf vocab (Task 0 Step 2 ids). Pair with the NeuCodec gguf.

- [ ] **Step 2: Add e2e cases** — drive `tts-cli synthesize --model <neucodec> --backbone <neutts-air> --ref-audio assets/refs/<x>.wav --ref-text "<phonemes>" --text "<phonemes of 'Hello world.'>"`; the smoke pre-phonemizes via `phonemizer`/espeak-ng in python so it works with espeak OFF. Assert Whisper `_try_cer` below threshold (start report-only, then enforce once stable) and stop via eos.

- [ ] **Step 3: Run** — `.venv/bin/python tests/e2e/ttscli_synthesize_smoke.py` → NeuTTS-Air + Nano round-trip.

- [ ] **Step 4: Publish** — once round-trip passes, upload the verified backbone ggufs to the BricksDisplay NeuTTS repos (like the Qwen3 fix), overwriting.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/ttscli_synthesize_smoke.py
git commit -m "test(e2e): NeuTTS-Air/Nano synthesize round-trip (Whisper)"
```

---

## Self-Review notes (author)

- **Spec coverage:** detection+range (T2), ICL prompt (T3), decode loop (T4), CLI+espeak option (T1/T5), reference asset (T5), no-codec-metadata (T2 derives from vocab), phonemization scoped to tts-cli+CMake (T1/T5), e2e+publish (T6). Task 0 gates the numeric unknowns (NeuCodec variant, phonemes, token ids, ref-encode).
- **Deferred/risk:** NeuCodec variant match is the hard gate (T0 Step 1/4) — escalate if codec.cpp can't reproduce the code space. Phoneme exactness (T0 Step 3) — validated against the reference phonemizer.
- **Type consistency:** `detect_neutts`, `neutts_audio_token_range`, `vocab_piece_id`, `build_neutts_prompt`, `run_token_single_cb`, `KIND_TOKEN_SINGLE_CB`, `ref_text`, `CODEC_TTS_ESPEAK` used consistently.
- **Verification points for the implementer:** the exact `codec_encode` invocation for a NeuCodec context (T0 Step 4 / T3 Step 5); whether an `audio_lm_encode_ref_codes` helper is warranted vs direct codec access; the `run_token_single_cb` feed-back `llama_decode` batch shape (mirror `run_codebook_ar`'s backbone-token path).
