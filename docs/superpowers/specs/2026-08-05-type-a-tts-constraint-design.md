# Type A TTS backbone-sampling constraints (generalized auto-grammar + logit-bias)

Status: design (awaiting user review)
Date: 2026-08-05

## Problem

`tts_auto_grammar(pi, text)` (`common/audio_lm.cpp`) currently emits a GBNF only for
MOSS-TTSD. Every other model — including Type A token-based models (OuteTTS, Orpheus) —
gets an empty string, so the backbone sampler is unconstrained. The runner can already
attach a *user*-supplied `--grammar`, but there is no model-derived constraint for Type A.

We want the runner to **automatically constrain the backbone sampler for Type A models**,
choosing the correct mechanism for the model's generation structure.

## Key facts established during design

1. **llama.cpp natively supports logit-bias, and `common_sampler` wires it.**
   - `llama_sampler_init_logit_bias(n_vocab, n, biases)` — `llama.cpp/include/llama.h:1450`,
     impl `llama.cpp/src/llama-sampler.cpp:3569`.
   - `common/sampling.cpp:313`: `if (params.has_logit_bias()) samplers.push_back(llama_sampler_init_logit_bias(...))`.
   - Our `SamplerChain` (`common/tts_runner.cpp:151`) builds a `common_params_sampling sp`; we
     already set `sp.grammar`. Adding `sp.logit_bias` is a one-line addition.

2. **GBNF constrains the DETOKENIZED CHARACTER STREAM, not token IDs.** MOSS-TTSD's grammar
   works only because its codes detokenize to literal `"<0>".."<1023>"` pieces. A generic
   "constrain to token-ID range" cannot be expressed as GBNF without enumerating each token's
   piece string (collision/byte-fallback/non-prefix-free risks, unverifiable without a
   checkpoint). For a *flat* contiguous ID range, **logit-bias is the correct, provable,
   metadata-only tool.**

3. **Type A splits by generation structure — one mechanism does not fit both:**
   - **Flat / direct-audio Type A**: sampled stream is purely audio codes ∪ eos. A static
     logit-bias mask over `[offset, offset+count) ∪ {eos}` is correct. (This is the ONLY case
     logit-bias fits.)
   - **Structured Type A (OuteTTS)**: word text + timestamps + per-word features + audio codes
     are *interleaved*; a whole-generation ID mask would forbid the structural tokens and break
     generation. **Only a grammar can express the sequence** — and OuteTTS's grammar is
     **prompt-dependent** (the input word sequence is compiled into the grammar).

4. **Type A range metadata already exists** but is not surfaced to the grammar builder:
   - Private `audio_lm_context` fields: `audio_tok_offset`, `audio_tok_count`,
     `audio_tok_eos` (`common/audio_lm.cpp:90-92`).
   - GGUF KVs read at init: `codec.audio_token.offset`, `codec.audio_token.count`,
     `codec.audio_token.eos_id` (`common/audio_lm.cpp:180`, doc `common/codec_common.h:186`).
   - Accessors: `audio_lm_set_audio_token_range` / `audio_lm_get_audio_token_range`
     (`common/audio_lm.cpp:555` / `:563`).

5. **No Type A model is wired up and no local checkpoint exists.** So the *detection wiring*
   and *actual OuteTTS audio* are not e2e-testable now. But the **constraint-generation logic
   IS testable offline** (GBNF string vs the reference, GBNF parses in llama.cpp; logit-bias
   membership over an arbitrary vocab).

## Design decision (mechanism per structure)

`logit-bias` = flat/direct-audio models ONLY. `grammar` = structured models (OuteTTS) and the
existing MOSS-TTSD path. Model family is **detected from backbone metadata** (preferred) or
codec metadata, then the matching constraint is generated at runtime.

### Constraint abstraction (lives in `tts_runner`, NOT codec_common)

`llama_logit_bias` comes from `llama.h`, which `common/audio_lm.cpp` (codec_common) must not
include — codec_common is built WITHOUT the backbone. So the unified struct and all logit-bias
materialization live in `tts_runner.cpp` (which links the backbone). codec_common stays
string-only.

```cpp
// tts_runner.cpp (backbone-linked)
struct tts_constraint {
    std::string                    grammar;      // GBNF; empty = none
    std::vector<llama_logit_bias>  logit_bias;   // ID mask; empty = none
    // Exactly one is populated for a constrained model; both empty = unconstrained.
};
```

### Resolution order (in `tts_runner`, where the backbone `vocab`/`lmodel` are available)

A single `resolve_constraint(...)` in `tts_runner` produces the `tts_constraint`:

1. **User `--grammar`** → `{ grammar = user, {} }`. (Unchanged precedence; wins over auto.)
2. **Backbone-family detection** → if the backbone is identified as OuteTTS (see below), build
   the **prompt-dependent OuteTTS grammar** (V2 or V3) from `text`. → `{ grammar, {} }`.
3. **Codec-metadata fallback**:
   - `tts_auto_grammar(pi, text)` (in `audio_lm.cpp`, backbone-agnostic, returns a string) —
     MOSS-TTSD merged-cb0 delay grammar (unchanged). Non-empty → `{ grammar, {} }`.
   - else Flat Type A (`pi.audio_tok_offset >= 0`, no known structure) → `tts_runner`
     materializes a **logit-bias mask** over `[offset, offset+count) ∪ {eos}` from `pi`'s
     id-range fields (needs `n_vocab`, available here). → `{ "", logit_bias }`.
   - otherwise → both empty.

### Location split (respect the no-backbone build)

Insight: the OuteTTS grammar STRING builders are pure `(version, text) -> string` with NO
llama dependency (only version *detection* touches the vocab). So the pure builders live in
codec_common — reusable by hook-hosts (rn-tts's existing `build_dynamic_grammar`, see
`codec_common.h:487`) and testable via ctypes/a C++ binary WITHOUT a backbone.

- **`audio_lm.cpp` / `codec_common.h` (codec_common, no backbone)**:
  - `tts_auto_grammar(pi, text)` — MOSS-TTSD grammar, unchanged output. Its only change is it
    can now also read `pi.audio_tok_*` (added below), but it does NOT build logit-bias (no
    `llama.h`).
  - NEW pure builders: `enum class outetts_version { V2, V3 };` and
    `std::string outetts_build_grammar(outetts_version v, const std::string & text);`
    (+ internal `word_to_grammar`). Pure string; no llama include.
  - New `pi` id-range fields, populated at prompt-build.
- **`tts_runner.cpp` (backbone-linked)**: owns `struct tts_constraint`, `resolve_constraint`,
  backbone-family/version detection (reads `lmodel` metadata + `vocab` signature tokens),
  flat-mask materialization (`llama_logit_bias`), and the `SamplerChain` wiring
  (`sp.grammar` / `sp.logit_bias`).

The existing call site (`tts_runner.cpp:1252`, currently `tts_auto_grammar(pi, a.text)`) is
replaced by `resolve_constraint(...)`, which internally calls `outetts_build_grammar` (detected
OuteTTS), else `tts_auto_grammar` (MOSS), else the flat mask.

### `audio_lm_prompt_info` additions

```cpp
enum kind { ..., KIND_TOKEN_SINGLE_CB /* Type A */ };   // new
int32_t audio_tok_offset = -1;   // copied from audio_lm_context at prompt-build
int32_t audio_tok_count  = 0;
int32_t audio_tok_eos    = -1;
```

Populated in `audio_lm_get_prompt_info` from the ctx fields so the metadata-only builder can
form the flat mask without reaching into the private context.

## OuteTTS grammar builder (V2 + V3) — verbatim reference

Ported from the user's reference (BricksDisplay/OuteTTS-Speaker-Creator `app.py`). Preserve
exactly; do not "fix" the interspersed-audioBlock or unused-rule quirks.

`word_to_grammar(word)`:
```python
if all(ord(c) < 128 for c in word):
    return f'"{word}"'                                   # ASCII → quoted literal
return f'[{"".join(OrderedDict.fromkeys(word))}]+'       # non-ASCII → unique-char class +
```

V2 (`words` joined into root positionally; word order is HARD-CODED):
```
root       ::= NL? {' audioBlock '.join(words)} audioEnd NL EOS?
audioBlock ::= TIME CODE* space NL?
TEXT ::= [A-Za-z0-9 .,?!]+
EOS      ::= "<|im_end|>"
emotionStart ::= "<|emotion_start|>"
emotionEnd ::= "<|emotion_end|>"
audioEnd   ::= "<|audio_end|>"
space      ::= "<|space|>"
WORD       ::= {' | '.join(words)}
NL         ::= [\n]
TIME  ::= "<|t_" DECIMAL "|>"
CODE    ::= "<|" DIGITS "|>"
DIGITS     ::= [0-9]+
DECIMAL    ::= [0-9]+ "." [0-9]+
punch ::= "<|" [a-z_]+ "|>"
```

V3 (root uses `WORD` alternation + `wordBlock*`; word order NOT hard-locked; paired c1_/c2_):
```
root       ::= leadWord wordBlock* audioEnd NL EOS?
leadWord ::= WORD audioBlock
wordBlock ::= wordStart WORD audioBlock
audioBlock ::= codeBlock wordEnd NL?
codeBlock ::= features TIME energy spectralCentroid pitch CODE CODES*
TEXT ::= [A-Za-z0-9.,!?]+
EOS      ::= "<|im_end|>"
audioEnd   ::= "<|audio_end|>"
wordStart ::= "<|word_start|>"
wordEnd ::= "<|word_end|>"
features ::= "<|features|>"
energy ::= "<|energy_" DIGITS "|>"
spectralCentroid ::= "<|spectral_centroid_" DIGITS "|>"
pitch ::= "<|pitch_" DIGITS "|>"
WORD       ::= {' | '.join(words)}
NL         ::= [\n]
TIME       ::= "<|t_" DECIMAL "|>"
CODE       ::= "<|code|>"
CODES      ::= CODE1 CODE2
CODE1      ::= "<|c1_" DIGITS "|>"
CODE2      ::= "<|c2_" DIGITS "|>"
DIGITS     ::= [0-9]+
DECIMAL    ::= [0-9]+ "." [0-9]+
```

### V3 word-constraint rationale (per author — do NOT "tighten")

V3 targets multilingual text. Without running a tokenizer, word-segmentation points cannot be
reliably determined, so the grammar deliberately does NOT enforce exact word order/count — it
lists the words as a **semi-open `WORD` alternation** (any listed word, any order, `wordBlock*`).
The goal is only **anti-hallucination** (keep generation from drifting far off the requested
words), not strict alignment. Do NOT refactor this into a fixed sequence — that breaks
multilingual inputs. V2, the older single-script path, DOES hard-code word order in `root`; keep
the V2/V3 asymmetry intentionally.

**Robustness addendum (safe deviation from the reference):** when emitting an ASCII word as a
`"{word}"` literal, escape `"` and `\` so a word containing those characters cannot produce
invalid GBNF. Non-ASCII words already go through the `[chars]+` class. This only affects
otherwise-broken inputs; valid outputs are byte-identical to the reference.

### Version detection (from backbone metadata)

Distinguish by signature tokens present in the backbone vocab / metadata:
- V3 iff the vocab contains V3-only markers, e.g. `<|word_start|>` / `<|c1_0|>` / `<|features|>`.
- V2 iff it has `<|space|>` + bare `<|N|>` code tokens and lacks the V3 markers.
- Optionally corroborate with `general.name` / `general.architecture`.

The builder itself takes `(version, text)` and is unit-tested independently of detection.

## Init-time validation (Fable's #1 failure mode)

When `audio_tok_offset >= 0`, at load: assert `offset + count <= n_vocab`; log the detokenized
piece of the first and last token of the range. A converter that miscounts added-token order
produces in-range-but-wrong IDs (silent garbage audio, no error) — the log makes it visible.
(Requires vocab; run this check in `tts_runner` init where the backbone vocab exists.)

## Testing

The repo has NO C++ unit framework — it tests exclusively via Python e2e smokes driving built
binaries (`tests/e2e/*.py`). But this feature's core logic has **no runnable model** (no Type A
backbone gguf exists), so a true e2e smoke is impossible now. Documented deviation: add one
**standalone C++ test binary** for the model-less logic. This is NOT throwaway parity
scaffolding that graduates to e2e (the norm the repo's e2e-only rule targets) — it is the
*permanent* regression coverage for logic that can never be e2e-tested until an OuteTTS backbone
exists. It can be retired if/when a real OuteTTS gguf enables a synthesize smoke.

- **`tests/tts_constraint_test.cpp`** (new `add_executable`, built only under
  `CODEC_TTS_BACKBONE=ON`; a thin Python wrapper `tests/e2e/tts_constraint_smoke.py` runs it via
  subprocess so it slots into the existing `tests/e2e` invocation pattern). Covers:
  - `outetts_build_grammar(V2, text)` / `(V3, text)`: assert EXACT GBNF string vs expected for
    sample inputs (ASCII words; a non-ASCII/CJK word → `[chars]+`; a word containing `"` → escaped).
  - Structural lint on the produced GBNF (model-less; a full parse needs a vocab): every rule
    name referenced on a RHS has an LHS definition, and `"` literals are balanced. The
    exact-string-match against the known-valid reference is the primary validity guarantee; a
    true GBNF parse + token-match is deferred to when a real model runs.
  - Flat logit-bias: `build_flat_audio_mask(offset,count,eos,n_vocab)` returns exactly the
    complement ids at `-INFINITY`, and the allowed set (range ∪ eos) unbiased.
  - `tts_auto_grammar` MOSS-TTSD output unchanged (regression-guard the string).
- **Deferred (needs assets)**: backbone-family/version detection against a real OuteTTS gguf;
  OuteTTS end-to-end audio. Gated behind "when an OuteTTS backbone gguf is available".

## Non-goals / deferred

- Wiring up OuteTTS/Orpheus/VibeVoice as runnable models (converters, decode) — separate specs.
- Multi-codebook Type A dispatch (Orpheus delay) — `observe_token` still falls through to
  PASSTHROUGH for `n_q != 1`; unchanged here.
- Phase-scoped logit-bias for structured models — unnecessary: structured models use grammar.
- Moshi — out of scope (no Helium arch in pinned llama.cpp).

## Failure modes to keep in mind (from Fable advisory)

1. Stale/wrong `codec.audio_token.offset` → in-range-but-wrong ids → silent garbage. Mitigated
   by the init assert + piece logging above.
2. A static ID mask breaks any model that emits structural/text tokens mid-generation. Mitigated
   by never using logit-bias for structured models — those go through grammar.
