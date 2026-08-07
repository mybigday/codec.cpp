# OuteTTS 1.0 Integration — Numeric Contract

**GATE: PASS**

All V3 format details, token ids, speaker profile, grammar delta, and DAC parity pinned from
official sources. No blockers found.

Evidence:
- Token ids extracted directly from tokenizer.json via hf_hub_download (OuteAI/OuteTTS-1.0-0.6B,
  OuteAI/Llama-OuteTTS-1.0-1B).
- Prompt serialization code read from `outetts` 0.4.4 package
  (`version/v3/prompt_processor.py`, `version/interface.py`).
- Speaker JSON read from bundled file in outetts 0.4.4 package; license = Apache 2.0.
- DAC GGUF metadata read via `gguf.GGUFReader` from
  `models/output/ibm-research--DAC.speech-F32.gguf`. MATCH confirmed.
- Grammar (audio_lm.cpp:1219-1275) cross-checked against real outetts token sequence; one
  non-blocking structural delta found (see §4).

---

## §1. V3 Prompt + Speaker-Profile Serialization

**SOURCE:** `outetts` 0.4.4 — `version/v3/prompt_processor.py`, `version/v3/tokens.py`

### Input prompt (prefill)

Template (from `PromptProcessor._init_prompt` + `get_completion_prompt`):

```
<|im_start|>
<|text_start|>TARGET_TEXT<|text_end|>
<|audio_start|>
```

With a speaker prefix, `TARGET_TEXT` becomes the merged text (speaker reference text + separator +
target text), and the speaker word-blocks are appended after `<|audio_start|>\n`. Generation is
primed with a trailing `<|word_start|>` so the model continues with the first target word.

```
<|im_start|>
<|text_start|>SPEAKER_TEXT. TARGET_TEXT<|text_end|>
<|audio_start|>
<speaker word blocks, one per line>
<|word_start|>
```

### Speaker profile word-block format

Each word in the speaker profile is serialized as one line:

```
<|word_start|>WORD<|features|><|t_D.DD|><|energy_N|><|spectral_centroid_N|><|pitch_N|><|code|><|c1_X|><|c2_X|><|c1_X|><|c2_X|>...<|word_end|>
```

- `WORD` — raw text of the word (may contain punctuation run-on from speaker text)
- `<|t_D.DD|>` — word duration in seconds, 2 decimal places (e.g. `<|t_0.20|>`)
- `<|energy_N|>`, `<|spectral_centroid_N|>`, `<|pitch_N|>` — per-word acoustic features, N in 0..100
  (scaled from [0,1] float)
- `<|code|>` — single frame-group sentinel for this word (appears ONCE, at start of all frames)
- `<|c1_X|><|c2_X|>` — one DAC frame (X = codebook index 0..1023); repeated once per frame
- Lines joined with `\n`

**Code producing this** (`create_codes`):
```python
word = i['word'] + "<|features|>" + "<|t_{:.2f}|>".format(duration)
word += "".join(f"<|{k}_{v}|>" for k,v in features.items())  # energy, sc, pitch
word += "<|code|>"
for idx in range(len(i['c1'])):
    word += f"<|c1_{i['c1'][idx]}|><|c2_{i['c2'][idx]}|>"
block = "<|word_start|>" + word + "<|word_end|>"
```

### Model output format (generated tokens)

The model generates one word block per target word, then stops:

```
<|word_start|>Hello<|features|><|t_0.34|><|energy_12|><|spectral_centroid_18|><|pitch_42|><|code|><|c1_N|><|c2_N|>...<|word_end|>
<|word_start|>world<|features|><|t_0.28|><|energy_10|><|spectral_centroid_15|><|pitch_38|><|code|><|c1_N|><|c2_N|>...<|word_end|>
<|audio_end|>
<|im_end|>
```

**Token extraction** (`extract_audio_from_tokens`):

```python
codebook1 = [c1_map[t] for t in tokens if t in c1_map]   # backbone_id -> DAC index
codebook2 = [c2_map[t] for t in tokens if t in c2_map]
# strips non-audio tokens (word_start, features, energy, sc, pitch, etc.) automatically
```

The `<|code|>` sentinel is NOT in c1_map or c2_map; it is also skipped.

### Worked example: word "The" (from en-female-1-neutral speaker, word[0])

```
<|word_start|>The<|features|><|t_0.20|><|energy_10|><|spectral_centroid_15|><|pitch_45|>
<|code|>
<|c1_720|><|c2_658|><|c1_720|><|c2_663|><|c1_474|><|c2_237|><|c1_691|><|c2_915|>
<|c1_607|><|c2_74|><|c1_126|><|c2_74|><|c1_597|><|c2_966|><|c1_607|><|c2_721|>
<|c1_897|><|c2_893|><|c1_288|><|c2_722|><|c1_362|><|c2_630|><|c1_903|><|c2_516|>
<|c1_333|><|c2_861|><|c1_1009|><|c2_385|><|c1_79|><|c2_149|>
<|word_end|>
```

Token count: 1 (word_start) + 1 (The) + 1 (features) + 1 (t_0.20) + 3 (energy/sc/pitch)
             + 1 (code) + 15×2 (c1c2 pairs) + 1 (word_end) = **39 tokens** for 15 frames (0.20s @ 75fps).

---

## §2. Exact Token IDs

**SOURCE:** `OuteAI/OuteTTS-1.0-0.6B` tokenizer.json and `OuteAI/Llama-OuteTTS-1.0-1B`
tokenizer.json via `hf_hub_download`.

### OuteTTS-1.0-0.6B (Qwen3ForCausalLM, vocab_size=157760)

| Token | ID |
|---|---|
| `<|im_start|>` (BOS) | **151644** |
| `<|im_end|>` (EOS) | **151645** |
| `<|text_start|>` | **156722** |
| `<|text_end|>` | **156723** |
| `<|audio_start|>` | **156728** |
| `<|audio_end|>` | **156729** |
| `<|code|>` (frame sentinel) | **156730** |
| `<|word_start|>` | **156731** |
| `<|word_end|>` | **156732** |
| `<|features|>` | **156733** |
| `<|global_features_start|>` | **156734** |
| `<|global_features_end|>` | **156735** |
| `<|t_0.00|>` (first time token) | **156736** |
| `<|endoftext|>` | 151643 |
| `<|end_of_text|>` | 157737 |

#### 0.6B Audio-token ranges (all verified contiguous)

| Range | First ID | Last ID (incl.) | Count | Notes |
|---|---|---|---|---|
| c1 (codebook 0) | **151669** | **152693** | 1025 | indices 0..1024; index 1024 is padding |
| c2 (codebook 1) | **152694** | **153718** | 1025 | indices 0..1024; index 1024 is padding |
| energy | 153719 | 154719 | 1001 | |
| spectral_centroid | 154720 | 155720 | 1001 | |
| pitch | 155721 | 156721 | 1001 | |
| t_ (time) | 156736 | 157736 | 1001 | t_0.00..t_10.00 |

**c1 and c2 blocks are directly adjacent (zero gap).** c1_1024=152693, c2_0=152694.

**Valid DAC code range per codebook: 0..1023.** Index 1024 (the 1025th token) is an extra
padding/sentinel beyond the actual codebook size; never generated by the audio codec.

**c1 offset for codec.cpp:** `c1_base = 151669`; DAC index = `token_id - 151669`  
**c2 offset for codec.cpp:** `c2_base = 152694`; DAC index = `token_id - 152694`

**Audio-token contiguous test:**  
`detect_outetts_version` fires on `<|c1_0|>` (id 151669) — present in vocab. Both c1 and c2
blocks are Type A in the sense that they live in the backbone vocab. The `<|code|>` sentinel
(156730) lies outside the c1/c2 blocks (there are 3004 ids between c2_end and `<|code|>`
occupied by energy/sc/pitch/t_ tokens).

### Llama-OuteTTS-1.0-1B (LlamaForCausalLM, vocab_size=134400)

| Token | ID |
|---|---|
| `<|im_start|>` (BOS) | **133309** |
| `<|im_end|>` (EOS) | **133310** |
| `<|text_start|>` | **133311** |
| `<|text_end|>` | **133312** |
| `<|audio_start|>` | **133317** |
| `<|audio_end|>` | **133318** |
| `<|code|>` | **133319** |
| `<|word_start|>` | **133320** |
| `<|word_end|>` | **133321** |
| `<|features|>` | **133322** |

#### 1B Audio-token ranges (all verified contiguous)

| Range | First ID | Last ID (incl.) | Count |
|---|---|---|---|
| c1 | **128256** | **129280** | 1025 |
| c2 | **129281** | **130305** | 1025 |

**c1 and c2 blocks are directly adjacent (zero gap).** c1_1024=129280, c2_0=129281.

**c1 offset:** `c1_base = 128256`  
**c2 offset:** `c2_base = 129281`

#### Architecture summary

| | 0.6B | 1B |
|---|---|---|
| arch | qwen3 | llama |
| hidden | 1024 | 2048 |
| layers | 28 | 16 |
| heads (Q/KV) | 16/8 | 32/8 |
| max_pos | 40960 | 131072 |
| rope_theta | 1e6 | 5e5 |
| rope_scaling | **None (plain 1D RoPE)** | llama3 (factor=32) |
| dtype | bfloat16 | bfloat16 |
| bos_token_id | 151644 | 133309 |
| eos_token_id | 151645 | 133310 |

**Note on 0.6B MRoPE:** `rope_scaling=None` confirmed from config.json — plain 1D RoPE, not
MRoPE. This resolves the prior risk item. The 0.6B is safe to load with standard Qwen3 llama.cpp
support (no special rope_scaling handling needed beyond standard theta=1e6).

---

## §3. Default Speaker

**SOURCE:** `outetts` 0.4.4 package — `version/v3/default_speakers/json/en-female-1-neutral.json`
and `version/interface.py` (`load_default_speaker`).

### Identification

- **Name:** `en-female-1-neutral`
- **Language:** English
- **Voice:** neutral female
- **Only bundled V3 speaker** in outetts 0.4.4 (confirmed by `print_default_speakers` code comment
  and single file in the json directory)

### How to obtain

```python
# API
interface.load_default_speaker("en-female-1-neutral")

# Direct file path
outetts/version/v3/default_speakers/json/en-female-1-neutral.json
```

### JSON schema

```json
{
  "text": "<the reference sentence spoken in this speaker profile>",
  "words": [
    {
      "word":     "The",
      "duration": 0.20,
      "c1":       [720, 720, 474, ...],
      "c2":       [658, 663, 237, ...],
      "features": {"energy": 10, "spectral_centroid": 15, "pitch": 45}
    },
    ...
  ],
  "global_features": {"energy": 13, "spectral_centroid": 20, "pitch": 28},
  "interface_version": 3
}
```

- `words`: array of word objects. Counts: 26 words for the bundled speaker.
- `c1`/`c2`: parallel arrays of DAC codebook indices (0..1023), length = `round(duration * 75)` frames.
- `features`: per-word quantized acoustic features (0..100 int).
- `global_features`: whole-utterance summary features (same schema).
- `interface_version`: 3 (marks as V3 speaker).

### Reference text

> "The cat watched from the windowsill, tail flicking with quiet curiosity as the first snowflakes of winter began to fall, dusting the world in fragile white."

### File size

25 KB (25295 bytes). Saved to `docs/superpowers/notes/outetts10-en-female-1-neutral.json`
for reference. **Not a large binary; safe to commit if needed.**

### License

**Apache 2.0** — `outetts` package METADATA: `License :: OSI Approved :: Apache Software License`.
The default speaker JSON is bundled inside the outetts pip package, which is Apache 2.0 licensed.
Freely redistributable.

### Timing constants (from AudioProcessor.create_speaker_from_dict)

- `tps = 75` frames/second (hardcoded literal in audio_processor.py)
- Matches `sr=24000 / hop=320 = 75.0` exactly

---

## §4. Grammar Match: `outetts_build_grammar(V3)` vs OuteTTS 1.0 Real Format

**SOURCE:** `common/audio_lm.cpp:1219-1275` + `outetts` 0.4.4 `version/v3/prompt_processor.py`

### Structural match (correct elements)

The V3 grammar in `audio_lm.cpp` correctly captures:
- `codeBlock ::= features TIME energy spectralCentroid pitch CODE CODES*` — exactly matches the
  per-word token order: `<|features|><|t_D.DD|><|energy_N|><|spectral_centroid_N|><|pitch_N|><|code|>(<|c1_N|><|c2_N|>)*`
- One `<|code|>` sentinel per word (not per frame) — correct.
- `CODES* = (CODE1 CODE2)*` = zero-or-more c1c2 pairs — correct.
- `<|audio_end|>` terminator — correct.
- `<|im_end|>` EOS — correct.
- `wordEnd NL?` after each block — matches `\n`-joined output.

### Delta: `leadWord` missing `<|word_start|>`

```
Grammar:  root       ::= leadWord wordBlock* audioEnd NL EOS?
          leadWord   ::= WORD audioBlock      ← NO <|word_start|>
          wordBlock  ::= wordStart WORD audioBlock

Real:     ALL words (including first) have <|word_start|>...<|word_end|>
```

The actual token sequence from both `create_codes` (training) and `get_completion_prompt`
(inference) wraps every word — including the first — with `<|word_start|>`...`<|word_end|>`.
There is no bare WORD opener without `<|word_start|>`.

**Impact:** If grammar is attached as a hard GBNF constraint, it will fail on token 1 of the
generated output (the model generates `<|word_start|>` but the grammar root starts with `WORD`).
The grammar is currently not wired to OuteTTS V3 in `tts_auto_grammar` (returns `""` for
non-PARALLEL_HEADS_DELAY models), so this is a latent bug, not a live one.

**Fix needed** (before attaching grammar): change `leadWord` to `wordStart WORD audioBlock`,
or equivalently make `wordStart?` optional and accept that both forms are valid.

### Minor delta: `<|audio_start|>` not in grammar

The real prompt includes `<|audio_start|>` in the prefill; it is not part of the generated
sequence constrained by the grammar. This is correct — the grammar constrains generation only.
No fix needed.

---

## §5. DAC Decode Order + Parity

### Token → DAC index extraction

**SOURCE:** `outetts` 0.4.4 `version/v3/prompt_processor.py:extract_audio_from_tokens`,
`version/interface.py:get_audio`, `dac/interface.py:decode`

```python
# extract_audio_from_tokens returns two Python lists:
codebook1 = [c1_map[t] for t in tokens if t in c1_map]   # DAC codebook 0 indices
codebook2 = [c2_map[t] for t in tokens if t in c2_map]   # DAC codebook 1 indices
# → [codebook1, codebook2]  shape semantically: [n_q=2][n_frames]

# get_audio then calls:
torch.tensor([output], dtype=torch.int64)  # shape: [1, 2, n_frames]
audio_codec.decode(codes)                  # dac.DAC receives [batch=1, n_q=2, n_frames]
```

**Ordering:** c1 = codebook 0 (coarsest RVQ level), c2 = codebook 1. This mirrors the encode
path: `full_codes = dac.encode(audio)` returns `[batch, n_q, n_frames]`, with `full_codes[0][0]`
= codebook 0 = c1.

**codec.cpp layout:** The `codec_token_buffer` (T×Q interleaved: `data[t*n_q + q]`) should store:
`data[t*2 + 0] = c1[t]` (DAC codebook 0), `data[t*2 + 1] = c2[t]` (DAC codebook 1).

### DAC GGUF parity

**SOURCE:** `gguf.GGUFReader` on `models/output/ibm-research--DAC.speech-F32.gguf`

```
general.name      = DAC.speech.v1.0
codec.sample_rate = 24000
codec.hop_size    = 320
codec.n_q         = 2
codec.codebook_size = 1024
codec.latent_dim  = 1024
codec.codebook_dim = 8
codec.has_encoder = True
codec.has_decoder = True
```

**Source repo:** `ibm-research/DAC.speech.v1.0` (also mirrored/used by `OuteAI/DAC-speech-v1.0`).

| Parameter | GGUF on disk | OuteTTS requirement |
|---|---|---|
| sample_rate | 24000 | 24000 ✓ |
| hop_size | 320 | 320 ✓ |
| n_q | 2 | 2 ✓ |
| codebook_size | 1024 | 1024 ✓ |
| has_encoder | True | True (needed for speaker creation) |
| has_decoder | True | True (needed for synthesis) |

**PARITY: MATCH.** `models/output/ibm-research--DAC.speech-F32.gguf` is the correct codec for
OuteTTS 1.0. No new GGUF conversion needed. Quantized variants (`Q8_0`, `Q5_K_M`, `Q4_K_M`) also
present in `models/output/`.

Note: ibm-research's `config.json` lists hop_size=512 (authoring error); the actual hop is
4×4×4×5 = 320 (verified from GGUF metadata and OuteAI ONNX config).

**Which codec.cpp model already uses this DAC:** None currently uses the
`ibm-research--DAC.speech-F32.gguf` file. The older 16kHz/12-codebook DAC is at
`models/dac/dac.gguf`; OuteTTS 1.0 will be the first model in this repo to use the 24kHz 2-cb
version.

---

## §6. First-Pass Scoping File Discrepancies

Comparing against `.git/sdd/outetts10-facts.md` (first-pass):

| Item | First-pass | This investigation | Delta |
|---|---|---|---|
| `<|audio_start|>` 0.6B id | 156728 | **156728** | confirmed |
| `<|audio_end|>` 0.6B id | 156729 | **156729** | confirmed |
| `<|code|>` 0.6B id | 156730 | **156730** | confirmed |
| `<|word_start|>` 0.6B id | "156724-ish" | **156731** | first-pass was wrong |
| `<|word_end|>` 0.6B id | "..." | **156732** | first-pass was missing |
| `<|im_start|>` 0.6B | 151644 | **151644** | confirmed |
| c1_0 0.6B | 151669 | **151669** | confirmed |
| c1 count | "1025" | **1025** (indices 0..1024) | confirmed; 1024 is padding |
| c2_0 0.6B | 152694 | **152694** | confirmed |
| c2 count | "1025" | **1025** | confirmed |
| gap between c1 and c2 | "~1 unused id (Llama-1B)" | **0 for both models** | first-pass wrong |
| Llama-1B `<|im_start|>` | 133311 | **133309** | first-pass was wrong |
| Llama-1B `<|text_start|>` | 133311 | **133311** | confirmed |
| Grammar leadWord | not analyzed | **missing word_start** | new finding |
| Default speaker name | "EN-FEMALE-1-NEUTRAL" | **en-female-1-neutral** | case confirmed |
| Default speaker license | not stated | **Apache 2.0** | new finding |
| DAC parity | MATCH | **MATCH** | confirmed |
| 0.6B rope_scaling | "no rope_scaling" | **None confirmed** | confirmed |

Key corrections from first-pass:
- `<|word_start|>` for 0.6B is **156731** (not "156724-ish")
- Gap between c1 and c2 is **0** for both models (directly adjacent), not "~1 unused id"
- Llama-1B `<|im_start|>` is **133309** (first-pass table had 133311 which is `<|text_start|>`)
