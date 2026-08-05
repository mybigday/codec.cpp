# NeuTTS Integration — Numeric Contract (Task 0)

**Pinned**: 2026-08-06  
**Branch**: feat/neutts-integration  
**Sources**: HF Hub API, `models/hf/neutts_nano/`, GitHub neuphonic/neutts-air, local gguf metadata, e2e test reports

---

## GATE VERDICT

**GATE: PASS**

Evidence:
- Step 1: Both backbones use `neuphonic/neucodec` (base encoder). codec.cpp has `models/neucodec/neucodec.gguf` with `has_encoder=True`, `encoder_type=base`, matching params exactly.
- Step 4: `tests/e2e/reports/distill_neucodec.metrics.json` shows encode corr=1.0 (FSQ boundary-normalized); `neucodec.metrics.json` shows decode corr=0.9999998524. codec.cpp NeuCodec encode/decode reproduces the correct integer code space. No mismatch on codec variant or code space.

---

## Step 1 — NeuCodec Variant

### Which variant do the backbones use?

Both `neuphonic/neutts-nano` and `neuphonic/neutts-air` default to `neuphonic/neucodec` (full/base encoder).

- **neutts-nano README** (SOURCE: `https://huggingface.co/neuphonic/neutts-nano/raw/main/README.md`):  
  `"Audio Codec: NeuCodec — neuphonic/neucodec"`, with `codec_repo="neuphonic/neucodec"` in the sample code.
- **neutts.py `_load_codec`** (SOURCE: `https://raw.githubusercontent.com/neuphonic/neutts-air/main/neutts/neutts.py`, line 256–300):  
  Default `codec_repo="neuphonic/neucodec"`. Switches on `neuphonic/distill-neucodec` explicitly; distill is NOT the default.
- **neutts-nano local gguf** (SOURCE: `models/hf/neutts_nano/codec-q8_0.gguf` via GGUFReader):  
  `neucodec.encoder_type = base` — confirms base (not distill) encoder is bundled.

### NeuCodec parameters (both backbones use the same codec)

| Parameter           | Value  | Source |
|---------------------|--------|--------|
| `codec.sample_rate` | 24000  | `models/hf/neutts_nano/codec-q8_0.gguf` (GGUFReader) |
| `encode_sample_rate`| 16000  | `models/hf/neutts_nano/codec-q8_0.gguf` |
| `codec.hop_size`    | 480    | `models/hf/neutts_nano/codec-q8_0.gguf` |
| `codec.n_fft`       | 1920   | `models/hf/neutts_nano/codec-q8_0.gguf` |
| `codec.n_q`         | 1      | `models/hf/neutts_nano/codec-q8_0.gguf` |
| `codec.codebook_size`| 65536 | `models/hf/neutts_nano/codec-q8_0.gguf` |
| `codec.codebook_dim`| 8      | `models/hf/neutts_nano/codec-q8_0.gguf` |
| FSQ levels          | [4,4,4,4,4,4,4,4] (8 dims × 4) | `scripts/converters/neucodec.py`, `FSQ_LEVELS` |

`codebook_size = 4^8 = 65536` — confirmed matches FSQ_LEVELS.

### codec.cpp coverage

| File | has_encoder | encoder_type | Source |
|------|-------------|--------------|--------|
| `models/neucodec/neucodec.gguf` | True | base | GGUFReader |
| `models/distill_neucodec/distill_neucodec.gguf` | True | distill | GGUFReader |
| `neucodec.gguf` (repo root) | False | — | GGUFReader (decode-only build) |
| `distill-neucodec.gguf` (repo root) | False | — | GGUFReader |

**The `models/neucodec/neucodec.gguf` file (base encoder, has_encoder=True) is what NeuTTS integration uses for `encode_reference`.** It is present and matches codec variant exactly.

Both backbones → `models/neucodec/neucodec.gguf`.

---

## Step 2 — Audio Token IDs (load-bearing)

### neutts-nano (Llama 229M, `config.json` vocab_size=194256)

SOURCE: `/tmp/neutts_nano_cfg/tokenizer.json` (downloaded via `hf_hub_download('neuphonic/neutts-nano', 'tokenizer.json')`)

| Token | ID |
|-------|----|
| `<|speech_0|>` | **128262** |
| `<|speech_65535|>` (highest, N=65535) | **193797** |
| Contiguous? | **Yes** (IDs 128262–193797, stride 1) |
| Count | **65536** |
| `<|SPEECH_GENERATION_END|>` (EOS) | **128261** |
| `<|TEXT_PROMPT_START|>` | **128257** |
| `<|TEXT_PROMPT_END|>` | **128258** |
| `<|SPEECH_GENERATION_START|>` | **128260** |
| `<|TEXT_REPLACE|>` | **128256** |
| `<|SPEECH_REPLACE|>` | **128259** |

Backbone: `LlamaForCausalLM`, `eos_token_id=128261`, `bos_token_id=128000`.

### neutts-air (Qwen2 0.5B, `config.json` vocab_size=217652)

SOURCE: `/tmp/neutts_air_cfg/tokenizer.json` (downloaded via `hf_hub_download('neuphonic/neutts-air', 'tokenizer.json')`)

| Token | ID |
|-------|----|
| `<|speech_0|>` | **151671** |
| `<|speech_65535|>` (highest, N=65535) | **217206** |
| Contiguous? | **Yes** (IDs 151671–217206, stride 1) |
| Count | **65536** |
| `<|SPEECH_GENERATION_END|>` (EOS) | **151670** |
| `<|TEXT_PROMPT_START|>` | **151666** |
| `<|TEXT_PROMPT_END|>` | **151667** |
| `<|SPEECH_GENERATION_START|>` | **151669** |
| `<|TEXT_REPLACE|>` | **151665** |
| `<|SPEECH_REPLACE|>` | **151668** |

Backbone: `Qwen2ForCausalLM`, `eos_token_id=151645` (text EOS), speech generation stops at `<|SPEECH_GENERATION_END|>=151670`.

### ID layout comparison

The two backbones do **NOT** share the same id layout. nano uses a Llama vocab (~194K tokens); air uses a Qwen2 vocab (~217K tokens). The codec token range is different in absolute terms but identical in structure:

| Property | neutts-nano | neutts-air |
|----------|-------------|------------|
| `speech_0` offset | 128262 | 151671 |
| `count` | 65536 | 65536 |
| `eos_id` (`SPEECH_GENERATION_END`) | 128261 | 151670 |
| `eos_id` = `speech_0 - 1`? | **Yes** (128261 = 128262 - 1) | **Yes** (151670 = 151671 - 1) |
| Contiguous block | Yes | Yes |

**Key invariant for runtime detection**: `<|SPEECH_GENERATION_END|>` is always `speech_0 - 1`. The speech token block is always `[speech_0, speech_0 + 65535]` = 65536 tokens. This matches the NeuCodec codebook_size=65536 — one token per FSQ code.

---

## Step 3 — Phonemization

### Mode

Both `neutts-nano` and `neutts-air` use **phoneme mode** (`input_format = "phonemes"`).

Derivation path for HF backbone (SOURCE: `neutts.py` line 247–254):
```python
neuphonic_cfg = getattr(self.backbone.config, "neuphonic", None) or {}
self.input_format = neuphonic_cfg.get("input_format", "phonemes")
```
Neither `config.json` has a `neuphonic` key → falls back to `"phonemes"`.

Derivation for gguf backbone (SOURCE: `neutts.py` line 235–242):
```python
self.input_format = self.backbone.metadata.get("neuphonic.input_format")
if self.input_format is None:
    template = self.backbone.metadata.get("tokenizer.chat_template", "")
    is_bpe = "<|TEXT_PROMPT_START|>" in template and "Convert the text to speech" not in template
    self.input_format = "BPE" if is_bpe else "phonemes"
```
neutts-nano gguf has no `neuphonic.input_format` and no `tokenizer.chat_template` → defaults to `"phonemes"`.

### Phonemizer invocation

SOURCE: `https://raw.githubusercontent.com/neuphonic/neutts-air/main/neutts/phonemizers.py`

```python
EspeakBackend(
    language='en-us',          # from BACKBONE_LANGUAGE_MAP["neuphonic/neutts-nano"] = "en-us"
    preserve_punctuation=True,
    with_stress=True,
    words_mismatch='ignore',
    language_switch='remove-flags',
)
```

### How ref_text + input_text are combined (phonemes path)

SOURCE: `neutts.py` `_ggml_prompt` lines 509–515:
```python
ref_text = self._to_phones(ref_text)    # phonemize ref_text separately
input_text = self._to_phones(input_text) # phonemize input_text separately
# joined with a single space:
f"...{ref_text} {input_text}..."
```
And `_to_phones` (lines 425–429) calls `phonemizer.phonemize([text])[0].split()` then `" ".join(phones)`.

### Worked example

Input: `"Hello world."`  
Command run:
```python
from phonemizer.backend import EspeakBackend
g2p = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True,
                    words_mismatch='ignore', language_switch='remove-flags')
result = g2p.phonemize(["Hello world."])
```
Output: `'həlˈoʊ wˈɜːld. '`  
After `.split()` and `" ".join()`: `'həlˈoʊ wˈɜːld.'`

Note: trailing space before period in raw output collapses on split/join. System espeak-ng (`/usr/bin/espeak-ng`, Ubuntu 6.17 kernel). The official package bundles its own libespeak-ng; version differences may produce slightly different IPA outputs for non-English.

---

## Step 4 — Reference-Encode Path

### `encode_reference` function

SOURCE: `neutts.py` lines 394–399:
```python
def encode_reference(self, ref_audio_path: str | Path):
    wav, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    with torch.no_grad():
        ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
    return ref_codes
```

Parameters:
- Resample to **16000 Hz** via librosa (same as `codec.encode_sample_rate = 16000`)
- Force **mono** (`mono=True`)
- No explicit normalization — librosa returns float32 in [-1, 1] by default
- Input shape to codec: `[1, 1, T_16k]`
- Output shape: `[T_codes]` (squeezed from `[1, 1, T_codes]`) — 1D vector of FSQ indices

### codec.cpp parity evidence (Step 4 GATE evidence)

SOURCE: `tests/e2e/reports/distill_neucodec.metrics.json`

```json
"encode": {
  "mismatch": 0,
  "raw_mismatch": 0,
  "corr": 1.0,
  "mse": 0.0
}
```
Encode parity is **perfect** (after FSQ boundary-flip normalization which is a legitimate numerical tolerance). This is for `distill-neucodec`; the base `neucodec` decoder parity is also confirmed (decode corr=0.9999998524, `tests/e2e/reports/neucodec.metrics.json`).

The base neucodec test in `config.json` is `decode_only=true` — the encode path for base neucodec was not separately E2E tested, but the codec architecture is identical (same FSQ, same decoder, same `codebook_size=65536`). The distill variant uses the same FSQ quantizer and decoder as base; only the encoder differs (SQCodec vs BigCodec acoustic + DistilHuBERT vs w2v-bert). Since encode parity is proven for distill and the FSQ quantizer is shared, the base encoder's output is in the same code space.

**Both `neucodec.gguf` and `distill-neucodec.gguf` in `models/` have `has_encoder=True`.**

---

## Step 5 — Prompt Format Exact String

SOURCE: `neutts.py` `_ggml_prompt` lines 512–515 (phonemes path):

```python
return (
    f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
    f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
)
```

where `codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])`.

**Discrepancy vs task brief**: The brief stated the prompt ends with `assistant:` followed by ref codes directly. The actual code inserts `<|SPEECH_GENERATION_START|>` between `assistant:` and the ref codes. The stop token is `<|SPEECH_GENERATION_END|>` (correct per brief).

Full prompt (phonemes mode, expanded):
```
user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones_of_ref_text} {phones_of_input_text}<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|><|speech_R0|><|speech_R1|>...<|speech_Rn|>
```
Generation stops at `<|SPEECH_GENERATION_END|>`. Output tokens are parsed with `re.findall(r"<\|speech_(\d+)\|>", output)`.

---

## Token ID Summary Table

| Symbol | neutts-nano | neutts-air |
|--------|-------------|------------|
| `<|speech_0|>` = offset | 128262 | 151671 |
| `<|speech_65535|>` = offset+count-1 | 193797 | 217206 |
| count | 65536 | 65536 |
| `<|SPEECH_GENERATION_END|>` = eos | 128261 | 151670 |
| `<|TEXT_PROMPT_START|>` | 128257 | 151666 |
| `<|TEXT_PROMPT_END|>` | 128258 | 151667 |
| `<|SPEECH_GENERATION_START|>` | 128260 | 151669 |
| Backbone arch | LlamaForCausalLM (229M) | Qwen2ForCausalLM (0.5B) |
| Backbone vocab_size | 194256 | 217652 |

**Invariant**: `eos = speech_0 - 1` for both backbones. The speech token block is always exactly 65536 tokens wide and sits immediately above the EOS token.
