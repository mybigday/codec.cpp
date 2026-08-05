# NeuTTS-Air / Nano integration (flat Type A TTS)

Status: design (awaiting user review)
Date: 2026-08-06

## Goal

Make `tts-cli synthesize` produce intelligible, round-trip-correct speech for **NeuTTS-Air** and **NeuTTS-Nano** (Text → TTS → Whisper → Text matches), validated against the BricksDisplay HF ggufs. These are currently unsupported (no Type A single-codebook decode path in `tts_runner`); the llama.rn report showed them producing garbage.

## Model family

NeuTTS-Air/Nano are **flat Type A** token TTS (official ref: github `neuphonic/neutts-air`, `neutts/neutts.py`):
- Backbone LM emits a stream of single-codebook audio tokens `<|speech_{i}|>` (NeuCodec codes), interleaved with nothing else, terminated by `<|SPEECH_GENERATION_END|>`.
- **ICL voice-clone**: a reference voice is injected as in-context codes. Flow: NeuCodec-**encode** the ref audio → integer codes → `<|speech_{i}|>` tokens, placed in the prompt alongside the reference transcript.
- Prompt (phoneme mode): `user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}<|TEXT_PROMPT_END|>\nassistant:` followed by the reference speaker's `<|speech_{i}|>` codes.
- Stop token: `<|SPEECH_GENERATION_END|>`. No grammar in the official code.
- Codec: **NeuCodec** (single codebook). Air backbone = qwen2, Nano = llama. Pipeline otherwise identical.

Build **Air first** (template); Nano differs only in backbone arch (both plain llama.cpp-convertible) and reference sample.

## Key architectural decisions (settled with user)

1. **No NeuTTS metadata on the codec gguf.** The generic NeuCodec gguf is reused unchanged. The `<|speech_N|>` range + control tokens are backbone-vocab properties → they live on the backbone side and are **derived at runtime from the backbone vocab**. (User: "token 資料應該要在 backbone 本身".)
2. **No large params baked into any gguf metadata.** (User, prior-session decision.)
3. **Reference voice = existing `--ref-audio` + new `--ref-text`** (Chatterbox-style; `audio_lm_input.ref_text` already exists for ICL). No baked default speaker. Ship one **CC0/public-domain** reference clip + transcript in-repo as the free example/default.
4. **Phonemization = espeak-ng, confined to `tts-cli` behind a CMake option `CODEC_TTS_ESPEAK` (default OFF).** Core `libcodec`/`codec_common` never links espeak → main license untouched (GPLv3 exposure limited to an opt-in example that dynamically links a user-installed system lib). When OFF, tts-cli expects pre-phonemized `--text`/`--ref-text`. (User correction: license concern is example-scoped + toggleable.)

## Components

### A. Backbone conversion (no new converter)
NeuTTS backbones convert as plain qwen2 (Air) / llama (Nano) via the existing path — the `<|speech_N|>`, `<|TEXT_PROMPT_START|>`, `<|TEXT_PROMPT_END|>`, `<|SPEECH_GENERATION_END|>` are added-tokens already in the vocab; the conversion must preserve them (verify added-token survival into the gguf vocab). Codec = existing NeuCodec gguf. **Task 0 must confirm** (from official source): the exact NeuCodec variant (`neuphonic/emilia-yodas-english-neucodec`) matches codec.cpp's NeuCodec decode/encode, and the phoneme vs BPE prompt details.

### B. Runtime detection + range derivation (`tts_runner`)
Mirror the OuteTTS-detection pattern (grammar branch): from the backbone `vocab`, detect NeuTTS by signature tokens (`<|SPEECH_GENERATION_END|>` + `<|TEXT_PROMPT_START|>` + `<|speech_0|>`). Derive the audio-token range: `offset` = id of `<|speech_0|>`, `eos_id` = id of `<|SPEECH_GENERATION_END|>`, `count` = number of contiguous `<|speech_N|>` tokens (equivalently NeuCodec codebook size). Call the existing `audio_lm_set_audio_token_range(offset,count,eos_id)` at runtime. No codec-gguf metadata involved.

### C. ICL prompt assembly (`tts_runner`/`audio_lm`, new Type A ICL branch)
1. NeuCodec-**encode** `ref_pcm` (via `codec_encode` / `codec_neucodec_encode`) → integer codes.
2. Render `<|speech_{c}|>` for each ref code → tokenize (they're single vocab tokens).
3. Assemble the NeuTTS template (hardcoded per detected family, like the OuteTTS grammar precedent) with `ref_text` + `input_text` (both already phonemized by tts-cli when `CODEC_TTS_ESPEAK`), then append the ref-code tokens. Feed as the backbone prefill.

### D. Type A decode loop `run_token_single_cb` (`tts_runner`, new)
New dispatch branch for `pi.model_kind == KIND_TOKEN_SINGLE_CB`:
```
loop:
  tok = BackboneSampler.sample(lctx)          // optional flat logit-bias [speech range]∪eos (grammar branch)
  act = audio_lm_observe_token(ctx, tok, ...) // CONSUMED → accumulate code; STOP → break
  feed tok back as next input (standard token path)
audio_lm_decode_audio(ctx, &out)              // accumulated codes → NeuCodec → PCM
```
Reuses the existing `audio_lm_observe_token` (accumulates codes, detects eos) + `audio_lm_decode_audio` (codes→PCM) primitives — already used by the rn-tts hook host. The optional flat logit-bias is the grammar branch's `build_flat_audio_mask`.

### E. CLI (`tts-cli`)
- Add `--ref-text` (‑‑ref-audio already exists). NeuTTS requires both.
- Behind `CODEC_TTS_ESPEAK`: phonemize `--text` and `--ref-text` via espeak-ng before passing to the runner. When OFF: pass through (caller pre-phonemizes).

### F. Reference asset
Ship `assets/refs/<name>.wav` (CC0/public-domain speech) + `<name>.txt` (its exact transcript) in-repo as the default example. Source: a public-domain clip (e.g. LibriVox / a self-recorded CC0 sample), NOT the neuphonic `samples/dave.wav` (licensing unknown).

## Data flow (tts-cli, espeak ON)
```
--text, --ref-text (raw) --espeak--> phonemes
--ref-audio --load/resample--> ref_pcm --NeuCodec.encode--> ref codes
runner: detect NeuTTS(vocab) → set audio_tok range → assemble ICL prompt (phonemes + ref codes)
      → run_token_single_cb (sample <|speech_N|> until <|SPEECH_GENERATION_END|>)
      → audio_lm_decode_audio (NeuCodec) → PCM → WAV
```

## Testing
- **e2e** `tests/e2e/ttscli_synthesize_smoke.py`: add `neutts_air` + `neutts_nano` cases driving `tts-cli synthesize --model <neucodec> --backbone <neutts> --ref-audio assets/refs/<x>.wav --ref-text "<transcript>" --text "Hello world."` → Whisper round-trip (`_try_cer`) below threshold. The smoke **pre-phonemizes** `--text`/`--ref-text` in python (phonemizer/espeak-ng) so it passes regardless of the tts-cli `CODEC_TTS_ESPEAK` build.
- **Staged validation** (per repo convention): before the full round-trip, confirm the ICL prompt token sequence + a short generated code sequence against the official `neutts` reference on the same input (deterministic/greedy) to catch prompt/format bugs early; delete that scaffolding once e2e passes.
- Convert Air/Nano backbones from `neuphonic/neutts-air` / `neuphonic/neutts-nano`; verify round-trip; then upload the (correct) ggufs to the BricksDisplay NeuTTS repos.

## Non-goals / deferred
- Soprano (different arch: hidden-states→Vocos) and OuteTTS (backbone under OuteAI org) — separate specs.
- Streaming.
- Non-English phonemization nuances (README warns espeak version matters); target English first.

## Risks
1. **Phoneme fidelity**: matching neuphonic's espeak-ng phoneme set exactly. Mitigate: use espeak-ng (same tool) + validate the phoneme string against the official `neutts` phonemizer output in Task 0/staged validation.
2. **NeuCodec variant mismatch**: the NeuTTS NeuCodec (`emilia-yodas-english-neucodec`) must match codec.cpp's NeuCodec numerically (encode for ref + decode for output). Confirm in Task 0 before building the loop.
3. **Added-token survival**: the `<|speech_N|>` block must land in the gguf vocab contiguously; verify after backbone conversion.
