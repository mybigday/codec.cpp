# TTS CLI (`examples/tts.py`)

A multi-model TTS driver that wires HF tokenizer + text-embed extraction
into llama.cpp's backbone + codec.cpp's codec_lm AR loop + codec.cpp's
codec decode, producing a WAV from a text prompt (and optional reference
audio for voice-clone models).

## Wired profiles

| Profile | Params | HF id | Kind | Speaker config |
|---|---|---|---|---|
| `csm` | ~1B | `sesame/csm-1b` | zero-shot TTS, Llama-3.2-1B + Mimi 32-codebook | no |
| `qwen3-tts` | ~0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | voice-clone TTS, Qwen3 talker + qwen3_tts_tokenizer 16-codebook | yes |
| `moss-ttsd-v0.5` | ~1.7B | `fnlp/MOSS-TTSD-v0.5` | two-speaker dialogue voice-clone, Qwen3-1.7B + XY-Tokenizer 8-codebook | yes |
| `moss-ttsd-v0.7` | ~2B | `fnlp/MOSS-TTSD-v0.7` | same shape as v0.5, more training data | yes |

### Speaker config (`--speaker-config alice.json`)

```json
{
  "ref_audio": "alice.wav",
  "ref_text":  "Hello, my name is Alice.",
  "language":  "Auto",
  "x_vector_only_mode": false
}
```

- `ref_audio` may be absolute or relative to the JSON file's directory.
- `ref_text` format is profile-specific (MOSS-TTSD wants `[S1]…[S2]…`).
- `language` is profile-specific (`Auto`, `Chinese`, `English`, …).
- `x_vector_only_mode` is Qwen3-TTS-only (skips the ICL ref-codes path
  and uses speaker-embedding-only).

### Sample runs

```bash
# CSM (zero-shot)
.venv/bin/python examples/tts.py --model csm \
    --text "Hello world." --speaker 0 \
    --output /tmp/hello_csm.wav --max-frames 100 --greedy

# Qwen3-TTS (voice clone)
.venv/bin/python examples/tts.py --model qwen3-tts \
    --text "Good morning, this is a parity check." \
    --speaker-config alice.json \
    --output /tmp/hello_qwen.wav \
    --temperature 0.9 --top-k 50 --top-p 1.0 --seed 7

# MOSS-TTSD (two-speaker dialogue)
.venv/bin/python examples/tts.py --model moss-ttsd-v0.5 \
    --text "[S1]Hello![S2]How are you?" \
    --speaker-config alice.json \
    --output /tmp/hello_moss.wav \
    --temperature 0.9 --top-k 50 --top-p 0.95 --seed 7
```

## Pending — under-3B models with conversion + profile gaps

The user asked for "MOSS-TTS 全系列盡力支援 + CLI <3B" — these are the
remaining under-3B models in the MOSS / LFM2 family, each with its
own architectural lift before a CLI profile lands:

### MOSS-TTS-Realtime (~2.3B)

- HF: `OpenMOSS-Team/MOSS-TTS-Realtime`
- Arch: custom `MossTTSRealtime` — Qwen3-2B backbone + 4-layer
  `MossTTSRealtimeLocalTransformer` depth decoder + 17 channels
  (1 text head + 16 RVQ audio heads, all summed at the input).
- **Converter gap**: `scripts/converters/lm_adaptor/moss_ttsd.py` doesn't
  cover `MossTTSRealtime` yet.  Conceptually fits `residual_depth_ar`
  (the local_transformer is a standard depth decoder), but the cb-0
  text head needs the same dual-vocab handling as MOSS-TTSD's parallel
  cb-0 (text-vocab merged with speech codes).
- **Runtime gap**: same as MOSS-TTSD — the codec_lm's cb-0 has a much
  wider vocab than cb-1..16; needs `allowed_token_range` masking and
  HF processor delegation for the final decode.

### MOSS-TTS-Nano-100M (100M)

- HF: `OpenMOSS-Team/MOSS-TTS-Nano-100M`
- Arch: custom `MossTTSNanoForCausalLM` — **GPT-2** backbone
  (n_layer=12, n_embd=768) + 4-layer local-GPT-2 depth decoder + 17
  channels (1 text head + 16 audio heads at audio_vocab=1024).
  Audio tokenizer: `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
  (48 kHz, 16 codebooks).  We already have this codec converted as
  `models/moss_audio_nano/moss_audio_nano.gguf`.
- **Converter gap**:
  - GPT-2 backbone converter for the `MossTTSNano` wrapper (rename
    `transformer.*` → standalone GPT-2 layout) — needs a new
    `prep_moss_tts_nano` in `scripts/convert-backbone-to-gguf.py`.
  - `scripts/converters/lm_adaptor/moss_ttsd.py` explicitly notes
    `MossTTSNanoForCausalLM` is pending M3 (needs `residual_depth_ar`
    Nano variant — the local_transformer has `wte = nn.Identity()`,
    consuming embeddings directly rather than token IDs).
- **Runtime gap**: same cb-0 dual-vocab + text-head split as Realtime.

### LFM2-Audio (1.5B)

- HF: `LiquidAI/LFM2-Audio-1.5B`
- Arch: `Lfm2AudioForConditionalGeneration` — LFM2-1.5B backbone +
  Mimi 8-codebook decoder, but it's fundamentally a **chat model**
  (audio-in + interleaved text/audio-out via `generate_interleaved`),
  not a pure TTS model.  For TTS mode, we'd:
  - Frame the prompt as `system + user + assistant` with text-only
    user input.
  - Drive an AR loop that **switches modality** per step (TEXT vs
    AUDIO_OUT), where TEXT positions come from the backbone's lm_head
    (over 65k LFM2 text vocab) and AUDIO positions come from
    codec_lm's depth decoder.
- **Runtime gap**: the `LlamaBackbone` wrapper currently only exposes
  hidden states (`embeddings=True` mode); for TEXT mode sampling we'd
  need a parallel path that reads logits via `llama_get_logits_ith`
  (or computes them via the backbone's lm_head weights externally).
- **Converter**: already done.
  Backbone GGUF at `models/lfm2_audio/lfm_backbone.gguf`;
  codec_lm + Mimi at `models/lfm2_audio/lfm2_audio.gguf` +
  `models/mimi/mimi.gguf`.

## Out of scope for the <3B CLI

- `OpenMOSS-Team/MOSS-TTSD-v1.0` (~8.4B), `OpenMOSS-Team/MOSS-TTS`
  (~8.5B), `OpenMOSS-Team/MOSS-TTS-Local-Transformer` (~3B): all
  `MossTTSDelayModel` arch — converters exist (the existing
  `lm_adaptor/moss_ttsd.py` covers it).  Same profile shape as
  MOSS-TTSD-v0.5/v0.7 would work; only the model size keeps them out
  of the CLI-<3B scope.
- `OpenMOSS-Team/MOSS-TTS-GGUF`: pre-quantized GGUF artefacts for
  MOSS-TTS; targets llama-cpp users directly.  Different consumption
  pattern — out of `examples/tts.py`'s scope.
- `OpenMOSS-Team/MOSS-VITS-*`: VITS-style end-to-end TTS, not
  codec-LM-based — different architecture entirely.

## Cross-cutting open items

- The TTS CLI currently leans on HF transformers (or `liquid_audio` /
  `qwen_tts`) for tokenization, text-embed lookup, and (for MOSS-TTSD)
  the final audio decode.  Decoupling that from HF — by serialising
  speaker embeddings + ref-audio codes into a "compiled" speaker
  artefact (e.g. `alice.compiled.json` with cached `ref_code.npz` +
  `spk_emb.npy`) — would let the CLI run from a pure C++ stack after
  a one-time prep.
- The `allowed_token_range` cb-0 masking is currently MOSS-TTSD only;
  Qwen3-TTS could benefit from the same (currently it relies on
  `suppress_tokens` not being applied since cb-0 is already a
  speech-only 3072 vocab).
- Free-running drift on F16 backbones is sensitive to the prompt
  content; if a model NaN's despite the parity smoke passing, try
  the BF16 backbone GGUF (see MOSS-TTSD's profile registry entry).
