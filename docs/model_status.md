# Model support status

✅ Full — 🟡 Limited / conditional (see note) — 🚧 WIP

## Codecs

| Model | Status | Note |
|---|---|---|
| Mimi | ✅ | |
| DAC | ✅ | |
| WavTokenizer | ✅ | |
| Qwen3-TTS-Tokenizer | ✅ | |
| NeMo Nano Codec | ✅ | |
| SNAC | ✅ | |
| XCodec2 | ✅ | |
| MOSS-Audio-Tokenizer | ✅ | |
| XY-Tokenizer | ✅ | |
| Pocket-TTS Mimi | ✅ | |
| BlueMagpie AudioVAE | ✅ | |
| Soprano (Vocos decoder) | ✅ | Decoder only (by design) |
| Chatterbox-S3T | ✅ | Encoder only (by design) |
| Chatterbox-S3G | 🟡 | Builtin conds only; no ref_wav embedding |
| NeuCodec | 🟡 | Base encoder unimplemented — use `distill_neucodec` for encode (same code space + decoder) |

## TTS (`tts-cli synthesize`)

| Model | Status | Note |
|---|---|---|
| CSM (Sesame) | ✅ | |
| Chatterbox (Llama T3) | ✅ | |
| BlueMagpie | ✅ | |
| MOSS-TTS-Realtime | ✅ | |
| Qwen3-TTS | ✅ | Base model requires reference audio |
| LFM2-Audio | ✅ | Must sample (greedy degenerates, as in reference) |
| Chatterbox Turbo | 🟡 | Builtin voice only (ref audio swaps only the VE speaker row); GPU untested |
| MOSS-TTSD | 🟡 | Requires BF16 backbone (F16/Q8 → babble) |
| NeuTTS-Air | 🟡 | Input must be pre-phonemized (in-CLI espeak is opt-in); ref encode needs `distill_neucodec` |
| NeuTTS-Nano | 🟡 | Same as Air; seed-fragile (~1/3 seeds collapse to silence) |
| OuteTTS-1.0 (0.6B / Llama-1B) | 🟡 | Default speaker only; eos emission seed-sensitive |
| Pocket-TTS | 🟡 | Free-run quality noise-sensitive (also in reference); voice clone not validated |
| MOSS-TTS-Nano-100M | 🟡 | AR loop runs in HF; codec.cpp decode only |
| Soprano | 🚧 | Continuous hidden → Vocos path not wired into tts-cli |

Moshi is out of scope for `synthesize` (full-duplex, no Helium arch in llama.cpp); tts-cli refuses it.
