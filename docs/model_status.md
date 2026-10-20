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
| Chatterbox-S3G | ✅ | Speakers: builtin (`spk.*`), saved files, or derived from reference audio when the GGUF bundles the encoders |
| CosyVoice (v1 / v2 / v3 token2wav) | ✅ | Speakers: builtin (`spk2info`), saved files, or derived from reference audio |
| NeuCodec | 🟡 | Base encoder unimplemented — use `distill_neucodec` for encode (same code space + decoder); NeuTTS voices as speakers ([models/neutts.md](models/neutts.md)) |

## TTS (`tts-cli synthesize`)

| Model | Status | Note |
|---|---|---|
| CSM (Sesame) | ✅ | |
| Chatterbox (Llama T3) | ✅ | See [models/chatterbox.md](models/chatterbox.md) |
| BlueMagpie | ✅ | |
| MOSS-TTS-Realtime | ✅ | |
| Qwen3-TTS | ✅ | Base model requires reference audio |
| LFM2-Audio | ✅ | Must sample (greedy degenerates, as in reference) |
| Chatterbox Turbo | ✅ | `--ref-audio` clones the voice (T3 VE row + S3G speaker); CPU and Vulkan verified; see [models/chatterbox.md](models/chatterbox.md) |
| CosyVoice-300M (base / SFT / Instruct) | 🟡 | Text normalization + sentence splitting not ported (pass normalized sentences); see [models/cosyvoice.md](models/cosyvoice.md) |
| CosyVoice2 / CosyVoice3 | 🟡 | Same as CosyVoice-300M |
| BreezyVoice | 🟡 | Same as CosyVoice-300M; g2pw 注音 annotation is external |
| KittenTTS 2 | 🟡 | 47 builtin voices + voice clone; English text normalizer not ported (pass spoken text); see [models/kitten_tts2.md](models/kitten_tts2.md) |
| MOSS-TTSD | 🟡 | Requires BF16 backbone (F16/Q8 → babble) |
| NeuTTS-Air | 🟡 | Input must be pre-phonemized (in-CLI espeak is opt-in); 9 builtin voices, `--ref-audio` needs the distill codec; see [models/neutts.md](models/neutts.md) |
| NeuTTS-Nano | 🟡 | Same as Air; seed-fragile (~1/3 seeds collapse to silence) |
| OuteTTS-1.0 (0.6B / Llama-1B) | 🟡 | Builtin voices in a per-model codec GGUF, no voice clone; see [models/outetts.md](models/outetts.md) |
| OuteTTS-0.2 / 0.3 | 🟡 | Same; text reduced to a-z words (numbers / romanization are the caller's job) |
| Pocket-TTS | 🟡 | Free-run quality noise-sensitive (also in reference); voice clone not validated |
| MOSS-TTS-Nano-100M | 🟡 | AR loop runs in HF; codec.cpp decode only |
| Soprano | 🚧 | Continuous hidden → Vocos path not wired into tts-cli |

Moshi is out of scope for `synthesize` (full-duplex, no Helium arch in llama.cpp); tts-cli refuses it.
