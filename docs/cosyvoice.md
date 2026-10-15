# CosyVoice family (CosyVoice-300M, BreezyVoice, CosyVoice2, CosyVoice3)

One `cosyvoice` codec architecture and one codec_lm adaptor cover every
released checkpoint; the variant is carried in GGUF metadata (`cosyvoice.*`,
`codec.lm.cosyvoice.*`).

| Checkpoint | Speech LM (llama.cpp backbone) | Tokens | Flow | Vocoder |
|---|---|---|---|---|
| [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) (`-SFT`, `-Instruct`) | WeNet rel-pos Transformer, arch `cosyvoice` (`patches/cosyvoice-llamacpp.patch`) | S3 v1, 50 Hz, 4096 | Conformer → length regulator → U-Net CFM | HiFT 22050 Hz |
| [MediaTek-Research/BreezyVoice](https://huggingface.co/MediaTek-Research/BreezyVoice) | same arch, fine-tuned `llm.pt` | same | same weights as 300M | same weights as 300M |
| [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Qwen2-0.5B (stock) | S3 v2, 25 Hz, 6561 | UpsampleConformer → causal U-Net CFM (= Chatterbox S3Gen) | HiFT 24 kHz, SineGen2 |
| [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Qwen2-0.5B (stock) | S3 v3 (12-layer v2), 6561 | PreLookahead → DiT CFM (22 layers) | causal HiFT 24 kHz |

BreezyVoice ships CosyVoice-300M's `flow.pt`, `hift.pt`, `campplus.onnx` and
`speech_tokenizer_v1.onnx` byte for byte; it differs in `llm.pt`, plain top-k
sampling (no RAS), LayerNorm eps 1e-5 (its WeNet fork) and its fork's length
regulator (one nearest interpolation over prompt + target instead of
separate linear ones).

## Conversion

```bash
# codec GGUF: flow + HiFT + speaker encoders (speech tokenizer, CAMPPlus from
# campplus.onnx, prompt-mel filterbank) + spk2info voices + the LM adaptor
.venv/bin/python scripts/convert-to-gguf.py --checkpoint-path models/cosyvoice/CosyVoice2-0.5B \
    --output models/cosyvoice/cosyvoice2.gguf

# backbone GGUF: Qwen2 from llm.pt (+ the CosyVoice special tokens) for v2/v3,
# the `cosyvoice` arch with the Whisper multilingual vocab for v1 / BreezyVoice
.venv/bin/python scripts/convert-backbone-to-gguf.py --model-id models/cosyvoice/CosyVoice2-0.5B \
    --output models/cosyvoice/cosyvoice2-backbone.gguf --outtype f16
```

The checkpoint directory is the Hugging Face snapshot (the `cosyvoice*.yaml`
selects the version). Conversion needs `s3tokenizer` and `openai-whisper`
(speech tokenizer ONNX → weights, Whisper mel filters / vocab); the v2/v3
special-token list is read from a CosyVoice checkout in `.model-src/cosyvoice`.
The v1 backbone needs `patches/cosyvoice-llamacpp.patch`, applied to the
llama.cpp submodule by `cmake/SetupLlamaPatches.cmake` at configure time.

## Synthesis

```bash
# zero-shot voice cloning: the transcript must match the reference audio
tts-cli synthesize --model cosyvoice2.gguf --backbone cosyvoice2-backbone.gguf \
    --ref-audio ref.wav --ref-text "transcript of ref.wav" --text "要合成的文字" --output out.wav

# cross-lingual: reference audio without a transcript (the LM sees no prompt)
tts-cli synthesize ... --ref-audio ref.wav --text "Hello there."

# SFT voices (CosyVoice-300M-SFT / -Instruct, BreezyVoice): tts-cli voices --model X.gguf
tts-cli synthesize --model cosyvoice-300m-sft.gguf --backbone ... --voice zh_female --text "..."

# instruct: CosyVoice-300M-Instruct (with --voice) or CosyVoice2/3 (with --ref-audio)
tts-cli synthesize ... --instruct "用四川话说这句话" --text "..."
```

Voices from `spk2info.pt`: 中文女 `zh_female`, 中文男 `zh_male`, 日语男 `ja_male`,
粤语女 `yue_female`, 英文女 `en_female`, 英文男 `en_male`, 韩语女 `ko_female`.
BreezyVoice's voices also carry prompt tokens / mel (no transcript, so they run
in cross-lingual mode).

## Runtime contract

- **Prompt** (`common/tts_runner.cpp::run_cosyvoice`, `cosyvoice/llm/llm.py`):
  v1 `[sos][spk_aff(x-vector)][text_encoder(prompt_text ++ text)][task_id][speech_emb(prompt tokens)]`,
  v2/v3 `[sos][backbone embed(prompt_text ++ text)][task_id][speech_emb(prompt tokens)]`.
  The v1 text encoder runs causally (`decoding_chunk_size=1`, `static_chunk_size=1`).
  v3 prepends `You are a helpful assistant.<|endofprompt|>` to the prompt text
  (to the target text in cross-lingual mode); instruct text must end with
  `<|endofprompt|>` (added for v2/v3 when missing).
- **Sampling**: `ras_sampling` (nucleus top_p 0.8 / top_k 25, redraw from the
  full distribution when the pick occurs in the last 10 tokens) or BreezyVoice's
  top-k 25; `min_len = 2 × text tokens` (target text only), `max_len = 20 ×`.
  As upstream, the min-length mask hits id `speech_token_size`, which in v3 is
  sos rather than eos, so v3 has no effective minimum. v3 drops runs of more
  than 5 silent / breath tokens before the flow.
- **Speakers** (`codec_speaker`): `prompt_token`, `prompt_feat` (Matcha mel of the
  16 kHz clip resampled to the model rate; 24 kHz variants trim to
  `2 × tokens`), `embedding` (CAMPPlus). `codec_speaker_from_audio` runs the
  frontend (resample, speech tokenizer, CAMPPlus, mel) as one graph; clips are
  limited to 30 s. CosyVoice's `campplus.onnx` averages a partial last 100-frame
  CAM segment over the full window (ONNX AveragePool), unlike the PyTorch
  CAMPPlus Chatterbox uses.
- **Decode**: one graph: flow encoder → (v1) length regulation → 10 Euler
  steps × CFG 0.7 → HiFT. v1's relative-position self-attention runs in
  llama.cpp as a KQ bias built from stock RoPE (see the patch's
  `src/models/cosyvoice.cpp`).

## Not ported

- Text normalization (wetext / ttsfrd: numbers, symbols) and the frontend's
  sentence splitting: pass normalized text of sentence length.
- BreezyVoice's g2pw polyphone annotation (注音 inserted after ambiguous
  characters, `single_inference.py`): plain text works; annotate externally for
  polyphone control.
- Streaming (chunked flow / HiFT caches); `speed`.

## Validation (CPU, F32 GGUFs, reference = upstream PyTorch / BreezyVoice fork)

- Speaker frontend: speech tokens 100 % match (v1, v2, v3), x-vector cosine 1.0000.
- Flow + HiFT with shared noise: mel max |Δ| 1.4e-5 (v1, BreezyVoice, v2),
  3e-3 (v3 DiT); waveform correlation 0.99 (residual is HiFT phase drift).
- LM, teacher-forced along a sampled path: argmax agrees at every step
  (v1 103/103, BreezyVoice 180/180, v2 99/99, v3 110/110), max |Δ log p| < 4e-3.
- `tests/e2e/cosyvoice_decode_smoke.py`, `tests/e2e/cosyvoice_synthesize_smoke.py`
  (Whisper CER 0 on zero-shot, SFT and cross-lingual cases). Vulkan decode verified.
