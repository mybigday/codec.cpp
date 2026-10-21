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

### Quantized GGUFs

Codec GGUF: `scripts/quantize-gguf.py`. Validated on BreezyVoice (= the
CosyVoice-300M flow / HiFT / frontend), CPU and Vulkan, Q8_0 / Q5_K / Q4_K:

```bash
CT='(hift\.up\.\d+|cfm\.up\.0\.x)\.w$'   # ConvTranspose kernels
.venv/bin/python scripts/quantize-gguf.py breezyvoice.gguf codec-q4_k_m.gguf \
    --type Q4_K --fallback Q8_0 --tensor-type '^lm\.=Q8_0' \
    --tensor-type 's3t\.q\.embed=KEEP' \
    --flatten "^(?!.*$CT).*\.w$" --ci-major "$CT"
```

- `s3t.q.embed` is the speech tokenizer codebook; keep it F32.
- ConvTranspose kernels take `--ci-major`, every other conv1d kernel
  `--flatten`. The U-Net's ConvTranspose is the up-sampling conv of each
  inner level (`cfm.up.<i>.x`, all levels but the last; level 0 for v1); the
  last level's `x` is a plain conv1d.
- The LM adaptor (`lm.*`, v1's text encoder and embeddings) at Q4_K
  degrades the speech LM (repeated phrases); Q8_0 there keeps Q4 / Q5 codec
  files close to F16 intelligibility.

Backbone GGUF: `llama-quantize` from the patched llama.cpp. For the v1
`cosyvoice` arch keep the rel-pos tensors at F32: the 2-D `pos_bias_u` /
`pos_bias_v` are added to the query, and a block-quantized operand there
aborts the graph.

```bash
llama-quantize --tensor-type attn_pos=f32 breezyvoice-backbone.gguf backbone-q4_k_m.gguf Q4_K_M
```

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

## Chunked decoding

There is no separate streaming API. Decode overlapping token windows with
`codec_decode` and the same speaker, then cross-fade the shared region (as
OuteTTS does). For example, use windows of 50 new tokens plus 10 tokens of
context on each side, place window `[lo, hi)` at sample `lo * sample_rate /
token_rate`, and fade linearly over the `2 * 10` overlapping tokens. The right
context also gives v2/v3 their 3-token pre-lookahead. On the reference clip, this
transcribes like a single `codec_decode` on v1, v2 and v3. Each window re-runs
the prompt through the flow, as upstream streaming does.

## Tunable metadata

Inference knobs are GGUF metadata with upstream defaults; change them per run
with `--override-kv KEY=TYPE:VALUE` (repeatable) or, through the C API, the
`kv_overrides` argument of `codec_model_load_from_file_ext`. Keys missing from
older GGUFs fall back to the defaults below.

| Key | Default | Meaning |
|---|---|---|
| `cosyvoice.flow.speed` | 1.0 | speaking rate: the flow mel is linearly resampled to `int(frames / speed)` before HiFT (`token2wav(speed=)`) |
| `cosyvoice.flow.n_timesteps` | 10 | Euler steps |
| `cosyvoice.flow.cfg_rate` | 0.7 | classifier-free guidance |
| `cosyvoice.flow.max_prompt_s` | 0 (whole prompt) | flow prompt kept, in trailing seconds of the speaker's prompt tokens + mel; the LM prompt is unchanged |
| `codec.lm.cosyvoice.min_token_text_ratio` / `max_token_text_ratio` | 2 / 20 | speech-length bounds per text token |
| `codec.lm.cosyvoice.max_silent_run` | 5 (v3), 0 (off) | drop longer runs of silent tokens |

```bash
tts-cli synthesize ... --override-kv cosyvoice.flow.speed=float:1.2
```

### Recommended sampling

The speech head (sampled codec-side) uses the request's llama.cpp settings:

| Model | Settings |
|---|---|
| CosyVoice 300M / 2 / 3 | `--temp 1.0 --top-k 25 --top-p 0.8 --min-p 0 --repeat-last-n 10 --repeat-penalty 1.1 --samplers "penalties;top_k;top_p"` |
| BreezyVoice | `--temp 1.0 --top-k 25 --top-p 1.0 --min-p 0` |

Upstream's `ras_sampling` (nucleus top-p 0.8 / top-k 25, redrawn from the full
distribution when the pick occurs in the last 10 tokens) is not a llama.cpp
sampler; a repetition penalty of 1.1 over its 10-token window stands in for it
(without one, CosyVoice-300M cross-lingual runs to max_len on some seeds;
1.1 and 1.2 both stop and transcribe on seeds 1-3, measured 2026-10-21). These ship in the backbone GGUF as `general.sampling.*` (llama.cpp's own keys); a host reads them as llama.cpp does and a setting given on the command line or in the request wins. `scripts/set-backbone-sampling.py` writes them into an existing backbone.

### Faster flow decoding

The flow decode is the CFM U-Net unrolled over `n_timesteps` × 2 passes (CFG
cond + uncond), so its cost is linear in both. Fewer steps or `cfg_rate=0`
trade a little speaker fidelity for speed; intelligibility is unaffected.
Measured 2026-10-17 on CPU (Ryzen AI MAX+ 395, 12 threads), CosyVoice2 with
`--ref-audio` cloning, 5 zh/en sentences: Whisper large-v3-turbo CER, speaker
cosine (CAMPPlus) of each output against the default-settings output, and the
flow decode time for 4 s of audio (16 threads). Chatterbox S3Gen
(`chatterbox_s3g.flow.n_timesteps` / `cfg_rate`, same defaults) measures
alike.

| `n_timesteps` / `cfg_rate` | CER | speaker cos vs default | flow decode |
|---|---|---|---|
| 10 / 0.7 (default) | 0.000 | 1 | 1× |
| 6 / 0.7 | 0.000 | 0.993 | ~1.6× faster |
| 4 / 0.7 | 0.000 | 0.985 | ~2.5× faster |
| 10 / 0 | 0.000 | 0.967 | ~2× faster |
| 5 / 0 | 0.000 | 0.957 | ~3.9× faster |

`n_timesteps=6` keeps the voice closest; dropping CFG moves it further than
cutting steps for the same speedup.

```bash
tts-cli synthesize ... --override-kv cosyvoice.flow.n_timesteps=int:6
```

The flow also runs over the whole reference prompt: a 15 s clip is ~70% of the
CFM sequence for a 4 s sentence. `cosyvoice.flow.max_prompt_s` keeps only the
prompt's trailing seconds (acoustic context next to the target; the x-vector
and the LM prompt still come from the whole clip). Measured 2026-10-19 on CPU
(8 threads), `--ref-audio` cloning with the 15 s reference clip, one zh
sentence: decode time, Whisper CER, CAMPPlus speaker cosine of output vs
reference clip.

| `max_prompt_s` | 300M decode / spk cos | BreezyVoice | CosyVoice2 | CosyVoice3 | CER |
|---|---|---|---|---|---|
| 0 (whole, 15 s) | 8.1 s / 0.857 | 8.3 s / 0.848 | 6.9 s / 0.819 | 9.8 s / 0.875 | 0 |
| 6 | 6.0 s / 0.843 | 6.6 s / 0.806 | 5.1 s / 0.790 | 8.0 s / 0.869 | 0 |
| 4 | 5.4 s / 0.829 | 5.4 s / 0.786 | 4.0 s / 0.768 | 6.5 s / 0.881 | 0 |
| 3 | 4.7 s / 0.797 | 4.9 s / 0.801 | 3.7 s / 0.777 | 5.6 s / 0.865 | 0 |
| 2 | 4.1 s / 0.810 | 4.7 s / 0.781 | 3.4 s / 0.766 | 5.1 s / 0.868 | 0 |

## Runtime contract

- **Prompt** (`common/tts_runner.cpp::run_cosyvoice`, `cosyvoice/llm/llm.py`):
  v1 `[sos][spk_aff(x-vector)][text_encoder(prompt_text ++ text)][task_id][speech_emb(prompt tokens)]`,
  v2/v3 `[sos][backbone embed(prompt_text ++ text)][task_id][speech_emb(prompt tokens)]`.
  The v1 text encoder runs causally (`decoding_chunk_size=1`, `static_chunk_size=1`).
  v3 prepends `You are a helpful assistant.<|endofprompt|>` to the prompt text
  (to the target text in cross-lingual mode); instruct text must end with
  `<|endofprompt|>` (added for v2/v3 when missing).
- **Sampling**: the request's llama.cpp settings (see "Recommended sampling");
  `min_len = 2 × text tokens` (target text only, `--min-len` overrides), `max_len = 20 ×`.
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
- Upstream's exact streaming (flow / HiFT caches). Chunked playback works
  with `codec_decode` alone; see below.

## Validation (CPU, F32 GGUFs, reference = upstream PyTorch / BreezyVoice fork)

- Speaker frontend: speech tokens 100 % match (v1, v2, v3), x-vector cosine 1.0000.
- Flow + HiFT with shared noise: mel max |Δ| 1.4e-5 (v1, BreezyVoice, v2),
  3e-3 (v3 DiT); waveform correlation 0.99 (residual is HiFT phase drift).
- LM, teacher-forced along a sampled path: argmax agrees at every step
  (v1 103/103, BreezyVoice 180/180, v2 99/99, v3 110/110), max |Δ log p| < 4e-3.
- `tests/e2e/cosyvoice_decode_smoke.py`, `tests/e2e/cosyvoice_synthesize_smoke.py`
  (Whisper CER 0 on zero-shot, SFT and cross-lingual cases). Vulkan decode verified.
