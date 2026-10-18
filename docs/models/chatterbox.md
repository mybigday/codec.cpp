# Chatterbox family (Chatterbox, Chatterbox Multilingual, Chatterbox Turbo)

ResembleAI's Chatterbox is a stack: the T3 speech-token LM, the S3 speech
tokenizer (S3T), and the S3Gen token-to-waveform decoder (S3G), plus a
VoiceEncoder and CAMPPlus for voice cloning. The T3 transformer runs in
llama.cpp; everything else is one codec GGUF in codec.cpp.

| Checkpoint | T3 backbone (llama.cpp) | T3 adaptor (codec_lm) | S3Gen |
|---|---|---|---|
| [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) (`t3_cfg`) | Llama-520M (stock `llama`) | own BPE text tokenizer, perceiver cond, learned PEs, CFG pair | CFM, 10 Euler steps × CFG 0.7 |
| multilingual (`t3_23lang`, `t3_mtl23ls_v2`, `t3_mtl23ls_v3`) | same | same, 2454-row text table | same |
| [ResembleAI/chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (`t3_turbo_v1`) | GPT-2 medium (stock `gpt2`, carries the GPT-2 tokenizer) | no perceiver / emotion / learned PEs, single lane (no CFG), speech head bias | MeanFlow, 2 steps, no CFG |

Speech tokens are S3 v2: 25 Hz, payload ids `0..6560`; T3's speech vocab is
8194 with start / stop speech tokens 6561 / 6562. S3T encodes 16 kHz audio,
S3G decodes at 24 kHz.

## Conversion

```bash
# codec GGUF: S3Gen + S3T + VoiceEncoder + CAMPPlus + builtin voice (conds.pt) + T3 adaptor
.venv/bin/python scripts/convert-to-gguf.py --checkpoint-path models/chatterbox_turbo \
    --model-type chatterbox_s3g --lm-source models/chatterbox_turbo \
    --output models/chatterbox_turbo/chatterbox_turbo.gguf

# T3 backbone GGUF (Llama-520M, or GPT-2 medium + tokenizer for Turbo)
.venv/bin/python scripts/convert-backbone-to-gguf.py --model-id models/chatterbox_turbo \
    --output models/chatterbox_turbo/gpt2_backbone.gguf --outtype f16
```

The same two commands convert `models/chatterbox` and
`models/chatterbox_multilingual`; the T3 checkpoint file name selects the
variant. Precision flags (`--outtype`, `--quantization`, `--lm-outtype`,
`--lm-quantization`) are described in the top-level README.

### Quantized codec GGUFs

`scripts/quantize-gguf.py` re-types an existing codec GGUF. Validated on
Turbo (CPU and Vulkan, Whisper CER 0 for Q8_0 / Q5_K / Q4_K, builtin voice and
`--ref-audio`):

```bash
CT='hift\.up\.\d+\.w$'   # ConvTranspose kernels
.venv/bin/python scripts/quantize-gguf.py chatterbox_turbo.gguf codec-q4_k_m.gguf \
    --type Q4_K --fallback Q8_0 --tensor-type '^lm\.=Q8_0' \
    --tensor-type 's3t\.q\.embed=KEEP' \
    --flatten "^(?!.*$CT).*\.w$" --ci-major "$CT"
```

- `s3t.q.embed` is the S3 tokenizer codebook; keep it F32.
- Conv1d kernels need `--flatten` and ConvTranspose kernels `--ci-major`
  (the quantizable layouts); depthwise kernels stay at their type.
- The LM adaptor (`lm.*`) is a separate part; Q8_0 there keeps Q4 / Q5 files
  closer to F16.

## Synthesis

```bash
# builtin voice (conds.pt)
tts-cli synthesize --model chatterbox_turbo.gguf --backbone gpt2_backbone.gguf \
    --text "Hello there." --output out.wav

# voice clone: --ref-audio derives the T3 VoiceEncoder row and the S3G speaker
tts-cli synthesize ... --ref-audio ref.wav --text "Hello there."
```

Multilingual T3 expects the text its tokenizer would see upstream
(`MTLTokenizer.encode` before BPE): lowercase, NFKD, the language conversion
(zh → Cangjie, ja → hiragana) and the `[lang]` prefix. That preparation is the
caller's; codec.cpp tokenizes what it is given.

## Tunable metadata

GGUF metadata, overridable with `--override-kv KEY=TYPE:VALUE`:

| Key | Default | Meaning |
|---|---|---|
| `chatterbox_s3g.flow.n_timesteps` | 10 (Turbo: 2) | flow steps |
| `chatterbox_s3g.flow.cfg_rate` | 0.7 (MeanFlow: not used) | flow classifier-free guidance |
| `codec.lm.chatterbox.sampling.temperature` | 0.8 | T3 sampling |
| `codec.lm.chatterbox.sampling.min_p` / `top_p` / `top_k` | 0.05 / 1.0 / – (Turbo: – / 0.95 / 1000) | T3 sampling |
| `codec.lm.chatterbox.sampling.repetition_penalty` | 1.2 | T3 sampling |
| `codec.lm.chatterbox.sampling.cfg_weight` | 0.5 (Turbo: none) | T3 CFG |

Fewer flow steps trade speaker fidelity for speed the same way as CosyVoice's
flow; see "Faster flow decoding" in [cosyvoice.md](cosyvoice.md).

## Runtime contract

- **T3 prompt** (`src/lm/chatterbox_t3.cpp`, `common/tts_runner.cpp`):
  cond rows, then text embeddings, then speech BOS. Llama T3: 34 cond rows
  (speaker projection, 32 perceiver queries over the cond speech tokens,
  emotion), start/stop text tokens, two speech BOS, learned text / speech
  position embeddings, CFG as a second sequence with the text zeroed.
  Turbo: speaker projection + 375 raw `speech_emb` rows, one speech BOS,
  3 trailing silence codes before S3Gen; sampling order temperature →
  top-k → top-p → repetition penalty (HF warper order).
- **Speakers** (`codec_speaker`, see [codec_speaker.md](../codec_speaker.md)):
  S3G slots `prompt_token` (S3 tokens), `prompt_feat` (24 kHz mel) and
  `embedding` (CAMPPlus x-vector); T3 takes the 256-d VoiceEncoder embedding
  (see [audio_speaker_encoders.md](../audio_speaker_encoders.md)).
  `codec_speaker_from_audio` derives all of them from one clip in-library.
- **Decode**: one graph: S3G flow encoder → unrolled CFM / MeanFlow steps →
  HiFT.

## Validation

- `tests/e2e/chatterbox_s3g_decode_smoke.py`, `chatterbox_s3g_conds_smoke.py`,
  `chatterbox_speaker_smoke.py` (T3 cond rows corr 1.000000 vs
  `T3CondEnc.forward`), `chatterbox_converter_smoke.py`.
- `tests/e2e/ttscli_synthesize_smoke.py` cases `chatterbox_t3`,
  `chatterbox_turbo`, `chatterbox_turbo_ref` (Whisper CER).
- Turbo greedy T3 speech tokens are identical to PyTorch.
