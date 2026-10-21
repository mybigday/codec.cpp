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

# T3 backbone GGUF (Llama-520M with T3's text tokenizer as its llama.cpp vocab,
# tokenizer.ggml.pre = chatterbox from patches/chatterbox-tokenizer-llamacpp.patch;
# or GPT-2 medium + tokenizer for Turbo). Older Llama backbones without the vocab
# fall back to the codec GGUF's tokenizer copy (deprecated).
.venv/bin/python scripts/convert-backbone-to-gguf.py --model-id models/chatterbox_turbo \
    --output models/chatterbox_turbo/gpt2_backbone.gguf --outtype f16
```

The same two commands convert `models/chatterbox` and
`models/chatterbox_multilingual`; the T3 checkpoint file name selects the
variant (`t3_mtl23ls_v3` before `v2` before `t3_23lang`). The multilingual
repo ships `s3gen.pt` / `ve.pt` instead of safetensors (the S3Gen weights are
identical to English Chatterbox's) and its tokenizer as
`grapheme_mtl_merged_expanded_v1.json` (2454 tokens, the T3 text-table size);
the converter reads both. A `config.json` next to the checkpoint sets release
defaults: the published multilingual codec files use
`{"n_timesteps": 6, "cfg_rate": 0.7}`. Precision flags (`--outtype`, `--quantization`, `--lm-outtype`,
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

# voice clone: --ref-audio derives one speaker holding T3's and S3Gen's conditioning
tts-cli synthesize ... --ref-audio ref.wav --text "Hello there." [--save-speaker voice.gguf]
```

A cloned voice is one codec speaker computed as `prepare_conditionals` does
(docs/codec_speaker.md): the S3Gen slots from the first 10 s, and for T3 the
VoiceEncoder embedding of the whole clip (librosa `trim(top_db=20)` first)
plus the S3 tokens of its first `speech_cond_prompt_len / 25` s (6 s, Turbo
15 s). Turbo first scales the clip to -27 LUFS (pyloudnorm BS.1770 gain,
`chatterbox_s3g.spk.ref_lufs`). Builtin voices use the GGUF's builtin T3
conds. Against the PyTorch reference on a 5.5 s clip (Turbo, F16 S3T):
VoiceEncoder cosine 1.00000, S3 tokens 92-93 % (FSQ boundary flips), output
VoiceEncoder similarity to the clip 0.737 (reference 0.727, 3-6 seeds).

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
| `chatterbox_s3g.flow.max_prompt_s` | 0 (whole prompt) | flow prompt kept, in trailing seconds of the speaker's prompt tokens + mel |
| `codec.lm.chatterbox.sampling.cfg_weight` | 0.5 (Turbo: none) | T3 CFG (`--cfg-weight`) |

### Recommended sampling

The host samples the T3 speech head (a codec-side head; codec.cpp hands it
the guided logits) with its llama.cpp settings.
The reference generate()'s, as tts-cli flags:

| Model | Settings |
|---|---|
| Chatterbox / Multilingual | `--temp 0.8 --top-k 0 --top-p 1.0 --min-p 0.05 --repeat-penalty 1.2 --repeat-last-n -1 --samplers "penalties;temperature;min_p;top_p"` |
| Chatterbox Turbo | `--temp 0.8 --top-k 1000 --top-p 0.95 --min-p 0 --repeat-penalty 1.2 --repeat-last-n -1 --samplers "temperature;top_k;top_p;penalties"` |

The penalty history of T3 starts with the start-of-speech token (sample_t3);
`mtl_tts.py`'s repetition_penalty 2.0 needs its AlignmentStreamAnalyzer, which
this runtime does not have, so the multilingual model keeps 1.2. These ship in the backbone GGUF as `general.sampling.*` (llama.cpp's own keys); a host reads them as llama.cpp does and a setting given on the command line or in the request wins. `scripts/set-backbone-sampling.py` writes them into an existing backbone.

Fewer flow steps trade speaker fidelity for speed the same way as CosyVoice's
flow; see "Faster flow decoding" in [cosyvoice.md](cosyvoice.md). So does a
shorter flow prompt: measured 2026-10-19 on CPU (8 threads), decode time and
CAMPPlus speaker cosine against the whole-prompt output, Whisper CER 0
throughout:

| `max_prompt_s` | Chatterbox (builtin voice) | Turbo `--ref-audio` | KittenTTS 2 |
|---|---|---|---|
| 0 (whole) | 5.2 s / 1 | 1.39 s / 1 | 1.46 s / 1 |
| 4 | 4.2 s / 0.958 | 0.94 s / 0.928 | 1.03 s / 0.963 |
| 3 | 3.7 s / 0.930 | 0.98 s / 0.926 | 0.91 s / 0.950 |
| 2 | 3.1 s / 0.957 | 0.81 s / 0.922 | 0.92 s / 0.943 |

## Runtime contract

- **T3 prompt** (`src/lm/chatterbox_t3.cpp`, `common/tts_runner.cpp`):
  cond rows, then text embeddings, then speech BOS. Llama T3: 34 cond rows
  (speaker projection, 32 perceiver queries over the cond speech tokens,
  emotion), start/stop text tokens, two speech BOS, learned text / speech
  position embeddings, CFG as a second sequence with the text zeroed.
  Turbo: speaker projection + 375 raw `speech_emb` rows, one speech BOS,
  3 trailing silence codes before S3Gen.
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
