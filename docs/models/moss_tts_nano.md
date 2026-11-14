# MOSS-TTS-Nano-100M

[OpenMOSS-Team/MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M):
a 100M multilingual (zh / ja / en and more) voice-cloning TTS over
[MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)
(48 kHz stereo, 16 RVQ codebooks of 1024 at 12.5 Hz).

| Part | Where | What |
|---|---|---|
| global transformer (GPT-2 + RoPE, 12 × 768) | llama.cpp, stock `starcoder2` | the backbone: embd / tokens in, hidden out |
| local transformer (1 GPT-2 layer, 17 positions) | codec_lm `residual_depth_ar`, `codec.lm.depth.arch = "gpt2"` | per frame: the text slot, then 16 codes |
| audio embeddings / heads (tied) | codec_lm | input tables, heads, backbone rows |
| MOSS-Audio-Tokenizer-Nano | codec.cpp `moss_audio_tokenizer` | voice codes (encode), audio (decode) |

## Conversion

```bash
# codec + LM adaptor (local transformer, embeddings, prompt template ids, the
# tokenizer's normalizer).  The LM checkpoint is BF16: keep the adaptor BF16.
python scripts/convert-to-gguf.py --checkpoint-path MOSS-Audio-Tokenizer-Nano \
    --lm-source MOSS-TTS-Nano-100M --outtype F16 --lm-outtype BF16 \
    -o moss_tts_nano_codec.gguf            # 80 MB; + --quantization Q8_0: 62 MB

# global transformer → starcoder2 backbone GGUF with the SentencePiece vocab
python scripts/convert-backbone-to-gguf.py --model-id MOSS-TTS-Nano-100M \
    --output backbone-bf16.gguf --outtype bf16
```

The voice is encoded by the codec's encoder, so its precision shows in the
reference codes: equal to the reference's `batch_encode` on 96 % (F32),
92.5–95.7 % (F16) and 56–72 % (Q8_0) of the codes on the eval clips. Intelligibility did
not differ measurably between F16 and Q8_0 (below), but F16 keeps the voice
prompt closest to the reference's.

The global transformer is pre-LN LayerNorm with shift, biased q/k/v/o and
gelu_new FFN, final LayerNorm, full-width RoPE (base 10000) and no position
table: llama.cpp's `starcoder2`, no arch patch. `c_attn` splits into q/k/v;
MOSS rotates interleaved pairs, starcoder2 is NEOX, so each head's q/k rows
and biases are reordered to [even, odd]. The output head is tied to `wte`
(`token_embd`). The checkpoint is BF16, so `--outtype f32` stays BF16 and no
F16 variant is published.

The tokenizer is SentencePiece BPE with the `nmt_nfkc` normalizer. It goes in
as llama.cpp's SPM vocab (score merges, byte fallback, space prefix), which
has no normalizer: the codec GGUF carries the precompiled charsmap
(`codec.lm.moss_tts_nano.text_norm_charsmap`) and the family normalizes the
text (NFKC + NMT rules, whitespace collapsed) before the host tokenizes. The
result equals `SentencePieceProcessor.encode` on every test sentence (zh, ja,
en, full-width punctuation, half-width kana); the one known difference is a
text that starts with a user-defined piece such as `<user_inst>` (no leading
`▁`). The prompt template's pieces are baked as token ids, because the
reference encodes each piece on its own.

## Runtime

```bash
tts-cli synthesize --backbone backbone-bf16.gguf --model moss_tts_nano_codec.gguf \
    --text "你也可以咨询除贵国以外其他政府的建议。" --ref-audio voice.wav --output out.wav
```

The family (`common/audio_lm_family_moss_nano.cpp`) follows
`modeling_moss_tts_nano.py` `inference(mode="voice_clone")`:

- prompt (`build_inference_input_ids`): `<|im_start|>user\n<user_inst>\n-
  Reference(s):\n` `<|audio_start|>` reference rows `<|audio_end|>` the
  instruction block, the text, `\n</user_inst><|im_end|>\n<|im_start|>assistant\n`
  `<|audio_start|>`. A reference row is `wte[user_slot] + Σ audio_embeddings[k][code_k]`.
  Without a voice the reference section is `None` (continuation mode with no
  prompt audio).
- per frame (`_iter_generation_events`): the backbone's last hidden enters
  the local transformer at position 0; its text head picks the assistant
  slot or `<|audio_end|>`. The audio end stops the generation before any
  code is drawn; otherwise position 1 reads `wte[assistant_slot]` and
  positions 2..17 the previous code's embedding, each emitting the next code
  (KV cache across the 17 positions, reset per frame). The next backbone row
  is `wte[assistant_slot] + Σ audio_embeddings[k][code_k]`, the sum over the
  17 input tables.
- long text: voice clone splits text above 50 tokens at sentence / clause
  punctuation and joins the chunks with a 0.24 s pause; the family splits at
  150 UTF-8 bytes (the same chunks as the reference on every test sentence)
  and inserts the same pause. `max_frames` defaults to 300 per chunk.
- a voice is the codec speaker `ref_codes` I32 [16, T]
  (docs/codec_speaker.md): `codec_speaker_from_audio` resamples the clip to
  48 kHz in-graph (torchaudio's sinc), copies mono to both channels, encodes
  the whole clip and keeps `floor(n / 3840)` frames. Stereo references are
  downmixed first (the reference encodes both channels as they are).

### Sampling (required settings)

**Use `--temp 1.7 --top-k 25 --top-p 1.0 --min-p 0 --samplers "temperature;top_k;top_p"`.** These are the
backbone GGUF's `general.sampling.*` defaults from the converter, so a host that honours them gets them
automatically.

The reference (`modeling_moss_tts_nano.py`, `inference(do_sample=True)`) samples two kinds of head:
- **The stop decision.** `_sample_next_assistant_text_token` keeps two `text_lm_head` logits, the assistant slot
  (continue) and `<|audio_end|>` (stop), and samples them at `text_temperature` 1.5 with no top_p.
- **The 16 codebooks**, at `audio_temperature` 1.7 / top_k 25 / top_p 0.8.

The host samples every head with one llama.cpp chain. The family hands the host only the two candidates: their
log-probabilities at 1.5, multiplied by `codec.lm.moss_tts_nano.audio_temperature` (1.7, the host's
temperature). The host's temperature step then reproduces the reference's stop draw exactly, as long as nothing
in the chain truncates:
- **top_p 0.8 would truncate.** With top_p after the temperature, it drops the minority whenever the other
  candidate's probability is ≥ 0.8, so the end can't be drawn while its probability is ≤ 0.2. With top_p before
  the temperature, the end is lost below 0.307.
- **So top_p is 1.0 for every head.** The codebooks lose the reference's top_p 0.8, at the cost measured in
  Evaluation: no read-back loss, and lengths within the reference's.

`--temp 0` makes the whole frame greedy, the stop included. Running with another `--temp` changes the stop's
temperature in proportion: 1.5 × T / 1.7.

## Parity

Against `modeling_moss_tts_nano.py` (float32, sdpa) on the zh voice-clone
prompt (331 rows):

| Stage | Result |
|---|---|
| tokenizer (normalizer + llama.cpp SPM) | equal to SentencePiece on 18 / 18 texts |
| backbone hidden (llama.cpp starcoder2) | rel. error 4e-4 (F32 weights), 4e-3 (BF16) |
| depth decoder, teacher-forced frame 0 (BF16 adaptor) | 17 / 17 argmax equal, logits max abs diff 0.05 |
| greedy generation, all F32, stages chained by hand | first 34 frames (17 tokens each) equal, then a float-noise near-tie flips |
| greedy `codec_cpp.TTS` (the family path: normalize, tokenize, voice rows, frame loop), all F32, same voice codes on both sides | first 244 frames equal (91 % of 300 frames' codes) |
| voice codes vs `batch_encode` (F32 codec) | 96 % equal (as plain `codec_encode`; deep RVQ levels) |

## Evaluation

Voice-clone read-back of `tts-cli synthesize` (BF16 backbone, F16 or Q8_0
codec + BF16 adaptor, model-default sampling) against the PyTorch reference
(`inference(mode="voice_clone", do_sample=True)`, float32) on the same texts
and voices: zh = 4 FLEURS cmn sentences in a FLEURS zh voice, ja = 4 FLEURS
ja sentences in a ja voice, en = 4 FLEURS en sentences in an en voice,
zh/en voice = 2 zh sentences in the en voice. ASR: whisper large-v3-turbo
(whisper.cpp, Q8_0, VAD-segmented) on the 16 kHz mono mix; CER for zh
(hypothesis to simplified) / ja and WER for en, punctuation and spaces
removed. Each side ran 5 seeds (101 + i ... 501 + i, 20 utterances per
language, 10 for zh/en voice); the two use different RNGs, so a seed is not
the same draw on both sides: compare the spreads, not the columns.

Sampling makes single runs vary a lot, on both sides: the ASR also
hallucinates repetitions on some clips (one en utterance scores WER 17.5),
so the table gives the corpus score per seed, its mean, the mean with each
utterance's errors capped at its length, and the median utterance. The table after it compares the stop decisions.

| Set | Metric | Reference: seeds (mean / capped / median) | codec.cpp Q8_0 codec | codec.cpp F16 codec |
|---|---|---|---|---|
| zh | CER | .42 .13 .04 .00 .06 (.131 / .131 / .057) | .22 .14 .64 .02 .08 (.219 / .163 / .029) | 1.11 .04 .02 .20 .05 (.283 / .142 / .031) |
| ja | CER | .54 .51 .47 .45 1.49 (.692 / .513 / .549) | .33 .43 .45 .44 .52 (.433 / .433 / .424) | .53 .39 .52 .43 .59 (.492 / .492 / .496) |
| en | WER | .55 .28 .10 .04 .35 (.264 / .264 / .165) | .51 4.03 .15 .31 .10 (1.021 / .344 / .189) | .12 .40 .12 .17 .50 (.259 / .256 / .134) |
| zh, en voice | CER | .25 .13 .49 .07 .44 (.274 / .274 / .125) | .15 .20 .14 .37 .23 (.219 / .219 / .195) | .34 .23 .10 .35 .25 (.253 / .253 / .318) |

Stop decision and host settings, F16 codec, same 5 seeds. Each cell is the read-back error, then the mean
duration / number of clips over 20 s (out of 20, or 10 for zh with the en voice):

| Set | Reference (PyTorch) | **top_p 1.0 (this PR)** | drawn in the library | 25 candidates + dummies, top_p 0.8 | 2 candidates, top_p 0.8 |
|---|---|---|---|---|---|
| zh CER | .131 · 9.1 s / 0 | **.105 · 8.5 s / 0** | .283 · 9.4 s / 0 | .119 · 9.4 s / 0 | .071 · 9.9 s / 0 |
| ja CER | .692 · 11.7 s / 1 | **.541 · 11.1 s / 0** | .492 · 12.8 s / 3 | .449 · 19.9 s / 10 | .451 · 20.8 s / 11 |
| en WER | .264 · 8.6 s / 0 | **.241 · 7.1 s / 0** | .259 · 7.4 s / 0 | .231 · 8.7 s / 1 | .200 · 8.7 s / 1 |
| zh, en voice CER | .274 · 12.7 s / 1 | **.151 · 10.9 s / 0** | .253 · 12.9 s / 1 | .155 · 13.7 s / 1 | .191 · 13.7 s / 1 |

The 25-candidate column keeps the card's top_p 0.8. The 23 dummy tokens soak up 0.195 of the mass, so top_p keeps
exactly continue and end. That is exact only while both candidates are ≥ 1.07 %, and below that the end is cut.
In these runs the end probability after the speech often sits under 1 %, so the Japanese clips still run long.

The read-back scores do not penalise trailing audio, so the length columns are the stop measure.

A single unseeded reference run (zh .042, ja .43, en .115, zh/en voice .085;
with this scorer zh .042, ja .406, en .115, zh/en voice .085) is one draw of
the same spread. Failure modes are the reference's too: an early audio end (a
clipped sentence; 2 sub-second clips in the reference's 70 utterances, 0 and 1
at 2.5 s in codec.cpp's two runs) and a repeated phrase.

A run without a voice (the `None` reference section) and the
`--save-speaker` / `--speaker-file` round trip (bit-identical audio) work;
`codec_audio_lm_stream_poll` streams (chunks match the whole decode up to the
crossfades).

Host speed, `tts-cli`, 8 threads, CPU only (AMD Ryzen AI Max+ 395), over
the 70 utterances of the F16 run (723 s of audio): generation (backbone +
depth decoder, prompt included) RTF 0.095, about 7.5 ms per 80 ms frame;
codec decode RTF 0.051; the voice (resample + encode) 0.35 s for a 6 s clip,
2.1 s for an 18 s one, once per voice.

`tests/e2e/moss_tts_nano_smoke.py` runs the voice, two sampled syntheses
and, with `MOSS_TTS_NANO_SRC` / `MOSS_AUDIO_TOKENIZER_SRC`, the parity checks
above; `ttscli_synthesize_smoke.py` and `python_tts_smoke.py` have a
`moss_tts_nano` case.
