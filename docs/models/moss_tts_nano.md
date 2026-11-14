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
reference codes: equal to the reference's `batch_encode` on 96 % (F32), 95 %
(F16) and 56–72 % (Q8_0) of the codes on the eval clips. Intelligibility did
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

### Sampling

The reference samples the codebooks at temperature 1.7 / top_k 25 / top_p 0.8
(the backbone GGUF's `general.sampling.*`, temperature first, no repetition
penalty), and the text slot apart at temperature 1.5 / top_p 1. A host
applies one setting to every codec-side head, and with the codebooks' top_p
the audio end could not be drawn while its probability is below 0.2:
generations ran past the end. So the text slot is drawn model-side, like a
stop head: the family samples it from its two logits at
`codec.lm.moss_tts_nano.text_temperature` (1.5; `<= 0` = argmax) with its own
stream seeded by `codec_audio_lm_set_seed`, and hands the host a one-hot
head. `--temp 0` makes the codebooks greedy; for a fully greedy run also pass
`--override-kv codec.lm.moss_tts_nano.text_temperature=float:0`.

## Parity

Against `modeling_moss_tts_nano.py` (float32, sdpa) on the zh voice-clone
prompt (331 rows):

| Stage | Result |
|---|---|
| tokenizer (normalizer + llama.cpp SPM) | equal to SentencePiece on 18 / 18 texts |
| backbone hidden (llama.cpp starcoder2) | rel. error 4e-4 (F32 weights), 4e-3 (BF16) |
| depth decoder, teacher-forced frame 0 (BF16 adaptor) | 17 / 17 argmax equal, logits max abs diff 0.05 |
| greedy generation, all F32 | first 34 frames (17 tokens each) equal, then a float-noise near-tie flips |
| voice codes vs `batch_encode` (F32 codec) | 96 % equal (as plain `codec_encode`; deep RVQ levels) |

## Evaluation

Voice-clone read-back of `tts-cli synthesize` (BF16 backbone, F16 or Q8_0
codec + BF16 adaptor, model-default sampling) against the PyTorch reference
(`inference(mode="voice_clone", do_sample=True)`, float32) on the same texts
and voices: zh = 4 FLEURS cmn sentences in a FLEURS zh voice, ja = 4 FLEURS
ja sentences in a ja voice, en = 4 FLEURS en sentences in an en voice,
zh/en voice = 2 zh sentences in the en voice. ASR: whisper large-v3-turbo
(live-translate `lt-eval`, cascade mode) on the 16 kHz mono mix; CER for zh
(hypothesis to simplified) / ja and WER for en, punctuation and spaces
removed. Both sides use seeds 101 + i, 201 + i, ... 501 + i (20 utterances
per language, 10 for zh/en voice).

Sampling makes single runs vary a lot, on both sides: the ASR also
hallucinates repetitions on some clips (one en utterance scores WER 17.5),
so the table gives the corpus score per seed, its mean, the mean with each
utterance's errors capped at its length, and the median utterance.

| Set | Metric | Reference: seeds (mean / capped / median) | codec.cpp Q8_0 codec | codec.cpp F16 codec |
|---|---|---|---|---|
| zh | CER | .42 .13 .04 .00 .06 (.131 / .131 / .057) | .22 .14 .64 .02 .08 (.219 / .163 / .029) | 1.11 .04 .02 .20 .05 (.283 / .142 / .031) |
| ja | CER | .54 .51 .47 .45 1.49 (.692 / .513 / .549) | .33 .43 .45 .44 .52 (.433 / .433 / .424) | .53 .39 .52 .43 .59 (.492 / .492 / .496) |
| en | WER | .55 .28 .10 .04 .35 (.264 / .264 / .165) | .51 4.03 .15 .31 .10 (1.021 / .344 / .189) | .12 .40 .12 .17 .50 (.259 / .256 / .134) |
| zh, en voice | CER | .25 .13 .49 .07 .44 (.274 / .274 / .125) | .15 .20 .14 .37 .23 (.219 / .219 / .195) | .34 .23 .10 .35 .25 (.253 / .253 / .318) |

The single unseeded reference run quoted earlier (zh .042, ja .43, en .115,
zh/en voice .085) is one draw of the same spread; re-scored with this scorer
it is zh .042, ja .406, en .115, zh/en voice .085. Failure modes are the
reference's too: an early audio end (a clipped sentence), a repeated phrase.

Host speed, `tts-cli`, 8 threads, CPU only (AMD Ryzen AI Max+ 395), over
the 70 utterances of the F16 run (723 s of audio): generation (backbone +
depth decoder, prompt included) RTF 0.095, about 7.5 ms per 80 ms frame;
codec decode RTF 0.051; the voice (resample + encode) 0.35 s for a 6 s clip,
2.1 s for an 18 s one, once per voice.

`tests/e2e/moss_tts_nano_smoke.py` runs the voice, two sampled syntheses
and, with `MOSS_TTS_NANO_SRC` / `MOSS_AUDIO_TOKENIZER_SRC`, the parity checks
above; `ttscli_synthesize_smoke.py` and `python_tts_smoke.py` have a
`moss_tts_nano` case.
