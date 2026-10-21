# NeuTTS (Air / Nano)

[Neuphonic](https://huggingface.co/neuphonic) speech LMs (Air: Qwen2 0.5B;
Nano: English, German, French and Spanish variants) that write NeuCodec
tokens as `<|speech_N|>` vocab rows. The backbone runs in llama.cpp; the
prompt is a phoneme-mode in-context reference:

```
user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref phonemes} {input phonemes}<|TEXT_PROMPT_END|>
assistant:<|SPEECH_GENERATION_START|>{ref codes}
```

Input text must be phonemized by the caller (espeak-ng, `with_stress=True`,
`preserve_punctuation=True`, the backbone's language; or build tts-cli with
`-DCODEC_TTS_ESPEAK=ON` for English).

## Codec GGUF and voices

The codec GGUF is NeuCodec with the distill encoder
([neuphonic/distill-neucodec](https://huggingface.co/neuphonic/distill-neucodec));
its decoder is bit-identical to NeuCodec's, and the base NeuCodec encoder
(w2v-bert-2.0) is not implemented. A speaker (docs/codec_speaker.md) is:

- `ref_codes` I32 [T]: NeuCodec tokens of the reference clip;
- `ref_text`: the reference transcript as phonemes.

Builtin voices are upstream's `samples/` (`.pt` codes from the NeuCodec
encoder, `.txt` phonemized as neutts does): `jo` (default), `dave`, `emily`,
`paul`, `sophie`, `steven` (en-us), `mateo` (es), `greta` (de), `juliette`
(fr-fr). Use a voice in the backbone's language for non-English backbones.
`--ref-audio` + `--ref-text` derive a speaker with the distill encoder
(`codec_speaker_from_audio`, any sample rate, resampled in-graph); its codes
differ from the base encoder's, as expected from a different encoder over the
same code space. `--save-speaker` keeps it for reuse.

```bash
python scripts/convert-to-gguf.py --checkpoint-path distill_neucodec \
    --model-type distill_neucodec --outtype F16 -o distill_neucodec-F16.gguf
python scripts/bake-speakers.py neutts distill_neucodec-F16.gguf neutts-codec-F16.gguf \
    --samples neutts/samples --voice jo=en-us --voice dave=en-us --voice greta=de \
    --voice juliette=fr-fr --default jo        # ... one --voice per sample
```

## Runtime

```bash
tts-cli synthesize --backbone neutts-air-q8_0.gguf --model neutts-codec-F16.gguf \
    --voice dave --text "ðə bˈɜːtʃ kənˈuː slˈɪd ɔnðə smˈuːð plˈæŋks." --output out.wav
```

`--ref-text` overrides the voice's transcript. Recommended sampling:
`--temp 1.0 --top-k 50 --top-p 1.0 --min-p 0`. The stream is masked to the
speech range plus the end token, and the completion's grammar forbids eos for
the first 25 frames (`--min-len`). These ship in the backbone GGUF as `general.sampling.*` (llama.cpp's own keys); a host reads them as llama.cpp does and a setting given on the command line or in the request wins. `scripts/set-backbone-sampling.py` writes them into an existing backbone.
