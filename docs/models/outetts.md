# OuteTTS (0.2 / 0.3 / 1.0)

[OuteAI](https://huggingface.co/OuteAI) speech LMs that read text and write
codec tokens as ordinary vocab rows, so the backbone runs unmodified in
llama.cpp and codec.cpp only decodes:

| Model | Interface | Codec | Codes in the vocab |
|---|---|---|---|
| OuteTTS-0.2-500M | v1 | WavTokenizer (75 tok/s, 4096) | `<|0|>`..`<|4095|>` |
| OuteTTS-0.3-500M / 0.3-1B | v2 | WavTokenizer | `<|0|>`..`<|4095|>` |
| OuteTTS-1.0-0.6B / Llama-OuteTTS-1.0-1B | v3 | DAC.speech (2 × 1024) | `<|c1_N|>`, `<|c2_N|>` |

tts-cli detects the interface from the backbone vocab (`<|c1_0|>` → v3,
`<|space|>` → v2, `<|text_sep|>` + `<|code_start|>` → v1).

## Codec GGUF and voices

Each OuteTTS model gets its own codec GGUF with the reference voices baked in
as builtin speakers (docs/codec_speaker.md). A speaker is the reference's
prompt in two parts:

- `ref_text`: the speaker's text section. 1.0: the transcript; 0.2/0.3: the
  processed words, space-separated (0.3 keeps punctuation tokens such as
  `<|comma|>` glued to their word).
- `audio_prompt` I32 [N]: the backbone token ids of the speaker's audio
  section, everything upstream's `get_completion_prompt` puts after
  `<|audio_start|>\n` (1.0 word blocks; 0.2/0.3 `word<|t_…|>codes` lines).

The token ids belong to that backbone's tokenizer, which is why the codec
GGUF is per model; the runtime rejects ids outside the backbone vocab.

The WavTokenizer for 0.2/0.3 is OuteAI's
[wavtokenizer-large-75token-interface](https://huggingface.co/OuteAI/wavtokenizer-large-75token-interface)
(the original WavTokenizer-large-speech-75token release). The novateur repo
was later replaced by a v2 checkpoint with different weights; decoding OuteTTS
codes with it is unintelligible.

```bash
# 0.2 / 0.3 codec: merge the interface encoder + decoder into one state dict
python - <<'EOF'
import torch
e = torch.load("oute_wavtok/encoder/encoder_model.pt", map_location="cpu", weights_only=False)
d = torch.load("oute_wavtok/decoder/decoder_model.pt", map_location="cpu", weights_only=False)
sd = dict(e); sd.update({k: v for k, v in d["model_state_dict"].items() if k != "codebook_weights"})
torch.save({"state_dict": sd}, "oute_wavtok_merged.ckpt")
EOF
python scripts/convert-to-gguf.py --checkpoint-path oute_wavtok_merged.ckpt \
    --model-type wavtokenizer --outtype F32 -o oute-wavtokenizer-F32.gguf

# voices: outetts 0.2.x package v1/default_speakers (0.2), 0.3.x package
# v2/default_speakers/parquet/speakers.parquet (0.3), 0.4.x v3 JSON (1.0)
python scripts/bake-speakers.py outetts oute-wavtokenizer-F32.gguf outetts-0.3-500m-codec.gguf \
    --interface 2 --tokenizer OuteTTS-0.3-500M --model-id OuteAI/OuteTTS-0.3-500M \
    --speakers speakers.parquet --default en_male_1
python scripts/bake-speakers.py outetts ibm-research--DAC.speech-F32.gguf outetts-1.0-0.6b-codec.gguf \
    --interface 3 --tokenizer OuteTTS-1.0-0.6B --model-id OuteAI/OuteTTS-1.0-0.6B \
    --speakers outetts/version/v3/default_speakers/json --default en_female_1_neutral
```

`--tokenizer` is the model's HF tokenizer directory. A custom outetts speaker
JSON becomes a speaker file with `--speaker-file` (then `tts-cli --speaker-file`).
OuteTTS cannot derive a speaker from audio in codec.cpp (upstream needs a
Whisper word alignment).

## Runtime

```bash
tts-cli synthesize --backbone OuteTTS-0.3-500M-Q8_0.gguf --model outetts-0.3-500m-codec.gguf \
    --voice en_female_1 --text "The birch canoe slid on the smooth planks." --output out.wav
```

- Prompt: upstream `prompt_processor.py` of the interface. 1.0 joins the
  speaker text and the input like `merge_speaker_text`.
- Input text: 1.0 is passed as is. 0.2/0.3 are lowercased and reduced to
  `a-z` words (0.3 keeps `. ! ? , "` as punctuation tokens). Numbers to words,
  romanization (uroman / MeCab) and other language prep are the caller's job.
- Grammar: the stream is constrained to the input's words, each followed by a
  complete audio block of at least one in-range code (0..4095 / 0..1023).
  Markers such as `<|audio_end|>` are matched as tokens, so they cannot be
  spelled out of text pieces; the loop also stops on any end-of-generation
  token.
- Recommended sampling: 1.0 `--temp 1.0 --top-k 50 --top-p 1.0 --min-p 0
  --repeat-penalty 1.1 --repeat-last-n 64` (README); 0.2/0.3 `--temp 0.1
  --top-k 50 --top-p 1.0 --min-p 0 --repeat-penalty 1.1 --repeat-last-n -1`
  (`GenerationConfig`). These ship in the backbone GGUF as `general.sampling.*` (llama.cpp's own keys); a host reads them as llama.cpp does and a setting given on the command line or in the request wins. `scripts/set-backbone-sampling.py` writes them into an existing backbone. Context: 8192 tokens for 1.0, 4096 for
  0.2/0.3 (upstream `max_length`).
