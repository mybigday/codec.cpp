# assets/refs — NeuTTS in-context learning reference clips

Reference audio clips used as `--ref-audio` for NeuTTS ICL synthesis.
Each clip must be accompanied by a matching `.txt` transcript (exact words spoken).

---

## gettysburg_espeak_en.wav

| Field | Value |
|-------|-------|
| License | Public domain / CC0 |
| Source | Generated with eSpeak-NG 1.51 (`espeak-ng -v en-us -s 150 -p 50`) |
| Source text | Opening line of the Gettysburg Address (1863, Abraham Lincoln; US government speech, no copyright) |
| Duration | 5.92 s |
| Format | 16-bit PCM WAV, 24000 Hz, mono |
| File size | ~278 KB |

**Transcript** (`gettysburg_espeak_en.txt`):
> Four score and seven years ago our fathers brought forth on this continent a new nation.

**Provenance / license statement:**
The source text (Gettysburg Address, 1863) is a US government work and is in the public domain.
The synthesized audio was generated entirely by eSpeak-NG, an open-source speech synthesizer
(Apache-2.0 / GPL). Machine-generated speech has no independent copyright under US law; the file
is dedicated to the public domain (CC0 1.0 Universal).

**How the transcript was derived:**
The text was supplied directly to eSpeak-NG as input. Whisper large-v3-turbo (openai/whisper-large-v3-turbo)
was run on the output and returned the transcript verbatim, confirming the two match exactly.

**How the file was produced:**
```bash
espeak-ng -v en-us -s 150 -p 50 \
  "Four score and seven years ago our fathers brought forth on this continent a new nation." \
  -w /tmp/espeak_ref_raw.wav
ffmpeg -i /tmp/espeak_ref_raw.wav -ar 24000 -ac 1 -c:a pcm_s16le assets/refs/gettysburg_espeak_en.wav
```

**Usage:**
```bash
./build/tts-cli synthesize \
  --model models/distill_neucodec/distill_neucodec.gguf \
  --backbone models/hf/neutts_nano/neutts-nano-q8_0.gguf \
  --ref-audio assets/refs/gettysburg_espeak_en.wav \
  --ref-text "Four score and seven years ago our fathers brought forth on this continent a new nation." \
  --text "Hello world." \
  -o /tmp/neutts_asset_smoke.wav
```
