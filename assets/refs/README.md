# assets/refs — NeuTTS in-context learning reference clips

Reference audio clips used as `--ref-audio` for NeuTTS ICL synthesis.
Each clip must be accompanied by a matching `.txt` transcript (exact words spoken).

---

## jfk_en.wav

| Field | Value |
|-------|-------|
| License | Public domain (US government work) |
| Source | https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav |
| Source content | Natural recording of John F. Kennedy's inaugural address (20 January 1961) |
| Trim | First 5.5 s of the source clip (trimmed with ffmpeg) |
| Format | 16-bit PCM WAV, 24000 Hz, mono |
| File size | ~258 KB |

**Transcript** (`jfk_en.txt`):
> And so, my fellow Americans, ask not what

**Provenance / license statement:**
The source recording is a US government work (presidential address delivered at a public ceremony)
and is in the public domain.  The clip is widely used in TTS and ASR CI under the same status
(see whisper.cpp samples/ for precedent).

**How the transcript was derived:**
The trimmed wav was transcribed with Whisper large-v3-turbo
(`openai/whisper-large-v3-turbo` via HuggingFace Transformers).  The transcript is the exact
text spoken in the 5.5 s clip — the sentence is incomplete because the clip ends mid-utterance.

**How the file was produced:**
```bash
curl -fsSL -o /tmp/jfk.wav https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
ffmpeg -y -i /tmp/jfk.wav -t 5.5 -ar 24000 -ac 1 -c:a pcm_s16le assets/refs/jfk_en.wav
```

**Usage:**
```bash
./build/tts-cli synthesize \
  --model models/distill_neucodec/distill_neucodec.gguf \
  --backbone models/hf/neutts_nano/neutts-nano-q8_0.gguf \
  --ref-audio assets/refs/jfk_en.wav \
  --ref-text "And so, my fellow Americans, ask not what" \
  --text "hello world" \
  -o /tmp/neutts_jfk_smoke.wav
```
