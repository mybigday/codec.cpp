# codec.cpp documentation

- [model_status.md](model_status.md): support status of every codec and TTS model.
- Models (`models/`): conversion, quantization, synthesis and runtime contract per family.
  - [chatterbox.md](models/chatterbox.md): Chatterbox, Chatterbox Multilingual, Chatterbox Turbo
  - [cosyvoice.md](models/cosyvoice.md): CosyVoice-300M, BreezyVoice, CosyVoice2, CosyVoice3
  - [kitten_tts2.md](models/kitten_tts2.md): KittenTTS 2
- Design and API:
  - [codec_lm.md](codec_lm.md): the LM adaptor (`codec_lm`) between a llama.cpp backbone and a codec
  - [codec_common_api.md](codec_common_api.md): `codec_audio_lm`, the audio-LM inference API a llama.cpp host drives, its model families, and the ready-made hosts (tts-cli, tts-server, Python)
  - [codec_speaker.md](codec_speaker.md): decoder-side voice conditioning (`codec_speaker`)
  - [audio_speaker_encoders.md](audio_speaker_encoders.md): LM-side speaker encoders (`codec_lm_speaker_encode*`)
