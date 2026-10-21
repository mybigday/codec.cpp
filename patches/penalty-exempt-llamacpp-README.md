# Penalty-exempt tokens for llama.cpp

`penalty-exempt-llamacpp.patch` lets the repetition / frequency / presence
penalties skip a list of tokens:

- `llama_sampler_init_penalties_exempt(last_n, repeat, freq, present, exempt, n_exempt)`
  (`llama_sampler_init_penalties` is the same with an empty list);
- `common_params_sampling::penalty_exempt`, used by `common_sampler_init`.

Model-independent: a speech LM's end-of-speech and pause tokens recur by
design, and penalizing them keeps the model from closing a segment cleanly
(KittenTTS 2 exempts `<|speech_end|>`, `[STOP]` and its silence token).
`codec_audio_lm_completion.penalty_exempt` (and a head group's) carries the list from the codec GGUF
to the host's sampler; a llama.cpp without this patch ignores it.

Stacked on `soprano-llamacpp.patch`; applied in order by
`cmake/SetupLlamaPatches.cmake`.
