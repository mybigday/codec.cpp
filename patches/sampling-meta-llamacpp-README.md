# Sampling metadata for freq / presence / DRY in llama.cpp

`sampling-meta-llamacpp.patch` extends the GGUF sampling defaults llama.cpp
already reads (`general.sampling.temp`, `top_k`, `top_p`, `min_p`,
`penalty_last_n`, `penalty_repeat`, `sequence`, ...) with:

- `general.sampling.penalty_freq`, `general.sampling.penalty_present`
- `general.sampling.dry_multiplier`, `general.sampling.dry_base`,
  `general.sampling.dry_allowed_length`, `general.sampling.dry_penalty_last_n`

They get `LLAMA_MODEL_META_KEY_SAMPLING_*` entries (appended to the enum, so
`llama_model_meta_key_str` resolves them) and `common_init_sampler_from_model`
applies them like the others: a value the user set on the command line
(`--frequency-penalty`, `--presence-penalty`, `--dry-*`, now tracked in
`user_sampling_config`) wins over the GGUF's.

Model-independent: a TTS backbone's recommended sampling is the model's, so it
ships in the backbone GGUF; KittenTTS 2's token-run penalty is DRY.

Stacked on `penalty-exempt-llamacpp.patch`; applied in order by
`cmake/SetupLlamaPatches.cmake`.
