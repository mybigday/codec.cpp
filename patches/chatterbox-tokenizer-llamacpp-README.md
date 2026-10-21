# Chatterbox T3 text tokenizer for llama.cpp

`chatterbox-tokenizer-llamacpp.patch` adds `tokenizer.ggml.pre = chatterbox`
(`LLAMA_VOCAB_PRE_TYPE_CHATTERBOX`): the T3 text tokenizers of Chatterbox
(`tokenizer.json`, 704 tokens) and Chatterbox Multilingual
(`grapheme_mtl_merged_expanded_v1.json`, 2454 tokens) are raw-UTF-8 BPEs behind
HF tokenizers' Whitespace pre-tokenizer (`\w+|[^\w\s]+`, the whitespace between
words dropped). Pieces missing from the vocab become `[UNK]`, as in HF.

`scripts/convert-backbone-to-gguf.py` writes the tokenizer into the T3
backbone GGUF (its `token_embd` is T3's `text_emb`, sized to the vocab), so the
host tokenizes the prompt with llama.cpp. Text normalization stays outside the
tokenizer: codec_audio_lm's punc_norm and `' '` -> `[SPACE]` (the prompt is
tokenized with parse_special); `[lang]` tags and language-specific conversions
are the caller's. Matches HF `tokenizers` on 24 multilingual test strings and
the English ones (2026-10-21).

Backbones converted before this carry no vocab (`tokenizer.ggml.model = none`);
codec_audio_lm then falls back to the codec GGUF's copy of the tokenizer
(deprecated).

Stacked on `sampling-meta-llamacpp.patch`; applied in order by
`cmake/SetupLlamaPatches.cmake`.
