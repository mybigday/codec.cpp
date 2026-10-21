# SentencePiece byte fallback in llama.cpp's UGM tokenizer

`ugm-byte-fallback-llamacpp.patch`: llama.cpp's unigram (UGM, `tokenizer.ggml.model = t5`)
tokenizer maps a code point no piece covers to `<unk>`. SentencePiece models
trained with `byte_fallback` (Pocket-TTS's text tokenizer) spell it as its
UTF-8 bytes (`<0xXX>` pieces) instead. With byte pieces in the vocab, the
patched backtrack emits those bytes for each unknown span; vocabularies without
them keep the merged `<unk>`. Matches SentencePiece on the Pocket-TTS test
strings (accents, emoji, CJK; 2026-10-21).

Stacked on `chatterbox-tokenizer-llamacpp.patch`; applied in order by
`cmake/SetupLlamaPatches.cmake`.
