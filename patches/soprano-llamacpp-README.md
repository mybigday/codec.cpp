# Soprano tokenizer for llama.cpp

`soprano-llamacpp.patch` adds the BPE pre-tokenizer `soprano`
(`tokenizer.ggml.pre = soprano`) for
[ekwek/Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M). The
model itself is stock `qwen3`; only its tokenizer is new:

- normalizer: Unicode lowercase, whitespace runs → one space;
- pre-tokenizer: one token per digit, then `\s+|\w+|[^\w\s]+`;
- raw UTF-8 BPE (no GPT-2 byte encoding); unknown pieces → `[UNK]` (id 0);
- `[STOP]` (id 3) is eos / end-of-generation, no BOS.

Stacked on `cosyvoice-llamacpp.patch` (it only touches `src/llama-vocab.*`);
applied in order by `cmake/SetupLlamaPatches.cmake`.

Validated: token ids identical to `tokenizers.Tokenizer.from_file(tokenizer.json)`
on mixed-case, digit, punctuation, whitespace, CJK and emoji inputs. See
[docs/models/soprano.md](../docs/models/soprano.md).
