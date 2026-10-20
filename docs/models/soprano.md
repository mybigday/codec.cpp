# Soprano

[ekwek/Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M): an 80M
Qwen3 LM whose final-norm hidden states are the decoder input of a Vocos
decoder (32 kHz). No audio codes: the LM samples `[0]`..`[7999]` tokens only
to advance, the decoder reads the hidden state that predicted each one.

## Conversion

```bash
# decoder (decoder.pth) → codec GGUF
python scripts/convert-to-gguf.py --checkpoint-path models/soprano/decoder.pth \
    --model-type soprano --output models/soprano/soprano.gguf

# LM → backbone GGUF, tokenizer.ggml.pre = soprano
python scripts/convert-backbone-to-gguf.py --model-id models/soprano \
    --output models/soprano/backbone-f16.gguf --outtype f16
```

The tokenizer (lowercase + whitespace collapse, one token per digit, raw UTF-8
BPE with `[UNK]` fallback) is not a stock llama.cpp pre-tokenizer;
`patches/soprano-llamacpp.patch` adds it as `soprano` (see
`patches/soprano-llamacpp-README.md`). A GGUF tagged `gpt-2` loads but drops
every space and never lowercases; retag it with
`gguf-py/gguf/scripts/gguf_new_metadata.py IN OUT --pre-tokenizer soprano`
(weights unchanged).

## Runtime

```bash
tts-cli synthesize --backbone backbone-f16.gguf --model soprano.gguf \
    --text "The birch canoe slid on the smooth planks." --output out.wav
```

`common/tts_runner.cpp` (`run_soprano`) follows `soprano/tts.py` with the
transformers backend:

- prompt `[STOP][TEXT]{text}[START]`, generation until `[STOP]` or 512 tokens;
- greedy (upstream temperature 0), top-p 0.95, repetition penalty 1.2 over
  the prompt and the generated ids; sampling is masked to `[0]`..`[7999]` +
  `[STOP]`;
- decoder input: the hidden state at the position that predicts each
  non-`[STOP]` token (the first is the prompt's last position), frame-major
  `[n, 512]`; the decoder output, `(n-1)·2048` samples, is the whole result.

Not ported: upstream's text cleaning (numbers, abbreviations, `unidecode`) and
sentence splitting; pass normalized text, a sentence or two per call.

## Validation

- `tests/e2e/ttscli_synthesize_smoke.py` case `soprano` (Whisper CER).
- F16 backbone + F32 codec vs `transformers` F32 greedy: same frame count,
  waveform corr 0.999995. Every published backbone quant (F16 … Q2_K) and
  codec quant transcribes correctly; Vulkan matches.
