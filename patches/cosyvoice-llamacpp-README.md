# CosyVoice arch for llama.cpp

`cosyvoice-llamacpp.patch` adds the llama.cpp arch `cosyvoice`: the speech-token
LM of CosyVoice-300M (`-SFT`, `-Instruct`) and BreezyVoice, a 14-layer WeNet
TransformerEncoder with Espnet relative-position self-attention, run causally
with a KV cache. CosyVoice2/3 use stock Qwen2 and need no patch.

- Stacked on `barbet-llamacpp.patch` (same pinned base, see
  `barbet-llamacpp-README.md`); both are applied in order by
  `cmake/SetupLlamaPatches.cmake` at configure time.
- Input is embeddings only (text-encoder rows, speaker row and speech-token rows
  are built by codec.cpp's CosyVoice adaptor); output is the final hidden state
  — the speech head runs in codec.cpp. A token batch maps to zero rows.
- The relative-position term `(q + v)·W_pos·pe(i - j)` is passed to
  `build_attn` as a KQ bias: with the interleaved Espnet sinusoid it equals
  `rope_i(W_pos,hᵀ(q + v)) · rope_j(c)` for `c = (0, 1, 0, 1, …)`, so stock RoPE
  computes it. Key positions are KV cell indices (one sequence, filled in order).
  KQ bias excludes flash attention (the tts runner disables it for this arch).
- GGUF: `scripts/convert-backbone-to-gguf.py` writes it directly from `llm.pt`
  (`linear_pos` pre-split per head) with the Whisper multilingual vocab as a
  GPT-2 BPE tokenizer.

Validated against the PyTorch TransformerLM: teacher-forced argmax agrees at
every step (CosyVoice-300M 103/103, BreezyVoice 180/180), max |Δ log p| < 3e-3.
