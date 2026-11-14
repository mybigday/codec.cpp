# Proposal: the backbone micro-batch follows llama.cpp (host n_ubatch, default 512)

Status: proposal for review (no implementation on this branch). Public headers touched if accepted:
`include/codec_audio_lm.h` (`codec_audio_lm_backbone_req::n_ubatch` meaning) and `common/tts_runner.h`
(`tts_runner_params::n_ubatch`), plus `python/codec_cpp/_ffi.py` if a field changes.

## Motivation

`tts_runner` creates the backbone context with `n_ubatch = req.n_ubatch ?: n_ctx`
(`common/tts_runner.cpp:569`). A micro-batch of the whole context makes llama.cpp reserve its compute buffer for
`n_ctx` tokens at once: **2502 MiB for Qwen3-TTS 0.6B** (n_ctx 4096), 172 MiB for Soprano, on every load, for
prompts that are a few hundred tokens. On a phone that is the difference between fitting next to an ASR model and
not (measured in Live Translate on a Pixel 7 Pro: 2.5 GB reserved vs 0.31 GB at 512).

## Where the model-specified value comes from (evidence)

- `examples/tts.py` (7afe2ab, the first Python host) set `cp.n_ubatch = max(n_ctx, 64)` with no stated reason;
  `tts-cli` copied it, and `3ac2156` moved it into `codec_tts_runner`.
- `f10a3d5` (OuteTTS, 8192-token context) set `cp.n_ubatch = 4096` next to the larger context, as a cap, not a
  requirement; `8ce8eb8` moved both into `codec_audio_lm_backbone_requirements` (`audio_lm_family_token.cpp:269`).
  The base family returns `n_ubatch = 0` (`audio_lm_family.cpp:161`).
- llama.cpp needs `n_ubatch >= n_tokens` only for **non-causal attention** (`src/llama-context.cpp:1750`,
  `GGML_ASSERT(cparams.causal_attn || cparams.n_ubatch >= n_tokens_all)`) and for encoders (`:1405`). Every
  codec.cpp backbone is a causal LM run embd-in / hidden-out; no family calls `llama_set_causal_attn` and no patch
  under `patches/` makes attention non-causal. So no model has a real requirement today.
- Splitting a long prompt only changes float summation order: Soprano with a ~500-token prompt (two micro-batches)
  matches a single micro-batch at 35-46 dB for the first 6 s, then a greedy step flips (as any float change can);
  Qwen3-TTS prompts below 512 tokens are bit-identical.

## Proposal

1. Hosts set the micro-batch like llama.cpp: `tts_runner_params::n_ubatch` (default 512 = llama.cpp's
   `n_ubatch` default), `tts-cli` / `tts-server` `-ub` / `--ubatch-size N` (llama.cpp's flag names).
   `cp.n_ubatch = max(params.n_ubatch, req.n_ubatch)`, never "0 means something".
2. `codec_audio_lm_backbone_req::n_ubatch` becomes a **minimum**: "the prompt must be decoded in one micro-batch of
   at least this many tokens (non-causal backbones); 0 = no requirement". Every family returns 0 today
   (OuteTTS's 4096 goes away). Same field, same type, so the struct layout and the Python mirror do not change;
   only the documented meaning does.

## Alternatives considered

- **A bool** `needs_single_ubatch` (the host then uses `n_ubatch = n_ctx` for that model). Clearer intent, but a
  struct layout change (ABI break for compiled hosts, `_ffi.py` update), and a minimum covers the same case.
- **Keep the field, change only `tts_runner`'s default** (the reverted first version of this PR): leaves the public
  meaning "0 = n_ctx" contradicted by the reference host.
- **Each family sets 512 explicitly**: keeps a model-specified value nobody needs.

## Compatibility

- Hosts that follow today's doc (0 = n_ctx) keep working; they only keep reserving the large buffer.
- Hosts that use `tts_runner` get llama.cpp's default and can override it.
- No model needs a larger micro-batch, so outputs only move by float summation order on prompts > 512 tokens.

## Models that benefit

All backbone models. Compute buffer (CPU, host): Qwen3-TTS 0.6B 2502 → 313 MiB; Soprano 172 → 22 MiB; OuteTTS
(n_ctx 8192) from a 4096-token micro-batch to 512. Peak RSS drops only when the reservation is touched (long
prompts): Soprano ~500-token prompt 932 → 872 MB.
