# Barbet arch for llama.cpp

Barbet = Open-Formosa R2 text-semantic LM (BlueMagpie-TTS backbone): a Mamba2 +
attention **hybrid**, motif `global, sliding, sliding, mamba2` over 28 layers.
Runs in llama.cpp (arch `barbet`); the acoustic adaptor + AudioVAE run in codec.cpp.

Clone base: `third-party/llama.cpp` @ f708a5b.

## Status

| Piece | State |
|---|---|
| **Converter** `convert_barbet_to_gguf.py` | ✅ DONE + validated (324 tensors, all fused Mamba2 shapes match `build_mamba2_layer`). In `barbet-llamacpp.patch`. |
| C++ arch (7 files below) | SPEC — mechanical, modeled on `granite-hybrid` + `qwen3` (qk_norm) + `gemma3` (SWA). |

## Converter — the Barbet-specific crux (done)

Megatron Mamba2 stores 5 in-projs + 3 convs; fused to llama.cpp's single tensors:
- `ssm_in = concat[in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt]` → `(n_embd, 6424)`, order `[z,x,B,C,dt]`.
- `ssm_conv1d = concat[conv_x, conv_b, conv_c].squeeze(1)` → `(channels=3328, d_conv=4)`.
- `ssm_a = -exp(A_log)` `(24,1)`; `ssm_d = D` `(24,1)`; `ssm_dt.bias = dt_bias`.
- `ssm_norm = norm.reshape(n_group=2, group_size=1536)`; `ssm_out = out_proj` `(d_inner, n_embd)`.
- Attention: `attn_{q,k,v,output}` + per-head `attn_{q,k}_norm` (qk_norm). MLP: SwiGLU. Tied embeddings (no `output.weight`); MTP dropped.

Dims: n_embd 1536, 28 layers, head_dim 128, n_head 16, n_head_kv 2, n_ff 5120,
d_inner 3072, n_group 2, d_state 64, n_ssm_head 24, d_conv 4, rope_theta 1e7,
sliding_window 8192, rms_eps 1e-6.

## C++ arch — remaining (mechanical)

Model on `src/models/granite-hybrid.cpp` (per layer: norm → mixer(mamba2|attn) → ffn),
adding the two Barbet specifics:

1. **gguf-py/gguf/constants.py** — `MODEL_ARCH.BARBET` + tensor list (token_embd,
   output_norm, attn_norm, attn_{q,k,v,output}, attn_{q,k}_norm, ffn_norm,
   ffn_{gate,up,down}, ssm_{in,conv1d,dt,a,d,norm,out}).
2. **src/llama-arch.h / .cpp** — `LLM_ARCH_BARBET` enum, name "barbet", the
   `LLM_TENSOR_*` → name map, KV keys (reuse SSM_* + ATTENTION_SLIDING_WINDOW;
   read `barbet.attention.global_layers` / `barbet.mamba_layers` / `barbet.attention.qk_norm`).
3. **src/llama-hparams.h** — already has `is_recr_impl[]`, `n_head_kv` per layer,
   `swa`/`swa_layers`. Mark mamba layers recurrent; mark non-global attn layers SWA.
4. **src/llama-model.h** — `struct llama_model_barbet : llama_model` with a nested
   `graph : llm_build_mamba_base` (so it gets `build_mamba2_layer`).
5. **src/models/barbet.cpp** — `load_arch_hparams` (read schedule, set is_recr +
   swa per layer, qk_norm), `load_arch_tensors` (per layer: attn_norm + ffn_norm +
   SwiGLU; if mamba → ssm tensors per nemotron-h; else → q/k/v/o + q/k_norm),
   `graph` (granite-hybrid loop; in `build_attention_layer` apply per-head RMSNorm to
   Q/K before RoPE — see `qwen3.cpp`; SWA mask comes from the per-layer swa flag;
   RoPE θ=1e7, NEOX type). qk_logit_softcap / attention_sink are OFF in this ckpt → skip.
6. **src/models/models.h** — declare `llama_model_barbet`.
7. **src/llama-model.cpp** — dispatch `LLM_ARCH_BARBET` in the model factory +
   `LLM_TYPE` (~1.6B).

### Integration note (TTS use)
The codec.cpp adaptor consumes Barbet's **hidden states** (`res->t_embd`), not logits
— run with embeddings enabled; feed inputs via `llama_batch.embd` (LocEnc feedback)
and text-token rows via the tied `token_embd`. No sampling on the audio path.

No new ggml kernels needed: `ggml_ssm_conv`/`ggml_ssm_scan` (Mamba2 variant) + the
existing attention/RoPE/RMSNorm/SwiGLU primitives cover everything.
