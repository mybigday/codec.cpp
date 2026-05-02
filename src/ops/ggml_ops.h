#ifndef CODEC_OPS_GGML_OPS_H
#define CODEC_OPS_GGML_OPS_H

#include "../codec_internal.h"

enum codec_unary_op {
    CODEC_UNARY_SIGMOID = 0,
    CODEC_UNARY_ELU = 1,
    CODEC_UNARY_SILU = 2,
    CODEC_UNARY_GELU_ERF = 3,
    CODEC_UNARY_MISH = 4,
};

ggml_tensor * codec_op_unary(ggml_context * ctx, ggml_tensor * x, codec_unary_op op);
ggml_tensor * codec_op_layer_norm(ggml_context * ctx, ggml_tensor * x, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_layer_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_layer_norm_tc(ggml_context * ctx, ggml_tensor * x_tc, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_rms_norm_ct(ggml_context * ctx, ggml_tensor * x_ct, float eps, ggml_tensor * gamma);
ggml_tensor * codec_op_group_norm(ggml_context * ctx, ggml_tensor * x, int32_t n_groups, float eps, ggml_tensor * gamma, ggml_tensor * beta);
ggml_tensor * codec_op_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * codec_op_linear_tc(ggml_context * ctx, ggml_tensor * x_tc, ggml_tensor * w, ggml_tensor * b);
ggml_tensor * codec_op_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, float eps);
ggml_tensor * codec_op_snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * inv_beta, float eps);
ggml_tensor * codec_op_pad_1d(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right);
ggml_tensor * codec_op_pad_1d_replicate(ggml_context * ctx, ggml_tensor * x, int32_t pad_left, int32_t pad_right);
ggml_tensor * codec_op_causal_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t target_t);
ggml_tensor * codec_op_crop_1d(ggml_context * ctx, ggml_tensor * x, int32_t crop_left, int32_t crop_right);
ggml_tensor * codec_op_channel_scale(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale);

ggml_tensor * codec_op_tokens_to_features(ggml_context * ctx, ggml_tensor * tokens, int32_t out_channels);

// ConvNeXt block (Vocos-style): residual + (depthwise conv → LayerNorm → linear → GELU → linear → channel-scale).
// `x_ct` is `[c, t]`; biases and `gamma` are optional (pass nullptr to skip).
// `dw_padding` is the symmetric (non-causal) padding for the depthwise conv;
// for causal variants compose your own block.
ggml_tensor * codec_op_convnext_block_ct(
    ggml_context * ctx,
    ggml_tensor * x_ct,
    ggml_tensor * dw_w,
    ggml_tensor * dw_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b,
    ggml_tensor * pw1_w,
    ggml_tensor * pw1_b,
    ggml_tensor * pw2_w,
    ggml_tensor * pw2_b,
    ggml_tensor * gamma,
    int32_t dw_padding);

// Matcha CausalBlock1D: causal conv1d (left-pad k-1) → LayerNorm along channels → Mish.
// `x_tc` is `[t, c]`; the conv weight is loaded directly, kernel is inferred from
// `conv_w->ne[0]`. Used by Matcha-TTS / Chatterbox CFM decoders and any other
// causal diffusion U-Net.
ggml_tensor * codec_op_causal_block1d_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * conv_w,
    ggml_tensor * conv_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b);

// One branch of a HiFi-GAN-style ResBlock: `x + conv1d(snake(conv1d(snake(x))))`
// where the first conv has user-provided dilation and the second conv runs at
// dilation=1. `x_tc` is `[t, c]`; padding is symmetric (`d*(k-1)/2` for the
// dilated conv, `(k-1)/2` for the second). Caller stacks branches with
// distinct dilations to build the full ResBlock.
ggml_tensor * codec_op_hifigan_resblock_branch_ct(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * a1,
    ggml_tensor * a2,
    ggml_tensor * c1_w,
    ggml_tensor * c1_b,
    ggml_tensor * c2_w,
    ggml_tensor * c2_b,
    int32_t kernel_size,
    int32_t dilation);

// Matcha CausalResnetBlock1D forward:
//   h  = causal_block(x)                   (block1: causal-conv → LN_tc → Mish)
//   h += linear(mish(t_emb)).unsqueeze(t)  (broadcast over time)
//   h  = causal_block(h)                   (block2 with the same shape pattern)
//   return h + res_conv1x1(x)
// `t_emb` is `[time_embed_dim]`; `mlp_w` has PyTorch shape `(out_c, time_embed_dim)`
// and `res_w` is a 1×1 conv of shape `(out_c, in_c, 1)`. `x_tc` is `[t, c=in_c]`.
ggml_tensor * codec_op_cfm_causal_resnet_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * t_emb,
    ggml_tensor * b1_conv_w, ggml_tensor * b1_conv_b,
    ggml_tensor * b1_ln_w,   ggml_tensor * b1_ln_b,
    ggml_tensor * b2_conv_w, ggml_tensor * b2_conv_b,
    ggml_tensor * b2_ln_w,   ggml_tensor * b2_ln_b,
    ggml_tensor * mlp_w,     ggml_tensor * mlp_b,
    ggml_tensor * res_w,     ggml_tensor * res_b);

// Diffusers BasicTransformerBlock without cross-attention (standard
// `LayerNorm → self-attn → +res` then `LayerNorm → GELU FFN → +res`). Used by
// any diffusion model that pulls in `diffusers.models.attention.BasicTransformerBlock`.
// `x_tc` is `[t, c]`; norms run along the channel dim. Q/K/V have no bias;
// only `out` and the FFN linears carry biases. `head_dim * num_heads` must
// equal the inner attention dim implied by `qw`'s rows.
ggml_tensor * codec_op_basic_transformer_block_tc(
    ggml_context * ctx,
    ggml_tensor * x_tc,
    ggml_tensor * norm1_w, ggml_tensor * norm1_b,
    ggml_tensor * qw, ggml_tensor * kw, ggml_tensor * vw,
    ggml_tensor * ow, ggml_tensor * ob,
    ggml_tensor * norm3_w, ggml_tensor * norm3_b,
    ggml_tensor * ff1_w, ggml_tensor * ff1_b,
    ggml_tensor * ff2_w, ggml_tensor * ff2_b,
    int32_t head_dim,
    int32_t num_heads);

// Sinusoidal time embedding (Diffusion / flow-matching SinusoidalPosEmb).
// Builds the embedding entirely in-graph from `ggml_arange + sin/cos`:
//   half = dim/2;  freq[k] = exp(-k * log(10000)/(half-1))
//   emb  = concat(sin(scale*t*freq), cos(scale*t*freq))   length = dim.
// `t_v` is a compile-time constant scalar; the graph bakes it into a
// `ggml_scale`. Output is a 1-D tensor of length `dim`.
ggml_tensor * codec_op_sinusoidal_time_emb(
    ggml_context * ctx,
    float t_v,
    int32_t dim,
    float scale);

// Espnet-style relative positional encoding for a Conformer with `T` query
// positions, built entirely in-graph. Output ne = (d_model, 2T-1):
//   row r covers position p_r ∈ [T-1, T-2, ..., 0, -1, ..., -(T-1)]
//   pe[r, 2k]   = sin(p_r * freq[k])
//   pe[r, 2k+1] = cos(p_r * freq[k])
// where freq[k] = exp(-2k * log(10000)/d_model). The interleaved sin/cos
// layout matches the Espnet `pe[:, 0::2] / 1::2` convention so a Linear(d, d)
// (`linear_pos`) trained on that ordering applies directly.
ggml_tensor * codec_op_espnet_rel_pos_emb(
    ggml_context * ctx,
    int32_t t,
    int32_t d_model);

#endif
