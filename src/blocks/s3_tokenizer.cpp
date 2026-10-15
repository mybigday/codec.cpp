#include "s3_tokenizer.h"

#include "../ops/audio_frontend.h"
#include "../ops/conv1d.h"
#include "../ops/ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/rope.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr int32_t kHop = 160;        // 10 ms at 16 kHz
constexpr int32_t kTokenHop = 640;   // 4 mel frames per token

// One encoder block: pre-LN self-attention (NeoX RoPE) + FSMN memory on V,
// pre-LN GELU MLP.
ggml_tensor * s3t_block(
        ggml_context * ctx, const codec_model * model, const std::string & base,
        ggml_tensor * x_ct, const codec_s3_tokenizer_params & p) {
    bool missing = false;
    auto W = [&](const char * n) -> ggml_tensor * {
        ggml_tensor * t = codec_graph_weight(ctx, model, base + n);
        if (t == nullptr) missing = true;
        return t;
    };
    ggml_tensor * attn_ln_w = W(".attn_ln.w");
    ggml_tensor * attn_ln_b = W(".attn_ln.b");
    ggml_tensor * q_w = W(".attn.q.w");
    ggml_tensor * q_b = W(".attn.q.b");
    ggml_tensor * k_w = W(".attn.k.w");
    ggml_tensor * v_w = W(".attn.v.w");
    ggml_tensor * v_b = W(".attn.v.b");
    ggml_tensor * o_w = W(".attn.o.w");
    ggml_tensor * o_b = W(".attn.o.b");
    ggml_tensor * fsmn_w = W(".attn.fsmn.w");
    ggml_tensor * mlp_ln_w = W(".mlp_ln.w");
    ggml_tensor * mlp_ln_b = W(".mlp_ln.b");
    ggml_tensor * fc1_w = W(".mlp.fc1.w");
    ggml_tensor * fc1_b = W(".mlp.fc1.b");
    ggml_tensor * fc2_w = W(".mlp.fc2.w");
    ggml_tensor * fc2_b = W(".mlp.fc2.b");
    if (missing) return nullptr;

    const int32_t hidden = (int32_t) x_ct->ne[0];
    const int32_t t = (int32_t) x_ct->ne[1];
    const int32_t head_dim = hidden / p.n_heads;

    ggml_tensor * h_ct = codec_op_layer_norm_ct(ctx, x_ct, 1e-5f, attn_ln_w, attn_ln_b);
    ggml_tensor * q_ct = codec_op_linear(ctx, h_ct, q_w, q_b);
    ggml_tensor * k_ct = codec_op_linear(ctx, h_ct, k_w, nullptr);
    ggml_tensor * v_ct = codec_op_linear(ctx, h_ct, v_w, v_b);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr) return nullptr;

    ggml_tensor * q_dth = ggml_permute(ctx, ggml_reshape_3d(ctx, q_ct, head_dim, p.n_heads, t), 0, 2, 1, 3);
    ggml_tensor * k_dth = ggml_permute(ctx, ggml_reshape_3d(ctx, k_ct, head_dim, p.n_heads, t), 0, 2, 1, 3);
    ggml_tensor * v_dth = ggml_permute(ctx, ggml_reshape_3d(ctx, v_ct, head_dim, p.n_heads, t), 0, 2, 1, 3);
    ggml_tensor * q_rope = codec_op_rope(ctx, q_dth, head_dim, p.rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
    ggml_tensor * k_rope = codec_op_rope(ctx, k_dth, head_dim, p.rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
    if (q_rope == nullptr || k_rope == nullptr) return nullptr;

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx, q_rope, k_rope, v_dth, &attn_p);
    if (attn_ctx == nullptr) return nullptr;
    ggml_tensor * attn_ct = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attn_ctx, 0, 2, 1, 3)), hidden, t);
    ggml_tensor * attn_proj = codec_op_linear(ctx, attn_ct, o_w, o_b);
    if (attn_proj == nullptr) return nullptr;

    ggml_tensor * v_tc = ggml_cont(ctx, ggml_transpose(ctx, v_ct));
    ggml_tensor * fsmn_tc = codec_conv1d_depthwise(ctx, v_tc, fsmn_w, nullptr, 1, 1, p.fsmn_kernel / 2);
    if (fsmn_tc == nullptr) return nullptr;
    ggml_tensor * fsmn_ct = ggml_add(ctx, ggml_cont(ctx, ggml_transpose(ctx, fsmn_tc)), v_ct);
    x_ct = ggml_add(ctx, x_ct, ggml_add(ctx, attn_proj, fsmn_ct));

    ggml_tensor * m_ct = codec_op_layer_norm_ct(ctx, x_ct, 1e-5f, mlp_ln_w, mlp_ln_b);
    ggml_tensor * ff = codec_op_linear(ctx, m_ct, fc1_w, fc1_b);
    if (ff == nullptr) return nullptr;
    ff = codec_op_linear(ctx, ggml_gelu_erf(ctx, ff), fc2_w, fc2_b);
    if (ff == nullptr) return nullptr;
    return ggml_add(ctx, x_ct, ff);
}

// Analysis window: `s3t.window` when the GGUF carries one, else periodic Hann.
bool s3t_window(const codec_model * model, int32_t n_fft, std::vector<float> * out) {
    ggml_tensor * w = codec_model_get_tensor(model, "s3t.window");
    if (w != nullptr) {
        return codec_tensor_as_vec_f32(w, out) && (int32_t) out->size() == n_fft;
    }
    out->resize((size_t) n_fft);
    for (int32_t i = 0; i < n_fft; ++i) {
        (*out)[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * (float) M_PI * (float) i / (float) n_fft);
    }
    return true;
}

}  // namespace

int32_t codec_s3_tokenizer_n_tokens(int64_t n_samples) {
    return (int32_t) ((n_samples + kTokenHop - 1) / kTokenHop);
}

ggml_tensor * codec_block_s3_tokenizer(
        ggml_context * ctx, const codec_model * model, const codec_s3_tokenizer_params & p, ggml_tensor * pcm16k) {
    if (ctx == nullptr || model == nullptr || pcm16k == nullptr || p.n_heads <= 0 || p.hidden % p.n_heads != 0 ||
        p.fsmn_kernel <= 0 || (p.fsmn_kernel % 2) == 0) {
        return nullptr;
    }
    const int32_t t_tok = codec_s3_tokenizer_n_tokens(ggml_nelements(pcm16k));
    std::vector<float> window;
    ggml_tensor * mel_fb = codec_graph_weight(ctx, model, "s3t.mel_filters");
    if (t_tok <= 0 || mel_fb == nullptr || mel_fb->ne[1] != p.n_mels || !s3t_window(model, p.n_fft, &window)) {
        return nullptr;
    }
    ggml_tensor * mel = codec_op_whisper_log_mel(ctx, pcm16k, mel_fb, window, p.n_fft, kHop, kTokenHop, "lut.s3t_stft");
    if (mel == nullptr) return nullptr;                                              // [t_mel, n_mels]

    auto W = [&](const char * n) { return codec_graph_weight(ctx, model, n); };
    ggml_tensor * x_tc = codec_conv1d(ctx, mel, W("s3t.enc.conv1.w"), W("s3t.enc.conv1.b"), 2, 1, 1);
    if (x_tc == nullptr) return nullptr;
    x_tc = codec_conv1d(ctx, ggml_gelu_erf(ctx, x_tc), W("s3t.enc.conv2.w"), W("s3t.enc.conv2.b"), 2, 1, 1);
    if (x_tc == nullptr) return nullptr;
    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, ggml_gelu_erf(ctx, x_tc)));
    for (int32_t li = 0; li < p.n_layers; ++li) {
        x_ct = s3t_block(ctx, model, "s3t.enc.blk." + std::to_string(li), x_ct, p);
        if (x_ct == nullptr) return nullptr;
    }

    // FSQ: round(0.999 * tanh(proj)) + 1 ∈ {0, 1, 2}, base-3 code over 8 dims.
    ggml_tensor * q_ct = codec_op_linear(ctx, x_ct, W("s3t.q.proj.w"), W("s3t.q.proj.b"));
    if (q_ct == nullptr) return nullptr;
    q_ct = codec_op_round_bounded(ctx, ggml_scale(ctx, ggml_tanh(ctx, q_ct), 0.9990000128746033f), 1);
    if (q_ct == nullptr) return nullptr;
    q_ct = ggml_scale_bias(ctx, q_ct, 1.0f, 1.0f);
    const std::vector<float> powers = {1.0f, 3.0f, 9.0f, 27.0f, 81.0f, 243.0f, 729.0f, 2187.0f};
    ggml_tensor * pw = codec_graph_const_f32(ctx, "lut.s3t_fsq_powers", powers, 8);
    if (pw == nullptr) return nullptr;
    ggml_tensor * idx = ggml_sum_rows(ctx, ggml_mul(ctx, q_ct, ggml_reshape_2d(ctx, pw, 8, 1)));
    return ggml_cast(ctx, ggml_reshape_1d(ctx, idx, t_tok), GGML_TYPE_I32);
}
