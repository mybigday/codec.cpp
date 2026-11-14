#include "qwen3_tts_tokenizer.h"

#include "../ops/audio_frontend.h"
#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/ggml_ops.h"
#include "../ops/rope.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

static constexpr int32_t CODEC_Q3T_RES_UNITS = 3;
static constexpr int32_t CODEC_Q3T_RES_DILATIONS[CODEC_Q3T_RES_UNITS] = { 1, 3, 9 };

enum codec_status codec_qwen3_tts_tokenizer_init(struct codec_model * model) {
    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(model->impl);
    codec_qwen3_tts_tokenizer & q3 = impl.q3;
    codec_mimi & mimi = impl.mimi;

    q3.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    q3.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 1920);
    q3.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", 16);
    q3.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 2048);
    q3.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", 1024);
    q3.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", 1024);
    q3.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", true);
    q3.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    q3.hidden_size = codec_read_i32_kv(model->gguf, "qwen3.decoder.hidden_size", 1024);
    q3.num_hidden_layers = codec_read_i32_kv(model->gguf, "qwen3.decoder.num_hidden_layers", 8);
    q3.num_attention_heads = codec_read_i32_kv(model->gguf, "qwen3.decoder.num_attention_heads", 16);
    q3.num_key_value_heads = codec_read_i32_kv(model->gguf, "qwen3.decoder.num_key_value_heads", q3.num_attention_heads);
    q3.head_dim = codec_read_i32_kv(model->gguf, "qwen3.decoder.head_dim", 64);
    q3.intermediate_size = codec_read_i32_kv(model->gguf, "qwen3.decoder.intermediate_size", 3072);
    q3.rope_theta = codec_read_f32_kv(model->gguf, "qwen3.decoder.rope_theta", 10000.0f);
    q3.sliding_window = codec_read_i32_kv(model->gguf, "qwen3.decoder.sliding_window", 0);
    q3.decoder_dim = codec_read_i32_kv(model->gguf, "qwen3.decoder.decoder_dim", 1536);

    // upsample arrays
    q3.n_upsample_rates = 0;
    q3.n_upsampling_ratios = 0;
    {
        const int key_id = gguf_find_key(model->gguf, "qwen3.decoder.upsample_rates");
        if (key_id >= 0 && gguf_get_kv_type(model->gguf, key_id) == GGUF_TYPE_ARRAY) {
            const enum gguf_type t = gguf_get_arr_type(model->gguf, key_id);
            const size_t n = gguf_get_arr_n(model->gguf, key_id);
            const size_t copy_n = std::min(n, (size_t) CODEC_Q3T_MAX_UPSAMPLE);
            q3.n_upsample_rates = (int32_t) copy_n;
            const void * data = gguf_get_arr_data(model->gguf, key_id);
            for (size_t i = 0; i < copy_n; ++i) {
                if (t == GGUF_TYPE_INT32) {
                    q3.upsample_rates[i] = ((const int32_t *) data)[i];
                } else if (t == GGUF_TYPE_UINT32) {
                    q3.upsample_rates[i] = (int32_t) ((const uint32_t *) data)[i];
                } else {
                    q3.upsample_rates[i] = 0;
                }
            }
        }
    }
    {
        const int key_id = gguf_find_key(model->gguf, "qwen3.decoder.upsampling_ratios");
        if (key_id >= 0 && gguf_get_kv_type(model->gguf, key_id) == GGUF_TYPE_ARRAY) {
            const enum gguf_type t = gguf_get_arr_type(model->gguf, key_id);
            const size_t n = gguf_get_arr_n(model->gguf, key_id);
            const size_t copy_n = std::min(n, (size_t) CODEC_Q3T_MAX_UPSAMPLE);
            q3.n_upsampling_ratios = (int32_t) copy_n;
            const void * data = gguf_get_arr_data(model->gguf, key_id);
            for (size_t i = 0; i < copy_n; ++i) {
                if (t == GGUF_TYPE_INT32) {
                    q3.upsampling_ratios[i] = ((const int32_t *) data)[i];
                } else if (t == GGUF_TYPE_UINT32) {
                    q3.upsampling_ratios[i] = (int32_t) ((const uint32_t *) data)[i];
                } else {
                    q3.upsampling_ratios[i] = 0;
                }
            }
        }
    }

    // Initialize Mimi encoder metadata from qwen3.encoder.* keys
    mimi.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    mimi.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 1920);
    mimi.n_q = codec_read_i32_kv(model->gguf, "qwen3.encoder.n_q", q3.n_q);
    mimi.num_semantic_quantizers = codec_read_i32_kv(model->gguf, "codec.num_semantic_quantizers", 1);
    mimi.codebook_size = codec_read_i32_kv(model->gguf, "qwen3.encoder.codebook_size", q3.codebook_size);
    mimi.codebook_dim = codec_read_i32_kv(model->gguf, "qwen3.encoder.codebook_dim", q3.codebook_dim);
    mimi.hidden_size = codec_read_i32_kv(model->gguf, "qwen3.encoder.hidden_size", 512);
    mimi.num_hidden_layers = codec_read_i32_kv(model->gguf, "qwen3.encoder.num_hidden_layers", 8);
    mimi.num_attention_heads = codec_read_i32_kv(model->gguf, "qwen3.encoder.num_attention_heads", 8);
    mimi.head_dim = codec_read_i32_kv(model->gguf, "qwen3.encoder.head_dim", 64);
    mimi.intermediate_size = codec_read_i32_kv(model->gguf, "qwen3.encoder.intermediate_size", 2048);
    mimi.rope_theta = codec_read_f32_kv(model->gguf, "qwen3.encoder.rope_theta", 10000.0f);
    mimi.rope_scaling_factor = codec_read_f32_kv(model->gguf, "qwen3.encoder.rope_scaling_factor", 1.0f);
    mimi.has_encoder = q3.has_encoder;
    mimi.has_decoder = false;

    model->sample_rate = q3.sample_rate;
    model->has_encoder = q3.has_encoder;
    model->has_decoder = q3.has_decoder;
    model->hop_size = q3.hop_size;
    model->n_q = q3.n_q;
    model->codebook_size = q3.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = q3.latent_dim;

    impl.has_spk_enc = q3.has_encoder && ggml_get_tensor(model->weights, "speaker.qwen3_tts.fc.weight") != nullptr;
    if (impl.has_spk_enc) {
        gguf_context * gf = model->gguf;
        codec_ecapa_params & e = impl.ecapa;
        e.n_mels        = codec_read_i32_kv(gf, "codec.speaker.ecapa.mel_dim", e.n_mels);
        e.res2net_scale = codec_read_i32_kv(gf, "codec.speaker.ecapa.enc_res2net_scale", e.res2net_scale);
        e.n_fft         = codec_read_i32_kv(gf, "codec.speaker.ecapa.n_fft", e.n_fft);
        e.hop           = codec_read_i32_kv(gf, "codec.speaker.ecapa.hop_size", e.hop);
        impl.spk_sample_rate = codec_read_i32_kv(gf, "codec.speaker.ref_sample_rate", impl.spk_sample_rate);
        std::vector<int32_t> channels, kernels, dilations;
        codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_channels", &channels);
        codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_kernel_sizes", &kernels);
        codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_dilations", &dilations);
        if (channels.size() < 3 || channels.size() > (size_t) kCodecEcapaMaxBlocks ||
            channels.size() != kernels.size() || channels.size() != dilations.size() ||
            impl.spk_sample_rate != q3.sample_rate) {
            return CODEC_STATUS_INVALID_ARG;
        }
        e.n_blocks = (int32_t) channels.size();
        for (size_t i = 0; i < channels.size(); ++i) {
            e.channels[i] = channels[i];
            e.kernels[i] = kernels[i];
            e.dilations[i] = dilations[i];
        }
    }
    model->can_speaker_from_audio = impl.has_spk_enc;

    return CODEC_STATUS_SUCCESS;
}

struct q3t_decode_build {
    int32_t t;
    int32_t q;
    int32_t n_sem;
    int32_t codebook_size;
    int32_t codebook_dim;
    int32_t codebook_dim_half;
    int32_t latent_dim;
    int32_t hidden_size;
    int32_t transformer_layers;
    int32_t transformer_heads;
    int32_t transformer_kv_heads;
    int32_t transformer_head_dim;
    int32_t transformer_intermediate;
    float rope_theta;
    int32_t sliding_window;
    int32_t n_upsample_rates;
    int32_t n_upsampling_ratios;
    int32_t upsample_rates[CODEC_Q3T_MAX_UPSAMPLE];
    int32_t upsampling_ratios[CODEC_Q3T_MAX_UPSAMPLE];
    int32_t decoder_dim;
    const codec_model * model;
    // Incremental decode (codec_stream.h): every stateful site reads its state from "q3t.st.<i>.in" and writes the
    // next one to "q3t.st.<i>.out"; attention takes absolute positions "q3t.st.pos" and the window mask
    // "q3t.st.mask" over [kv_len cached frames + t new ones].
    bool stream;
    int32_t kv_len;
    int32_t n_sites;  // counted while building
};

// Stream state site: [state frames ; x] along time (dim 0, x is [t, c]); the last n frames are the next state.
static ggml_tensor * codec_q3t_st_cat(ggml_context * ctx, q3t_decode_build * p, ggml_tensor * x, int32_t n, int dim) {
    const int32_t i = p->n_sites++;
    int64_t ne[4] = { x->ne[0], x->ne[1], x->ne[2], x->ne[3] };
    ne[dim] = n;
    ggml_tensor * in = ggml_new_tensor(ctx, GGML_TYPE_F32, ggml_n_dims(x) > 2 ? 3 : 2, ne);
    ggml_set_name(in, ("q3t.st." + std::to_string(i) + ".in").c_str());
    ggml_set_input(in);
    ggml_tensor * cat = ggml_concat(ctx, in, ggml_cont(ctx, x), dim);
    const size_t off = (size_t) (cat->ne[dim] - n) * cat->nb[dim];
    ggml_tensor * tail = dim == 0 ? ggml_view_3d(ctx, cat, n, cat->ne[1], cat->ne[2], cat->nb[1], cat->nb[2], off)
                                  : ggml_view_3d(ctx, cat, cat->ne[0], n, cat->ne[2], cat->nb[1], cat->nb[2], off);
    tail = ggml_cont(ctx, tail);
    ggml_set_name(tail, ("q3t.st." + std::to_string(i) + ".out").c_str());
    ggml_set_output(tail);
    return cat;
}

// Causal conv (stride 1); in stream mode the left context comes from the state instead of zero padding.
static ggml_tensor * codec_q3t_conv(ggml_context * ctx, q3t_decode_build * p, ggml_tensor * x, ggml_tensor * w,
                                    ggml_tensor * b, int32_t dilation, bool depthwise) {
    if (!p->stream) {
        return depthwise ? codec_conv1d_depthwise_causal(ctx, x, w, b, 1, dilation)
                         : codec_conv1d_causal(ctx, x, w, b, 1, dilation);
    }
    const int32_t k = depthwise ? (int32_t) w->ne[0] : (int32_t) codec_conv1d_kernel_size(w, x->ne[1]);
    const int32_t pad = (k - 1) * dilation;
    if (pad <= 0) return codec_conv1d_causal(ctx, x, w, b, 1, dilation);
    ggml_tensor * cat = codec_q3t_st_cat(ctx, p, x, pad, 0);
    ggml_tensor * y = depthwise ? codec_conv1d_depthwise_causal(ctx, cat, w, b, 1, dilation)
                                : codec_conv1d_causal(ctx, cat, w, b, 1, dilation);
    return y == nullptr ? nullptr : ggml_cont(ctx, codec_op_crop_1d(ctx, y, pad, 0));
}

// Causal ConvTranspose (right-cropped); in stream mode the previous input frames come from the state.
static ggml_tensor * codec_q3t_convtr(ggml_context * ctx, q3t_decode_build * p, ggml_tensor * x, ggml_tensor * w,
                                      ggml_tensor * b, int32_t stride) {
    if (!p->stream) return codec_convtr1d_causal(ctx, x, w, b, stride, 1);
    const int32_t k = (int32_t) codec_convtr1d_kernel_size(w, x->ne[1]);
    const int32_t n_prev = std::max(0, (k + stride - 1) / stride - 1);
    if (n_prev == 0) return codec_convtr1d_causal(ctx, x, w, b, stride, 1);
    ggml_tensor * cat = codec_q3t_st_cat(ctx, p, x, n_prev, 0);
    ggml_tensor * y = codec_convtr1d_causal(ctx, cat, w, b, stride, 1);
    return y == nullptr ? nullptr : ggml_cont(ctx, codec_op_crop_1d(ctx, y, n_prev * stride, 0));
}

// Add bias broadcast over channels; no-op when bias is null. Used for
// attention q/k/v/o biases which are optional in some Qwen3 checkpoints.
static ggml_tensor * codec_q3t_add_bias_ct(ggml_context * ctx_eval, ggml_tensor * x, ggml_tensor * b) {
    if (b == nullptr) {
        return x;
    }
    return ggml_add(ctx_eval, x, ggml_repeat(ctx_eval, ggml_reshape_2d(ctx_eval, b, x->ne[0], 1), x));
}

static ggml_tensor * codec_q3t_convnext_block(
    ggml_context * ctx_eval,
    q3t_decode_build * p,
    ggml_tensor * x_tc,
    ggml_tensor * dw_w,
    ggml_tensor * dw_b,
    ggml_tensor * ln_w,
    ggml_tensor * ln_b,
    ggml_tensor * pw1_w,
    ggml_tensor * pw1_b,
    ggml_tensor * pw2_w,
    ggml_tensor * pw2_b,
    ggml_tensor * gamma) {

    if (ctx_eval == nullptr || x_tc == nullptr) {
        return nullptr;
    }

    ggml_tensor * res = x_tc;
    ggml_tensor * h = codec_q3t_conv(ctx_eval, p, x_tc, dw_w, dw_b, 1, true);
    if (h == nullptr) {
        return nullptr;
    }

    ggml_tensor * h_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, h)); // [c, t]
    h_ct = codec_op_layer_norm_ct(ctx_eval, h_ct, 1e-6f, ln_w, ln_b);
    if (h_ct == nullptr) {
        return nullptr;
    }
    ggml_tensor * pw1 = codec_mul_mat(ctx_eval, pw1_w, h_ct);
    if (pw1_b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx_eval, pw1_b, pw1_w->ne[1], 1);
        pw1 = ggml_add(ctx_eval, pw1, b2);
    }
    pw1 = ggml_gelu_erf(ctx_eval, pw1);
    ggml_tensor * pw2 = codec_mul_mat(ctx_eval, pw2_w, pw1);
    if (pw2_b != nullptr) {
        ggml_tensor * b2 = ggml_reshape_2d(ctx_eval, pw2_b, pw2_w->ne[1], 1);
        pw2 = ggml_add(ctx_eval, pw2, b2);
    }
    pw2 = codec_op_channel_scale(ctx_eval, pw2, gamma);
    ggml_tensor * out_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, pw2)); // [t, c]
    return ggml_add(ctx_eval, res, out_tc);
}

static ggml_tensor * codec_q3t_apply_sliding_mask(
    ggml_context * ctx_eval,
    ggml_tensor * attn_scores,
    int32_t window) {

    if (ctx_eval == nullptr || attn_scores == nullptr) {
        return nullptr;
    }
    attn_scores = ggml_diag_mask_inf_inplace(ctx_eval, attn_scores, 0);
    if (window <= 0) {
        return attn_scores;
    }
    const int32_t n_past = std::max(0, window - 1);
    ggml_tensor * t = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, attn_scores));
    t = ggml_diag_mask_inf_inplace(ctx_eval, t, n_past);
    return ggml_cont(ctx_eval, ggml_transpose(ctx_eval, t));
}

static ggml_tensor * codec_q3t_repeat_kv(
    ggml_context * ctx_eval,
    ggml_tensor * x,
    int32_t n_heads,
    int32_t n_kv) {

    if (n_kv <= 0 || n_heads <= 0 || n_kv == n_heads) {
        return x;
    }
    if (n_heads % n_kv != 0) {
        return nullptr;
    }
    ggml_tensor * target = ggml_new_tensor_3d(ctx_eval, x->type, x->ne[0], x->ne[1], n_heads);
    return ggml_repeat(ctx_eval, x, target);
}

static std::string codec_q3t_decode_codebook_tensor_name(int32_t qi) {
    return "q3t.dec.q.l" + std::to_string(qi) + ".codebook";
}

static std::string codec_q3t_decode_idx_tensor_name(int32_t qi) {
    return "q3t.dec.q" + std::to_string(qi) + ".idx";
}

static std::string codec_q3t_decode_pt_layer_name(int32_t li, const char * suffix) {
    return "q3t.dec.pt.l" + std::to_string(li) + "." + suffix;
}

static bool codec_q3t_build_decode(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    q3t_decode_build * p = static_cast<q3t_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->t <= 0 || p->q <= 0 ||
        p->codebook_size <= 1 || p->codebook_dim <= 0 || p->codebook_dim_half <= 0 ||
        p->latent_dim <= 0 || p->hidden_size <= 0 || p->transformer_layers <= 0 ||
        p->transformer_heads <= 0 || p->transformer_head_dim <= 0 || p->transformer_intermediate <= 0 ||
        p->model == nullptr) {
        return false;
    }
    const int32_t attn_dim = p->transformer_heads * p->transformer_head_dim;
    const int32_t kv_dim = p->transformer_kv_heads * p->transformer_head_dim;

    auto W = [&](const std::string & name) -> ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };
    auto Wopt = [&](const std::string & name) -> ggml_tensor * {
        return codec_graph_weight_or_null(ctx_eval, p->model, name);
    };

    // Codebook lookup per-quantizer
    ggml_tensor * sem_sum = nullptr;
    ggml_tensor * acu_sum = nullptr;
    for (int32_t qi = 0; qi < p->q; ++qi) {
        ggml_tensor * t_idx = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, p->t);
        ggml_set_name(t_idx, codec_q3t_decode_idx_tensor_name(qi).c_str());

        ggml_tensor * t_codebook = W(codec_q3t_decode_codebook_tensor_name(qi));
        if (t_codebook == nullptr) {
            return false;
        }

        ggml_tensor * t_qi = ggml_get_rows(ctx_eval, t_codebook, t_idx); // [cb_dim_half, t]
        if (t_qi == nullptr) {
            return false;
        }
        if (qi == 0) {
            ggml_set_name(t_qi, "q3t.dec.q0");
        } else if (qi == 1) {
            ggml_set_name(t_qi, "q3t.dec.q1");
        }
        if (qi < p->n_sem) {
            sem_sum = sem_sum == nullptr ? t_qi : ggml_add(ctx_eval, sem_sum, t_qi);
        } else {
            acu_sum = acu_sum == nullptr ? t_qi : ggml_add(ctx_eval, acu_sum, t_qi);
        }
    }
    if (sem_sum == nullptr) {
        return false;
    }

    ggml_tensor * t_sem_op_w = W("q3t.dec.q.s.op.w");
    if (t_sem_op_w == nullptr) {
        return false;
    }
    ggml_tensor * sem_ct = ggml_cont(ctx_eval, sem_sum); // [cb_dim_half, t]
    ggml_tensor * sem_out = codec_mul_mat(ctx_eval, t_sem_op_w, sem_ct); // [cb_dim, t]
    sem_out = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, sem_out)); // [t, cb_dim]
    if (sem_out == nullptr) {
        return false;
    }
    ggml_set_name(sem_out, "q3t.dec.sem");

    ggml_tensor * x_tc = sem_out;
    if (acu_sum != nullptr) {
        ggml_tensor * t_acu_op_w = W("q3t.dec.q.a.op.w");
        if (t_acu_op_w == nullptr) {
            return false;
        }
        ggml_tensor * acu_ct = ggml_cont(ctx_eval, acu_sum); // [cb_dim_half, t]
        ggml_tensor * acu_out = codec_mul_mat(ctx_eval, t_acu_op_w, acu_ct); // [cb_dim, t]
        acu_out = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, acu_out)); // [t, cb_dim]
        if (acu_out == nullptr) {
            return false;
        }
        x_tc = ggml_add(ctx_eval, sem_out, acu_out);
    }
    ggml_set_name(x_tc, "q3t.dec.qsum");

    ggml_tensor * t_pre_w = W("q3t.dec.pre.conv.w");
    ggml_tensor * t_pre_b = W("q3t.dec.pre.conv.b");
    if (t_pre_w == nullptr || t_pre_b == nullptr) {
        return false;
    }
    x_tc = codec_q3t_conv(ctx_eval, p, x_tc, t_pre_w, t_pre_b, 1, false);
    if (x_tc == nullptr) {
        return false;
    }
    ggml_set_name(x_tc, "q3t.dec.pre");

    ggml_tensor * x_ct = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_tc)); // [latent_dim, t]

    // pre-transformer input/output projections
    ggml_tensor * t_in_w = W("q3t.dec.pt.in.w");
    ggml_tensor * t_in_b = W("q3t.dec.pt.in.b");
    ggml_tensor * t_out_w = W("q3t.dec.pt.out.w");
    ggml_tensor * t_out_b = W("q3t.dec.pt.out.b");
    if (t_in_w == nullptr || t_in_b == nullptr || t_out_w == nullptr || t_out_b == nullptr) {
        return false;
    }

    x_ct = codec_mul_mat(ctx_eval, t_in_w, x_ct);
    ggml_tensor * in_b2 = ggml_reshape_2d(ctx_eval, t_in_b, t_in_w->ne[1], 1);
    x_ct = ggml_add(ctx_eval, x_ct, in_b2);

    ggml_tensor * t_pos = nullptr;
    ggml_tensor * t_mask = nullptr;
    if (p->stream) {
        t_pos = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, p->t);
        ggml_set_name(t_pos, "q3t.st.pos");
        ggml_set_input(t_pos);
        t_mask = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, p->kv_len + p->t, p->t);
        ggml_set_name(t_mask, "q3t.st.mask");
        ggml_set_input(t_mask);
    }

    for (int32_t li = 0; li < p->transformer_layers; ++li) {
        ggml_tensor * inln_w = W(codec_q3t_decode_pt_layer_name(li, "inln.w"));
        ggml_tensor * paln_w = W(codec_q3t_decode_pt_layer_name(li, "paln.w"));
        ggml_tensor * q_w = W(codec_q3t_decode_pt_layer_name(li, "attn.q.w"));
        ggml_tensor * k_w = W(codec_q3t_decode_pt_layer_name(li, "attn.k.w"));
        ggml_tensor * v_w = W(codec_q3t_decode_pt_layer_name(li, "attn.v.w"));
        ggml_tensor * o_w = W(codec_q3t_decode_pt_layer_name(li, "attn.o.w"));
        ggml_tensor * q_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.q.b"));
        ggml_tensor * k_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.k.b"));
        ggml_tensor * v_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.v.b"));
        ggml_tensor * o_b = Wopt(codec_q3t_decode_pt_layer_name(li, "attn.o.b"));
        ggml_tensor * fc_gate = W(codec_q3t_decode_pt_layer_name(li, "mlp.gate.w"));
        ggml_tensor * fc_up = W(codec_q3t_decode_pt_layer_name(li, "mlp.up.w"));
        ggml_tensor * fc_down = W(codec_q3t_decode_pt_layer_name(li, "mlp.down.w"));
        ggml_tensor * sa_scale = W(codec_q3t_decode_pt_layer_name(li, "sa.scale"));
        ggml_tensor * mlp_scale = W(codec_q3t_decode_pt_layer_name(li, "mlp.scale"));
        if (inln_w == nullptr || paln_w == nullptr || q_w == nullptr || k_w == nullptr ||
            v_w == nullptr || o_w == nullptr || fc_gate == nullptr || fc_up == nullptr ||
            fc_down == nullptr || sa_scale == nullptr || mlp_scale == nullptr) {
            return false;
        }

        ggml_tensor * h = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-5f, inln_w);
        if (h == nullptr) {
            return false;
        }

        ggml_tensor * q = codec_mul_mat(ctx_eval, q_w, h);
        ggml_tensor * k = codec_mul_mat(ctx_eval, k_w, h);
        ggml_tensor * v = codec_mul_mat(ctx_eval, v_w, h);
        if (q == nullptr || k == nullptr || v == nullptr) {
            return false;
        }
        q = codec_q3t_add_bias_ct(ctx_eval, q, q_b);
        k = codec_q3t_add_bias_ct(ctx_eval, k, k_b);
        v = codec_q3t_add_bias_ct(ctx_eval, v, v_b);

        const int64_t t_cur = q->ne[1];
        ggml_tensor * q_dth = ggml_permute(ctx_eval, ggml_reshape_3d(ctx_eval, q, p->transformer_head_dim, p->transformer_heads, t_cur), 0, 2, 1, 3);
        ggml_tensor * k_dth = ggml_permute(ctx_eval, ggml_reshape_3d(ctx_eval, k, p->transformer_head_dim, p->transformer_kv_heads, t_cur), 0, 2, 1, 3);
        ggml_tensor * v_dth = ggml_permute(ctx_eval, ggml_reshape_3d(ctx_eval, v, p->transformer_head_dim, p->transformer_kv_heads, t_cur), 0, 2, 1, 3);

        ggml_tensor * q_rope = nullptr;
        ggml_tensor * k_rope = nullptr;
        if (!p->stream) {
            q_rope = codec_op_rope(ctx_eval, q_dth, p->transformer_head_dim, p->rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
            k_rope = codec_op_rope(ctx_eval, k_dth, p->transformer_head_dim, p->rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
        } else {
            // absolute positions of the new frames (the whole generation's, as in one decode)
            auto rope_at = [&](ggml_tensor * x_dth) {
                ggml_tensor * x_dht = ggml_permute(ctx_eval, x_dth, 0, 2, 1, 3);
                ggml_tensor * y = ggml_rope_ext(ctx_eval, x_dht, t_pos, nullptr, p->transformer_head_dim, CODEC_ROPE_MODE_NEOX,
                                                0, p->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                return ggml_cont(ctx_eval, ggml_permute(ctx_eval, y, 0, 2, 1, 3));
            };
            q_rope = rope_at(q_dth);
            k_rope = rope_at(k_dth);
            // [kv_len cached frames ; new frames]; the last kv_len are the next state
            k_rope = codec_q3t_st_cat(ctx_eval, p, k_rope, p->kv_len, 1);
            v_dth = codec_q3t_st_cat(ctx_eval, p, v_dth, p->kv_len, 1);
        }
        if (q_rope == nullptr || k_rope == nullptr) {
            return false;
        }
        q_rope = codec_q3t_repeat_kv(ctx_eval, q_rope, p->transformer_heads, p->transformer_kv_heads);
        k_rope = codec_q3t_repeat_kv(ctx_eval, k_rope, p->transformer_heads, p->transformer_kv_heads);
        ggml_tensor * v_rep = codec_q3t_repeat_kv(ctx_eval, v_dth, p->transformer_heads, p->transformer_kv_heads);
        if (q_rope == nullptr || k_rope == nullptr || v_rep == nullptr) {
            return false;
        }

        ggml_tensor * attn_scores = codec_mul_mat(ctx_eval, ggml_cont(ctx_eval, k_rope), q_rope); // [t, t, h]
        attn_scores = ggml_scale_inplace(ctx_eval, attn_scores, 1.0f / std::sqrt((float) p->transformer_head_dim));
        if (!p->stream) {
            attn_scores = codec_q3t_apply_sliding_mask(ctx_eval, attn_scores, p->sliding_window);
        } else {
            attn_scores = ggml_add(ctx_eval, attn_scores, t_mask);  // causal + window + unfilled cache slots
        }
        ggml_tensor * attn_probs = ggml_soft_max(ctx_eval, attn_scores);

        ggml_tensor * v_tdh = ggml_permute(ctx_eval, v_rep, 1, 0, 2, 3);
        ggml_tensor * attn_ctx = codec_mul_mat(ctx_eval, ggml_cont(ctx_eval, v_tdh), attn_probs); // [d, t, h]
        ggml_tensor * attn_ct = ggml_reshape_2d(
            ctx_eval,
            ggml_cont(ctx_eval, ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
            attn_dim,
            t_cur);
        ggml_tensor * attn_proj = codec_mul_mat(ctx_eval, o_w, attn_ct);
        attn_proj = codec_q3t_add_bias_ct(ctx_eval, attn_proj, o_b);
        x_ct = ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, attn_proj, sa_scale));

        ggml_tensor * m = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-5f, paln_w);
        ggml_tensor * gate = codec_mul_mat(ctx_eval, fc_gate, m);
        ggml_tensor * up = codec_mul_mat(ctx_eval, fc_up, m);
        gate = ggml_silu(ctx_eval, gate);
        ggml_tensor * prod = ggml_mul(ctx_eval, gate, up);
        ggml_tensor * down = codec_mul_mat(ctx_eval, fc_down, prod);
        x_ct = ggml_add(ctx_eval, x_ct, codec_op_channel_scale(ctx_eval, down, mlp_scale));
    }

    ggml_tensor * t_norm_w = W("q3t.dec.pt.norm.w");
    if (t_norm_w == nullptr) {
        return false;
    }
    x_ct = codec_op_rms_norm_ct(ctx_eval, x_ct, 1e-5f, t_norm_w);

    x_ct = codec_mul_mat(ctx_eval, t_out_w, x_ct);
    ggml_tensor * out_b2 = ggml_reshape_2d(ctx_eval, t_out_b, t_out_w->ne[1], 1);
    x_ct = ggml_add(ctx_eval, x_ct, out_b2);

    x_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, x_ct)); // [t, latent_dim]
    ggml_set_name(x_tc, "q3t.dec.pt");

    // upsampling ratios (convtr + convnext)
    for (int32_t ui = 0; ui < p->n_upsampling_ratios; ++ui) {
        const std::string base = "q3t.dec.up" + std::to_string(ui);
        ggml_tensor * t_up_w = W(base + ".tr.w");
        ggml_tensor * t_up_b = W(base + ".tr.b");
        ggml_tensor * t_dw_w = W(base + ".cnx.dw.w");
        ggml_tensor * t_dw_b = W(base + ".cnx.dw.b");
        ggml_tensor * t_ln_w = W(base + ".cnx.norm.w");
        ggml_tensor * t_ln_b = W(base + ".cnx.norm.b");
        ggml_tensor * t_pw1_w = W(base + ".cnx.pw1.w");
        ggml_tensor * t_pw1_b = W(base + ".cnx.pw1.b");
        ggml_tensor * t_pw2_w = W(base + ".cnx.pw2.w");
        ggml_tensor * t_pw2_b = W(base + ".cnx.pw2.b");
        ggml_tensor * t_gamma = W(base + ".cnx.gamma");
        if (t_up_w == nullptr || t_up_b == nullptr || t_dw_w == nullptr || t_dw_b == nullptr ||
            t_ln_w == nullptr || t_ln_b == nullptr || t_pw1_w == nullptr || t_pw1_b == nullptr ||
            t_pw2_w == nullptr || t_pw2_b == nullptr || t_gamma == nullptr) {
            return false;
        }
        x_tc = codec_q3t_convtr(ctx_eval, p, x_tc, t_up_w, t_up_b, p->upsampling_ratios[ui]);
        x_tc = codec_q3t_convnext_block(ctx_eval, p, x_tc, t_dw_w, t_dw_b, t_ln_w, t_ln_b, t_pw1_w, t_pw1_b, t_pw2_w, t_pw2_b, t_gamma);
        ggml_set_name(x_tc, (base + ".out").c_str());
    }

    // decoder conv stack
    ggml_tensor * t_dec0_w = W("q3t.dec.d0.w");
    ggml_tensor * t_dec0_b = W("q3t.dec.d0.b");
    if (t_dec0_w == nullptr || t_dec0_b == nullptr) {
        return false;
    }
    x_tc = codec_q3t_conv(ctx_eval, p, x_tc, t_dec0_w, t_dec0_b, 1, false);

    int32_t cur_dim = p->decoder_dim;
    for (int32_t bi = 0; bi < p->n_upsample_rates; ++bi) {
        const std::string base = "q3t.dec.b" + std::to_string(bi);
        const int32_t out_dim = p->decoder_dim / (1 << (bi + 1));
        const int32_t up_rate = p->upsample_rates[bi];

        ggml_tensor * t_s0_a = W(base + ".s0.a");
        ggml_tensor * t_s0_b = W(base + ".s0.binv");
        ggml_tensor * t_tr_w = W(base + ".tr.w");
        ggml_tensor * t_tr_b = W(base + ".tr.b");
        if (t_s0_a == nullptr || t_s0_b == nullptr || t_tr_w == nullptr || t_tr_b == nullptr) {
            return false;
        }
        x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_s0_a, t_s0_b, 1e-9f);
        x_tc = codec_q3t_convtr(ctx_eval, p, x_tc, t_tr_w, t_tr_b, up_rate);

        for (int32_t ri = 0; ri < CODEC_Q3T_RES_UNITS; ++ri) {
            const std::string rbase = base + ".r" + std::to_string(ri);
            ggml_tensor * t_s1_a = W(rbase + ".s1.a");
            ggml_tensor * t_s1_b = W(rbase + ".s1.binv");
            ggml_tensor * t_c1_w = W(rbase + ".c1.w");
            ggml_tensor * t_c1_b = W(rbase + ".c1.b");
            ggml_tensor * t_s2_a = W(rbase + ".s2.a");
            ggml_tensor * t_s2_b = W(rbase + ".s2.binv");
            ggml_tensor * t_c2_w = W(rbase + ".c2.w");
            ggml_tensor * t_c2_b = W(rbase + ".c2.b");
            if (t_s1_a == nullptr || t_s1_b == nullptr || t_c1_w == nullptr || t_c1_b == nullptr ||
                t_s2_a == nullptr || t_s2_b == nullptr || t_c2_w == nullptr || t_c2_b == nullptr) {
                return false;
            }

            ggml_tensor * res = x_tc;
            x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_s1_a, t_s1_b, 1e-9f);
            x_tc = codec_q3t_conv(ctx_eval, p, x_tc, t_c1_w, t_c1_b, CODEC_Q3T_RES_DILATIONS[ri], false);
            x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_s2_a, t_s2_b, 1e-9f);
            x_tc = codec_q3t_conv(ctx_eval, p, x_tc, t_c2_w, t_c2_b, 1, false);
            x_tc = ggml_add(ctx_eval, res, x_tc);
        }
        cur_dim = out_dim;
    }
    ggml_set_name(x_tc, "q3t.dec.dec_out");

    ggml_tensor * t_fs_a = W("q3t.dec.final.s.a");
    ggml_tensor * t_fs_b = W("q3t.dec.final.s.binv");
    ggml_tensor * t_out_final_w = W("q3t.dec.final.w");
    ggml_tensor * t_out_final_b = W("q3t.dec.final.b");
    if (t_fs_a == nullptr || t_fs_b == nullptr || t_out_final_w == nullptr || t_out_final_b == nullptr) {
        return false;
    }
    x_tc = codec_op_snake_beta(ctx_eval, x_tc, t_fs_a, t_fs_b, 1e-9f);

    ggml_tensor * t_out = codec_q3t_conv(ctx_eval, p, x_tc, t_out_final_w, t_out_final_b, 1, false);
    if (t_out == nullptr) {
        return false;
    }
    t_out = ggml_clamp(ctx_eval, t_out, -1.0f, 1.0f);
    ggml_set_name(t_out, "q3t.dec.out");
    *out = t_out;
    return true;
}

static bool codec_q3t_init_decode_build(codec_context * ctx, int32_t t, int32_t q, q3t_decode_build * build, std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || t <= 0 || q <= 0) {
        if (err != nullptr) {
            *err = "invalid Qwen3 decode build arguments";
        }
        return false;
    }
    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl);
    codec_qwen3_tts_tokenizer & q3 = impl.q3;
    build->t = t;
    build->q = q;
    build->n_sem = 1;
    build->codebook_size = std::max(2, q3.codebook_size);
    build->codebook_dim = std::max(1, q3.codebook_dim);
    if (build->codebook_dim % 2 != 0) {
        if (err != nullptr) {
            *err = "Qwen3 codebook_dim must be even";
        }
        return false;
    }
    build->codebook_dim_half = build->codebook_dim / 2;
    build->latent_dim = std::max(1, q3.latent_dim);
    build->hidden_size = std::max(1, q3.hidden_size);
    build->transformer_layers = std::max(1, q3.num_hidden_layers);
    build->transformer_heads = std::max(1, q3.num_attention_heads);
    build->transformer_kv_heads = std::max(1, q3.num_key_value_heads);
    build->transformer_head_dim = std::max(1, q3.head_dim);
    build->transformer_intermediate = std::max(1, q3.intermediate_size);
    build->rope_theta = q3.rope_theta;
    build->sliding_window = q3.sliding_window;
    build->decoder_dim = std::max(1, q3.decoder_dim);

    build->n_upsample_rates = std::max(0, std::min(q3.n_upsample_rates, CODEC_Q3T_MAX_UPSAMPLE));
    build->n_upsampling_ratios = std::max(0, std::min(q3.n_upsampling_ratios, CODEC_Q3T_MAX_UPSAMPLE));
    for (int32_t i = 0; i < build->n_upsample_rates; ++i) {
        build->upsample_rates[i] = std::max(1, q3.upsample_rates[i]);
    }
    for (int32_t i = 0; i < build->n_upsampling_ratios; ++i) {
        build->upsampling_ratios[i] = std::max(1, q3.upsampling_ratios[i]);
    }
    build->model = ctx->model;
    return true;
}


enum codec_status codec_qwen3_tts_tokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || tokens == nullptr || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl);
    codec_qwen3_tts_tokenizer & q3 = impl.q3;
    if (!q3.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, q3.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "Qwen3 decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    if (tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid Qwen3 token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    codec_graph_eval_guard eval_guard(ctx);
    q3t_decode_build build = {};
    std::string err;
    if (!codec_q3t_init_decode_build(ctx, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_Q3T_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/build.codebook_dim, /*n_in=*/0, /*latent_dim=*/build.latent_dim },
            codec_q3t_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "q3t.dec.out");
    if (t_out == nullptr) {
        codec_context_set_error(ctx, "cached Qwen3 decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // tokens -> per-quantizer index vectors (column-major layout)
    std::vector<int32_t> tok_i32((size_t) t * (size_t) q, 0);
    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < q; ++qi) {
            int32_t tok = tokens->data[(size_t) ti * (size_t) tokens->n_q + (size_t) qi];
            tok = std::max(0, std::min(build.codebook_size - 1, tok));
            tok_i32[(size_t) qi * (size_t) t + (size_t) ti] = tok;
        }
    }
    for (int32_t qi = 0; qi < q; ++qi) {
        const std::string name = codec_q3t_decode_idx_tensor_name(qi);
        ggml_tensor * t_idx = codec_graph_get_tensor(ctx, entry, name.c_str());
        if (t_idx == nullptr) {
            codec_context_set_error(ctx, "cached Qwen3 decode graph is invalid");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        const size_t offset = (size_t) qi * (size_t) t;
        if (!codec_runtime_write_tensor(t_idx, tok_i32.data() + offset, (size_t) t * sizeof(int32_t), &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    const int32_t n_threads = ctx->n_threads;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_samples = (int32_t) t_out->ne[0];
    std::vector<float> out(n_samples, 0.0f);
    if (!codec_runtime_read_tensor(t_out, out.data(), out.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    float * data = static_cast<float *>(std::malloc(out.size() * sizeof(float)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate output PCM");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::copy(out.begin(), out.end(), data);

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = data;
    out_pcm->n_samples = n_samples;
    out_pcm->sample_rate = q3.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

// ─── Incremental decode (codec_stream.h) ────────────────────────────
// The decoder is strictly causal: left-padded convs, right-cropped ConvTranspose (kernel 2x stride), attention
// with a sliding window and RoPE positions. So the state between pushes is: every conv's last (k-1)*dilation input
// frames, every ConvTranspose's previous input frame, and per attention layer the last sliding_window-1 K (after
// RoPE) and V frames plus the position count.

struct q3t_stream_state {
    int32_t n_q = 0;
    int64_t n_past = 0;
    std::vector<std::vector<float>> st;  // per state site, in build order
};

static void * codec_q3t_stream_new(codec_context * ctx, int32_t n_q) {
    if (ctx == nullptr || ctx->model == nullptr) return nullptr;
    const auto & q3 = static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl)->q3;
    if (!q3.has_decoder || q3.sliding_window <= 0) return nullptr;  // full causal attention: no bounded state
    const int32_t model_n_q = std::max(1, q3.n_q);
    if (n_q < 0 || n_q > model_n_q) return nullptr;
    auto * s = new (std::nothrow) q3t_stream_state();
    if (s != nullptr) s->n_q = n_q == 0 ? model_n_q : n_q;
    return s;
}

static void * codec_q3t_stream_clone(const void * state) {
    return state == nullptr ? nullptr : new (std::nothrow) q3t_stream_state(*static_cast<const q3t_stream_state *>(state));
}

static void codec_q3t_stream_free(void * state) { delete static_cast<q3t_stream_state *>(state); }

static enum codec_status codec_q3t_stream_push(
    codec_context * ctx, void * state, const codec_token_buffer * tokens, codec_pcm_buffer * out_pcm) {
    auto * s = static_cast<q3t_stream_state *>(state);
    if (ctx == nullptr || s == nullptr || tokens == nullptr || out_pcm == nullptr || tokens->data == nullptr ||
        tokens->n_frames <= 0 || tokens->n_q < s->n_q) {
        codec_context_set_error(ctx, "invalid Qwen3 stream push");
        return CODEC_STATUS_INVALID_ARG;
    }
    auto & q3 = static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl)->q3;
    const int32_t t = tokens->n_frames;
    const int32_t q = s->n_q;
    const int32_t kv_len = q3.sliding_window - 1;

    codec_graph_eval_guard eval_guard(ctx);
    q3t_decode_build build = {};
    std::string err;
    if (!codec_q3t_init_decode_build(ctx, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    build.stream = true;
    build.kv_len = kv_len;
    build.n_sites = 0;
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx, { CODEC_GRAPH_Q3T_DECODE_STREAM, t, q, build.codebook_dim, kv_len, build.latent_dim },
            codec_q3t_build_decode, &build, sizeof(build), &entry, &err) ||
        !codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "q3t.dec.out");
    ggml_tensor * t_pos = codec_graph_get_tensor(ctx, entry, "q3t.st.pos");
    ggml_tensor * t_mask = codec_graph_get_tensor(ctx, entry, "q3t.st.mask");
    if (t_out == nullptr || t_pos == nullptr || t_mask == nullptr) {
        codec_context_set_error(ctx, "cached Qwen3 stream graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    // codes
    for (int32_t qi = 0; qi < q; ++qi) {
        std::vector<int32_t> idx((size_t) t);
        for (int32_t ti = 0; ti < t; ++ti) {
            idx[(size_t) ti] = std::max(0, std::min(build.codebook_size - 1, tokens->data[(size_t) ti * tokens->n_q + qi]));
        }
        ggml_tensor * t_idx = codec_graph_get_tensor(ctx, entry, codec_q3t_decode_idx_tensor_name(qi).c_str());
        if (t_idx == nullptr || !codec_runtime_write_tensor(t_idx, idx.data(), idx.size() * sizeof(int32_t), &err)) {
            codec_context_set_error(ctx, err.empty() ? "cached Qwen3 stream graph is invalid" : err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }
    // positions and the window mask over [kv_len cached ; t new]
    std::vector<int32_t> pos((size_t) t);
    for (int32_t i = 0; i < t; ++i) pos[(size_t) i] = (int32_t) (s->n_past + i);
    const int32_t n_k = kv_len + t;
    std::vector<float> mask((size_t) n_k * t);
    for (int32_t i = 0; i < t; ++i) {
        const int64_t aq = s->n_past + i;
        for (int32_t j = 0; j < n_k; ++j) {
            const int64_t ak = s->n_past - kv_len + j;
            const bool ok = ak >= 0 && ak <= aq && aq - ak <= (int64_t) kv_len;
            mask[(size_t) i * n_k + j] = ok ? 0.0f : -INFINITY;
        }
    }
    if (!codec_runtime_write_tensor(t_pos, pos.data(), pos.size() * sizeof(int32_t), &err) ||
        !codec_runtime_write_tensor(t_mask, mask.data(), mask.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    // state in (zeros before the first push = the zero left padding of one decode)
    std::vector<ggml_tensor *> outs;
    for (int32_t i = 0;; ++i) {
        const std::string base = "q3t.st." + std::to_string(i);
        ggml_tensor * tin = codec_graph_get_tensor(ctx, entry, (base + ".in").c_str());
        ggml_tensor * tout = codec_graph_get_tensor(ctx, entry, (base + ".out").c_str());
        if (tin == nullptr) break;
        if (tout == nullptr || ggml_nelements(tin) != ggml_nelements(tout)) {
            codec_context_set_error(ctx, "cached Qwen3 stream graph is invalid");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        if ((size_t) i >= s->st.size()) s->st.emplace_back((size_t) ggml_nelements(tin), 0.0f);
        if (s->st[(size_t) i].size() != (size_t) ggml_nelements(tin) ||
            !codec_runtime_write_tensor(tin, s->st[(size_t) i].data(), s->st[(size_t) i].size() * sizeof(float), &err)) {
            codec_context_set_error(ctx, err.empty() ? "Qwen3 stream state does not match the graph" : err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        outs.push_back(tout);
    }
    if (!codec_graph_compute(ctx, entry, ctx->n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_samples = (int32_t) t_out->ne[0];
    float * data = static_cast<float *>(std::malloc((size_t) n_samples * sizeof(float)));
    if (data == nullptr || !codec_runtime_read_tensor(t_out, data, (size_t) n_samples * sizeof(float), &err)) {
        std::free(data);
        codec_context_set_error(ctx, err.empty() ? "failed to allocate output PCM" : err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    for (size_t i = 0; i < outs.size(); ++i) {
        if (!codec_runtime_read_tensor(outs[i], s->st[i].data(), s->st[i].size() * sizeof(float), &err)) {
            std::free(data);
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }
    s->n_past += t;
    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = data;
    out_pcm->n_samples = n_samples;
    out_pcm->sample_rate = q3.sample_rate;
    out_pcm->n_channels = 1;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_qwen3_tts_tokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || ctx->model->impl == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_qwen3_tts_tokenizer_impl & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl);
    return codec_mimi_encode_with(ctx, &impl.mimi, pcm, out_tokens, params);
}

// create_voice_clone_prompt as one graph: the reference resampled to 24 kHz
// -> the ECAPA-TDNN x-vector (extract_speaker_embedding) and the codec's own
// codes of it (speech_tokenizer.encode, the ICL ref_code).
struct q3t_speaker_build {
    int32_t n_samples = 0;
    int32_t sample_rate = 0;
    int32_t n24 = 0;
    codec_ecapa_params ecapa;
    mimi_encode_build mimi;
    const codec_model * model = nullptr;
};

static bool q3t_build_speaker(ggml_context * ctx, void * user_data, ggml_tensor ** out) {
    const auto * p = static_cast<const q3t_speaker_build *>(user_data);
    if (ctx == nullptr || p == nullptr || p->model == nullptr || out == nullptr) return false;
    ggml_tensor * pcm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p->n_samples);
    ggml_set_name(pcm, "q3t.spk.pcm");
    ggml_tensor * wav24 = codec_op_resample_sinc_hann(ctx, pcm, p->sample_rate, 24000);
    if (wav24 == nullptr || ggml_nelements(wav24) != p->n24) return false;
    ggml_tensor * xv = codec_block_ecapa_tdnn(ctx, p->model, "speaker.qwen3_tts", p->ecapa, wav24);
    ggml_tensor * codes = nullptr;
    if (xv == nullptr ||
        !codec_mimi_build_encode_from(ctx, &p->mimi, ggml_reshape_2d(ctx, ggml_cont(ctx, wav24), p->n24, 1), &codes) ||
        codes == nullptr) {
        return false;
    }
    ggml_set_name(xv, "q3t.spk.xvector");
    ggml_set_name(codes, "q3t.spk.ref_codes");
    ggml_set_output(xv);
    ggml_set_output(codes);
    *out = xv;
    return true;
}

static enum codec_status codec_qwen3_tts_tokenizer_speaker_from_audio(
        codec_context * ctx, const std::vector<float> & pcm, int32_t sample_rate, codec_speaker * out) {
    auto & impl = *static_cast<codec_qwen3_tts_tokenizer_impl *>(ctx->model->impl);
    if (pcm.empty() || sample_rate <= 0 || pcm.size() > (size_t) INT32_MAX) {
        codec_context_set_error(ctx, "invalid reference audio");
        return CODEC_STATUS_INVALID_ARG;
    }
    q3t_speaker_build build;
    build.n_samples = (int32_t) pcm.size();
    build.sample_rate = sample_rate;
    build.n24 = (int32_t) codec_resample_out_len((int64_t) pcm.size(), sample_rate, 24000);
    build.ecapa = impl.ecapa;
    build.model = ctx->model;
    std::string err;
    if (codec_ecapa_n_frames(impl.ecapa, build.n24) < 2 ||
        !codec_mimi_init_encode_build(ctx, &impl.mimi, build.n24, &build.mimi, &err)) {
        codec_context_set_error(ctx, err.empty() ? "reference audio too short for the speaker encoder" : err);
        return CODEC_STATUS_INVALID_ARG;
    }
    build.mimi.model = ctx->model;

    codec_graph_eval_guard guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(ctx, {CODEC_GRAPH_QWEN3_TTS_SPEAKER, build.n_samples, 0, sample_rate, 0, 0},
                                        q3t_build_speaker, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "q3t.spk.pcm");
    ggml_tensor * t_xv = codec_graph_get_tensor(ctx, entry, "q3t.spk.xvector");
    ggml_tensor * t_codes = codec_graph_get_tensor(ctx, entry, "q3t.spk.ref_codes");
    if (t_pcm == nullptr || t_xv == nullptr || t_codes == nullptr) {
        codec_context_set_error(ctx, "invalid speaker graph IO");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<float> xv((size_t) ggml_nelements(t_xv));
    std::vector<int32_t> codes_tq;
    if (!codec_graph_prepare_io(ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_graph_compute(ctx, entry, ctx->n_threads, &err) ||
        !codec_runtime_read_tensor(t_xv, xv.data(), xv.size() * sizeof(float), &err) ||
        !codec_runtime_read_tensor_i32_2d_tq(t_codes, &codes_tq, &err)) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int64_t n_t = t_codes->ne[0], n_q = t_codes->ne[1];
    if (n_q < impl.q3.n_q) {
        codec_context_set_error(ctx, "speaker graph: fewer codebooks than the codec");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<int32_t> ref((size_t) n_t * impl.q3.n_q);
    for (int64_t t = 0; t < n_t; ++t) {
        for (int32_t q = 0; q < impl.q3.n_q; ++q) ref[(size_t) (t * impl.q3.n_q + q)] = codes_tq[(size_t) (t * n_q + q)];
    }
    const int64_t ne_xv[1] = {(int64_t) xv.size()};
    const int64_t ne_rc[2] = {impl.q3.n_q, n_t};
    codec_speaker_add_slot(out, "xvector", GGML_TYPE_F32, ne_xv, 1, xv.data());
    codec_speaker_add_slot(out, "ref_codes", GGML_TYPE_I32, ne_rc, 2, ref.data());
    return CODEC_STATUS_SUCCESS;
}

static void * codec_qwen3_tts_tokenizer_create_impl() {
    return new (std::nothrow) codec_qwen3_tts_tokenizer_impl();
}

static void codec_qwen3_tts_tokenizer_destroy_impl(void * ptr) {
    delete static_cast<codec_qwen3_tts_tokenizer_impl *>(ptr);
}

static enum codec_status codec_qwen3_tts_tokenizer_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * /*out_latent*/,
    struct codec_encode_params params) {
    return codec_qwen3_tts_tokenizer_encode(ctx, pcm, out_tokens, params);
}

static enum codec_status codec_qwen3_tts_tokenizer_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_qwen3_tts_tokenizer_decode(ctx, tokens, out_pcm, params);
}

// Qwen3-TTS voices condition the talker prompt. CustomVoice builtin voices:
// `spk_id` I32 [1] is the codec-vocab row the talker embeds as its speaker
// row, `language_id` I32 [1] (dialect voices) the language tag used when the
// language is auto. Base voices (cloned): `xvector` F32 [hidden] is the
// speaker row, and `ref_codes` I32 [n_q, T] (the reference's codes) with the
// speaker's ref_text select the in-context prompt (upstream's ICL mode).
static bool codec_qwen3_tts_tokenizer_speaker_validate(
        const codec_model * model, const codec_speaker * spk, std::string * err) {
    if (spk->slot("spk_id") == nullptr && spk->slot("xvector") != nullptr) {
        const auto & impl = *static_cast<const codec_qwen3_tts_tokenizer_impl *>(model->impl);
        const codec_speaker_slot * xv = spk->slot("xvector");
        if (xv->type != GGML_TYPE_F32 || xv->ne[1] != 1 || xv->ne[0] < 1) {
            *err = "Qwen3-TTS xvector slot must be F32 [hidden]";
            return false;
        }
        const codec_speaker_slot * rc = spk->slot("ref_codes");
        if (rc != nullptr) {
            const auto * v = reinterpret_cast<const int32_t *>(rc->data.data());
            bool ok = rc->type == GGML_TYPE_I32 && rc->ne[0] == impl.q3.n_q && rc->ne[1] >= 1 && rc->ne[2] == 1;
            for (int64_t i = 0; ok && i < rc->nelements(); ++i) ok = v[i] >= 0 && v[i] < impl.q3.codebook_size;
            if (!ok) {
                *err = "Qwen3-TTS ref_codes slot must be I32 [" + std::to_string(impl.q3.n_q) + ", T] codes";
                return false;
            }
        }
        return true;
    }
    auto tag_ok = [&](const char * name, bool required) {
        const codec_speaker_slot * s = spk->slot(name);
        if (s == nullptr && !required) return true;
        if (s != nullptr && s->type == GGML_TYPE_I32 && s->nelements() == 1 &&
            *reinterpret_cast<const int32_t *>(s->data.data()) >= 0) {
            return true;
        }
        *err = std::string("Qwen3-TTS speaker needs an I32 [1] ") + name + " slot";
        return false;
    };
    return tag_ok("spk_id", true) && tag_ok("language_id", false);
}

const struct codec_model_vtable * codec_qwen3_tts_tokenizer_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_QWEN3_TTS_TOKENIZER,
        "Qwen3-TTS-Tokenizer",
        codec_qwen3_tts_tokenizer_create_impl,
        codec_qwen3_tts_tokenizer_destroy_impl,
        codec_qwen3_tts_tokenizer_init,
        codec_graph_size_exact,
        codec_qwen3_tts_tokenizer_encode_wrap,
        codec_qwen3_tts_tokenizer_decode_wrap,
        nullptr,
        codec_qwen3_tts_tokenizer_speaker_validate,
        codec_qwen3_tts_tokenizer_speaker_from_audio,
        codec_q3t_stream_new,
        codec_q3t_stream_clone,
        codec_q3t_stream_free,
        codec_q3t_stream_push,
    };
    return &vtable;
}
