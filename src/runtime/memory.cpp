#include "memory.h"

#include "graph.h"
#include "tensor_utils.h"
#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), for the model fingerprint in the state blob.

namespace {

struct sha256_ctx {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint8_t buf[64] = {};
    size_t n = 0;
    uint64_t len = 0;
};

constexpr uint32_t k_sha256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256_block(sha256_ctx & c, const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3], e = c.h[4], f = c.h[5], g = c.h[6], h = c.h[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k_sha256[i] + w[i];
        const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
        h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c.h[0] += a; c.h[1] += b; c.h[2] += cc; c.h[3] += d; c.h[4] += e; c.h[5] += f; c.h[6] += g; c.h[7] += h;
}

void sha256_update(sha256_ctx & c, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    c.len += n;
    while (n > 0) {
        const size_t k = std::min(n, (size_t) 64 - c.n);
        std::memcpy(c.buf + c.n, p, k);
        c.n += k; p += k; n -= k;
        if (c.n == 64) { sha256_block(c, c.buf); c.n = 0; }
    }
}

void sha256_final(sha256_ctx & c, uint8_t out[32]) {
    const uint64_t bits = c.len * 8;
    const uint8_t one = 0x80, zero = 0;
    sha256_update(c, &one, 1);
    while (c.n != 56) sha256_update(c, &zero, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; ++i) lb[i] = (uint8_t) (bits >> (56 - 8 * i));
    sha256_update(c, lb, 8);
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t) (c.h[i] >> 24); out[4 * i + 1] = (uint8_t) (c.h[i] >> 16);
        out[4 * i + 2] = (uint8_t) (c.h[i] >> 8); out[4 * i + 3] = (uint8_t) c.h[i];
    }
}

// Tensor names, types and shapes of the model's weights, plus the arch.
void codec_memory_fingerprint(const codec_model * model, uint8_t out[32]) {
    sha256_ctx c;
    const int32_t arch = (int32_t) model->arch;
    sha256_update(c, &arch, sizeof(arch));
    if (model->weights != nullptr) {
        for (ggml_tensor * t = ggml_get_first_tensor(model->weights); t != nullptr; t = ggml_get_next_tensor(model->weights, t)) {
            sha256_update(c, t->name, std::strlen(t->name) + 1);
            const int32_t type = (int32_t) t->type;
            sha256_update(c, &type, sizeof(type));
            sha256_update(c, t->ne, sizeof(t->ne));
        }
    }
    sha256_final(c, out);
}

constexpr char     kStateMagic[4] = {'C', 'D', 'C', 'M'};
constexpr uint32_t kStateFormatVersion = 1;
constexpr uint32_t kStateImplVersion = 1;

std::string slot_name(int32_t side, size_t i) {
    return std::string(side == CODEC_MEMORY_ENCODER ? "mem.enc.s" : "mem.dec.s") + std::to_string(i);
}

}  // namespace

// ---------------------------------------------------------------------------
// Lifecycle

bool codec_memory_create(codec_context * ctx, int32_t side, codec_memory ** out, std::string * err) {
    *out = nullptr;
    codec_memory * mem = new codec_memory();
    mem->ctx = ctx;
    mem->side = side;
    codec_model * model = ctx->model;
    const bool causal = side == CODEC_MEMORY_ENCODER ? codec_model_encoder_is_causal(model) : codec_model_decoder_is_causal(model);
    const bool has_side = side == CODEC_MEMORY_ENCODER ? model->has_encoder : model->has_decoder;
    if (causal && has_side && model->vtable != nullptr && model->vtable->memory_plan != nullptr) {
        mem->planning = true;
        const bool ok = model->vtable->memory_plan(ctx, side, mem, err);
        mem->planning = false;
        if (!ok) {
            delete mem;
            return false;
        }
    }
    if (!mem->specs.empty()) {
        ggml_init_params p = { ggml_tensor_overhead() * (mem->specs.size() + 1), nullptr, true };
        mem->tctx = ggml_init(p);
        if (mem->tctx == nullptr) {
            if (err != nullptr) *err = "failed to create the memory context";
            delete mem;
            return false;
        }
        for (size_t i = 0; i < mem->specs.size(); ++i) {
            const codec_memory_slot_spec & s = mem->specs[i];
            ggml_tensor * t = ggml_new_tensor_4d(mem->tctx, s.type, s.ne[0], s.ne[1], s.ne[2], s.ne[3]);
            ggml_set_name(t, slot_name(side, i).c_str());
            mem->slots.push_back(t);
        }
        mem->buf = ggml_backend_alloc_ctx_tensors(mem->tctx, ctx->backend);
        if (mem->buf == nullptr) {
            if (err != nullptr) *err = "failed to allocate the memory buffer";
            codec_memory_free(mem);
            return false;
        }
    }
    codec_memory_clear(mem);
    *out = mem;
    return true;
}

void codec_memory_free(codec_memory * mem) {
    if (mem == nullptr) return;
    if (mem->buf != nullptr) ggml_backend_buffer_free(mem->buf);
    if (mem->tctx != nullptr) ggml_free(mem->tctx);
    delete mem;
}

codec_memory * codec_memory_active(codec_context * ctx, int32_t side) {
    if (ctx == nullptr || ctx->memory_bypass) return nullptr;
    codec_memory * mem = side == CODEC_MEMORY_ENCODER ? ctx->enc_mem : ctx->dec_mem;
    return mem != nullptr && !mem->slots.empty() ? mem : nullptr;
}

bool codec_memory_plan_build(codec_context * ctx, codec_memory * mem, codec_graph_build_fn build_fn, void * user_data,
                             std::string * err) {
    mem->specs.clear();
    mem->clocks.clear();
    mem->planning = true;
    const bool ok = codec_graph_build_scratch(ctx, build_fn, user_data, err);
    mem->next_slot = mem->next_clock = 0;
    return ok;
}

// ---------------------------------------------------------------------------
// Build-time helpers

void codec_memory_begin_build(codec_memory * mem) {
    if (mem == nullptr) return;
    mem->next_slot = 0;
    mem->next_clock = 0;
    if (mem->planning) {
        mem->specs.clear();
        mem->clocks.clear();
    }
}

ggml_tensor * codec_memory_slot(ggml_context * ctx_eval, codec_memory * mem, uint32_t kind, ggml_type type,
                                int64_t ne0, int64_t ne1, int64_t ne2) {
    if (mem == nullptr || ctx_eval == nullptr) return nullptr;
    if (mem->planning) {
        codec_memory_slot_spec s;
        s.kind = kind;
        s.type = type;
        s.ne[0] = ne0; s.ne[1] = ne1; s.ne[2] = ne2;
        mem->specs.push_back(s);
        mem->next_slot++;
        return ggml_new_tensor_3d(ctx_eval, type, ne0, ne1, ne2);
    }
    if (mem->next_slot >= mem->slots.size()) return nullptr;
    ggml_tensor * t = mem->slots[mem->next_slot];
    const codec_memory_slot_spec & s = mem->specs[mem->next_slot];
    mem->next_slot++;
    if (s.kind != kind || t->type != type || t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != ne2) {
        return nullptr;   // graph structure differs from the plan
    }
    return t;
}

void codec_memory_store(ggml_context * ctx_eval, codec_memory * mem, ggml_tensor * src, ggml_tensor * slot) {
    if (mem == nullptr || mem->planning || src == nullptr || slot == nullptr) return;
    ggml_tensor * c = ggml_cpy(ctx_eval, src, slot);
    ggml_set_output(c);
}

bool codec_memory_clock_inputs(ggml_context * ctx_eval, codec_memory * mem, int32_t cache_len, int32_t window, int64_t t,
                               ggml_tensor ** pos, ggml_tensor ** mask) {
    if (mem == nullptr || ctx_eval == nullptr || t <= 0 || cache_len < 0) return false;
    const size_t id = mem->next_clock++;
    if (mem->planning) {
        codec_memory_clock c;
        c.cache_len = cache_len;
        c.window = window;
        mem->clocks.push_back(c);
    } else if (id >= mem->clocks.size() || mem->clocks[id].cache_len != cache_len || mem->clocks[id].window != window) {
        return false;
    }
    const std::string base = "mem.clk" + std::to_string(id);
    *pos = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, t);
    ggml_set_name(*pos, (base + ".pos").c_str());
    *mask = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, cache_len + t, t);
    ggml_set_name(*mask, (base + ".mask").c_str());
    return true;
}

ggml_tensor * codec_memory_fresh_input(ggml_context * ctx_eval, codec_memory * mem) {
    if (mem == nullptr) return nullptr;
    ggml_tensor * t = ggml_get_tensor(ctx_eval, "mem.fresh");
    if (t == nullptr) {
        t = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 1);
        ggml_set_name(t, "mem.fresh");
    }
    return t;
}

// ---------------------------------------------------------------------------
// Per call

bool codec_memory_set_inputs(codec_context * ctx, codec_graph_cache_entry * entry, codec_memory * mem, std::string * err) {
    if (mem == nullptr) return true;
    for (size_t id = 0; id < mem->clocks.size(); ++id) {
        codec_memory_clock & c = mem->clocks[id];
        const std::string base = "mem.clk" + std::to_string(id);
        ggml_tensor * tp = codec_graph_get_tensor(ctx, entry, (base + ".pos").c_str());
        ggml_tensor * tm = codec_graph_get_tensor(ctx, entry, (base + ".mask").c_str());
        if (tp == nullptr || tm == nullptr) {
            c.t_call = 0;
            continue;
        }
        const int64_t t = tp->ne[0];
        const int64_t C = c.cache_len;
        c.t_call = (int32_t) t;
        std::vector<int32_t> pos((size_t) t);
        for (int64_t i = 0; i < t; ++i) pos[(size_t) i] = (int32_t) (c.n + i);
        std::vector<float> m((size_t) ((C + t) * t));
        const int64_t W = c.window;
        for (int64_t i = 0; i < t; ++i) {
            const int64_t p = c.n + i;
            for (int64_t j = 0; j < C + t; ++j) {
                const int64_t kp = j < C ? c.n - C + j : c.n + (j - C);
                const bool ok = kp >= 0 && kp <= p && (W <= 0 || kp >= p - W + 1);
                m[(size_t) (i * (C + t) + j)] = ok ? 0.0f : -INFINITY;
            }
        }
        if (!codec_runtime_write_tensor(tp, pos.data(), pos.size() * sizeof(int32_t), err) ||
            !codec_runtime_write_tensor(tm, m.data(), m.size() * sizeof(float), err)) {
            return false;
        }
    }
    if (ggml_tensor * tf = codec_graph_get_tensor(ctx, entry, "mem.fresh")) {
        const float f = mem->fresh ? 1.0f : 0.0f;
        if (!codec_runtime_write_tensor(tf, &f, sizeof(f), err)) return false;
    }
    return true;
}

void codec_memory_commit(codec_memory * mem, int64_t n_frames) {
    if (mem == nullptr) return;
    for (codec_memory_clock & c : mem->clocks) {
        c.n += c.t_call;
        c.t_call = 0;
    }
    mem->fresh = false;
    mem->position += n_frames;
}

// ---------------------------------------------------------------------------
// Stateful ops

static ggml_tensor * codec_memory_conv_common(ggml_context * ctx, codec_memory * mem, ggml_tensor * x, ggml_tensor * w,
                                              ggml_tensor * b, int32_t stride, int32_t dilation, bool depthwise,
                                              bool replicate) {
    const int32_t pad_left = codec_conv1d_causal_pad_left(w, x->ne[1], stride, dilation, depthwise);
    if (pad_left < 0) return nullptr;
    ggml_tensor * prev = nullptr;
    ggml_tensor * slot = nullptr;
    if (pad_left > 0) {
        slot = codec_memory_slot(ctx, mem, CODEC_MEMORY_SLOT_CONV_TAIL, GGML_TYPE_F32, pad_left, x->ne[1]);
        if (slot == nullptr) return nullptr;
        prev = slot;
        if (replicate) {
            // First call after a clear: the batch op's replicate padding of the first value.
            ggml_tensor * f = codec_memory_fresh_input(ctx, mem);
            ggml_tensor * first = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], 0);
            ggml_tensor * rep = ggml_repeat(ctx, first, slot);
            prev = ggml_add(ctx, ggml_mul(ctx, slot, ggml_scale_bias(ctx, f, -1.0f, 1.0f)), ggml_mul(ctx, rep, f));
        }
    }
    ggml_tensor * x_cat = nullptr;
    ggml_tensor * y = codec_conv1d_causal_ctx(ctx, x, prev, w, b, stride, dilation, depthwise, &x_cat, replicate);
    if (y == nullptr) return nullptr;
    if (pad_left > 0) {
        ggml_tensor * tail = ggml_view_2d(ctx, x_cat, pad_left, x_cat->ne[1], x_cat->nb[1],
                                          (size_t) (x_cat->ne[0] - pad_left) * x_cat->nb[0]);
        codec_memory_store(ctx, mem, tail, slot);
    }
    return y;
}

ggml_tensor * codec_memory_conv1d_causal(ggml_context * ctx, codec_memory * mem, ggml_tensor * x, ggml_tensor * w,
                                         ggml_tensor * b, int32_t stride, int32_t dilation, bool replicate) {
    if (mem == nullptr) {
        return replicate ? codec_conv1d_causal_replicate(ctx, x, w, b, stride, dilation)
                         : codec_conv1d_causal(ctx, x, w, b, stride, dilation);
    }
    return codec_memory_conv_common(ctx, mem, x, w, b, stride, dilation, false, replicate);
}

ggml_tensor * codec_memory_conv1d_depthwise_causal(ggml_context * ctx, codec_memory * mem, ggml_tensor * x, ggml_tensor * w,
                                                   ggml_tensor * b, int32_t stride, int32_t dilation) {
    if (mem == nullptr) {
        return codec_conv1d_depthwise_causal(ctx, x, w, b, stride, dilation);
    }
    return codec_memory_conv_common(ctx, mem, x, w, b, stride, dilation, true, false);
}

ggml_tensor * codec_memory_convtr1d_causal(ggml_context * ctx, codec_memory * mem, ggml_tensor * x, ggml_tensor * w,
                                           ggml_tensor * b, int32_t stride) {
    if (mem == nullptr) {
        return codec_convtr1d_causal(ctx, x, w, b, stride, 1);
    }
    const int32_t o = codec_convtr1d_causal_overlap(w, x->ne[1], stride);
    if (o < 0) return nullptr;
    ggml_tensor * slot = nullptr;
    if (o > 0) {
        slot = codec_memory_slot(ctx, mem, CODEC_MEMORY_SLOT_CONVTR_OVERLAP, GGML_TYPE_F32, o, w->ne[1]);
        if (slot == nullptr) return nullptr;
    }
    ggml_tensor * next = nullptr;
    ggml_tensor * y = codec_convtr1d_causal_ctx(ctx, x, slot, w, b, stride, &next);
    if (y == nullptr) return nullptr;
    if (o > 0) {
        codec_memory_store(ctx, mem, next, slot);
    }
    return y;
}

ggml_tensor * codec_memory_attn(ggml_context * ctx, codec_memory * mem, ggml_tensor * q_dht, ggml_tensor * k_dht,
                                ggml_tensor * v_dht, const codec_memory_attn_params & p) {
    if (ctx == nullptr || mem == nullptr || q_dht == nullptr || k_dht == nullptr || v_dht == nullptr || p.window <= 0 ||
        p.head_dim <= 0 || p.n_heads <= 0) {
        return nullptr;
    }
    const int32_t n_kv = p.n_kv_heads > 0 ? p.n_kv_heads : p.n_heads;
    const int64_t t = q_dht->ne[2];
    const int32_t C = p.window - 1;

    ggml_tensor * pos = nullptr;
    ggml_tensor * mask = nullptr;
    if (!codec_memory_clock_inputs(ctx, mem, C, p.window, t, &pos, &mask)) return nullptr;

    ggml_tensor * q_r = ggml_rope_ext(ctx, q_dht, pos, nullptr, p.head_dim, p.rope_mode, 0, p.rope_theta,
                                      p.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_r = ggml_rope_ext(ctx, k_dht, pos, nullptr, p.head_dim, p.rope_mode, 0, p.rope_theta,
                                      p.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * k_all = k_r;
    ggml_tensor * v_all = v_dht;
    if (C > 0) {
        ggml_tensor * k_slot = codec_memory_slot(ctx, mem, CODEC_MEMORY_SLOT_K, GGML_TYPE_F32, p.head_dim, n_kv, C);
        ggml_tensor * v_slot = codec_memory_slot(ctx, mem, CODEC_MEMORY_SLOT_V, GGML_TYPE_F32, p.head_dim, n_kv, C);
        if (k_slot == nullptr || v_slot == nullptr) return nullptr;
        k_all = ggml_concat(ctx, k_slot, ggml_cont(ctx, k_r), 2);   // [d, h_kv, C + t]
        v_all = ggml_concat(ctx, v_slot, ggml_cont(ctx, v_dht), 2);
        codec_memory_store(ctx, mem, ggml_view_3d(ctx, k_all, p.head_dim, n_kv, C, k_all->nb[1], k_all->nb[2], (size_t) t * k_all->nb[2]), k_slot);
        codec_memory_store(ctx, mem, ggml_view_3d(ctx, v_all, p.head_dim, n_kv, C, v_all->nb[1], v_all->nb[2], (size_t) t * v_all->nb[2]), v_slot);
    }
    const int64_t n_k = C + t;

    ggml_tensor * q_dth = ggml_permute(ctx, q_r, 0, 2, 1, 3);                         // [d, t, h]
    ggml_tensor * k_dth = ggml_cont(ctx, ggml_permute(ctx, k_all, 0, 2, 1, 3));      // [d, n_k, h_kv]
    ggml_tensor * v_dth = ggml_permute(ctx, v_all, 0, 2, 1, 3);                       // [d, n_k, h_kv]
    if (n_kv != p.n_heads) {
        k_dth = ggml_repeat(ctx, k_dth, ggml_new_tensor_3d(ctx, k_dth->type, p.head_dim, n_k, p.n_heads));
        v_dth = ggml_repeat(ctx, ggml_cont(ctx, v_dth), ggml_new_tensor_3d(ctx, v_dth->type, p.head_dim, n_k, p.n_heads));
    }
    const float scale = p.scale > 0.0f ? p.scale : 1.0f / std::sqrt((float) p.head_dim);
    ggml_tensor * scores = codec_mul_mat(ctx, k_dth, q_dth);                           // [n_k, t, h]
    const bool padded = p.n_valid > 0 && p.n_valid < t;
    if (padded) {
        std::vector<float> pad((size_t) n_k, 0.0f);
        for (int64_t j = C + p.n_valid; j < n_k; ++j) pad[(size_t) j] = -INFINITY;
        const std::string name = "mem.padmask." + std::to_string(n_k) + "." + std::to_string(p.n_valid);
        mask = ggml_add(ctx, mask, codec_graph_const_f32(ctx, name.c_str(), pad, n_k, 1));
    }
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * v_tdh = ggml_cont(ctx, ggml_permute(ctx, v_dth, 1, 0, 2, 3));       // [n_k, d, h]
    ggml_tensor * out = codec_mul_mat(ctx, v_tdh, probs);                              // [d, t, h]
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));                         // [d, h, t]
    out = ggml_reshape_2d(ctx, out, (int64_t) p.head_dim * p.n_heads, t);
    if (padded) {
        std::vector<float> rows((size_t) t, 0.0f);
        for (int64_t i = 0; i < p.n_valid; ++i) rows[(size_t) i] = 1.0f;
        const std::string name = "mem.padrows." + std::to_string(t) + "." + std::to_string(p.n_valid);
        out = ggml_mul(ctx, out, codec_graph_const_f32(ctx, name.c_str(), rows, 1, t));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Public API

bool codec_model_encoder_is_causal(const struct codec_model * model) {
    return model != nullptr && model->has_encoder && model->enc_look_ahead == 0;
}

bool codec_model_decoder_is_causal(const struct codec_model * model) {
    return model != nullptr && model->has_decoder && model->dec_look_ahead == 0;
}

codec_memory_t codec_get_encoder_memory(const struct codec_context * ctx) {
    return ctx != nullptr ? ctx->enc_mem : nullptr;
}

codec_memory_t codec_get_decoder_memory(const struct codec_context * ctx) {
    return ctx != nullptr ? ctx->dec_mem : nullptr;
}

void codec_memory_clear(codec_memory_t mem) {
    if (mem == nullptr) return;
    if (mem->buf != nullptr) ggml_backend_buffer_clear(mem->buf, 0);
    for (codec_memory_clock & c : mem->clocks) {
        c.n = 0;
        c.t_call = 0;
    }
    mem->fresh = true;
    mem->position = 0;
}

// Blob (little-endian): header, then one section per slot.
//   char[4] "CDCM" | u32 format_version | u32 arch | u32 side | u8[32] fingerprint
//   u32 impl_version | i64 position | u8 fresh | u8[7] pad | u32 n_clocks | i64 clock_n[n_clocks]
//   u32 n_sections | per section: u32 kind, u32 ggml_type, i64 ne[4], u64 nbytes, data (8-byte aligned)
namespace {

struct blob_writer {
    uint8_t * dst;
    size_t cap;
    size_t n = 0;
    void put(const void * p, size_t k) {
        if (dst != nullptr && n + k <= cap) std::memcpy(dst + n, p, k);
        n += k;
    }
    template <typename T> void put(T v) { put(&v, sizeof(v)); }
    void align8() { static const uint8_t z[8] = {}; put(z, (8 - n % 8) % 8); }
};

struct blob_reader {
    const uint8_t * src;
    size_t cap;
    size_t n = 0;
    bool ok = true;
    void get(void * p, size_t k) {
        if (!ok || n + k > cap) { ok = false; return; }
        std::memcpy(p, src + n, k);
        n += k;
    }
    template <typename T> T get() { T v{}; get(&v, sizeof(v)); return v; }
    void align8() { const size_t k = (8 - n % 8) % 8; if (n + k > cap) ok = false; else n += k; }
};

size_t codec_memory_serialize(codec_memory * mem, uint8_t * dst, size_t cap) {
    blob_writer w{dst, cap};
    w.put(kStateMagic, 4);
    w.put<uint32_t>(kStateFormatVersion);
    w.put<uint32_t>((uint32_t) mem->ctx->model->arch);
    w.put<uint32_t>((uint32_t) mem->side);
    uint8_t fp[32];
    codec_memory_fingerprint(mem->ctx->model, fp);
    w.put(fp, 32);
    w.put<uint32_t>(kStateImplVersion);
    w.put<int64_t>(mem->position);
    w.put<uint8_t>(mem->fresh ? 1 : 0);
    static const uint8_t pad[7] = {};
    w.put(pad, 7);
    w.put<uint32_t>((uint32_t) mem->clocks.size());
    for (const codec_memory_clock & c : mem->clocks) w.put<int64_t>(c.n);
    w.put<uint32_t>((uint32_t) mem->slots.size());
    std::vector<uint8_t> tmp;
    for (size_t i = 0; i < mem->slots.size(); ++i) {
        ggml_tensor * t = mem->slots[i];
        w.put<uint32_t>(mem->specs[i].kind);
        w.put<uint32_t>((uint32_t) t->type);
        for (int d = 0; d < 4; ++d) w.put<int64_t>(t->ne[d]);
        const size_t nb = ggml_nbytes(t);
        w.put<uint64_t>(nb);
        w.align8();
        if (dst != nullptr && w.n + nb <= cap) {
            ggml_backend_tensor_get(t, dst + w.n, 0, nb);
        }
        w.n += nb;
    }
    return w.n;
}

}  // namespace

size_t codec_memory_state_get_size(codec_memory_t mem) {
    if (mem == nullptr || mem->slots.empty()) return 0;
    return codec_memory_serialize(mem, nullptr, 0);
}

size_t codec_memory_state_get_data(codec_memory_t mem, uint8_t * dst, size_t size) {
    if (mem == nullptr || mem->slots.empty()) return 0;
    const size_t need = codec_memory_serialize(mem, nullptr, 0);
    if (dst == nullptr || size < need) return 0;
    return codec_memory_serialize(mem, dst, size);
}

size_t codec_memory_state_set_data(codec_memory_t mem, const uint8_t * src, size_t size) {
    if (mem == nullptr) return 0;
    if (mem->slots.empty()) return 0;   // nothing to restore (size 0 blob)
    if (src == nullptr) return 0;
    blob_reader r{src, size};
    char magic[4];
    r.get(magic, 4);
    if (!r.ok || std::memcmp(magic, kStateMagic, 4) != 0) return 0;
    if (r.get<uint32_t>() != kStateFormatVersion) return 0;
    if (r.get<uint32_t>() != (uint32_t) mem->ctx->model->arch) return 0;
    if (r.get<uint32_t>() != (uint32_t) mem->side) return 0;
    uint8_t fp[32], fp_model[32];
    r.get(fp, 32);
    codec_memory_fingerprint(mem->ctx->model, fp_model);
    if (!r.ok || std::memcmp(fp, fp_model, 32) != 0) return 0;
    if (r.get<uint32_t>() != kStateImplVersion) return 0;
    const int64_t position = r.get<int64_t>();
    const bool fresh = r.get<uint8_t>() != 0;
    uint8_t pad[7];
    r.get(pad, 7);
    const uint32_t n_clocks = r.get<uint32_t>();
    if (!r.ok || n_clocks != mem->clocks.size()) return 0;
    std::vector<int64_t> clock_n(n_clocks);
    for (uint32_t i = 0; i < n_clocks; ++i) clock_n[i] = r.get<int64_t>();
    const uint32_t n_sections = r.get<uint32_t>();
    if (!r.ok || n_sections != mem->slots.size()) return 0;
    // validate every section before touching the cache
    std::vector<size_t> offsets(n_sections);
    for (uint32_t i = 0; i < n_sections; ++i) {
        ggml_tensor * t = mem->slots[i];
        const uint32_t kind = r.get<uint32_t>();
        const uint32_t type = r.get<uint32_t>();
        int64_t ne[4];
        for (int d = 0; d < 4; ++d) ne[d] = r.get<int64_t>();
        const uint64_t nb = r.get<uint64_t>();
        r.align8();
        if (!r.ok || kind != mem->specs[i].kind || type != (uint32_t) t->type || nb != ggml_nbytes(t) ||
            std::memcmp(ne, t->ne, sizeof(ne)) != 0 || r.n + nb > size) {
            return 0;
        }
        offsets[i] = r.n;
        r.n += nb;
    }
    for (uint32_t i = 0; i < n_sections; ++i) {
        ggml_backend_tensor_set(mem->slots[i], src + offsets[i], 0, ggml_nbytes(mem->slots[i]));
    }
    for (uint32_t i = 0; i < n_clocks; ++i) {
        mem->clocks[i].n = clock_n[i];
        mem->clocks[i].t_call = 0;
    }
    mem->position = position;
    mem->fresh = fresh;
    return r.n;
}
