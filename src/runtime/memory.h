#ifndef CODEC_RUNTIME_MEMORY_H
#define CODEC_RUNTIME_MEMORY_H

// Codec memory (docs/proposals/codec_memory.md): the context a causal encoder
// or decoder caches between calls: convolution left tails, transposed-conv
// overlap partial sums, sliding-window K/V and the attention positions.
//
// A graph builder that supports memory takes a `codec_memory *` (nullptr =
// the stateless batch graph) and, through the helpers below:
//   * reads a slot (persistent tensor, zero after a clear) as the left context
//     of an op and writes the context the call leaves behind back into it, in
//     the same graph (ggml_cpy side roots, as llama.cpp writes its KV cache);
//   * takes per-call inputs (attention positions / validity mask, the
//     "fresh stream" flag) that codec_memory_set_inputs fills after
//     codec_graph_prepare_io.
// Slots and clocks are numbered in build order, which does not depend on the
// chunk length, so a plan build at context creation sizes them once.

#include "../codec_internal.h"

#include <string>
#include <vector>

enum codec_memory_slot_kind : uint32_t {
    CODEC_MEMORY_SLOT_CONV_TAIL      = 1,
    CODEC_MEMORY_SLOT_CONVTR_OVERLAP = 2,
    CODEC_MEMORY_SLOT_K              = 3,
    CODEC_MEMORY_SLOT_V              = 4,
};

struct codec_memory_slot_spec {
    uint32_t kind = 0;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
};

// One attention stack's position counter: tokens seen since the clear, at
// that stack's frame rate. `cache_len` past tokens are kept; a query at
// position p sees keys in [p - window + 1, p] (window <= 0: all cached).
struct codec_memory_clock {
    int32_t cache_len = 0;
    int32_t window = 0;
    int64_t n = 0;
    int32_t t_call = 0;   // tokens of the call being computed
};

struct codec_memory {
    codec_context * ctx = nullptr;
    int32_t side = 0;     // 0 = encoder, 1 = decoder

    std::vector<codec_memory_slot_spec> specs;
    std::vector<ggml_tensor *> slots;   // in tctx, allocated in buf
    std::vector<codec_memory_clock> clocks;
    ggml_context * tctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    bool fresh = true;      // nothing fed since the clear
    int64_t position = 0;   // codec frames fed since the clear

    // build state
    bool planning = false;
    size_t next_slot = 0;
    size_t next_clock = 0;
};

enum { CODEC_MEMORY_ENCODER = 0, CODEC_MEMORY_DECODER = 1 };

// Context lifecycle (codec.cpp).
bool codec_memory_create(codec_context * ctx, int32_t side, codec_memory ** out, std::string * err);
void codec_memory_free(codec_memory * mem);

// The memory a model's encode / decode should use for this call: nullptr when
// not enabled, empty (non-causal side), or bypassed (codec_decode_batch).
codec_memory * codec_memory_active(codec_context * ctx, int32_t side);

// Runs `build_fn` in planning mode (slot / clock shapes are recorded, nothing
// is computed). Called by a model's memory_plan hook.
bool codec_memory_plan_build(codec_context * ctx, codec_memory * mem, codec_graph_build_fn build_fn, void * user_data,
                             std::string * err);

// --- build-time helpers (call from a build_fn; mem may be nullptr) ---------
void codec_memory_begin_build(codec_memory * mem);
ggml_tensor * codec_memory_slot(ggml_context * ctx_eval, codec_memory * mem, uint32_t kind, ggml_type type,
                                int64_t ne0, int64_t ne1, int64_t ne2 = 1);
// Writes `src` (same shape as `slot`) into the slot at the end of the compute.
void codec_memory_store(ggml_context * ctx_eval, codec_memory * mem, ggml_tensor * src, ggml_tensor * slot);
// Registers an attention clock; returns its per-call inputs:
//   pos  I32 [t]                   absolute positions of the call's tokens
//   mask F32 [cache_len + t, t]    0 / -inf: cached slots then the call's tokens
bool codec_memory_clock_inputs(ggml_context * ctx_eval, codec_memory * mem, int32_t cache_len, int32_t window, int64_t t,
                               ggml_tensor ** pos, ggml_tensor ** mask);
// F32 [1]: 1 on the first call after a clear, else 0.
ggml_tensor * codec_memory_fresh_input(ggml_context * ctx_eval, codec_memory * mem);

// --- per call ---------------------------------------------------------------
bool codec_memory_set_inputs(codec_context * ctx, codec_graph_cache_entry * entry, codec_memory * mem, std::string * err);
// After a successful compute: advance the clocks.
void codec_memory_commit(codec_memory * mem, int64_t n_frames);

// --- stateful ops (mem == nullptr: the batch op) -----------------------------
// Causal conv1d whose left padding comes from the cache. `replicate`: the
// batch op pads by repeating the first value (Mimi's downsample).
ggml_tensor * codec_memory_conv1d_causal(ggml_context * ctx, codec_memory * mem, ggml_tensor * x, ggml_tensor * w,
                                         ggml_tensor * b, int32_t stride, int32_t dilation, bool replicate = false);
ggml_tensor * codec_memory_conv1d_depthwise_causal(ggml_context * ctx, codec_memory * mem, ggml_tensor * x, ggml_tensor * w,
                                                   ggml_tensor * b, int32_t stride, int32_t dilation);
// Causal ConvTranspose1d: the overlap the previous call's last frames add to
// this call's first samples is carried (before the bias).
ggml_tensor * codec_memory_convtr1d_causal(ggml_context * ctx, codec_memory * mem, ggml_tensor * x, ggml_tensor * w,
                                           ggml_tensor * b, int32_t stride);

// Cached causal attention. q_dht / k_dht / v_dht are [head_dim, n_heads(_kv), t]
// *before* RoPE; RoPE (`rope_mode`, positions from the clock) is applied here.
// Returns the context [head_dim * n_heads, t]. n_kv_heads < n_heads repeats K/V (GQA).
// `window` <= 0: unbounded causal attention, which memory cannot cache (the
// caller must pass a window when mem != nullptr).
struct codec_memory_attn_params {
    int32_t head_dim = 0;
    int32_t n_heads = 0;
    int32_t n_kv_heads = 0;
    int32_t window = 0;
    float rope_theta = 10000.0f;
    float rope_freq_scale = 1.0f;
    int32_t rope_mode = 0;
    float scale = 0.0f;
    // 0 < n_valid < t: the call's tokens from n_valid on are padding; as the
    // batch op (codec_lm_attn_params.n_valid) keys there are masked and their
    // output rows are zero.
    int32_t n_valid = 0;
};
ggml_tensor * codec_memory_attn(ggml_context * ctx, codec_memory * mem, ggml_tensor * q_dht, ggml_tensor * k_dht,
                                ggml_tensor * v_dht, const codec_memory_attn_params & p);

#endif
