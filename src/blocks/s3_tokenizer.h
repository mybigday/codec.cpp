#ifndef CODEC_BLOCKS_S3_TOKENIZER_H
#define CODEC_BLOCKS_S3_TOKENIZER_H

#include "../codec_internal.h"

// S3TokenizerV2 (CosyVoice 2 / Chatterbox speech tokenizer, 25 Hz, 6561 FSQ
// codes): Whisper-style log-mel → 2 strided convs → FSMN transformer → FSQ.
// Weights live under `s3t.` (`s3t.mel_filters`, optional `s3t.window`,
// `s3t.enc.*`, `s3t.q.proj.*`).
struct codec_s3_tokenizer_params {
    int32_t n_fft = 400;
    int32_t n_mels = 128;
    int32_t hidden = 1280;
    int32_t n_heads = 20;
    int32_t n_layers = 6;
    int32_t fsmn_kernel = 31;
    float rope_theta = 10000.0f;
};

// Tokens produced for `n_samples` of 16 kHz audio.
int32_t codec_s3_tokenizer_n_tokens(int64_t n_samples);

// 16 kHz mono PCM [n] → token ids I32 [n_tokens]. Call from a graph build_fn.
ggml_tensor * codec_block_s3_tokenizer(
    ggml_context * ctx,
    const codec_model * model,
    const codec_s3_tokenizer_params & p,
    ggml_tensor * pcm16k);

#endif
