#ifndef CODEC_MODEL_QWEN3_TTS_TOKENIZER_H
#define CODEC_MODEL_QWEN3_TTS_TOKENIZER_H

#include "../codec_internal.h"

enum codec_status codec_qwen3_tts_tokenizer_init(struct codec_model * model);
enum codec_status codec_qwen3_tts_tokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

#endif
