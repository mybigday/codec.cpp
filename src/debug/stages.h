#ifndef CODEC_DEBUG_STAGES_H
#define CODEC_DEBUG_STAGES_H

#include <cstdint>

const char * codec_mimi_debug_transformer_env(void);
bool codec_mimi_debug_transformer_enabled(void);
bool codec_mimi_debug_transformer_dump_all_layers(void);
bool codec_mimi_debug_should_dump_layer(bool dump_tf, bool dump_all_layers, int32_t layer_idx);

#endif
