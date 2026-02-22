#include "stages.h"

#include <cstdlib>
#include <cstring>

const char * codec_mimi_debug_transformer_env(void) {
    return std::getenv("CODEC_MIMI_DEBUG_TRANSFORMER");
}

bool codec_mimi_debug_transformer_enabled(void) {
    const char * debug_tf = codec_mimi_debug_transformer_env();
    return debug_tf != nullptr && debug_tf[0] != '\0';
}

bool codec_mimi_debug_transformer_dump_all_layers(void) {
    const char * debug_tf = codec_mimi_debug_transformer_env();
    return debug_tf != nullptr && debug_tf[0] != '\0' && std::strcmp(debug_tf, "all") == 0;
}

bool codec_mimi_debug_should_dump_layer(bool dump_tf, bool dump_all_layers, int32_t layer_idx) {
    if (!dump_tf || layer_idx < 0) {
        return false;
    }
    if (!dump_all_layers && layer_idx != 0) {
        return false;
    }
    return true;
}
