#include "dump.h"

#include <algorithm>
#include <cstdio>

bool codec_mimi_debug_match_sem_l0_codebook_name(const std::string & name) {
    if (name.find("q.s.layers.0") == std::string::npos) {
        return false;
    }
    return name.find("codebook.embed") != std::string::npos ||
           name.find(".cb.embed") != std::string::npos ||
           name.find("cb.embed") != std::string::npos;
}

void codec_mimi_debug_print_tensor_lookup(const std::string & requested_name, const std::string & actual_name, const struct ggml_tensor * t) {
    if (!codec_mimi_debug_match_sem_l0_codebook_name(requested_name)) {
        return;
    }
    std::printf("DEBUG: mimi_tensor_lookup requested_name=%s actual_name=%s tensor=%s\n",
                requested_name.c_str(),
                actual_name.empty() ? "<none>" : actual_name.c_str(),
                t == nullptr ? "<null>" : "<set>");
}

void codec_mimi_dump_tc_stage(
    bool dump_tf,
    bool dump_all_layers,
    int32_t t,
    int32_t c,
    FILE * dbg_tf_meta,
    const char * stage_name,
    int32_t layer_idx,
    const std::vector<float> & tc,
    int32_t stage_c) {

    (void) dump_all_layers;
    (void) tc;
    const int32_t dump_c = stage_c > 0 ? stage_c : c;
    if (!dump_tf || dbg_tf_meta == nullptr || t <= 0 || dump_c <= 0) {
        return;
    }
    std::fprintf(dbg_tf_meta, "stage=%s layer=%d shape=%dx%d\n", stage_name, layer_idx, dump_c, t);
}

void codec_mimi_dump_mat_stage(
    bool dump_tf,
    bool dump_all_layers,
    FILE * dbg_tf_meta,
    const char * stage_name,
    int32_t layer_idx,
    const std::vector<float> & mat,
    int32_t rows,
    int32_t cols) {

    (void) dump_all_layers;
    (void) mat;
    if (!dump_tf || dbg_tf_meta == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    std::fprintf(dbg_tf_meta, "stage=%s layer=%d mat=%dx%d\n", stage_name, layer_idx, rows, cols);
}
