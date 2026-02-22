#ifndef CODEC_DEBUG_DUMP_H
#define CODEC_DEBUG_DUMP_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct ggml_tensor;

bool codec_mimi_debug_match_sem_l0_codebook_name(const std::string & name);
void codec_mimi_debug_print_tensor_lookup(const std::string & requested_name, const std::string & actual_name, const struct ggml_tensor * t);
void codec_mimi_dump_tc_stage(
    bool dump_tf,
    bool dump_all_layers,
    int32_t t,
    int32_t c,
    FILE * dbg_tf_meta,
    const char * stage_name,
    int32_t layer_idx,
    const std::vector<float> & tc,
    int32_t stage_c = -1);
void codec_mimi_dump_mat_stage(
    bool dump_tf,
    bool dump_all_layers,
    FILE * dbg_tf_meta,
    const char * stage_name,
    int32_t layer_idx,
    const std::vector<float> & mat,
    int32_t rows,
    int32_t cols);

#endif
