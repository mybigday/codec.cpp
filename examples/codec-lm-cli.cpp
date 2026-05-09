// codec-lm-cli — thin CLI for parity-testing the codec_lm runtime
// against an HF reference.  Two subcommands:
//
//   step    --model <gguf> --hidden <h.npy> --logits-prefix <pfx>
//                                            [--codes-out <c.npy>]
//   compose --model <gguf> --codes <c.npy> --embd-out <e.npy>
//
// `step` runs codec_lm_step_begin on the given hidden state, copies each
// codebook's logits out as <pfx>_<i>.npy, and (optionally) greedy-samples
// to produce a codes_out.npy.  `compose` runs codec_lm_compose_audio_embd
// on a code vector and writes the resulting hidden_dim float buffer.
//
// The Python parity test (tests/e2e/moss_ttsd_lm_smoke.py) drives both
// against the HF reference and asserts max-abs / corr thresholds.

#include "codec.h"
#include "codec_lm.h"
#include "utils/npy_io.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int usage(const char * me) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s step    --model <gguf> --hidden <h.npy> --logits-prefix <pfx> [--codes-out <c.npy>]\n"
        "  %s compose --model <gguf> --codes <c.npy> --embd-out <e.npy>\n",
        me, me);
    return 2;
}

static const char * arg_value(int argc, char ** argv, int * i) {
    if (*i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", argv[*i]);
        std::exit(2);
    }
    return argv[++*i];
}

static int cmd_step(int argc, char ** argv) {
    const char * model_path  = nullptr;
    const char * hidden_path = nullptr;
    const char * logits_pfx  = nullptr;
    const char * codes_out   = nullptr;

    for (int i = 2; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--model")          == 0) model_path  = arg_value(argc, argv, &i);
        else if (std::strcmp(argv[i], "--hidden")         == 0) hidden_path = arg_value(argc, argv, &i);
        else if (std::strcmp(argv[i], "--logits-prefix")  == 0) logits_pfx  = arg_value(argc, argv, &i);
        else if (std::strcmp(argv[i], "--codes-out")      == 0) codes_out   = arg_value(argc, argv, &i);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!model_path || !hidden_path || !logits_pfx) {
        return usage(argv[0]);
    }

    codec_example_npy_array h_arr;
    std::string err;
    if (!codec_example_load_npy(hidden_path, &h_arr, &err)) {
        std::fprintf(stderr, "load hidden npy failed: %s\n", err.c_str());
        return 3;
    }
    if (h_arr.dtype != CODEC_EXAMPLE_NPY_DTYPE_F32) {
        std::fprintf(stderr, "hidden must be float32, got dtype=%d\n", (int) h_arr.dtype);
        return 3;
    }

    codec_model_params mp = codec_model_default_params();
    codec_model * codec = codec_model_load_from_file(model_path, mp);
    if (!codec) { std::fprintf(stderr, "codec_model_load_from_file failed\n"); return 4; }

    codec_lm * lm = codec_lm_create(codec);
    if (!lm) {
        std::fprintf(stderr, "codec_lm_create failed: GGUF has no codec.lm.* metadata?\n");
        codec_model_free(codec);
        return 5;
    }

    const codec_lm_info * info = codec_lm_get_info(lm);
    if ((int32_t) h_arr.f32.size() != info->hidden_dim) {
        std::fprintf(stderr, "hidden length %zu != info->hidden_dim %d\n",
                     h_arr.f32.size(), info->hidden_dim);
        codec_lm_free(lm); codec_model_free(codec);
        return 6;
    }

    codec_lm_state * st = codec_lm_state_new(lm);
    if (!st) {
        std::fprintf(stderr, "codec_lm_state_new failed\n");
        codec_lm_free(lm); codec_model_free(codec);
        return 7;
    }

    enum codec_status rc = codec_lm_step_begin(st, h_arr.f32.data());
    if (rc != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "step_begin rc=%d err='%s'\n",
                     (int) rc, codec_lm_state_get_last_error(st));
        codec_lm_state_free(st); codec_lm_free(lm); codec_model_free(codec);
        return 8;
    }

    std::vector<int32_t> codes((size_t) info->n_codebook, 0);
    char path_buf[1024];
    while (codec_lm_step_pending(st)) {
        int32_t cb_idx = -1;
        int32_t n      = 0;
        const float * lg = codec_lm_step_logits(st, &cb_idx, &n);
        if (!lg || n <= 0) {
            std::fprintf(stderr, "step_logits failed at cb=%d: %s\n",
                         cb_idx, codec_lm_state_get_last_error(st));
            codec_lm_state_free(st); codec_lm_free(lm); codec_model_free(codec);
            return 9;
        }

        std::snprintf(path_buf, sizeof(path_buf), "%s_%d.npy", logits_pfx, cb_idx);
        if (!codec_example_save_npy_f32_1d(path_buf, lg, n, &err)) {
            std::fprintf(stderr, "save logits npy failed: %s\n", err.c_str());
            codec_lm_state_free(st); codec_lm_free(lm); codec_model_free(codec);
            return 10;
        }

        // greedy sample (the python test only really reads logits; codes
        // are emitted for convenience / smoke-checking the state machine).
        int32_t best = 0;
        float   best_v = lg[0];
        for (int32_t i = 1; i < n; ++i) {
            if (lg[i] > best_v) { best = i; best_v = lg[i]; }
        }
        codec_lm_step_push_code(st, best);
        codes[(size_t) cb_idx] = best;
    }

    int32_t finished[64] = {0};
    codec_lm_step_finish(st, finished);

    if (codes_out) {
        if (!codec_example_save_npy_i32_1d(codes_out, finished, info->n_codebook, &err)) {
            std::fprintf(stderr, "save codes npy failed: %s\n", err.c_str());
            codec_lm_state_free(st); codec_lm_free(lm); codec_model_free(codec);
            return 11;
        }
    }

    codec_lm_state_free(st);
    codec_lm_free(lm);
    codec_model_free(codec);
    return 0;
}

static int cmd_compose(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * codes_path = nullptr;
    const char * embd_out   = nullptr;
    for (int i = 2; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--model")    == 0) model_path = arg_value(argc, argv, &i);
        else if (std::strcmp(argv[i], "--codes")    == 0) codes_path = arg_value(argc, argv, &i);
        else if (std::strcmp(argv[i], "--embd-out") == 0) embd_out   = arg_value(argc, argv, &i);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!model_path || !codes_path || !embd_out) return usage(argv[0]);

    codec_example_npy_array c_arr;
    std::string err;
    if (!codec_example_load_npy(codes_path, &c_arr, &err)) {
        std::fprintf(stderr, "load codes npy failed: %s\n", err.c_str());
        return 3;
    }
    if (c_arr.dtype != CODEC_EXAMPLE_NPY_DTYPE_I32) {
        std::fprintf(stderr, "codes must be int32, got dtype=%d\n", (int) c_arr.dtype);
        return 3;
    }

    codec_model_params mp = codec_model_default_params();
    codec_model * codec = codec_model_load_from_file(model_path, mp);
    if (!codec) { std::fprintf(stderr, "codec_model_load_from_file failed\n"); return 4; }

    codec_lm * lm = codec_lm_create(codec);
    if (!lm) {
        std::fprintf(stderr, "codec_lm_create failed\n");
        codec_model_free(codec);
        return 5;
    }
    const codec_lm_info * info = codec_lm_get_info(lm);
    if ((int32_t) c_arr.i32.size() != info->n_codebook) {
        std::fprintf(stderr, "codes length %zu != n_codebook %d\n",
                     c_arr.i32.size(), info->n_codebook);
        codec_lm_free(lm); codec_model_free(codec);
        return 6;
    }

    std::vector<float> embd((size_t) info->audio_embed_dim, 0.0f);
    if (codec_lm_compose_audio_embd(lm, c_arr.i32.data(), embd.data()) != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "compose_audio_embd failed: %s\n", codec_lm_get_last_error(lm));
        codec_lm_free(lm); codec_model_free(codec);
        return 7;
    }

    if (!codec_example_save_npy_f32_1d(embd_out, embd.data(), info->audio_embed_dim, &err)) {
        std::fprintf(stderr, "save embd npy failed: %s\n", err.c_str());
        codec_lm_free(lm); codec_model_free(codec);
        return 8;
    }
    codec_lm_free(lm); codec_model_free(codec);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) return usage(argv[0]);
    if (std::strcmp(argv[1], "step")    == 0) return cmd_step(argc, argv);
    if (std::strcmp(argv[1], "compose") == 0) return cmd_compose(argc, argv);
    return usage(argv[0]);
}
