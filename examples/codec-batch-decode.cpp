#include "codec.h"
#include "utils/npy_io.h"
#include "utils/wav_io.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

struct codec_batch_decode_params {
    const char * model_path = nullptr;
    const char * codes_path = nullptr;
    const char * latent_path = nullptr;
    const char * out_path = nullptr;
    int32_t n_threads = 4;
    int32_t n_q = 0;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s --model <gguf> --codes <codes.npy> --out <path> [--threads N] [--nq N]\n"
        "  %s --model <gguf> --latent <latent.npy> --out <path> [--threads N]\n",
        prog, prog);
}

static bool parse_i32(const char * value, int32_t * out) {
    if (value == nullptr || out == nullptr) {
        return false;
    }

    char * end = nullptr;
    long v = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    if (v < INT32_MIN || v > INT32_MAX) {
        return false;
    }

    *out = (int32_t)v;
    return true;
}

static bool parse_args(int argc, char ** argv, struct codec_batch_decode_params * out, bool * show_help) {
    if (out == nullptr) {
        return false;
    }
    if (show_help != nullptr) {
        *show_help = false;
    }

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            if (show_help != nullptr) {
                *show_help = true;
            }
            return true;
        }

        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --model\n");
                return false;
            }
            out->model_path = argv[++i];
            continue;
        }

        if (std::strcmp(arg, "--codes") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --codes\n");
                return false;
            }
            out->codes_path = argv[++i];
            continue;
        }

        if (std::strcmp(arg, "--latent") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --latent\n");
                return false;
            }
            out->latent_path = argv[++i];
            continue;
        }

        if (std::strcmp(arg, "--out") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --out\n");
                return false;
            }
            out->out_path = argv[++i];
            continue;
        }

        if (std::strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --threads\n");
                return false;
            }
            if (!parse_i32(argv[++i], &out->n_threads) || out->n_threads <= 0) {
                std::fprintf(stderr, "invalid --threads value: %s\n", argv[i]);
                return false;
            }
            continue;
        }

        if (std::strcmp(arg, "--nq") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --nq\n");
                return false;
            }
            if (!parse_i32(argv[++i], &out->n_q) || out->n_q < 0) {
                std::fprintf(stderr, "invalid --nq value: %s\n", argv[i]);
                return false;
            }
            continue;
        }

        std::fprintf(stderr, "unknown argument: %s\n", arg);
        return false;
    }

    if (out->model_path == nullptr || out->out_path == nullptr) {
        std::fprintf(stderr, "--model and --out are required\n");
        return false;
    }

    const bool has_codes = out->codes_path != nullptr;
    const bool has_latent = out->latent_path != nullptr;
    if (has_codes == has_latent) {
        std::fprintf(stderr, "exactly one of --codes or --latent is required\n");
        return false;
    }

    if (has_latent && out->n_q != 0) {
        std::fprintf(stderr, "--nq is only valid with --codes\n");
        return false;
    }

    return true;
}


int main(int argc, char ** argv) {
    struct codec_batch_decode_params args;
    bool show_help = false;
    if (!parse_args(argc, argv, &args, &show_help)) {
        print_usage(argv[0]);
        return 1;
    }
    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }

    struct codec_model_params mparams = codec_model_default_params();
    mparams.n_threads = args.n_threads;

    struct codec_model * model = codec_model_load_from_file(args.model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_path);
        return 2;
    }

    struct codec_context * ctx = codec_init_from_model(model, codec_context_default_params());
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize context\n");
        codec_model_free(model);
        return 3;
    }

    struct codec_decode_params dparams = codec_decode_default_params();
    dparams.n_threads = args.n_threads;
    dparams.n_q = args.n_q;

    std::string npy_err;
    struct codec_example_npy_array arr;
    const bool codes_mode = args.codes_path != nullptr;
    const char * input_path = codes_mode ? args.codes_path : args.latent_path;
    if (!codec_example_load_npy(input_path, &arr, &npy_err)) {
        std::fprintf(stderr, "failed to load npy: %s (%s)\n", input_path, npy_err.c_str());
        codec_free(ctx);
        codec_model_free(model);
        return 4;
    }

    if (arr.shape.size() != 2 && arr.shape.size() != 3) {
        std::fprintf(stderr, "unsupported npy rank: %zu (expected 2D or 3D)\n", arr.shape.size());
        codec_free(ctx);
        codec_model_free(model);
        return 5;
    }

    if (codes_mode && arr.dtype != CODEC_EXAMPLE_NPY_DTYPE_I32) {
        std::fprintf(stderr, "codes mode requires int32 npy\n");
        codec_free(ctx);
        codec_model_free(model);
        return 6;
    }
    if (!codes_mode && arr.dtype != CODEC_EXAMPLE_NPY_DTYPE_F32) {
        std::fprintf(stderr, "latent mode requires float32 npy\n");
        codec_free(ctx);
        codec_model_free(model);
        return 6;
    }

    const bool is_batch = arr.shape.size() == 3;

    if (!is_batch) {
        const int32_t n_frames = arr.shape[0];
        const int32_t width = arr.shape[1];

        struct codec_pcm_buffer pcm = {};
        enum codec_status st = CODEC_STATUS_SUCCESS;

        if (codes_mode) {
            const int32_t n_q = width;
            const struct codec_token_buffer tokens = {
                const_cast<int32_t *>(arr.i32.data()),
                n_frames * n_q,
                n_frames,
                n_q,
                codec_model_codebook_size(model),
                codec_model_sample_rate(model),
                codec_model_hop_size(model)
            };
            st = codec_decode(ctx, &tokens, &pcm, dparams);
        } else {
            const int32_t latent_dim = width;
            st = codec_decode_quantized_representation(ctx, arr.f32.data(), latent_dim, n_frames, &pcm, dparams);
        }

        if (st != CODEC_STATUS_SUCCESS) {
            std::fprintf(stderr, "decode failed: status=%d err=%s\n", (int)st, codec_get_last_error(ctx));
            codec_pcm_buffer_free(&pcm);
            codec_free(ctx);
            codec_model_free(model);
            return 7;
        }

        std::string wav_err;
        if (!codec_example_write_wav_pcm16(args.out_path, pcm.data, pcm.n_samples, pcm.sample_rate, &wav_err)) {
            std::fprintf(stderr, "failed to write wav: %s\n", wav_err.c_str());
            codec_pcm_buffer_free(&pcm);
            codec_free(ctx);
            codec_model_free(model);
            return 8;
        }

        std::printf("decoded single sequence: frames=%d dim=%d samples=%d sr=%d\n",
            n_frames, width, pcm.n_samples, pcm.sample_rate);
        std::printf("wrote: %s\n", args.out_path);

        codec_pcm_buffer_free(&pcm);
        codec_free(ctx);
        codec_model_free(model);
        return 0;
    }

    const int32_t n_seq = arr.shape[0];
    const int32_t n_frames = arr.shape[1];
    const int32_t width = arr.shape[2];

    if (n_seq <= 0 || n_frames <= 0 || width <= 0) {
        std::fprintf(stderr, "invalid 3D shape in npy\n");
        codec_free(ctx);
        codec_model_free(model);
        return 9;
    }

    struct codec_batch batch = {};
    if (codes_mode) {
        const int64_t total_codes64 = (int64_t)n_seq * (int64_t)n_frames * (int64_t)width;
        if (total_codes64 <= 0 || total_codes64 > INT32_MAX) {
            std::fprintf(stderr, "codes payload too large\n");
            codec_free(ctx);
            codec_model_free(model);
            return 10;
        }

        batch = codec_batch_init_codes(n_seq, (int32_t)total_codes64, n_seq);
        if (batch.codes == nullptr || batch.seq_id == nullptr || batch.n_frames == nullptr || batch.n_q == nullptr) {
            std::fprintf(stderr, "failed to initialize codes batch\n");
            codec_batch_free(batch);
            codec_free(ctx);
            codec_model_free(model);
            return 11;
        }

        const int32_t seq_stride = n_frames * width;
        for (int32_t i = 0; i < n_seq; ++i) {
            const int32_t * seq_codes = arr.i32.data() + (size_t)i * (size_t)seq_stride;
            if (codec_batch_add_seq_codes(&batch, i, n_frames, width, seq_codes) < 0) {
                std::fprintf(stderr, "failed to add codes sequence %d to batch\n", i);
                codec_batch_free(batch);
                codec_free(ctx);
                codec_model_free(model);
                return 12;
            }
        }
    } else {
        const int64_t total_latent64 = (int64_t)n_seq * (int64_t)n_frames * (int64_t)width;
        if (total_latent64 <= 0 || total_latent64 > INT32_MAX) {
            std::fprintf(stderr, "latent payload too large\n");
            codec_free(ctx);
            codec_model_free(model);
            return 10;
        }

        batch = codec_batch_init_latent(n_seq, width, (int32_t)total_latent64, n_seq);
        if (batch.latent == nullptr || batch.seq_id == nullptr || batch.n_frames == nullptr) {
            std::fprintf(stderr, "failed to initialize latent batch\n");
            codec_batch_free(batch);
            codec_free(ctx);
            codec_model_free(model);
            return 11;
        }

        const int32_t seq_stride = n_frames * width;
        for (int32_t i = 0; i < n_seq; ++i) {
            const float * seq_latent = arr.f32.data() + (size_t)i * (size_t)seq_stride;
            if (codec_batch_add_seq_latent(&batch, i, n_frames, seq_latent, width) < 0) {
                std::fprintf(stderr, "failed to add latent sequence %d to batch\n", i);
                codec_batch_free(batch);
                codec_free(ctx);
                codec_model_free(model);
                return 12;
            }
        }
    }

    std::vector<struct codec_pcm_buffer> pcm((size_t)n_seq);
    const enum codec_status st = codec_decode_batch(ctx, &batch, pcm.data(), dparams);
    codec_batch_free(batch);

    if (st != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "batch decode failed: status=%d err=%s\n", (int)st, codec_get_last_error(ctx));
        for (int32_t i = 0; i < n_seq; ++i) {
            codec_pcm_buffer_free(&pcm[(size_t)i]);
        }
        codec_free(ctx);
        codec_model_free(model);
        return 13;
    }

    std::error_code ec;
    std::filesystem::create_directories(args.out_path, ec);
    if (ec) {
        std::fprintf(stderr, "failed to create output directory: %s\n", ec.message().c_str());
        for (int32_t i = 0; i < n_seq; ++i) {
            codec_pcm_buffer_free(&pcm[(size_t)i]);
        }
        codec_free(ctx);
        codec_model_free(model);
        return 14;
    }

    bool write_ok = true;
    for (int32_t i = 0; i < n_seq; ++i) {
        std::string out_wav = std::string(args.out_path) + "/seq_" + std::to_string(i) + ".wav";
        std::string wav_err;
        if (!codec_example_write_wav_pcm16(out_wav.c_str(), pcm[(size_t)i].data, pcm[(size_t)i].n_samples, pcm[(size_t)i].sample_rate, &wav_err)) {
            std::fprintf(stderr, "failed to write %s: %s\n", out_wav.c_str(), wav_err.c_str());
            write_ok = false;
            break;
        }
        std::printf("wrote: %s (samples=%d sr=%d)\n", out_wav.c_str(), pcm[(size_t)i].n_samples, pcm[(size_t)i].sample_rate);
    }

    for (int32_t i = 0; i < n_seq; ++i) {
        codec_pcm_buffer_free(&pcm[(size_t)i]);
    }

    codec_free(ctx);
    codec_model_free(model);

    if (!write_ok) {
        return 15;
    }

    std::printf("decoded batch: n_seq=%d n_frames=%d dim=%d\n", n_seq, n_frames, width);
    return 0;
}
