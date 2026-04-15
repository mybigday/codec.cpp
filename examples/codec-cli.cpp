#include "codec.h"
#include "utils/npy_io.h"
#include "utils/wav_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s e2e --model <gguf> --in <wav> --out <wav> [--threads N] [--nq N] [--use-gpu]\n"
        "  %s encode --model <gguf> --in <wav> --out <codes.npy> [--threads N] [--nq N] [--use-gpu]\n"
        "  %s decode --model <gguf> --codes <codes.npy> --out <wav> [--threads N] [--nq N] [--use-gpu]\n"
        "  %s decode-latent --model <gguf> --latent <latent.npy> --out <wav> [--threads N] [--use-gpu]\n",
        prog, prog, prog, prog);
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

struct e2e_args {
    const char * model_path = nullptr;
    const char * input_wav = nullptr;
    const char * out_wav = nullptr;
    int32_t n_threads = 4;
    int32_t n_q = 0;
    bool use_gpu = false;
};

struct decode_args {
    const char * model_path = nullptr;
    const char * codes_npy = nullptr;
    const char * out_wav = nullptr;
    int32_t n_threads = 4;
    int32_t n_q = 0;
    bool use_gpu = false;
};

struct encode_args {
    const char * model_path = nullptr;
    const char * input_wav = nullptr;
    const char * out_codes = nullptr;
    int32_t n_threads = 4;
    int32_t n_q = 0;
    bool use_gpu = false;
};

struct decode_latent_args {
    const char * model_path = nullptr;
    const char * latent_npy = nullptr;
    const char * out_wav = nullptr;
    int32_t n_threads = 4;
    bool use_gpu = false;
};

static bool parse_e2e_args(int argc, char ** argv, e2e_args * out) {
    for (int i = 2; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --model\n");
                return false;
            }
            out->model_path = argv[++i];
        } else if (std::strcmp(arg, "--in") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --in\n");
                return false;
            }
            out->input_wav = argv[++i];
        } else if (std::strcmp(arg, "--out") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --out\n");
                return false;
            }
            out->out_wav = argv[++i];
        } else if (std::strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc || !parse_i32(argv[i + 1], &out->n_threads) || out->n_threads < 1) {
                std::fprintf(stderr, "invalid value for --threads\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--nq") == 0) {
            if (i + 1 >= argc || !parse_i32(argv[i + 1], &out->n_q) || out->n_q < 0) {
                std::fprintf(stderr, "invalid value for --nq\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--use-gpu") == 0) {
            out->use_gpu = true;
        } else {
            std::fprintf(stderr, "unknown argument for e2e: %s\n", arg);
            return false;
        }
    }

    if (out->model_path == nullptr || out->input_wav == nullptr || out->out_wav == nullptr) {
        std::fprintf(stderr, "e2e requires --model, --in and --out\n");
        return false;
    }

    return true;
}

static bool parse_decode_args(int argc, char ** argv, decode_args * out) {
    for (int i = 2; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --model\n");
                return false;
            }
            out->model_path = argv[++i];
        } else if (std::strcmp(arg, "--codes") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --codes\n");
                return false;
            }
            out->codes_npy = argv[++i];
        } else if (std::strcmp(arg, "--out") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --out\n");
                return false;
            }
            out->out_wav = argv[++i];
        } else if (std::strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc || !parse_i32(argv[i + 1], &out->n_threads) || out->n_threads < 1) {
                std::fprintf(stderr, "invalid value for --threads\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--nq") == 0) {
            if (i + 1 >= argc || !parse_i32(argv[i + 1], &out->n_q) || out->n_q < 0) {
                std::fprintf(stderr, "invalid value for --nq\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--use-gpu") == 0) {
            out->use_gpu = true;
        } else {
            std::fprintf(stderr, "unknown argument for decode: %s\n", arg);
            return false;
        }
    }

    if (out->model_path == nullptr || out->codes_npy == nullptr || out->out_wav == nullptr) {
        std::fprintf(stderr, "decode requires --model, --codes and --out\n");
        return false;
    }

    return true;
}

static bool parse_encode_args(int argc, char ** argv, encode_args * out) {
    for (int i = 2; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --model\n");
                return false;
            }
            out->model_path = argv[++i];
        } else if (std::strcmp(arg, "--in") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --in\n");
                return false;
            }
            out->input_wav = argv[++i];
        } else if (std::strcmp(arg, "--out") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --out\n");
                return false;
            }
            out->out_codes = argv[++i];
        } else if (std::strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc || !parse_i32(argv[i + 1], &out->n_threads) || out->n_threads < 1) {
                std::fprintf(stderr, "invalid value for --threads\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--nq") == 0) {
            if (i + 1 >= argc || !parse_i32(argv[i + 1], &out->n_q) || out->n_q < 0) {
                std::fprintf(stderr, "invalid value for --nq\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--use-gpu") == 0) {
            out->use_gpu = true;
        } else {
            std::fprintf(stderr, "unknown argument for encode: %s\n", arg);
            return false;
        }
    }

    if (out->model_path == nullptr || out->input_wav == nullptr || out->out_codes == nullptr) {
        std::fprintf(stderr, "encode requires --model, --in and --out\n");
        return false;
    }

    return true;
}

static bool parse_decode_latent_args(int argc, char ** argv, decode_latent_args * out) {
    for (int i = 2; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --model\n");
                return false;
            }
            out->model_path = argv[++i];
        } else if (std::strcmp(arg, "--latent") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --latent\n");
                return false;
            }
            out->latent_npy = argv[++i];
        } else if (std::strcmp(arg, "--out") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after --out\n");
                return false;
            }
            out->out_wav = argv[++i];
        } else if (std::strcmp(arg, "--threads") == 0) {
            if (i + 1 >= argc || !parse_i32(argv[i + 1], &out->n_threads) || out->n_threads < 1) {
                std::fprintf(stderr, "invalid value for --threads\n");
                return false;
            }
            ++i;
        } else if (std::strcmp(arg, "--use-gpu") == 0) {
            out->use_gpu = true;
        } else {
            std::fprintf(stderr, "unknown argument for decode-latent: %s\n", arg);
            return false;
        }
    }

    if (out->model_path == nullptr || out->latent_npy == nullptr || out->out_wav == nullptr) {
        std::fprintf(stderr, "decode-latent requires --model, --latent and --out\n");
        return false;
    }

    return true;
}

static int cmd_e2e(const e2e_args & args) {
    struct codec_model_params mparams = codec_model_default_params();
    mparams.n_threads = args.n_threads;
    mparams.use_gpu = args.use_gpu;

    struct codec_model * model = codec_model_load_from_file(args.model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_path);
        return 2;
    }

    if (!codec_model_has_encoder(model)) {
        std::fprintf(stderr, "model does not support encode for e2e: arch=%s\n", codec_arch_name(codec_model_arch(model)));
        codec_model_free(model);
        return 3;
    }

    struct codec_context * ctx = codec_init_from_model(model, codec_context_default_params());
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        codec_model_free(model);
        return 4;
    }

    codec_example_wav_data wav;
    std::string wav_err;
    if (!codec_example_load_wav_pcm16(args.input_wav, &wav, &wav_err)) {
        std::fprintf(stderr, "failed to load wav: %s (%s)\n", args.input_wav, wav_err.c_str());
        codec_free(ctx);
        codec_model_free(model);
        return 5;
    }

    struct codec_audio in_audio = {
        /*.data =*/ wav.pcm_i16.data(),
        /*.n_samples =*/ (int32_t)(wav.pcm_i16.size() / (size_t)wav.n_channels),
        /*.sample_rate =*/ wav.sample_rate,
        /*.n_channels =*/ wav.n_channels,
        /*.pcm_type =*/ CODEC_PCM_TYPE_I16,
    };

    struct codec_token_buffer tokens = {};
    struct codec_latent_buffer latent = {};
    struct codec_encode_params eparams = codec_encode_default_params();
    eparams.n_threads = args.n_threads;
    eparams.n_q = args.n_q;

    enum codec_status st = codec_encode_latent(ctx, &in_audio, &tokens, &latent, eparams);
    if (st != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "codec_encode_latent failed: status=%d err=%s\n", (int)st, codec_get_last_error(ctx));
        codec_free(ctx);
        codec_model_free(model);
        return 6;
    }

    struct codec_pcm_buffer pcm = {};
    struct codec_decode_params dparams = codec_decode_default_params();
    dparams.n_threads = args.n_threads;
    dparams.n_q = args.n_q;

    st = codec_decode(ctx, &tokens, &pcm, dparams);
    if (st != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "codec_decode failed: status=%d err=%s\n", (int)st, codec_get_last_error(ctx));
        codec_latent_buffer_free(&latent);
        codec_token_buffer_free(&tokens);
        codec_free(ctx);
        codec_model_free(model);
        return 7;
    }

    std::string write_err;
    if (!codec_example_write_wav_pcm16(args.out_wav, pcm.data, pcm.n_samples, pcm.sample_rate, &write_err)) {
        std::fprintf(stderr, "failed to write wav: %s\n", write_err.c_str());
        codec_pcm_buffer_free(&pcm);
        codec_latent_buffer_free(&latent);
        codec_token_buffer_free(&tokens);
        codec_free(ctx);
        codec_model_free(model);
        return 8;
    }

    std::printf("model: %s (%s)\n", codec_model_name(model), codec_arch_name(codec_model_arch(model)));
    std::printf("encoded: n_frames=%d n_q=%d n_tokens=%d\n", tokens.n_frames, tokens.n_q, tokens.n_tokens);
    std::printf("decoded: n_samples=%d sr=%d\n", pcm.n_samples, pcm.sample_rate);
    std::printf("wrote: %s\n", args.out_wav);

    codec_pcm_buffer_free(&pcm);
    codec_latent_buffer_free(&latent);
    codec_token_buffer_free(&tokens);
    codec_free(ctx);
    codec_model_free(model);
    return 0;
}

static int cmd_decode(const decode_args & args) {
    std::string npy_err;
    std::vector<int32_t> tokens_data;
    int32_t n_q = 0;
    int32_t n_frames = 0;
    if (!codec_example_load_npy_i32_2d_tq(args.codes_npy, &tokens_data, &n_q, &n_frames, &npy_err)) {
        std::fprintf(stderr, "failed to read codes npy: %s (%s)\n", args.codes_npy, npy_err.c_str());
        return 2;
    }

    struct codec_model_params mparams = codec_model_default_params();
    mparams.n_threads = args.n_threads;
    mparams.use_gpu = args.use_gpu;
    struct codec_model * model = codec_model_load_from_file(args.model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_path);
        return 3;
    }

    if (!codec_model_has_decoder(model)) {
        std::fprintf(stderr, "model does not support decode: arch=%s\n", codec_arch_name(codec_model_arch(model)));
        codec_model_free(model);
        return 4;
    }

    struct codec_context * ctx = codec_init_from_model(model, codec_context_default_params());
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        codec_model_free(model);
        return 5;
    }

    struct codec_token_buffer tokens = {};
    tokens.data = tokens_data.data();
    tokens.n_q = n_q;
    tokens.n_frames = n_frames;
    tokens.n_tokens = n_q * n_frames;
    tokens.codebook_size = codec_model_codebook_size(model);
    tokens.sample_rate = codec_model_sample_rate(model);
    tokens.hop_size = codec_model_hop_size(model);

    struct codec_pcm_buffer pcm = {};
    struct codec_decode_params dparams = codec_decode_default_params();
    dparams.n_threads = args.n_threads;
    dparams.n_q = args.n_q;

    enum codec_status st = codec_decode(ctx, &tokens, &pcm, dparams);
    if (st != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "codec_decode failed: status=%d err=%s\n", (int)st, codec_get_last_error(ctx));
        codec_free(ctx);
        codec_model_free(model);
        return 6;
    }

    std::string wav_err;
    if (!codec_example_write_wav_pcm16(args.out_wav, pcm.data, pcm.n_samples, pcm.sample_rate, &wav_err)) {
        std::fprintf(stderr, "failed to write wav: %s\n", wav_err.c_str());
        codec_pcm_buffer_free(&pcm);
        codec_free(ctx);
        codec_model_free(model);
        return 7;
    }

    const int32_t effective_n_q = args.n_q == 0 ? codec_model_n_q(model) : args.n_q;
    std::printf("model: %s (%s)\n", codec_model_name(model), codec_arch_name(codec_model_arch(model)));
    std::printf("decoded: input_n_q=%d use_n_q=%d n_frames=%d n_samples=%d sr=%d\n",
        n_q, effective_n_q, n_frames, pcm.n_samples, pcm.sample_rate);
    std::printf("wrote: %s\n", args.out_wav);

    codec_pcm_buffer_free(&pcm);
    codec_free(ctx);
    codec_model_free(model);
    return 0;
}

static int cmd_decode_latent(const decode_latent_args & args) {
    std::string npy_err;
    std::vector<float> latent_data;
    int32_t n_frames = 0;
    int32_t latent_dim = 0;
    if (!codec_example_load_npy_f32_2d(args.latent_npy, &latent_data, &n_frames, &latent_dim, &npy_err)) {
        std::fprintf(stderr, "failed to read latent npy: %s (%s)\n", args.latent_npy, npy_err.c_str());
        return 2;
    }

    struct codec_model_params mparams = codec_model_default_params();
    mparams.n_threads = args.n_threads;
    mparams.use_gpu = args.use_gpu;
    struct codec_model * model = codec_model_load_from_file(args.model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_path);
        return 3;
    }

    if (!codec_model_has_decoder(model)) {
        std::fprintf(stderr, "model does not support decode: arch=%s\n", codec_arch_name(codec_model_arch(model)));
        codec_model_free(model);
        return 4;
    }

    struct codec_context * ctx = codec_init_from_model(model, codec_context_default_params());
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        codec_model_free(model);
        return 5;
    }

    struct codec_pcm_buffer pcm = {};
    struct codec_decode_params dparams = codec_decode_default_params();
    dparams.n_threads = args.n_threads;

    enum codec_status st = codec_decode_quantized_representation(ctx, latent_data.data(), latent_dim, n_frames, &pcm, dparams);
    if (st != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "codec_decode_quantized_representation failed: status=%d err=%s\n", (int)st, codec_get_last_error(ctx));
        codec_free(ctx);
        codec_model_free(model);
        return 6;
    }

    std::string wav_err;
    if (!codec_example_write_wav_pcm16(args.out_wav, pcm.data, pcm.n_samples, pcm.sample_rate, &wav_err)) {
        std::fprintf(stderr, "failed to write wav: %s\n", wav_err.c_str());
        codec_pcm_buffer_free(&pcm);
        codec_free(ctx);
        codec_model_free(model);
        return 7;
    }

    std::printf("model: %s (%s)\n", codec_model_name(model), codec_arch_name(codec_model_arch(model)));
    std::printf("decoded latent: n_frames=%d latent_dim=%d n_samples=%d sr=%d\n",
        n_frames, latent_dim, pcm.n_samples, pcm.sample_rate);
    std::printf("wrote: %s\n", args.out_wav);

    codec_pcm_buffer_free(&pcm);
    codec_free(ctx);
    codec_model_free(model);
    return 0;
}

static int cmd_encode(const encode_args & args) {
    struct codec_model_params mparams = codec_model_default_params();
    mparams.n_threads = args.n_threads;
    mparams.use_gpu = args.use_gpu;

    struct codec_model * model = codec_model_load_from_file(args.model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_path);
        return 2;
    }

    if (!codec_model_has_encoder(model)) {
        std::fprintf(stderr, "model does not support encode: arch=%s\n", codec_arch_name(codec_model_arch(model)));
        codec_model_free(model);
        return 3;
    }

    struct codec_context * ctx = codec_init_from_model(model, codec_context_default_params());
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        codec_model_free(model);
        return 4;
    }

    codec_example_wav_data wav;
    std::string wav_err;
    if (!codec_example_load_wav_pcm16(args.input_wav, &wav, &wav_err)) {
        std::fprintf(stderr, "failed to load wav: %s (%s)\n", args.input_wav, wav_err.c_str());
        codec_free(ctx);
        codec_model_free(model);
        return 5;
    }

    struct codec_audio in_audio = {
        /*.data =*/ wav.pcm_i16.data(),
        /*.n_samples =*/ (int32_t)(wav.pcm_i16.size() / (size_t)wav.n_channels),
        /*.sample_rate =*/ wav.sample_rate,
        /*.n_channels =*/ wav.n_channels,
        /*.pcm_type =*/ CODEC_PCM_TYPE_I16,
    };

    struct codec_token_buffer tokens = {};
    struct codec_encode_params eparams = codec_encode_default_params();
    eparams.n_threads = args.n_threads;
    eparams.n_q = args.n_q;

    enum codec_status st = codec_encode(ctx, &in_audio, &tokens, eparams);
    if (st != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "codec_encode failed: status=%d err=%s\n", (int)st, codec_get_last_error(ctx));
        codec_token_buffer_free(&tokens);
        codec_free(ctx);
        codec_model_free(model);
        return 6;
    }

    std::string npy_err;
    if (!codec_example_save_npy_i32_2d_qt(args.out_codes, tokens.data, tokens.n_q, tokens.n_frames, &npy_err)) {
        std::fprintf(stderr, "failed to write codes npy: %s (%s)\n", args.out_codes, npy_err.c_str());
        codec_token_buffer_free(&tokens);
        codec_free(ctx);
        codec_model_free(model);
        return 7;
    }

    std::printf("model: %s (%s)\n", codec_model_name(model), codec_arch_name(codec_model_arch(model)));
    std::printf("encoded: n_frames=%d n_q=%d n_tokens=%d\n", tokens.n_frames, tokens.n_q, tokens.n_tokens);
    std::printf("wrote: %s\n", args.out_codes);

    codec_token_buffer_free(&tokens);
    codec_free(ctx);
    codec_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char * cmd = argv[1];
    if (std::strcmp(cmd, "e2e") == 0) {
        e2e_args args;
        if (!parse_e2e_args(argc, argv, &args)) {
            print_usage(argv[0]);
            return 1;
        }
        return cmd_e2e(args);
    }

    if (std::strcmp(cmd, "encode") == 0) {
        encode_args args;
        if (!parse_encode_args(argc, argv, &args)) {
            print_usage(argv[0]);
            return 1;
        }
        return cmd_encode(args);
    }

    if (std::strcmp(cmd, "decode") == 0) {
        decode_args args;
        if (!parse_decode_args(argc, argv, &args)) {
            print_usage(argv[0]);
            return 1;
        }
        return cmd_decode(args);
    }

    if (std::strcmp(cmd, "decode-latent") == 0) {
        decode_latent_args args;
        if (!parse_decode_latent_args(argc, argv, &args)) {
            print_usage(argv[0]);
            return 1;
        }
        return cmd_decode_latent(args);
    }

    std::fprintf(stderr, "unknown command: %s\n", cmd);
    print_usage(argv[0]);
    return 1;
}
