#include "codec.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

enum npy_dtype {
    NPY_DTYPE_UNKNOWN = 0,
    NPY_DTYPE_I32 = 1,
    NPY_DTYPE_F32 = 2,
};

struct npy_array {
    enum npy_dtype dtype = NPY_DTYPE_UNKNOWN;
    std::vector<int32_t> shape;
    std::vector<int32_t> i32;
    std::vector<float> f32;
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

static bool write_wav_pcm16(const char * path, const float * pcm, int32_t n_samples, int32_t sample_rate, std::string * err) {
    FILE * fp = std::fopen(path, "wb");
    if (fp == nullptr) {
        if (err != nullptr) {
            *err = "failed to open output wav";
        }
        return false;
    }

    const uint16_t audio_format = 1;
    const uint16_t n_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = n_channels * bits_per_sample / 8;
    const uint32_t byte_rate = (uint32_t)sample_rate * block_align;
    const uint32_t data_size = (uint32_t)n_samples * block_align;
    const uint32_t riff_size = 36 + data_size;

    std::fwrite("RIFF", 1, 4, fp);
    std::fwrite(&riff_size, 4, 1, fp);
    std::fwrite("WAVE", 1, 4, fp);
    std::fwrite("fmt ", 1, 4, fp);
    const uint32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, fp);
    std::fwrite(&audio_format, 2, 1, fp);
    std::fwrite(&n_channels, 2, 1, fp);
    std::fwrite(&sample_rate, 4, 1, fp);
    std::fwrite(&byte_rate, 4, 1, fp);
    std::fwrite(&block_align, 2, 1, fp);
    std::fwrite(&bits_per_sample, 2, 1, fp);
    std::fwrite("data", 1, 4, fp);
    std::fwrite(&data_size, 4, 1, fp);

    for (int32_t i = 0; i < n_samples; ++i) {
        const float x = std::max(-1.0f, std::min(1.0f, pcm[i]));
        const int32_t q = (int32_t)std::lround(x * 32767.0f);
        const int16_t s = (int16_t)std::max(-32768, std::min(32767, q));
        std::fwrite(&s, sizeof(s), 1, fp);
    }

    std::fclose(fp);
    return true;
}

static bool parse_shape_dims(const std::string & header, std::vector<int32_t> * out_shape, std::string * err) {
    const size_t p0 = header.find('(');
    const size_t p1 = header.find(')', p0 == std::string::npos ? 0 : p0 + 1);
    if (p0 == std::string::npos || p1 == std::string::npos || p1 <= p0 + 1) {
        if (err != nullptr) {
            *err = "invalid npy shape";
        }
        return false;
    }

    out_shape->clear();
    const std::string inside = header.substr(p0 + 1, p1 - p0 - 1);

    size_t i = 0;
    while (i < inside.size()) {
        while (i < inside.size() && (std::isspace((unsigned char)inside[i]) || inside[i] == ',')) {
            ++i;
        }
        if (i >= inside.size()) {
            break;
        }

        size_t j = i;
        while (j < inside.size() && std::isdigit((unsigned char)inside[j])) {
            ++j;
        }
        if (j == i) {
            if (err != nullptr) {
                *err = "invalid npy shape contents";
            }
            return false;
        }

        const std::string token = inside.substr(i, j - i);
        char * end = nullptr;
        long v = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0' || v <= 0 || v > INT32_MAX) {
            if (err != nullptr) {
                *err = "invalid npy dimensions";
            }
            return false;
        }

        out_shape->push_back((int32_t)v);
        i = j;
    }

    if (out_shape->empty()) {
        if (err != nullptr) {
            *err = "npy shape is empty";
        }
        return false;
    }

    return true;
}

static bool parse_npy_header(std::ifstream & ifs, enum npy_dtype * dtype, std::vector<int32_t> * shape, std::string * err) {
    char magic[6] = { 0 };
    if (!ifs.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        if (err != nullptr) {
            *err = "invalid npy magic";
        }
        return false;
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    if (!ifs.read(reinterpret_cast<char *>(&major), 1) || !ifs.read(reinterpret_cast<char *>(&minor), 1)) {
        if (err != nullptr) {
            *err = "invalid npy version";
        }
        return false;
    }

    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        if (!ifs.read(reinterpret_cast<char *>(&h16), 2)) {
            if (err != nullptr) {
                *err = "invalid npy header length";
            }
            return false;
        }
        hlen = h16;
    } else if (major == 2 || major == 3) {
        if (!ifs.read(reinterpret_cast<char *>(&hlen), 4)) {
            if (err != nullptr) {
                *err = "invalid npy header length";
            }
            return false;
        }
    } else {
        if (err != nullptr) {
            *err = "unsupported npy version";
        }
        return false;
    }

    std::string header(hlen, '\0');
    if (!ifs.read(header.data(), (std::streamsize)hlen)) {
        if (err != nullptr) {
            *err = "failed to read npy header";
        }
        return false;
    }

    if (header.find("fortran_order") == std::string::npos || header.find("False") == std::string::npos) {
        if (err != nullptr) {
            *err = "fortran-order npy is not supported";
        }
        return false;
    }

    if (header.find("'descr': '<i4'") != std::string::npos || header.find("\"descr\": \"<i4\"") != std::string::npos) {
        *dtype = NPY_DTYPE_I32;
    } else if (header.find("'descr': '<f4'") != std::string::npos || header.find("\"descr\": \"<f4\"") != std::string::npos) {
        *dtype = NPY_DTYPE_F32;
    } else {
        if (err != nullptr) {
            *err = "only little-endian int32/float32 npy is supported";
        }
        return false;
    }

    return parse_shape_dims(header, shape, err);
}

static bool load_npy(const char * path, struct npy_array * out, std::string * err) {
    if (path == nullptr || out == nullptr) {
        if (err != nullptr) {
            *err = "invalid npy load arguments";
        }
        return false;
    }

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (err != nullptr) {
            *err = "failed to open npy file";
        }
        return false;
    }

    out->dtype = NPY_DTYPE_UNKNOWN;
    out->shape.clear();
    out->i32.clear();
    out->f32.clear();

    if (!parse_npy_header(ifs, &out->dtype, &out->shape, err)) {
        return false;
    }

    size_t n_elem = 1;
    for (size_t i = 0; i < out->shape.size(); ++i) {
        if (out->shape[i] <= 0) {
            if (err != nullptr) {
                *err = "npy dimensions must be positive";
            }
            return false;
        }
        if (n_elem > SIZE_MAX / (size_t)out->shape[i]) {
            if (err != nullptr) {
                *err = "npy element count overflow";
            }
            return false;
        }
        n_elem *= (size_t)out->shape[i];
    }

    if (out->dtype == NPY_DTYPE_I32) {
        out->i32.resize(n_elem);
        if (!ifs.read(reinterpret_cast<char *>(out->i32.data()), (std::streamsize)(n_elem * sizeof(int32_t)))) {
            if (err != nullptr) {
                *err = "failed to read npy int32 payload";
            }
            return false;
        }
    } else if (out->dtype == NPY_DTYPE_F32) {
        out->f32.resize(n_elem);
        if (!ifs.read(reinterpret_cast<char *>(out->f32.data()), (std::streamsize)(n_elem * sizeof(float)))) {
            if (err != nullptr) {
                *err = "failed to read npy float32 payload";
            }
            return false;
        }
    } else {
        if (err != nullptr) {
            *err = "unsupported npy dtype";
        }
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
    struct npy_array arr;
    const bool codes_mode = args.codes_path != nullptr;
    const char * input_path = codes_mode ? args.codes_path : args.latent_path;
    if (!load_npy(input_path, &arr, &npy_err)) {
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

    if (codes_mode && arr.dtype != NPY_DTYPE_I32) {
        std::fprintf(stderr, "codes mode requires int32 npy\n");
        codec_free(ctx);
        codec_model_free(model);
        return 6;
    }
    if (!codes_mode && arr.dtype != NPY_DTYPE_F32) {
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
        if (!write_wav_pcm16(args.out_path, pcm.data, pcm.n_samples, pcm.sample_rate, &wav_err)) {
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
        if (!write_wav_pcm16(out_wav.c_str(), pcm[(size_t)i].data, pcm[(size_t)i].n_samples, pcm[(size_t)i].sample_rate, &wav_err)) {
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
