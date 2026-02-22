#include "codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <fstream>
#include <string>
#include <vector>

struct wav_data {
    int32_t sample_rate = 0;
    int32_t n_channels = 0;
    std::vector<int16_t> pcm_i16;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s e2e --model <gguf> --in <wav> --out <wav> [--threads N] [--nq N] [--use-gpu]\n"
        "  %s decode --model <gguf> --codes <codes.npy> --out <wav> [--threads N] [--nq N] [--use-gpu]\n",
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

static bool read_exact(FILE * fp, void * ptr, size_t n) {
    return std::fread(ptr, 1, n, fp) == n;
}

static bool load_wav_pcm16(const char * path, wav_data * out, std::string * err) {
    FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        if (err != nullptr) {
            *err = "failed to open wav file";
        }
        return false;
    }

    char riff[4] = { 0 };
    uint32_t riff_size = 0;
    char wave[4] = { 0 };
    if (!read_exact(fp, riff, 4) || !read_exact(fp, &riff_size, 4) || !read_exact(fp, wave, 4)) {
        std::fclose(fp);
        if (err != nullptr) {
            *err = "invalid wav header";
        }
        return false;
    }

    (void)riff_size;
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fclose(fp);
        if (err != nullptr) {
            *err = "not a RIFF/WAVE file";
        }
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t n_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<uint8_t> pcm_bytes;

    while (true) {
        char chunk_id[4] = { 0 };
        uint32_t chunk_size = 0;
        if (!read_exact(fp, chunk_id, 4) || !read_exact(fp, &chunk_size, 4)) {
            break;
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t block_align = 0;
            uint32_t byte_rate = 0;
            if (!read_exact(fp, &audio_format, 2) ||
                !read_exact(fp, &n_channels, 2) ||
                !read_exact(fp, &sample_rate, 4) ||
                !read_exact(fp, &byte_rate, 4) ||
                !read_exact(fp, &block_align, 2) ||
                !read_exact(fp, &bits_per_sample, 2)) {
                std::fclose(fp);
                if (err != nullptr) {
                    *err = "invalid fmt chunk";
                }
                return false;
            }
            (void)byte_rate;
            (void)block_align;
            if (chunk_size > 16) {
                std::fseek(fp, (long)(chunk_size - 16), SEEK_CUR);
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            pcm_bytes.resize(chunk_size);
            if (chunk_size > 0 && !read_exact(fp, pcm_bytes.data(), chunk_size)) {
                std::fclose(fp);
                if (err != nullptr) {
                    *err = "failed to read data chunk";
                }
                return false;
            }
        } else {
            std::fseek(fp, (long)chunk_size, SEEK_CUR);
        }

        if ((chunk_size & 1) != 0) {
            std::fseek(fp, 1, SEEK_CUR);
        }
    }

    std::fclose(fp);

    if (audio_format != 1 || bits_per_sample != 16) {
        if (err != nullptr) {
            *err = "only PCM 16-bit wav is supported";
        }
        return false;
    }

    if (n_channels == 0 || sample_rate == 0 || pcm_bytes.empty()) {
        if (err != nullptr) {
            *err = "missing fmt/data chunks";
        }
        return false;
    }

    out->sample_rate = (int32_t)sample_rate;
    out->n_channels = (int32_t)n_channels;
    out->pcm_i16.resize(pcm_bytes.size() / sizeof(int16_t));
    std::memcpy(out->pcm_i16.data(), pcm_bytes.data(), pcm_bytes.size());
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

static bool parse_npy_header(std::ifstream & ifs, int32_t * n_q, int32_t * n_frames, std::string * err) {
    char magic[6] = { 0 };
    if (!ifs.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        if (err != nullptr) {
            *err = "invalid npy magic";
        }
        return false;
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    ifs.read(reinterpret_cast<char *>(&major), 1);
    ifs.read(reinterpret_cast<char *>(&minor), 1);

    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        ifs.read(reinterpret_cast<char *>(&h16), 2);
        hlen = h16;
    } else {
        ifs.read(reinterpret_cast<char *>(&hlen), 4);
    }

    std::string header(hlen, '\0');
    if (!ifs.read(header.data(), (std::streamsize)hlen)) {
        if (err != nullptr) {
            *err = "failed to read npy header";
        }
        return false;
    }

    if (header.find("'descr': '<i4'") == std::string::npos && header.find("\"descr\": \"<i4\"") == std::string::npos) {
        if (err != nullptr) {
            *err = "only little-endian int32 npy is supported";
        }
        return false;
    }
    if (header.find("False") == std::string::npos) {
        if (err != nullptr) {
            *err = "fortran-order npy is not supported";
        }
        return false;
    }

    const size_t p0 = header.find('(');
    const size_t p1 = header.find(')', p0 == std::string::npos ? 0 : p0 + 1);
    if (p0 == std::string::npos || p1 == std::string::npos || p1 <= p0 + 1) {
        if (err != nullptr) {
            *err = "invalid npy shape";
        }
        return false;
    }

    std::string shape = header.substr(p0 + 1, p1 - p0 - 1);
    int q = 0;
    int t = 0;
    if (std::sscanf(shape.c_str(), " %d , %d", &q, &t) != 2 && std::sscanf(shape.c_str(), "%d,%d", &q, &t) != 2) {
        if (err != nullptr) {
            *err = "expected 2D npy shape (n_q, n_frames)";
        }
        return false;
    }

    if (q <= 0 || t <= 0) {
        if (err != nullptr) {
            *err = "invalid npy dimensions";
        }
        return false;
    }

    *n_q = q;
    *n_frames = t;
    return true;
}

static bool load_npy_i32_2d(const char * path, std::vector<int32_t> * out, int32_t * n_q, int32_t * n_frames, std::string * err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (err != nullptr) {
            *err = "failed to open npy file";
        }
        return false;
    }

    if (!parse_npy_header(ifs, n_q, n_frames, err)) {
        return false;
    }

    const size_t n = (size_t)(*n_q) * (size_t)(*n_frames);
    std::vector<int32_t> tmp(n);
    if (!ifs.read(reinterpret_cast<char *>(tmp.data()), (std::streamsize)(n * sizeof(int32_t)))) {
        if (err != nullptr) {
            *err = "failed to read npy data";
        }
        return false;
    }

    out->assign(n, 0);
    for (int32_t q = 0; q < *n_q; ++q) {
        for (int32_t t = 0; t < *n_frames; ++t) {
            (*out)[(size_t)t * (size_t)(*n_q) + (size_t)q] = tmp[(size_t)q * (size_t)(*n_frames) + (size_t)t];
        }
    }

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

    wav_data wav;
    std::string wav_err;
    if (!load_wav_pcm16(args.input_wav, &wav, &wav_err)) {
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
    if (!write_wav_pcm16(args.out_wav, pcm.data, pcm.n_samples, pcm.sample_rate, &write_err)) {
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
    if (!load_npy_i32_2d(args.codes_npy, &tokens_data, &n_q, &n_frames, &npy_err)) {
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
    if (!write_wav_pcm16(args.out_wav, pcm.data, pcm.n_samples, pcm.sample_rate, &wav_err)) {
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

    if (std::strcmp(cmd, "decode") == 0) {
        decode_args args;
        if (!parse_decode_args(argc, argv, &args)) {
            print_usage(argv[0]);
            return 1;
        }
        return cmd_decode(args);
    }

    std::fprintf(stderr, "unknown command: %s\n", cmd);
    print_usage(argv[0]);
    return 1;
}
