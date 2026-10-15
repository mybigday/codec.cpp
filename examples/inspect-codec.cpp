#include "codec.h"

#include <cstdio>

static void print_optional_i32(const char * key, int32_t v) {
    if (v >= 0) {
        std::printf("  %-14s %d\n", key, v);
    } else {
        std::printf("  %-14s N/A\n", key);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    struct codec_model_params mparams = codec_model_default_params();
    mparams.use_gpu = false;

    struct codec_model * model = codec_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", argv[1]);
        return 2;
    }

    std::printf("name:       %s\n", codec_model_name(model));
    std::printf("arch:       %s\n", codec_arch_name(codec_model_arch(model)));
    std::printf("n_tensors:  %d\n", codec_model_n_tensors(model));
    std::printf("codec parameters:\n");
    std::printf("  %-14s %d\n", "sample_rate", codec_model_sample_rate(model));
    std::printf("  %-14s %s\n", "has_encoder", codec_model_has_encoder(model) ? "true" : "false");
    std::printf("  %-14s %s\n", "has_decoder", codec_model_has_decoder(model) ? "true" : "false");
    std::printf("  %-14s %d\n", "hop_size", codec_model_hop_size(model));
    std::printf("  %-14s %d\n", "n_q", codec_model_n_q(model));
    std::printf("  %-14s %d\n", "codebook_size", codec_model_codebook_size(model));
    print_optional_i32("n_fft", codec_model_n_fft(model));
    print_optional_i32("win_length", codec_model_win_length(model));
    print_optional_i32("n_mels", codec_model_n_mels(model));
    print_optional_i32("latent_dim", codec_model_latent_dim(model));

    const int32_t n_spk = codec_model_n_speakers(model);
    if (n_spk > 0) {
        const struct codec_speaker * def = codec_get_speaker(model, nullptr);
        std::printf("speakers (%d):", n_spk);
        for (int32_t i = 0; i < n_spk; ++i) {
            const char * name = codec_model_speaker_name(model, i);
            const bool is_def = def != nullptr && codec_get_speaker(model, name) == def;
            std::printf(" %s%s", name, is_def ? "*" : "");
        }
        std::printf("\n");
    }

    const struct codec_gguf_metadata * meta = codec_model_metadata(model);
    if (meta != nullptr) {
        std::printf("metadata:\n");
        for (size_t i = 0; i < meta->n_items; ++i) {
            std::printf("  %s = %s\n", meta->items[i].key, meta->items[i].value);
        }
    }

    codec_model_free(model);
    return 0;
}
