// Drives the codec_common continuous-latent path end-to-end on a few dummy
// backbone hidden states: observe_hidden → feedback loop → decode_audio.
// Validates the BlueMagpie integration into codec_common (audio-output path).
#include "codec_common.h"
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    using namespace codec_common;
    const char * gguf = argc > 1 ? argv[1] : "models/bluemagpie/bluemagpie.gguf";

    audio_lm_params p;
    p.codec_path = gguf;
    std::string err;
    audio_lm_context * ctx = audio_lm_init(p, &err);
    if (!ctx) { std::printf("init failed: %s\n", err.c_str()); return 1; }

    std::printf("is_continuous = %d\n", (int) audio_lm_is_continuous(ctx));
    std::printf("hidden_dim    = %d\n", audio_lm_hidden_dim(ctx));
    std::printf("modality      = 0x%x\n", audio_lm_modality(ctx));
    audio_lm_set_continuous_params(ctx, 2.0f, 8);

    const int hd = audio_lm_hidden_dim(ctx);
    std::vector<float> hidden((size_t) hd, 0.0f);
    std::vector<float> noise(4 * 64, 0.0f);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = 0.01f * (float) i - 1.0f;

    int steps = 0;
    const int max_steps = 5;
    for (; steps < max_steps; ++steps) {
        observe_action a = audio_lm_observe_hidden(ctx, hidden.data(), hd, noise.data());
        if (a == OBSERVE_STOP) { std::printf("step %d: STOP\n", steps); break; }
        if (a == OBSERVE_CONSUMED_EMBED) {
            int32_t dim = 0;
            const float * fb = audio_lm_get_next_embed(ctx, &dim);
            if (!fb || dim != hd) { std::printf("step %d: bad feedback\n", steps); return 1; }
            for (int i = 0; i < hd; ++i) hidden[i] = fb[i];   // feed feedback as next hidden
            std::printf("step %d: CONSUMED_EMBED (fb dim=%d)\n", steps, dim);
        } else {
            std::printf("step %d: unexpected action %d\n", steps, (int) a); return 1;
        }
    }

    audio_lm_audio_output out;
    if (!audio_lm_decode_audio(ctx, &out)) {
        std::printf("decode_audio failed: %s\n", audio_lm_last_error(ctx));
        return 1;
    }
    std::printf("decoded: %zu samples @ %d Hz, %d ch  (from %d steps)\n",
                out.pcm.size(), out.sample_rate, out.n_channels, steps + (steps < max_steps ? 1 : 0));
    audio_lm_free(ctx);
    std::printf("CODEC_COMMON CONTINUOUS PATH OK\n");
    return 0;
}
