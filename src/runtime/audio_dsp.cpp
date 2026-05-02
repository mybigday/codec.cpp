#include "audio_dsp.h"

#include <algorithm>
#include <cmath>

bool codec_runtime_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    const std::vector<float> * window,
    bool skip_dc_nyquist,
    int32_t trim_pad_override,
    std::vector<float> * out_pcm,
    std::string * err) {

    if (out_pcm == nullptr || out_dim <= 0 || n_frames <= 0 || hop <= 0 || (out_dim % 2) != 0) {
        if (err != nullptr) {
            *err = "invalid ISTFT arguments";
        }
        return false;
    }
    const int32_t n_bins = out_dim / 2;
    const int32_t n_fft = 2 * (n_bins - 1);
    if (n_fft <= 0) {
        if (err != nullptr) {
            *err = "invalid ISTFT head output dimension";
        }
        return false;
    }
    const float pi = 3.14159265358979323846f;

    std::vector<float> win((size_t) n_fft, 0.0f);
    if (window != nullptr && (int32_t) window->size() == n_fft) {
        win = *window;
    } else {
        for (int32_t n = 0; n < n_fft; ++n) {
            win[(size_t) n] = 0.5f - 0.5f * std::cos(2.0f * pi * (float) n / (float) (n_fft - 1));
        }
    }

    const int32_t pad = trim_pad_override >= 0
        ? trim_pad_override
        : (skip_dc_nyquist ? (n_fft / 2) : ((n_fft - hop) / 2));
    const int32_t out_size = (n_frames - 1) * hop + n_fft;
    std::vector<float> y((size_t) out_size, 0.0f);
    std::vector<float> env((size_t) out_size, 0.0f);
    std::vector<float> frame((size_t) n_fft, 0.0f);

    for (int32_t ti = 0; ti < n_frames; ++ti) {
        for (int32_t n = 0; n < n_fft; ++n) {
            float sum = 0.0f;
            if (!skip_dc_nyquist) {
                float mag0 = std::exp(head[(size_t) 0 + (size_t) out_dim * (size_t) ti]);
                if (mag0 > 1e2f) mag0 = 1e2f;
                const float re0 = mag0 * std::cos(head[(size_t) n_bins + (size_t) out_dim * (size_t) ti]);
                sum += re0;
                float magn = std::exp(head[(size_t) (n_bins - 1) + (size_t) out_dim * (size_t) ti]);
                if (magn > 1e2f) magn = 1e2f;
                const float ren = magn * std::cos(head[(size_t) (2 * n_bins - 1) + (size_t) out_dim * (size_t) ti]);
                sum += ren * ((n & 1) ? -1.0f : 1.0f);
            }
            for (int32_t k = 1; k < n_bins - 1; ++k) {
                float mag = std::exp(head[(size_t) k + (size_t) out_dim * (size_t) ti]);
                if (mag > 1e2f) mag = 1e2f;
                const float ph = head[(size_t) (n_bins + k) + (size_t) out_dim * (size_t) ti];
                const float re = mag * std::cos(ph);
                const float im = mag * std::sin(ph);
                const float ang = 2.0f * pi * (float) k * (float) n / (float) n_fft;
                sum += 2.0f * (re * std::cos(ang) - im * std::sin(ang));
            }
            frame[(size_t) n] = (sum / (float) n_fft) * win[(size_t) n];
        }
        const int32_t off = ti * hop;
        for (int32_t n = 0; n < n_fft; ++n) {
            y[(size_t) (off + n)] += frame[(size_t) n];
            env[(size_t) (off + n)] += win[(size_t) n] * win[(size_t) n];
        }
    }

    const int32_t out_begin = std::max(0, pad);
    const int32_t out_end = std::max(out_begin, out_size - pad);
    out_pcm->assign((size_t) (out_end - out_begin), 0.0f);
    for (int32_t i = out_begin; i < out_end; ++i) {
        const float den = env[(size_t) i] > 1e-11f ? env[(size_t) i] : 1.0f;
        (*out_pcm)[(size_t) (i - out_begin)] = y[(size_t) i] / den;
    }
    return true;
}


void codec_runtime_periodic_hann_window(int32_t n_fft, std::vector<float> * out) {
    if (out == nullptr || n_fft <= 0) return;
    out->assign((size_t) n_fft, 0.0f);
    for (int32_t n = 0; n < n_fft; ++n) {
        (*out)[(size_t) n] = 0.5f - 0.5f * std::cos(2.0f * (float) M_PI * (float) n / (float) n_fft);
    }
}

void codec_runtime_stft_basis_kernels(
    int32_t n_fft,
    std::vector<float> * out_re_kernel,
    std::vector<float> * out_im_kernel,
    std::vector<float> * out_hann) {
    if (out_re_kernel == nullptr || out_im_kernel == nullptr || n_fft <= 0) return;
    const int32_t n_bins = n_fft / 2 + 1;
    std::vector<float> hann;
    codec_runtime_periodic_hann_window(n_fft, &hann);
    if (out_hann != nullptr) *out_hann = hann;

    out_re_kernel->assign((size_t) n_fft * (size_t) n_bins, 0.0f);
    out_im_kernel->assign((size_t) n_fft * (size_t) n_bins, 0.0f);
    // ggml conv1d weight ne[0]=k, ne[1]=in=1, ne[2]=out=n_bins → flat index = k + out*K.
    for (int32_t k_bin = 0; k_bin < n_bins; ++k_bin) {
        for (int32_t n = 0; n < n_fft; ++n) {
            const float ang = 2.0f * (float) M_PI * (float) k_bin * (float) n / (float) n_fft;
            const float w = hann[(size_t) n];
            const size_t off = (size_t) n + (size_t) k_bin * (size_t) n_fft;
            (*out_re_kernel)[off] =  w * std::cos(ang);
            (*out_im_kernel)[off] = -w * std::sin(ang);
        }
    }
}

void codec_runtime_istft_synthesis_basis(
    int32_t n_fft,
    const std::vector<float> & hann,
    std::vector<float> * out_re,
    std::vector<float> * out_im) {
    if (out_re == nullptr || out_im == nullptr || n_fft <= 0 || (int32_t) hann.size() != n_fft) return;
    const int32_t n_bins = n_fft / 2 + 1;
    out_re->assign((size_t) n_bins * (size_t) n_fft, 0.0f);
    out_im->assign((size_t) n_bins * (size_t) n_fft, 0.0f);
    // ggml ne[0]=n_bins, ne[1]=n_fft → flat index = k_bin + n*n_bins.
    for (int32_t n = 0; n < n_fft; ++n) {
        for (int32_t k_bin = 0; k_bin < n_bins; ++k_bin) {
            const float ang = 2.0f * (float) M_PI * (float) k_bin * (float) n / (float) n_fft;
            float coef_re = 0.0f, coef_im = 0.0f;
            if (k_bin == 0) {
                coef_re = 1.0f;
            } else if (k_bin == n_bins - 1) {
                coef_re = (n & 1) ? -1.0f : 1.0f;
            } else {
                coef_re = 2.0f * std::cos(ang);
                coef_im = 2.0f * std::sin(ang);
            }
            const size_t off = (size_t) k_bin + (size_t) n * (size_t) n_bins;
            (*out_re)[off] = coef_re * hann[(size_t) n];
            (*out_im)[off] = coef_im * hann[(size_t) n];
        }
    }
}

void codec_runtime_ola_identity_kernel(int32_t n_fft, std::vector<float> * out) {
    if (out == nullptr || n_fft <= 0) return;
    // ggml convtr1d weight ne = (k, out=1, in) → flat index = k + out*K + in*K (out=1).
    out->assign((size_t) n_fft * (size_t) n_fft, 0.0f);
    for (int32_t i = 0; i < n_fft; ++i) {
        (*out)[(size_t) i + (size_t) i * (size_t) n_fft] = 1.0f;
    }
}

