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


bool codec_runtime_w2v_bert_features(
    const std::vector<float> & pcm,
    const std::vector<float> & mel_filters,
    int32_t n_freq,
    int32_t n_mels,
    const std::vector<float> & window,
    int32_t n_fft,
    int32_t win,
    int32_t hop,
    float preemphasis,
    float mel_floor,
    int32_t stride,
    std::vector<float> * out_features,
    int32_t * out_n_frames,
    std::string * err) {

    if (out_features == nullptr || out_n_frames == nullptr) {
        if (err != nullptr) *err = "null output";
        return false;
    }
    if (n_fft <= 0 || win <= 0 || hop <= 0 || n_mels <= 0 || stride <= 0) {
        if (err != nullptr) *err = "invalid mel-fbank arguments";
        return false;
    }
    if ((int32_t) window.size() != win) {
        if (err != nullptr) *err = "window size mismatch";
        return false;
    }
    if (n_freq != n_fft / 2 + 1 ||
        (int32_t) mel_filters.size() != n_freq * n_mels) {
        if (err != nullptr) *err = "mel filter shape mismatch";
        return false;
    }
    const int64_t n = (int64_t) pcm.size();
    if (n < win) {
        if (err != nullptr) *err = "input shorter than win";
        return false;
    }

    const int32_t n_frames = (int32_t) ((n - win) / hop + 1);
    if (n_frames <= 0) {
        if (err != nullptr) *err = "no frames";
        return false;
    }

    // Compute log-mel features per frame.  Matches transformers' reference
    // exactly (kaldi-compliance scale 2^15, per-frame DC remove, preemphasis,
    // window, FFT, |X|^2 mel, log(max(., mel_floor))).
    const float pi = 3.14159265358979323846f;
    std::vector<float> log_mel((size_t) n_frames * (size_t) n_mels, 0.0f);
    std::vector<double> buffer((size_t) n_fft, 0.0);
    std::vector<double> re_v((size_t) n_freq, 0.0);
    std::vector<double> im_v((size_t) n_freq, 0.0);

    // Precompute DFT basis (real + imag) at double precision for parity.
    std::vector<double> dft_cos((size_t) n_freq * (size_t) n_fft, 0.0);
    std::vector<double> dft_sin((size_t) n_freq * (size_t) n_fft, 0.0);
    for (int32_t k = 0; k < n_freq; ++k) {
        for (int32_t m = 0; m < n_fft; ++m) {
            const double ang = -2.0 * pi * (double) k * (double) m / (double) n_fft;
            dft_cos[(size_t) k * (size_t) n_fft + (size_t) m] = std::cos(ang);
            dft_sin[(size_t) k * (size_t) n_fft + (size_t) m] = std::sin(ang);
        }
    }

    for (int32_t ti = 0; ti < n_frames; ++ti) {
        // 1. Extract frame (Kaldi-compliance: scale by 2^15) into a double buffer.
        const int64_t off = (int64_t) ti * hop;
        std::fill(buffer.begin(), buffer.end(), 0.0);
        for (int32_t k = 0; k < win; ++k) {
            buffer[(size_t) k] = (double) pcm[(size_t) (off + k)] * 32768.0;
        }
        // 2. Remove DC offset (subtract mean over the win samples).
        double mean = 0.0;
        for (int32_t k = 0; k < win; ++k) mean += buffer[(size_t) k];
        mean /= (double) win;
        for (int32_t k = 0; k < win; ++k) buffer[(size_t) k] -= mean;
        // 3. Pre-emphasis applied IN-FRAME (note: must go from k=win-1 down to k=1
        //    to avoid clobbering buffer[k-1] before it's used).
        for (int32_t k = win - 1; k >= 1; --k) {
            buffer[(size_t) k] -= (double) preemphasis * buffer[(size_t) (k - 1)];
        }
        buffer[0] *= (double) (1.0f - preemphasis);
        // 4. Window.
        for (int32_t k = 0; k < win; ++k) buffer[(size_t) k] *= (double) window[(size_t) k];

        // 5. DFT (zero-padded to n_fft, but win <= n_fft so trailing slots are 0).
        for (int32_t k = 0; k < n_freq; ++k) {
            double re = 0.0, im = 0.0;
            const double * cos_row = &dft_cos[(size_t) k * (size_t) n_fft];
            const double * sin_row = &dft_sin[(size_t) k * (size_t) n_fft];
            for (int32_t m = 0; m < n_fft; ++m) {
                re += buffer[(size_t) m] * cos_row[(size_t) m];
                im += buffer[(size_t) m] * sin_row[(size_t) m];
            }
            re_v[(size_t) k] = re;
            im_v[(size_t) k] = im;
        }
        // 6. Power spectrogram |X|^2 then mel matmul.
        for (int32_t mi = 0; mi < n_mels; ++mi) {
            double acc = 0.0;
            for (int32_t k = 0; k < n_freq; ++k) {
                const double power = re_v[(size_t) k] * re_v[(size_t) k] +
                                     im_v[(size_t) k] * im_v[(size_t) k];
                acc += power * (double) mel_filters[(size_t) k * (size_t) n_mels + (size_t) mi];
            }
            // 7. log(max(., mel_floor)).
            if (acc < (double) mel_floor) acc = (double) mel_floor;
            log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi] = (float) std::log(acc);
        }
    }

    // 8. Per-mel-bin (time) zero-mean unit-variance normalize.  ddof=1 sample
    //    variance to match torch/numpy reference.
    if (n_frames > 1) {
        for (int32_t mi = 0; mi < n_mels; ++mi) {
            double sum = 0.0;
            for (int32_t ti = 0; ti < n_frames; ++ti) {
                sum += (double) log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi];
            }
            const double m = sum / (double) n_frames;
            double var = 0.0;
            for (int32_t ti = 0; ti < n_frames; ++ti) {
                const double d = (double) log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi] - m;
                var += d * d;
            }
            var /= (double) (n_frames - 1);  // ddof=1
            const double s = 1.0 / std::sqrt(var + 1e-7);
            for (int32_t ti = 0; ti < n_frames; ++ti) {
                const float x = log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi];
                log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi] = (float) (((double) x - m) * s);
            }
        }
    }

    // 9. Stride-2 stacking: drop trailing remainder, reshape (T, n_mels) →
    //    (T/stride, n_mels * stride).  Memory layout is identical (contiguous
    //    row-major), so we just truncate and reinterpret the buffer.
    const int32_t remainder = n_frames % stride;
    const int32_t n_frames_kept = n_frames - remainder;
    const int32_t n_frames_out = n_frames_kept / stride;
    const int32_t out_dim = n_mels * stride;

    out_features->assign((size_t) n_frames_out * (size_t) out_dim, 0.0f);
    for (int32_t ti = 0; ti < n_frames_out; ++ti) {
        for (int32_t s = 0; s < stride; ++s) {
            for (int32_t mi = 0; mi < n_mels; ++mi) {
                (*out_features)[(size_t) ti * (size_t) out_dim + (size_t) (s * n_mels + mi)] =
                    log_mel[(size_t) (ti * stride + s) * (size_t) n_mels + (size_t) mi];
            }
        }
    }
    *out_n_frames = n_frames_out;
    (void) err;
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

