#ifndef CODEC_RUNTIME_AUDIO_DSP_H
#define CODEC_RUNTIME_AUDIO_DSP_H

#include <cstdint>
#include <string>
#include <vector>

// Inverse short-time Fourier transform from a Vocos/ISTFTHead-style head tensor.
// `head` is laid out as `[out_dim, n_frames]` (column-major in time): for each
// time frame, the first `n_bins = out_dim/2` values are log-magnitudes and the
// next `n_bins` values are phases. `n_fft = 2 * (n_bins - 1)`. If `window` is
// non-null and matches `n_fft`, it is used; otherwise a symmetric Hann window
// is computed. `skip_dc_nyquist` controls bin handling (Soprano zeros DC/Nyquist;
// Vocos-style includes them). `trim_pad_override` overrides the default trim:
// pass -1 to use the default (`n_fft/2` if `skip_dc_nyquist`, otherwise
// `(n_fft - hop)/2`); pass `n_fft/2` for HiFi-GAN-style `center=True` iSTFT.
bool codec_runtime_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    const std::vector<float> * window,
    bool skip_dc_nyquist,
    int32_t trim_pad_override,
    std::vector<float> * out_pcm,
    std::string * err);

// Periodic Hann window of length `n_fft` (matches scipy.get_window("hann", n,
// fftbins=True)). Used by HiFi-GAN-style STFT pipelines.
void codec_runtime_periodic_hann_window(int32_t n_fft, std::vector<float> * out);

// STFT analysis basis kernels with the periodic Hann window pre-multiplied.
// Both outputs are laid out as ggml conv1d weights with ne = (n_fft, 1, n_bins):
//   re_kernel[k, 0, n] =  hann[n] * cos(2π k n / n_fft)
//   im_kernel[k, 0, n] = -hann[n] * sin(2π k n / n_fft)
// where `n_bins = n_fft/2 + 1`. Conv1d(signal, re_kernel) yields the real part,
// Conv1d(signal, im_kernel) the imaginary part.
void codec_runtime_stft_basis_kernels(
    int32_t n_fft,
    std::vector<float> * out_re_kernel,
    std::vector<float> * out_im_kernel,
    std::vector<float> * out_hann);

// iSTFT synthesis basis matrices, window pre-multiplied. Used by the
// matmul + ConvTranspose1d-OLA in-graph iSTFT path. Layout matches
// `codec_op_linear` weights — ggml ne[0]=n_bins, ne[1]=n_fft. Mid bins carry
// the ×2 conjugate-symmetry factor (HiFi-GAN style: includes DC and Nyquist).
//   re_basis[n_bins, n] = hann[n] * coef_re(k_bin, n)
//   im_basis[n_bins, n] = hann[n] * coef_im(k_bin, n)
// where coef_re/im are the standard inverse-DFT weights with k=0 → 1, k=N/2 →
// (-1)^n, mid → 2cos / 2sin.
void codec_runtime_istft_synthesis_basis(
    int32_t n_fft,
    const std::vector<float> & hann,
    std::vector<float> * out_re,
    std::vector<float> * out_im);

// OLA identity kernel for ConvTranspose1d-based overlap-add. Output layout:
// ggml ne = (k=n_fft, out=1, in=n_fft) with `weight[k=i, 0, in=i] = 1` and
// zeros elsewhere. ConvTranspose1d with this kernel and stride=hop scatters
// each input frame back to a length-`n_fft` slot offset by `hop`.
void codec_runtime_ola_identity_kernel(int32_t n_fft, std::vector<float> * out);

#endif
