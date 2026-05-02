#ifndef CODEC_RUNTIME_AUDIO_DSP_H
#define CODEC_RUNTIME_AUDIO_DSP_H

#include <cstdint>
#include <string>
#include <vector>

// Inverse short-time Fourier transform from a Vocos/ISTFTHead-style head tensor.
// `head` is laid out as `[out_dim, n_frames]` (column-major in time): for each
// time frame, the first `n_bins = out_dim/2` values are log-magnitudes and the
// next `n_bins` values are phases. `n_fft = 2 * (n_bins - 1)`. If `window` is
// non-null and matches `n_fft`, it is used; otherwise a Hann window is computed.
//
// Two algorithm variants are supported via `skip_dc_nyquist`:
//  - false (Vocos / Wavtokenizer / NeuCodec): DC and Nyquist bins are included
//    once per frame, mid bins folded with the standard 2x conjugate-symmetry
//    weighting; output trim uses `pad = (n_fft - hop) / 2`.
//  - true  (Soprano):                          DC and Nyquist bins are zeroed,
//    output trim uses `pad = n_fft / 2`.
bool codec_runtime_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    const std::vector<float> * window,
    bool skip_dc_nyquist,
    std::vector<float> * out_pcm,
    std::string * err);

#endif
