#pragma once

// Output marshalling for tts-server: resample to the response rate and encode
// to an OpenAI `response_format`, whole or as a stream of chunks.  wav / pcm
// are always available; mp3, opus, aac and flac need their encoder library at
// build time (CODEC_TTS_SERVER_MP3 / _OPUS / _AAC / _FLAC).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Mono F32 resampler with the torchaudio sinc_interp_hann kernel
// (lowpass_filter_width 6, rolloff 0.99): the same filter as the in-graph
// codec_op_resample_sinc_hann.  Fed in pieces, it gives the same samples as
// resampling the whole signal at once.
class codec_example_resampler {
public:
    codec_example_resampler(int32_t in_sr, int32_t out_sr);
    void push(const float * x, size_t n, std::vector<float> * out);
    void finish(std::vector<float> * out);

private:
    void run(size_t n_out, std::vector<float> * out);

    int64_t orig_ = 1, new_ = 1, width_ = 0, klen_ = 0;
    std::vector<float> kernel_;
    std::vector<float> in_;     // every input sample so far
    size_t produced_ = 0;
};

std::vector<float> codec_example_resample_sinc_hann(const std::vector<float> & in, int32_t in_sr, int32_t out_sr);

// Interleaved multi-channel → mono (channel average).
std::vector<float> codec_example_downmix_mono(const float * in, size_t n_samples, int32_t n_channels);

// The formats this build can encode, in OpenAI's order.
std::vector<std::string> codec_example_audio_formats();

// True when this build can encode `format`; else *err says why.
bool codec_example_audio_format_ok(const std::string & format, std::string * err);

// Mono F32 → encoded bytes, a piece at a time.  push / finish append the
// bytes ready so far to *out.
class codec_example_audio_encoder {
public:
    virtual ~codec_example_audio_encoder() = default;
    virtual bool push(const float * pcm, size_t n, std::string * out, std::string * err) = 0;
    virtual bool finish(std::string * out, std::string * err) = 0;
    const std::string & content_type() const { return content_type_; }

    // total_samples < 0: length unknown up front (streaming; a wav header then
    // carries the "unknown size" 0xFFFFFFFF).
    static std::unique_ptr<codec_example_audio_encoder> create(const std::string & format, int32_t sample_rate,
                                                               int64_t total_samples, std::string * err);

protected:
    std::string content_type_;
};

// Whole-signal encode.
bool codec_example_encode_audio(const std::string & format, const std::vector<float> & pcm, int32_t sample_rate,
                                std::string * out, std::string * content_type, std::string * err);
