#include "audio_encode.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <numeric>

#ifdef CODEC_TTS_SERVER_MP3
#include <lame/lame.h>
#endif
#ifdef CODEC_TTS_SERVER_OPUS
#include <ogg/ogg.h>
#include <opus/opus.h>
#endif
#ifdef CODEC_TTS_SERVER_AAC
#include <fdk-aac/aacenc_lib.h>
#endif
#ifdef CODEC_TTS_SERVER_FLAC
#include <FLAC/stream_encoder.h>
#endif

// ─── Resampler ──────────────────────────────────────────────────────────────

codec_example_resampler::codec_example_resampler(int32_t in_sr, int32_t out_sr) {
    if (in_sr == out_sr) return;
    const int64_t g = std::gcd(in_sr, out_sr);
    orig_ = in_sr / g;
    new_  = out_sr / g;
    const double lpw       = 6.0;
    const double base_freq = (double) std::min(orig_, new_) * 0.99;
    width_ = (int64_t) std::ceil(lpw * (double) orig_ / base_freq);
    klen_  = 2 * width_ + orig_;

    // kernel[i][k], k over idx = (k - width) / orig (torchaudio _get_sinc_resample_kernel).
    kernel_.resize((size_t) (new_ * klen_));
    const double scale = base_freq / (double) orig_;
    for (int64_t i = 0; i < new_; ++i) {
        for (int64_t k = 0; k < klen_; ++k) {
            double t = (-(double) i / (double) new_ + (double) (k - width_) / (double) orig_) * base_freq;
            t = std::clamp(t, -lpw, lpw);
            const double w = std::pow(std::cos(t * M_PI / lpw / 2.0), 2.0);
            t *= M_PI;
            kernel_[(size_t) (i * klen_ + k)] = (float) ((t == 0.0 ? 1.0 : std::sin(t) / t) * w * scale);
        }
    }
}

// Output sample o is block o / new, phase o % new: a correlation over the
// input padded by (width, width + orig) zeros, at stride orig.
void codec_example_resampler::run(size_t n_out, std::vector<float> * out) {
    const int64_t n_in = (int64_t) in_.size();
    for (; produced_ < n_out; ++produced_) {
        const int64_t j = (int64_t) produced_ / new_, i = (int64_t) produced_ % new_;
        const float * kr = kernel_.data() + i * klen_;
        double acc = 0.0;
        for (int64_t k = 0; k < klen_; ++k) {
            const int64_t src = j * orig_ + k - width_;
            if (src >= 0 && src < n_in) acc += (double) in_[(size_t) src] * kr[k];
        }
        out->push_back((float) acc);
    }
}

void codec_example_resampler::push(const float * x, size_t n, std::vector<float> * out) {
    if (kernel_.empty()) { out->insert(out->end(), x, x + n); return; }
    in_.insert(in_.end(), x, x + n);
    // Outputs whose whole kernel window is already in hand.
    const int64_t n_in = (int64_t) in_.size();
    const int64_t blocks = n_in + width_ - klen_ >= 0 ? (n_in + width_ - klen_) / orig_ + 1 : 0;
    run((size_t) std::max<int64_t>(0, blocks * new_), out);
}

void codec_example_resampler::finish(std::vector<float> * out) {
    if (kernel_.empty()) return;
    run((size_t) ((new_ * (int64_t) in_.size() + orig_ - 1) / orig_), out);   // ceil(new * n / orig)
}

std::vector<float> codec_example_downmix_mono(const float * in, size_t n_samples, int32_t n_channels) {
    if (n_channels <= 1) return std::vector<float>(in, in + n_samples);
    std::vector<float> out(n_samples);
    for (size_t t = 0; t < n_samples; ++t) {
        float s = 0.0f;
        for (int32_t c = 0; c < n_channels; ++c) s += in[t * (size_t) n_channels + (size_t) c];
        out[t] = s / (float) n_channels;
    }
    return out;
}

// ─── Encoders ───────────────────────────────────────────────────────────────

namespace {

std::vector<int16_t> to_s16(const float * pcm, size_t n) {
    std::vector<int16_t> s(n);
    for (size_t i = 0; i < n; ++i) s[i] = (int16_t) std::lrint(std::clamp(pcm[i], -1.0f, 1.0f) * 32767.0f);
    return s;
}

void put_u32(std::string * o, uint32_t v) { for (int i = 0; i < 4; ++i) o->push_back((char) ((v >> (8 * i)) & 0xff)); }
void put_u16(std::string * o, uint16_t v) { o->push_back((char) (v & 0xff)); o->push_back((char) (v >> 8)); }

class pcm_encoder : public codec_example_audio_encoder {
public:
    pcm_encoder() { content_type_ = "audio/pcm"; }
    bool push(const float * pcm, size_t n, std::string * out, std::string *) override {
        const std::vector<int16_t> s = to_s16(pcm, n);
        out->append(reinterpret_cast<const char *>(s.data()), s.size() * sizeof(int16_t));
        return true;
    }
    bool finish(std::string *, std::string *) override { return true; }
};

class wav_encoder : public pcm_encoder {
public:
    wav_encoder(int32_t sr, int64_t total) : sr_(sr), total_(total) { content_type_ = "audio/wav"; }
    bool push(const float * pcm, size_t n, std::string * out, std::string * err) override {
        header(out);
        return pcm_encoder::push(pcm, n, out, err);
    }
    bool finish(std::string * out, std::string *) override { header(out); return true; }

private:
    void header(std::string * out) {
        if (written_) return;
        written_ = true;
        const uint32_t data_bytes = total_ < 0 ? 0xFFFFFFFFu : (uint32_t) (total_ * 2);
        out->append("RIFF");  put_u32(out, total_ < 0 ? 0xFFFFFFFFu : 36 + data_bytes);
        out->append("WAVEfmt ");
        put_u32(out, 16); put_u16(out, 1); put_u16(out, 1);
        put_u32(out, (uint32_t) sr_); put_u32(out, (uint32_t) sr_ * 2); put_u16(out, 2); put_u16(out, 16);
        out->append("data"); put_u32(out, data_bytes);
    }
    int32_t sr_;
    int64_t total_;
    bool    written_ = false;
};

#ifdef CODEC_TTS_SERVER_MP3
class mp3_encoder : public codec_example_audio_encoder {
public:
    explicit mp3_encoder(lame_global_flags * lf) : lf_(lf) { content_type_ = "audio/mpeg"; }
    ~mp3_encoder() override { lame_close(lf_); }
    static std::unique_ptr<codec_example_audio_encoder> create(int32_t sr, std::string * err) {
        lame_global_flags * lf = lame_init();
        if (lf == nullptr) { *err = "lame_init failed"; return nullptr; }
        lame_set_in_samplerate(lf, sr);
        lame_set_out_samplerate(lf, sr);
        lame_set_num_channels(lf, 1);
        lame_set_mode(lf, MONO);
        lame_set_brate(lf, 64);
        lame_set_quality(lf, 2);
        if (lame_init_params(lf) < 0) { lame_close(lf); *err = "lame_init_params failed"; return nullptr; }
        return std::make_unique<mp3_encoder>(lf);
    }
    bool push(const float * pcm, size_t n, std::string * out, std::string * err) override {
        for (size_t off = 0; off < n; off += k_chunk) {
            const int m = (int) std::min(k_chunk, n - off);
            const int w = lame_encode_buffer_ieee_float(lf_, pcm + off, nullptr, m, buf_.data(), (int) buf_.size());
            if (w < 0) { *err = "lame encode failed"; return false; }
            out->append(reinterpret_cast<const char *>(buf_.data()), (size_t) w);
        }
        return true;
    }
    bool finish(std::string * out, std::string *) override {
        const int w = lame_encode_flush(lf_, buf_.data(), (int) buf_.size());
        if (w > 0) out->append(reinterpret_cast<const char *>(buf_.data()), (size_t) w);
        return true;
    }

private:
    static constexpr size_t k_chunk = 8192;
    lame_global_flags * lf_;
    std::vector<unsigned char> buf_ = std::vector<unsigned char>((size_t) (1.25 * k_chunk + 7200));
};
#endif

#ifdef CODEC_TTS_SERVER_OPUS
// Ogg Opus (RFC 7845), as OpenAI's `opus` response.  Granule positions are in
// 48 kHz units whatever the input rate; every push flushes its pages.
class opus_encoder : public codec_example_audio_encoder {
public:
    opus_encoder(OpusEncoder * enc, int32_t sr) : enc_(enc), sr_(sr) {
        content_type_ = "audio/ogg";
        opus_encoder_ctl(enc_, OPUS_GET_LOOKAHEAD(&lookahead_));
        to48_     = 48000 / sr_;
        pre_skip_ = lookahead_ * to48_;
        frame_    = sr_ / 50;   // 20 ms
        ogg_stream_init(&os_, 0x636f6463);   // "codc"
    }
    ~opus_encoder() override { ogg_stream_clear(&os_); opus_encoder_destroy(enc_); }
    static std::unique_ptr<codec_example_audio_encoder> create(int32_t sr, std::string * err) {
        int oerr = 0;
        OpusEncoder * enc = opus_encoder_create(sr, 1, OPUS_APPLICATION_AUDIO, &oerr);
        if (enc == nullptr) { *err = std::string("opus_encoder_create: ") + opus_strerror(oerr); return nullptr; }
        return std::make_unique<opus_encoder>(enc, sr);
    }
    bool push(const float * pcm, size_t n, std::string * out, std::string * err) override {
        headers(out);
        n_in_ += (int64_t) n;
        buf_.insert(buf_.end(), pcm, pcm + n);
        size_t off = 0;
        for (; buf_.size() - off >= (size_t) frame_; off += frame_) {
            if (!frame(buf_.data() + off, false, err)) return false;
        }
        buf_.erase(buf_.begin(), buf_.begin() + (ptrdiff_t) off);
        pages(out, true);
        return true;
    }
    bool finish(std::string * out, std::string * err) override {
        headers(out);
        // Encode until the decoder's output (minus pre-skip) covers every input sample.
        do {
            buf_.resize((size_t) frame_, 0.0f);
            const bool last = (n_frames_ + 1) * frame_ >= n_in_ + lookahead_;
            if (!frame(buf_.data(), last, err)) return false;
            buf_.clear();
            if (last) break;
        } while (true);
        pages(out, true);
        return true;
    }

private:
    void packet(const unsigned char * d, long n, int64_t granule, bool bos, bool eos) {
        ogg_packet op;
        op.packet = const_cast<unsigned char *>(d);
        op.bytes = n; op.b_o_s = bos; op.e_o_s = eos; op.granulepos = granule; op.packetno = packetno_++;
        ogg_stream_packetin(&os_, &op);
    }
    void pages(std::string * out, bool force) {
        ogg_page pg;
        while (force ? ogg_stream_flush(&os_, &pg) : ogg_stream_pageout(&os_, &pg)) {
            out->append(reinterpret_cast<const char *>(pg.header), (size_t) pg.header_len);
            out->append(reinterpret_cast<const char *>(pg.body), (size_t) pg.body_len);
        }
    }
    void headers(std::string * out) {
        if (headers_done_) return;
        headers_done_ = true;
        std::string head = "OpusHead";
        head.push_back(1); head.push_back(1);
        put_u16(&head, (uint16_t) pre_skip_); put_u32(&head, (uint32_t) sr_); put_u16(&head, 0); head.push_back(0);
        packet(reinterpret_cast<const unsigned char *>(head.data()), (long) head.size(), 0, true, false);
        pages(out, true);
        std::string tags = "OpusTags";
        const std::string vendor = "codec.cpp";
        put_u32(&tags, (uint32_t) vendor.size()); tags += vendor; put_u32(&tags, 0);
        packet(reinterpret_cast<const unsigned char *>(tags.data()), (long) tags.size(), 0, false, false);
        pages(out, true);
    }
    bool frame(const float * x, bool last, std::string * err) {
        const opus_int32 nb = opus_encode_float(enc_, x, frame_, pkt_.data(), (opus_int32) pkt_.size());
        if (nb < 0) { *err = std::string("opus_encode_float: ") + opus_strerror(nb); return false; }
        ++n_frames_;
        const int64_t granule = last ? pre_skip_ + n_in_ * to48_ : n_frames_ * frame_ * to48_;
        packet(pkt_.data(), nb, granule, false, last);
        return true;
    }

    OpusEncoder * enc_;
    ogg_stream_state os_;
    int32_t sr_;
    opus_int32 lookahead_ = 0;
    int64_t to48_ = 2, pre_skip_ = 0, frame_ = 480, n_in_ = 0, n_frames_ = 0, packetno_ = 0;
    bool headers_done_ = false;
    std::vector<float> buf_;
    std::vector<unsigned char> pkt_ = std::vector<unsigned char>(4000);
};
#endif

#ifdef CODEC_TTS_SERVER_AAC
// AAC-LC in ADTS, as OpenAI's `aac` response.
class aac_encoder : public codec_example_audio_encoder {
public:
    explicit aac_encoder(HANDLE_AACENCODER enc) : enc_(enc) { content_type_ = "audio/aac"; }
    ~aac_encoder() override { aacEncClose(&enc_); }
    static std::unique_ptr<codec_example_audio_encoder> create(int32_t sr, std::string * err) {
        HANDLE_AACENCODER enc = nullptr;
        if (aacEncOpen(&enc, 0, 1) != AACENC_OK) { *err = "aacEncOpen failed"; return nullptr; }
        if (aacEncoder_SetParam(enc, AACENC_AOT, AOT_AAC_LC) != AACENC_OK ||
            aacEncoder_SetParam(enc, AACENC_SAMPLERATE, (UINT) sr) != AACENC_OK ||
            aacEncoder_SetParam(enc, AACENC_CHANNELMODE, MODE_1) != AACENC_OK ||
            aacEncoder_SetParam(enc, AACENC_BITRATE, 64000) != AACENC_OK ||
            aacEncoder_SetParam(enc, AACENC_TRANSMUX, TT_MP4_ADTS) != AACENC_OK ||
            aacEncoder_SetParam(enc, AACENC_AFTERBURNER, 1) != AACENC_OK ||
            aacEncEncode(enc, nullptr, nullptr, nullptr, nullptr) != AACENC_OK) {
            aacEncClose(&enc);
            *err = "fdk-aac encoder setup failed";
            return nullptr;
        }
        return std::make_unique<aac_encoder>(enc);
    }
    bool push(const float * pcm, size_t n, std::string * out, std::string * err) override {
        const std::vector<int16_t> s = to_s16(pcm, n);
        for (size_t off = 0; off < s.size();) {
            INT consumed = 0;
            if (!encode(s.data() + off, (INT) (s.size() - off), &consumed, out, err)) return false;
            off += (size_t) consumed;
        }
        return true;
    }
    bool finish(std::string * out, std::string * err) override {
        for (;;) {
            INT consumed = 0;
            const AACENC_ERROR e = encode_raw(nullptr, -1, &consumed, out);
            if (e == AACENC_ENCODE_EOF) return true;
            if (e != AACENC_OK) { *err = "aacEncEncode failed"; return false; }
        }
    }

private:
    bool encode(const int16_t * x, INT n, INT * consumed, std::string * out, std::string * err) {
        if (encode_raw(x, n, consumed, out) != AACENC_OK) { *err = "aacEncEncode failed"; return false; }
        return true;
    }
    AACENC_ERROR encode_raw(const int16_t * x, INT n, INT * consumed, std::string * out) {
        void * in_ptr = (void *) x;
        INT in_id = IN_AUDIO_DATA, in_size = n > 0 ? n * (INT) sizeof(int16_t) : 0, in_el = sizeof(int16_t);
        AACENC_BufDesc ib = {};
        ib.numBufs = 1; ib.bufs = &in_ptr; ib.bufferIdentifiers = &in_id; ib.bufSizes = &in_size; ib.bufElSizes = &in_el;
        void * out_ptr = obuf_.data();
        INT out_id = OUT_BITSTREAM_DATA, out_size = (INT) obuf_.size(), out_el = 1;
        AACENC_BufDesc ob = {};
        ob.numBufs = 1; ob.bufs = &out_ptr; ob.bufferIdentifiers = &out_id; ob.bufSizes = &out_size; ob.bufElSizes = &out_el;
        AACENC_InArgs ia = {};
        ia.numInSamples = n;
        AACENC_OutArgs oa = {};
        const AACENC_ERROR e = aacEncEncode(enc_, &ib, &ob, &ia, &oa);
        if (e == AACENC_OK && oa.numOutBytes > 0) out->append(reinterpret_cast<const char *>(obuf_.data()), (size_t) oa.numOutBytes);
        *consumed = oa.numInSamples;
        return e;
    }

    HANDLE_AACENCODER enc_;
    std::vector<uint8_t> obuf_ = std::vector<uint8_t>(8192);
};
#endif

#ifdef CODEC_TTS_SERVER_FLAC
class flac_encoder : public codec_example_audio_encoder {
public:
    explicit flac_encoder(FLAC__StreamEncoder * enc) : enc_(enc) { content_type_ = "audio/flac"; }
    ~flac_encoder() override { FLAC__stream_encoder_delete(enc_); }
    static std::unique_ptr<codec_example_audio_encoder> create(int32_t sr, int64_t total, std::string * err) {
        FLAC__StreamEncoder * enc = FLAC__stream_encoder_new();
        if (enc == nullptr) { *err = "FLAC__stream_encoder_new failed"; return nullptr; }
        auto e = std::make_unique<flac_encoder>(enc);
        FLAC__stream_encoder_set_channels(enc, 1);
        FLAC__stream_encoder_set_bits_per_sample(enc, 16);
        FLAC__stream_encoder_set_sample_rate(enc, (uint32_t) sr);
        FLAC__stream_encoder_set_compression_level(enc, 5);
        // No seek callback: STREAMINFO is written once, so give it the length
        // when it is known (0 = unknown).
        FLAC__stream_encoder_set_total_samples_estimate(enc, total > 0 ? (FLAC__uint64) total : 0);
        if (FLAC__stream_encoder_init_stream(enc, write_cb, nullptr, nullptr, nullptr, e.get()) !=
            FLAC__STREAM_ENCODER_INIT_STATUS_OK) {
            *err = "FLAC__stream_encoder_init_stream failed";
            return nullptr;
        }
        return e;
    }
    bool push(const float * pcm, size_t n, std::string * out, std::string * err) override {
        sink_ = out;
        std::vector<FLAC__int32> buf(n);
        const std::vector<int16_t> s = to_s16(pcm, n);
        for (size_t i = 0; i < n; ++i) buf[i] = s[i];
        const bool ok = n == 0 || FLAC__stream_encoder_process_interleaved(enc_, buf.data(), (uint32_t) n);
        if (!ok) *err = "FLAC encode failed";
        return ok;
    }
    bool finish(std::string * out, std::string * err) override {
        sink_ = out;
        const bool ok = FLAC__stream_encoder_finish(enc_);
        if (!ok) *err = "FLAC encode failed";
        return ok;
    }

private:
    static FLAC__StreamEncoderWriteStatus write_cb(const FLAC__StreamEncoder *, const FLAC__byte buffer[], size_t bytes,
                                                   uint32_t, uint32_t, void * client) {
        auto * self = static_cast<flac_encoder *>(client);
        if (self->sink_ != nullptr) self->sink_->append(reinterpret_cast<const char *>(buffer), bytes);
        return FLAC__STREAM_ENCODER_WRITE_STATUS_OK;
    }
    FLAC__StreamEncoder * enc_;
    std::string * sink_ = nullptr;
};
#endif

struct format_entry {
    const char * name;
    bool         built;
};

const format_entry k_formats[] = {
#ifdef CODEC_TTS_SERVER_MP3
    { "mp3",  true },
#else
    { "mp3",  false },
#endif
#ifdef CODEC_TTS_SERVER_OPUS
    { "opus", true },
#else
    { "opus", false },
#endif
#ifdef CODEC_TTS_SERVER_AAC
    { "aac",  true },
#else
    { "aac",  false },
#endif
#ifdef CODEC_TTS_SERVER_FLAC
    { "flac", true },
#else
    { "flac", false },
#endif
    { "wav",  true },
    { "pcm",  true },
};

}  // namespace

std::vector<std::string> codec_example_audio_formats() {
    std::vector<std::string> v;
    for (const auto & f : k_formats) if (f.built) v.emplace_back(f.name);
    return v;
}

bool codec_example_audio_format_ok(const std::string & format, std::string * err) {
    for (const auto & f : k_formats) {
        if (format != f.name) continue;
        if (f.built) return true;
        std::string opt = format;
        for (char & c : opt) c = (char) std::toupper((unsigned char) c);
        *err = "response_format '" + format + "' is not built into this server (rebuild with -DCODEC_TTS_SERVER_" +
               opt + "=ON)";
        return false;
    }
    *err = "unknown response_format '" + format + "'";
    return false;
}

std::unique_ptr<codec_example_audio_encoder> codec_example_audio_encoder::create(
        const std::string & format, int32_t sample_rate, int64_t total_samples, std::string * err) {
    if (!codec_example_audio_format_ok(format, err)) return nullptr;
    if (format == "wav") return std::make_unique<wav_encoder>(sample_rate, total_samples);
    if (format == "pcm") return std::make_unique<pcm_encoder>();
#ifdef CODEC_TTS_SERVER_MP3
    if (format == "mp3") return mp3_encoder::create(sample_rate, err);
#endif
#ifdef CODEC_TTS_SERVER_OPUS
    if (format == "opus") return opus_encoder::create(sample_rate, err);
#endif
#ifdef CODEC_TTS_SERVER_AAC
    if (format == "aac") return aac_encoder::create(sample_rate, err);
#endif
#ifdef CODEC_TTS_SERVER_FLAC
    if (format == "flac") return flac_encoder::create(sample_rate, total_samples, err);
#endif
    *err = "response_format '" + format + "' has no encoder";
    return nullptr;
}

