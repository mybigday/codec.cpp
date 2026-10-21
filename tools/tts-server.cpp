// tts-server — OpenAI-compatible speech endpoint over the TTS runner.
//
//   tts-server --model CODEC.gguf [--backbone LLAMA.gguf] [--host H] [--port P]
//
// One codec (+ backbone) pair is loaded at start-up and serves
//
//   POST /v1/audio/speech   {model, input, voice?, instructions?, response_format, speed?}
//   GET  /v1/audio/voices   the names `voice` accepts
//   GET  /v1/models
//   GET  /health
//
// Differences from OpenAI's API: `response_format` is required (mp3 / opus /
// aac / flac only when built with their encoder), `voice` names a speaker of
// the loaded model (omitted = the model default), `speed` must be 1.0 and
// `instructions` is taken only by models with an instruct mode (CosyVoice).
// Audio is resampled to 24 kHz mono as OpenAI's formats specify.
//
// Responses stream: the input is synthesized a segment at a time
// (--segment-bytes) and, on models that support it, each segment is decoded
// in chunks of about --stream-chunk-ms while it generates.  The body is sent
// with chunked transfer encoding, or as `speech.audio.delta` /
// `speech.audio.done` events with `stream_format: "sse"`.  An error after the
// first chunk ends the stream early (an `error` event under SSE).
//
// TODO: requests run one at a time behind a mutex; llama-server style
// parallel slots (--parallel N) would serve several at once.

#include "codec_common.h"
#include "tts_runner.h"
#include "audio_encode.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace {

constexpr int32_t k_out_sample_rate = 24000;
constexpr size_t  k_max_input_chars = 4096;   // OpenAI's input limit

struct args {
    std::string model;
    std::string backbone;
    std::string speaker_file;
    std::string voice_dir;      // custom voices: stored here (*.gguf + *.json), served by id
    bool        allow_voice_create = false;   // POST /v1/audio/voices (needs voice_dir)
    std::string model_id;       // `id` reported by /v1/models
    std::string host = "127.0.0.1";
    int32_t     port = 8080;
    std::string api_key;
    int32_t     n_threads  = 0;
    bool        use_gpu    = false;
    int32_t     max_frames = 0;
    int32_t     segment_bytes   = 160;    // ~10 s of speech per segment
    int32_t     stream_chunk_ms = 1000;
    std::vector<codec_model_kv_override> kv_overrides;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model CODEC.gguf [--backbone LLAMA.gguf] [--speaker-file SPK.gguf]\n"
        "          [--voice-dir DIR [--allow-voice-create]]\n"
        "          [--host 127.0.0.1] [--port 8080] [--api-key KEY] [--model-id NAME]\n"
        "          [--gpu] [--n-threads N] [--max-frames N] [--override-kv KEY=TYPE:VALUE]\n"
        "          [--segment-bytes N (160; 0 = whole input)] [--stream-chunk-ms N (1000; 0 = per segment)]\n"
        "response formats in this build:", argv0);
    for (const auto & f : codec_example_audio_formats()) std::fprintf(stderr, " %s", f.c_str());
    std::fprintf(stderr, "\n");
}

bool parse_args(int argc, char ** argv, args * out) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&]() -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); return nullptr; }
            return argv[++i];
        };
        const char * v = nullptr;
        if      (a == "--model")        { if (!(v = need())) return false; out->model = v; }
        else if (a == "--backbone")     { if (!(v = need())) return false; out->backbone = v; }
        else if (a == "--speaker-file") { if (!(v = need())) return false; out->speaker_file = v; }
        else if (a == "--model-id")     { if (!(v = need())) return false; out->model_id = v; }
        else if (a == "--voice-dir")    { if (!(v = need())) return false; out->voice_dir = v; }
        else if (a == "--allow-voice-create") { out->allow_voice_create = true; }
        else if (a == "--host")         { if (!(v = need())) return false; out->host = v; }
        else if (a == "--port")         { if (!(v = need())) return false; out->port = std::atoi(v); }
        else if (a == "--api-key")      { if (!(v = need())) return false; out->api_key = v; }
        else if (a == "--n-threads")    { if (!(v = need())) return false; out->n_threads = std::atoi(v); }
        else if (a == "--max-frames")   { if (!(v = need())) return false; out->max_frames = std::atoi(v); }
        else if (a == "--segment-bytes")   { if (!(v = need())) return false; out->segment_bytes = std::atoi(v); }
        else if (a == "--stream-chunk-ms") { if (!(v = need())) return false; out->stream_chunk_ms = std::atoi(v); }
        else if (a == "--gpu")          { out->use_gpu = true; }
        else if (a == "--override-kv") {
            if (!(v = need())) return false;
            codec_model_kv_override ov;
            if (!codec_common::parse_kv_override(v, &ov)) {
                std::fprintf(stderr, "invalid --override-kv '%s' (want KEY=int|float|bool|str:VALUE)\n", v);
                return false;
            }
            out->kv_overrides.push_back(ov);
        }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return false; }
    }
    if (out->model_id.empty() && !out->model.empty()) {
        std::string base = out->model.substr(out->model.find_last_of("/\\") + 1);
        if (base.size() > 5 && base.compare(base.size() - 5, 5, ".gguf") == 0) base.resize(base.size() - 5);
        out->model_id = base;
    }
    if (out->allow_voice_create && out->voice_dir.empty()) {
        std::fprintf(stderr, "--allow-voice-create needs --voice-dir\n");
        return false;
    }
    return !out->model.empty();
}

// Custom voices (OpenAI custom voices: POST /v1/audio/voices, then
// "voice": {"id": ...}): codec speakers cloned from an uploaded sample, saved
// as <dir>/<id>.gguf (the voice) + <dir>/<id>.json (name, created_at) and
// loaded back at start.
struct custom_voice {
    std::string     id, name;
    int64_t         created_at = 0;
    codec_speaker * spk = nullptr;
};

struct voice_store {
    std::map<std::string, custom_voice> voices;
    ~voice_store() { for (auto & kv : voices) codec_speaker_free(kv.second.spk); }

    json describe(const custom_voice & v) const {
        return { { "object", "audio.voice" }, { "id", v.id }, { "name", v.name }, { "created_at", v.created_at } };
    }
    void load(codec_audio_lm * alm, const std::string & dir) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
            if (e.path().extension() != ".json") continue;
            std::ifstream f(e.path());
            const json meta = json::parse(f, nullptr, /*allow_exceptions=*/false);
            if (!meta.is_object() || !meta.contains("id") || !meta["id"].is_string()) continue;
            custom_voice v;
            v.id = meta["id"].get<std::string>();
            v.name = meta.value("name", v.id);
            v.created_at = meta.value("created_at", (int64_t) 0);
            const std::string gguf = (std::filesystem::path(dir) / (v.id + ".gguf")).string();
            v.spk = codec_audio_lm_load_voice(alm, gguf.c_str(), nullptr);
            if (v.spk == nullptr) {
                std::fprintf(stderr, "voice %s skipped: %s\n", v.id.c_str(), codec_audio_lm_last_error(alm));
                continue;
            }
            voices[v.id] = v;
        }
    }
    const codec_speaker * find(const std::string & id) const {
        const auto it = voices.find(id);
        return it == voices.end() ? nullptr : it->second.spk;
    }
};

std::string new_voice_id() {
    static std::mt19937_64 rng(std::random_device{}() ^ (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count());
    char buf[32];
    std::snprintf(buf, sizeof buf, "voice_%016llx", (unsigned long long) rng());
    return buf;
}

json error_json(const std::string & message, const char * type, const char * param) {
    json err = { { "message", message }, { "type", type }, { "param", nullptr }, { "code", nullptr } };
    if (param != nullptr) err["param"] = param;
    return err;
}

void send_error(httplib::Response & res, int status, const std::string & message,
                const char * type = "invalid_request_error", const char * param = nullptr) {
    res.status = status;
    res.set_content(json{ { "error", error_json(message, type, param) } }.dump(), "application/json");
}

std::string base64(const std::string & in) {
    static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    for (size_t i = 0; i < in.size(); i += 3) {
        const uint32_t b0 = (unsigned char) in[i];
        const uint32_t b1 = i + 1 < in.size() ? (unsigned char) in[i + 1] : 0;
        const uint32_t b2 = i + 2 < in.size() ? (unsigned char) in[i + 2] : 0;
        const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(i + 1 < in.size() ? tbl[(v >> 6) & 63] : '=');
        out.push_back(i + 2 < in.size() ? tbl[v & 63] : '=');
    }
    return out;
}


// Encoded audio handed from a synthesis worker to the HTTP response.
class speech_stream {
public:
    enum class item { bytes, done, failed };

    std::thread worker;

    // Worker side.  put returns false once the client is gone.
    bool put(std::string bytes) {
        std::lock_guard<std::mutex> lock(m_);
        if (cancelled_) return false;
        if (!bytes.empty()) queue_.push_back(std::move(bytes));
        cv_.notify_all();
        return true;
    }
    void finish(bool ok, const std::string & err) {
        std::lock_guard<std::mutex> lock(m_);
        finished_ = true;
        ok_ = ok;
        error_ = err;
        cv_.notify_all();
    }

    // Response side.  wait_first: true once bytes are queued, or the run
    // ended well; false when it failed before producing any.
    bool wait_first() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return !queue_.empty() || finished_; });
        return !queue_.empty() || ok_;
    }
    item take(std::string * bytes) {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return !queue_.empty() || finished_; });
        if (!queue_.empty()) {
            *bytes = std::move(queue_.front());
            queue_.pop_front();
            return item::bytes;
        }
        return ok_ ? item::done : item::failed;
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(m_);
        cancelled_ = true;
    }
    bool cancelled() {
        std::lock_guard<std::mutex> lock(m_);
        return cancelled_;
    }
    std::string error() {
        std::lock_guard<std::mutex> lock(m_);
        return error_;
    }

private:
    std::mutex              m_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    bool                    finished_  = false;
    bool                    ok_        = false;
    bool                    cancelled_ = false;
    std::string             error_;
};

}  // namespace

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, &a)) { print_usage(argv[0]); return 1; }
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // runner progress lines show up as they happen in a log file
    if (a.segment_bytes <= 0 && a.stream_chunk_ms > 0) {
        std::fprintf(stderr, "warning: --segment-bytes 0 with frame streaming re-decodes the whole growing clip "
                             "every chunk; long inputs get slow\n");
    }

    codec_common::tts_runner_params lp;
    lp.codec_path    = a.model;
    lp.backbone_path = a.backbone;
    lp.kv_overrides  = a.kv_overrides;
    lp.n_threads     = a.n_threads;
    lp.use_gpu       = a.use_gpu;
    lp.max_frames    = a.max_frames;
    lp.segment_bytes   = a.segment_bytes;
    lp.stream_chunk_ms = a.stream_chunk_ms;

    std::string err;
    codec_common::tts_runner_session * session = codec_common::tts_runner_load(lp, &err);
    if (session == nullptr) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const codec_common::tts_runner_info info = codec_common::tts_runner_get_info(session, a.speaker_file);
    if (!info.error.empty()) {
        std::fprintf(stderr, "%s\n", info.error.c_str());
        codec_common::tts_runner_free(session);
        return 1;
    }
    std::mutex run_mutex;
    voice_store custom;
    if (!a.voice_dir.empty()) {
        custom.load(codec_common::tts_runner_audio_lm(session), a.voice_dir);
        std::fprintf(stderr, "tts-server: %zu custom voices in %s%s\n", custom.voices.size(), a.voice_dir.c_str(),
                     a.allow_voice_create ? " (creation enabled)" : "");
    }

    httplib::Server svr;
    svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (a.api_key.empty() || req.path == "/health") return httplib::Server::HandlerResponse::Unhandled;
        if (req.get_header_value("Authorization") != "Bearer " + a.api_key) {
            send_error(res, 401, "invalid API key", "authentication_error");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        const json m = { { "id", a.model_id }, { "object", "model" }, { "created", 0 }, { "owned_by", "codec.cpp" } };
        res.set_content(json{ { "object", "list" }, { "data", json::array({ m }) } }.dump(), "application/json");
    });

    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & res) {
        json j = { { "voices", info.voices } };
        j["default"] = info.default_voice.empty() ? json(nullptr) : json(info.default_voice);
        json list = json::array();
        {
            std::lock_guard<std::mutex> lock(run_mutex);
            for (const auto & kv : custom.voices) list.push_back(custom.describe(kv.second));
        }
        j["custom"] = list;
        res.set_content(j.dump(), "application/json");
    });

    // Create a voice from a recording: multipart `name` + `audio_sample`
    // (WAV, at most 30 s); `ref_text`, the sample's transcript, is a
    // codec.cpp extension models that prompt with it use. OpenAI's `consent`
    // is not checked here.
    svr.Post("/v1/audio/voices", [&](const httplib::Request & req, httplib::Response & res) {
        if (a.voice_dir.empty() || !a.allow_voice_create) {
            send_error(res, 403, "voice creation is disabled (start the server with --voice-dir DIR --allow-voice-create)");
            return;
        }
        if (!req.is_multipart_form_data() || !req.form.has_field("name") || !req.form.has_file("audio_sample")) {
            send_error(res, 400, "multipart 'name' and 'audio_sample' are required");
            return;
        }
        const std::string name = req.form.get_field("name");
        const std::string ref_text = req.form.has_field("ref_text") ? req.form.get_field("ref_text") : "";
        const std::string id = new_voice_id();
        const std::filesystem::path dir(a.voice_dir);
        const std::string upload = (dir / ("." + id + ".wav")).string();
        {
            std::ofstream f(upload, std::ios::binary);
            const std::string & bytes = req.form.get_file("audio_sample").content;
            f.write(bytes.data(), (std::streamsize) bytes.size());
        }
        std::vector<float> pcm;
        int32_t n = 0, sr = 0;
        std::string err;
        const bool read = codec_common::load_ref_audio(upload, pcm, &n, &sr, &err);
        std::error_code ec;
        std::filesystem::remove(upload, ec);
        if (!read || n <= 0) { send_error(res, 400, "audio_sample is not a PCM16 WAV file", "invalid_request_error", "audio_sample"); return; }
        if (n > (int64_t) 30 * sr) { send_error(res, 400, "audio_sample is longer than 30 seconds", "invalid_request_error", "audio_sample"); return; }

        std::lock_guard<std::mutex> lock(run_mutex);
        codec_audio_lm * alm = codec_common::tts_runner_audio_lm(session);
        custom_voice v;
        v.id = id;
        v.name = name;
        v.created_at = (int64_t) std::time(nullptr);
        v.spk = codec_audio_lm_create_voice(alm, pcm.data(), n, sr, ref_text.empty() ? nullptr : ref_text.c_str());
        if (v.spk == nullptr) { send_error(res, 400, codec_audio_lm_last_error(alm), "invalid_request_error", "audio_sample"); return; }
        if (!codec_audio_lm_save_voice(alm, v.spk, (dir / (id + ".gguf")).string().c_str(), id.c_str())) {
            codec_speaker_free(v.spk);
            send_error(res, 500, codec_audio_lm_last_error(alm), "server_error");
            return;
        }
        std::ofstream((dir / (id + ".json")).string()) << json{ { "id", id }, { "name", name }, { "created_at", v.created_at } }.dump();
        custom.voices[id] = v;
        std::fprintf(stderr, "voice %s (%s): %.1fs reference\n", id.c_str(), name.c_str(), (double) n / sr);
        res.set_content(custom.describe(v).dump(), "application/json");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (!body.is_object()) { send_error(res, 400, "request body is not a JSON object"); return; }
        auto str_field = [&](const char * key, std::string * out) -> bool {
            if (!body.contains(key) || body[key].is_null()) return true;
            if (!body[key].is_string()) { send_error(res, 400, std::string("'") + key + "' must be a string", "invalid_request_error", key); return false; }
            *out = body[key].get<std::string>();
            return true;
        };
        std::string model, input, voice, instructions, format, stream_format;
        const codec_speaker * custom_spk = nullptr;
        if (body.contains("voice") && body["voice"].is_object()) {
            // A custom voice: {"id": "voice_..."}.
            const json & vid = body["voice"]["id"];
            {
                std::lock_guard<std::mutex> lock(run_mutex);
                custom_spk = vid.is_string() ? custom.find(vid.get<std::string>()) : nullptr;
            }
            if (custom_spk == nullptr) {
                send_error(res, 400, "unknown custom voice id", "invalid_request_error", "voice");
                return;
            }
            body.erase("voice");
        }
        if (!str_field("model", &model) || !str_field("input", &input) || !str_field("voice", &voice) ||
            !str_field("instructions", &instructions) || !str_field("response_format", &format) ||
            !str_field("stream_format", &stream_format)) {
            return;
        }
        if (input.empty()) { send_error(res, 400, "'input' is required", "invalid_request_error", "input"); return; }
        size_t n_chars = 0;   // UTF-8 code points
        for (unsigned char c : input) n_chars += (c & 0xC0) != 0x80;
        if (n_chars > k_max_input_chars) {
            send_error(res, 400, "'input' exceeds 4096 characters", "invalid_request_error", "input");
            return;
        }
        const std::vector<std::string> formats = codec_example_audio_formats();
        if (format.empty()) {
            std::string list;
            for (const auto & f : formats) list += (list.empty() ? "" : ", ") + f;
            send_error(res, 400, "'response_format' is required (this server encodes: " + list + ")",
                       "invalid_request_error", "response_format");
            return;
        }
        if (!stream_format.empty() && stream_format != "audio" && stream_format != "sse") {
            send_error(res, 400, "stream_format '" + stream_format + "' is not supported",
                       "invalid_request_error", "stream_format");
            return;
        }
        if (body.contains("speed") && !body["speed"].is_null()) {
            if (!body["speed"].is_number() || body["speed"].get<double>() != 1.0) {
                send_error(res, 400, "'speed' other than 1.0 is not supported", "invalid_request_error", "speed");
                return;
            }
        }
        if (!instructions.empty() && !info.takes_instruct) {
            send_error(res, 400, "this model takes no 'instructions'", "invalid_request_error", "instructions");
            return;
        }
        if (!voice.empty() && custom_spk == nullptr) {
            std::lock_guard<std::mutex> lock(run_mutex);
            custom_spk = custom.find(voice);   // a custom voice's id given as a string
            if (custom_spk != nullptr) voice.clear();
        }
        if (!voice.empty()) {
            bool known = false;
            for (const auto & v : info.voices) known = known || v == voice;
            if (!known) {
                std::string list;
                for (const auto & v : info.voices) list += (list.empty() ? "" : ", ") + v;
                send_error(res, 400, "unknown voice '" + voice + "' (available: " + (list.empty() ? "none" : list) + ")",
                           "invalid_request_error", "voice");
                return;
            }
        }
        // Reject a format this build lacks before spending the synthesis on it.
        {
            std::string ferr;
            if (!codec_example_audio_format_ok(format, &ferr)) {
                send_error(res, 400, ferr, "invalid_request_error", "response_format");
                return;
            }
        }

        std::string enc_err;
        std::unique_ptr<codec_example_audio_encoder> enc =
            codec_example_audio_encoder::create(format, k_out_sample_rate, /*total_samples=*/-1, &enc_err);
        if (!enc) { send_error(res, 500, enc_err, "server_error"); return; }
        const std::string content_type = enc->content_type();

        codec_common::tts_runner_params rp = lp;
        rp.text         = input;
        rp.voice        = voice;
        rp.speaker_file = custom_spk ? "" : a.speaker_file;
        rp.voice_spk    = custom_spk;
        rp.instruct     = instructions;
        const bool sse  = stream_format == "sse";

        // Synthesis runs on a worker that queues the encoded bytes.  The
        // handler waits for the first bytes (or the failure) before choosing
        // the response, so a run that fails before any audio still gets an
        // error status, not a truncated 200.
        auto st = std::make_shared<speech_stream>();
        st->worker = std::thread([&, st, rp, enc = std::move(enc), format, n_chars, sse]() mutable {
            std::lock_guard<std::mutex> run_lock(run_mutex);
            std::unique_ptr<codec_example_resampler> rs;
            size_t n_out = 0;
            std::string err;
            auto encode = [&](const std::vector<float> & pcm) {
                n_out += pcm.size();
                std::string bytes;
                if (!enc->push(pcm.data(), pcm.size(), &bytes, &err)) return false;
                return st->put(std::move(bytes));
            };
            rp.on_audio = [&](const float * pcm, int32_t n, int32_t sr, int32_t nch) {
                if (!rs) rs = std::make_unique<codec_example_resampler>(sr, k_out_sample_rate);
                const std::vector<float> mono = codec_common::downmix_mono(pcm, (size_t) n, nch);
                std::vector<float> out;
                rs->push(mono.data(), mono.size(), &out);
                return encode(out);
            };
            codec_common::tts_runner_result r;
            bool ok = codec_common::tts_runner_run(session, rp, &r);
            if (!ok) err = r.error;
            if (ok && rs) {
                std::vector<float> out;
                rs->finish(&out);
                ok = encode(out);
            }
            std::string tail;
            ok = ok && enc->finish(&tail, &err) && st->put(std::move(tail));
            if (ok) {
                std::fprintf(stderr, "speech: %zu chars -> %.2fs %s%s (%d frames, stop=%s, %.0f ms)\n",
                             n_chars, (double) n_out / k_out_sample_rate, format.c_str(), sse ? " sse" : "",
                             r.n_frames, r.stop_reason.c_str(), r.total_ms);
            } else {
                std::fprintf(stderr, "speech failed: %s\n", st->cancelled() ? "client disconnected" : err.c_str());
            }
            st->finish(ok, err);
        });

        if (!st->wait_first()) {
            st->worker.join();
            send_error(res, 500, st->error(), "server_error");
            return;
        }
        res.set_chunked_content_provider(sse ? "text/event-stream" : content_type,
            [st, sse](size_t, httplib::DataSink & sink) {
                std::string bytes;
                for (;;) {
                    const speech_stream::item it = st->take(&bytes);
                    if (it == speech_stream::item::bytes) {
                        const std::string data = sse
                            ? "data: " + json{ { "type", "speech.audio.delta" }, { "audio", base64(bytes) } }.dump() + "\n\n"
                            : bytes;
                        if (!sink.write(data.data(), data.size())) { st->cancel(); return false; }
                        continue;
                    }
                    if (it == speech_stream::item::done) {
                        if (sse) {
                            const std::string ev = "data: " + json{ { "type", "speech.audio.done" } }.dump() + "\n\n";
                            sink.write(ev.data(), ev.size());
                        }
                        sink.done();
                        return true;
                    }
                    // Failed after audio started: an error event under SSE, else a cut-off body.
                    if (!sse) return false;
                    const std::string ev = "data: " + json{ { "type", "error" },
                                                           { "error", error_json(st->error(), "server_error", nullptr) } }.dump() + "\n\n";
                    sink.write(ev.data(), ev.size());
                    sink.done();
                    return true;
                }
            },
            [st](bool) {
                st->cancel();
                if (st->worker.joinable()) st->worker.join();
            });
    });

    std::fprintf(stderr, "tts-server: %s on http://%s:%d (%zu voices)\n",
                 a.model_id.c_str(), a.host.c_str(), a.port, info.voices.size());
    const bool listened = svr.listen(a.host, a.port);
    codec_common::tts_runner_free(session);
    if (!listened) {
        std::fprintf(stderr, "failed to listen on %s:%d\n", a.host.c_str(), a.port);
        return 1;
    }
    return 0;
}
