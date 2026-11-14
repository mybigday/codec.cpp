// MOSS-TTS-Nano prompt helpers (lm_models.h): the prompt template's token ids
// and the tokenizer's text normalizer, from `codec.lm.moss_tts_nano.*`.
//
// The reference tokenizer is SentencePiece BPE with the nmt_nfkc normalizer
// (precompiled charsmap) and remove_extra_whitespaces.  llama.cpp's SPM vocab
// has no normalizer, so the family normalizes the text first; llama.cpp then
// adds the space prefix and merges like SentencePiece.

#include "lm_internal.h"

#include "../runtime/gguf_kv.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

struct moss_nano_state {
    codec_lm_moss_nano_info info = {};
    std::vector<int32_t> user_prefix, no_reference, after_reference, assistant_prefix;
    std::vector<uint8_t> charsmap;
    const uint32_t * xcda = nullptr;   // XOR-compressed double array (SentencePiece's darts)
    size_t           n_xcda = 0;
    const char *     repl = nullptr;   // NUL-terminated replacement strings
    size_t           n_repl = 0;
};

// The darts lookup of SentencePiece's precompiled charsmap (as llama.cpp's UGM
// tokenizer reads it): the longest input prefix with a replacement.
struct xcda_view {
    const uint32_t * a; size_t n;
    uint32_t node(size_t i) const { return i < n ? a[i] : 0; }
    uint32_t base(size_t i) const { const uint32_t p = node(i); return (p >> 10) << ((p & (1u << 9)) >> 6); }
    uint32_t lcheck(size_t i) const { return node(i) & ((1u << 31) | 0xff); }
    bool     leaf(size_t i) const { return (node(i) >> 8) & 1; }
    uint32_t value(size_t i) const { return node(i) & ((1u << 31) - 1); }
};

size_t utf8_len(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 1;
}

moss_nano_state * get_state(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr || lm->codec->gguf == nullptr) return nullptr;
    if (lm->moss_nano != nullptr) return static_cast<moss_nano_state *>(lm->moss_nano);
    gguf_context * gf = lm->codec->gguf;
    if (gguf_find_key(gf, "codec.lm.moss_tts_nano.user_prefix") < 0) return nullptr;
    auto * s = new moss_nano_state();
    codec_read_i32_array_kv_vec(gf, "codec.lm.moss_tts_nano.user_prefix", &s->user_prefix);
    codec_read_i32_array_kv_vec(gf, "codec.lm.moss_tts_nano.no_reference", &s->no_reference);
    codec_read_i32_array_kv_vec(gf, "codec.lm.moss_tts_nano.after_reference", &s->after_reference);
    codec_read_i32_array_kv_vec(gf, "codec.lm.moss_tts_nano.assistant_prefix", &s->assistant_prefix);
    const int64_t kc = gguf_find_key(gf, "codec.lm.moss_tts_nano.text_norm_charsmap");
    if (kc >= 0 && gguf_get_kv_type(gf, kc) == GGUF_TYPE_ARRAY &&
        (gguf_get_arr_type(gf, kc) == GGUF_TYPE_UINT8 || gguf_get_arr_type(gf, kc) == GGUF_TYPE_INT8)) {
        const auto * p = static_cast<const uint8_t *>(gguf_get_arr_data(gf, kc));
        s->charsmap.assign(p, p + gguf_get_arr_n(gf, kc));
        uint32_t blob = 0;
        if (s->charsmap.size() > 4) std::memcpy(&blob, s->charsmap.data(), 4);   // little-endian, as written
        if (blob > 0 && (size_t) blob + 4 < s->charsmap.size()) {
            s->xcda   = reinterpret_cast<const uint32_t *>(s->charsmap.data() + 4);
            s->n_xcda = blob / 4;
            s->repl   = reinterpret_cast<const char *>(s->charsmap.data() + 4 + blob);
            s->n_repl = s->charsmap.size() - 4 - blob;
        }
    }
    codec_lm_moss_nano_info & i = s->info;
    i.hidden_dim       = lm->info.hidden_dim;
    i.n_codebook       = lm->info.n_codebook;
    i.audio_start      = codec_read_i32_kv(gf, "codec.lm.moss_tts_nano.audio_start", -1);
    i.audio_end        = codec_read_i32_kv(gf, "codec.lm.moss_tts_nano.audio_end", -1);
    i.user_slot        = codec_read_i32_kv(gf, "codec.lm.moss_tts_nano.audio_user_slot", -1);
    i.assistant_slot   = codec_read_i32_kv(gf, "codec.lm.moss_tts_nano.audio_assistant_slot", -1);
    i.user_prefix      = s->user_prefix.data();      i.n_user_prefix      = (int32_t) s->user_prefix.size();
    i.no_reference     = s->no_reference.data();     i.n_no_reference     = (int32_t) s->no_reference.size();
    i.after_reference  = s->after_reference.data();  i.n_after_reference  = (int32_t) s->after_reference.size();
    i.assistant_prefix = s->assistant_prefix.data(); i.n_assistant_prefix = (int32_t) s->assistant_prefix.size();
    i.has_text_norm    = s->xcda != nullptr ? 1 : 0;
    lm->moss_nano = s;
    return s;
}

}  // namespace

void codec_lm_moss_nano_free_state(struct codec_lm * lm) {
    if (lm == nullptr || lm->moss_nano == nullptr) return;
    delete static_cast<moss_nano_state *>(lm->moss_nano);
    lm->moss_nano = nullptr;
}

const struct codec_lm_moss_nano_info * codec_lm_moss_nano_get_info(struct codec_lm * lm) {
    moss_nano_state * s = get_state(lm);
    return s != nullptr ? &s->info : nullptr;
}

int32_t codec_lm_moss_nano_normalize_text(struct codec_lm * lm, const char * text, char * out, int32_t cap) {
    moss_nano_state * s = get_state(lm);
    if (s == nullptr || text == nullptr) return -1;
    // 1. charsmap (NFKC + the NMT rules: control characters dropped, every
    //    whitespace character to ' '), longest match first; other UTF-8
    //    characters pass through.
    const std::string in(text);
    std::string norm;
    norm.reserve(in.size());
    const xcda_view xv{s->xcda, s->n_xcda};
    for (size_t off = 0; off < in.size();) {
        size_t best_len = 0, best_off = 0;
        if (s->xcda != nullptr) {
            uint32_t node = xv.base(0);
            for (size_t p = off; p < in.size(); ++p) {
                const unsigned char c = (unsigned char) in[p];
                if (c == 0) break;
                node ^= c;
                if (xv.lcheck(node) != c) break;
                const bool leaf = xv.leaf(node);
                node ^= xv.base(node);
                if (leaf) { best_len = p - off + 1; best_off = xv.value(node); }
            }
        }
        if (best_len > 0 && best_off < s->n_repl) {
            norm += std::string(s->repl + best_off, strnlen(s->repl + best_off, s->n_repl - best_off));
            off += best_len;
        } else {
            const size_t n = std::min(utf8_len((unsigned char) in[off]), in.size() - off);
            norm.append(in, off, n);
            off += n;
        }
    }
    // 2. remove_extra_whitespaces: no leading / trailing spaces, runs to one.
    std::string res;
    res.reserve(norm.size());
    for (char c : norm) {
        if (c == ' ' && (res.empty() || res.back() == ' ')) continue;
        res.push_back(c);
    }
    while (!res.empty() && res.back() == ' ') res.pop_back();
    const int32_t n = (int32_t) res.size();
    if (out != nullptr && cap > n) {
        std::memcpy(out, res.data(), (size_t) n);
        out[n] = '\0';
    }
    return n;
}
