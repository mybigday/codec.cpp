# codec_audio_lm — running audio LMs with a llama.cpp backbone

`include/codec_audio_lm.h` describes how to run an audio LM whose backbone is
a llama.cpp model and whose audio side is a codec.cpp GGUF. It plays the role
of llama.cpp's `common` + MTMD: the library says **what to feed the backbone,
how to sample, when to stop**, and turns the generation into audio. The host
keeps every `llama_*` call: tokenization, `llama_decode`, the sampler, the KV
cache, batching. Nothing in the API names a model; every model-specific detail
sits behind it.

Text-to-speech is the first prompt builder (`codec_audio_lm_tts_prompt`).
Other tasks (audio-to-audio) will add their own builders that produce the same
completion description, so the host loop stays the same.

## Layers

| Layer | Owns |
|---|---|
| **Host** (llama.rn, `common/tts_runner`, a Python script) | the llama.cpp model and context, tokenize / decode / sample, the loop |
| **codec_audio_lm** (`common/audio_lm_c.cpp`, in libcodec) | capabilities, voices, context requirements, segmentation and joining, prompt chunks, sampler settings, per-step verdicts, codec-side heads and their sampling, decode and streaming |
| **families** (`common/audio_lm_family_*.cpp`, internal) | one class per model family: the prompt format, conditioning, heads, stop rules |
| **codec_lm / codec** (`src/`) | the adaptor graphs (heads, embeddings, speaker encoders) and the codec decoder |

## Host loop

```c
codec_audio_lm * alm = codec_audio_lm_init(&params, err, sizeof err);

// Backbone: load the model, then size its context as the model asks.
codec_audio_lm_backbone_req req;
codec_audio_lm_backbone_requirements(alm, &vocab, max_frames, &req);
// llama_context_params: n_ctx = req.n_ctx, n_ubatch = req.n_ubatch ?: n_ctx,
// n_seq_max = req.n_seq_max, embeddings = req.embeddings (pooling none),
// flash_attn_type = req.flash_attn ? auto : disabled

// The voice: builtin, cloned, or loaded (NULL = the model default).
codec_speaker * voice = codec_audio_lm_create_voice(alm, pcm, n_samples, sample_rate, ref_text);
int n = codec_audio_lm_tts_split(alm, text, 0);      // the model's segmentation
for (int i = 0; i < n; ++i) {
    req = codec_audio_lm_default_tts_request();
    req.text = codec_audio_lm_tts_segment(alm, i);
    req.voice = voice;
    codec_audio_lm_tts_prompt(alm, &req, &vocab);
    codec_audio_lm_completion c;
    codec_audio_lm_get_completion(alm, &c);
    if (c.flow == CODEC_AUDIO_LM_FLOW_SELF_CONTAINED) {
        codec_audio_lm_generate_self_contained(alm);
    } else {
        // clear the KV cache; build a sampler from c.sampler (+ c.grammar / c.allowed)
        // decode chunks 0..n_chunks-1 of every sequence (chunk i of all sequences in one batch)
        // if c.needs_prompt_hidden: codec_audio_lm_prefill(alm, every prompt position's hidden, n)
        bool sample = c.sample_first;
        for (int frames = 0; frames < c.max_frames; ) {
            tok = sample ? sample_backbone() : -1;
            codec_audio_lm_observe(alm, sample ? &tok : NULL, c.needs_embeddings ? hiddens : NULL, &v);
            frames += v.is_audio;
            if (v.action == CODEC_AUDIO_LM_STOP) break;
            if (v.action == CODEC_AUDIO_LM_FEED_TOKEN) decode_token(v.token >= 0 ? v.token : tok);
            else for each sequence s: decode_embd(codec_audio_lm_next_embd(alm, s));
            sample = v.sample_backbone;
        }
    }
    codec_audio_lm_decode(alm, &pcm);                 // or stream_poll / stream_finish
    gap = codec_audio_lm_tts_join(alm, &pcm, i, n);   // then `gap` samples of silence
}
```

`common/tts_runner.cpp` (`run_one`) is this loop over llama.cpp in C++;
`tests/e2e/audio_lm_host_smoke.py` is the same loop over llama-cpp-python.

### The vocabulary

Prompt builders look up the backbone's vocabulary through
`codec_audio_lm_vocab` callbacks (`tokenize`, `token_to_piece`, `n_tokens`,
`eos`, `is_eog`): some models recognize their backbone from it, place special
tokens, or restrict sampling to their audio tokens. The library never decodes.

### The completion

| Field | Meaning for the host |
|---|---|
| `flow` | `TOKENS`: audio codes are backbone tokens. `CODEC_HEAD`: codec heads read the backbone hidden. `HIDDEN`: tokens drive, hiddens are the audio latent. `SELF_CONTAINED`: no backbone. |
| `n_seq`, `chunks`, `n_chunks` | the prompt of each sequence: TEXT (tokenize), TOKENS or EMBD (`n_embd`-wide rows) chunks, decoded in order; chunk `i` of sequence `s` is `chunks[s * n_chunks + i]` |
| `sampler` | backbone sampling, named after llama.cpp's `common_params_sampling` (temp, top_k, top_p, min_p, penalties, `penalty_exempt`, DRY, sampler order) |
| `grammar` / `allowed` | GBNF, or the allowed backbone token ranges as a logit bias |
| `needs_embeddings` | the context returns hidden states; pass the last position's hidden of every sequence to `observe` |
| `needs_prompt_hidden` | decode the prompt with every position as output and call `codec_audio_lm_prefill` |
| `sample_first` | the first step samples a backbone token; afterwards each verdict's `sample_backbone` says |
| `streamable` | `codec_audio_lm_stream_poll` works during the generation |
| `max_frames` | audio frames (`is_audio` verdicts) after which to stop |
| `min_frames`, `stop_token` | minimum length: while fewer than `min_frames` frames exist, a sampled `stop_token` is replaced by one draw from a second sampler (same temp / top_k / top_p, no penalties, no grammar, `allowed` minus `stop_token`); sample with the primary sampler first so its random stream advances every step |

`penalty_exempt` needs `patches/penalty-exempt-llamacpp.patch` (applied to
the bundled llama.cpp); a stock llama.cpp ignores the list.

### Verdicts

`codec_audio_lm_observe` takes the sampled token per sequence (or NULL) and
the hidden state per sequence (or NULL) and answers:

- `FEED_TOKEN`: decode `verdict.token` when it is >= 0 (the model picked the
  token itself), else the sampled token;
- `FEED_EMBD`: decode `codec_audio_lm_next_embd(alm, s)` on every sequence `s`;
- `STOP`: the generation ended; `codec_audio_lm_stop_reason` says why.

`is_audio` marks steps that produced audio (count them against `max_frames`,
poll the stream after them).

### Audio

`codec_audio_lm_decode` returns the generation's audio. For streaming, call
`codec_audio_lm_stream_poll(alm, chunk_ms, &pcm)` after each audio step: it
returns the audio that became final since the last call. Each window is
re-decoded from the first frame (causal decoders need the whole left context),
the right edge the decoder may still revise is held back (0.32 s), and the
joins are crossfaded (20 ms). `codec_audio_lm_stream_finish` returns the rest.
`decodable_frames` says how many frames are decodable so far.

To decode later (out of order, several generations on, more than once),
`codec_audio_lm_get_generation` returns the generation's decoder input:
frame-major codec codes (`CODEC_AUDIO_LM_DATA_CODES`, `width` codebooks) or
latents (`CODEC_AUDIO_LM_DATA_LATENTS`, `width` = latent dim), already in codec
form (control codebooks dropped, delay pattern undone, trailing silence
tokens appended). The arrays stay valid until the next call; copy them.
`codec_audio_lm_decode_generation` decodes such a copy with the voice it was
made with (pass the request's voice) and leaves the in-progress generation alone; for the
generation just made it gives exactly the PCM of `codec_audio_lm_decode`.

### Voices

A voice is a `codec_speaker` holding all of the model's voice conditioning,
whichever part of the model reads it: the backbone prompt (an x-vector, a
reference's tokens / latents / transcript), the decoder (S3Gen's prompt), or
both (Chatterbox, Qwen3-TTS ICL); see `docs/codec_speaker.md` for each
model's slots. One structure for every model, so a host manages voices the
same way everywhere:

| Call | Voice |
|---|---|
| `codec_audio_lm_list_voices` / `codec_audio_lm_voice` | names of the builtin voices, or those in a speaker file |
| `codec_audio_lm_get_voice(name)` | builtin (model-owned; NULL = the default, `<arch>.spk.default`) |
| `codec_audio_lm_create_voice(pcm, n, sr, ref_text)` | cloned from a reference clip, all conditioning computed once (caller frees) |
| `codec_audio_lm_load_voice(path, name)` / `codec_audio_lm_save_voice(voice, path, name)` | speaker GGUF of this model (hash-checked) |

The voice goes in each request (`codec_audio_lm_tts_request.voice`) and each
`codec_audio_lm_decode_generation`, never in the context, so one context can
serve requests in different voices. A request borrows its voice until its
generation is decoded. The default voice comes from the GGUF metadata;
`--kv-override <arch>.spk.default=str:NAME` changes it.

### Segmentation and joining

Long text is generated one segment at a time. `codec_audio_lm_tts_split` and
`codec_audio_lm_tts_join` give every model the same general behaviour, tuned
by the family's parameters:

- **split**: sentences / clauses packed up to `max_bytes` UTF-8 bytes in any
  script (`tts_split_text`); a family may pick another mode instead (KittenTTS:
  the English sentence packing of its reference, with a minimum chunk size);
- **join**: leading and trailing silence trimmed, 8 ms fades, a gap of
  silence between segments (160 ms by default, per-family).

A family that needs something else entirely can still override both.

## Families

| Family | Models | Flow | Notes |
|---|---|---|---|
| `codebook` | CSM, Qwen3-TTS, MOSS-TTSD | CODEC_HEAD | depth decoder or delayed parallel heads; MOSS-TTSD samples codebook 0 from the backbone under a grammar |
| `realtime` | MOSS-TTS-Realtime | CODEC_HEAD | text lane from the backbone's embedding table (`params.backbone_path`) |
| `lfm2` | LFM2-Audio | CODEC_HEAD | text warm-up from the tied embedding head (`verdict.token`), then audio frames |
| `chatterbox` | Chatterbox T3, Turbo, Multilingual | CODEC_HEAD | guidance pair (`n_seq` 2) for T3 |
| `cosyvoice` | CosyVoice 1/2/3, BreezyVoice | CODEC_HEAD | SFT / zero-shot / cross-lingual / instruct modes; v1 disables flash attention |
| `continuous` | BlueMagpie | HIDDEN | prompt hiddens prime the CFM head; stop head ends it |
| `flow` | Pocket-TTS | CODEC_HEAD | starcoder2 backbone; the BOS frame input ends the prompt |
| `neutts`, `outetts` | NeuTTS, OuteTTS | TOKENS | audio codes are backbone tokens |
| `kitten` | KittenTTS 2 | TOKENS | penalty exemption + DRY in the sampler, sentence-pack segmentation |
| `soprano` | Soprano | HIDDEN | the hidden that predicts each token is the decoder input |

`alm_make_family` (`common/audio_lm_family.cpp`) picks the family from the
codec GGUF's metadata. Moshi is refused: it is a full-duplex dialogue model
with no audio end (no one-shot stop condition) and its Helium backbone has no
llama.cpp architecture.

### Adding a model

1. Write `common/audio_lm_family_<name>.cpp`: subclass `alm_family`; implement
   `tts_prompt` (fill `alm_completion`) and `observe`; override `caps`,
   `set_voice`, `backbone_requirements`, `segment_split` / `segment_join`,
   `next_embd`, `decode`, `decodable_frames` where the model needs it.
   Model-specific codec_lm helpers stay internal (`src/lm/lm_models.h`).
2. Detect it in `alm_make_family`.
3. Add a case to `tests/e2e/ttscli_synthesize_smoke.py`. Every host
   (tts-cli, tts-server, `codec_cpp.TTS` / `AudioLM`, llama.rn) runs it with
   no further change.

## Ready-made hosts

- **`common/tts_runner`**: the loop above over the bundled llama.cpp
  (libttsbackbone), with segmentation, voices and streaming;
  `tools/tts-cli` and `tools/tts-server` drive it.
- **Python `codec_cpp.TTS`**: the same loop in Python, over `codec_cpp.AudioLM`
  and the bundled llama.cpp's C API (ctypes on libttsbackbone).
- **Python `codec_cpp.AudioLM`**: `codec_audio_lm.h` call for call, with
  `codec_cpp.Vocab.from_llama_cpp` for llama-cpp-python hosts.
