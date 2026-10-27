# codec_audio_lm — running audio LMs with a llama.cpp backbone

`include/codec_audio_lm.h` describes how to run an audio LM whose backbone is
a llama.cpp model and whose audio side is a codec.cpp GGUF. It plays the role
of llama.cpp's `common` + MTMD: the library says **what to feed the backbone,
how to sample, when to stop**, and turns the generation into audio. The host
keeps every `llama_*` call: tokenization, `llama_decode`, the sampler, the KV
cache, batching. Nothing in the API names a model; every model-specific detail
sits behind it.

Text-to-speech is the first prompt builder (`codec_audio_lm_tts_format` /
`codec_audio_lm_tts_build`). Other tasks (audio-to-audio) will add their own builders that produce the same
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
    codec_audio_lm_format f;
    codec_audio_lm_tts_format(alm, &req, &vocab, &f);         // the prompt as text
    codec_audio_lm_completion c;
    codec_audio_lm_tts_build(alm, &req, f.prompt, &vocab, &c); // its chunks
    if (c.flow == CODEC_AUDIO_LM_FLOW_SELF_CONTAINED) {
        codec_audio_lm_generate_self_contained(alm);
    } else {
        // clear the KV cache; build the host's llama.cpp sampler (+ c.grammar / c.allowed /
        // c.penalty_exempt) and one sampler per c.head_groups[g]
        // decode chunks 0..n_chunks-1 of every sequence (chunk i of all sequences in one batch)
        // if c.needs_prompt_hidden: codec_audio_lm_prefill(alm, every prompt position's hidden, n)
        bool sample = c.sample_first;
        for (int frames = 0; frames < c.max_frames; ) {
            tok = sample ? sample_backbone() : -1;
            codec_audio_lm_observe(alm, sample ? &tok : NULL, c.needs_embeddings ? hiddens : NULL, &v);
            while (v.action == CODEC_AUDIO_LM_SAMPLE_HEAD)   // codec-side heads: the host draws
                codec_audio_lm_head_token(alm, draw(group[v.head_group], codec_audio_lm_head_logits(alm), v.n_vocab), &v);
            frames += v.frame_done;
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

### Several requests at once

A context runs one generation at a time. For more, open one context per
concurrent request with `codec_audio_lm_init_shared(base, ...)`: it shares
`base`'s weights (the codec GGUF, and the backbone's embedding table for the
models that read it) and has its own generation state, codec graphs and
backend instances, so each context can run on its own thread. Pair each
with its own llama.cpp context on the same `llama_model`. `base` must outlive
the contexts made from it. `tts-server --parallel N` (`tts_runner_load`
with `n_parallel`) works this way.

### The prompt

As in llama.cpp's MTMD, the prompt is text: `codec_audio_lm_tts_format`
returns the model's template with the request filled in (roles, instruction,
the normalized text, text-form voices written as their vocab pieces) and the
completion's constraints and flags, with no generation state, so a host can
show or edit it. `codec_audio_lm_tts_build` turns that text (the format's or
the host's edit) into the completion: text is tokenized by the backbone vocab
with parse_special, and markers stand where the backbone takes input the vocab
cannot spell:

- `<__voice__>`: the voice's embedding-form conditioning;
- `<__codec:NAME__>`: a model row (an embedding row, or a token without a
  piece), or `<__codec:text__>`: the text to speak starts here (models that
  tokenize it apart from what precedes).

Language tags (`[en]`), phonemes and script conversions stay the caller's;
format only applies the model's deterministic normalization.

| Model | Prompt |
|---|---|
| Soprano | `[STOP][TEXT]{text}[START]` |
| NeuTTS | `user: Convert the text to speech:<\|TEXT_PROMPT_START\|>{ref phonemes} {phonemes}<\|TEXT_PROMPT_END\|>\nassistant:<\|SPEECH_GENERATION_START\|>` + the reference codes as `<\|speech_N\|>` |
| OuteTTS | `<\|im_start\|>\n<\|text_start\|>{speaker text + words}<\|text_end\|>\n<\|audio_start\|>\n` + the speaker's audio section as pieces (+ `<\|word_start\|>` for 1.0) |
| KittenTTS 2 | `<__voice__>` (lm_cond row) + `<__codec:ref_text_start__>{transcript}<__codec:ref_text_end__><__codec:ref_speech_start__><\|a_N\|>…<__codec:ref_speech_end__>` + `[{emo: 1}]<\|text_start\|>{text}<final_seg><\|speech_start\|>` |
| Pocket-TTS | `{prepared text}<__voice__><__codec:bos__>` |
| BlueMagpie, LFM2-Audio | `{the GGUF's prefix}{text}{suffix}` (BOS added) |
| CSM, MOSS-TTSD | `[<__voice__>]{prefix}{text}{suffix}` (MOSS-TTSD: `[S1]` → `<speaker1>`) |
| Qwen3-TTS | `[<\|im_start\|>user\n{instruct}<\|im_end\|>\n]<\|im_start\|>assistant\n<__voice__>[{reference transcript}]<__codec:text__>{text}` |
| MOSS-TTS-Realtime | `{the GGUF's context}<__codec:text__>{text}` |
| Chatterbox | `<__voice__>{punc_norm'd text}` (T3: ' ' as `[SPACE]`) |
| CosyVoice | `<__codec:sos__>[<__voice__> (v1 x-vector)]{prompt text}<__codec:text__>{text}<__codec:task__>[<__voice__> (zero-shot reference tokens)]` |

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
| `penalty_exempt` | backbone tokens the host's penalties should not touch (count the prompt tokens for the penalties, as llama-server does) |
| `head_groups` | codec-side head groups the host samples (`SAMPLE_HEAD` verdicts), one host sampler each |
| `grammar` / `allowed` | GBNF (the tokens the model may emit and its minimum length, `min_frames` frames before the end), and the allowed backbone token ranges as a logit bias |
| `needs_embeddings` | the context returns hidden states; pass the last position's hidden of every sequence to `observe` |
| `needs_prompt_hidden` | decode the prompt with every position as output and call `codec_audio_lm_prefill` |
| `sample_first` | the first step samples a backbone token; afterwards each verdict's `sample_backbone` says |
| `streamable` | `codec_audio_lm_stream_poll` works during the generation |
| `min_frames` | audio frames before the generation may end: the request's `min_frames` when >= 0, else the model's own minimum (0 = none). The generation keeps to it (grammar, or an end code masked in the codec-side head); `codec_audio_lm_tts_format` reports it too, so a host can pass it to its own minimum-length setting and back as the request's `min_frames` |
| `max_frames` | audio frames (`frame_done` verdicts) after which to stop |

`penalty_exempt` needs `patches/penalty-exempt-llamacpp.patch` (applied to
the bundled llama.cpp); a stock llama.cpp ignores the list.

### Sampling

The host samples every token with llama.cpp's sampler: the backbone's, and
each codec-side head's (`SAMPLE_HEAD` verdicts: `codec_audio_lm_head_logits`
gives the head's logits, the model's constraints already applied as -inf and
guidance already combined; `codec_audio_lm_head_token` takes the id back and
gives the next verdict). codec_audio_lm holds no sampler and no sampling
settings; it only suggests constraints: `grammar` / `allowed` /
`penalty_exempt` for the backbone, and `head_groups` for the heads. Build one
host sampler per group (one random stream and penalty history; seed group `g`
with seed + 1 + g), with the same llama.cpp settings as the backbone; a group
differs only by `penalties` (off for e.g. Qwen3-TTS's depth codebooks),
`history` (accepted first: Chatterbox's start-of-speech token) and
`penalty_exempt`. Heads have no vocabulary, so no grammar or DRY. External
logits go through `llama_sampler_apply` on a host-built
`llama_token_data_array`, then `llama_sampler_accept`.

A model's recommended settings ship in its **backbone** GGUF as llama.cpp's
`general.sampling.*` keys (with `patches/sampling-meta-llamacpp.patch` for
freq / presence / DRY), read the way llama.cpp's
`common_init_sampler_from_model` does. The model's own randomness (the init
noise of a codec-side flow / CFM head: Pocket-TTS, BlueMagpie) is not token
sampling: `codec_audio_lm_set_seed` seeds it after `tts_build` (unset =
random), and hosts pass the seed the user gave the completion. Decoders are
deterministic (Chatterbox S3G / CosyVoice use a fixed noise seed). Decoder settings
(`lm_guidance`, `flow_guidance`, `flow_steps`) are not sampling: < 0 takes the
codec GGUF's.

### Verdicts

`codec_audio_lm_observe` takes the sampled token per sequence (or NULL) and
the hidden state per sequence (or NULL) and answers:

- `FEED_TOKEN`: decode `verdict.token` when it is >= 0 (the model picked the
  token itself), else the sampled token;
- `FEED_EMBD`: decode `codec_audio_lm_next_embd(alm, s)` on every sequence `s`;
- `STOP`: the generation ended; `codec_audio_lm_stop_reason` says why.

`is_audio` marks steps whose token belongs to the audio: every codebook token
and, for models with an audio start / end, the markup between them (OuteTTS
word pieces, timestamps, code_start / code_end). OuteTTS, NeuTTS and Kitten
prefill their audio start, so all their steps are audio; LFM2's text warm-up
is not (its `<|audio_start|>` step is). The host emits the fed-back tokens of
the other steps as text. `frame_done` marks steps that completed an audio frame
(count them against `max_frames`, poll the stream after them). A STOP verdict
does not report the end token.

### Audio

`codec_audio_lm_decode` returns the generation's audio. For streaming, call
`codec_audio_lm_stream_poll(alm, chunk_ms, &pcm)` after each `frame_done` step: it
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
   `tts_format` (the prompt text and `alm_completion`'s constraints) and
   `tts_build` (read the prompt, fill the chunks, set up the generation) and
   `observe`; override `caps`,
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
  `tools/cli/tts-cli` and `tools/server/tts-server` drive it.
- **Python `codec_cpp.TTS`**: the same loop in Python, over `codec_cpp.AudioLM`
  and the bundled llama.cpp's C API (ctypes on libttsbackbone).
- **Python `codec_cpp.AudioLM`**: `codec_audio_lm.h` call for call, with
  `codec_cpp.Vocab.from_llama_cpp` for llama-cpp-python hosts.
