# Chatterbox T3 LM-adaptor integration plan (backlog)

Status: **backlog only — not implemented**.  Adds TTS profile support
for Resemble AI's Chatterbox T3 family (English + multilingual variants)
on top of our existing Chatterbox-S3G codec.

## 1. Architecture summary

Chatterbox = `T3` (Token-To-Token TTS LM) + `S3G` (speech-to-waveform
decoder) + `VE` (voice encoder).  S3G is already shipped as
`CODEC_ARCH_CHATTERBOX_S3G` in `src/models/chatterbox_s3g.cpp`.

T3 layout (from `.model-src/chatterbox/src/chatterbox/models/t3/`):

```
T3
├── tfmr               : LlamaModel (520M)   — generic Llama, embd-driven
│     Llama_520M_CONFIG: 30 layers, hidden 1024, 16 MHA heads,
│                        head_dim=64, FFN 4096, llama3 RoPE
│                        (factor=8, lo=1, hi=4, orig_max=8192),
│                        rope_theta=500000, rms_norm_eps=1e-5,
│                        tie_word_embeddings=False, bfloat16
│     NOTE: vocab_size=8 placeholder — tfmr's own embed_tokens is
│           unused; everything is fed via inputs_embeds.
│
├── cond_enc           : T3CondEnc (perceiver resampler + speaker proj
│                        + optional emotion proj) → (B, len_cond, 1024)
├── text_emb           : nn.Embedding(text_dict, 1024)
│                        text_dict = 704 (English) or 2454 (multilingual)
├── speech_emb         : nn.Embedding(8194, 1024)    ← single codebook
├── text_pos_emb       : LearnedPositionEmbeddings(max_text_tokens+2, 1024)
├── speech_pos_emb     : LearnedPositionEmbeddings(max_speech_tokens+4, 1024)
├── text_head          : Linear(1024, text_dict, bias=False)
└── speech_head        : Linear(1024, 8194, bias=False)
```

Special tokens (from `T3Config`):

```
start_text_token   = 255
stop_text_token    = 0
start_speech_token = 6561
stop_speech_token  = 6562
speech_tokens_dict = 8194  (= 6561 codebook codes + 2 specials + slack)
max_text_tokens    = 2048
max_speech_tokens  = 4096
```

## 2. Inference flow

```python
# 1. Build conditioning (run voice encoder on ref audio, build T3Cond)
ve_emb = VE.encode(ref_audio_pcm)                       # (B, 256)
t3_cond = T3Cond(speaker=ve_emb, emotion=...)
cond_emb = T3.cond_enc(t3_cond)                         # (B, len_cond, 1024)

# 2. Embed text tokens (NO autoregressive text emission at inference time —
#    text is provided as a complete prompt segment)
text_emb = T3.text_emb(text_tokens) + T3.text_pos_emb(text_tokens)
                                                        # (B, T_text, 1024)

# 3. Embed BOS speech token
bos_emb = T3.speech_emb([start_speech_token]) +
          T3.speech_pos_emb([0])                        # (B, 1, 1024)

# 4. Concatenate full prompt; feed to Llama backbone
prompt = cat([cond_emb, text_emb, bos_emb], dim=1)
hidden = backbone(prompt)                               # (B, T_prompt, 1024)

# 5. AR loop over speech tokens
for step in range(max_new_tokens):
    h = backbone.last_hidden                            # (B, 1024)
    logits = T3.speech_head(h)                          # (B, 8194)
    code   = sample(logits, temperature, top_p, ...)
    if code == stop_speech_token: break
    next_emb = T3.speech_emb([code]) +
               T3.speech_pos_emb([step + 1])            # (B, 1, 1024)
    hidden = backbone.feed_embeds(next_emb)

# 6. Hand the speech token stream to S3G for waveform synthesis
wav = S3G.decode(speech_tokens, ref_x_vector=...)
```

Inference also supports **classifier-free guidance** (`cfg_weight`,
default 0.5): the backbone is run twice per step (conditional +
unconditional batch), and final logits are
`uncond + cfg_weight * (cond - uncond)`.  v1 can skip CFG (cfg_weight=0)
and revisit later.

## 3. Maps onto existing codec.cpp framework

| Piece | Our framework | Notes |
|---|---|---|
| Backbone (Llama 520M) | llama.cpp `qwen3`/`llama` arch via standard backbone GGUF | identical pattern to CSM's Llama-3.2-1B; reuse `prep_csm`-style flattening |
| codec_lm kind | `parallel_heads_delay` with `n_cb=1, delay_pattern=[0]` | single-codebook, no depth decoder; cb-0 head = speech_head |
| Codec out (S3G) | `CODEC_ARCH_CHATTERBOX_S3G` (shipped) | uses `ref_x_vector` from VE encoder for speaker conditioning at decode time |
| Profile | `ChatterboxProfile` + `ChatterboxSession` | embd-driven (text in prompt, speech AR out) — same shape as MOSS-TTS-Realtime |
| Voice encoder (VE) | new module OR HF-fallback | runs once at session start to produce speaker embedding |
| Conditioning encoder | bundle into codec_lm GGUF OR run HF-side at session start | perceiver resampler + linear projections; can be precomputed once per ref audio |

The fundamental shape lines up with what we already ship — Llama
backbone + single-cb parallel head + dedicated codec for final
waveform.  The differences vs. MOSS-TTS-Realtime are:

  1. **Single codebook** (n_cb=1) instead of 16 — `parallel_heads_delay`
     just has one head and no delay.
  2. **Learned positional embedding** added to each speech-token embed
     before feeding to the backbone — easy to handle Python-side in
     the profile's `compose_next_embed`.
  3. **Conditioning encoder** prepended to the prompt — produces a
     fixed-length `(len_cond, 1024)` block that gets concatenated
     before text+speech.
  4. **Voice encoder** preprocessing step.

## 4. Pieces needed

### 4.1 Backbone converter — `prep_chatterbox_t3`

Add to `scripts/convert-backbone-to-gguf.py`.  Mechanical Llama
extraction from `t3.tfmr` (resemble-ai/chatterbox repo:
`t3_cfg.safetensors` for English, `t3_mtl23ls_v3.safetensors` for
multilingual).  Rename `tfmr.*` → `model.*`; emit standalone Llama
config matching `LLAMA_520M_CONFIG_DICT` (with the placeholder
vocab_size=8 left as-is — the backbone's own token_embd is unused at
inference, since we feed embeds via the `embeddings=True` path).

Both English and multilingual variants share the same backbone arch;
only the LM-adaptor side differs (different text_emb table size,
different tokenizer).

### 4.2 LM-adaptor converter — `scripts/converters/lm_adaptor/chatterbox.py`

Write `codec.lm.*` metadata + `lm.*` tensors for the chatterbox-bundled
GGUF (similar to existing `qwen3_tts.py` / `moss_ttsd.py` / `csm.py`):

```
codec.lm.has_adaptor               = true
codec.lm.kind                      = "parallel_heads_delay"
codec.lm.host_arch                 = "llama"
codec.lm.hidden_dim                = 1024
codec.lm.audio_embed_dim           = 1024
codec.lm.n_codebook                = 1
codec.lm.codebook_sizes            = [8194]
codec.lm.delay_pattern             = [0]
codec.lm.parallel.tied_heads_to_embd = false

# Chatterbox-specific extensions:
codec.lm.chatterbox.text_vocab_size      = 704 / 2454
codec.lm.chatterbox.start_text_token     = 255
codec.lm.chatterbox.stop_text_token      = 0
codec.lm.chatterbox.start_speech_token   = 6561
codec.lm.chatterbox.stop_speech_token    = 6562
codec.lm.chatterbox.max_text_tokens      = 2048
codec.lm.chatterbox.max_speech_tokens    = 4096
codec.lm.chatterbox.is_multilingual      = bool
codec.lm.chatterbox.has_emotion_cond     = true
codec.lm.chatterbox.speaker_embed_dim    = 256
codec.lm.chatterbox.cond_len             = 32   (or whatever cond_enc emits)
```

Tensors:

```
# Speech side (the audio "codebook" head + embed)
lm.audio_embd_0.weight        ← speech_emb.weight              (8194, 1024)
lm.c0_head.weight             ← speech_head.weight             (8194, 1024)

# Text-prompt side (used by the profile's prompt builder, not the AR step loop)
lm.chatterbox.text_emb.weight ← text_emb.weight                (text_dict, 1024)
lm.chatterbox.text_head.weight ← text_head.weight              (text_dict, 1024)
            # text_head only useful for auxiliary scoring / future bidirectional;
            # at inference text isn't emitted, so this is optional

# Learned positional embeddings
lm.chatterbox.text_pos_emb.weight    ← text_pos_emb.weight     (max_text+2, 1024)
lm.chatterbox.speech_pos_emb.weight  ← speech_pos_emb.weight   (max_speech+4, 1024)

# Conditioning encoder (perceiver resampler + projections)
lm.chatterbox.cond_enc.*    ← cond_enc.*                       (various)
```

Modify `scripts/converters/lm_adaptor/__init__.py` dispatch to route
the chatterbox T3 checkpoint to this new module.

### 4.3 Voice encoder (VE)

Two options:

A. **Codec.cpp module** — port `ve.safetensors` (LSTM-based ECAPA-style
   voice encoder, small) as a new graph kind under `src/models/`,
   producing a 256-d speaker embedding from PCM.  Cleanest long-term.
B. **HF-fallback at session start** — load `ve.safetensors` Python-side,
   run once on `--speaker-config` ref audio to produce the 256-d
   embedding, then continue with codec.cpp AR.  Quickest first
   implementation; same pattern as MOSS-TTS-Nano's HF AR loop.

Recommend **B for v1, A as follow-up**.

### 4.4 Profile in `examples/tts.py`

`ChatterboxProfile` + `ChatterboxSession`:

```python
class ChatterboxSession(TTSSession):
    def initial_prompt_embeds(self, text):
        # 1. HF-load VE on ref_audio (if speaker config provided), produce
        #    speaker_embed (256-d).  Else use baked-in conds.pt fallback.
        # 2. Run cond_enc Python-side OR via codec_lm to get cond_emb
        #    (len_cond, 1024).
        # 3. Tokenize text via Chatterbox tokenizer (704-vocab English,
        #    2454-vocab multilingual); wrap with start/stop_text tokens.
        # 4. Build text_embd = text_emb(text_ids) + text_pos_emb(positions)
        #    using tables from the codec_lm GGUF.
        # 5. Build bos_embd = speech_emb([start_speech]) + speech_pos_emb([0]).
        # 6. Return prompt = concat([cond_emb, text_embd, bos_embd]).
    
    def compose_next_embed(self, codes, step):
        # speech_emb(codes[0]) + speech_pos_emb([step + 1])
        # codec_lm.compose_audio_embd returns the speech_emb side;
        # add speech_pos_emb on top Python-side.
    
    def detect_stop(self, codes, step):
        return codes[0] == STOP_SPEECH_TOKEN  # 6562
    
    def synthesize(self, codes_all, codec):
        # codec is Chatterbox-S3G; needs ref_x_vector from VE as
        # speaker conditioning.  Existing codec_decode_params has n_q
        # field; we may need to extend with a speaker-vector pointer.
```

Speaker config (same JSON shape as existing voice-clone profiles):

```json
{
  "ref_audio": "alice.wav",
  "ref_text":  "Hello, my name is Alice.",   // optional, currently unused
  "emotion":   0.5,                          // optional, only T3 with emotion_adv
  "cfg_weight": 0.0                          // CFG; default 0 (no CFG)
}
```

### 4.5 Runtime extensions

Minimal changes needed:

1. **Learned PE compose** — for Chatterbox, the per-step compose needs
   `speech_emb(code) + speech_pos_emb(step+1)`, not just the embed
   lookup.  Easiest: do the addition Python-side in
   `compose_next_embed` using the speech_pos_emb table from the
   codec_lm GGUF (loaded once at session start).  No C-side change.

2. **(Optional) CFG dual-batch** — for v1 skip; if added later, the
   profile runs the backbone twice per step and blends logits.  Pure
   Python; no codec_lm runtime change.

3. **(Optional) Conditioning encoder as ggml graph** — bundling
   cond_enc into codec_lm would let `codec_lm_compose_cond` produce
   `(len_cond, hidden)` from a speaker embedding + emotion scalar.
   Cleaner long-term.  v1 can run cond_enc in Python via the HF model.

4. **S3G decode with ref speaker vector** — current
   `codec_decode_params` may not pass a speaker vector to the codec.
   Check `src/models/chatterbox_s3g.cpp`'s `decode_with` signature;
   if it already accepts the vector via the token buffer or a
   side channel, no change.  Otherwise extend.

## 5. Implementation order

Recommended phased landing:

**Phase A (HF-fallback profile, no C-side changes, ~1 model-day):**

1. Wire `ChatterboxProfile` in `tts.py` with `bypass_standard_run=True`
   running the entire T3 inference through HF
   (`.model-src/chatterbox` is already vendored); use our codec.cpp
   Chatterbox-S3G GGUF for the final waveform decode.  Same pattern
   as the MOSS-TTS-Nano-100M profile (`run_full`).
2. Smoke-test with English (`t3_cfg.safetensors`) and multilingual
   (`t3_mtl23ls_v3.safetensors`) variants.

This validates the speaker config + S3G end-to-end with minimum code,
and gives users a working CLI immediately.

**Phase B (native LM adaptor, ~1 model-week):**

3. Implement `prep_chatterbox_t3` + run conversion → standalone Llama
   520M backbone GGUF.
4. Implement `scripts/converters/lm_adaptor/chatterbox.py` to write
   the codec_lm side (single-cb head + embed + learned PE tables +
   cond_enc tensors); dispatch update.
5. Run conversion → bundled `chatterbox_t3.gguf` (codec_lm + S3G in
   one file) or split (codec_lm GGUF + s3g GGUF separately).
6. Implement `ChatterboxSession` native path: prompt build via
   codec_lm tables, AR through codec_lm + llama.cpp, S3G decode.
7. Switch the profile to native, retire the HF fallback.

**Phase C (optional refinements, no fixed deadline):**

8. Voice encoder as a codec.cpp ggml graph (drops the last HF
   dependency at inference).
9. Conditioning encoder as a ggml graph (drops HF cond_enc load).
10. CFG dual-batch sampling.
11. Alignment stream analyzer (multilingual only — improves robustness
    of stop-token detection at multilingual inference; layer-9
    text-alignment monitor).

## 6. Multilingual variants

Same architecture, different weight bundle:

| File | Vocab | Languages | Notes |
|---|---|---|---|
| `t3_cfg.safetensors` | 704 | English only | original release |
| `t3_23lang.safetensors` | 2454 | 23 langs | uses `mtl_tokenizer.json` instead of `tokenizer.json`; needs grapheme normalization via `grapheme_mtl_merged_expanded_v1.json` |
| `t3_mtl23ls_v2.safetensors` | 2454 | 23 langs | v2 multilingual |
| `t3_mtl23ls_v3.safetensors` | 2454 | 23 langs | v3 multilingual (latest) |

All four share `s3gen.safetensors` / `s3gen_v3.safetensors` for the
codec and `ve.safetensors` for the voice encoder.  Profile registry
can expose them as separate entries (`chatterbox-en`,
`chatterbox-multilingual-v3`, etc.) pointing at different LM-adaptor
GGUFs.

## 7. Risks / open questions

- **Conditioning encoder weights' shape** — need to read
  `T3CondEnc` source to figure out the perceiver resampler layout and
  emit a clean tensor namespace.  Not yet inspected here.
- **VE checkpoint format** — `ve.safetensors` arch needs decoding
  before either HF-fallback or native port.
- **CFG inference** — Resemble's default `cfg_weight=0.5` matters for
  quality.  Phase A's HF fallback runs it natively; Phase B needs to
  decide whether to support batched cond+uncond before declaring
  Phase B complete.
- **S3G's `ref_x_vector` pass-through** — verify whether the existing
  S3G `codec_decode_with` already accepts the per-call speaker
  vector or if `codec_decode_params` needs a new field.
- **Tokenizer integration** — multilingual variant uses
  `mtl_tokenizer.json` (a HF tokenizers JSON) plus a
  Cangjie-style grapheme prefix map.  Profile-side, Python load is
  fine; long-term these could be embedded into the codec_lm GGUF.
