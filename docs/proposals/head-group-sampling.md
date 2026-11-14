# Proposal: per-head-group sampler settings in `codec_audio_lm_head_group`

Status: proposal for review, not implemented. If accepted it changes `include/codec_audio_lm.h`
(`struct codec_audio_lm_head_group`), `common/tts_runner.cpp` (the reference host), `python/codec_cpp/_ffi.py`
(the struct mirror) and `docs/codec_common_api.md`.

## Motivation

Some models sample different codec-side heads with different settings, and the host can't reproduce that today:
the header says "All groups use the same llama.cpp settings as the backbone" (`codec_audio_lm.h:255`).

- **MOSS-TTS-Nano**: the stop decision is a two-way text slot (continue / `<|audio_end|>`) sampled at
  temperature 1.5, top_p 1.0, and the 16 codebooks at 1.7 / top_k 25 / top_p 0.8 (`modeling_moss_tts_nano.py`,
  `_sample_next_assistant_text_token` and `inference(text_temperature=1.5, text_top_p=1.0, audio_temperature=1.7,
  audio_top_p=0.8)`). Under one shared setting, llama.cpp's chain applies top_p 0.8 before the temperature and
  drops the end while its reference probability is below ~0.31. Measured (5 seeds, 70 utterances): Japanese clips
  run 20.8 s on average against the reference's 11.7 s (11 of 20 over 20 s against 1). Scaling the logits
  in the family fixes the temperature but can't undo the truncation, because temperature is the last step.
- **Qwen3-TTS** ships separate `subtalker_temperature / top_k / top_p` for the code predictor next to the
  talker's (`generation_config.json`). Today they're equal (0.9 / 50 / 1.0), so nothing is lost yet.

What PR #7 does today is draw the decision inside the library, from the host's seed. That matches the
reference's lengths, and the continuous and flow families already draw their noise the same way. But it puts
sampling in the library, out of the host's control.

## Proposal

Append an optional override to the group, with an explicit flag (no "0 means inherit"):

```c
struct codec_audio_lm_head_group {
    bool            penalties;
    const int32_t * history;
    int32_t         n_history;
    const int32_t * penalty_exempt;
    int32_t         n_penalty_exempt;
    // new
    bool            own_sampling;     // true: sample this group with the settings below, not the backbone's
    float           temp;             // llama.cpp semantics (<= 0: greedy)
    int32_t         top_k;            // llama.cpp semantics (<= 0: off)
    float           top_p;            // llama.cpp semantics (1.0: off)
};
```

The family fills them from model metadata: MOSS-Nano's text-slot group gets `{true, 1.5, 0, 1.0}`, and the
codebook group keeps `own_sampling = false`. A host that builds one sampler per group (as `tts_runner` already
does, `tts_runner.cpp:402`) applies the override when `own_sampling` is set. The user's `--temp` etc. keep
applying to groups without it. A user who wants a greedy run still gets one: `--temp 0` makes every group greedy.

## Alternatives

1. **Keep the library draw** (PR #7 as it stands): exact and needs no API change, but the library samples.
2. **Host sampling with logit scaling** (tried, measured above): no API change and no library sampling, but the
   stop comes late. The smoke's 3 s English sentence ran to the 24 s frame cap.
3. **Logit scaling plus a high "top_p floor" hint** (a per-group `min top_p`): a narrower version of this
   proposal, fixing only this case.
4. **A new verdict field** carrying sampling settings per SAMPLE_HEAD step: more flexible per step, but larger,
   and every host's loop changes.

## Compatibility

- ABI: the struct grows at the end. Hosts compiled against the old header read the old fields at the same
  offsets, but iterate `head_groups[g]` with the old stride, which breaks for `n_head_groups > 1`. So this needs a
  version bump, or the array has to come through a size-carrying accessor (`codec_audio_lm_head_group_get(ctx, g,
  &out, sizeof out)`), which keeps old binaries working.
- Source: existing hosts compile unchanged and ignore the new fields, so they get today's behaviour.
- Python: `_ffi.py` appends the four fields.

## Migration

1. Add the fields (or the accessor) and document them.
2. `tts_runner`: `HeadSampler::build` takes the group's settings when `own_sampling` is set.
3. MOSS-Nano: the text slot becomes its own head group (group 1) with `{1.5, top_k off, top_p 1.0}`, and the
   logit scaling goes away.
4. Optional: Qwen3-TTS fills `subtalker_*` into its code-predictor group, from the converter.
