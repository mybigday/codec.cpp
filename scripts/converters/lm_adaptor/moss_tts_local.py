"""MOSS-TTS-{Realtime,Nano} LM-adaptor dump — `residual_depth_ar`.

Covers two MOSS architectures that share the same shape:

  - `MossTTSRealtime` (OpenMOSS-Team/MOSS-TTS-Realtime, ~2.3B):
      backbone     = Qwen3 2B (`language_model`)
      depth        = 4-layer Qwen3-style local_transformer
      per-step     = 16 audio codebooks (text comes from backbone)
      cb-0 input   = backbone hidden (no embed at depth pos 0)
      cb-i input (i≥1) = local_transformer.model.embed_tokens.{i-1}(c_{i-1})
      cb-i head    = local_transformer.local_lm_heads.{i}(depth_pos_i_hidden)
      compose      = sum_i outer embed_tokens.{i+1}(c_i)   for i in 0..15
                     (text embed added externally by the driver)

  - `MossTTSNanoForCausalLM` (OpenMOSS-Team/MOSS-TTS-Nano-100M, ~100M):
      backbone     = GPT-2 12L (`transformer`)
      depth        = 4-layer GPT-2-style local_transformer
      per-step     = 17 channels (cb-0 = text via text_lm_head, cb-1..16 = audio)
      cb-0 input   = backbone hidden (no embed at depth pos 0)
      cb-0 head    = text_lm_head(depth_pos_0_hidden)   (text vocab)
      cb-i head (i≥1) = audio_lm_heads.{i-1}(depth_pos_i_hidden)
      compose      = transformer.wte(text_token) + sum_i audio_embeddings.{i}(c_i)

Both fit the `residual_depth_ar` kind with `depth_emits_c0 = true`
(`c0_input_modality = "none"`).  The differences land in metadata
(`codec.lm.depth.arch ∈ {"qwen3", "gpt2"}`, `codec.lm.residual.cb0_vocab`)
and in two extra Nano-only tensors (`lm.depth.heads_0` = text head,
`lm.depth.cb0_input_embd.weight` = wte for compose).

NOT yet covered: GPT-2 depth-block runtime support (Nano) — landing
in a follow-up commit.  This file already emits the GGUF in a shape
the runtime can consume once the gpt2 depth block ships; until then,
only `MossTTSRealtime` is exposed via the dispatcher.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def dump(writer, sd: Dict[str, np.ndarray], cfg: Dict[str, Any],
         *, verbose: bool = False) -> None:
    archs = cfg.get("architectures") or []
    arch  = archs[0] if archs else ""

    if arch == "MossTTSRealtime":
        return _dump_realtime(writer, sd, cfg, verbose=verbose)
    if arch == "MossTTSNanoForCausalLM":
        return _dump_nano(writer, sd, cfg, verbose=verbose)
    raise NotImplementedError(
        f"moss_tts_local dispatched on unexpected arch {arch!r}")


# ---------------------------------------------------------------------
# MOSS-TTS-Realtime
# ---------------------------------------------------------------------

def _dump_realtime(writer, sd: Dict[str, np.ndarray],
                   cfg: Dict[str, Any], *, verbose: bool) -> None:
    lcfg = cfg["language_config"]
    ocfg = cfg["local_config"]

    rvq           = int(cfg["rvq"])                # 16
    audio_vocab   = int(cfg["audio_vocab_size"])   # 1027 (= 1024 + bos/eos/pad)
    backbone_hid  = int(lcfg["hidden_size"])       # 2048
    text_vocab    = int(lcfg["vocab_size"])        # 151936

    depth_hidden  = int(ocfg["hidden_size"])       # 2048
    depth_layers  = int(ocfg["num_hidden_layers"]) # 4
    depth_nh      = int(ocfg["num_attention_heads"])   # 16
    depth_nkvh    = int(ocfg["num_key_value_heads"])   # 8
    depth_hd      = int(ocfg["head_dim"])              # 128
    depth_inter   = int(ocfg["intermediate_size"])     # 6144
    depth_eps     = float(ocfg["rms_norm_eps"])
    depth_rope    = float(ocfg["rope_theta"])
    depth_maxpos  = int(ocfg["max_position_embeddings"])   # 33

    n_codebook    = rvq                            # all audio (text from backbone)
    has_in_proj   = (backbone_hid != depth_hidden) # 2048 == 2048 → False

    # --- metadata --------------------------------------------------------
    writer.add_bool   ("codec.lm.has_adaptor",     True)
    writer.add_string ("codec.lm.kind",            "residual_depth_ar")
    writer.add_string ("codec.lm.host_arch",       "qwen3")
    writer.add_uint32 ("codec.lm.hidden_dim",      backbone_hid)
    writer.add_uint32 ("codec.lm.audio_embed_dim", depth_hidden)
    writer.add_uint32 ("codec.lm.n_codebook",      n_codebook)
    writer.add_array  ("codec.lm.codebook_sizes",  [audio_vocab] * n_codebook)
    writer.add_array  ("codec.lm.delay_pattern",   [0] * n_codebook)
    writer.add_bool   ("codec.lm.parallel.tied_heads_to_embd", False)

    # End-of-audio: MOSS-TTS-Realtime channel 0 is an AUDIO codebook
    # (text comes from the backbone).  The streaming reference stops when
    # `audio_tokens[:, 0] == audio_eos_token`.  audio_bos/eos are class
    # defaults on the streaming inferencer (1025 / 1026), not stored in
    # config.json — the audio_vocab_size=1027 layout is [0..1023 codes,
    # 1024 pad, 1025 bos, 1026 eos].  Read overrides from config if present.
    audio_eos_token = int(cfg.get("audio_eos_token", audio_vocab - 1))  # 1026
    writer.add_int32 ("codec.lm.eos_code_c0", audio_eos_token)
    writer.add_int32 ("codec.lm.eos_min_step", 0)
    if "audio_bos_token" in cfg:
        writer.add_int32 ("codec.lm.bos_code_c0", int(cfg["audio_bos_token"]))
    else:
        writer.add_int32 ("codec.lm.bos_code_c0", audio_vocab - 2)          # 1025

    writer.add_uint32 ("codec.lm.residual.depth_layers",     depth_layers)
    writer.add_uint32 ("codec.lm.residual.depth_hidden",     depth_hidden)
    writer.add_uint32 ("codec.lm.residual.depth_n_heads",    depth_nh)
    writer.add_uint32 ("codec.lm.residual.depth_n_kv_heads", depth_nkvh)
    writer.add_uint32 ("codec.lm.residual.depth_head_dim",   depth_hd)
    writer.add_uint32 ("codec.lm.residual.depth_intermediate", depth_inter)
    writer.add_uint32 ("codec.lm.residual.depth_max_position", depth_maxpos)
    writer.add_float32("codec.lm.residual.depth_rms_norm_eps", depth_eps)
    writer.add_float32("codec.lm.residual.depth_rope_theta",   depth_rope)
    writer.add_bool   ("codec.lm.residual.depth_has_in_proj",  has_in_proj)
    writer.add_bool   ("codec.lm.residual.depth_has_qk_norm",  True)
    writer.add_bool   ("codec.lm.residual.depth_use_rope",     True)
    writer.add_bool   ("codec.lm.residual.depth_emits_c0",     True)
    writer.add_string ("codec.lm.residual.weight_layout",      "shared")
    writer.add_string ("codec.lm.residual.c0_input_modality",  "none")
    writer.add_string ("codec.lm.depth.arch",                  "qwen3")
    # Driver-side: text embed at each backbone step lives outside the
    # codec_lm (it's the host LLM's token_embd lookup of the just-sampled
    # text token); compose returns the audio-only sum and the driver
    # adds the text embed on top.
    writer.add_bool   ("codec.lm.compose.text_externally_added", True)

    # Optional in_proj.weight (used only if backbone_hid != depth_hidden).
    # MOSS-TTS-Realtime: 2048 == 2048 → skipped.  Kept here so future
    # variants that diverge can flip it.
    if has_in_proj:
        raise NotImplementedError(
            "MOSS-TTS-Realtime variant with backbone_hid != depth_hidden "
            "needs an `in_proj` tensor — not currently present in HF "
            "checkpoint")

    # --- per-cb depth-input embeddings ----------------------------------
    # `local_transformer.model.embed_tokens.{i}` (i=0..14, 15 tables) is
    # the depth decoder's per-position input embedding for cb-i — used
    # at depth pos (i+1) to embed the just-sampled cb-i code.  cb-15 is
    # never an INPUT (only an output) so position 15's input doesn't
    # exist; we still write a placeholder at slot 15 so the runtime
    # sees `n_codebook` tables for shape uniformity.
    for i in range(rvq):
        if i < rvq - 1:
            key = f"local_transformer.model.embed_tokens.{i}.weight"
            if key not in sd:
                raise RuntimeError(f"missing tensor: {key}")
            arr = sd[key]
            if arr.shape != (audio_vocab, depth_hidden):
                raise RuntimeError(
                    f"{key} shape {arr.shape} != ({audio_vocab}, {depth_hidden})")
            writer.add_tensor(f"lm.depth.audio_embd_{i}.weight",
                              arr.astype(np.float32), st_dtype="F16")
        else:
            # Placeholder for slot 15 — never read by the runtime
            # (depth pos 16 doesn't exist).  Reuse slot 14's weights to
            # keep size bounded and avoid an all-zero header that some
            # codecs treat as "missing".
            base_key = f"local_transformer.model.embed_tokens.{rvq - 2}.weight"
            placeholder = sd[base_key]
            writer.add_tensor(f"lm.depth.audio_embd_{i}.weight",
                              placeholder.astype(np.float32), st_dtype="F16")

    # --- per-cb depth heads --------------------------------------------
    for i in range(rvq):
        key = f"local_transformer.local_lm_heads.{i}.weight"
        if key not in sd:
            raise RuntimeError(f"missing tensor: {key}")
        arr = sd[key]
        if arr.shape != (audio_vocab, depth_hidden):
            raise RuntimeError(
                f"{key} shape {arr.shape} != ({audio_vocab}, {depth_hidden})")
        writer.add_tensor(f"lm.depth.heads_{i}.weight",
                          arr.astype(np.float32), st_dtype="F16")

    # --- depth transformer layers (Qwen3-style, 4 layers) ---------------
    q_dim    = depth_nh   * depth_hd
    kv_dim   = depth_nkvh * depth_hd
    expect_q    = (q_dim,        depth_hidden)
    expect_kv   = (kv_dim,       depth_hidden)
    expect_o    = (depth_hidden, q_dim)
    expect_gate = (depth_inter,  depth_hidden)
    expect_up   = (depth_inter,  depth_hidden)
    expect_down = (depth_hidden, depth_inter)

    for l in range(depth_layers):
        prefix_in  = f"local_transformer.model.layers.{l}"
        prefix_out = f"lm.depth.blk_{l}"
        for src_suf, dst_suf, expect in [
            ("input_layernorm.weight",          "attn_norm.weight",  None),
            ("self_attn.q_proj.weight",         "q.weight",          expect_q),
            ("self_attn.k_proj.weight",         "k.weight",          expect_kv),
            ("self_attn.v_proj.weight",         "v.weight",          expect_kv),
            ("self_attn.o_proj.weight",         "o.weight",          expect_o),
            ("self_attn.q_norm.weight",         "q_norm.weight",     (depth_hd,)),
            ("self_attn.k_norm.weight",         "k_norm.weight",     (depth_hd,)),
            ("post_attention_layernorm.weight", "ffn_norm.weight",   None),
            ("mlp.gate_proj.weight",            "ffn_gate.weight",   expect_gate),
            ("mlp.up_proj.weight",              "ffn_up.weight",     expect_up),
            ("mlp.down_proj.weight",            "ffn_down.weight",   expect_down),
        ]:
            src = f"{prefix_in}.{src_suf}"
            if src not in sd:
                raise RuntimeError(f"missing tensor: {src}")
            arr = sd[src]
            if expect is not None and arr.shape != expect:
                raise RuntimeError(f"{src} shape {arr.shape} != {expect}")
            dt = "F32" if dst_suf.endswith("_norm.weight") else "F16"
            writer.add_tensor(f"{prefix_out}.{dst_suf}",
                              arr.astype(np.float32), st_dtype=dt)

    # final norm
    if "local_transformer.model.norm.weight" not in sd:
        raise RuntimeError("missing local_transformer.model.norm.weight")
    writer.add_tensor("lm.depth.output_norm.weight",
                      sd["local_transformer.model.norm.weight"].astype(np.float32),
                      st_dtype="F32")

    # --- compose-side audio embeds (outer embed_tokens.{1..16}) ---------
    # `embed_tokens.0` is the text embed (151936 × 2048) — lives in the
    # backbone GGUF as `model.embed_tokens.weight` via the backbone
    # converter.  embed_tokens.{1..16} are the per-cb audio embeds used
    # to build the BACKBONE input at the next AR step.  We bake them
    # here as a single fused (rvq * audio_vocab, backbone_hid) compose
    # table indexed by `cb * audio_vocab + code`; runtime sums over the
    # 16 audio cb to produce the audio portion of next-step input.  The
    # driver adds the text embed on top of this (via the host LLM's
    # `token_embd.weight` for the sampled text token).
    rows = []
    for i in range(rvq):
        key = f"embed_tokens.{i + 1}.weight"
        if key not in sd:
            raise RuntimeError(f"missing tensor: {key}")
        arr = sd[key]
        if arr.shape != (audio_vocab, backbone_hid):
            raise RuntimeError(
                f"{key} shape {arr.shape} != ({audio_vocab}, {backbone_hid})")
        rows.append(arr.astype(np.float32))
    compose_w = np.concatenate(rows, axis=0)  # (rvq * audio_vocab, backbone_hid)
    writer.add_tensor("lm.compose.audio_embd.weight", compose_w, st_dtype="F16")
    writer.add_uint32("codec.lm.compose.audio_embed_dim", backbone_hid)
    writer.add_uint32("codec.lm.compose.codebook_stride", audio_vocab)

    # --- prompt-side metadata for the driver ----------------------------
    # Caller-facing fields — codec_lm itself doesn't consume them but
    # writing them keeps the GGUF self-describing for the host driver
    # (tts-cli / rn-tts).
    for k in ("text_pad", "reference_audio_pad", "audio_pad_token"):
        if k in cfg:
            writer.add_uint32(f"codec.lm.{k}", int(cfg[k]))

    if verbose:
        print(f"[lm_adaptor:moss_tts_local:realtime] residual_depth_ar: "
              f"n_codebook={n_codebook} backbone_h={backbone_hid} "
              f"depth=({depth_layers}L, {depth_hidden}h, "
              f"{depth_nh}q/{depth_nkvh}kv heads, head_dim={depth_hd}, "
              f"ffn={depth_inter}, audio_vocab={audio_vocab}) "
              f"compose=({backbone_hid}d, fused)")


# ---------------------------------------------------------------------
# MOSS-TTS-Nano-100M (stub for now — GPT-2 depth-block runtime pending)
# ---------------------------------------------------------------------

def _dump_nano(writer, sd, cfg, *, verbose: bool) -> None:
    raise NotImplementedError(
        "MOSS-TTS-Nano dump: pending `codec.lm.depth.arch=gpt2` runtime "
        "(LayerNorm + GELU + fused c_attn split + absolute wpe).  Will "
        "land in a follow-up commit."
    )
