"""Qwen3-TTS LM-adaptor dump — `residual_depth_ar` kind.

Covers `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,VoiceDesign}`,
which all share the same talker + code-predictor topology:

  - Talker (backbone): Qwen3-style transformer with 3D MRoPE.  Lives
    in llama.cpp; gated on upstream MRoPE support on the `qwen3` arch
    (see CLAUDE.md).
  - Code predictor (depth decoder): 5-layer Qwen3 transformer with
    per-position input embeddings + per-position lm_head linears.
    No MRoPE (1D RoPE), QK-norm on the head dim, optional in_proj
    (small_to_mtp_projection) with bias when talker.hidden !=
    code_predictor.hidden.

This handler writes only the codec_lm side — backbone weights are
expected to be extracted separately for llama.cpp consumption.

Tensor mapping (HF → codec_lm):

  talker.codec_head.weight                    (V_c0, H_t)
    → lm.c0_head.weight                       (V_c0, H_t)

  talker.model.codec_embedding.weight         (V_c0, H_t)
    → lm.audio_embd_0.weight                  (V_c0, H_t)
      Talker's c0 input embedding.  At depth position 1 the runtime
      uses this table to embed c0 (matches HF's
      `forward_sub_talker_finetune` and the `generate` prefill at
      `inputs_embeds[1] = self.get_input_embeddings()(c0)`).  Also
      the c0 row used by compose_audio_embd for the next backbone
      input.

  talker.code_predictor.model.codec_embedding.{i}.weight
                                              (V, H_d)  for i=0..N-2
    → lm.audio_embd_{i+1}.weight              (V, H_d)
      Per-position depth input embeddings.  At depth position j+1
      (j≥1), the runtime looks up `audio_embd_{j}` at code c_{j}.

  talker.code_predictor.lm_head.{i}.weight    (V, H_d)  for i=0..N-2
    → lm.depth.heads_{i}.weight               (V, H_d)
      Per-position depth heads.  `heads[k-1]` is applied to depth
      position k's hidden to produce c_k logits.

  talker.code_predictor.model.inputs_embeds_projector{,...} — only
  emitted in 1.7B variants where talker.hidden != cp.hidden.  For
  0.6B (both share hidden=1024), the HF `small_to_mtp_projection`
  is an `nn.Identity()` and NOT stored — we set
  `codec.lm.residual.depth_has_in_proj=false`.

  Standard Qwen3 transformer block weights for each of the
  `num_hidden_layers` layers of the code predictor:
    code_predictor.model.layers.{l}.{
        self_attn.{q,k,v,o}_proj.weight,
        self_attn.{q,k}_norm.weight,
        mlp.{gate,up,down}_proj.weight,
        input_layernorm.weight,
        post_attention_layernorm.weight
    }
    → lm.depth.blk_{l}.{q,k,v,o,q_norm,k_norm,
                        ffn_gate,ffn_up,ffn_down,
                        attn_norm,ffn_norm}.weight

  code_predictor.model.norm.weight            → lm.depth.output_norm.weight

Metadata (`codec.lm.*`):

  has_adaptor                = true
  kind                       = "residual_depth_ar"
  host_arch                  = "qwen3"
  hidden_dim                 = talker.hidden_size
  audio_embed_dim            = talker.hidden_size       (== H_d for 0.6B)
  n_codebook                 = talker_cfg.num_code_groups (16 for 0.6B)
  codebook_sizes             = [V_c0] + [V] * (N-1)     (non-homogeneous)
  delay_pattern              = [0] * N                  (no inter-cb delay)
  parallel.tied_heads_to_embd= false                    (per-cb heads + embds
                                                         are independent weights)
  residual.depth_layers      = cp_cfg.num_hidden_layers
  residual.depth_hidden      = cp_cfg.hidden_size
  residual.depth_n_heads     = cp_cfg.num_attention_heads
  residual.depth_n_kv_heads  = cp_cfg.num_key_value_heads
  residual.depth_head_dim    = cp_cfg.head_dim
  residual.depth_intermediate= cp_cfg.intermediate_size
  residual.depth_rope_theta  = cp_cfg.rope_theta
  residual.depth_rms_norm_eps= cp_cfg.rms_norm_eps
  residual.depth_max_position= cp_cfg.max_position_embeddings
  residual.depth_has_in_proj = (talker.hidden != cp.hidden)
  residual.depth_has_qk_norm = true                     (Qwen3 family)
  residual.weight_layout     = "shared"
  residual.c0_input_modality = "audio"
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def dump(writer, sd: Dict[str, np.ndarray], cfg: Dict[str, Any],
         *, verbose: bool = False) -> None:
    archs = cfg.get("architectures") or []
    if "Qwen3TTSForConditionalGeneration" not in archs:
        raise RuntimeError(
            f"Qwen3-TTS lm-adaptor dispatched on unexpected architectures {archs!r}"
        )

    tk_cfg = cfg["talker_config"]
    cp_cfg = tk_cfg["code_predictor_config"]

    talker_hidden     = int(tk_cfg["hidden_size"])
    talker_vocab      = int(tk_cfg["vocab_size"])
    n_codebook        = int(tk_cfg["num_code_groups"])
    depth_hidden      = int(cp_cfg["hidden_size"])
    depth_layers      = int(cp_cfg["num_hidden_layers"])
    depth_nh          = int(cp_cfg["num_attention_heads"])
    depth_nkvh        = int(cp_cfg["num_key_value_heads"])
    depth_hd          = int(cp_cfg["head_dim"])
    depth_inter       = int(cp_cfg["intermediate_size"])
    depth_eps         = float(cp_cfg["rms_norm_eps"])
    depth_rope        = float(cp_cfg["rope_theta"])
    depth_maxpos      = int(cp_cfg["max_position_embeddings"])
    depth_vocab       = int(cp_cfg["vocab_size"])

    has_in_proj = (talker_hidden != depth_hidden)

    # --- metadata --------------------------------------------------------
    writer.add_bool  ("codec.lm.has_adaptor",     True)
    writer.add_string("codec.lm.kind",            "residual_depth_ar")
    writer.add_string("codec.lm.host_arch",       "qwen3")
    writer.add_uint32("codec.lm.hidden_dim",      talker_hidden)
    writer.add_uint32("codec.lm.audio_embed_dim", talker_hidden)
    writer.add_uint32("codec.lm.n_codebook",      n_codebook)
    # cb 0 is talker-vocab (includes specials); cb 1..N-1 are codec-only.
    codebook_sizes = [talker_vocab] + [depth_vocab] * (n_codebook - 1)
    writer.add_array ("codec.lm.codebook_sizes",  codebook_sizes)
    writer.add_array ("codec.lm.delay_pattern",   [0] * n_codebook)
    writer.add_bool  ("codec.lm.parallel.tied_heads_to_embd", False)

    # End-of-audio: the talker stops when codebook-0 samples
    # `codec_eos_token_id` (2150 for 0.6B).  Also record bos/pad codes
    # since the config carries them.  All read from the checkpoint's own
    # talker_config.  eos_min_step=0 (honored from the first frame).
    if "codec_eos_token_id" in tk_cfg:
        writer.add_int32 ("codec.lm.eos_code_c0", int(tk_cfg["codec_eos_token_id"]))
        writer.add_int32 ("codec.lm.eos_min_step", 0)
    if "codec_bos_id" in tk_cfg:
        writer.add_int32 ("codec.lm.bos_code_c0", int(tk_cfg["codec_bos_id"]))
    if "codec_pad_id" in tk_cfg:
        writer.add_int32 ("codec.lm.pad_code_c0", int(tk_cfg["codec_pad_id"]))

    # --- talker prompt-assembly control tags (Qwen3-TTS specific) --------
    # The talker prefix interleaves two lanes summed position-wise:
    #   text lane  = text_projection(text_embedding[text_tok])   (2048->1024)
    #   codec lane = codec_embedding[control_tag]                (already 1024)
    # The codec-tag stream (auto-language) is:
    #   [nothink, think_bos, think_eos, <X-VECTOR>, pad, bos]
    # and the text lane is [tts_pad * (len-2), tts_bos] aligned to it, then
    # the first payload text token is summed with codec_bos.  See
    # modeling_qwen3_tts.py::Qwen3TTSTalkerModel.generate.  These ids come
    # straight from the checkpoint's talker_config so the runtime can
    # assemble the prefix without re-reading config.json.
    for meta_key, cfg_key in [
        ("codec.lm.qwen3tts.nothink_id",   "codec_nothink_id"),
        ("codec.lm.qwen3tts.think_id",     "codec_think_id"),
        ("codec.lm.qwen3tts.think_bos_id", "codec_think_bos_id"),
        ("codec.lm.qwen3tts.think_eos_id", "codec_think_eos_id"),
        ("codec.lm.qwen3tts.tts_pad_id",   "tts_pad_token_id"),
        ("codec.lm.qwen3tts.tts_bos_id",   "tts_bos_token_id"),
        ("codec.lm.qwen3tts.tts_eos_id",   "tts_eos_token_id"),
    ]:
        val = tk_cfg.get(cfg_key, cfg.get(cfg_key))
        if val is not None:
            writer.add_int32(meta_key, int(val))
    # Language-id map (codec-vocab tokens looked up in codec_embedding).
    # The runtime default is "auto" (no explicit language → 3-tag nothink
    # form), so only the common per-language ids are surfaced as scalars for
    # an optional explicit-language path.  (String-keyed maps aren't cleanly
    # representable as GGUF metadata the runtime reads.)
    lang_map = tk_cfg.get("codec_language_id", cfg.get("codec_language_id")) or {}
    for lang in ("chinese", "english"):
        if lang in lang_map:
            writer.add_int32(f"codec.lm.qwen3tts.language_{lang}",
                             int(lang_map[lang]))

    writer.add_uint32 ("codec.lm.residual.depth_layers",     depth_layers)
    writer.add_uint32 ("codec.lm.residual.depth_hidden",     depth_hidden)
    writer.add_uint32 ("codec.lm.residual.depth_n_heads",    depth_nh)
    writer.add_uint32 ("codec.lm.residual.depth_n_kv_heads", depth_nkvh)
    writer.add_uint32 ("codec.lm.residual.depth_head_dim",   depth_hd)
    writer.add_uint32 ("codec.lm.residual.depth_intermediate", depth_inter)
    writer.add_float32("codec.lm.residual.depth_rope_theta", depth_rope)
    writer.add_uint32 ("codec.lm.residual.depth_max_position", depth_maxpos)
    writer.add_float32("codec.lm.residual.depth_rms_norm_eps", depth_eps)
    writer.add_bool   ("codec.lm.residual.depth_has_in_proj", has_in_proj)
    writer.add_bool   ("codec.lm.residual.depth_has_qk_norm", True)
    writer.add_string ("codec.lm.residual.weight_layout",      "shared")
    writer.add_string ("codec.lm.residual.c0_input_modality",  "audio")

    # --- audio embedding tables -----------------------------------------
    # audio_embd_0 = talker.model.codec_embedding (V_c0=talker_vocab, H_t).
    talker_embd = sd["talker.model.codec_embedding.weight"]
    if talker_embd.shape != (talker_vocab, talker_hidden):
        raise RuntimeError(
            f"talker.codec_embedding shape {talker_embd.shape} != "
            f"({talker_vocab}, {talker_hidden})"
        )
    writer.add_tensor("lm.audio_embd_0.weight",
                      talker_embd.astype(np.float32), st_dtype="F16")

    # audio_embd_{i+1} = code_predictor.codec_embedding[i] (V=depth_vocab, H_d).
    # ModuleList has n_codebook - 1 entries.
    for i in range(n_codebook - 1):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if key not in sd:
            raise RuntimeError(f"missing tensor: {key}")
        arr = sd[key]
        if arr.shape != (depth_vocab, depth_hidden):
            raise RuntimeError(
                f"{key} shape {arr.shape} != ({depth_vocab}, {depth_hidden})"
            )
        writer.add_tensor(f"lm.audio_embd_{i+1}.weight",
                          arr.astype(np.float32), st_dtype="F16")

    # --- talker text projection + text embedding ------------------------
    # The talker prompt projects backbone/text-vocab embeds (2048) into
    # talker hidden (1024) via a 2-layer SiLU MLP (Qwen3TTSTalkerResizeMLP:
    # linear_fc2(silu(linear_fc1(x))), bias=True), then sums them with the
    # codec-tag lane.  Both the projection weights and the talker's own text
    # embedding table (151936 x 2048) are needed runtime-side, so bake them.
    for src, dst, expect in [
        ("talker.text_projection.linear_fc1.weight", "lm.text_projection.fc1.weight",
         (talker_hidden * 2, 2048)),
        ("talker.text_projection.linear_fc1.bias",   "lm.text_projection.fc1.bias",   None),
        ("talker.text_projection.linear_fc2.weight", "lm.text_projection.fc2.weight",
         (talker_hidden, talker_hidden * 2)),
        ("talker.text_projection.linear_fc2.bias",   "lm.text_projection.fc2.bias",   None),
    ]:
        if src not in sd:
            raise RuntimeError(f"missing tensor: {src}")
        arr = sd[src]
        if expect is not None and arr.shape != expect:
            raise RuntimeError(f"{src} shape {arr.shape} != expected {expect}")
        dt = "F32" if dst.endswith(".bias") else "F16"
        writer.add_tensor(dst, arr.astype(np.float32), st_dtype=dt)

    txt_embd = sd["talker.model.text_embedding.weight"]
    text_vocab = int(txt_embd.shape[0])
    writer.add_uint32("codec.lm.qwen3tts.text_vocab", text_vocab)
    writer.add_uint32("codec.lm.qwen3tts.text_embed_dim", int(txt_embd.shape[1]))
    writer.add_tensor("lm.text_embd.weight", txt_embd.astype(np.float32), st_dtype="F16")

    # --- c0 head --------------------------------------------------------
    c0 = sd["talker.codec_head.weight"]
    if c0.shape != (talker_vocab, talker_hidden):
        raise RuntimeError(
            f"talker.codec_head shape {c0.shape} != ({talker_vocab}, {talker_hidden})"
        )
    writer.add_tensor("lm.c0_head.weight", c0.astype(np.float32), st_dtype="F16")

    # --- depth heads ----------------------------------------------------
    for i in range(n_codebook - 1):
        key = f"talker.code_predictor.lm_head.{i}.weight"
        if key not in sd:
            raise RuntimeError(f"missing tensor: {key}")
        arr = sd[key]
        if arr.shape != (depth_vocab, depth_hidden):
            raise RuntimeError(
                f"{key} shape {arr.shape} != ({depth_vocab}, {depth_hidden})"
            )
        writer.add_tensor(f"lm.depth.heads_{i}.weight",
                          arr.astype(np.float32), st_dtype="F16")

    # --- in_proj (optional) ---------------------------------------------
    if has_in_proj:
        wkey = "talker.code_predictor.small_to_mtp_projection.weight"
        bkey = "talker.code_predictor.small_to_mtp_projection.bias"
        if wkey not in sd:
            raise RuntimeError(
                f"missing tensor: {wkey} "
                f"(expected when talker.hidden={talker_hidden} != "
                f"depth.hidden={depth_hidden})"
            )
        w = sd[wkey]
        if w.shape != (depth_hidden, talker_hidden):
            raise RuntimeError(
                f"{wkey} shape {w.shape} != ({depth_hidden}, {talker_hidden})"
            )
        writer.add_tensor("lm.depth.in_proj.weight",
                          w.astype(np.float32), st_dtype="F16")
        if bkey in sd:
            b = sd[bkey]
            if b.shape != (depth_hidden,):
                raise RuntimeError(
                    f"{bkey} shape {b.shape} != ({depth_hidden},)"
                )
            writer.add_tensor("lm.depth.in_proj.bias",
                              b.astype(np.float32), st_dtype="F32")

    # --- depth transformer layers ---------------------------------------
    # Qwen3 attention: head_dim * n_heads != hidden in general (here
    # 128 * 16 = 2048 vs hidden=1024 for the 0.6B variant — the
    # attention output projects from (n_heads * head_dim) back into
    # hidden via o_proj).
    expect_q    = (depth_nh   * depth_hd, depth_hidden)
    expect_kv   = (depth_nkvh * depth_hd, depth_hidden)
    expect_o    = (depth_hidden, depth_nh * depth_hd)
    expect_gate = (depth_inter, depth_hidden)
    expect_up   = (depth_inter, depth_hidden)
    expect_down = (depth_hidden, depth_inter)

    for l in range(depth_layers):
        prefix_in  = f"talker.code_predictor.model.layers.{l}"
        prefix_out = f"lm.depth.blk_{l}"

        for src_suf, dst_suf, expect in [
            ("input_layernorm.weight",         "attn_norm.weight",  None),
            ("self_attn.q_proj.weight",        "q.weight",          expect_q),
            ("self_attn.k_proj.weight",        "k.weight",          expect_kv),
            ("self_attn.v_proj.weight",        "v.weight",          expect_kv),
            ("self_attn.o_proj.weight",        "o.weight",          expect_o),
            ("self_attn.q_norm.weight",        "q_norm.weight",     (depth_hd,)),
            ("self_attn.k_norm.weight",        "k_norm.weight",     (depth_hd,)),
            ("post_attention_layernorm.weight","ffn_norm.weight",   None),
            ("mlp.gate_proj.weight",           "ffn_gate.weight",   expect_gate),
            ("mlp.up_proj.weight",             "ffn_up.weight",     expect_up),
            ("mlp.down_proj.weight",           "ffn_down.weight",   expect_down),
        ]:
            src = f"{prefix_in}.{src_suf}"
            if src not in sd:
                raise RuntimeError(f"missing tensor in Qwen3-TTS checkpoint: {src}")
            arr = sd[src]
            if expect is not None and arr.shape != expect:
                raise RuntimeError(
                    f"{src} shape {arr.shape} != expected {expect}"
                )
            dt = "F32" if dst_suf.endswith("_norm.weight") else "F16"
            writer.add_tensor(f"{prefix_out}.{dst_suf}",
                              arr.astype(np.float32), st_dtype=dt)

    out_norm = sd["talker.code_predictor.model.norm.weight"]
    writer.add_tensor("lm.depth.output_norm.weight",
                      out_norm.astype(np.float32), st_dtype="F32")

    # --- speaker_encoder (ECAPA-TDNN) ----------------------------------
    # Bundled only if the HF checkpoint exposed it (some Qwen3-TTS
    # variants are zero-shot — no x-vector speaker encoder).
    _dump_qwen3_tts_speaker_encoder(writer, sd, cfg, verbose=verbose)

    if verbose:
        print(f"[lm_adaptor:qwen3_tts] residual_depth_ar: "
              f"n_codebook={n_codebook} "
              f"talker_hidden={talker_hidden} depth=({depth_layers}L, "
              f"{depth_hidden}h, {depth_nh}q/{depth_nkvh}kv heads, "
              f"head_dim={depth_hd}, ffn={depth_inter}, "
              f"rope_theta={depth_rope}, has_in_proj={has_in_proj})")


def _dump_qwen3_tts_speaker_encoder(writer, sd, cfg, *, verbose=False):
    """Bundle Qwen3-TTS's ECAPA-TDNN speaker encoder + mel front-end
    constants into the same GGUF.

    Tensor namespace (matches what `src/lm/speaker_qwen3_tts.cpp`
    consumes — block_idx mirrors `Qwen3TTSSpeakerEncoder.blocks`):

      speaker.qwen3_tts.blocks.0.conv.{weight,bias}        ; TDNN (k=5)
      speaker.qwen3_tts.blocks.{1..N-2}.tdnn1.conv.{w,b}   ; SE-Res2Net
      speaker.qwen3_tts.blocks.{1..N-2}.res2net.{0..S-2}.conv.{w,b}
      speaker.qwen3_tts.blocks.{1..N-2}.tdnn2.conv.{w,b}
      speaker.qwen3_tts.blocks.{1..N-2}.se.{conv1,conv2}.{w,b}
      speaker.qwen3_tts.mfa.conv.{weight,bias}             ; MFA TDNN
      speaker.qwen3_tts.asp.tdnn.conv.{w,b}                ; ASP
      speaker.qwen3_tts.asp.conv.{w,b}
      speaker.qwen3_tts.fc.{weight,bias}                   ; final Conv1d
      speaker.qwen3_tts.mel_basis                          ; (n_mels, n_freq)
      speaker.qwen3_tts.window                             ; (win,)
    """
    # The speaker_encoder lives on the talker side of the model.  Detect
    # by probing for the initial TDNN weight; if missing, the checkpoint
    # is zero-shot and we skip.
    probe_key = "speaker_encoder.blocks.0.conv.weight"
    if probe_key not in sd:
        if verbose:
            print(f"[lm_adaptor:qwen3_tts] no speaker_encoder in checkpoint "
                  f"(zero-shot variant); skipping speaker section.")
        return

    # speaker_encoder_config lives at top-level of Qwen3-TTS's config.
    # Most fields are dataclass defaults from Qwen3TTSSpeakerEncoderConfig
    # — the published config only overrides enc_dim / sample_rate.
    se_cfg = cfg.get("speaker_encoder_config") or {}
    mel_dim       = int(se_cfg.get("mel_dim", 128))
    enc_dim       = int(se_cfg.get("enc_dim", 1024))
    enc_channels  = list(se_cfg.get("enc_channels", [512, 512, 512, 512, 1536]))
    enc_kernels   = list(se_cfg.get("enc_kernel_sizes", [5, 3, 3, 3, 1]))
    enc_dilations = list(se_cfg.get("enc_dilations", [1, 2, 3, 4, 1]))
    enc_attn_ch   = int(se_cfg.get("enc_attention_channels", 128))
    enc_res2net   = int(se_cfg.get("enc_res2net_scale", 8))
    enc_se_ch     = int(se_cfg.get("enc_se_channels", 128))
    sample_rate   = int(se_cfg.get("sample_rate", 24000))

    # Mel front-end defaults (per upstream `mel_spectrogram` in
    # modeling_qwen3_tts.py — n_fft=1024, hop=256, win=1024, fmin=0,
    # fmax=None → 12000, center=False, Slaney mel).
    n_fft   = 1024
    hop     = 256
    win     = 1024
    fmin    = 0
    fmax    = sample_rate // 2     # librosa default when fmax=None
    center  = False

    n_blocks = len(enc_channels)   # 5 by default — block 0 + (N-2) SE-Res2Net + 1 MFA

    def _emit(name, arr, st_dtype="F16"):
        a = arr.astype(np.float32, copy=False)
        writer.add_tensor(name, a,
                          st_dtype="F32" if name.endswith((".bias", "norm.weight")) else st_dtype)

    # block 0: TDNN (in=mel_dim, out=enc_channels[0], k=enc_kernels[0])
    _emit("speaker.qwen3_tts.blocks.0.conv.weight",
          sd["speaker_encoder.blocks.0.conv.weight"])
    _emit("speaker.qwen3_tts.blocks.0.conv.bias",
          sd["speaker_encoder.blocks.0.conv.bias"])

    # blocks 1..N-2: SE-Res2Net
    for bi in range(1, n_blocks - 1):
        p = f"speaker_encoder.blocks.{bi}"
        _emit(f"speaker.qwen3_tts.blocks.{bi}.tdnn1.conv.weight", sd[f"{p}.tdnn1.conv.weight"])
        _emit(f"speaker.qwen3_tts.blocks.{bi}.tdnn1.conv.bias",   sd[f"{p}.tdnn1.conv.bias"])
        for ri in range(enc_res2net - 1):
            _emit(f"speaker.qwen3_tts.blocks.{bi}.res2net.{ri}.conv.weight",
                  sd[f"{p}.res2net_block.blocks.{ri}.conv.weight"])
            _emit(f"speaker.qwen3_tts.blocks.{bi}.res2net.{ri}.conv.bias",
                  sd[f"{p}.res2net_block.blocks.{ri}.conv.bias"])
        _emit(f"speaker.qwen3_tts.blocks.{bi}.tdnn2.conv.weight", sd[f"{p}.tdnn2.conv.weight"])
        _emit(f"speaker.qwen3_tts.blocks.{bi}.tdnn2.conv.bias",   sd[f"{p}.tdnn2.conv.bias"])
        _emit(f"speaker.qwen3_tts.blocks.{bi}.se.conv1.weight",   sd[f"{p}.se_block.conv1.weight"])
        _emit(f"speaker.qwen3_tts.blocks.{bi}.se.conv1.bias",     sd[f"{p}.se_block.conv1.bias"])
        _emit(f"speaker.qwen3_tts.blocks.{bi}.se.conv2.weight",   sd[f"{p}.se_block.conv2.weight"])
        _emit(f"speaker.qwen3_tts.blocks.{bi}.se.conv2.bias",     sd[f"{p}.se_block.conv2.bias"])

    # MFA TDNN (operates on concat of SE-Res2Net outputs):
    _emit("speaker.qwen3_tts.mfa.conv.weight", sd["speaker_encoder.mfa.conv.weight"])
    _emit("speaker.qwen3_tts.mfa.conv.bias",   sd["speaker_encoder.mfa.conv.bias"])

    # ASP
    _emit("speaker.qwen3_tts.asp.tdnn.conv.weight", sd["speaker_encoder.asp.tdnn.conv.weight"])
    _emit("speaker.qwen3_tts.asp.tdnn.conv.bias",   sd["speaker_encoder.asp.tdnn.conv.bias"])
    _emit("speaker.qwen3_tts.asp.conv.weight",      sd["speaker_encoder.asp.conv.weight"])
    _emit("speaker.qwen3_tts.asp.conv.bias",        sd["speaker_encoder.asp.conv.bias"])

    # Final fc Conv1d
    _emit("speaker.qwen3_tts.fc.weight", sd["speaker_encoder.fc.weight"])
    _emit("speaker.qwen3_tts.fc.bias",   sd["speaker_encoder.fc.bias"])

    # Mel basis (slaney) + window (periodic Hann).  Baked at convert time
    # so the runtime has no librosa/scipy dep.
    import librosa
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=mel_dim,
        fmin=fmin, fmax=fmax, htk=False, norm="slaney",
    ).astype(np.float32)
    window = np.asarray(
        0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(win, dtype=np.float64) / win),
        dtype=np.float32,
    )
    writer.add_tensor("speaker.qwen3_tts.mel_basis", mel_basis, st_dtype="F32")
    writer.add_tensor("speaker.qwen3_tts.window",    window,    st_dtype="F32")

    # ---- Metadata block ----------------------------------------------
    # Qwen3-TTS uses `speaker_emb` as a single row in the talker prompt,
    # so n_rows = 1, hidden_dim = enc_dim (1024).  Talker hidden is
    # also 1024 for the 0.6B base, so the output is directly consumable.
    talker_hidden = int(cfg["talker_config"]["hidden_size"])

    writer.add_bool   ("codec.speaker.has_encoder", True)
    writer.add_uint32 ("codec.speaker.n_rows", 1)
    writer.add_uint32 ("codec.speaker.hidden_dim", talker_hidden)
    writer.add_bool   ("codec.speaker.needs_ref_pcm", True)
    writer.add_bool   ("codec.speaker.needs_ref_speech_tokens", False)
    writer.add_bool   ("codec.speaker.needs_emotion_scalar", False)
    writer.add_uint32 ("codec.speaker.ref_sample_rate", sample_rate)
    writer.add_uint32 ("codec.speaker.speaker_emb_dim", enc_dim)
    writer.add_float32("codec.speaker.emotion_default", 0.5)
    writer.add_string ("codec.speaker.encoder_arch", "qwen3_tts_ecapa_tdnn")

    # ECAPA-TDNN op constants the runtime reads to size graphs.
    writer.add_uint32 ("codec.speaker.ecapa.mel_dim", mel_dim)
    writer.add_array  ("codec.speaker.ecapa.enc_channels", enc_channels)
    writer.add_array  ("codec.speaker.ecapa.enc_kernel_sizes", enc_kernels)
    writer.add_array  ("codec.speaker.ecapa.enc_dilations", enc_dilations)
    writer.add_uint32 ("codec.speaker.ecapa.enc_attention_channels", enc_attn_ch)
    writer.add_uint32 ("codec.speaker.ecapa.enc_res2net_scale", enc_res2net)
    writer.add_uint32 ("codec.speaker.ecapa.enc_se_channels", enc_se_ch)
    writer.add_uint32 ("codec.speaker.ecapa.enc_dim", enc_dim)
    writer.add_uint32 ("codec.speaker.ecapa.n_fft", n_fft)
    writer.add_uint32 ("codec.speaker.ecapa.hop_size", hop)
    writer.add_uint32 ("codec.speaker.ecapa.win_size", win)
    writer.add_uint32 ("codec.speaker.ecapa.fmin", fmin)
    writer.add_uint32 ("codec.speaker.ecapa.fmax", fmax)
    writer.add_bool   ("codec.speaker.ecapa.center", center)

    if verbose:
        print(f"[lm_adaptor:qwen3_tts] speaker_encoder ECAPA-TDNN: "
              f"mel_dim={mel_dim} enc_dim={enc_dim} "
              f"channels={enc_channels} (1+{n_blocks-2}+1) "
              f"@ sr={sample_rate} → (1, {talker_hidden})")
