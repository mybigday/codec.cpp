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

    if verbose:
        print(f"[lm_adaptor:qwen3_tts] residual_depth_ar: "
              f"n_codebook={n_codebook} "
              f"talker_hidden={talker_hidden} depth=({depth_layers}L, "
              f"{depth_hidden}h, {depth_nh}q/{depth_nkvh}kv heads, "
              f"head_dim={depth_hd}, ffn={depth_inter}, "
              f"rope_theta={depth_rope}, has_in_proj={has_in_proj})")
