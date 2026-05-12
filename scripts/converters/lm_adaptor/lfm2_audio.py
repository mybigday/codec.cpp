"""LFM2-Audio LM-adaptor dump — `residual_depth_ar` with shared layers
+ per-position in_proj (3D) + per-cb pre-head RMSNorm.

Covers `LiquidAI/LFM2-Audio-1.5B` (and the newer LFM2.5-Audio variants).
Reference implementation: the `liquid_audio` pypi package
(`liquid_audio.model.lfm2_audio.LFM2AudioModel._sample_audio_frame`).

Architecture in one paragraph:
  - 6-layer SwiGLU transformer (`depthformer`) with GQA (32 heads, 8 kv
    heads, head_dim=32), QK-norm on the head dim, RoPE θ=1e6.  Weights
    are SHARED across all N=8 depth positions.
  - Pre-step setup: `depth_linear` is a single (8192, 2048) Linear with
    bias that projects the LFM hidden into 8 stacked per-position
    contexts of dim 1024.  We store it as a 3D `(N, depth_hidden,
    hidden_dim)` per-position in_proj (same format as Moshi's flexible
    in_proj), with a 2D `(depth_hidden, N)` bias view.
  - Step k input embedding = audio_embd[k-1][c_{k-1}] (or zero at k=0)
    + in_proj[k] @ h_in.  No text embed (c0_input_modality="none").
  - Step k head = `to_logits[k] @ embedding_norm[k](depth_h_last)` —
    per-cb 2D head with its own pre-head RMSNorm.  Depth emits ALL N
    codes (no separate c0_head).

  - SharedEmbedding combines `embedding`, `embedding_norm`, and
    `to_logits` per cb.  Optionally `tie_embedding=True` ties embedding
    and to_logits to the same parameter; on `LFM2-Audio-1.5B` the
    config says `depthformer.tie=true` so the converter sanity-checks
    tensor equality and emits separate tensors (the runtime always
    addresses them as two distinct names; ties are an HF storage
    optimisation, not a runtime contract).

Tensor mapping (HF -> codec_lm):

  depth_linear.weight  (8192, 2048)
    -> lm.depth.in_proj.weight  (8, 1024, 2048)   [3D, per-pos]
  depth_linear.bias    (8192,)
    -> lm.depth.in_proj.bias    (1024, 8)         [2D, per-pos slice]

  depth_embeddings.{i}.embedding.weight       (2049, 1024)  i=0..N-1
    -> lm.depth.audio_embd_{i}.weight         (2049, 1024)
  depth_embeddings.{i}.to_logits.weight       (2049, 1024)
    -> lm.depth.heads_{i}.weight              (2049, 1024)
  depth_embeddings.{i}.embedding_norm.weight  (1024,)
    -> lm.depth.heads_{i}_norm.weight         (1024,)

  depthformer.layers.{l}.operator.qkv_proj.weight  (1536, 1024)
    -> split into q (1024, 1024) + k (256, 1024) + v (256, 1024)
    -> lm.depth.blk_{l}.q.weight, .k.weight, .v.weight
  depthformer.layers.{l}.operator.out_proj.weight  (1024, 1024)
    -> lm.depth.blk_{l}.o.weight
  depthformer.layers.{l}.operator.bounded_attention.q_layernorm.weight  (32,)
    -> lm.depth.blk_{l}.q_norm.weight  (head_dim)
  depthformer.layers.{l}.operator.bounded_attention.k_layernorm.weight  (32,)
    -> lm.depth.blk_{l}.k_norm.weight
  depthformer.layers.{l}.operator_norm.weight  (1024,)
    -> lm.depth.blk_{l}.attn_norm.weight
  depthformer.layers.{l}.ffn_norm.weight       (1024,)
    -> lm.depth.blk_{l}.ffn_norm.weight
  depthformer.layers.{l}.feed_forward.w1.weight  (2816, 1024)
    -> lm.depth.blk_{l}.ffn_gate.weight
  depthformer.layers.{l}.feed_forward.w3.weight  (2816, 1024)
    -> lm.depth.blk_{l}.ffn_up.weight
  depthformer.layers.{l}.feed_forward.w2.weight  (1024, 2816)
    -> lm.depth.blk_{l}.ffn_down.weight

Metadata (`codec.lm.*`):

  kind                       = "residual_depth_ar"
  host_arch                  = "lfm2"
  hidden_dim                 = lfm.hidden_size           (2048)
  audio_embed_dim            = depthformer.dim           (1024 — depth space)
  n_codebook                 = config.codebooks          (8)
  codebook_sizes             = [audio_vocab_size]*N      (2049)
  delay_pattern              = [0]*N
  parallel.tied_heads_to_embd= depthformer.tie

  residual.weight_layout       = "shared"
  residual.c0_input_modality   = "none"         (no embed at pos 0)
  residual.depth_emits_c0      = true           (all N cb come from depth)
  residual.depth_in_proj_per_pos = true         (lm.depth.in_proj.weight is 3D)
  residual.depth_in_proj_has_bias= true         (lm.depth.in_proj.bias is set)
  residual.depth_has_pre_head_norm = true       (per-cb RMSNorm before head)
  residual.depth_has_in_proj   = true
  residual.depth_has_qk_norm   = true
  residual.depth_has_output_norm = false        (depthformer has no final norm)
  residual.depth_use_rope      = true
  residual.depth_layers / hidden / n_heads / n_kv_heads / head_dim /
    intermediate / rope_theta / rms_norm_eps / max_position
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def dump(writer, sd: Dict[str, np.ndarray], cfg: Dict[str, Any],
         *, verbose: bool = False) -> None:
    archs = cfg.get("architectures") or []
    if "Lfm2AudioForConditionalGeneration" not in archs:
        raise RuntimeError(
            f"LFM2-Audio lm-adaptor dispatched on unexpected architectures {archs!r}"
        )

    lfm_cfg          = cfg["lfm"]
    dpf_cfg          = cfg["depthformer"]

    n_codebook       = int(cfg["codebooks"])
    audio_vocab      = int(cfg.get("audio_vocab_size", 2048)) + 1  # +1 for EOAudio (matches LFM2AudioModel.audio_vocab_size)
    backbone_hid     = int(lfm_cfg["hidden_size"])
    depth_hidden     = int(dpf_cfg["dim"])
    depth_layers     = int(dpf_cfg["layers"])
    depth_tie        = bool(dpf_cfg.get("tie", False))

    # MHA defaults (from liquid_audio.model.transformer.MHA.__init__).
    depth_nh         = 32           # num_heads
    depth_nkvh       = 8            # gqa_dim
    depth_hd         = depth_hidden // depth_nh  # = 32
    depth_eps        = 0.00001
    depth_rope       = 1_000_000.0
    depth_maxpos     = 128_000

    # FFN intermediate inferred from the actual checkpoint (the GLU's
    # multiple_of / dim_multiplier defaults aren't all stashed in
    # config.json, but the saved tensor shape is authoritative).
    fc_gate          = sd[f"depthformer.layers.0.feed_forward.w1.weight"]
    depth_inter      = int(fc_gate.shape[0])

    # --- metadata --------------------------------------------------------
    writer.add_bool  ("codec.lm.has_adaptor",      True)
    writer.add_string("codec.lm.kind",             "residual_depth_ar")
    writer.add_string("codec.lm.host_arch",        "lfm2")
    writer.add_uint32("codec.lm.hidden_dim",       backbone_hid)
    writer.add_uint32("codec.lm.audio_embed_dim",  depth_hidden)
    writer.add_uint32("codec.lm.n_codebook",       n_codebook)
    writer.add_array ("codec.lm.codebook_sizes",   [audio_vocab] * n_codebook)
    writer.add_array ("codec.lm.delay_pattern",    [0] * n_codebook)
    writer.add_bool  ("codec.lm.parallel.tied_heads_to_embd", depth_tie)

    writer.add_uint32 ("codec.lm.residual.depth_layers",     depth_layers)
    writer.add_uint32 ("codec.lm.residual.depth_hidden",     depth_hidden)
    writer.add_uint32 ("codec.lm.residual.depth_n_heads",    depth_nh)
    writer.add_uint32 ("codec.lm.residual.depth_n_kv_heads", depth_nkvh)
    writer.add_uint32 ("codec.lm.residual.depth_head_dim",   depth_hd)
    writer.add_uint32 ("codec.lm.residual.depth_intermediate", depth_inter)
    writer.add_uint32 ("codec.lm.residual.depth_max_position", depth_maxpos)
    writer.add_float32("codec.lm.residual.depth_rms_norm_eps", depth_eps)
    writer.add_float32("codec.lm.residual.depth_rope_theta",   depth_rope)
    writer.add_bool   ("codec.lm.residual.depth_has_in_proj",      True)
    writer.add_bool   ("codec.lm.residual.depth_has_qk_norm",      True)
    writer.add_bool   ("codec.lm.residual.depth_has_output_norm",  False)
    writer.add_bool   ("codec.lm.residual.depth_use_rope",         True)
    # LFM2's apply_rotary_emb uses interleaved cos/sin pairs
    # (`rearrange "(D two) -> D two"` + complex multiply), i.e. the
    # GPT-J convention.  CSM / Qwen3-TTS use Llama's NEOX rotate_half
    # convention.  ggml mode: NORMAL=interleaved, NEOX=half-rotation.
    writer.add_bool   ("codec.lm.residual.depth_rope_interleaved", True)
    writer.add_bool   ("codec.lm.residual.depth_in_proj_per_pos",  True)
    writer.add_bool   ("codec.lm.residual.depth_in_proj_has_bias", True)
    writer.add_bool   ("codec.lm.residual.depth_has_pre_head_norm", True)
    writer.add_bool   ("codec.lm.residual.depth_emits_c0",         True)
    writer.add_string ("codec.lm.residual.weight_layout",          "shared")
    writer.add_string ("codec.lm.residual.c0_input_modality",      "none")

    # --- in_proj (3D weight + 2D bias) ----------------------------------
    dl_w = sd["depth_linear.weight"]   # HF (8192, 2048)
    dl_b = sd["depth_linear.bias"]     # HF (8192,)
    if dl_w.shape != (depth_hidden * n_codebook, backbone_hid):
        raise RuntimeError(
            f"depth_linear.weight shape {dl_w.shape} != "
            f"({depth_hidden}*{n_codebook}, {backbone_hid})"
        )
    if dl_b.shape != (depth_hidden * n_codebook,):
        raise RuntimeError(
            f"depth_linear.bias shape {dl_b.shape} != "
            f"({depth_hidden}*{n_codebook},)"
        )
    in_proj_3d = dl_w.reshape(n_codebook, depth_hidden, backbone_hid)
    in_proj_3d = np.ascontiguousarray(in_proj_3d).astype(np.float32)
    writer.add_tensor("lm.depth.in_proj.weight", in_proj_3d, st_dtype="F16")

    in_proj_bias_2d = dl_b.reshape(n_codebook, depth_hidden)
    in_proj_bias_2d = np.ascontiguousarray(in_proj_bias_2d).astype(np.float32)
    # Stored as (N, depth_hidden) HF -> ggml ne=(depth_hidden, N).  The
    # runtime slices the first T rows along the leading axis and adds
    # to (depth_hidden, T) via broadcast.
    writer.add_tensor("lm.depth.in_proj.bias", in_proj_bias_2d, st_dtype="F32")

    # --- backbone-side compose embedding -------------------------------
    # `audio_embedding.embedding` is used by `LFM2AudioModel.generate_*`
    # to embed the just-sampled audio frame back into the BACKBONE for
    # the next AR step.  Per liquid_audio's `_sample_audio_frame` +
    # `generate_sequential` (line 231 of lfm2_audio.py):
    #
    #   in_emb = audio_embedding(next_token + codebook_offsets).sum(0)
    #
    # where `next_token` is the (N,) sampled cb codes and
    # `codebook_offsets = arange(N) * (audio_vocab + 1)`.  The result is
    # a single (backbone_hidden,) vector that becomes the next backbone
    # input embed.  We keep this table in the codec_lm GGUF so
    # `codec_lm_compose_audio_embd` can produce the next-step embed
    # without the caller having to dig it out of the original checkpoint.
    if "audio_embedding.embedding.weight" not in sd:
        raise RuntimeError(
            "missing audio_embedding.embedding.weight — needed for TTS "
            "compose; please re-download the checkpoint")
    compose_w = sd["audio_embedding.embedding.weight"]
    expected_rows = (audio_vocab) * n_codebook  # 2049 * 8 = 16392
    if compose_w.shape != (expected_rows, backbone_hid):
        raise RuntimeError(
            f"audio_embedding.embedding shape {compose_w.shape} != "
            f"({expected_rows}, {backbone_hid})")
    writer.add_tensor("lm.compose.audio_embd.weight",
                      compose_w.astype(np.float32), st_dtype="F16")
    # The compose table is in BACKBONE-hidden-dim space (not depth_hidden).
    # Runtime publishes this via codec_lm_info.compose_audio_embed_dim.
    writer.add_uint32("codec.lm.compose.audio_embed_dim", backbone_hid)
    # Codebook offsets are `i * audio_vocab` (= 2049 for LFM2; the +1 for
    # EOS is already baked into audio_vocab here).  Stride is uniform so
    # the runtime can compute on the fly — we store it for clarity.
    writer.add_uint32("codec.lm.compose.codebook_stride", audio_vocab)

    # --- per-cb audio embed + heads + pre-head norm ---------------------
    for i in range(n_codebook):
        emb = sd[f"depth_embeddings.{i}.embedding.weight"]
        if emb.shape != (audio_vocab, depth_hidden):
            raise RuntimeError(
                f"depth_embeddings.{i}.embedding shape {emb.shape} != "
                f"({audio_vocab}, {depth_hidden})"
            )
        writer.add_tensor(f"lm.depth.audio_embd_{i}.weight",
                          emb.astype(np.float32), st_dtype="F16")

        tol = sd[f"depth_embeddings.{i}.to_logits.weight"]
        if tol.shape != (audio_vocab, depth_hidden):
            raise RuntimeError(
                f"depth_embeddings.{i}.to_logits shape {tol.shape} != "
                f"({audio_vocab}, {depth_hidden})"
            )
        writer.add_tensor(f"lm.depth.heads_{i}.weight",
                          tol.astype(np.float32), st_dtype="F16")

        norm = sd[f"depth_embeddings.{i}.embedding_norm.weight"]
        if norm.shape != (depth_hidden,):
            raise RuntimeError(
                f"depth_embeddings.{i}.embedding_norm shape {norm.shape} != "
                f"({depth_hidden},)"
            )
        writer.add_tensor(f"lm.depth.heads_{i}_norm.weight",
                          norm.astype(np.float32), st_dtype="F32")

    # --- transformer layers --------------------------------------------
    q_dim  = depth_nh * depth_hd            # 1024
    kv_dim = depth_nkvh * depth_hd          # 256
    qkv_w  = q_dim + 2 * kv_dim             # 1536

    expect_qkv  = (qkv_w, depth_hidden)
    expect_o    = (depth_hidden, depth_hidden)
    expect_gate = (depth_inter, depth_hidden)
    expect_up   = (depth_inter, depth_hidden)
    expect_down = (depth_hidden, depth_inter)

    for l in range(depth_layers):
        prefix_in  = f"depthformer.layers.{l}"
        prefix_out = f"lm.depth.blk_{l}"

        qkv = sd[f"{prefix_in}.operator.qkv_proj.weight"]
        if qkv.shape != expect_qkv:
            raise RuntimeError(
                f"{prefix_in}.qkv_proj shape {qkv.shape} != {expect_qkv}"
            )
        # Split along output dim (axis 0): [q, k, v] = [0:q_dim, q_dim:q_dim+kv_dim, ...]
        q = qkv[0:q_dim, :]
        k = qkv[q_dim:q_dim + kv_dim, :]
        v = qkv[q_dim + kv_dim:, :]
        for name_out, arr in [
            (f"{prefix_out}.q.weight", q),
            (f"{prefix_out}.k.weight", k),
            (f"{prefix_out}.v.weight", v),
        ]:
            writer.add_tensor(
                name_out, np.ascontiguousarray(arr).astype(np.float32),
                st_dtype="F16")

        for src_suf, dst_suf, expect in [
            ("operator.out_proj.weight",                              "o.weight",       expect_o),
            ("operator.bounded_attention.q_layernorm.weight",         "q_norm.weight",  (depth_hd,)),
            ("operator.bounded_attention.k_layernorm.weight",         "k_norm.weight",  (depth_hd,)),
            ("operator_norm.weight",                                  "attn_norm.weight", None),
            ("ffn_norm.weight",                                       "ffn_norm.weight",  None),
            ("feed_forward.w1.weight",                                "ffn_gate.weight", expect_gate),
            ("feed_forward.w3.weight",                                "ffn_up.weight",   expect_up),
            ("feed_forward.w2.weight",                                "ffn_down.weight", expect_down),
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

    if verbose:
        print(f"[lm_adaptor:lfm2_audio] residual_depth_ar (shared + per-pos in_proj): "
              f"n_codebook={n_codebook} backbone_h={backbone_hid} "
              f"depth=({depth_layers}L, {depth_hidden}h, "
              f"{depth_nh}q/{depth_nkvh}kv heads, head_dim={depth_hd}, "
              f"ffn={depth_inter}, audio_vocab={audio_vocab})")
