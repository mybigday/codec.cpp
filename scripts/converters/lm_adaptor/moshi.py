"""Moshi (Kyutai) LM-adaptor dump — `residual_depth_ar` with
`weight_layout="flexible"`.

Covers `MoshiForConditionalGeneration` (e.g. `kmhf/hf-moshiko`,
`kmhf/hf-moshika`).  Moshi's depth decoder is structurally different
from CSM / Qwen3-TTS:

  - Every Linear in the depth decoder (q/k/v/o, fc1, fc2,
    input_projections, lm_heads) is a `MoshiFlexibleLinear` — weights
    stored as `(num_codebooks, out, in)` and the model gathers
    `weight[layer_idx]` per position, so depth position p uses its own
    weight slice.  Only the RMSNorms are shared across positions.
  - The depth decoder uses **no RoPE** (`use_rope=False`).
  - The first depth position consumes a **text token** (from the
    backbone's text head), not an audio code — `c0_input_modality="text"`.
  - The backbone hidden state is added to **every** depth position via
    `input_projections` (also flexible), not just position 0.
  - There is no final `output_norm`; `lm_heads` is applied directly to
    the last layer's output.
  - `fc1` is fused `[gate; up]` concatenated (`out = 2 * intermediate_size`);
    the converter splits it into `ffn_gate.weight` + `ffn_up.weight` so
    the runtime can apply standard SwiGLU.

This handler writes the codec_lm side only — the Helium backbone is
expected to be extracted separately for llama.cpp consumption (the
extractor + smoke test land alongside the flexible-weight runtime
support).  The runtime is gated behind a metadata check at init time
until `weight_layout="flexible"` is implemented in
`src/lm/residual_depth_ar.cpp`.

Tensor mapping (HF → codec_lm):

  depth_decoder.text_embed_tokens.weight  (V_text+1, H_d)
    → lm.depth.text_embd.weight           (V_text+1, H_d)
      Used at depth position 0 to embed the seed text token.

  depth_decoder.embed_tokens.{i}.weight   (V_audio+1, H_d)  for i=0..N-2
    → lm.depth.audio_embd_{i}.weight      (V_audio+1, H_d)
      Used at depth position i+1 to embed code c_i.  N-1 tables (the
      last codebook is predicted but never an input).

  depth_decoder.input_projections.weight  (N, H_d, H_b)
    → lm.depth.in_proj.weight             (N, H_d, H_b)  [3D flexible]
      At depth position p, projects the backbone hidden h_in to depth
      hidden via `weight[p] @ h_in`.  Added to `embed[p]`.

  depth_decoder.lm_heads.weight           (N, V_audio, H_d)
    → lm.depth.heads.weight               (N, V_audio, H_d)  [3D flexible]
      At depth position p, `weight[p]` produces the c_p logits.

  Per-layer (l = 0..L-1):
    depth_decoder.layers.{l}.self_attn.{q,k,v,o}_proj.linear.weight
                                          (N, H_d, H_d)
      → lm.depth.blk_{l}.{q,k,v,o}.weight (N, H_d, H_d)  [3D flexible]

    depth_decoder.layers.{l}.mlp.fc1.weight  (N, 2*I, H_d)  fused gate+up
      → lm.depth.blk_{l}.ffn_gate.weight  (N, I, H_d)  [first half]
      → lm.depth.blk_{l}.ffn_up.weight    (N, I, H_d)  [second half]

    depth_decoder.layers.{l}.mlp.fc2.weight  (N, H_d, I)
      → lm.depth.blk_{l}.ffn_down.weight  (N, H_d, I)   [3D flexible]

    depth_decoder.layers.{l}.input_layernorm.weight        (H_d,)
      → lm.depth.blk_{l}.attn_norm.weight (H_d,)  [SHARED across positions]
    depth_decoder.layers.{l}.post_attention_layernorm.weight (H_d,)
      → lm.depth.blk_{l}.ffn_norm.weight  (H_d,)  [SHARED]

No `lm.depth.output_norm.weight` — Moshi's depth decoder applies
`lm_heads` to the last layer's residual stream directly.  Runtime must
treat output_norm as optional.

Metadata (`codec.lm.*`):

  has_adaptor                  = true
  kind                         = "residual_depth_ar"
  host_arch                    = "llama"     (Helium = Llama-style)
  hidden_dim                   = config.hidden_size      (backbone)
  audio_embed_dim              = depth_decoder.hidden_size
                                   (sized to the depth decoder's input;
                                    backbone-side compose lives in the
                                    caller for Moshi)
  n_codebook                   = config.num_codebooks
  codebook_sizes               = [V_audio] * N (all audio outputs)
  delay_pattern                = [0] * N
  parallel.tied_heads_to_embd  = false

  residual.weight_layout       = "flexible"
  residual.c0_input_modality   = "text"
  residual.depth_layers        = depth_decoder.num_hidden_layers
  residual.depth_hidden        = depth_decoder.hidden_size
  residual.depth_n_heads       = depth_decoder.num_attention_heads
  residual.depth_n_kv_heads    = depth_decoder.num_key_value_heads
  residual.depth_head_dim      = depth_decoder.head_dim
  residual.depth_intermediate  = ffn_dim / 2            (per gate / up; fc1 stores fused 2 * I)
  residual.depth_rms_norm_eps  = depth_decoder.rms_norm_eps
  residual.depth_max_position  = config.num_codebooks   (depth prefix max = N positions)
  residual.depth_has_in_proj   = true
  residual.depth_has_qk_norm   = false
  residual.depth_has_output_norm = false                (no final norm)
  residual.depth_use_rope      = false                  (no RoPE in depth)
  residual.depth_sliding_window= depth_decoder.sliding_window  (== N = no-op)
  residual.depth_text_vocab    = depth_decoder.vocab_size       (text vocab for c0 input)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def dump(writer, sd: Dict[str, np.ndarray], cfg: Dict[str, Any],
         *, verbose: bool = False) -> None:
    archs = cfg.get("architectures") or []
    if "MoshiForConditionalGeneration" not in archs:
        raise RuntimeError(
            f"Moshi lm-adaptor dispatched on unexpected architectures {archs!r}"
        )

    dc = cfg["depth_decoder_config"]

    n_codebook    = int(cfg["num_codebooks"])
    backbone_hid  = int(cfg["hidden_size"])
    text_vocab    = int(dc["vocab_size"])      # 32000 (table has +1 padding row)
    audio_vocab   = int(dc["audio_vocab_size"])
    depth_hidden  = int(dc["hidden_size"])
    depth_layers  = int(dc["num_hidden_layers"])
    depth_nh      = int(dc["num_attention_heads"])
    depth_nkvh    = int(dc["num_key_value_heads"])
    depth_hd      = int(dc["head_dim"])
    depth_eps     = float(dc["rms_norm_eps"])
    depth_sw      = int(dc.get("sliding_window") or n_codebook)

    # MoshiGatingMLP stores fused (gate, up) concatenated along the
    # output axis in fc1 of shape (N, 2*I, H_d).  The convention is
    # `gate, up = fc1(x).chunk(2, dim=-1)` — first half is gate, second
    # half is up.  Recover the per-half intermediate size from the
    # actual checkpoint to avoid relying on `ffn_dim` (config field
    # name varies across releases).
    fc1_probe = sd[f"depth_decoder.layers.0.mlp.fc1.weight"]
    fused_out = int(fc1_probe.shape[1])  # 2 * I
    if fused_out % 2 != 0:
        raise RuntimeError(
            f"fc1 fused output {fused_out} not even; cannot split gate / up"
        )
    depth_inter = fused_out // 2

    # --- metadata --------------------------------------------------------
    writer.add_bool  ("codec.lm.has_adaptor",     True)
    writer.add_string("codec.lm.kind",            "residual_depth_ar")
    writer.add_string("codec.lm.host_arch",       "llama")
    writer.add_uint32("codec.lm.hidden_dim",      backbone_hid)
    writer.add_uint32("codec.lm.audio_embed_dim", depth_hidden)
    writer.add_uint32("codec.lm.n_codebook",      n_codebook)
    writer.add_array ("codec.lm.codebook_sizes",  [audio_vocab] * n_codebook)
    writer.add_array ("codec.lm.delay_pattern",   [0] * n_codebook)
    writer.add_bool  ("codec.lm.parallel.tied_heads_to_embd", False)

    # NO codec.lm.eos_code_c0 for Moshi: channel 0 is a TEXT token
    # (c0_input_modality="text") and Moshi is a full-duplex streaming
    # model with no audio-codebook-0 end-of-audio sentinel.  Its
    # `depth_decoder_config.eos_token_id` is None and the audio
    # `pad_token_id=audio_vocab_size` is a delay-pattern pad, not an EOS.
    # Termination is text-EOS on the backbone text head, handled by the
    # host, so no eos_code_c0 is written (the runtime defaults to -1 →
    # codec_lm_step_is_eos always reports not-EOS).

    writer.add_uint32 ("codec.lm.residual.depth_layers",     depth_layers)
    writer.add_uint32 ("codec.lm.residual.depth_hidden",     depth_hidden)
    writer.add_uint32 ("codec.lm.residual.depth_n_heads",    depth_nh)
    writer.add_uint32 ("codec.lm.residual.depth_n_kv_heads", depth_nkvh)
    writer.add_uint32 ("codec.lm.residual.depth_head_dim",   depth_hd)
    writer.add_uint32 ("codec.lm.residual.depth_intermediate", depth_inter)
    writer.add_uint32 ("codec.lm.residual.depth_max_position", n_codebook)
    writer.add_float32("codec.lm.residual.depth_rms_norm_eps", depth_eps)
    writer.add_bool   ("codec.lm.residual.depth_has_in_proj",   True)
    writer.add_bool   ("codec.lm.residual.depth_has_qk_norm",   False)
    writer.add_bool   ("codec.lm.residual.depth_has_output_norm", False)
    writer.add_bool   ("codec.lm.residual.depth_use_rope",      False)
    writer.add_uint32 ("codec.lm.residual.depth_sliding_window", depth_sw)
    writer.add_uint32 ("codec.lm.residual.depth_text_vocab",     text_vocab)
    writer.add_string ("codec.lm.residual.weight_layout",        "flexible")
    writer.add_string ("codec.lm.residual.c0_input_modality",    "text")

    # --- text embed (depth pos 0) ---------------------------------------
    text_embd = sd["depth_decoder.text_embed_tokens.weight"]
    if text_embd.shape != (text_vocab + 1, depth_hidden):
        raise RuntimeError(
            f"text_embed_tokens shape {text_embd.shape} != "
            f"({text_vocab + 1}, {depth_hidden})"
        )
    writer.add_tensor("lm.depth.text_embd.weight",
                      text_embd.astype(np.float32), st_dtype="F16")

    # --- audio embed tables (depth pos 1..N-1) --------------------------
    # N-1 tables; the last codebook is never an input.
    for i in range(n_codebook - 1):
        key = f"depth_decoder.embed_tokens.{i}.weight"
        if key not in sd:
            raise RuntimeError(f"missing tensor: {key}")
        arr = sd[key]
        if arr.shape != (audio_vocab + 1, depth_hidden):
            raise RuntimeError(
                f"{key} shape {arr.shape} != ({audio_vocab + 1}, {depth_hidden})"
            )
        writer.add_tensor(f"lm.depth.audio_embd_{i}.weight",
                          arr.astype(np.float32), st_dtype="F16")

    # --- input_projections (3D flexible) --------------------------------
    in_proj = sd["depth_decoder.input_projections.weight"]
    if in_proj.shape != (n_codebook, depth_hidden, backbone_hid):
        raise RuntimeError(
            f"input_projections shape {in_proj.shape} != "
            f"({n_codebook}, {depth_hidden}, {backbone_hid})"
        )
    writer.add_tensor("lm.depth.in_proj.weight",
                      in_proj.astype(np.float32), st_dtype="F16")

    # --- lm_heads (3D flexible) -----------------------------------------
    heads = sd["depth_decoder.lm_heads.weight"]
    if heads.shape != (n_codebook, audio_vocab, depth_hidden):
        raise RuntimeError(
            f"lm_heads shape {heads.shape} != "
            f"({n_codebook}, {audio_vocab}, {depth_hidden})"
        )
    writer.add_tensor("lm.depth.heads.weight",
                      heads.astype(np.float32), st_dtype="F16")

    # --- depth transformer layers ---------------------------------------
    expect_qkvo = (n_codebook, depth_hidden, depth_hidden)
    expect_fc1  = (n_codebook, 2 * depth_inter, depth_hidden)
    expect_fc2  = (n_codebook, depth_hidden, depth_inter)

    for l in range(depth_layers):
        prefix_in  = f"depth_decoder.layers.{l}"
        prefix_out = f"lm.depth.blk_{l}"

        # Attention: q/k/v/o (3D flexible).
        for src_suf, dst_suf in [
            ("self_attn.q_proj.linear.weight", "q.weight"),
            ("self_attn.k_proj.linear.weight", "k.weight"),
            ("self_attn.v_proj.linear.weight", "v.weight"),
            ("self_attn.o_proj.linear.weight", "o.weight"),
        ]:
            src = f"{prefix_in}.{src_suf}"
            if src not in sd:
                raise RuntimeError(f"missing tensor: {src}")
            arr = sd[src]
            if arr.shape != expect_qkvo:
                raise RuntimeError(
                    f"{src} shape {arr.shape} != {expect_qkvo}"
                )
            writer.add_tensor(f"{prefix_out}.{dst_suf}",
                              arr.astype(np.float32), st_dtype="F16")

        # Shared norms.
        for src_suf, dst_suf in [
            ("input_layernorm.weight",          "attn_norm.weight"),
            ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ]:
            src = f"{prefix_in}.{src_suf}"
            if src not in sd:
                raise RuntimeError(f"missing tensor: {src}")
            arr = sd[src]
            if arr.shape != (depth_hidden,):
                raise RuntimeError(
                    f"{src} shape {arr.shape} != ({depth_hidden},)"
                )
            writer.add_tensor(f"{prefix_out}.{dst_suf}",
                              arr.astype(np.float32), st_dtype="F32")

        # MLP: split fc1 into gate / up; fc2 is straight ffn_down.
        fc1 = sd[f"{prefix_in}.mlp.fc1.weight"]
        if fc1.shape != expect_fc1:
            raise RuntimeError(
                f"{prefix_in}.mlp.fc1 shape {fc1.shape} != {expect_fc1}"
            )
        gate, up = np.split(fc1, 2, axis=1)
        writer.add_tensor(f"{prefix_out}.ffn_gate.weight",
                          np.ascontiguousarray(gate).astype(np.float32),
                          st_dtype="F16")
        writer.add_tensor(f"{prefix_out}.ffn_up.weight",
                          np.ascontiguousarray(up).astype(np.float32),
                          st_dtype="F16")

        fc2 = sd[f"{prefix_in}.mlp.fc2.weight"]
        if fc2.shape != expect_fc2:
            raise RuntimeError(
                f"{prefix_in}.mlp.fc2 shape {fc2.shape} != {expect_fc2}"
            )
        writer.add_tensor(f"{prefix_out}.ffn_down.weight",
                          fc2.astype(np.float32), st_dtype="F16")

    if verbose:
        print(f"[lm_adaptor:moshi] residual_depth_ar (flexible): "
              f"n_codebook={n_codebook} backbone_h={backbone_hid} "
              f"depth=({depth_layers}L, {depth_hidden}h, "
              f"{depth_nh}q/{depth_nkvh}kv heads, head_dim={depth_hd}, "
              f"ffn={depth_inter} per-gate, text_vocab={text_vocab}, "
              f"audio_vocab={audio_vocab})")
