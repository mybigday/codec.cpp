"""CSM (Sesame) LM-adaptor dump — `residual_depth_ar` kind.

Tensor mapping (HF `sesame/csm-1b` → codec.lm.* schema):

  backbone_model.embed_tokens.embed_audio_tokens.weight
      shape (V*N, H) = (65632, 2048)    fused per-cb input embedding
        → split into N tensors `lm.audio_embd_{i}.weight`, each (V, H)
          (depth_decoder.model.embed_tokens.weight is tied to this; we
           write a single copy under `lm.audio_embd_{i}` and the runtime
           uses it as both the backbone-side input embed and the
           depth-decoder-side codebook embedding.)

  lm_head.weight       (V, H_b)         = (2051, 2048)   c0 head
        → `lm.c0_head.weight`

  depth_decoder.codebooks_head.weight   (N-1, H_d, V) = (31, 1024, 2051)
                                        Parameter, computation
                                        `decoder_h_last @ codebooks_head[i]`
                                        → for cb i+1 logits.
        → split into N-1 tensors `lm.depth.heads_{i}.weight`, each
          (H_d, V) per slice (matches the (in, out) layout that ggml
          mul_mat expects so logits = mul_mat(head, decoder_h)).

  depth_decoder.model.inputs_embeds_projector.weight   (H_d, H_b) = (1024, 2048)
        → `lm.depth.in_proj.weight`     2048 → 1024 projection from
          backbone hidden into depth_hidden.

  depth_decoder.model.layers.{l}.input_layernorm.weight       (H_d,)
        → `lm.depth.blk_{l}.attn_norm.weight`
  depth_decoder.model.layers.{l}.self_attn.{q,k,v,o}_proj.weight
        → `lm.depth.blk_{l}.{q,k,v,o}.weight`
  depth_decoder.model.layers.{l}.post_attention_layernorm.weight
        → `lm.depth.blk_{l}.ffn_norm.weight`
  depth_decoder.model.layers.{l}.mlp.{gate,up,down}_proj.weight
        → `lm.depth.blk_{l}.{ffn_gate,ffn_up,ffn_down}.weight`
  depth_decoder.model.norm.weight                              (H_d,)
        → `lm.depth.output_norm.weight`

Metadata (codec.lm.*):

  codec.lm.has_adaptor                                = true
  codec.lm.kind                                       = "residual_depth_ar"
  codec.lm.host_arch                                  = "llama"
  codec.lm.hidden_dim                                 = backbone H_b
  codec.lm.audio_embed_dim                            = backbone H_b
  codec.lm.n_codebook                                 = config.audio_num_codebooks
  codec.lm.codebook_sizes                             = [V] * N
  codec.lm.delay_pattern                              = [0] * N
  codec.lm.parallel.tied_heads_to_embd                = false (we ship c0_head + per-cb heads explicitly)
  codec.lm.residual.depth_layers                      = depth_decoder_config.num_hidden_layers
  codec.lm.residual.depth_hidden                      = depth_decoder_config.hidden_size
  codec.lm.residual.depth_n_heads                     = depth_decoder_config.num_attention_heads
  codec.lm.residual.depth_n_kv_heads                  = depth_decoder_config.num_key_value_heads
  codec.lm.residual.depth_head_dim                    = depth_decoder_config.head_dim
  codec.lm.residual.depth_intermediate                = depth_decoder_config.intermediate_size
  codec.lm.residual.depth_rope_theta                  = depth_decoder_config.rope_theta
  codec.lm.residual.depth_max_position                = depth_decoder_config.max_position_embeddings
  codec.lm.residual.depth_rms_norm_eps                = depth_decoder_config.rms_norm_eps
  codec.lm.residual.depth_has_in_proj                 = true
  codec.lm.residual.depth_has_qk_norm                 = false  (Llama-3.2-style; not Qwen3)
  codec.lm.residual.weight_layout                     = "shared"
  codec.lm.residual.c0_input_modality                 = "audio"
  # llama3-style RoPE scaling factors:
  codec.lm.residual.depth_rope_scaling_factor         = scaling.factor
  codec.lm.residual.depth_rope_scaling_low_freq       = scaling.low_freq_factor
  codec.lm.residual.depth_rope_scaling_high_freq      = scaling.high_freq_factor
  codec.lm.residual.depth_rope_scaling_orig_max_pos   = scaling.original_max_position_embeddings

Plus pre-computed RoPE scaling tensor:
  lm.depth.rope_freq_factors                          shape (head_dim/2,)
        Llama3-style scaled inverse frequencies; the runtime feeds this
        into ggml_rope_ext via the `freq_factors` parameter.  Computing
        it at convert time avoids replicating the piecewise smoothing
        logic in ggml.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def dump(writer, sd: Dict[str, np.ndarray], cfg: Dict[str, Any],
         *, verbose: bool = False) -> None:
    """Write codec.lm.* metadata + lm.* tensors for a CSM checkpoint into
    the supplied GGUFWriter.  `sd` is the full CSM state dict (numpy);
    `cfg` is its config.json."""

    archs = cfg.get("architectures") or []
    if "CsmForConditionalGeneration" not in archs:
        raise RuntimeError(
            f"CSM lm-adaptor dispatched on unexpected architectures {archs!r}"
        )

    n_codebook   = int(cfg["audio_num_codebooks"])
    audio_vocab  = int(cfg["audio_vocab_size"])
    backbone_h   = int(cfg["hidden_size"])
    dcfg         = cfg["depth_decoder_config"]
    depth_layers = int(dcfg["num_hidden_layers"])
    depth_h      = int(dcfg["hidden_size"])
    depth_nh     = int(dcfg["num_attention_heads"])
    depth_nkvh   = int(dcfg["num_key_value_heads"])
    depth_hd     = int(dcfg["head_dim"])
    depth_inter  = int(dcfg["intermediate_size"])
    depth_eps    = float(dcfg["rms_norm_eps"])
    depth_rope   = float(dcfg["rope_theta"])
    depth_maxpos = int(dcfg["max_position_embeddings"])

    # --- metadata -------------------------------------------------------
    writer.add_bool  ("codec.lm.has_adaptor",     True)
    writer.add_string("codec.lm.kind",            "residual_depth_ar")
    writer.add_string("codec.lm.host_arch",       "llama")
    writer.add_uint32("codec.lm.hidden_dim",      backbone_h)
    writer.add_uint32("codec.lm.audio_embed_dim", backbone_h)
    writer.add_uint32("codec.lm.n_codebook",      n_codebook)
    writer.add_array ("codec.lm.codebook_sizes",  [audio_vocab] * n_codebook)
    writer.add_array ("codec.lm.delay_pattern",   [0] * n_codebook)
    writer.add_bool  ("codec.lm.parallel.tied_heads_to_embd", False)

    # End-of-audio: CSM stops when codebook-0 samples the audio-EOS code.
    # The reference (CsmForConditionalGeneration.generate) ignores frame 0
    # and stops at frame >= 1 when cb0 == codebook_eos_token_id (=0).  So
    # eos_min_step=1.  Read the code from the checkpoint's own config
    # rather than hardcoding.
    eos_c0 = int(cfg.get("codebook_eos_token_id", 0))
    writer.add_int32("codec.lm.eos_code_c0", eos_c0)
    writer.add_int32("codec.lm.eos_min_step", 1)

    writer.add_uint32 ("codec.lm.residual.depth_layers",     depth_layers)
    writer.add_uint32 ("codec.lm.residual.depth_hidden",     depth_h)
    writer.add_uint32 ("codec.lm.residual.depth_n_heads",    depth_nh)
    writer.add_uint32 ("codec.lm.residual.depth_n_kv_heads", depth_nkvh)
    writer.add_uint32 ("codec.lm.residual.depth_head_dim",   depth_hd)
    writer.add_uint32 ("codec.lm.residual.depth_intermediate", depth_inter)
    writer.add_float32("codec.lm.residual.depth_rope_theta", depth_rope)
    writer.add_uint32 ("codec.lm.residual.depth_max_position", depth_maxpos)
    writer.add_float32("codec.lm.residual.depth_rms_norm_eps", depth_eps)
    writer.add_bool   ("codec.lm.residual.depth_has_in_proj", True)
    writer.add_bool   ("codec.lm.residual.depth_has_qk_norm", False)
    writer.add_string ("codec.lm.residual.weight_layout",      "shared")
    writer.add_string ("codec.lm.residual.c0_input_modality",  "audio")

    # rope_scaling: Llama3 piecewise — write the scaling parameters and a
    # precomputed `freq_factors` tensor so the runtime doesn't need to
    # replicate the smoothing logic.
    rope_scaling = dcfg.get("rope_scaling") or {}
    if rope_scaling.get("rope_type") == "llama3":
        s_factor = float(rope_scaling["factor"])
        s_low    = float(rope_scaling["low_freq_factor"])
        s_high   = float(rope_scaling["high_freq_factor"])
        s_origm  = int(rope_scaling["original_max_position_embeddings"])
        writer.add_float32("codec.lm.residual.depth_rope_scaling_factor",   s_factor)
        writer.add_float32("codec.lm.residual.depth_rope_scaling_low_freq", s_low)
        writer.add_float32("codec.lm.residual.depth_rope_scaling_high_freq", s_high)
        writer.add_uint32 ("codec.lm.residual.depth_rope_scaling_orig_max_pos", s_origm)

        freqs = _llama3_rope_freq_factors(
            depth_hd, depth_rope, s_factor, s_low, s_high, s_origm).astype(np.float32)
        writer.add_tensor("lm.depth.rope_freq_factors", freqs, st_dtype="F32")

    # --- audio embed tables (split fused) -------------------------------
    embd = sd["backbone_model.embed_tokens.embed_audio_tokens.weight"]
    if embd.shape != (n_codebook * audio_vocab, backbone_h):
        raise RuntimeError(
            f"audio embed table shape mismatch {embd.shape} != "
            f"({n_codebook}*{audio_vocab}, {backbone_h})"
        )
    for i in range(n_codebook):
        slc = embd[i * audio_vocab : (i + 1) * audio_vocab].astype(np.float32)
        writer.add_tensor(f"lm.audio_embd_{i}.weight", slc, st_dtype="F16")

    # --- c0 head --------------------------------------------------------
    c0 = sd["lm_head.weight"]
    if c0.shape != (audio_vocab, backbone_h):
        raise RuntimeError(
            f"lm_head shape mismatch {c0.shape} != ({audio_vocab}, {backbone_h})"
        )
    writer.add_tensor("lm.c0_head.weight", c0.astype(np.float32), st_dtype="F16")

    # --- depth heads (split codebooks_head Parameter) -------------------
    # `audio_head` is stored as a torch.Parameter with shape (N-1, in=H_d,
    # out=V) — that's the upstream `(in, out)` layout used by the explicit
    # `decoder_h @ audio_head[i]` matmul in CSM's reference forward.  Every
    # other Linear-derived weight in this checkpoint follows PyTorch's
    # `(out, in)` Linear convention, which lands in ggml as
    # `ne = [in, out]` and works directly with `mul_mat(W, x)`.  Transpose
    # this one slice to match so the runtime can use the standard
    # `mul_mat(head, h_last)` pattern.
    heads = sd["depth_decoder.codebooks_head.weight"]
    if heads.shape != (n_codebook - 1, depth_h, audio_vocab):
        raise RuntimeError(
            f"codebooks_head shape mismatch {heads.shape} != "
            f"({n_codebook - 1}, {depth_h}, {audio_vocab})"
        )
    for i in range(n_codebook - 1):
        head_slice = heads[i].T   # (H_d, V) -> (V, H_d) i.e. (out, in)
        head_slice = np.ascontiguousarray(head_slice).astype(np.float32)
        writer.add_tensor(f"lm.depth.heads_{i}.weight", head_slice, st_dtype="F16")

    # --- in_proj --------------------------------------------------------
    in_proj = sd["depth_decoder.model.inputs_embeds_projector.weight"]
    if in_proj.shape != (depth_h, backbone_h):
        raise RuntimeError(
            f"in_proj shape mismatch {in_proj.shape} != ({depth_h}, {backbone_h})"
        )
    writer.add_tensor("lm.depth.in_proj.weight",
                      in_proj.astype(np.float32), st_dtype="F16")

    # --- depth decoder transformer layers -------------------------------
    expect_q = (depth_h, depth_h)
    expect_kv = (depth_nkvh * depth_hd, depth_h)
    expect_o = (depth_h, depth_h)
    expect_gate = (depth_inter, depth_h)
    expect_up   = (depth_inter, depth_h)
    expect_down = (depth_h, depth_inter)

    for l in range(depth_layers):
        prefix_in  = f"depth_decoder.model.layers.{l}"
        prefix_out = f"lm.depth.blk_{l}"

        for src_suf, dst_suf, expect in [
            ("input_layernorm.weight",         "attn_norm.weight",  None),
            ("self_attn.q_proj.weight",        "q.weight",          expect_q),
            ("self_attn.k_proj.weight",        "k.weight",          expect_kv),
            ("self_attn.v_proj.weight",        "v.weight",          expect_kv),
            ("self_attn.o_proj.weight",        "o.weight",          expect_o),
            ("post_attention_layernorm.weight","ffn_norm.weight",   None),
            ("mlp.gate_proj.weight",           "ffn_gate.weight",   expect_gate),
            ("mlp.up_proj.weight",             "ffn_up.weight",     expect_up),
            ("mlp.down_proj.weight",           "ffn_down.weight",   expect_down),
        ]:
            src = f"{prefix_in}.{src_suf}"
            if src not in sd:
                raise RuntimeError(f"missing tensor in CSM checkpoint: {src}")
            arr = sd[src]
            if expect is not None and arr.shape != expect:
                raise RuntimeError(
                    f"{src} shape {arr.shape} != expected {expect}"
                )
            dt = "F32" if dst_suf.endswith("_norm.weight") else "F16"
            writer.add_tensor(f"{prefix_out}.{dst_suf}", arr.astype(np.float32),
                              st_dtype=dt)

    # final norm
    out_norm = sd["depth_decoder.model.norm.weight"]
    writer.add_tensor("lm.depth.output_norm.weight",
                      out_norm.astype(np.float32), st_dtype="F32")

    if verbose:
        print(f"[lm_adaptor:csm] residual_depth_ar: n_codebook={n_codebook} "
              f"backbone_h={backbone_h} depth=({depth_layers}L, {depth_h}h, "
              f"{depth_nh}q/{depth_nkvh}kv heads, head_dim={depth_hd}, "
              f"ffn={depth_inter}, rope_theta={depth_rope}, "
              f"rope_scaling={rope_scaling.get('rope_type', 'none')})")


# ---------------------------------------------------------------------
# Llama3 RoPE scaling — matches the formula in the upstream reference
# (`transformers.modeling_rope_utils._compute_llama3_parameters`):
#
#   inv_freq[i] = base ** (-2*i / D)                  for i in 0..D/2
#   wavelen[i]  = 2π / inv_freq[i]
#   smooth[i]   = (orig_max / wavelen - low) / (high - low)
#   inv_freq_llama[i] = inv_freq[i]                                   if wavelen < high_wavelen
#                     = inv_freq[i] / factor                           if wavelen > low_wavelen
#                     = (1 - smooth) * inv_freq / factor + smooth * inv_freq  otherwise
#
# ggml's `rope_ext` takes `freq_factors[i]` such that effective inverse
# frequency = inv_freq[i] / freq_factors[i].  We match by writing
# `freq_factors[i] = inv_freq[i] / inv_freq_llama[i]`.
# ---------------------------------------------------------------------

def _llama3_rope_freq_factors(
    head_dim: int, base: float, factor: float,
    low_freq_factor: float, high_freq_factor: float,
    original_max_pos: int) -> np.ndarray:
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    inv_freq = base ** (-2.0 * i / head_dim)        # (half,)
    wavelen  = 2.0 * math.pi / inv_freq

    low_wavelen  = original_max_pos / low_freq_factor
    high_wavelen = original_max_pos / high_freq_factor

    smooth = (original_max_pos / wavelen - low_freq_factor) / \
             (high_freq_factor - low_freq_factor)

    inv_freq_llama = np.where(
        wavelen < high_wavelen, inv_freq,
        np.where(wavelen > low_wavelen, inv_freq / factor,
                 (1.0 - smooth) * inv_freq / factor + smooth * inv_freq))

    # ggml's freq_factors divides inv_freq, so we want
    # freq_factors[i] = inv_freq[i] / inv_freq_llama[i].
    return (inv_freq / inv_freq_llama).astype(np.float64)
