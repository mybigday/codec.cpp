"""MOSS-TTS family LM-adaptor dump.

Covers three top-level architecture classes published by OpenMOSS:

  * `MossTTSDForCausalLM` — MOSS-TTSD v0.5 / v0.7 (Qwen3 2B backbone, 8
    fully-summed channels in `model.embedding_list.{0..7}`; channel 0 is
    text+speech merged vocab, channels 1..7 are speech-only).  All heads
    are tied to the corresponding embedding via `_tie_or_clone_weights`.

  * `MossTTSDelayModel` — MOSS-TTSD v1.0 / MOSS-TTS (Qwen3-8B backbone in
    `language_model`; channel 0 piggybacks on the backbone's
    `embed_tokens`, channels 1..n_vq live in `emb_ext.{0..n_vq-1}`).
    All heads tied to corresponding embeddings.

  * `AsteroidTTSModel` — MOSS-TTSD v0 (early prototype, declared as plain
    Qwen3 with channel metadata bolted on).  Same channel layout as v0.5
    by config, but the safetensors layout follows the upstream Qwen3
    naming (with `embedding_list` added).  Treated as v0.5-shaped.

All three map onto the codec_lm `parallel_heads_delay` kind: N parallel
linear heads off the backbone hidden, no intra-step dependency, optional
delay (always `[0, 1, …, N-1]` for MOSS-TTS).

NOT covered here: `MossTTSNanoForCausalLM` (uses a `local_transformer`
depth-AR decoder; needs codec_lm `residual_depth_ar`, pending M3).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def dump(writer, sd: Dict[str, np.ndarray], cfg: Dict[str, Any],
         *, arch_name: str, verbose: bool = False) -> None:
    """Write codec.lm.* metadata + lm.* tensors for a MOSS-TTS-family
    LM into `writer`.  Caller (the codec converter) keeps the writer's
    lifecycle."""

    if arch_name in ("MossTTSDForCausalLM", "AsteroidTTSModel"):
        n_codebook, hidden, codebook_sizes, embd_keys = _layout_v05(cfg)
    elif arch_name == "MossTTSDelayModel":
        n_codebook, hidden, codebook_sizes, embd_keys = _layout_v10(cfg)
    else:
        raise NotImplementedError(f"MOSS-TTS LM arch not handled: {arch_name!r}")

    for k in embd_keys:
        if k not in sd:
            raise RuntimeError(
                f"missing tensor {k!r} in MOSS-TTS-family LM checkpoint "
                f"({arch_name})"
            )

    # MOSS-TTS-family processor pre-shifts channel j by j positions and
    # reverses on output, so the model itself sees a flat layout.  Caller
    # is expected to apply the same shift; metadata records the values
    # so it's there for inspection / future use.
    delay_pattern = list(range(n_codebook))

    # ---- codec.lm.* metadata ----------------------------------------
    writer.add_bool  ("codec.lm.has_adaptor",     True)
    writer.add_string("codec.lm.kind",            "parallel_heads_delay")
    writer.add_string("codec.lm.host_arch",       _host_arch_for(arch_name, cfg))
    writer.add_uint32("codec.lm.hidden_dim",      hidden)
    writer.add_uint32("codec.lm.audio_embed_dim", hidden)
    writer.add_uint32("codec.lm.n_codebook",      n_codebook)
    writer.add_array ("codec.lm.codebook_sizes",  codebook_sizes)
    writer.add_array ("codec.lm.delay_pattern",   delay_pattern)

    # All MOSS-TTS variants tie heads to the input embeddings; runtime
    # uses the audio_embd_{i} tensor as the head weight.  Saves the
    # ~600 MB-1.2 GB it'd cost to duplicate channel-0's vocab table.
    writer.add_bool  ("codec.lm.parallel.tied_heads_to_embd", True)

    # Caller-side prompt assembly fields — informational; codec_lm
    # itself doesn't consume them, but writing them lets a future
    # helper read everything from one file.
    _write_prompt_metadata(writer, cfg, arch_name)

    # ---- lm.audio_embd_{i}.weight -----------------------------------
    if not hasattr(writer, "_lm_adaptor_add_tensor"):
        # Try to match the codec converter's quantization pipeline if
        # it's exposed; otherwise dump as F16 (default for embeddings).
        pass

    for i, src_key in enumerate(embd_keys):
        t = _to_f32(sd[src_key])
        expected_shape = (codebook_sizes[i], hidden)
        if t.shape != expected_shape:
            raise RuntimeError(
                f"tensor {src_key!r} has shape {t.shape}, expected {expected_shape}"
            )
        writer.add_tensor(f"lm.audio_embd_{i}.weight", t, st_dtype="F16")

    if verbose:
        print(f"[lm_adaptor:moss_ttsd] {arch_name}: n_codebook={n_codebook} "
              f"hidden={hidden} codebook_sizes={codebook_sizes} "
              f"delay={delay_pattern}")


# ---------------------------------------------------------------------
# Per-class layout extractors
# ---------------------------------------------------------------------

def _layout_v05(cfg: Dict[str, Any]) -> Tuple[int, int, List[int], List[str]]:
    """MossTTSDForCausalLM (v0.5/v0.7) and AsteroidTTSModel (v0).

    Channels are 0..n-1, all live in `model.embedding_list.{i}.weight`.
    Channel 0 is text+speech vocab (`vocab_size`), channels 1..n-1 are
    speech-only (`speech_vocab_size`).
    """
    n_codebook = int(cfg["channels"])
    hidden     = int(cfg["hidden_size"])

    text_vocab = int(cfg["vocab_size"])
    speech_v   = int(cfg.get("speech_vocab_size", 1025))

    if "vocab_size_list" in cfg:
        codebook_sizes = [int(v) for v in cfg["vocab_size_list"]]
    else:
        # AsteroidTTSModel (v0) — config doesn't always carry the list.
        codebook_sizes = [text_vocab] + [speech_v] * (n_codebook - 1)
    if len(codebook_sizes) != n_codebook:
        raise RuntimeError(
            f"vocab_size_list length {len(codebook_sizes)} != channels={n_codebook}"
        )

    embd_keys = [f"model.embedding_list.{i}.weight" for i in range(n_codebook)]
    return n_codebook, hidden, codebook_sizes, embd_keys


def _layout_v10(cfg: Dict[str, Any]) -> Tuple[int, int, List[int], List[str]]:
    """MossTTSDelayModel (v1.0 / MOSS-TTS).

    Channel 0 is the language model's `embed_tokens`; channels 1..n_vq
    live in `emb_ext.{0..n_vq-1}.weight` with vocab `audio_vocab_size+1`.
    """
    if "language_config" not in cfg:
        raise RuntimeError("MossTTSDelayModel config missing 'language_config'")
    lcfg = cfg["language_config"]

    n_vq      = int(cfg["n_vq"])
    n_codebook = 1 + n_vq
    hidden    = int(lcfg["hidden_size"])
    text_vocab = int(lcfg["vocab_size"])
    audio_v   = int(cfg["audio_vocab_size"]) + 1   # +1 per emb_ext sizing
    codebook_sizes = [text_vocab] + [audio_v] * n_vq

    embd_keys = ["language_model.embed_tokens.weight"] + [
        f"emb_ext.{i}.weight" for i in range(n_vq)
    ]
    return n_codebook, hidden, codebook_sizes, embd_keys


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _host_arch_for(arch_name: str, cfg: Dict[str, Any]) -> str:
    """The llama.cpp arch name the host LLM should be loaded under."""
    if arch_name == "MossTTSDelayModel":
        # v1.0 / MOSS-TTS — `language_config` describes Qwen3-8B; llama.cpp
        # handles via `qwen3` (or `qwen3moe` for MoE variants).
        lcfg = cfg.get("language_config", {})
        larchs = lcfg.get("architectures") or []
        if any("Qwen3MoE" in a for a in larchs):
            return "qwen3moe"
        return "qwen3"
    # MossTTSDForCausalLM / AsteroidTTSModel — Qwen3 2B backbone.
    return "qwen3"


def _write_prompt_metadata(writer, cfg: Dict[str, Any], arch_name: str) -> None:
    """Carry text BOS / EOS / pad tokens + speech ranges so a future
    caller-side helper can assemble prompts without re-reading
    config.json.  codec_lm itself doesn't consume these."""

    if arch_name in ("MossTTSDForCausalLM", "AsteroidTTSModel"):
        if "bos_token_id" in cfg:
            writer.add_uint32("codec.lm.text_bos_id", int(cfg["bos_token_id"]))
        if "eos_token_id" in cfg:
            writer.add_uint32("codec.lm.text_eos_id", int(cfg["eos_token_id"]))
            # channel 0 is the text-vocab channel; end-of-audio = text EOS
            # sampled on cb0.  Mirror the value under the uniform eos_code_c0
            # key (eos_min_step=0).  text_eos_id is kept for back-compat.
            writer.add_int32("codec.lm.eos_code_c0", int(cfg["eos_token_id"]))
            writer.add_int32("codec.lm.eos_min_step", 0)
        if "pad_token" in cfg:
            writer.add_array("codec.lm.pad_token_per_channel",
                             [int(v) for v in cfg["pad_token"]])
        if "speech_token_range" in cfg:
            writer.add_array("codec.lm.speech_token_range",
                             [int(v) for v in cfg["speech_token_range"]])
        if "speech_pad_token" in cfg:
            writer.add_uint32("codec.lm.speech_pad_token", int(cfg["speech_pad_token"]))

    elif arch_name == "MossTTSDelayModel":
        lcfg = cfg.get("language_config", {})
        if "bos_token_id" in lcfg:
            writer.add_uint32("codec.lm.text_bos_id", int(lcfg["bos_token_id"]))
        if "eos_token_id" in lcfg:
            writer.add_uint32("codec.lm.text_eos_id", int(lcfg["eos_token_id"]))
            # channel 0 (text vocab) EOS also signals end-of-audio.
            writer.add_int32("codec.lm.eos_code_c0", int(lcfg["eos_token_id"]))
            writer.add_int32("codec.lm.eos_min_step", 0)
        if "audio_pad_code" in cfg:
            writer.add_uint32("codec.lm.audio_pad_code", int(cfg["audio_pad_code"]))
        for k in ("audio_start_token_id", "audio_end_token_id",
                  "audio_user_slot_token_id",
                  "audio_assistant_gen_slot_token_id",
                  "audio_assistant_delay_slot_token_id"):
            if k in cfg:
                writer.add_uint32(f"codec.lm.{k}", int(cfg[k]))


def _to_f32(x: np.ndarray) -> np.ndarray:
    """Standardise to F32 for the writer's quantisation pipeline.  The
    writer downcasts to F16/Q4/etc. internally based on `st_dtype`."""
    if x.dtype == np.float32:
        return x
    return x.astype(np.float32)
