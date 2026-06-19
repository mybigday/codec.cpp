"""Convert the backbone LLM of a bundled codec_lm checkpoint to GGUF.

Single entry point for all backbones we currently support:

    sesame/csm-1b                          (CSM, Llama-3.2-1B)
    Qwen/Qwen3-TTS-12Hz-*                  (Qwen3 talker)
    fnlp/MOSS-TTSD-v0.5                    (Qwen3 language_model)
    LiquidAI/LFM2-Audio-1.5B               (Lfm2 hybrid conv+attn)
    kmhf/hf-moshiko, kmhf/hf-moshika       (Helium = Llama-style)

The codec_lm GGUF (produced by `convert-to-gguf.py`) covers the depth
decoder + compose tables.  This script covers the OTHER half — the
backbone causal LM that produces the hidden states codec_lm consumes.

Each bundled checkpoint stuffs the backbone under its own tensor-name
prefix (`backbone_model.*` for CSM, `talker.model.*` for Qwen3-TTS,
`model.language_model.*` for MOSS-TTSD, `lfm.*` for LFM2-Audio,
`decoder.*` for Moshi).  We

  1. resolve the source HF id / path,
  2. rename the backbone tensors into the standalone layout that
     llama.cpp's `convert_hf_to_gguf.py` expects (`model.layers.*` /
     `model.embed_tokens.weight` / `lm_head.weight`),
  3. synthesise a `config.json` matching the backbone arch (llama /
     qwen3 / lfm2),
  4. copy or replace tokenizer files,
  5. invoke llama.cpp's converter on the prepared dir.

If a model's bundled tokenizer has a pre-tokenizer-hash that llama.cpp
doesn't recognise (CSM uses a Mistral-flavoured Llama-3 regex variant),
we patch the converter source in-memory to fall back to a known
equivalent pre-tokenizer (`llama-bpe` for the Llama-3 case) — the
tokenizer is never used at inference for codec_lm-driven generation
because the caller feeds `llama_batch.embd` directly.

Usage:

    python scripts/convert-backbone-to-gguf.py \\
        --model-id sesame/csm-1b \\
        --output models/csm/llama_backbone.gguf \\
        [--llama-cpp /home/hans/Projects/llama.cpp] \\
        [--outtype f16]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict

import torch
from safetensors.torch import load_file, save_file


DEFAULT_LLAMA_CPP = Path.home() / "Projects" / "llama.cpp"

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
]


# ---------------------------------------------------------------------
# Source resolution + shard loading
# ---------------------------------------------------------------------

def resolve_source(src: str) -> Path:
    p = Path(src)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id=src,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt"],
    )
    return Path(local)


def load_state_dict(d: Path) -> Dict[str, torch.Tensor]:
    """Load the full state dict from a directory.  Handles both the
    `model.safetensors{.index.json}` and `transformers.safetensors.index.json`
    conventions — CSM ships the latter alongside a different (older?)
    `model.safetensors` that omits some tensors, so the transformers
    index takes precedence."""
    for idx_name in ("transformers.safetensors.index.json",
                     "model.safetensors.index.json"):
        idx = d / idx_name
        if idx.is_file():
            manifest = json.loads(idx.read_text())["weight_map"]
            shards = sorted({d / fn for fn in manifest.values()})
            sd: Dict[str, torch.Tensor] = {}
            for s in shards:
                sd.update(load_file(str(s)))
            return sd
    return load_file(str(d / "model.safetensors"))


# ---------------------------------------------------------------------
# Per-model preparation
# ---------------------------------------------------------------------
#
# Each preparer returns nothing; it writes:
#   {dst}/config.json        — HF config for the backbone arch
#   {dst}/model.safetensors  — renamed backbone state dict
#   {dst}/generation_config.json (minimal)
#   tokenizer files (either copied from src or downloaded fresh)
#
# Models may also return a list of `llama.cpp converter source
# patches`: `(needle, replacement)` string pairs applied at runtime to
# bypass missing-feature checks.  Empty list means use llama.cpp's
# converter unmodified.


def _save_state_dict(sd: Dict[str, torch.Tensor], dst: Path) -> None:
    out = {k: v.clone().contiguous() for k, v in sd.items()}
    save_file(out, str(dst / "model.safetensors"))


def _copy_tokenizer(src: Path, dst: Path) -> None:
    for fn in TOKENIZER_FILES:
        if (src / fn).exists():
            shutil.copy(src / fn, dst / fn)


def _write_generation_config(dst: Path, bos: int | None, eos: int | None) -> None:
    gen_cfg = {"_from_model_config": True, "transformers_version": "4.40.0"}
    if bos is not None: gen_cfg["bos_token_id"] = bos
    if eos is not None: gen_cfg["eos_token_id"] = eos
    (dst / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))


# ---- CSM (Llama-3.2-1B) --------------------------------------------------

def prep_csm(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    sd = load_state_dict(src)
    text_vocab = int(cfg["text_vocab_size"])
    hidden     = int(cfg["hidden_size"])

    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("backbone_model.layers."):
            out_sd["model." + k[len("backbone_model."):]] = v
        elif k == "backbone_model.norm.weight":
            out_sd["model.norm.weight"] = v

    embd = sd["embed_text_tokens.weight"]
    assert embd.shape == (text_vocab, hidden), \
        f"embed_text_tokens shape {tuple(embd.shape)} != ({text_vocab}, {hidden})"
    out_sd["model.embed_tokens.weight"] = embd.clone().contiguous()

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size":              text_vocab,
        "hidden_size":             hidden,
        "intermediate_size":       int(cfg["intermediate_size"]),
        "num_hidden_layers":       int(cfg["num_hidden_layers"]),
        "num_attention_heads":     int(cfg["num_attention_heads"]),
        "num_key_value_heads":     int(cfg["num_key_value_heads"]),
        "head_dim":                int(cfg["head_dim"]),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
        "rms_norm_eps":            float(cfg["rms_norm_eps"]),
        "rope_theta":              float(cfg["rope_theta"]),
        "hidden_act":              cfg.get("hidden_act", "silu"),
        "attention_bias":          bool(cfg.get("attention_bias", False)),
        "attention_dropout":       float(cfg.get("attention_dropout", 0.0)),
        "mlp_bias":                bool(cfg.get("mlp_bias", False)),
        "initializer_range":       float(cfg.get("initializer_range", 0.02)),
        "tie_word_embeddings":     True,
        "torch_dtype":             "float32",
        "use_cache":               True,
        "bos_token_id":            cfg.get("bos_token_id", 128000),
        "eos_token_id":            cfg.get("eos_token_id", 128001),
        "pad_token_id":            cfg.get("pad_token_id", 128004),
    }
    if cfg.get("rope_scaling"):
        llama_cfg["rope_scaling"] = cfg["rope_scaling"]
    (dst / "config.json").write_text(json.dumps(llama_cfg, indent=2))

    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, llama_cfg["bos_token_id"], llama_cfg["eos_token_id"])
    _copy_tokenizer(src, dst)

    # CSM ships a Mistral-flavoured Llama-3 tokenizer whose
    # pre-tokenizer hash isn't in llama.cpp's get_vocab_base_pre() map.
    # The regex family is the same as llama-bpe; the tokenizer is unused
    # at codec_lm-driven inference (we feed embeddings via batch.embd),
    # so patch the converter's fallback branch to map any unknown hash
    # to llama-bpe.
    return [(
        'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
        'res = "llama-bpe"  # csm: Llama-3 regex family, unused at codec_lm inference',
    )]


# ---- Qwen3-TTS (Qwen3 talker) -------------------------------------------

def prep_qwen3_tts(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    tk = cfg["talker_config"]
    hidden  = int(tk["hidden_size"])
    vocab   = int(tk["vocab_size"])

    # Qwen3-TTS ships the full Qwen3 text tokenizer (151936) but the
    # talker's codec_embedding only has `vocab` rows.  Pad to the
    # tokenizer size so llama.cpp's set_vocab assertion passes; the
    # padded rows are never read (we feed embeddings directly).
    PADDED_VOCAB = 151936

    sd = load_state_dict(src)
    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("talker.model.layers."):
            out_sd["model." + k[len("talker.model."):]] = v
        elif k == "talker.model.norm.weight":
            out_sd["model.norm.weight"] = v

    codec_embd = sd["talker.model.codec_embedding.weight"]
    assert codec_embd.shape == (vocab, hidden)
    padded = torch.zeros((PADDED_VOCAB, hidden), dtype=codec_embd.dtype)
    padded[:vocab] = codec_embd
    out_sd["model.embed_tokens.weight"] = padded.contiguous()

    qwen3_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size":              PADDED_VOCAB,
        "hidden_size":             hidden,
        "intermediate_size":       int(tk["intermediate_size"]),
        "num_hidden_layers":       int(tk["num_hidden_layers"]),
        "num_attention_heads":     int(tk["num_attention_heads"]),
        "num_key_value_heads":     int(tk["num_key_value_heads"]),
        "head_dim":                int(tk["head_dim"]),
        "max_position_embeddings": int(tk["max_position_embeddings"]),
        "rms_norm_eps":            float(tk["rms_norm_eps"]),
        "rope_theta":              float(tk["rope_theta"]),
        "hidden_act":              tk.get("hidden_act", "silu"),
        "attention_bias":          bool(tk.get("attention_bias", False)),
        "attention_dropout":       float(tk.get("attention_dropout", 0.0)),
        "initializer_range":       float(tk.get("initializer_range", 0.02)),
        "tie_word_embeddings":     True,
        "torch_dtype":             "float32",
        "use_cache":               True,
        "sliding_window":          None,
        "use_sliding_window":      False,
    }
    (dst / "config.json").write_text(json.dumps(qwen3_cfg, indent=2))
    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, None, None)
    _copy_tokenizer(src, dst)
    return []


# ---- MOSS-TTSD (Qwen3 language_model) -----------------------------------

def prep_moss_ttsd(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    archs = cfg.get("architectures") or []
    if not any(a in archs for a in ("MossTTSDForCausalLM", "AsteroidTTSModel")):
        raise SystemExit(f"unsupported MOSS source architectures={archs}")

    sd = load_state_dict(src)
    out_sd: Dict[str, torch.Tensor] = {}
    n_renamed = 0
    for k, v in sd.items():
        if k.startswith("model.language_model."):
            out_sd["model." + k[len("model.language_model."):]] = v
            n_renamed += 1

    embd0 = sd.get("model.embedding_list.0.weight")
    if embd0 is None:
        raise SystemExit("missing model.embedding_list.0.weight")
    out_sd["model.embed_tokens.weight"] = embd0.clone().contiguous()

    qwen3_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size":              int(cfg["vocab_size"]),
        "hidden_size":             int(cfg["hidden_size"]),
        "intermediate_size":       int(cfg["intermediate_size"]),
        "num_hidden_layers":       int(cfg["num_hidden_layers"]),
        "num_attention_heads":     int(cfg["num_attention_heads"]),
        "num_key_value_heads":     int(cfg["num_key_value_heads"]),
        "head_dim":                int(cfg["head_dim"]),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
        "rms_norm_eps":            float(cfg["rms_norm_eps"]),
        "rope_theta":              float(cfg["rope_theta"]),
        "hidden_act":              cfg.get("hidden_act", "silu"),
        "attention_bias":          bool(cfg.get("attention_bias", False)),
        "attention_dropout":       float(cfg.get("attention_dropout", 0.0)),
        "initializer_range":       float(cfg.get("initializer_range", 0.02)),
        "tie_word_embeddings":     True,
        "torch_dtype":             cfg.get("torch_dtype", "bfloat16"),
        "use_cache":               True,
        "bos_token_id":            cfg.get("bos_token_id"),
        "eos_token_id":            cfg.get("eos_token_id"),
    }
    if cfg.get("rope_scaling"):
        qwen3_cfg["rope_scaling"] = cfg["rope_scaling"]
    if cfg.get("sliding_window") is not None:
        qwen3_cfg["sliding_window"] = cfg["sliding_window"]
    (dst / "config.json").write_text(json.dumps(qwen3_cfg, indent=2))

    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, qwen3_cfg["bos_token_id"], qwen3_cfg["eos_token_id"])
    _copy_tokenizer(src, dst)
    return []


# ---- Chatterbox T3 (Llama 520M with llama3 RoPE scaling) ----------------

# Mirror of `.model-src/chatterbox/src/chatterbox/models/t3/llama_configs.py`
# `LLAMA_520M_CONFIG_DICT`.  The vocab_size=8 placeholder is intentional:
# T3 bypasses the backbone's own embed_tokens (the real text/speech
# embeddings live in `text_emb` / `speech_emb` outside the tfmr), so the
# backbone GGUF only needs to load the transformer layers + a vestigial
# embed table.  At inference we feed embeddings via batch.embd, never
# token IDs, so the tiny vocab is unused.
CHATTERBOX_T3_LLAMA_CFG = {
    "vocab_size":              8,
    "hidden_size":             1024,
    "intermediate_size":       4096,
    "num_hidden_layers":       30,
    "num_attention_heads":     16,
    "num_key_value_heads":     16,
    "head_dim":                64,
    "max_position_embeddings": 131072,
    "rms_norm_eps":            1e-05,
    "rope_theta":              500000.0,
    "hidden_act":              "silu",
    "attention_bias":          False,
    "attention_dropout":       0.0,
    "mlp_bias":                False,
    "initializer_range":       0.02,
    "tie_word_embeddings":     False,
    "torch_dtype":             "float32",
    "use_cache":               True,
    "rope_scaling":            dict(
        factor=8.0,
        high_freq_factor=4.0,
        low_freq_factor=1.0,
        original_max_position_embeddings=8192,
        rope_type="llama3",
    ),
}


def _chatterbox_find_t3_safetensors(src: Path) -> Path:
    """Find the T3 LM checkpoint inside a Chatterbox-format directory.
    Supports English (`t3_cfg.safetensors`) + multilingual variants
    (`t3_23lang.safetensors`, `t3_mtl23ls_v2.safetensors`,
    `t3_mtl23ls_v3.safetensors`)."""
    for fn in ("t3_cfg.safetensors", "t3_mtl23ls_v3.safetensors",
               "t3_mtl23ls_v2.safetensors", "t3_23lang.safetensors"):
        p = src / fn
        if p.is_file():
            return p
    raise SystemExit(
        f"no Chatterbox T3 checkpoint in {src}; expected one of "
        f"t3_cfg / t3_mtl23ls_v2 / t3_mtl23ls_v3 / t3_23lang .safetensors")


def prep_chatterbox_t3(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    """Unwrap Chatterbox T3's Llama-520M backbone.

    Chatterbox's `t3_cfg.safetensors` (and the multilingual variants)
    bundles BOTH the Llama tfmr and the T3 LM-adaptor side
    (text/speech embeds + heads + learned PEs + cond_enc) under
    `tfmr.*` and various other prefixes.  This prep extracts only the
    `tfmr.*` tensors → standalone Llama layout; the LM-adaptor side
    is handled separately by `scripts/converters/lm_adaptor/chatterbox.py`.
    """
    t3_path = _chatterbox_find_t3_safetensors(src)
    sd = load_file(str(t3_path))

    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("tfmr.layers."):
            out_sd["model." + k[len("tfmr."):]] = v
        elif k == "tfmr.norm.weight":
            out_sd["model.norm.weight"] = v
        elif k == "tfmr.embed_tokens.weight":
            # vocab=8 placeholder; kept for round-trip but unused at inference.
            out_sd["model.embed_tokens.weight"] = v

    if "model.embed_tokens.weight" not in out_sd:
        raise SystemExit(f"missing tfmr.embed_tokens.weight in {t3_path}")

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        **CHATTERBOX_T3_LLAMA_CFG,
    }
    (dst / "config.json").write_text(json.dumps(llama_cfg, indent=2))
    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, None, None)
    # No tokenizer to copy — Chatterbox's tokenizer.json is the speech-LM
    # text tokenizer (704-vocab EN or 2454-vocab multilingual), not the
    # backbone's vocab (which is just an 8-row placeholder).  We
    # synthesise a minimal HF tokenizer dir so llama.cpp's converter
    # doesn't reject the source.
    _write_chatterbox_placeholder_tokenizer(dst)

    # Force llama.cpp's converter into the `no_vocab` path for
    # Chatterbox.  Our 8-token placeholder isn't a real tokenizer
    # (and the backbone is embd-driven at inference); writing a
    # `tokenizer.ggml.model = "none"` marker tells llama.cpp's loader
    # to skip vocab validation.  We patch LlamaModel.set_vocab to
    # early-return _set_vocab_none() for any model with vocab_size<=8.
    return [(
        'if self.origin_hf_arch == "GlmasrModel":',
        'if (self.hparams.get("vocab_size", 0) or 0) <= 8:\n'
        '            return self._set_vocab_none()\n'
        '        if self.origin_hf_arch == "GlmasrModel":',
    )]


def _write_chatterbox_placeholder_tokenizer(dst: Path) -> None:
    """T3's backbone has a vocab=8 placeholder embed table (real text +
    speech embeddings live in the LM-adaptor side).  llama.cpp's
    converter expects a HF tokenizer.json — synthesise a minimal one
    with the same 8-token vocab.  `<unk>` must be in the vocab for
    `tokenizers.Tokenizer` to load it; we put it at id 0 and fill the
    rest with `<t1>`..`<t7>` placeholders."""
    # WordLevel tokenizer with no merges — simplest type that
    # `tokenizers.Tokenizer.from_file()` accepts.  We never tokenize
    # through this vocab (backbone is embd-driven only); it just exists
    # so llama.cpp's converter has *some* vocab to write.
    vocab = {"<unk>": 0}
    vocab.update({f"<t{i}>": i for i in range(1, 8)})
    placeholder = {
        "version": "1.0",
        "model": {
            "type":         "WordLevel",
            "unk_token":    "<unk>",
            "vocab":        vocab,
        },
        "added_tokens": [
            {"id": tid, "content": tok, "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False,
             "special": True}
            for tok, tid in vocab.items()
        ],
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "normalizer": None,
    }
    (dst / "tokenizer.json").write_text(json.dumps(placeholder, indent=2))
    (dst / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": CHATTERBOX_T3_LLAMA_CFG["max_position_embeddings"],
    }, indent=2))


# ---- MOSS-TTS-Realtime (Qwen3 language_model) --------------------------

def prep_moss_tts_realtime(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    """Unwrap the Qwen3 language_model from MOSS-TTS-Realtime.

    Layout in HF:
      - language_model.{embed_tokens,layers.*,norm}     ← we want these
      - embed_tokens.{0..16}                            ← compose embeds, go to codec_lm
      - local_transformer.*                             ← depth decoder, goes to codec_lm

    The language_model is a stock Qwen3 with `tie_word_embeddings=true`
    inferred from the absence of a separate `lm_head.weight` tensor in
    the checkpoint."""
    archs = cfg.get("architectures") or []
    if "MossTTSRealtime" not in archs:
        raise SystemExit(f"unsupported source architectures={archs}")
    if "language_config" not in cfg:
        raise SystemExit("MOSS-TTS-Realtime config missing 'language_config'")

    lcfg = cfg["language_config"]

    sd = load_state_dict(src)
    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("language_model."):
            out_sd["model." + k[len("language_model."):]] = v

    if "model.embed_tokens.weight" not in out_sd:
        raise SystemExit("missing language_model.embed_tokens.weight")

    qwen3_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size":              int(lcfg["vocab_size"]),
        "hidden_size":             int(lcfg["hidden_size"]),
        "intermediate_size":       int(lcfg["intermediate_size"]),
        "num_hidden_layers":       int(lcfg["num_hidden_layers"]),
        "num_attention_heads":     int(lcfg["num_attention_heads"]),
        "num_key_value_heads":     int(lcfg["num_key_value_heads"]),
        "head_dim":                int(lcfg["head_dim"]),
        "max_position_embeddings": int(lcfg["max_position_embeddings"]),
        "rms_norm_eps":            float(lcfg["rms_norm_eps"]),
        "rope_theta":              float(lcfg["rope_theta"]),
        "hidden_act":              lcfg.get("hidden_act", "silu"),
        "attention_bias":          bool(lcfg.get("attention_bias", False)),
        "attention_dropout":       float(lcfg.get("attention_dropout", 0.0)),
        "initializer_range":       float(lcfg.get("initializer_range", 0.02)),
        # No separate lm_head tensor in MOSS-TTS-Realtime safetensors → tied.
        "tie_word_embeddings":     True,
        "torch_dtype":             lcfg.get("dtype", "bfloat16"),
        "use_cache":               True,
        "bos_token_id":            lcfg.get("bos_token_id"),
        "eos_token_id":            lcfg.get("eos_token_id"),
    }
    if lcfg.get("rope_scaling"):
        qwen3_cfg["rope_scaling"] = lcfg["rope_scaling"]
    if lcfg.get("sliding_window") is not None:
        qwen3_cfg["sliding_window"] = lcfg["sliding_window"]
    (dst / "config.json").write_text(json.dumps(qwen3_cfg, indent=2))

    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, qwen3_cfg["bos_token_id"], qwen3_cfg["eos_token_id"])
    _copy_tokenizer(src, dst)
    return []


# ---- LFM2-Audio (Lfm2-1.2B hybrid conv+attn) ---------------------------

def prep_lfm2_audio(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    lfm_cfg = cfg["lfm"]

    sd = load_state_dict(src)
    out_sd: Dict[str, torch.Tensor] = {}
    # Rename `lfm.*` -> `model.*` so the layout matches the standalone
    # LiquidAI/LFM2-1.2B HF dir.  `embedding_norm` is a separate norm
    # in Lfm2 (not collapsed into model.norm); both `embed_tokens` and
    # `embedding_norm` get the `model.` prefix verbatim.
    for k, v in sd.items():
        if k.startswith("lfm."):
            out_sd["model." + k[len("lfm."):]] = v

    out_cfg = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        # Preserve all the Lfm2-specific config fields verbatim; llama.cpp's
        # Lfm2Model converter looks at the same set of keys as the
        # standalone LiquidAI/LFM2-1.2B HF checkpoint.
        **{k: v for k, v in lfm_cfg.items()
           if k not in ("_name_or_path", "architectures")},
        "tie_word_embeddings":     True,
        "torch_dtype":             lfm_cfg.get("torch_dtype", "bfloat16"),
    }
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2))
    _save_state_dict(out_sd, dst)
    _write_generation_config(
        dst, out_cfg.get("bos_token_id"), out_cfg.get("eos_token_id"))
    _copy_tokenizer(src, dst)
    return []


# ---- Moshi (Helium = Llama-style) --------------------------------------

def prep_moshi(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    # Moshi's Helium backbone lives under `decoder.model.*` and uses the
    # MoshiLinear wrapper, so attention projections are stored as
    # `self_attn.{q,k,v,o}_proj.linear.weight` (the extra ".linear" is
    # the inner nn.Linear under MoshiLinear).  The MLP uses
    # MoshiGatingMLP, so `mlp.fc1.weight` is fused [gate; up] along the
    # output axis and `mlp.fc2.weight` is down_proj.  Llama arch wants
    # `self_attn.{q,k,v,o}_proj.weight` + `mlp.{gate,up,down}_proj.weight`,
    # so rename + split.
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["ffn_dim"]) // 2   # fused fc1 is 2*intermediate

    sd = load_state_dict(src)
    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith("decoder.model.layers."):
            continue
        suffix = k[len("decoder.model.layers."):]  # `{l}.<rest>`
        new_k = "model.layers." + suffix
        # strip MoshiLinear wrapper's ".linear" infix
        if ".linear.weight" in new_k:
            new_k = new_k.replace(".linear.weight", ".weight")
        if "mlp.fc1.weight" in new_k:
            # split fused gate+up along axis 0
            gate, up = v.chunk(2, dim=0)
            base = new_k.replace("mlp.fc1.weight", "")
            out_sd[base + "mlp.gate_proj.weight"] = gate.contiguous()
            out_sd[base + "mlp.up_proj.weight"]   = up.contiguous()
            continue
        if "mlp.fc2.weight" in new_k:
            out_sd[new_k.replace("mlp.fc2.weight", "mlp.down_proj.weight")] = v
            continue
        out_sd[new_k] = v

    # Final norm.
    norm_key = "decoder.model.norm.weight"
    if norm_key not in sd:
        raise SystemExit(f"missing {norm_key}")
    out_sd["model.norm.weight"] = sd[norm_key].clone().contiguous()

    # Embed: `decoder.model.embed_tokens.weight` has +1 vocab (32001)
    # for an EOS pad; the LM head is 32000 wide.  Truncate the embed to
    # match the canonical Helium vocab so tied / untied both work.
    embd = sd.get("decoder.model.embed_tokens.weight")
    head = sd.get("decoder.lm_head.weight")
    if embd is None or head is None:
        raise SystemExit("missing decoder.model.embed_tokens.weight or decoder.lm_head.weight")
    vocab = int(cfg["vocab_size"])
    if embd.shape[0] > vocab:
        embd = embd[:vocab].contiguous()
    if head.shape[0] != vocab:
        raise SystemExit(f"lm_head rows {head.shape[0]} != vocab {vocab}")
    out_sd["model.embed_tokens.weight"] = embd
    out_sd["lm_head.weight"] = head

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size":              vocab,
        "hidden_size":             hidden,
        "intermediate_size":       intermediate,
        "num_hidden_layers":       int(cfg["num_hidden_layers"]),
        "num_attention_heads":     int(cfg["num_attention_heads"]),
        "num_key_value_heads":     int(cfg["num_key_value_heads"]),
        "head_dim":                int(cfg["head_dim"]),
        "max_position_embeddings": int(cfg["max_position_embeddings"]),
        "rms_norm_eps":            float(cfg["rms_norm_eps"]),
        "rope_theta":              float(cfg["rope_theta"]),
        "hidden_act":              cfg.get("hidden_act", "silu"),
        "attention_bias":          False,
        "tie_word_embeddings":     False,  # separate lm_head shipped
        "torch_dtype":             cfg.get("torch_dtype", "bfloat16"),
        "use_cache":               True,
        "sliding_window":          int(cfg.get("sliding_window", 0)) or None,
    }
    (dst / "config.json").write_text(json.dumps(llama_cfg, indent=2))
    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, None, None)
    _copy_tokenizer(src, dst)
    # The Helium tokenizer (used by hf-moshi{ko,ka}) has its own
    # pre-tokenizer regex that isn't in llama.cpp's
    # get_vocab_base_pre() table.  codec_lm doesn't use the text
    # tokenizer at inference time (caller drives via embeddings), so
    # map any unknown hash to `llama-bpe`, same as CSM.
    return [(
        'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
        'res = "llama-bpe"  # moshi: Helium tokenizer, unused at codec_lm inference',
    )]


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------

PREPARERS: Dict[str, Callable[[Path, Path, dict], list[tuple[str, str]]]] = {
    "CsmForConditionalGeneration":    prep_csm,
    "Qwen3TTSForConditionalGeneration": prep_qwen3_tts,
    "MossTTSDForCausalLM":            prep_moss_ttsd,
    "AsteroidTTSModel":               prep_moss_ttsd,
    "MossTTSRealtime":                prep_moss_tts_realtime,
    "Lfm2AudioForConditionalGeneration": prep_lfm2_audio,
    "MoshiForConditionalGeneration":  prep_moshi,
    "ChatterboxT3":                   prep_chatterbox_t3,
}


def detect_arch(cfg: dict) -> str:
    archs = cfg.get("architectures") or []
    for a in archs:
        if a in PREPARERS:
            return a
    raise SystemExit(
        f"unsupported architectures={archs}; supported: "
        f"{sorted(PREPARERS.keys())}")


def chatterbox_layout_detected(src: Path) -> bool:
    """Detect a Chatterbox-format checkpoint dir.  Chatterbox doesn't
    ship a `config.json` with `architectures[]`, so we autodetect by
    looking for the bundled T3 checkpoint filename(s)."""
    if not src.is_dir():
        return False
    for fn in ("t3_cfg.safetensors", "t3_mtl23ls_v3.safetensors",
               "t3_mtl23ls_v2.safetensors", "t3_23lang.safetensors"):
        if (src / fn).is_file():
            return True
    return False


# ---------------------------------------------------------------------
# llama.cpp converter invocation
# ---------------------------------------------------------------------

def run_llama_cpp_converter(
        hf_dir: Path,
        output: Path,
        llama_cpp: Path,
        outtype: str,
        patches: list[tuple[str, str]],
) -> None:
    """Spawn llama.cpp's `convert_hf_to_gguf.py` in-process via exec,
    applying source patches before execution.  Patches handle the
    handful of cases where bundled-checkpoint quirks (unrecognised
    tokenizer hashes) need a one-line tweak to the converter."""
    conv = llama_cpp / "convert_hf_to_gguf.py"
    if not conv.is_file():
        raise SystemExit(f"missing {conv}; pass --llama-cpp /path/to/llama.cpp")

    saved_argv = sys.argv
    sys.argv = [
        "convert_hf_to_gguf.py", str(hf_dir),
        "--outfile", str(output),
        "--outtype", outtype,
    ]
    try:
        src = conv.read_text()
        for needle, repl in patches:
            if needle not in src:
                raise SystemExit(
                    f"patch failed: needle not found in {conv}: {needle!r}")
            src = src.replace(needle, repl)
        exec(compile(src, str(conv), "exec"),
             {"__name__": "__main__", "__file__": str(conv)})
    finally:
        sys.argv = saved_argv


def load_config(src: Path) -> dict:
    return json.loads((src / "config.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", required=True,
                    help="HF repo id (e.g. sesame/csm-1b) or local checkpoint dir")
    ap.add_argument("--output", required=True, type=Path,
                    help="output backbone GGUF path")
    ap.add_argument("--llama-cpp", default=str(DEFAULT_LLAMA_CPP),
                    help=f"path to llama.cpp checkout (default: {DEFAULT_LLAMA_CPP})")
    ap.add_argument("--outtype", default="f16",
                    help="dtype to pass through to convert_hf_to_gguf.py")
    args = ap.parse_args()

    src = resolve_source(args.model_id)
    print(f"[bb] source: {src}", flush=True)

    # Chatterbox doesn't ship a config.json (its T3 weights live in
    # `t3_cfg.safetensors`); autodetect the layout and synthesise a
    # config dict for the dispatch.
    if chatterbox_layout_detected(src):
        cfg = {"architectures": ["ChatterboxT3"]}
    else:
        cfg = load_config(src)
    arch = detect_arch(cfg)
    print(f"[bb] arch: {arch}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="codec-bb-") as tmp:
        hf_dir = Path(tmp)
        print(f"[bb] preparing HF dir at {hf_dir} …", flush=True)
        patches = PREPARERS[arch](src, hf_dir, cfg)
        if patches:
            print(f"[bb] applying {len(patches)} llama.cpp converter patch(es)",
                  flush=True)
        print(f"[bb] running llama.cpp converter …", flush=True)
        run_llama_cpp_converter(hf_dir, args.output, Path(args.llama_cpp),
                                args.outtype, patches)

    print(f"[bb] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
