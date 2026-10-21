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

import numpy as np
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


CHATTERBOX_T3_CHECKPOINTS = (
    "t3_cfg.safetensors", "t3_mtl23ls_v3.safetensors",
    "t3_mtl23ls_v2.safetensors", "t3_23lang.safetensors",
    "t3_turbo_v1.safetensors",
)

# Turbo swaps the Llama-520M tfmr for GPT-2 medium (learned wpe, LayerNorm,
# fused c_attn).  Mirror of `llama_configs.py::GPT2_MEDIUM_CONFIG` with the
# real Turbo text vocab: its `wte` rows line up 1:1 with the shipped GPT-2
# tokenizer (50257 BPE + 19 paralinguistic tags such as `[laugh]`), so the
# backbone GGUF carries the actual text tokenizer and the runtime tokenizes
# with llama_tokenize.  Inference still feeds embeddings, never token ids.
CHATTERBOX_T3_TURBO_GPT2_CFG = {
    "architectures":      ["GPT2LMHeadModel"],
    "model_type":         "gpt2",
    "activation_function": "gelu_new",
    "vocab_size":         50276,
    "n_ctx":              8196,
    "n_positions":        8196,
    "n_embd":             1024,
    "n_head":             16,
    "n_layer":            24,
    "n_inner":            None,
    "layer_norm_epsilon": 1e-05,
    "bos_token_id":       50256,
    "eos_token_id":       50256,
    "tie_word_embeddings": True,
    "torch_dtype":        "float32",
}


def prep_chatterbox_t3_turbo(src: Path, t3_path: Path, dst: Path) -> list[tuple[str, str]]:
    """Unwrap Chatterbox Turbo's GPT-2 medium tfmr (`tfmr.h.*`, `tfmr.ln_f`,
    `tfmr.wpe`, `tfmr.wte`) into a GPT2LMHeadModel dir + its tokenizer.
    The T3 adaptor side (text/speech embeds, speech head with bias,
    spkr_enc) goes through `scripts/converters/lm_adaptor/chatterbox.py`."""
    sd = load_file(str(t3_path))
    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("tfmr."):
            out_sd[k[len("tfmr."):]] = v
    for need in ("wte.weight", "wpe.weight", "ln_f.weight", "h.0.attn.c_attn.weight"):
        if need not in out_sd:
            raise SystemExit(f"missing tfmr.{need} in {t3_path}")
    if out_sd["wte.weight"].shape[0] != CHATTERBOX_T3_TURBO_GPT2_CFG["vocab_size"]:
        raise SystemExit(f"unexpected Turbo wte shape {tuple(out_sd['wte.weight'].shape)}")
    (dst / "config.json").write_text(json.dumps(CHATTERBOX_T3_TURBO_GPT2_CFG, indent=2))
    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, 50256, 50256)
    _copy_tokenizer(src, dst)
    return []


def _chatterbox_find_t3_safetensors(src: Path) -> Path:
    """Find the T3 LM checkpoint inside a Chatterbox-format directory.
    Supports English (`t3_cfg.safetensors`) + multilingual variants
    (`t3_23lang.safetensors`, `t3_mtl23ls_v2.safetensors`,
    `t3_mtl23ls_v3.safetensors`)."""
    for fn in CHATTERBOX_T3_CHECKPOINTS:
        p = src / fn
        if p.is_file():
            return p
    raise SystemExit(
        f"no Chatterbox T3 checkpoint in {src}; expected one of "
        f"{' / '.join(CHATTERBOX_T3_CHECKPOINTS)}")


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
    if t3_path.name == "t3_turbo_v1.safetensors":
        return prep_chatterbox_t3_turbo(src, t3_path, dst)
    sd = load_file(str(t3_path))

    out_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("tfmr.layers."):
            out_sd["model." + k[len("tfmr."):]] = v
        elif k == "tfmr.norm.weight":
            out_sd["model.norm.weight"] = v
        elif k == "text_emb.weight":
            # The token embedding of the backbone vocab: T3's text tokenizer
            # (the host feeds codec-composed rows; the table sizes the vocab).
            out_sd["model.embed_tokens.weight"] = v

    if "model.embed_tokens.weight" not in out_sd:
        raise SystemExit(f"missing text_emb.weight in {t3_path}")

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        **CHATTERBOX_T3_LLAMA_CFG,
        "vocab_size": int(out_sd["model.embed_tokens.weight"].shape[0]),
    }
    (dst / "config.json").write_text(json.dumps(llama_cfg, indent=2))
    _save_state_dict(out_sd, dst)
    _write_generation_config(dst, None, None)
    # The backbone vocab is T3's text tokenizer (704-token EN tokenizer.json /
    # 2454-token multilingual grapheme BPE): a raw-UTF-8 BPE behind HF's
    # Whitespace pre-tokenizer, which patches/chatterbox-tokenizer-llamacpp.patch
    # implements as `tokenizer.ggml.pre = chatterbox`.  The text normalization
    # (punc_norm, lowercase / NFKD, ' ' -> [SPACE]) is codec_audio_lm's format
    # step; [lang] tags are the caller's.
    tok = src / "tokenizer.json"
    if t3_path.name != "t3_cfg.safetensors" or not tok.is_file():
        tok = src / "grapheme_mtl_merged_expanded_v1.json"
    if not tok.is_file():
        raise SystemExit(f"no T3 tokenizer (tokenizer.json / grapheme_mtl_merged_expanded_v1.json) in {src}")
    shutil.copy(tok, dst / "tokenizer.json")
    (dst / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": CHATTERBOX_T3_LLAMA_CFG["max_position_embeddings"],
        "unk_token": "[UNK]",
    }, indent=2))
    return [(
        'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
        'res = "chatterbox"  # patches/chatterbox-tokenizer-llamacpp.patch',
    )]


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


# ---- KittenTTS 2 (Qwen3-1.7B speech LM) ----------------------------------

def prep_kitten_tts2(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    """KittenTTS 2 keeps a plain Qwen3ForCausalLM under `lm/`, plus a
    `spk_proj.*` speaker head that is not part of the backbone (it lives in
    the codec GGUF, see scripts/converters/kitten_tts2.py) and is dropped.
    Audio tokens are ordinary vocab rows; `<|speech_end|>` is the eos."""
    lm = src / cfg.get("lm_dir", "lm")
    sd = {k: v for k, v in load_state_dict(lm).items() if not k.startswith("spk_proj.")}
    _save_state_dict(sd, dst)
    shutil.copy(lm / "config.json", dst / "config.json")
    _copy_tokenizer(lm, dst)
    tm = cfg["token_map"]
    _write_generation_config(dst, bos=151643, eos=int(tm["speech_end_id"]))
    return []


# ---- Soprano (Qwen3 LM next to a Vocos decoder.pth) ----------------------

def prep_soprano(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    """Soprano is a stock Qwen3ForCausalLM, but its tokenizer (lowercase +
    whitespace collapse, per-digit split, raw-UTF-8 BPE) is no llama.cpp
    pre-tokenizer: tag it `soprano`, which patches/soprano-llamacpp.patch
    implements.  The decoder.pth half is the codec GGUF (--model-type soprano)."""
    shutil.copy(src / "config.json", dst / "config.json")
    _save_state_dict(load_state_dict(src), dst)
    _copy_tokenizer(src, dst)
    _write_generation_config(dst, bos=cfg.get("bos_token_id"), eos=cfg.get("eos_token_id"))
    return [(
        'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
        'res = "soprano"  # patches/soprano-llamacpp.patch',
    )]


# ---- CosyVoice2 / CosyVoice3 (Qwen2-0.5B inside llm.pt) -------------------

def _cosyvoice_special_tokens(version: int) -> list[str]:
    """CosyVoice{2,3}Tokenizer.add_special_tokens lists (cosyvoice/tokenizer/tokenizer.py).
    The order fixes the added-token ids (<|endofprompt|> = 151646)."""
    import importlib.util
    root = Path(__file__).resolve().parents[1] / ".model-src" / "cosyvoice" / "cosyvoice" / "tokenizer" / "tokenizer.py"
    if root.is_file():
        import ast
        tree = ast.parse(root.read_text())
        cls = {2: "CosyVoice2Tokenizer", 3: "CosyVoice3Tokenizer"}[version]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Dict):
                        keys = [k.value for k in sub.keys if isinstance(k, ast.Constant)]
                        if "additional_special_tokens" in keys:
                            lst = sub.values[keys.index("additional_special_tokens")]
                            return [e.value for e in lst.elts]
    raise SystemExit("CosyVoice special-token list not found: clone FunAudioLLM/CosyVoice into .model-src/cosyvoice")


def prep_cosyvoice(src: Path, dst: Path, cfg: dict) -> list[tuple[str, str]]:
    """CosyVoice2/3 `llm.pt` wraps a Qwen2ForCausalLM under `llm.model.`; the
    speech embedding / head (`speech_embedding`, `llm_decoder`, `llm_embedding`)
    belong to the codec GGUF's LM adaptor. Text is tokenized by the backbone
    vocab, so the CosyVoice special tokens are added before conversion."""
    from transformers import AutoTokenizer
    version = int(cfg["cosyvoice_version"])
    sd = torch.load(src / "llm.pt", map_location="cpu", weights_only=True)
    # lm_head is never used (the speech head is llm_decoder) and the config ties it.
    bb = {k[len("llm.model."):]: v for k, v in sd.items()
          if k.startswith("llm.model.") and k != "llm.model.lm_head.weight"}
    _save_state_dict(bb, dst)
    blank = src / "CosyVoice-BlankEN"
    shutil.copy(blank / "config.json", dst / "config.json")
    tok = AutoTokenizer.from_pretrained(str(blank))
    tok.add_special_tokens({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>",
                            "additional_special_tokens": _cosyvoice_special_tokens(version)})
    assert tok.convert_tokens_to_ids("<|endofprompt|>") == 151646
    tok.save_pretrained(str(dst))
    _write_generation_config(dst, bos=151643, eos=151643)
    # The added CosyVoice specials change llama.cpp's tokenizer fingerprint;
    # the pre-tokenizer is still Qwen2's (text IS tokenized by the backbone).
    return [(
        'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
        'res = "qwen2"  # cosyvoice: Qwen2 tokenizer + CosyVoice special tokens',
    ), (
        # The byte-level decoder mangles non-ASCII added tokens ("[iāng]" ->
        # "[i\x01ng]"), which then collide; keep the added tokens verbatim.
        'token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))',
        'pass  # cosyvoice: added tokens are stored verbatim',
    )]


# ---- CosyVoice v1 (CosyVoice-300M / BreezyVoice): llama.cpp `cosyvoice` arch ----

def _whisper_vocab() -> tuple[list[str], list[int], list[str]]:
    """Whisper multilingual tiktoken (num_languages=100, 51866 ids) as a GPT-2
    byte-level BPE vocab: tokens, token types and merges (rebuilt from ranks,
    as convert_hf_to_gguf.py does for tiktoken vocabs)."""
    import whisper.tokenizer as wt
    enc = wt.get_encoding("multilingual", num_languages=100)
    ranks = enc._mergeable_ranks

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = dict(zip(bs, map(chr, cs)))

    def to_str(b: bytes) -> str:
        return "".join(byte_encoder[x] for x in b)

    def bpe(token: bytes, max_rank: int) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            best = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                r = ranks.get(pair[0] + pair[1])
                if r is not None and (best is None or r < best[1]):
                    best = (i, r)
            if best is None or best[1] >= max_rank:
                break
            i = best[0]
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        return parts

    tokens = [""] * enc.n_vocab
    types = [1] * enc.n_vocab               # NORMAL
    merges = []
    for tok, rank in ranks.items():
        tokens[rank] = to_str(tok)
        if len(tok) > 1:
            m = bpe(tok, rank)
            assert len(m) == 2
            merges.append(" ".join(map(to_str, m)))
    for tok, rank in enc._special_tokens.items():
        tokens[rank] = tok
        types[rank] = 3                     # CONTROL
    return tokens, types, merges


def write_cosyvoice1_backbone(src: Path, output: Path, outtype: str) -> None:
    """CosyVoice v1 `llm.pt` `llm.*` (WeNet TransformerEncoder, rel-pos attention)
    → GGUF arch `cosyvoice` (patches/cosyvoice-llamacpp.patch).  The text
    encoder, speech embedding and speech head belong to the codec GGUF."""
    import gguf
    sd = {k[len("llm."):]: v.float().numpy() for k, v in
          torch.load(src / "llm.pt", map_location="cpu", weights_only=True).items() if k.startswith("llm.")}
    readme = (src / "README.md").read_text(errors="ignore") if (src / "README.md").is_file() else ""
    breezy = "BreezyVoice" in readme
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("encoders."))
    n_embd = sd["embed.out.0.weight"].shape[0]
    n_head = sd["encoders.0.self_attn.pos_bias_u"].shape[0]
    head_dim = n_embd // n_head
    n_ff = sd["encoders.0.feed_forward.w_1.weight"].shape[0]
    ftype = {"f32": gguf.GGMLQuantizationType.F32, "f16": gguf.GGMLQuantizationType.F16}[outtype]

    w = gguf.GGUFWriter(str(output), "cosyvoice")
    w.add_name("BreezyVoice LM" if breezy else "CosyVoice-300M LM")
    w.add_context_length(8192)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    # BreezyVoice's fork builds encoder layers with LayerNorm eps 1e-5; upstream WeNet uses 1e-12.
    w.add_layer_norm_eps(1e-5 if breezy else 1e-12)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if outtype == "f16" else gguf.LlamaFileType.ALL_F32)
    tokens, types, merges = _whisper_vocab()
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(50257)
    w.add_eos_token_id(50257)
    w.add_add_bos_token(False)

    def add(name: str, arr, keep_f32: bool = False):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if ftype == gguf.GGMLQuantizationType.F16 and arr.ndim >= 2 and not keep_f32:
            w.add_tensor(name, arr.astype(np.float16), raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            w.add_tensor(name, arr)

    add("inp_proj.weight", sd["embed.out.0.weight"])
    add("inp_proj.bias", sd["embed.out.0.bias"])
    add("inp_norm.weight", sd["embed.out.1.weight"])
    add("inp_norm.bias", sd["embed.out.1.bias"])
    add("output_norm.weight", sd["after_norm.weight"])
    add("output_norm.bias", sd["after_norm.bias"])
    for i in range(n_layer):
        p, d = f"encoders.{i}.", f"blk.{i}."
        a, f = p + "self_attn.", p + "feed_forward."
        add(d + "attn_norm.weight", sd[p + "norm1.weight"])
        add(d + "attn_norm.bias", sd[p + "norm1.bias"])
        for dd, ss in (("attn_q", "linear_q"), ("attn_k", "linear_k"), ("attn_v", "linear_v"), ("attn_output", "linear_out")):
            add(d + dd + ".weight", sd[a + ss + ".weight"])
            add(d + dd + ".bias", sd[a + ss + ".bias"])
        # linear_pos rows of each head, laid out [head, n_embd(in), head_dim] (ggml [d, in, head]).
        wp = sd[a + "linear_pos.weight"].reshape(n_head, head_dim, n_embd).transpose(0, 2, 1)
        add(d + "attn_pos.weight", wp, keep_f32=True)
        add(d + "attn_pos_bias_u.weight", sd[a + "pos_bias_u"], keep_f32=True)   # added to F32 q
        add(d + "attn_pos_bias_v.weight", sd[a + "pos_bias_v"], keep_f32=True)
        add(d + "ffn_norm.weight", sd[p + "norm2.weight"])
        add(d + "ffn_norm.bias", sd[p + "norm2.bias"])
        add(d + "ffn_up.weight", sd[f + "w_1.weight"])
        add(d + "ffn_up.bias", sd[f + "w_1.bias"])
        add(d + "ffn_down.weight", sd[f + "w_2.weight"])
        add(d + "ffn_down.bias", sd[f + "w_2.bias"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def pocket_tts_checkpoint(src: Path) -> Path | None:
    """kyutai/pocket-tts `model.safetensors` (local dir, or the HF repo's
    languages/english/), or None when `src` is not a Pocket-TTS checkpoint."""
    for st in (src / "model.safetensors", src / "languages" / "english" / "model.safetensors"):
        if st.is_file():
            from safetensors import safe_open
            with safe_open(str(st), "np") as f:
                if "flow_lm.transformer.layers.0.self_attn.in_proj.weight" in f.keys():
                    return st
    return None


def write_pocket_tts_backbone(st_path: Path, output: Path, outtype: str) -> None:
    """Pocket-TTS FlowLM transformer (`flow_lm.transformer.*` + `out_norm`) →
    stock GGUF arch `starcoder2`: pre-LN affine LayerNorm, separate q/k/v,
    non-gated GELU FFN, final LayerNorm.  The checkpoint has no linear biases;
    starcoder2 requires them (q/k/v only in older llama.cpp), so they are zeros.
    pocket_tts rotates interleaved pairs (2i, 2i+1); starcoder2 is NEOX
    (i, i + d/2), so each head's q/k rows are reordered to [even, odd] — the
    same permutation on q and k leaves q·k unchanged.  The text LUT
    (`conditioner.embed`, n_bins + 1 rows) is `token_embd`: text positions are
    fed as `b.token`; latent / voice positions as `b.embd` (codec_lm side)."""
    import gguf
    from safetensors import safe_open
    sd = {}
    with safe_open(str(st_path), "pt") as f:
        for k in f.keys():
            if k.startswith(("flow_lm.transformer.", "flow_lm.out_norm.")) or k == "flow_lm.conditioner.embed.weight":
                sd[k[len("flow_lm."):]] = f.get_tensor(k).float().numpy()
    # Fixed for every pocket-tts release (pocket_tts/config/*.yaml flow_lm.transformer).
    n_head, max_period = 16, 10000.0
    n_embd = sd["out_norm.weight"].shape[0]
    n_layer = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.layers."))
    n_ff = sd["transformer.layers.0.linear1.weight"].shape[0]
    n_vocab = sd["conditioner.embed.weight"].shape[0]
    head_dim = n_embd // n_head
    # The checkpoint is BF16: f32 is capped at bf16 (never wider than the source).
    qtype = {"f16": gguf.GGMLQuantizationType.F16, "bf16": gguf.GGMLQuantizationType.BF16,
             "f32": gguf.GGMLQuantizationType.BF16}[outtype]

    w = gguf.GGUFWriter(str(output), "starcoder2")
    w.add_name("Pocket-TTS FlowLM")
    w.add_context_length(8192)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head)
    w.add_rope_freq_base(max_period)
    w.add_layer_norm_eps(1e-5)
    w.add_vocab_size(n_vocab)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16 if qtype == gguf.GGMLQuantizationType.BF16
                    else gguf.LlamaFileType.MOSTLY_F16)
    w.add_tokenizer_model("none")   # tokenized by codec_lm's SentencePiece model

    def add(name: str, arr) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim >= 2:
            w.add_tensor(name, gguf.quants.quantize(arr, qtype), raw_dtype=qtype)
        else:
            w.add_tensor(name, arr)

    def neox_rows(m: np.ndarray) -> np.ndarray:
        m = m.reshape(n_head, head_dim // 2, 2, n_embd)
        return m.transpose(0, 2, 1, 3).reshape(n_embd, n_embd)

    add("token_embd.weight", sd["conditioner.embed.weight"])
    add("output_norm.weight", sd["out_norm.weight"])
    add("output_norm.bias", sd["out_norm.bias"])
    zero = np.zeros(n_embd, dtype=np.float32)
    for i in range(n_layer):
        p, d = f"transformer.layers.{i}.", f"blk.{i}."
        q, k, v = np.split(sd[p + "self_attn.in_proj.weight"], 3, axis=0)
        add(d + "attn_norm.weight", sd[p + "norm1.weight"])
        add(d + "attn_norm.bias", sd[p + "norm1.bias"])
        add(d + "attn_q.weight", neox_rows(q))
        add(d + "attn_k.weight", neox_rows(k))
        add(d + "attn_v.weight", v)
        for b in ("attn_q", "attn_k", "attn_v"):   # optional in current llama.cpp, required by older builds
            add(d + b + ".bias", zero)
        add(d + "attn_output.weight", sd[p + "self_attn.out_proj.weight"])
        add(d + "attn_output.bias", zero)
        add(d + "ffn_norm.weight", sd[p + "norm2.weight"])
        add(d + "ffn_norm.bias", sd[p + "norm2.bias"])
        add(d + "ffn_up.weight", sd[p + "linear1.weight"])
        add(d + "ffn_up.bias", np.zeros(n_ff, dtype=np.float32))
        add(d + "ffn_down.weight", sd[p + "linear2.weight"])
        add(d + "ffn_down.bias", zero)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


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
    "KittenTTS2":                     prep_kitten_tts2,
    "CosyVoice":                      prep_cosyvoice,
    "Soprano":                        prep_soprano,
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
    return any((src / fn).is_file() for fn in CHATTERBOX_T3_CHECKPOINTS)


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
    saved_path = list(sys.path)
    saved_modules = {k: v for k, v in sys.modules.items() if k == "conversion" or k.startswith("conversion.")}
    # Newer llama.cpp splits the converter into a `conversion/` package next
    # to convert_hf_to_gguf.py; exec() doesn't add the script dir itself.
    sys.path[:0] = [str(llama_cpp.resolve()), str((llama_cpp / "gguf-py").resolve())]
    # A patch whose needle lives in the package (not the script) is applied to
    # a copy of it put first on sys.path.
    src = conv.read_text()
    pkg = llama_cpp / "conversion"
    pkg_patches = [(n, r) for n, r in patches if n not in src]
    pkg_tmp = None
    if pkg_patches and pkg.is_dir():
        pkg_tmp = Path(tempfile.mkdtemp(prefix="codec-conv-"))
        shutil.copytree(pkg, pkg_tmp / "conversion")
        for needle, repl in pkg_patches:
            hits = [p for p in (pkg_tmp / "conversion").rglob("*.py") if needle in p.read_text()]
            if not hits:
                raise SystemExit(f"patch failed: needle not found in {conv} or {pkg}: {needle!r}")
            for p in hits:
                p.write_text(p.read_text().replace(needle, repl))
        patches = [(n, r) for n, r in patches if n in src]
        for k in saved_modules:
            del sys.modules[k]
        sys.path.insert(0, str(pkg_tmp))
    sys.argv = [
        "convert_hf_to_gguf.py", str(hf_dir),
        "--outfile", str(output),
        "--outtype", outtype,
    ]
    try:
        for needle, repl in patches:
            if needle not in src:
                raise SystemExit(
                    f"patch failed: needle not found in {conv}: {needle!r}")
            src = src.replace(needle, repl)
        exec(compile(src, str(conv), "exec"),
             {"__name__": "__main__", "__file__": str(conv)})
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "conversion" or k.startswith("conversion.")]:
            del sys.modules[k]
        sys.modules.update(saved_modules)
        if pkg_tmp is not None:
            shutil.rmtree(pkg_tmp, ignore_errors=True)


_ARCH_SAMPLING = {
    "CsmForConditionalGeneration": "csm",
    "Qwen3TTSForConditionalGeneration": "qwen3_tts",
    "MossTTSDForCausalLM": "moss_ttsd",
    "AsteroidTTSModel": "moss_ttsd",
    "MossTTSRealtime": "moss_tts_realtime",
    "Lfm2AudioForConditionalGeneration": "lfm2_audio",
    "KittenTTS2": "kitten_tts2",
    "CosyVoice": "cosyvoice",
    "Soprano": "soprano",
}


def sampling_model(arch: str, src: Path, model_id: str) -> str | None:
    """The scripts/utils/sampling.py entry of a backbone (None: no TTS sampling)."""
    if arch == "ChatterboxT3":
        return "chatterbox_turbo" if _chatterbox_find_t3_safetensors(src).name == "t3_turbo_v1.safetensors" \
            else "chatterbox"
    if arch in ("CosyVoice1", "CosyVoice") and "breezy" in model_id.lower():
        return "breezyvoice"
    if arch == "CosyVoice1":
        return "cosyvoice"
    if arch == "PocketTTS":
        return "pocket_tts"
    return _ARCH_SAMPLING.get(arch)


def write_sampling(output: Path, model: str | None) -> None:
    """Rewrite the backbone GGUF with the model's general.sampling.* keys."""
    if model is None:
        return
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parent / "common" / "third-party" / "llama.cpp" / "gguf-py")]
    from utils.sampling import add_sampling_to_gguf
    tmp = output.with_name(output.name + ".sampling.tmp")
    add_sampling_to_gguf(output, tmp, model)
    tmp.replace(output)
    print(f"[bb] general.sampling.*: {model}", flush=True)


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
    if (src / "cosyvoice.yaml").is_file():   # CosyVoice v1: custom arch, written directly
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_cosyvoice1_backbone(src, args.output, args.outtype)
        write_sampling(args.output, sampling_model("CosyVoice1", src, args.model_id))
        print(f"[bb] wrote {args.output}", flush=True)
        return 0
    pocket = pocket_tts_checkpoint(src)
    if pocket is not None:                    # Pocket-TTS: stock starcoder2, written directly
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_pocket_tts_backbone(pocket, args.output, args.outtype)
        write_sampling(args.output, sampling_model("PocketTTS", src, args.model_id))
        print(f"[bb] wrote {args.output}", flush=True)
        return 0
    if chatterbox_layout_detected(src):
        cfg = {"architectures": ["ChatterboxT3"]}
    elif (src / "cosyvoice2.yaml").is_file() or (src / "cosyvoice3.yaml").is_file():
        cfg = {"architectures": ["CosyVoice"],
               "cosyvoice_version": 3 if (src / "cosyvoice3.yaml").is_file() else 2}
    else:
        cfg = load_config(src)
        if cfg.get("type") == "KITTEN2":   # kittenml repo layout: LM under lm/
            cfg["architectures"] = ["KittenTTS2"]
        elif (src / "decoder.pth").is_file() and cfg.get("model_type") == "qwen3":   # Soprano
            cfg["architectures"] = ["Soprano"]
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
    write_sampling(args.output, sampling_model(arch, src, args.model_id))

    print(f"[bb] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
