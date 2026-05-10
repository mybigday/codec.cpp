"""Extract a CSM (Sesame) checkpoint's `backbone_model.*` submodule as a
standalone HF Llama model, ready for llama.cpp's `convert_hf_to_gguf.py`.

CSM's checkpoint stores the Llama-3.2-1B backbone under
`backbone_model.{embed_tokens.embed_audio_tokens, layers, norm}.*`, plus
a separate text-token embedding `embed_text_tokens.weight` and a c0
`lm_head.weight` (over the audio vocab).

llama.cpp's converter expects a flat HF Llama layout
(`model.{embed_tokens, layers, norm}.*` + `lm_head.weight`).  We:
  - rename `backbone_model.layers.* → model.layers.*` and
    `backbone_model.norm.weight → model.norm.weight`,
  - set `model.embed_tokens.weight ← embed_text_tokens.weight` (the text
    vocab the backbone was trained against),
  - tie `lm_head.weight` to `embed_tokens.weight` (CSM's actual lm_head
    is over a different vocab — the audio side — and unused by us; we
    only feed embeds and read hidden states),
  - drop `backbone_model.embed_tokens.embed_audio_tokens.weight` (it's
    the audio embed table, lives in our codec_lm GGUF).

At codec_lm runtime the host LLM only ever receives input via
`llama_batch.embd` (composed audio + text embeddings produced by
`codec_lm_compose_audio_embd`), so the embed_tokens weight is unused
at inference — but llama.cpp requires it to be present + shaped
correctly to load the model.

Usage:
    python scripts/extract_csm_backbone.py \\
        --csm <hf-id-or-local-dir> --out <hf-style-dir>

Then:
    python <llama.cpp>/convert_hf_to_gguf.py <hf-style-dir> \\
        --outfile <out>/llama_backbone.gguf --outtype f16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
]


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


def load_state_dict(d: Path) -> dict[str, torch.Tensor]:
    # CSM ships shards as `transformers-NNNNN-of-NNNNN.safetensors` with a
    # `transformers.safetensors.index.json`; the legacy `model.safetensors`
    # alongside is a different (older?) layout that does NOT contain
    # `embed_text_tokens.weight`.  Always prefer the transformers index.
    for idx_name in ("transformers.safetensors.index.json",
                     "model.safetensors.index.json"):
        idx = d / idx_name
        if idx.is_file():
            manifest = json.loads(idx.read_text())["weight_map"]
            shards = sorted({d / fn for fn in manifest.values()})
            sd: dict[str, torch.Tensor] = {}
            for s in shards:
                sd.update(load_file(str(s)))
            return sd
    return load_file(str(d / "model.safetensors"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csm", required=True,
                    help="HF id or local dir of the CSM checkpoint")
    ap.add_argument("--out", required=True,
                    help="Output dir to write the standalone HF Llama model")
    args = ap.parse_args()

    src = resolve_source(args.csm)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[extract] source: {src}")
    print(f"[extract] dest:   {dst}")

    cfg_in = json.loads((src / "config.json").read_text())
    archs = cfg_in.get("architectures") or []
    if "CsmForConditionalGeneration" not in archs:
        raise SystemExit(
            f"unsupported source architectures={archs}; "
            f"this extractor handles CsmForConditionalGeneration"
        )

    text_vocab = int(cfg_in["text_vocab_size"])
    hidden     = int(cfg_in["hidden_size"])
    inter      = int(cfg_in["intermediate_size"])
    n_layers   = int(cfg_in["num_hidden_layers"])
    n_heads    = int(cfg_in["num_attention_heads"])
    n_kv       = int(cfg_in["num_key_value_heads"])
    head_dim   = int(cfg_in["head_dim"])
    max_pos    = int(cfg_in["max_position_embeddings"])
    rms_eps    = float(cfg_in["rms_norm_eps"])
    rope_theta = float(cfg_in["rope_theta"])

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size":               text_vocab,
        "hidden_size":              hidden,
        "intermediate_size":        inter,
        "num_hidden_layers":        n_layers,
        "num_attention_heads":      n_heads,
        "num_key_value_heads":      n_kv,
        "head_dim":                 head_dim,
        "max_position_embeddings":  max_pos,
        "rms_norm_eps":             rms_eps,
        "rope_theta":               rope_theta,
        "hidden_act":               cfg_in.get("hidden_act", "silu"),
        "attention_bias":           bool(cfg_in.get("attention_bias", False)),
        "attention_dropout":        float(cfg_in.get("attention_dropout", 0.0)),
        "mlp_bias":                 bool(cfg_in.get("mlp_bias", False)),
        "initializer_range":        float(cfg_in.get("initializer_range", 0.02)),
        "tie_word_embeddings":      True,
        "torch_dtype":              cfg_in.get("torch_dtype", "float32"),
        "use_cache":                True,
        "bos_token_id":             cfg_in.get("bos_token_id", 128000),
        "eos_token_id":             cfg_in.get("eos_token_id", 128001),
        "pad_token_id":             cfg_in.get("pad_token_id", 128004),
    }
    if cfg_in.get("rope_scaling"):
        llama_cfg["rope_scaling"] = cfg_in["rope_scaling"]
    (dst / "config.json").write_text(json.dumps(llama_cfg, indent=2))

    print("[extract] loading state dict …")
    sd = load_state_dict(src)
    out_sd: dict[str, torch.Tensor] = {}

    n_renamed = 0
    for k, v in sd.items():
        # `backbone_model.layers.*`, `backbone_model.norm.weight`
        if k.startswith("backbone_model.layers."):
            new_k = "model." + k[len("backbone_model."):]
            out_sd[new_k] = v
            n_renamed += 1
        elif k == "backbone_model.norm.weight":
            out_sd["model.norm.weight"] = v
            n_renamed += 1

    embd_text = sd.get("embed_text_tokens.weight")
    if embd_text is None:
        raise SystemExit("missing embed_text_tokens.weight in source state dict")
    if embd_text.shape != (text_vocab, hidden):
        raise SystemExit(
            f"embed_text_tokens shape {tuple(embd_text.shape)} != "
            f"({text_vocab}, {hidden})")
    out_sd["model.embed_tokens.weight"] = embd_text.clone().contiguous()

    print(f"[extract] renamed {n_renamed} backbone_model.* tensors")
    print(f"[extract] using embed_text_tokens "
          f"({tuple(embd_text.shape)}, {embd_text.dtype}) as embed_tokens "
          f"+ tied lm_head")

    save_file(out_sd, str(dst / "model.safetensors"))

    gen_cfg = {
        "_from_model_config": True,
        "bos_token_id": llama_cfg.get("bos_token_id"),
        "eos_token_id": llama_cfg.get("eos_token_id"),
        "transformers_version": "4.40.0",
    }
    (dst / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))

    for fn in TOKENIZER_FILES:
        if (src / fn).exists():
            shutil.copy(src / fn, dst / fn)

    print(f"[extract] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
