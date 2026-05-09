"""Extract a MOSS-TTSD LM's `model.language_model.*` submodule as a
standalone HF Qwen3 model, ready for llama.cpp's `convert_hf_to_gguf.py`.

MOSS-TTSD's checkpoint stores the Qwen3 backbone under the
`model.language_model.*` prefix and 8 audio-channel embedding tables
under `model.embedding_list.{0..7}.weight`.  llama.cpp's converter
expects a flat HF Qwen3 layout (`model.{embed_tokens,layers,norm}.*` +
`lm_head`).

We rename `model.language_model.*` → `model.*`, and drop in
`model.embedding_list.0.weight` as `model.embed_tokens.weight` /
`lm_head.weight` (tied).  At codec_lm runtime the host LLM only ever
receives input via `llama_batch.embd` (the channel-summed audio
embedding produced by codec_lm_compose_audio_embd), so the embed_tokens
weight is unused — but llama.cpp requires it to be present + shaped
correctly to load the model.

Usage:
    python scripts/extract_qwen3_backbone.py \
        --moss <hf-id-or-local-dir> --out <hf-style-dir>

Then:
    python <llama.cpp>/convert_hf_to_gguf.py <hf-style-dir> \
        --outfile <out>/qwen3_backbone.gguf --outtype f16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Files we copy verbatim from the source dir (when present) — the
# tokenizer side is identical to MOSS-TTSD's, since channel-0 vocab IS
# the host LLM's tokenizer.
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
    # HF id — snapshot_download
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        repo_id=src,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "merges.txt"],
    )
    return Path(local)


def load_state_dict(d: Path) -> dict[str, torch.Tensor]:
    idx = d / "model.safetensors.index.json"
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
    ap.add_argument("--moss", required=True,
                    help="HF id or local dir of the MOSS-TTSD checkpoint")
    ap.add_argument("--out", required=True,
                    help="Output dir to write the standalone HF Qwen3 model")
    ap.add_argument("--keep-bf16", action="store_true",
                    help="Keep bf16 dtype (default: convert to bf16 if not "
                         "already, to match Qwen3's preferred dtype)")
    args = ap.parse_args()

    src = resolve_source(args.moss)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[extract] source: {src}")
    print(f"[extract] dest:   {dst}")

    # ── 1. configuration ────────────────────────────────────────────
    cfg_in = json.loads((src / "config.json").read_text())
    if cfg_in.get("model_type") not in ("moss_ttsd",):
        # MOSS-TTSD-v0 declares model_type=qwen3 explicitly; v0.5/v0.7 use
        # moss_ttsd; v1.0/MOSS-TTS use moss_tts_delay (different layout).
        # Be permissive and just check the architectures field.
        archs = cfg_in.get("architectures") or []
        if not any(a in archs for a in
                   ("MossTTSDForCausalLM", "AsteroidTTSModel")):
            raise SystemExit(
                f"unsupported MOSS source architectures={archs}; "
                f"this extractor handles MossTTSDForCausalLM / AsteroidTTSModel"
            )

    # MOSS-TTSD v0.5/v0.7 inherits Qwen3's transformer config verbatim,
    # nested at the top level rather than under `language_config` (which
    # is the v1.0/MOSS-TTS pattern).  Pull the fields llama.cpp's
    # converter looks at.
    qwen3_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size":               int(cfg_in["vocab_size"]),
        "hidden_size":              int(cfg_in["hidden_size"]),
        "intermediate_size":        int(cfg_in["intermediate_size"]),
        "num_hidden_layers":        int(cfg_in["num_hidden_layers"]),
        "num_attention_heads":      int(cfg_in["num_attention_heads"]),
        "num_key_value_heads":      int(cfg_in["num_key_value_heads"]),
        "head_dim":                 int(cfg_in["head_dim"]),
        "max_position_embeddings":  int(cfg_in["max_position_embeddings"]),
        "rms_norm_eps":             float(cfg_in["rms_norm_eps"]),
        "rope_theta":               float(cfg_in["rope_theta"]),
        "hidden_act":               cfg_in.get("hidden_act", "silu"),
        "attention_bias":           bool(cfg_in.get("attention_bias", False)),
        "attention_dropout":        float(cfg_in.get("attention_dropout", 0.0)),
        "initializer_range":        float(cfg_in.get("initializer_range", 0.02)),
        "tie_word_embeddings":      True,    # we tie embed ↔ lm_head
        "torch_dtype":              cfg_in.get("torch_dtype", "bfloat16"),
        "use_cache":                True,
        "bos_token_id":             cfg_in.get("bos_token_id"),
        "eos_token_id":             cfg_in.get("eos_token_id"),
    }
    if cfg_in.get("rope_scaling"):
        qwen3_cfg["rope_scaling"] = cfg_in["rope_scaling"]
    if cfg_in.get("sliding_window") is not None:
        qwen3_cfg["sliding_window"] = cfg_in["sliding_window"]
    (dst / "config.json").write_text(json.dumps(qwen3_cfg, indent=2))

    # ── 2. state dict (rename + use embedding_list[0] as embed_tokens) ──
    print("[extract] loading state dict …")
    sd = load_state_dict(src)
    out_sd: dict[str, torch.Tensor] = {}

    n_renamed = 0
    for k, v in sd.items():
        if k.startswith("model.language_model."):
            new_k = "model." + k[len("model.language_model."):]
            out_sd[new_k] = v
            n_renamed += 1

    embd0 = sd.get("model.embedding_list.0.weight")
    if embd0 is None:
        raise SystemExit("missing model.embedding_list.0.weight in source state dict")
    # safetensors refuses shared-memory tensors; with
    # tie_word_embeddings=True, HF's loader regenerates lm_head from
    # embed_tokens at load time, so saving only embed_tokens is
    # sufficient.  llama.cpp's converter also handles tied embeddings
    # correctly (writes one tok_embd tensor, derives output via tying).
    out_sd["model.embed_tokens.weight"] = embd0.clone().contiguous()

    # Drop any stray embed_tokens that might have been saved (it isn't
    # in this checkpoint, but be defensive in case other variants do).
    out_sd.pop("model.embed_tokens.weight.dup", None)

    print(f"[extract] renamed {n_renamed} language_model.* tensors")
    print(f"[extract] using embedding_list.0 ({tuple(embd0.shape)}, {embd0.dtype}) "
          f"as embed_tokens + lm_head")

    save_file(out_sd, str(dst / "model.safetensors"))

    # Generation config — minimal, with the BOS/EOS the source declared.
    gen_cfg = {
        "_from_model_config": True,
        "bos_token_id": qwen3_cfg.get("bos_token_id"),
        "eos_token_id": qwen3_cfg.get("eos_token_id"),
        "transformers_version": "4.40.0",
    }
    (dst / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))

    # ── 3. tokenizer ────────────────────────────────────────────────
    for fn in TOKENIZER_FILES:
        if (src / fn).exists():
            shutil.copy(src / fn, dst / fn)

    print(f"[extract] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
