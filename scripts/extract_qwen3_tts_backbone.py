"""Extract Qwen3-TTS's `talker.model.*` submodule as a standalone HF
Qwen3 model, ready for llama.cpp's `convert_hf_to_gguf.py`.

Qwen3-TTS's checkpoint stores the talker (28L Qwen3) under the
`talker.model.{layers,norm}.*` prefix, plus separate text and audio
embeddings (`talker.model.{codec,text}_embedding.weight`).  Llama.cpp's
converter expects a flat HF Qwen3 layout
(`model.{embed_tokens,layers,norm}.*` + `lm_head`).

We:
  - rename `talker.model.layers.* → model.layers.*` and
    `talker.model.norm.weight → model.norm.weight`,
  - set `model.embed_tokens.weight ← talker.model.codec_embedding.weight`
    (the audio token table — same hidden size as the talker).  The
    talker's text embedding lives at a different hidden size (2048)
    and goes through `text_projection` before joining the codec stream,
    so it isn't a direct fit for llama.cpp's embed slot,
  - tie `lm_head.weight` to `embed_tokens.weight` (the talker's own
    `codec_head` is the c0 head and lives in our codec_lm GGUF; the
    standalone Qwen3 dir's lm_head is unused at inference because we
    feed embeddings directly).

The talker uses MRoPE on its inputs, but for pure-text and pure-audio
sequences all 3 MRoPE channels share the same position ids (per the
upstream `get_rope_index` docstring), and the interleaved-MRoPE
machinery collapses to 1D RoPE.  Llama.cpp's `qwen3` arch applies 1D
RoPE — that matches the reduced case exactly.  For mixed audio+vision
sequences this would diverge, but Qwen3-TTS doesn't take vision input.

Usage:

    python scripts/extract_qwen3_tts_backbone.py \\
        --qwen Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --out  /tmp/qwen3_tts_talker_hf

    python <llama.cpp>/convert_hf_to_gguf.py /tmp/qwen3_tts_talker_hf \\
        --outfile models/qwen3_tts/qwen3_tts_talker.gguf --outtype f16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

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
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    return Path(local)


def load_state_dict(d: Path) -> dict[str, torch.Tensor]:
    for idx_name in ("model.safetensors.index.json",):
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
    ap.add_argument("--qwen", required=True,
                    help="HF id or local dir of the Qwen3-TTS checkpoint")
    ap.add_argument("--out", required=True,
                    help="Output dir to write the standalone HF Qwen3 model")
    args = ap.parse_args()

    src = resolve_source(args.qwen)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[extract] source: {src}")
    print(f"[extract] dest:   {dst}")

    cfg_in = json.loads((src / "config.json").read_text())
    archs = cfg_in.get("architectures") or []
    if "Qwen3TTSForConditionalGeneration" not in archs:
        raise SystemExit(
            f"unsupported source architectures={archs}; "
            f"this extractor handles Qwen3TTSForConditionalGeneration"
        )

    tk = cfg_in["talker_config"]
    hidden     = int(tk["hidden_size"])
    inter      = int(tk["intermediate_size"])
    n_layers   = int(tk["num_hidden_layers"])
    n_heads    = int(tk["num_attention_heads"])
    n_kv       = int(tk["num_key_value_heads"])
    head_dim   = int(tk["head_dim"])
    vocab      = int(tk["vocab_size"])
    max_pos    = int(tk["max_position_embeddings"])
    rms_eps    = float(tk["rms_norm_eps"])
    rope_theta = float(tk["rope_theta"])

    # Use the audio-codec vocab (talker.model.codec_embedding) as the
    # standalone Qwen3's embed_tokens.  At inference we feed embeddings
    # directly via `llama_batch.embd`, so the tokenizer / merges are
    # never exercised — but llama.cpp's converter still needs a vocab to
    # write `tok_embd`.
    #
    # Qwen3-TTS ships with the standard Qwen3 text tokenizer (vocab
    # 151936) but the talker's codec_embedding is only 3072 wide.
    # llama.cpp's `set_vocab` asserts the tokenizer fits the model's
    # vocab_size, so we pad the codec_embedding out to 151936 with
    # zeros and pretend the model has the full Qwen3 vocab.  The padded
    # rows are never read at inference (we feed embeds directly), and
    # `tie_word_embeddings=True` keeps the lm_head consistent.
    PADDED_VOCAB = 151936  # Qwen3-TTS text tokenizer vocab
    qwen3_cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size":               PADDED_VOCAB,
        "hidden_size":              hidden,
        "intermediate_size":        inter,
        "num_hidden_layers":        n_layers,
        "num_attention_heads":      n_heads,
        "num_key_value_heads":      n_kv,
        "head_dim":                 head_dim,
        "max_position_embeddings":  max_pos,
        "rms_norm_eps":             rms_eps,
        "rope_theta":               rope_theta,
        "hidden_act":               tk.get("hidden_act", "silu"),
        "attention_bias":           bool(tk.get("attention_bias", False)),
        "attention_dropout":        float(tk.get("attention_dropout", 0.0)),
        "initializer_range":        float(tk.get("initializer_range", 0.02)),
        "tie_word_embeddings":      True,
        "torch_dtype":              "float32",
        "use_cache":                True,
        "sliding_window":           None,
        "use_sliding_window":       False,
        # rope_scaling is intentionally omitted — Qwen3-TTS's
        # MRoPE-with-rope_type=default + mrope_section collapses to 1D
        # RoPE for non-vision sequences (the only case we use), so the
        # standalone Qwen3 just needs vanilla 1D RoPE.
    }
    (dst / "config.json").write_text(json.dumps(qwen3_cfg, indent=2))

    print("[extract] loading state dict …")
    sd = load_state_dict(src)
    out_sd: dict[str, torch.Tensor] = {}

    n_renamed = 0
    for k, v in sd.items():
        if k.startswith("talker.model.layers."):
            new_k = "model." + k[len("talker.model."):]
            out_sd[new_k] = v
            n_renamed += 1
        elif k == "talker.model.norm.weight":
            out_sd["model.norm.weight"] = v
            n_renamed += 1

    embd_codec = sd.get("talker.model.codec_embedding.weight")
    if embd_codec is None:
        raise SystemExit("missing talker.model.codec_embedding.weight in source state dict")
    if embd_codec.shape != (vocab, hidden):
        raise SystemExit(
            f"codec_embedding shape {tuple(embd_codec.shape)} != ({vocab}, {hidden})"
        )

    # Pad to PADDED_VOCAB rows with zeros.  cast to F32 first for ergonomics;
    # llama.cpp's converter will quantize back to the requested outtype.
    padded = torch.zeros((PADDED_VOCAB, hidden), dtype=embd_codec.dtype)
    padded[:vocab] = embd_codec
    out_sd["model.embed_tokens.weight"] = padded.contiguous()

    print(f"[extract] renamed {n_renamed} talker.model.* tensors")
    print(f"[extract] using talker.model.codec_embedding "
          f"({tuple(embd_codec.shape)}, {embd_codec.dtype}) padded to "
          f"({PADDED_VOCAB}, {hidden}) as embed_tokens + tied lm_head")

    from safetensors.torch import save_file
    save_file(out_sd, str(dst / "model.safetensors"))

    gen_cfg = {
        "_from_model_config": True,
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
