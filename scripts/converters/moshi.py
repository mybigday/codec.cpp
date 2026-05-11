"""Moshi (Kyutai) converter — bundles Mimi codec + residual_depth_ar
(flexible) LM adaptor into a single GGUF.

`kmhf/hf-moshiko` and `kmhf/hf-moshika` pack three things into one
safetensors:
  - `audio_encoder.*`: the Mimi codec (shared with sesame/csm-1b's
    `codec_model.*`).
  - `decoder.*`: a 32-layer Helium backbone (Llama-style); consumed by
    callers via llama.cpp's `llama` arch.  Extracted separately
    (extractor lands with the runtime work).
  - `depth_decoder.*` + `embed_tokens.*`: the codec_lm side.  This
    converter maps the `depth_decoder.*` half to `lm.*` via
    `lm_adaptor/moshi.py`.  The `embed_tokens.*` half (dual-stream
    backbone-side audio + text tables) lives with the extracted
    backbone, since llama.cpp owns input composition for Moshi.

Usage (auto-detected by `convert-to-gguf.py` when `architectures`
includes `MoshiForConditionalGeneration`):

    python scripts/convert-to-gguf.py --model-id kmhf/hf-moshiko \\
        --output models/moshi/moshi.gguf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import BaseConverter
from .mimi import MimiConverter
from utils.gguf_writer import GGUFWriter


_MOSHI_CODEC_PREFIX = "audio_encoder."


def _strip_codec_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sub-state-dict with `audio_encoder.*` entries only,
    with the prefix removed so MimiConverter's mappers see canonical
    HF Mimi names."""
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith(_MOSHI_CODEC_PREFIX):
            out[k[len(_MOSHI_CODEC_PREFIX):]] = v
    return out


def _moshi_to_mimi_config(moshi_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Lift Moshi's `audio_encoder_config` (Mimi-shaped) verbatim."""
    cc = moshi_cfg.get("audio_encoder_config")
    if cc is None:
        raise RuntimeError("Moshi config is missing `audio_encoder_config` block")
    return cc


class MoshiConverter(BaseConverter):
    """Convert a Kyutai Moshi checkpoint (bundled Mimi + flexible
    depth-AR LM)."""

    @property
    def model_type(self) -> str:
        return "moshi"

    @property
    def architecture(self) -> str:
        # The codec section uses Mimi's arch tag; the lm.* adaptor is
        # orthogonal and discovered via `codec.lm.*` metadata.
        return "mimi"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        path = Path(checkpoint_dir)
        cfg_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config.json next to {path}")
        self.config = json.loads(cfg_path.read_text())
        archs = self.config.get("architectures") or []
        if "MoshiForConditionalGeneration" not in archs:
            raise RuntimeError(
                f"unexpected architectures {archs!r}; "
                f"MoshiConverter expects MoshiForConditionalGeneration"
            )

        from safetensors.torch import load_file as load_st_torch
        idx_path = path / "model.safetensors.index.json"
        if idx_path.is_file():
            idx = json.loads(idx_path.read_text())
            shard_files = sorted({path / fn for fn in idx["weight_map"].values()})
        else:
            single = path / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(
                    f"no model.safetensors.index.json or model.safetensors in {path}"
                )
            shard_files = [single]

        import torch
        sd: Dict[str, Any] = {}
        for fn in shard_files:
            shard = load_st_torch(str(fn))
            for k, v in shard.items():
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float32)
                sd[k] = v.detach().cpu().numpy()
        self.state_dict = sd

    def load_from_huggingface(self, model_id: str) -> None:
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
        )
        self.load_from_checkpoint(Path(local))

    def convert_and_save(self, output_path: Path) -> None:
        if self.state_dict is None or self.config is None:
            raise RuntimeError("No model loaded.")

        # --- Codec section (Mimi) -------------------------------------
        codec_sd  = _strip_codec_prefix(self.state_dict)
        codec_cfg = _moshi_to_mimi_config(self.config)
        if not codec_sd:
            raise RuntimeError("Moshi checkpoint has no `audio_encoder.*` tensors")

        mimi = MimiConverter(
            quantization=self.quantization,
            quantize_codebook=self.quantize_codebook,
            verbose=self.verbose,
        )
        mimi.state_dict = codec_sd
        mimi.config     = codec_cfg

        writer = GGUFWriter(output_path, self.architecture)
        self._reset_quant_stats()
        mimi._reset_quant_stats()
        mimi.write_into(writer)

        # --- LM section (residual_depth_ar flexible) ------------------
        from .lm_adaptor.moshi import dump as moshi_lm_dump
        moshi_lm_dump(writer, self.state_dict, self.config,
                      verbose=self.verbose)

        self._warn_if_no_quantized()
        writer.write()
        self.log(
            f"Wrote Moshi (Mimi codec + flexible residual_depth_ar LM "
            f"adaptor) GGUF to {output_path}"
        )
