"""CSM (Sesame) converter — bundles the Mimi codec + residual_depth_ar
LM adaptor into a single GGUF.

The HF release `sesame/csm-1b` packs three things into one safetensors:
- `codec_model.*`: a 32-codebook Mimi audio codec (24 kHz, 12.5 Hz frames).
- `backbone_model.*`: a Llama-3.2-1B with `tok_embeddings` set to Identity;
  consumed by codec_lm's caller via llama.cpp's `llama` arch (not in this
  GGUF — the user runs `convert_hf_to_gguf.py` on a separately-extracted
  HF dir, mirroring the MOSS-TTSD-v0.5 flow).
- `depth_decoder.*` + `lm_head` + `embed_text_tokens` + audio embed table:
  the codec_lm side, mapped to `lm.*` via the `lm_adaptor/csm.py` handler.

This converter handles the codec + lm part in one pass — the user only
needs the backbone GGUF as a separate side input at inference time.

Usage (auto-detected by `convert-to-gguf.py` when `architectures` includes
`CsmForConditionalGeneration`):

    python scripts/convert-to-gguf.py --model-id sesame/csm-1b \\
        --output models/csm/csm.gguf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import BaseConverter
from .mimi import (
    MimiConverter,
    transform_tensor_for_codec,
    map_tensor_names,
    shorten_tensor_name,
    build_weight_transforms,
)
from utils.gguf_writer import GGUFWriter


# Tensor prefixes inside the CSM checkpoint that belong to the audio
# codec (we strip the leading `codec_model.` to reuse Mimi's
# tensor-name mapping verbatim).
_CSM_CODEC_PREFIX = "codec_model."


def _strip_codec_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a sub-state-dict containing only `codec_model.*` entries,
    with the prefix removed so MimiConverter's mappers see HF Mimi names."""
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith(_CSM_CODEC_PREFIX):
            out[k[len(_CSM_CODEC_PREFIX):]] = v
    return out


def _csm_to_mimi_config(csm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The codec sub-config CSM ships under `codec_config` already follows
    the kyutai/mimi schema 1:1; just lift it out."""
    cc = csm_cfg.get("codec_config")
    if cc is None:
        raise RuntimeError("CSM config is missing `codec_config` block")
    return cc


class CsmConverter(BaseConverter):
    """Convert a Sesame CSM checkpoint (bundled Mimi + residual depth-AR LM)."""

    @property
    def model_type(self) -> str:
        return "csm"

    @property
    def architecture(self) -> str:
        # Architecture written into the GGUF is `mimi` so codec_model loads
        # via the existing Mimi vtable; the lm.* adaptor sits orthogonal
        # to the codec arch.
        return "mimi"

    def load_from_checkpoint(self, checkpoint_dir: Path) -> None:
        from huggingface_hub import snapshot_download  # noqa: F401  (parity with siblings)
        path = Path(checkpoint_dir)
        cfg_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config.json next to {path}")
        self.config = json.loads(cfg_path.read_text())
        archs = self.config.get("architectures") or []
        if "CsmForConditionalGeneration" not in archs:
            raise RuntimeError(
                f"unexpected architectures {archs!r}; "
                f"CsmConverter expects CsmForConditionalGeneration"
            )

        # CSM ships its weights as `transformers-NNNNN-of-NNNNN.safetensors`
        # shards (note the `transformers-` prefix, not the HF default
        # `model-`).  Use the index file to enumerate.
        from safetensors.torch import load_file as load_st_torch
        idx_path = path / "transformers.safetensors.index.json"
        if idx_path.is_file():
            idx = json.loads(idx_path.read_text())
            shard_files = sorted({path / fn for fn in idx["weight_map"].values()})
        else:
            single = path / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError(
                    f"no transformers.safetensors.index.json or model.safetensors in {path}"
                )
            shard_files = [single]

        import torch
        sd: Dict[str, Any] = {}
        for fn in shard_files:
            shard = load_st_torch(str(fn))
            for k, v in shard.items():
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float32)
                # Cast to numpy via torch — keeps backbone_model.* as-is for
                # the lm adaptor handler, which expects torch tensors via
                # _load_lm_source(); but Mimi-side tensors get numpy via
                # transform_tensor_for_codec.  We standardise on numpy
                # everywhere for consistency.
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

        # --- 1. Codec section (Mimi) ----------------------------------
        codec_sd  = _strip_codec_prefix(self.state_dict)
        codec_cfg = _csm_to_mimi_config(self.config)
        if not codec_sd:
            raise RuntimeError("CSM checkpoint has no `codec_model.*` tensors")

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

        # --- 2. LM section (residual_depth_ar) ------------------------
        # Drive the lm_adaptor handler directly with the CSM state_dict
        # we already have in memory, bypassing the usual HF-id → snapshot
        # download path (the LM weights live in the same checkpoint dir).
        from .lm_adaptor.csm import dump as csm_lm_dump
        csm_lm_dump(writer, self.state_dict, self.config, verbose=self.verbose)

        self._warn_if_no_quantized()
        writer.write()
        self.log(
            f"Wrote CSM (Mimi codec + residual_depth_ar LM adaptor) GGUF to {output_path}"
        )
