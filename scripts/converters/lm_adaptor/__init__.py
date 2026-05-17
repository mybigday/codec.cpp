"""LM-adaptor dump helpers.

A codec converter (e.g. XYTokenizerConverter) writes the codec.* tensors;
when the user also passes a `--lm-source`, the codec converter calls
`dump_lm_into(writer, lm_source)` here.  Dispatch on the LM source's
`config.json` `architectures[0]` and hand off to a per-arch handler that
writes `lm.*` tensors and `codec.lm.*` metadata into the same GGUF.

Goal: codec + lm adaptor in one GGUF, with the codec converter as the
entry point.  The LM is auxiliary; new LM source families plug in here
without touching the codec converter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def dump_lm_into(writer, lm_source, *, verbose: bool = False) -> None:
    """Load an LM-source checkpoint, dispatch to the right handler, and
    write `lm.*` + `codec.lm.*` into the supplied GGUFWriter."""
    sd, cfg = _load_lm_source(lm_source, verbose=verbose)

    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""

    if arch in {"MossTTSDForCausalLM", "MossTTSDelayModel", "AsteroidTTSModel"}:
        from . import moss_ttsd
        moss_ttsd.dump(writer, sd, cfg, arch_name=arch, verbose=verbose)
        return
    if arch == "Qwen3TTSForConditionalGeneration":
        from . import qwen3_tts
        qwen3_tts.dump(writer, sd, cfg, verbose=verbose)
        return
    if arch == "MoshiForConditionalGeneration":
        from . import moshi
        moshi.dump(writer, sd, cfg, verbose=verbose)
        return
    if arch == "Lfm2AudioForConditionalGeneration":
        from . import lfm2_audio
        lfm2_audio.dump(writer, sd, cfg, verbose=verbose)
        return
    if arch == "MossTTSRealtime":
        from . import moss_tts_local
        moss_tts_local.dump(writer, sd, cfg, verbose=verbose)
        return
    if arch == "MossTTSNanoForCausalLM":
        # Stub: GGUF emission is shaped, but the GPT-2 depth-block
        # runtime (LayerNorm + GELU + fused c_attn split + abs wpe) is
        # still pending — the dump() will raise until that lands.
        from . import moss_tts_local
        moss_tts_local.dump(writer, sd, cfg, verbose=verbose)
        return

    raise NotImplementedError(
        f"unsupported LM-source architecture: {arch!r}; "
        f"add a handler in scripts/converters/lm_adaptor/"
    )


# ---------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------

def _load_lm_source(lm_source, *, verbose: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Resolve `lm_source` (a local dir/file path or a HF repo id) into
    (state_dict, config) tuples.  Mirrors the codec converters' approach
    so users can pass either."""

    if isinstance(lm_source, Path) or _looks_like_path(lm_source):
        local = Path(lm_source)
    else:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(
            repo_id=str(lm_source),
            allow_patterns=["*.safetensors", "*.json", "*.py"],
        ))

    cfg_path = local / "config.json" if local.is_dir() else local.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json next to {local}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    sd = _load_state_dict(local)
    if verbose:
        print(f"[lm_adaptor] loaded {len(sd)} tensors from {local}")
    return sd, cfg


def _looks_like_path(s) -> bool:
    return ("/" in str(s) and Path(str(s)).exists()) or Path(str(s)).is_dir()


def _load_state_dict(path: Path) -> Dict[str, np.ndarray]:
    """Load a state-dict from a directory that contains either a single
    `model.safetensors` or a sharded `model.safetensors.index.json` set.

    Goes through safetensors.torch to handle bfloat16 (numpy has no
    native bf16 type) — every tensor is decoded to torch and cast to
    float32 before being handed back as a numpy array.  Per-arch
    handlers can downcast back to F16 / quantize as needed via the
    GGUFWriter pipeline.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required for LM-source loading "
            "(needed to decode bfloat16 safetensors that `safetensors.numpy` "
            "can't handle natively)"
        ) from e
    from safetensors.torch import load_file as load_st_torch

    def _shard_to_numpy(shard_path: Path) -> Dict[str, np.ndarray]:
        sd_torch = load_st_torch(str(shard_path))
        out: Dict[str, np.ndarray] = {}
        for k, v in sd_torch.items():
            # bfloat16 -> float32 is the only safe round-trip via numpy.
            # Per-arch handlers downcast as needed during writing.
            if v.dtype == torch.bfloat16:
                v = v.to(torch.float32)
            out[k] = v.detach().cpu().numpy()
        return out

    if path.is_file() and path.suffix == ".safetensors":
        return _shard_to_numpy(path)

    if path.is_dir():
        idx = path / "model.safetensors.index.json"
        single = path / "model.safetensors"
        if idx.is_file():
            with open(idx) as f:
                index = json.load(f)
            shards = sorted({path / fn for fn in index.get("weight_map", {}).values()})
            sd: Dict[str, np.ndarray] = {}
            for shard in shards:
                sd.update(_shard_to_numpy(shard))
            return sd
        if single.is_file():
            return _shard_to_numpy(single)

    raise FileNotFoundError(
        f"could not find safetensors at {path}; expected model.safetensors "
        f"or model.safetensors.index.json"
    )
