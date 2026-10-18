"""
Storage-precision policy for codec.cpp GGUFs.

codec.cpp computes every tensor at the type stored in the GGUF, so the converter
decides precision.  Two separate settings: the codec part and the LM-side adaptor
(tensors named ``lm.*``), each with

  float type   F32 / F16 / BF16 / AUTO (= the checkpoint's own precision).  Not
               quantization: storing a tensor at a float type is a precision choice.
  quantization optional Q8_0 / Q5_K_M / Q4_K_M for the tensors a converter marks
               quantizable (codec) or every eligible 2-D+ weight (LM part).

A float type is never wider than the checkpoint: asking for F32 from an F16 / BF16
checkpoint stores the checkpoint's precision.  Converter-generated DSP tables (mel
filterbanks, windows) and tensors a converter must store at F32 are exempt.

Converters keep writing their historical literals: ``"F16"`` means "this part's
float type", ``"F32"`` means "F32, capped at the checkpoint's precision".
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

FLOAT_TYPES = ("F32", "F16", "BF16")
QUANT_TYPES = ("Q8_0", "Q4_K_M", "Q5_K_M", "Q4_K", "Q5_K")
QUANT_BLOCK = {"Q8_0": 32, "Q4_K_M": 256, "Q5_K_M": 256, "Q4_K": 256, "Q5_K": 256}

# Generated F32 tables: never capped to the checkpoint precision.
DSP_TABLE = re.compile(r"mel_filters|mel\.filters|mel_basis|mel_fb|(?:^|\.)window$")


@dataclass
class PartPolicy:
    float_type: str = "F16"            # F32 / F16 / BF16 / AUTO
    quantization: Optional[str] = None
    source: Optional[str] = None       # checkpoint precision: F32 / F16 / BF16 / None (unknown)

    def resolved_float(self) -> str:
        if self.float_type == "AUTO":
            return self.source or "F16"
        return self.float_type

    def cap(self, t: str) -> str:
        # never wider than the checkpoint
        if t == "F32" and self.source in ("F16", "BF16"):
            return self.source
        return t


@dataclass
class PrecisionPolicy:
    codec: PartPolicy = field(default_factory=PartPolicy)
    lm: PartPolicy = field(default_factory=PartPolicy)
    must_f32: tuple = ()               # extra regexes stored F32 regardless of the checkpoint

    def part(self, name: str) -> PartPolicy:
        return self.lm if name.startswith("lm.") else self.codec

    def resolve(self, name: str, shape, st_dtype: Optional[str], np_dtype_name: str) -> Optional[str]:
        """Final storage type for one tensor (None: leave the writer's own inference)."""
        if st_dtype is None:
            if np_dtype_name not in ("float32", "float16"):
                return None
            st_dtype = "F32" if np_dtype_name == "float32" else "F16"
        if st_dtype not in FLOAT_TYPES and st_dtype not in QUANT_TYPES:
            return st_dtype                                  # I32 etc.
        p = self.part(name)
        if DSP_TABLE.search(name) or any(re.search(m, name) for m in self.must_f32):
            return "F32" if st_dtype in FLOAT_TYPES else st_dtype
        lm = p is self.lm
        if st_dtype in QUANT_TYPES:
            if not lm:
                return st_dtype                              # the codec converter chose it
            st_dtype = "F16"                                 # LM part: decided below
        if lm and p.quantization and st_dtype == "F16" and len(shape) >= 2 and \
                int(shape[-1]) % QUANT_BLOCK[p.quantization] == 0:
            return p.quantization
        if st_dtype == "F16":
            return p.cap(p.resolved_float())
        if st_dtype == "BF16":
            return p.cap("BF16") if p.source != "F16" else "BF16"
        return p.cap("F32")


_POLICY: Optional[PrecisionPolicy] = None


def set_policy(policy: Optional[PrecisionPolicy]) -> None:
    global _POLICY
    _POLICY = policy


def get_policy() -> Optional[PrecisionPolicy]:
    return _POLICY


# ---------------------------------------------------------------------------
# Checkpoint precision detection (dominant float type by bytes)
# ---------------------------------------------------------------------------

_ST = {"F32": "F32", "F16": "F16", "BF16": "BF16", "F64": "F32"}
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def _safetensors_dtypes(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for k, v in header.items():
        if k == "__metadata__":
            continue
        d = _ST.get(v.get("dtype"))
        if d is not None:
            lo, hi = v["data_offsets"]
            yield d, hi - lo


def _torch_dtypes(path: Path):
    import torch
    try:
        obj = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        return
    stack = [obj]
    names = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.float64: "F32"}
    while stack:
        o = stack.pop()
        if isinstance(o, torch.Tensor):
            d = names.get(o.dtype)
            if d is not None:
                yield d, o.numel() * o.element_size()
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
        elif hasattr(o, "state_dict"):
            try:
                stack.append(o.state_dict())
            except Exception:
                pass


def _weight_files(src) -> Iterable[Path]:
    p = Path(str(src))
    if not p.exists():
        try:
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(repo_id=str(src), local_files_only=True))
        except Exception:
            return []
    if p.is_file():
        return [p]
    return sorted(q for q in p.rglob("*") if q.is_file() and q.suffix in _WEIGHT_SUFFIXES)


def detect_source_precision(src) -> Optional[str]:
    """Dominant float type of a checkpoint (path or cached HF repo id); None if unknown."""
    if src is None:
        return None
    total = {}
    for f in _weight_files(src):
        it = _safetensors_dtypes(f) if f.suffix == ".safetensors" else _torch_dtypes(f)
        for d, nbytes in it:
            total[d] = total.get(d, 0) + nbytes
    if not total:
        return None
    return max(total.items(), key=lambda kv: kv[1])[0]
