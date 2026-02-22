"""
GGUF file writer utility.

Writes GGUF format files with support for multiple tensor types
including quantized formats (Q4_K_M, Q5_K_M, Q8_0).
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

# GGUF constants
MAX_TENSOR_NAME = 63
KV_UINT32 = 4
KV_BOOL = 7
KV_STRING = 8
ALIGNMENT = 32

# GGML type mapping (matches ggml.h)
TENSOR_TYPE_MAP = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,   # Q4_K_M uses this
    "Q4_K_M": 12, # Alias for Q4_K (4-bit K-quants variant)
    "Q5_K": 13,   # Q5_K_M uses this
    "Q5_K_M": 13, # Alias for Q5_K (5-bit K-quants variant)
    "Q6_K": 14,
}


def _u64(n: int) -> bytes:
    return struct.pack("<Q", int(n))


def _i64(n: int) -> bytes:
    return struct.pack("<q", int(n))


def _u32(n: int) -> bytes:
    return struct.pack("<I", int(n))


def _i32(n: int) -> bytes:
    return struct.pack("<i", int(n))


def _str_bytes(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


class GGUFWriter:
    """Simple GGUF file writer with quantization support."""
    
    def __init__(self, path: Union[str, Path], architecture: str):
        self.path = Path(path)
        self.kv: List[Tuple[str, int, Any]] = [("general.architecture", KV_STRING, architecture)]
        self.tensors: List[Tuple[str, str, List[int], bytes]] = []
        
    def add_name(self, name: str):
        """Add model name metadata."""
        self.kv.append(("general.name", KV_STRING, name))
        
    def add_uint32(self, key: str, val: int):
        """Add uint32 metadata."""
        self.kv.append((key, KV_UINT32, int(val)))
        
    def add_bool(self, key: str, val: bool):
        """Add boolean metadata."""
        self.kv.append((key, KV_BOOL, bool(val)))
        
    def add_string(self, key: str, val: str):
        """Add string metadata."""
        self.kv.append((key, KV_STRING, str(val)))
        
    def add_tensor(self, name: str, arr: np.ndarray, st_dtype: str = None):
        """
        Add a tensor to the GGUF file.
        
        Args:
            name: Tensor name
            arr: Numpy array with data
            st_dtype: Storage type ("F32", "F16", "Q8_0", "Q4_K_M", "Q5_K_M")
                     If None, inferred from arr.dtype
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        arr = np.ascontiguousarray(arr)

        # Determine storage type
        if st_dtype is None:
            dtype_name = str(arr.dtype)
            if dtype_name == "float32":
                st_dtype = "F32"
            elif dtype_name == "float16":
                st_dtype = "F16"
            else:
                raise ValueError(f"Unsupported dtype: {arr.dtype} ({name})")

        if st_dtype not in TENSOR_TYPE_MAP:
            raise ValueError(f"Unsupported tensor type: {st_dtype} ({name})")

        # Prepare data based on type
        if st_dtype in ("F32", "F16"):
            if st_dtype == "F32" and arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if st_dtype == "F16" and arr.dtype != np.float16:
                arr = arr.astype(np.float16)
            data = arr.tobytes(order="C")
        else:
            # Import here to avoid circular dependency
            from .quantization import (
                quantize_tensor_q8_0, 
                quantize_tensor_q4_k_m, 
                quantize_tensor_q5_k_m
            )
            if st_dtype in ("Q4_0", "Q4_1", "Q4_K", "Q4_K_M"):
                data = quantize_tensor_q4_k_m(arr)
                st_dtype = "Q4_K"  # Use base type for GGML
            elif st_dtype in ("Q5_0", "Q5_1", "Q5_K", "Q5_K_M"):
                data = quantize_tensor_q5_k_m(arr)
                st_dtype = "Q5_K"
            elif st_dtype == "Q8_0":
                data = quantize_tensor_q8_0(arr)
            else:
                raise ValueError(f"Quantization not implemented for {st_dtype}")

        self.tensors.append((name, st_dtype, list(arr.shape), data))
        
    def write(self):
        """Write the GGUF file to disk."""
        n_tensors = len(self.tensors)
        n_kv = len(self.kv)

        # Encode KV data
        kv_blob = bytearray()
        for key, t, v in self.kv:
            kv_blob += self._encode_kv(key, t, v)

        # Build tensor info and data blob
        tensor_infos = bytearray()
        tensor_meta = []
        cur_off = 0
        
        for name, st_dtype, shape_rev, data in self.tensors:
            ggml_type = TENSOR_TYPE_MAP[st_dtype]
            data_off = _align_up(cur_off, ALIGNMENT)
            tensor_meta.append((name, ggml_type, shape_rev, data_off, data))
            cur_off = data_off + len(data)

        # Materialize single data blob
        data_blob = bytearray(cur_off)
        for name, ggml_type, shape_rev, data_off, data in tensor_meta:
            data_blob[data_off:data_off + len(data)] = data
            tensor_infos += _str_bytes(name)
            tensor_infos += _u32(len(shape_rev))
            for dim in shape_rev:
                tensor_infos += _i64(int(dim))
            tensor_infos += _i32(ggml_type)
            tensor_infos += _u64(data_off)

        # Build header
        header = bytearray()
        header += b"GGUF"
        header += _u32(3)  # version
        header += _i64(n_tensors)
        header += _i64(n_kv)
        header += kv_blob
        header += tensor_infos
        
        # Pad header
        pad = _align_up(len(header), ALIGNMENT) - len(header)
        if pad > 0:
            header += b"\x00" * pad

        # Write file
        with open(self.path, "wb") as f:
            f.write(header)
            f.write(data_blob)
            
    def _encode_kv(self, key: str, t: int, v) -> bytes:
        """Encode a key-value pair."""
        out = bytearray()
        out += _str_bytes(key)
        out += _i32(t)
        if t == KV_STRING:
            out += _str_bytes(str(v))
        elif t == KV_UINT32:
            out += _u32(int(v))
        elif t == KV_BOOL:
            out += struct.pack("<b", 1 if v else 0)
        else:
            raise ValueError(f"Unsupported KV type: {t}")
        return bytes(out)