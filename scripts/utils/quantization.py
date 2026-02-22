"""
Quantization utilities for GGUF conversion.

Implements GGML-compatible quantization schemes:
- Q8_0: 8-bit per-block quantization (block_size=32)
- Q4_K_M: 4-bit K-quants (block_size=256, with scales/mins)
- Q5_K_M: 5-bit K-quants (block_size=256, with scales/mins)
"""

import numpy as np
from typing import Tuple

# Block sizes
QK8_0 = 32
QK_K = 256
K_SCALE_SIZE = 12


def _pack_scale_min_k4(ls: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Pack 8 scales and 8 mins into 12 bytes (GGML K-quants format)."""
    out = np.zeros((K_SCALE_SIZE,), dtype=np.uint8)
    for j in range(8):
        lsj = int(ls[j]) & 63
        lmj = int(lm[j]) & 63
        if j < 4:
            out[j] = lsj
            out[j + 4] = lmj
        else:
            out[j + 4] = (lsj & 0x0F) | ((lmj & 0x0F) << 4)
            out[j - 4] |= ((lsj >> 4) << 6)
            out[j - 0] |= ((lmj >> 4) << 6)
    return out


def _quantize_row_q8_0(row: np.ndarray) -> np.ndarray:
    """Quantize a row to Q8_0 format."""
    if row.size % QK8_0 != 0:
        raise ValueError(f"Q8_0 row length must be divisible by {QK8_0}, got {row.size}")

    blocks = row.reshape(-1, QK8_0)
    out = bytearray()
    for b in blocks:
        amax = float(np.max(np.abs(b)))
        d = amax / 127.0 if amax > 0 else 0.0
        inv_d = (1.0 / d) if d != 0.0 else 0.0
        q = np.rint(b * inv_d).astype(np.int8)
        out += np.float16(d).tobytes()
        out += q.tobytes()
    return np.frombuffer(bytes(out), dtype=np.uint8)


def _quantize_row_q4_k(row: np.ndarray) -> np.ndarray:
    """Quantize a row to Q4_K format (used by Q4_K_M)."""
    if row.size % QK_K != 0:
        raise ValueError(f"Q4_K row length must be divisible by {QK_K}, got {row.size}")

    out = bytearray()
    for sb in row.reshape(-1, QK_K):
        sub = sb.reshape(8, 32)
        mins = np.zeros((8,), dtype=np.float32)
        scales = np.zeros((8,), dtype=np.float32)
        q = np.zeros((8, 32), dtype=np.uint8)

        for i in range(8):
            x = sub[i]
            xmin = float(np.min(x))
            xmax = float(np.max(x))
            scale = (xmax - xmin) / 15.0 if xmax > xmin else 0.0
            mins[i] = -xmin
            scales[i] = scale
            if scale > 0:
                q[i] = np.clip(np.rint((x - xmin) / scale), 0, 15).astype(np.uint8)

        max_scale = float(np.max(scales))
        max_min = float(np.max(mins))
        d = max_scale / 63.0 if max_scale > 0 else 0.0
        dmin = max_min / 63.0 if max_min > 0 else 0.0
        ls = np.clip(np.rint(scales / d), 0, 63).astype(np.uint8) if d > 0 else np.zeros((8,), dtype=np.uint8)
        lm = np.clip(np.rint(mins / dmin), 0, 63).astype(np.uint8) if dmin > 0 else np.zeros((8,), dtype=np.uint8)
        scale_bytes = _pack_scale_min_k4(ls, lm)

        qs = np.zeros((QK_K // 2,), dtype=np.uint8)
        l_all = q.reshape(QK_K)
        qoff = 0
        for j in range(0, QK_K, 64):
            lo = l_all[j:j + 32]
            hi = l_all[j + 32:j + 64]
            qs[qoff:qoff + 32] = lo | (hi << 4)
            qoff += 32

        out += np.float16(d).tobytes()
        out += np.float16(dmin).tobytes()
        out += scale_bytes.tobytes()
        out += qs.tobytes()

    return np.frombuffer(bytes(out), dtype=np.uint8)


def _quantize_row_q5_k(row: np.ndarray) -> np.ndarray:
    """Quantize a row to Q5_K format (used by Q5_K_M)."""
    if row.size % QK_K != 0:
        raise ValueError(f"Q5_K row length must be divisible by {QK_K}, got {row.size}")

    out = bytearray()
    for sb in row.reshape(-1, QK_K):
        sub = sb.reshape(8, 32)
        mins = np.zeros((8,), dtype=np.float32)
        scales = np.zeros((8,), dtype=np.float32)
        q = np.zeros((8, 32), dtype=np.uint8)

        for i in range(8):
            x = sub[i]
            xmin = float(np.min(x))
            xmax = float(np.max(x))
            scale = (xmax - xmin) / 31.0 if xmax > xmin else 0.0
            mins[i] = -xmin
            scales[i] = scale
            if scale > 0:
                q[i] = np.clip(np.rint((x - xmin) / scale), 0, 31).astype(np.uint8)

        max_scale = float(np.max(scales))
        max_min = float(np.max(mins))
        d = max_scale / 63.0 if max_scale > 0 else 0.0
        dmin = max_min / 63.0 if max_min > 0 else 0.0
        ls = np.clip(np.rint(scales / d), 0, 63).astype(np.uint8) if d > 0 else np.zeros((8,), dtype=np.uint8)
        lm = np.clip(np.rint(mins / dmin), 0, 63).astype(np.uint8) if dmin > 0 else np.zeros((8,), dtype=np.uint8)
        scale_bytes = _pack_scale_min_k4(ls, lm)

        qh = np.zeros((QK_K // 8,), dtype=np.uint8)
        ql = np.zeros((QK_K // 2,), dtype=np.uint8)
        l_all = q.reshape(QK_K)
        qoff = 0
        m1 = 1
        m2 = 2
        for n in range(0, QK_K, 64):
            for j in range(32):
                l1 = int(l_all[n + j])
                l2 = int(l_all[n + j + 32])
                if l1 > 15:
                    l1 -= 16
                    qh[j] |= m1
                if l2 > 15:
                    l2 -= 16
                    qh[j] |= m2
                ql[qoff + j] = (l1 & 0x0F) | ((l2 & 0x0F) << 4)
            qoff += 32
            m1 <<= 2
            m2 <<= 2

        out += np.float16(d).tobytes()
        out += np.float16(dmin).tobytes()
        out += scale_bytes.tobytes()
        out += qh.tobytes()
        out += ql.tobytes()

    return np.frombuffer(bytes(out), dtype=np.uint8)


def quantize_tensor_q8_0(arr: np.ndarray) -> bytes:
    """Quantize a tensor to Q8_0 format."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim < 1:
        raise ValueError(f"Q8_0 tensor rank must be >= 1, got {arr.shape}")
    if arr.shape[0] % QK8_0 != 0:
        raise ValueError(f"Q8_0 tensor ne0 must be divisible by {QK8_0}, got {arr.shape[0]}")

    rows = arr.reshape(-1, arr.shape[0])
    out = bytearray()
    for row in rows:
        out += _quantize_row_q8_0(row).tobytes()
    return bytes(out)


def quantize_tensor_q4_k_m(arr: np.ndarray) -> bytes:
    """Quantize a tensor to Q4_K_M format."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim < 1:
        raise ValueError(f"Q4_K_M tensor rank must be >= 1, got {arr.shape}")
    if arr.shape[0] % QK_K != 0:
        raise ValueError(f"Q4_K_M tensor ne0 must be divisible by {QK_K}, got {arr.shape[0]}")

    rows = arr.reshape(-1, arr.shape[0])
    out = bytearray()
    for row in rows:
        out += _quantize_row_q4_k(row).tobytes()
    return bytes(out)


def quantize_tensor_q5_k_m(arr: np.ndarray) -> bytes:
    """Quantize a tensor to Q5_K_M format."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim < 1:
        raise ValueError(f"Q5_K_M tensor rank must be >= 1, got {arr.shape}")
    if arr.shape[0] % QK_K != 0:
        raise ValueError(f"Q5_K_M tensor ne0 must be divisible by {QK_K}, got {arr.shape[0]}")

    rows = arr.reshape(-1, arr.shape[0])
    out = bytearray()
    for row in rows:
        out += _quantize_row_q5_k(row).tobytes()
    return bytes(out)