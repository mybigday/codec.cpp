"""Minimal ONNX reader (initializers + graph nodes) without the `onnx` package.

Parses the protobuf wire format of ModelProto directly, so converters can read
ONNX-only checkpoints (CosyVoice's campplus / speech tokenizers) in environments
whose protobuf is too old for `import onnx`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

# TensorProto.DataType → numpy dtype
_DTYPES = {
    1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32, 7: np.int64,
    9: np.bool_, 10: np.float16, 11: np.float64, 12: np.uint32, 13: np.uint64,
}


def _varint(buf: memoryview, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _fields(buf: memoryview) -> Iterator[Tuple[int, int, object]]:
    """Yield (field_number, wire_type, value); length-delimited values are memoryviews."""
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _varint(buf, pos)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = _varint(buf, pos)
        elif wt == 1:
            v = buf[pos:pos + 8]
            pos += 8
        elif wt == 2:
            ln, pos = _varint(buf, pos)
            v = buf[pos:pos + ln]
            pos += ln
        elif wt == 5:
            v = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fno, wt, v


def _packed_varints(v: object, wt: int) -> List[int]:
    if wt == 0:
        return [int(v)]
    out = []
    pos = 0
    mv = v  # type: ignore[assignment]
    while pos < len(mv):
        x, pos = _varint(mv, pos)
        out.append(x)
    return out


def _tensor(buf: memoryview) -> Tuple[str, np.ndarray]:
    dims: List[int] = []
    dtype = 1
    name = ""
    raw = None
    floats: List[float] = []
    ints: List[int] = []
    for fno, wt, v in _fields(buf):
        if fno == 1:
            dims += _packed_varints(v, wt)
        elif fno == 2:
            dtype = int(v)
        elif fno == 8:
            name = bytes(v).decode("utf-8")
        elif fno == 9:
            raw = bytes(v)
        elif fno == 4:
            floats += list(np.frombuffer(bytes(v), dtype=np.float32)) if wt == 2 else \
                [float(np.frombuffer(bytes(v), dtype=np.float32)[0])]
        elif fno in (5, 7):
            ints += _packed_varints(v, wt)
    np_dtype = _DTYPES[dtype]
    if raw is not None:
        arr = np.frombuffer(raw, dtype=np_dtype).copy()
    elif floats:
        arr = np.asarray(floats, dtype=np_dtype)
    else:
        arr = np.asarray(ints, dtype=np.int64).astype(np_dtype)
    return name, arr.reshape(dims) if dims else arr.reshape(())


@dataclass
class OnnxNode:
    op_type: str
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


def _node(buf: memoryview) -> OnnxNode:
    n = OnnxNode("", "")
    for fno, _, v in _fields(buf):
        if fno == 1:
            n.inputs.append(bytes(v).decode("utf-8"))
        elif fno == 2:
            n.outputs.append(bytes(v).decode("utf-8"))
        elif fno == 3:
            n.name = bytes(v).decode("utf-8")
        elif fno == 4:
            n.op_type = bytes(v).decode("utf-8")
    return n


def read_onnx(path: str | Path) -> Tuple[Dict[str, np.ndarray], List[OnnxNode]]:
    """Return ({initializer name: array}, [nodes]) of an ONNX model (no external data)."""
    data = memoryview(Path(path).read_bytes())
    inits: Dict[str, np.ndarray] = {}
    nodes: List[OnnxNode] = []
    for fno, _, v in _fields(data):
        if fno != 7:  # ModelProto.graph
            continue
        for gno, _, gv in _fields(v):  # type: ignore[arg-type]
            if gno == 5:      # GraphProto.initializer
                name, arr = _tensor(gv)  # type: ignore[arg-type]
                inits[name] = arr
            elif gno == 1:    # GraphProto.node
                nodes.append(_node(gv))  # type: ignore[arg-type]
    return inits, nodes
