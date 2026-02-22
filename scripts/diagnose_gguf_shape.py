#!/usr/bin/env python3
import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

GGUF_VALUE_TYPE_UINT8 = 0
GGUF_VALUE_TYPE_INT8 = 1
GGUF_VALUE_TYPE_UINT16 = 2
GGUF_VALUE_TYPE_INT16 = 3
GGUF_VALUE_TYPE_UINT32 = 4
GGUF_VALUE_TYPE_INT32 = 5
GGUF_VALUE_TYPE_FLOAT32 = 6
GGUF_VALUE_TYPE_BOOL = 7
GGUF_VALUE_TYPE_STRING = 8
GGUF_VALUE_TYPE_ARRAY = 9
GGUF_VALUE_TYPE_UINT64 = 10
GGUF_VALUE_TYPE_INT64 = 11
GGUF_VALUE_TYPE_FLOAT64 = 12

SCALAR_SIZES = {
    GGUF_VALUE_TYPE_UINT8: 1,
    GGUF_VALUE_TYPE_INT8: 1,
    GGUF_VALUE_TYPE_UINT16: 2,
    GGUF_VALUE_TYPE_INT16: 2,
    GGUF_VALUE_TYPE_UINT32: 4,
    GGUF_VALUE_TYPE_INT32: 4,
    GGUF_VALUE_TYPE_FLOAT32: 4,
    GGUF_VALUE_TYPE_BOOL: 1,
    GGUF_VALUE_TYPE_UINT64: 8,
    GGUF_VALUE_TYPE_INT64: 8,
    GGUF_VALUE_TYPE_FLOAT64: 8,
}


def _read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_i32(f) -> int:
    return struct.unpack("<i", f.read(4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_i64(f) -> int:
    return struct.unpack("<q", f.read(8))[0]


def _read_str(f) -> str:
    n = _read_u64(f)
    return f.read(n).decode("utf-8")


def _skip_gguf_value(f, vtype: int) -> None:
    if vtype == GGUF_VALUE_TYPE_STRING:
        _ = _read_str(f)
        return
    if vtype == GGUF_VALUE_TYPE_ARRAY:
        elem_type = _read_i32(f)
        n = _read_u64(f)
        if elem_type == GGUF_VALUE_TYPE_STRING:
            for _ in range(n):
                _ = _read_str(f)
            return
        if elem_type == GGUF_VALUE_TYPE_ARRAY:
            raise ValueError("nested GGUF arrays are unsupported")
        size = SCALAR_SIZES.get(elem_type)
        if size is None:
            raise ValueError(f"unsupported GGUF array element type: {elem_type}")
        f.read(size * n)
        return
    size = SCALAR_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"unsupported GGUF value type: {vtype}")
    f.read(size)


def read_gguf_tensor_dims(path: Path) -> dict[str, tuple[int, list[int]]]:
    out: dict[str, tuple[int, list[int]]] = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {path}")
        version = _read_u32(f)
        if version != 3:
            raise ValueError(f"unsupported GGUF version: {version}")
        n_tensors = _read_u64(f)
        n_kv = _read_u64(f)

        for _ in range(n_kv):
            _ = _read_str(f)
            kv_type = _read_i32(f)
            _skip_gguf_value(f, kv_type)

        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = _read_u32(f)
            dims = [_read_i64(f) for _ in range(n_dims)]
            _ = _read_i32(f)  # ggml_type
            _ = _read_u64(f)  # offset
            out[name] = (n_dims, dims)
    return out


def read_safetensors_shape(path: Path, keys: list[str]) -> tuple[str, tuple[int, ...]]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    for key in keys:
        if key in header:
            shape = header[key]["shape"]
            return key, tuple(int(x) for x in shape)
    raise KeyError(f"missing tensor in safetensors (tried): {keys}")


def choose_tensor_name(names: list[str]) -> str:
    preferred = (
        "q.s.layers.0.cb.embed",
        "q.s.layers.0.codebook.embed",
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed",
    )
    for p in preferred:
        if p in names:
            return p
    raise KeyError("semantic layer0 codebook.embed tensor not found in GGUF")


def run_cmd(cmd: list[str], cwd: Path, allow_fail: bool = False) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        out += f"\n[diagnose note] command exit code={proc.returncode}\n"
    return out


def parse_cpp_debug(stdout_stderr: str) -> tuple[str | None, str | None]:
    line = None
    for s in stdout_stderr.splitlines():
        if "DEBUG: mimi_sem_l0_t0_lookup" in s:
            line = s.strip()
            break
    txt = Path("/tmp/mimi_debug_sem_layer0_t0.txt")
    txt_content = txt.read_text(encoding="utf-8") if txt.is_file() else None
    return line, txt_content


def parse_shape_from_txt(content: str) -> tuple[int, int, str | None]:
    m_shape = re.search(r"^cb_shape=(\d+)x(\d+)$", content, flags=re.MULTILINE)
    m_layout = re.search(r"^layout=([^\n]+)$", content, flags=re.MULTILINE)
    if m_shape is None:
        raise ValueError(f"missing cb_shape in debug txt:\n{content}")
    ne0 = int(m_shape.group(1))
    ne1 = int(m_shape.group(2))
    layout = m_layout.group(1).strip() if m_layout is not None else None
    return ne0, ne1, layout


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Mimi GGUF codebook.embed shape handling")
    parser.add_argument("--model-dir", default="/home/node/.openclaw/workspace/checkpoints/mimi")
    parser.add_argument("--gguf", default="mimi.gguf")
    parser.add_argument("--root-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--codes", default="/tmp/mimi_codes.npy")
    parser.add_argument("--out-wav", default="/tmp/mimi_diagnose_shape.wav")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    gguf_path = Path(args.gguf).resolve()
    codes_path = Path(args.codes)
    mimi_bin = root_dir / "build/mimi-decode"
    inspect_py = root_dir / "scripts/inspect_mimi_hf.py"

    st_path = model_dir / "model.safetensors"
    hf_key, hf_shape = read_safetensors_shape(
        st_path,
        [
            "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed",
            "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum",
        ],
    )

    gguf_dims_map = read_gguf_tensor_dims(gguf_path)
    selected = choose_tensor_name(list(gguf_dims_map.keys()))
    n_dims, dims = gguf_dims_map[selected]
    shape_no_reverse = tuple(dims)
    shape_reversed = tuple(reversed(dims))

    if not codes_path.is_file():
        run_cmd(
            [
                sys.executable,
                str(inspect_py),
                "--model-id",
                str(model_dir),
                "--offline",
                "--input-audio",
                str(root_dir / "input_audio/reference_10_2.mp3"),
                "--save-codes",
                str(codes_path),
            ],
            root_dir,
        )

    if not mimi_bin.is_file():
        run_cmd(["cmake", "-S", str(root_dir), "-B", str(root_dir / "build")], root_dir)
        run_cmd(["cmake", "--build", str(root_dir / "build"), "-j"], root_dir)

    cpp_out = run_cmd([str(mimi_bin), str(gguf_path), str(codes_path), str(args.out_wav)], root_dir, allow_fail=True)
    cpp_lookup_line, cpp_txt = parse_cpp_debug(cpp_out)

    print("=== Mimi GGUF Shape Diagnose ===")
    print("tensor:")
    print(f"  hf_name            : {hf_key}")
    print(f"  gguf_name          : {selected}")
    print("shapes:")
    print(f"  hf_shape           : {hf_shape}")
    print(f"  gguf_header_n_dims : {n_dims}")
    print(f"  gguf_header_dims   : {tuple(dims)}")
    print(f"  gguf_shape_no_rev  : {shape_no_reverse}")
    print(f"  gguf_shape_rev     : {shape_reversed}")

    if cpp_lookup_line is None:
        print("cpp_lookup_line      : (not found in mimi-decode output)")
    else:
        print(f"cpp_lookup_line      : {cpp_lookup_line}")

    if cpp_txt is None:
        print("cpp_debug_txt        : /tmp/mimi_debug_sem_layer0_t0.txt not found")
    else:
        ne0, ne1, layout = parse_shape_from_txt(cpp_txt)
        print("ggml_interpretation:")
        print(f"  ne                 : ({ne0}, {ne1}, 1, 1)")
        print(f"  layout             : {layout}")
        print("cpp_debug_txt_raw:")
        print(cpp_txt.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
