#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMI_DIR="${1:-/home/node/.openclaw/workspace/checkpoints/mimi}"
OUT_GGUF="${2:-$ROOT_DIR/mimi.gguf}"
CODES_NPY="${3:-/tmp/mimi_codes.npy}"
REF_WAV="${4:-/tmp/mimi_ref.wav}"
OUT_WAV="${5:-$ROOT_DIR/recon_mimi_cpp.wav}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

INSPECT_PY="$ROOT_DIR/scripts/inspect_mimi_hf.py"
CONVERT_PY="$ROOT_DIR/scripts/convert-to-gguf.py"
INSPECT_BIN="$ROOT_DIR/build/inspect-codec"
CODEC_CLI_BIN="$ROOT_DIR/build/codec-cli"

if [[ ! -f "$MIMI_DIR/model.safetensors" || ! -f "$MIMI_DIR/config.json" ]]; then
    echo "ERROR: Mimi checkpoint missing in: $MIMI_DIR" >&2
    exit 1
fi
if [[ ! -f "$INSPECT_PY" || ! -f "$CONVERT_PY" ]]; then
    echo "ERROR: missing Mimi scripts in $ROOT_DIR/scripts" >&2
    exit 1
fi
if [[ ! -x "$INSPECT_BIN" || ! -x "$CODEC_CLI_BIN" ]]; then
    echo "ERROR: binaries missing; run: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
INSPECT_LOG="$TMP_DIR/inspect.log"

if [[ ! -f "$CODES_NPY" || ! -f "$REF_WAV" ]]; then
    echo "Generating Python reference (codes + wav)..."
    "$PYTHON_BIN" "$INSPECT_PY" \
        --model-id "$MIMI_DIR" \
        --offline \
        --input-audio "$ROOT_DIR/input_audio/reference_10_2.mp3" \
        --save-codes "$CODES_NPY" \
        --save-reference "$REF_WAV"
fi

if [[ ! -f "$CODES_NPY" || ! -f "$REF_WAV" ]]; then
    echo "ERROR: missing reference artifacts: $CODES_NPY / $REF_WAV" >&2
    exit 1
fi

echo "Converting Mimi safetensors -> GGUF..."
"$PYTHON_BIN" "$CONVERT_PY" --input-dir "$MIMI_DIR" --output "$OUT_GGUF" --model-type mimi

"$INSPECT_BIN" "$OUT_GGUF" > "$INSPECT_LOG"
arch="$(grep -E '^arch:' "$INSPECT_LOG" | awk '{print $2}' || true)"
sr="$(grep -E '^[[:space:]]*sample_rate[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
hop="$(grep -E '^[[:space:]]*hop_size[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
nq="$(grep -E '^[[:space:]]*n_q[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
codebook="$(grep -E '^[[:space:]]*codebook_size[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"

if [[ "$arch" != "Mimi" ]]; then
    echo "ERROR: inspect arch mismatch: $arch" >&2
    exit 1
fi
if [[ "$sr" -ne 24000 || "$hop" -ne 1920 || "$nq" -ne 32 || "$codebook" -ne 2048 ]]; then
    echo "ERROR: Mimi GGUF metadata mismatch" >&2
    echo "  got sample_rate=$sr hop=$hop n_q=$nq codebook=$codebook" >&2
    exit 1
fi

echo "Running C++ Mimi decode..."
compare_pair() {
    local ref_wav="$1"
    local out_wav="$2"
    local label="$3"
    "$PYTHON_BIN" - "$ref_wav" "$out_wav" "$label" <<'PY'
import math
import sys
import wave
from array import array

ref_path, out_path, label = sys.argv[1:]

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        data = wf.readframes(n)
    if ch != 1 or sw != 2:
        raise RuntimeError(f"{path}: expected mono PCM16")
    arr = array("h")
    arr.frombytes(data)
    return sr, arr

sr_ref, a = read_wav(ref_path)
sr_out, b = read_wav(out_path)
n_ref = len(a)
n_out = len(b)
n = min(n_ref, n_out)
if n <= 0:
    raise RuntimeError("empty output")
sum_err = 0.0
sum_aa = 0.0
sum_bb = 0.0
sum_ab = 0.0
for i in range(n):
    x = a[i] / 32768.0
    y = b[i] / 32768.0
    d = x - y
    sum_err += d * d
    sum_aa += x * x
    sum_bb += y * y
    sum_ab += x * y
mse = sum_err / n
den = math.sqrt(max(sum_aa * sum_bb, 1e-20))
corr = sum_ab / den
print(f"  {label}: sr_ref={sr_ref} sr_cpp={sr_out} n_ref={n_ref} n_cpp={n_out} mse={mse:.8f} corr={corr:.6f}")
PY
}

declare -a MIMI_CASES=(32 8)
for case_nq in "${MIMI_CASES[@]}"; do
    out_case="$TMP_DIR/recon_mimi_cpp_nq${case_nq}.wav"
    "$CODEC_CLI_BIN" decode --model "$OUT_GGUF" --codes "$CODES_NPY" --out "$out_case" --nq "$case_nq"
    if [[ ! -s "$out_case" ]]; then
        echo "ERROR: output wav missing for n_q=$case_nq: $out_case" >&2
        exit 1
    fi
done

cp "$TMP_DIR/recon_mimi_cpp_nq32.wav" "$OUT_WAV"

echo "Regression summary:"
echo "  mimi_dir:      $MIMI_DIR"
echo "  gguf:          $OUT_GGUF"
echo "  codes:         $CODES_NPY"
echo "  reference:     $REF_WAV"
echo "  output:        $OUT_WAV"
echo "  metadata:      sample_rate=$sr hop_size=$hop n_q=$nq codebook_size=$codebook"
echo "  note:          prefer unified CLI: build/codec-cli decode ..."
for case_nq in "${MIMI_CASES[@]}"; do
    compare_pair "$REF_WAV" "$TMP_DIR/recon_mimi_cpp_nq${case_nq}.wav" "n_q=$case_nq"
done
