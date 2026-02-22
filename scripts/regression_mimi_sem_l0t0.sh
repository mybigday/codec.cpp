#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMI_DIR="${1:-/home/node/.openclaw/workspace/checkpoints/mimi}"
MIMI_GGUF="${2:-$ROOT_DIR/mimi.gguf}"
CODES_NPY="/tmp/mimi_codes.npy"
OUT_WAV="/tmp/mimi_regression_sem_l0t0.wav"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

INSPECT_PY="$ROOT_DIR/scripts/inspect_mimi_hf.py"
HF_DEBUG_PY="$ROOT_DIR/scripts/debug_mimi_sem_layer0_hf.py"
COMPARE_PY="$ROOT_DIR/scripts/compare_mimi_sem_l0t0.py"
MIMI_BIN="$ROOT_DIR/build/mimi-decode"

if [[ ! -f "$INSPECT_PY" || ! -f "$HF_DEBUG_PY" || ! -f "$COMPARE_PY" ]]; then
    echo "ERROR: required scripts are missing under $ROOT_DIR/scripts" >&2
    exit 1
fi
if [[ ! -f "$MIMI_DIR/model.safetensors" || ! -f "$MIMI_DIR/config.json" ]]; then
    echo "ERROR: Mimi checkpoint missing in: $MIMI_DIR" >&2
    exit 1
fi

rm -f \
    "$CODES_NPY" \
    /tmp/mimi_debug_sem_layer0_t0.txt /tmp/mimi_debug_sem_layer0_t0.bin \
    /tmp/mimi_debug_sem_layer0_t0_hf.txt /tmp/mimi_debug_sem_layer0_t0_hf.bin

echo "[1/6] Generate /tmp/mimi_codes.npy via HF Mimi encode"
"$PYTHON_BIN" "$INSPECT_PY" \
    --model-id "$MIMI_DIR" \
    --offline \
    --input-audio "$ROOT_DIR/input_audio/reference_10_2.mp3" \
    --save-codes "$CODES_NPY"

echo "[2/6] Build codec.cpp"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build"
cmake --build "$ROOT_DIR/build" -j

echo "[3/6] Run C++ Mimi decode (dumps /tmp/mimi_debug_sem_layer0_t0.*)"
"$MIMI_BIN" "$MIMI_GGUF" "$CODES_NPY" "$OUT_WAV"

echo "[4/6] Run HF semantic layer0/t0 debug dump"
"$PYTHON_BIN" "$HF_DEBUG_PY" \
    --model-dir "$MIMI_DIR" \
    --codes "$CODES_NPY"

echo "[5/6] Compare C++ vs HF single lookup"
"$PYTHON_BIN" "$COMPARE_PY"

echo "[6/6] Done"
