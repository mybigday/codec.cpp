#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMI_DIR="${1:-/home/node/.openclaw/workspace/checkpoints/mimi}"
MIMI_GGUF="${2:-$ROOT_DIR/mimi.gguf}"
CODES_NPY="/tmp/mimi_codes.npy"
OUT_WAV="/tmp/mimi_regression_rvq_debug.wav"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

INSPECT_PY="$ROOT_DIR/scripts/inspect_mimi_hf.py"
DUMP_HF_PY="$ROOT_DIR/scripts/dump_mimi_rvq_debug_hf.py"
COMPARE_PY="$ROOT_DIR/scripts/compare_mimi_rvq_debug.py"
MIMI_BIN="$ROOT_DIR/build/mimi-decode"

if [[ ! -f "$INSPECT_PY" || ! -f "$DUMP_HF_PY" || ! -f "$COMPARE_PY" ]]; then
    echo "ERROR: required scripts are missing under $ROOT_DIR/scripts" >&2
    exit 1
fi
if [[ ! -f "$MIMI_DIR/model.safetensors" || ! -f "$MIMI_DIR/config.json" ]]; then
    echo "ERROR: Mimi checkpoint missing in: $MIMI_DIR" >&2
    exit 1
fi

rm -f \
    "$CODES_NPY" \
    /tmp/mimi_debug_sem_decoded.bin /tmp/mimi_debug_acu_decoded.bin /tmp/mimi_debug_sem_acu_sum.bin /tmp/mimi_debug_latent_final.bin \
    /tmp/mimi_debug_sem_decoded_hf.bin /tmp/mimi_debug_acu_decoded_hf.bin /tmp/mimi_debug_sem_acu_sum_hf.bin /tmp/mimi_debug_latent_final_hf.bin

echo "[1/6] Generate /tmp/mimi_codes.npy via HF Mimi encode"
"$PYTHON_BIN" "$INSPECT_PY" \
    --model-id "$MIMI_DIR" \
    --offline \
    --input-audio "$ROOT_DIR/input_audio/reference_10_2.mp3" \
    --save-codes "$CODES_NPY"

echo "[2/6] Build codec.cpp"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build"
cmake --build "$ROOT_DIR/build" -j

echo "[3/6] Run C++ Mimi decode (dumps RVQ checkpoints under /tmp)"
"$MIMI_BIN" "$MIMI_GGUF" "$CODES_NPY" "$OUT_WAV"

echo "[4/6] Run HF RVQ debug dumps"
"$PYTHON_BIN" "$DUMP_HF_PY" \
    --model-dir "$MIMI_DIR" \
    --codes "$CODES_NPY"

echo "[5/6] Compare 4 RVQ checkpoints"
"$PYTHON_BIN" "$COMPARE_PY"

echo "[6/6] Done"
