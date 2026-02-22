#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMI_DIR="${1:-/home/node/.openclaw/workspace/checkpoints/mimi}"
MIMI_GGUF="${2:-$ROOT_DIR/mimi.gguf}"
CODES_NPY="/tmp/mimi_codes.npy"
CPP_LATENT="/tmp/mimi_latent_cpp.bin"
HF_LATENT="/tmp/mimi_latent_hf.bin"
OUT_WAV="/tmp/mimi_regression_latent.wav"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

INSPECT_PY="$ROOT_DIR/scripts/inspect_mimi_hf.py"
DUMP_HF_PY="$ROOT_DIR/scripts/dump_mimi_latent_hf.py"
COMPARE_PY="$ROOT_DIR/scripts/compare_mimi_latent.py"
MIMI_BIN="$ROOT_DIR/build/mimi-decode"

if [[ ! -f "$INSPECT_PY" || ! -f "$DUMP_HF_PY" || ! -f "$COMPARE_PY" ]]; then
    echo "ERROR: required scripts are missing under $ROOT_DIR/scripts" >&2
    exit 1
fi
if [[ ! -f "$MIMI_DIR/model.safetensors" || ! -f "$MIMI_DIR/config.json" ]]; then
    echo "ERROR: Mimi checkpoint missing in: $MIMI_DIR" >&2
    exit 1
fi

rm -f "$CODES_NPY" "$CPP_LATENT" "$HF_LATENT"

echo "[1/5] Generate /tmp/mimi_codes.npy via HF Mimi encode"
"$PYTHON_BIN" "$INSPECT_PY" \
    --model-id "$MIMI_DIR" \
    --offline \
    --input-audio "$ROOT_DIR/input_audio/reference_10_2.mp3" \
    --save-codes "$CODES_NPY"

echo "[2/5] Build codec.cpp"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build"
cmake --build "$ROOT_DIR/build" -j

echo "[3/5] Run C++ Mimi decode (dumps latent to $CPP_LATENT)"
"$MIMI_BIN" "$MIMI_GGUF" "$CODES_NPY" "$OUT_WAV"

echo "[4/5] Dump HF RVQ latent to $HF_LATENT"
"$PYTHON_BIN" "$DUMP_HF_PY" \
    --model-dir "$MIMI_DIR" \
    --codes "$CODES_NPY" \
    --out "$HF_LATENT"

echo "[5/5] Compare latents"
if "$PYTHON_BIN" "$COMPARE_PY" --cpp "$CPP_LATENT" --hf "$HF_LATENT"; then
    echo "Latent test: PASS"
else
    echo "Latent test: FAIL"
    exit 1
fi
