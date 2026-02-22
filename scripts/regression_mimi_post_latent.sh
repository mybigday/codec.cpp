#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMI_DIR="${1:-/home/node/.openclaw/workspace/checkpoints/mimi}"
MIMI_GGUF="${2:-$ROOT_DIR/mimi.gguf}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

REGRESSION_MIMI_SH="$ROOT_DIR/scripts/regression_mimi.sh"
DUMP_HF_PY="$ROOT_DIR/scripts/dump_mimi_post_latent_hf.py"
COMPARE_PY="$ROOT_DIR/scripts/compare_mimi_post_latent.py"

if [[ ! -f "$REGRESSION_MIMI_SH" || ! -f "$DUMP_HF_PY" || ! -f "$COMPARE_PY" ]]; then
    echo "ERROR: required scripts are missing under $ROOT_DIR/scripts" >&2
    exit 1
fi
if [[ ! -f "$MIMI_DIR/model.safetensors" || ! -f "$MIMI_DIR/config.json" ]]; then
    echo "ERROR: Mimi checkpoint missing in: $MIMI_DIR" >&2
    exit 1
fi

rm -f \
    /tmp/mimi_dbg_z_before_upsample.bin /tmp/mimi_dbg_z_after_upsample.bin /tmp/mimi_dbg_z_after_transformer.bin /tmp/mimi_dbg_y_pre_tanh.bin \
    /tmp/mimi_dbg_z_before_upsample_hf.bin /tmp/mimi_dbg_z_after_upsample_hf.bin /tmp/mimi_dbg_z_after_transformer_hf.bin /tmp/mimi_dbg_y_pre_tanh_hf.bin \
    /tmp/mimi_dbg_post_latent_meta.txt /tmp/mimi_dbg_post_latent_hf_meta.txt

echo "[1/4] Build codec.cpp"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build"
cmake --build "$ROOT_DIR/build" -j

echo "[2/4] Run base Mimi regression (generates codes + C++ decode with dumps)"
"$REGRESSION_MIMI_SH" "$MIMI_DIR" "$MIMI_GGUF"

echo "[3/4] Dump HF post-latent checkpoints"
"$PYTHON_BIN" "$DUMP_HF_PY" \
    --model-dir "$MIMI_DIR" \
    --codes /tmp/mimi_codes.npy

echo "[4/4] Compare C++ vs HF post-latent checkpoints"
"$PYTHON_BIN" "$COMPARE_PY"
