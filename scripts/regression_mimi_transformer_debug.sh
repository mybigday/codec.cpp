#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Build codec.cpp"
cmake --build build -j

echo "[2/5] Run base C++ regression (with transformer dumps)"
# Respect user-provided CODEC_MIMI_DEBUG_TRANSFORMER (e.g. "all"); default to layer0 dumps.
export CODEC_MIMI_DEBUG_TRANSFORMER="${CODEC_MIMI_DEBUG_TRANSFORMER:-1}"
./build/mimi-decode mimi.gguf /tmp/mimi_codes.npy /tmp/mimi_regression_transformer_debug.wav 2>&1 | head -50

echo "[3/5] Run HF transformer dumps"
./.audio-env/bin/python scripts/dump_mimi_transformer_hf.py

echo "[4/5] Compare C++ vs HF"
./.audio-env/bin/python scripts/compare_mimi_transformer.py

echo "[5/5] Done"
