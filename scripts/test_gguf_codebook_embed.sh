#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_gguf_vs_hf_codebook_embed.py" "$@" && echo "GGUF embed matches HF embed" || { rc=$?; echo "GGUF embed does NOT match HF embed"; exit "$rc"; }
