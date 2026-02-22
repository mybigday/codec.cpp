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

REGRESSION_POST_LATENT_SH="$ROOT_DIR/scripts/regression_mimi_post_latent.sh"
COMPARE_PY="$ROOT_DIR/scripts/compare_mimi_post_latent.py"

if [[ ! -x "$REGRESSION_POST_LATENT_SH" || ! -f "$COMPARE_PY" ]]; then
    echo "ERROR: required scripts are missing under $ROOT_DIR/scripts" >&2
    exit 1
fi

LOG_FILE="$(mktemp /tmp/mimi_upsample_vs_hf.XXXXXX.log)"
trap 'rm -f "$LOG_FILE"' EXIT

"$REGRESSION_POST_LATENT_SH" "$MIMI_DIR" "$MIMI_GGUF" | tee "$LOG_FILE"

echo
echo "[verify] checking z_after_upsample + FIRST_DIVERGENCE"
UPSAMPLE_ALLCLOSE="$("$PYTHON_BIN" - "$LOG_FILE" <<'PY'
import sys

log_path = sys.argv[1]
upsample_allclose = None
first_div = None

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("z_after_upsample"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 7:
                upsample_allclose = parts[6]
        if line.startswith("FIRST_DIVERGENCE:"):
            first_div = line.split(":", 1)[1].strip()

if upsample_allclose is None:
    print("ERROR: could not find z_after_upsample row", file=sys.stderr)
    raise SystemExit(2)
if first_div is None:
    print("ERROR: could not find FIRST_DIVERGENCE line", file=sys.stderr)
    raise SystemExit(2)

print(f"{upsample_allclose}|{first_div}")
PY
)"

ALLCLOSE_VAL="${UPSAMPLE_ALLCLOSE%%|*}"
FIRST_DIV_VAL="${UPSAMPLE_ALLCLOSE#*|}"

echo "z_after_upsample allclose: $ALLCLOSE_VAL"
echo "FIRST_DIVERGENCE: $FIRST_DIV_VAL"

if [[ "$ALLCLOSE_VAL" != "True" ]]; then
    echo "ERROR: z_after_upsample does not match HF allclose criteria" >&2
    exit 1
fi

if [[ "$FIRST_DIV_VAL" == "z_after_upsample" ]]; then
    echo "ERROR: FIRST_DIVERGENCE is still z_after_upsample" >&2
    exit 1
fi

echo "PASS: Mimi upsample matches HF and first divergence moved past upsample."
