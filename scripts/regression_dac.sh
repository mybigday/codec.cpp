#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-$ROOT_DIR/dac_speech_v1.gguf}"
WEIGHTS_PTH="${2:-$ROOT_DIR/checkpoints/weights_24khz_3kbps_v1.0.pth}"
INPUT_AUDIO="${3:-$ROOT_DIR/input_audio/reference_10_2.mp3}"

# Mode:
# - quick (default): only n_q=4 (strict check)
# - full: run multiple n_q cases (non-strict for n_q!=4)
DAC_MODE="${DAC_MODE:-quick}"
HF_MODEL_ID="${HF_MODEL_ID:-ibm-research/DAC.speech.v1.0}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model not found: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$WEIGHTS_PTH" ]]; then
    echo "ERROR: weights not found: $WEIGHTS_PTH" >&2
    exit 1
fi
if [[ ! -f "$INPUT_AUDIO" ]]; then
    echo "ERROR: input audio not found: $INPUT_AUDIO" >&2
    exit 1
fi

INSPECT_BIN="$ROOT_DIR/build/inspect-codec"
CODEC_CLI_BIN="$ROOT_DIR/build/codec-cli"
PY_INSPECT="$ROOT_DIR/scripts/inspect_dac_hf.py"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

if [[ ! -x "$INSPECT_BIN" || ! -x "$CODEC_CLI_BIN" ]]; then
    echo "ERROR: binaries missing; run: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi
if [[ ! -f "$PY_INSPECT" ]]; then
    echo "ERROR: missing script: $PY_INSPECT" >&2
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is required for input normalization" >&2
    exit 1
fi

EXPECTED_ARCH="DAC"
EXPECTED_SR=24000
EXPECTED_NQ=4
EXPECTED_CODEBOOK=1024
EXPECTED_LATENT=1024
MAX_MSE=0.0001
MIN_CORR=0.99

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

INSPECT_LOG="$TMP_DIR/inspect.log"
HF_LOG="$TMP_DIR/hf.log"
CPP_LOG="$TMP_DIR/cpp.log"
INPUT_WAV="$TMP_DIR/input_24k_mono.wav"
REF_WAV="$TMP_DIR/reference_dac_py_e2e.wav"

"$INSPECT_BIN" "$MODEL_PATH" > "$INSPECT_LOG"

arch="$(grep -E '^arch:' "$INSPECT_LOG" | awk '{print $2}' || true)"
sr="$(grep -E '^[[:space:]]*sample_rate[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
hop="$(grep -E '^[[:space:]]*hop_size[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
nq="$(grep -E '^[[:space:]]*n_q[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
codebook="$(grep -E '^[[:space:]]*codebook_size[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
latent="$(grep -E '^[[:space:]]*latent_dim[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"

if [[ "$arch" != "$EXPECTED_ARCH" ]]; then
    echo "ERROR: unexpected architecture: $arch" >&2
    exit 1
fi
if [[ "$sr" -ne "$EXPECTED_SR" || "$nq" -ne "$EXPECTED_NQ" || "$codebook" -ne "$EXPECTED_CODEBOOK" || "$latent" -ne "$EXPECTED_LATENT" ]]; then
    echo "ERROR: metadata mismatch" >&2
    echo "  got sample_rate=$sr hop_size=$hop n_q=$nq codebook_size=$codebook latent_dim=$latent" >&2
    exit 1
fi

ffmpeg -hide_banner -loglevel error -y -i "$INPUT_AUDIO" -ac 1 -ar "$EXPECTED_SR" "$INPUT_WAV"

PY_CMD=(
    "$PYTHON_BIN" "$PY_INSPECT"
    "--weights-pth" "$WEIGHTS_PTH"
    "--input-audio" "$INPUT_WAV"
    "--save-reference" "$REF_WAV"
)
if [[ -n "$HF_LOCAL_DIR" ]]; then
    PY_CMD+=("--local-dir" "$HF_LOCAL_DIR" "--offline")
else
    PY_CMD+=("--model-id" "$HF_MODEL_ID")
fi
"${PY_CMD[@]}" > "$HF_LOG"

compare_pair() {
    local ref_wav="$1"
    local out_wav="$2"
    local hop_size="$3"
    local max_mse="$4"
    local min_corr="$5"
    local out_json="$6"

    "$PYTHON_BIN" - "$ref_wav" "$out_wav" "$hop_size" "$max_mse" "$min_corr" "$out_json" <<'PY'
import json
import math
import sys
import wave
from array import array

ref_path, out_path, hop_size, max_mse, min_corr, out_json = sys.argv[1:]
hop_size = int(hop_size)
max_mse = float(max_mse)
min_corr = float(min_corr)

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        data = wf.readframes(n)
    if ch != 1 or sw != 2:
        raise RuntimeError(f"{path}: expected mono PCM16, got channels={ch} sample_width={sw}")
    arr = array("h")
    arr.frombytes(data)
    return sr, arr

sr_ref, a = read_wav(ref_path)
sr_out, b = read_wav(out_path)
if sr_ref != sr_out:
    raise RuntimeError(f"sample_rate mismatch: ref={sr_ref}, out={sr_out}")

n_ref = len(a)
n_out = len(b)
n_min = min(n_ref, n_out)
if n_min <= 0:
    raise RuntimeError("empty waveform")

len_diff = abs(n_ref - n_out)

sum_err = 0.0
sum_aa = 0.0
sum_bb = 0.0
sum_ab = 0.0
for i in range(n_min):
    x = a[i] / 32768.0
    y = b[i] / 32768.0
    d = x - y
    sum_err += d * d
    sum_aa += x * x
    sum_bb += y * y
    sum_ab += x * y

mse = sum_err / n_min
den = math.sqrt(sum_aa * sum_bb)
corr = (sum_ab / den) if den > 0 else 0.0

ok = True
checks = []
if len_diff > max(hop_size, 1):
    ok = False
    checks.append(f"length diff {len_diff} > hop_size {hop_size}")
if mse > max_mse:
    ok = False
    checks.append(f"mse {mse:.8f} > {max_mse:.8f}")
if corr < min_corr:
    ok = False
    checks.append(f"corr {corr:.6f} < {min_corr:.6f}")

report = {
    "sample_rate": sr_ref,
    "n_samples_ref": n_ref,
    "n_samples_cpp": n_out,
    "length_diff": len_diff,
    "hop_size": hop_size,
    "mse": mse,
    "corr": corr,
    "ok": ok,
    "checks": checks,
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f)

if not ok:
    raise SystemExit("; ".join(checks))
PY
}

print_metrics() {
    local compare_json="$1"
    local label="$2"
    readarray -t compare_fields < <("$PYTHON_BIN" - "$compare_json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d["sample_rate"])
print(d["n_samples_ref"])
print(d["n_samples_cpp"])
print(d["length_diff"])
print(d["mse"])
print(d["corr"])
PY
    )

    local cmp_sr="${compare_fields[0]}"
    local cmp_n_ref="${compare_fields[1]}"
    local cmp_n_cpp="${compare_fields[2]}"
    local cmp_len_diff="${compare_fields[3]}"
    local cmp_mse="${compare_fields[4]}"
    local cmp_corr="${compare_fields[5]}"
    echo "  $label: sr=$cmp_sr n_ref=$cmp_n_ref n_cpp=$cmp_n_cpp len_diff=$cmp_len_diff mse=$cmp_mse corr=$cmp_corr"
}

if [[ ! -s "$REF_WAV" ]]; then
    echo "ERROR: missing reference wav: $REF_WAV" >&2
    exit 1
fi

declare -a DAC_CASES=(4)
if [[ "$DAC_MODE" == "full" ]]; then
    DAC_CASES=(1 2 4)
fi
for case_nq in "${DAC_CASES[@]}"; do
    out_wav="$TMP_DIR/recon_dac_cpp_nq${case_nq}.wav"
    compare_json="$TMP_DIR/compare_nq${case_nq}.json"
    "$CODEC_CLI_BIN" e2e --model "$MODEL_PATH" --in "$INPUT_WAV" --out "$out_wav" --nq "$case_nq" > "$CPP_LOG.nq${case_nq}"
    if [[ ! -s "$out_wav" ]]; then
        echo "ERROR: missing regression output for n_q=$case_nq" >&2
        exit 1
    fi
    if [[ "$case_nq" -eq 4 ]]; then
        compare_pair "$REF_WAV" "$out_wav" "$hop" "$MAX_MSE" "$MIN_CORR" "$compare_json"
    else
        compare_pair "$REF_WAV" "$out_wav" "$hop" "1000000" "-1" "$compare_json"
    fi
done

echo "Regression summary:"
echo "  model:         $MODEL_PATH"
echo "  weights:       $WEIGHTS_PTH"
echo "  input_audio:   $INPUT_AUDIO"
echo "  input_wav:     $INPUT_WAV"
if [[ -n "$HF_LOCAL_DIR" ]]; then
    echo "  hf source:     local-dir=$HF_LOCAL_DIR (offline)"
else
    echo "  hf source:     model-id=$HF_MODEL_ID"
fi
echo "  metadata:      sample_rate=$sr hop_size=$hop n_q=$nq codebook_size=$codebook latent_dim=$latent"
echo "  note:          prefer unified CLI: build/codec-cli e2e ..."
echo "  reference wav: $REF_WAV"
for case_nq in "${DAC_CASES[@]}"; do
    print_metrics "$TMP_DIR/compare_nq${case_nq}.json" "n_q=$case_nq"
done
echo "  tolerance:     strict only for n_q=4 -> len_diff<=hop_size mse<=$MAX_MSE corr>=$MIN_CORR"
