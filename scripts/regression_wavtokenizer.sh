#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-$ROOT_DIR/wavtokenizer_v2.gguf}"
INPUT_WAV="${2:-}"

if [[ -z "$INPUT_WAV" ]]; then
    for candidate in \
        "$ROOT_DIR/out_wavtokenizer_mp3/ref_24k.wav" \
        "$ROOT_DIR/out_wavtokenizer/ref_24k.wav" \
        "$ROOT_DIR/input_audio/ref_24k.wav"; do
        if [[ -f "$candidate" ]]; then
            INPUT_WAV="$candidate"
            break
        fi
    done
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ -z "$INPUT_WAV" || ! -f "$INPUT_WAV" ]]; then
    echo "ERROR: input wav not found. Pass: scripts/regression_wavtokenizer.sh <model.gguf> <input.wav>" >&2
    exit 1
fi

INSPECT_BIN="$ROOT_DIR/build/inspect-codec"
CODEC_CLI_BIN="$ROOT_DIR/build/codec-cli"
if [[ ! -x "$INSPECT_BIN" || ! -x "$CODEC_CLI_BIN" ]]; then
    echo "ERROR: binaries missing; run: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

EXPECTED_FRAMES=659
EXPECTED_NQ=1
EXPECTED_TOKENS=659
EXPECTED_SR=24000
EXPECTED_CHANNELS=1
N_THREADS="${N_THREADS:-4}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

INSPECT_LOG="$TMP_DIR/inspect.log"
RUN_LOG="$TMP_DIR/run.log"
OUT_WAV="$TMP_DIR/recon.wav"

"$INSPECT_BIN" "$MODEL_PATH" > "$INSPECT_LOG"
"$CODEC_CLI_BIN" e2e --model "$MODEL_PATH" --in "$INPUT_WAV" --out "$OUT_WAV" --threads "$N_THREADS" > "$RUN_LOG"

encoded_line="$(grep -E '^encoded:' "$RUN_LOG" || true)"
decoded_line="$(grep -E '^decoded:' "$RUN_LOG" || true)"
if [[ -z "$encoded_line" || -z "$decoded_line" ]]; then
    echo "ERROR: missing encoded/decoded line in run output" >&2
    cat "$RUN_LOG" >&2
    exit 1
fi

n_frames="$(echo "$encoded_line" | sed -E 's/.*n_frames=([0-9]+).*/\1/')"
n_q="$(echo "$encoded_line" | sed -E 's/.*n_q=([0-9]+).*/\1/')"
n_tokens="$(echo "$encoded_line" | sed -E 's/.*n_tokens=([0-9]+).*/\1/')"

decoded_samples="$(echo "$decoded_line" | sed -E 's/.*n_samples=([0-9]+).*/\1/')"
decoded_sr="$(echo "$decoded_line" | sed -E 's/.*sr=([0-9]+).*/\1/')"

hop_size="$(grep -E '^[[:space:]]*hop_size[[:space:]]+[0-9]+' "$INSPECT_LOG" | awk '{print $2}' || true)"
if [[ -z "$hop_size" ]]; then
    echo "ERROR: failed to read hop_size from inspect output" >&2
    cat "$INSPECT_LOG" >&2
    exit 1
fi

if [[ "$n_frames" -ne "$EXPECTED_FRAMES" || "$n_q" -ne "$EXPECTED_NQ" || "$n_tokens" -ne "$EXPECTED_TOKENS" ]]; then
    echo "ERROR: token regression mismatch" >&2
    echo "  expected: n_frames=$EXPECTED_FRAMES n_q=$EXPECTED_NQ n_tokens=$EXPECTED_TOKENS" >&2
    echo "  actual:   n_frames=$n_frames n_q=$n_q n_tokens=$n_tokens" >&2
    exit 1
fi

if [[ "$decoded_sr" -ne "$EXPECTED_SR" ]]; then
    echo "ERROR: decoded sample_rate mismatch: expected $EXPECTED_SR, got $decoded_sr" >&2
    exit 1
fi

expected_samples=$(( n_frames * hop_size ))
sample_diff=$(( decoded_samples - expected_samples ))
abs_diff=$(( sample_diff < 0 ? -sample_diff : sample_diff ))
if [[ "$abs_diff" -gt "$hop_size" ]]; then
    echo "ERROR: decoded sample count mismatch (outside padding tolerance)" >&2
    echo "  n_frames=$n_frames hop_size=$hop_size expected=$expected_samples actual=$decoded_samples diff=$sample_diff" >&2
    exit 1
fi

if [[ ! -s "$OUT_WAV" ]]; then
    echo "ERROR: output wav missing or empty: $OUT_WAV" >&2
    exit 1
fi

le_u16() {
    od -An -t u2 -N 2 -j "$2" "$1" | tr -d ' \n'
}

le_u32() {
    od -An -t u4 -N 4 -j "$2" "$1" | tr -d ' \n'
}

riff="$(dd if="$OUT_WAV" bs=1 count=4 2>/dev/null)"
wave="$(dd if="$OUT_WAV" bs=1 skip=8 count=4 2>/dev/null)"
fmt_id="$(dd if="$OUT_WAV" bs=1 skip=12 count=4 2>/dev/null)"
if [[ "$riff" != "RIFF" || "$wave" != "WAVE" || "$fmt_id" != "fmt " ]]; then
    echo "ERROR: invalid WAV header (RIFF/WAVE/fmt)" >&2
    exit 1
fi

audio_format="$(le_u16 "$OUT_WAV" 20)"
header_channels="$(le_u16 "$OUT_WAV" 22)"
header_sr="$(le_u32 "$OUT_WAV" 24)"
bits_per_sample="$(le_u16 "$OUT_WAV" 34)"
if [[ "$audio_format" -ne 1 || "$header_channels" -ne "$EXPECTED_CHANNELS" || "$header_sr" -ne "$EXPECTED_SR" || "$bits_per_sample" -ne 16 ]]; then
    echo "ERROR: WAV format mismatch" >&2
    echo "  audio_format=$audio_format channels=$header_channels sample_rate=$header_sr bits=$bits_per_sample" >&2
    exit 1
fi

if command -v ffprobe >/dev/null 2>&1; then
    if ! ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$OUT_WAV" >/dev/null 2>&1; then
        echo "ERROR: ffprobe failed to read output wav" >&2
        exit 1
    fi
    ffprobe_status="ok"
else
    ffprobe_status="skipped (ffprobe not found)"
fi

echo "Regression summary:"
echo "  input:         $INPUT_WAV"
echo "  encoded:       n_frames=$n_frames n_q=$n_q n_tokens=$n_tokens"
echo "  decoded:       n_samples=$decoded_samples sample_rate=$decoded_sr channels=$header_channels"
echo "  expected:      n_samples~=n_frames*hop_size ($expected_samples, tolerance +/-$hop_size)"
echo "  wav header:    OK (PCM16 mono ${EXPECTED_SR}Hz)"
echo "  ffprobe:       $ffprobe_status"
echo "  note:          prefer unified CLI: build/codec-cli e2e ..."
echo "  output wav:    $OUT_WAV"
