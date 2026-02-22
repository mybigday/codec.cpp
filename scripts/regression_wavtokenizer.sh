#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-$ROOT_DIR/wavtokenizer_v2.gguf}"
INPUT_AUDIO="${2:-$ROOT_DIR/input_audio/reference_10_2.mp3}"
WT_CKPT="${3:-${WT_CKPT:-$HOME/.cache/huggingface/hub/models--novateur--WavTokenizer-large-speech-75token/snapshots/9ecf0f435d6f8a75390457f31c6ed8f5f0d1af1b/wavtokenizer_large_speech_320_24k_v2.ckpt}}"
WT_CONFIG="${4:-${WT_CONFIG:-$ROOT_DIR/../WavTokenizer-source/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml}}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model not found: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$INPUT_AUDIO" ]]; then
    echo "ERROR: input audio not found: $INPUT_AUDIO" >&2
    exit 1
fi
if [[ ! -f "$WT_CKPT" ]]; then
    echo "ERROR: wavtokenizer ckpt not found: $WT_CKPT" >&2
    exit 1
fi
if [[ ! -f "$WT_CONFIG" ]]; then
    echo "ERROR: wavtokenizer config not found: $WT_CONFIG" >&2
    exit 1
fi

INSPECT_BIN="$ROOT_DIR/build/inspect-codec"
CODEC_CLI_BIN="$ROOT_DIR/build/codec-cli"
if [[ ! -x "$INSPECT_BIN" || ! -x "$CODEC_CLI_BIN" ]]; then
    echo "ERROR: binaries missing; run: cmake -S . -B build && cmake --build build -j" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg is required" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "$ROOT_DIR/.audio-env/bin/python" ]]; then
        PYTHON_BIN="$ROOT_DIR/.audio-env/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi

MAX_MSE=0.0001
MIN_CORR=0.99

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

INSPECT_LOG="$TMP_DIR/inspect.log"
CPP_LOG="$TMP_DIR/cpp.log"
INPUT_WAV="$TMP_DIR/input_24k_mono.wav"
REF_WAV="$TMP_DIR/reference_wt_py.wav"
OUT_WAV="$TMP_DIR/recon_wt_cpp.wav"
COMPARE_JSON="$TMP_DIR/compare.json"

"$INSPECT_BIN" "$MODEL_PATH" > "$INSPECT_LOG"
ffmpeg -hide_banner -loglevel error -y -i "$INPUT_AUDIO" -ac 1 -ar 24000 "$INPUT_WAV"

"$PYTHON_BIN" - "$ROOT_DIR" "$WT_CKPT" "$WT_CONFIG" "$INPUT_WAV" "$REF_WAV" <<'PY'
import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile

root_dir, ckpt, cfg, in_wav, out_wav = sys.argv[1:]
repo_root = Path(root_dir).resolve().parent
sys.path.append(str(repo_root / "WavTokenizer-source"))
from decoder.pretrained import WavTokenizer

sr, x = wavfile.read(in_wav)
if x.ndim > 1:
    x = x.mean(axis=1)
if np.issubdtype(x.dtype, np.integer):
    x = x.astype(np.float32) / 32768.0
else:
    x = x.astype(np.float32)

wav = torch.from_numpy(np.clip(x, -1.0, 1.0)).unsqueeze(0)
model = WavTokenizer.from_pretrained0802(cfg, ckpt)
model.eval()
with torch.inference_mode():
    bw = torch.tensor([0], dtype=torch.long)
    feat, _ = model.encode(wav, bandwidth_id=bw)
    y = model.decode(feat, bandwidth_id=bw)

y = y[0].detach().cpu().numpy().astype(np.float32)
y = np.clip(y, -1.0, 1.0)
wavfile.write(out_wav, 24000, np.round(y * 32767.0).astype(np.int16))
PY

"$CODEC_CLI_BIN" e2e --model "$MODEL_PATH" --in "$INPUT_WAV" --out "$OUT_WAV" > "$CPP_LOG"

"$PYTHON_BIN" - "$REF_WAV" "$OUT_WAV" "$MAX_MSE" "$MIN_CORR" "$COMPARE_JSON" <<'PY'
import json
import math
import sys
import wave
from array import array

ref_path, out_path, max_mse, min_corr, out_json = sys.argv[1:]
max_mse = float(max_mse)
min_corr = float(min_corr)

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate(); ch = wf.getnchannels(); sw = wf.getsampwidth(); n = wf.getnframes(); d = wf.readframes(n)
    if ch != 1 or sw != 2:
        raise RuntimeError(f"{path}: expected mono PCM16")
    a = array("h"); a.frombytes(d)
    return sr, a

sr_a, a = read_wav(ref_path)
sr_b, b = read_wav(out_path)
if sr_a != sr_b:
    raise RuntimeError("sample_rate mismatch")

n = min(len(a), len(b))
if n <= 0:
    raise RuntimeError("empty wav")

aa = bb = ab = err = 0.0
for i in range(n):
    x = a[i] / 32768.0
    y = b[i] / 32768.0
    d = x - y
    aa += x * x
    bb += y * y
    ab += x * y
    err += d * d
mse = err / n
corr = ab / math.sqrt(max(1e-20, aa * bb))
ok = mse <= max_mse and corr >= min_corr
rep = {"mse": mse, "corr": corr, "ok": ok, "n_ref": len(a), "n_cpp": len(b)}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(rep, f)
if not ok:
    raise SystemExit(f"mse={mse:.8f} corr={corr:.6f}")
PY

readarray -t fields < <("$PYTHON_BIN" - "$COMPARE_JSON" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
print(j['mse']); print(j['corr']); print(j['n_ref']); print(j['n_cpp'])
PY
)

echo "Regression summary:"
echo "  model:       $MODEL_PATH"
echo "  input_audio: $INPUT_AUDIO"
echo "  ckpt:        $WT_CKPT"
echo "  config:      $WT_CONFIG"
echo "  mse:         ${fields[0]}"
echo "  corr:        ${fields[1]}"
echo "  n_ref:       ${fields[2]}"
echo "  n_cpp:       ${fields[3]}"
echo "  target:      mse <= $MAX_MSE, corr >= $MIN_CORR"
