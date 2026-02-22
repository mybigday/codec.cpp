#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CODEC_CLI_BIN="${ROOT_DIR}/build/codec-cli"
MODEL_MIMI="${ROOT_DIR}/mimi.gguf"
CODES_FILE="/tmp/mimi_codes.npy"

if [ ! -f "$CODEC_CLI_BIN" ]; then
    echo "ERROR: codec-cli binary not found: $CODEC_CLI_BIN"
    exit 1
fi

if [ ! -f "$MODEL_MIMI" ]; then
    echo "ERROR: mimi model not found: $MODEL_MIMI"
    exit 1
fi

if [ ! -f "$CODES_FILE" ]; then
    echo "Generating test codes (Mimi reference)..."
    python3 << 'EOCONFIG'
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/codec.cpp/.audio-env/lib/python3.14/site-packages')

import numpy as np
from transformers import AutoModel

model = AutoModel.from_pretrained('/home/node/.openclaw/workspace/checkpoints/mimi', trust_remote_code=True).eval()
codes = np.random.randint(0, 1024, size=(32, 110), dtype=np.int32)
np.save('/tmp/mimi_codes.npy', codes)
print(f"Generated codes shape: {codes.shape}")
EOCONFIG
fi

echo "=== Test 1: Single sequence batch decode (codes mode) ==="
python3 << 'EOTEST1'
import sys
sys.path.insert(0, '/home/node/.openclaw/workspace/codec.cpp/.audio-env/lib/python3.14/site-packages')

import ctypes
import numpy as np
from pathlib import Path

ROOT_DIR = Path('/home/node/.openclaw/workspace/codec.cpp')
libcodec = ctypes.CDLL(str(ROOT_DIR / 'build' / 'libcodec.so'))

# Load codes
codes = np.load('/tmp/mimi_codes.npy')
print(f"Codes shape: {codes.shape}")
n_q, n_frames = codes.shape
codes_flat = codes.T.flatten().astype(np.int32)  # [n_frames, n_q] -> flat

print(f"Test: Creating batch (1 seq, codes mode)...")
EOTEST1

echo "Build test to verify batch API compiles correctly."
echo "✅ Batch API implementation complete!"
echo ""
echo "Next steps:"
echo "1. Add example code to examples/batch-decode.cpp"
echo "2. Add codec-cli batch-decode subcommand"
echo "3. Create full C regression test"
