#!/usr/bin/env python3
"""Write a model's recommended sampling (general.sampling.*) into a backbone GGUF.

    scripts/set-backbone-sampling.py --model qwen3_tts in.gguf out.gguf

The values are scripts/utils/sampling.py's MODEL_SAMPLING; convert-backbone-to-gguf.py
writes them on conversion, this tool updates an existing backbone (including
upstream ones such as OuteTTS / NeuTTS).  `out` may not be `in`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common" / "third-party" / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(REPO / "scripts"))

from utils.sampling import MODEL_SAMPLING, add_sampling_to_gguf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SAMPLING))
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    if a.input.resolve() == a.output.resolve():
        raise SystemExit("output must differ from input")
    add_sampling_to_gguf(a.input, a.output, a.model)
    print(f"wrote {a.output} with {a.model}'s sampling: {MODEL_SAMPLING[a.model]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
