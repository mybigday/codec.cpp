"""Run llama.cpp's `convert_hf_to_gguf.py` on a CSM backbone HF dir,
patched to accept the CSM tokenizer's pre-tokenizer hash.

CSM ships a Llama-3 tokenizer with a Mistral-flavoured regex twist; its
pre-tokenizer hash isn't in llama.cpp's `get_vocab_base_pre()` table
and the standard converter aborts with `BPE pre-tokenizer was not
recognized`.  At codec_lm runtime the host LLM is fed embeddings
directly via `llama_batch.embd`, so the tokenizer/BPE merges are never
exercised.  This wrapper monkey-patches the unrecognised-hash branch
to use `"llama-bpe"` (the same regex family) and runs the converter
unchanged otherwise.

Usage:
    python scripts/convert_csm_backbone_to_gguf.py \\
        --hf-dir /tmp/csm_backbone_hf \\
        --out    models/csm/llama_backbone.gguf \\
        [--llama-cpp /path/to/llama.cpp]   # default: ~/Projects/llama.cpp
        [--outtype f16]                    # default: f16

Combines naturally with `scripts/extract_csm_backbone.py`:
    python scripts/extract_csm_backbone.py --csm sesame/csm-1b --out /tmp/csm_backbone_hf
    python scripts/convert_csm_backbone_to_gguf.py \\
        --hf-dir /tmp/csm_backbone_hf --out models/csm/llama_backbone.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_LLAMA_CPP = Path.home() / "Projects" / "llama.cpp"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True,
                    help="HF directory produced by extract_csm_backbone.py")
    ap.add_argument("--out", required=True,
                    help="Output GGUF path")
    ap.add_argument("--llama-cpp", default=str(DEFAULT_LLAMA_CPP),
                    help=f"Path to llama.cpp checkout (default: {DEFAULT_LLAMA_CPP})")
    ap.add_argument("--outtype", default="f16",
                    help="Output dtype passed through to convert_hf_to_gguf.py "
                         "(default: f16)")
    args = ap.parse_args()

    conv = Path(args.llama_cpp) / "convert_hf_to_gguf.py"
    if not conv.is_file():
        print(f"FATAL: missing {conv}", file=sys.stderr)
        return 1

    sys.argv = [
        "convert_hf_to_gguf.py", str(args.hf_dir),
        "--outfile", str(args.out),
        "--outtype", args.outtype,
    ]

    src = conv.read_text().replace(
        'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
        'res = "llama-bpe"  # CSM tokenizer pre is the same regex family as llama-3'
    )
    exec(compile(src, str(conv), "exec"),
         {"__name__": "__main__", "__file__": str(conv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
