#!/usr/bin/env python3
"""E2E wrapper: runs the model-less C++ constraint test binary.

Run from repo root:  .venv/bin/python tests/e2e/tts_constraint_smoke.py
"""
import subprocess
import sys

from _common import REPO

BIN = REPO / "build" / "tts-constraint-test"

def main() -> int:
    if not BIN.exists():
        print(f"not built: {BIN}; run "
              f"cmake --build build --target tts-constraint-test")
        return 1
    proc = subprocess.run([str(BIN)], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode

if __name__ == "__main__":
    raise SystemExit(main())
