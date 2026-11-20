#!/bin/sh
# Smoke-test an installed release tree (cmake --install output): every binary
# resolves its libraries through the install rpath and starts.  Needs no model.
#   release-smoke.sh STAGE_DIR
# Runs on the host or inside a clean container (sh only, no bash).
set -eu
stage=$1
cd /   # not the build tree: libraries must come from the rpath, not the cwd

if command -v ldd >/dev/null 2>&1; then
    for f in "$stage"/bin/* "$stage"/lib/*.so; do
        if ldd "$f" 2>/dev/null | grep -q "not found"; then
            echo "missing libraries for $f:"; ldd "$f"; exit 1
        fi
    done
elif command -v otool >/dev/null 2>&1; then
    for f in "$stage"/bin/*; do otool -L "$f"; done
fi

"$stage/bin/tts-cli" --version
if [ -x "$stage/bin/tts-server" ]; then "$stage/bin/tts-server" --version; fi
# codec-cli has no --version; its usage text proves it loaded.
"$stage/bin/codec-cli" 2>&1 | grep -q usage
echo "smoke ok: $stage"
