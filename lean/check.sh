#!/bin/sh
# Kernel-check every Lean file (Lean 4.19.0, core only, no Mathlib). Output goes to ../logs/lean_check.out.
set -e
cd "$(dirname "$0")"
mkdir -p build
export LEAN_PATH="$PWD/build"
python3 gen_certdata.py
git diff --quiet -- CertData.lean 2>/dev/null || echo "note: CertData.lean differs from the committed copy"
for f in Forest4 CertData; do echo "== $f"; lean -o "build/$f.olean" "$f.lean"; done
for f in Cert4 Words4; do echo "== $f"; lean "$f.lean"; done
echo "all Lean checks passed"
