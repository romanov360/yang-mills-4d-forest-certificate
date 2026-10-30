#!/bin/sh
# Kernel-check every Lean file (Lean 4.19.0, core only, no Mathlib). Output goes to ../logs/lean_check.out.
# The all-phase check (Sos*) is split into 8 chunks run JOBS at a time (default 4): about 10 min on 4 cores.
set -e
cd "$(dirname "$0")"
mkdir -p build
export LEAN_PATH="$PWD/build"
python3 gen_certdata.py
git diff --quiet -- CertData.lean 2>/dev/null || echo "note: CertData.lean differs from the committed copy"
# SosData.lean is committed; `python3 gen_sosdata.py` rewrites it (SDP, about 30 s). It is not trusted either way.
for f in Forest4 CertData Cert4 SosData Sos4; do echo "== $f"; lean -o "build/$f.olean" "$f.lean"; done
echo "== Words4"; lean Words4.lean
echo "== SosG0..3, SosP0..3"
printf '%s\n' SosG0 SosG1 SosG2 SosG3 SosP0 SosP1 SosP2 SosP3 |
  xargs -P "${JOBS:-4}" -I{} sh -c 'lean -o build/{}.olean {}.lean && echo "{} ok"'
echo "== SosAll"; lean SosAll.lean
echo "all Lean checks passed"
