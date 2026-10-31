#!/bin/sh
# Kernel-check the block arithmetic (Lean core only).  Usage: sh check.sh
set -e
cd "$(dirname "$0")"
lean -o BlockData.olean BlockData.lean
LEAN_PATH=. lean BlockCheck.lean && echo "BlockCheck: all theorems checked"
