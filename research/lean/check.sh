#!/bin/sh
# Compile one file against a Mathlib that is already built, without going through lake.
#
# `lake build` is the standard route (README.md). This is the quick one for a single file once
# Mathlib's oleans exist: after `lake exe cache get` they are under this directory's own .lake/,
# which is the default. MATHLIB may name any other built tree of the same Mathlib.
#
#     ./check.sh Infer/Permutation.lean -o Infer/Permutation.olean
#
# The order matters: `SecondMoment` imports `Permutation`, so its olean has to exist first, and a
# file compiled out of order reports a missing object rather than a broken proof.
MATHLIB="${MATHLIB:-$(cd "$(dirname "$0")" && pwd)}"
if [ ! -f "$MATHLIB/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean" ]; then
  echo "no built Mathlib under $MATHLIB — set MATHLIB, or run this on a host that has one" >&2
  exit 2
fi
LP="$(pwd)"
for d in "$MATHLIB"/.lake/packages/*/.lake/build/lib/lean; do LP="$LP:$d"; done
export PATH="$HOME/.elan/bin:$PATH"
LEAN_PATH="$LP" exec lean "$@"
