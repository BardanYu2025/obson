#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
stage="${1:-train}"
case "$stage" in train|evaluate) ;; *) echo 'Usage: bash scripts/babel_repr_autodl.sh {train|evaluate}' >&2; exit 2 ;; esac
"${PYTHON_BIN:-python}" -m obson.babel.representation "$stage" \
  --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" \
  --reference "${BABEL_REFERENCE:-checkpoints/babel_r1_s42/manifest.json}" \
  --out "${BABEL_REPR_RUN:-checkpoints/babel_repr_r1_s42}" \
  --epochs "${BABEL_EPOCHS:-30}" --batch-size "${BABEL_BATCH_SIZE:-32}"
