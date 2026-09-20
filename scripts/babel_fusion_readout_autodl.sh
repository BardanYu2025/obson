#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
run="${BABEL_FUSION_RUN:-checkpoints/babel_fusion512_s42}"
out="$run/readout_audit"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}/babel_fusion512_readout_audit"
log="${BABEL_READOUT_LOG:-logs/babel_fusion_readout.log}"
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  if [[ -f "$out/readout_audit.json" ]]; then cp "$out/readout_audit.json" "$download/"; fi
  if [[ -f "$log" ]]; then cp "$log" "$download/run.log"; fi
  printf 'exit_code=%s\n' "$status" > "$download/run_status.txt"
  echo "Download: $download (exit_code=$status)"
  exit "$status"
}
trap finish EXIT
"${PYTHON_BIN:-python}" -m obson.babel.fusion_readout_audit --source "$run" --out "$out"
