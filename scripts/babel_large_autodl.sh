#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
run="${BABEL_LARGE_RUN:-checkpoints/babel_large512_s42}"
log="${BABEL_LARGE_LOG:-logs/babel_large512_s42.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}/$(basename "$run")"
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  for name in manifest.json preflight.json coverage.json local_metrics.json hierarchical_metrics.json hierarchical_examples.json hierarchical_examples.html local_examples.html artifacts.json; do
    if [[ -f "$run/$name" ]]; then cp "$run/$name" "$download/$name"; fi
  done
  for phase in local aggregate joint; do
    if [[ -f "$run/$phase/history.jsonl" ]]; then cp "$run/$phase/history.jsonl" "$download/${phase}_history.jsonl"; fi
    if [[ -f "$run/$phase/manifest.json" ]]; then cp "$run/$phase/manifest.json" "$download/${phase}_manifest.json"; fi
  done
  if [[ -f "$log" ]]; then cp "$log" "$download/training.log"; fi
  cp docs/BABEL_LARGE512.md "$download/experiment_notes.md"
  printf 'exit_code=%s\n' "$status" > "$download/run_status.txt"
  echo "Download: $download (exit_code=$status)"
  exit "$status"
}
case "${1:-all}" in all|preflight|evaluate|export) ;; *) echo 'Usage: bash scripts/babel_large_autodl.sh [all|preflight|evaluate|export]' >&2; exit 2 ;; esac
trap finish EXIT
if [[ "${1:-all}" != export ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.large_history "${1:-all}" \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" \
    --reference "${BABEL_REFERENCE:-checkpoints/babel_r1_s42/manifest.json}" --out "$run" \
    --local-epochs "${BABEL_LARGE_LOCAL_EPOCHS:-200}" --local-micro "${BABEL_LARGE_LOCAL_MICRO:-16}" --local-effective "${BABEL_LARGE_LOCAL_EFFECTIVE:-32}" \
    --aggregate-epochs "${BABEL_LARGE_AGG_EPOCHS:-30}" --aggregate-micro "${BABEL_LARGE_AGG_MICRO:-2}" --aggregate-effective "${BABEL_LARGE_AGG_EFFECTIVE:-16}" \
    --joint-epochs "${BABEL_LARGE_JOINT_EPOCHS:-20}" --joint-micro "${BABEL_LARGE_JOINT_MICRO:-1}" --joint-effective "${BABEL_LARGE_JOINT_EFFECTIVE:-8}"
fi
