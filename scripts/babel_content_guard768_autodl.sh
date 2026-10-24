#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run="${BABEL_CONTENT_GUARD_RUN:-checkpoints/babel_content_guard768}"
source_run="${BABEL_CONTENT_GUARD_SOURCE:-checkpoints/babel_layer_supervision768}"
log="${BABEL_CONTENT_GUARD_LOG:-logs/babel_content_guard768.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
python="${PYTHON_BIN:-python}"
mode="${1:-all}"
case "$mode" in
  all|evaluate|export) ;;
  *) echo 'Usage: bash scripts/babel_content_guard768_autodl.sh [all|evaluate|export]' >&2; exit 2 ;;
esac
if [[ "$#" -gt 1 ]]; then
  echo 'No extra arguments: single worker, effective batch128 and micro8 are fixed.' >&2
  exit 2
fi
child=""
finish() {
  code=$?
  trap - EXIT
  set +e
  "$python" -m obson.babel.content_guard_export --out "$run" --download "$download" \
    --command-exit-code "$code" --mode "$mode" --log "$log"
  export_code=$?
  if [[ "$export_code" != 0 ]]; then
    echo "Report export failed (exit=$export_code); original command exit=$code." >&2
    if [[ "$code" == 0 ]]; then code=$export_code; fi
  fi
  exit "$code"
}
trap finish EXIT
stop() {
  trap '' INT TERM
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" || true
  fi
  exit 130
}
trap stop INT TERM
if [[ "$mode" != export ]]; then
  args=(--source "$source_run" --out "$run")
  if [[ "$mode" == evaluate ]]; then args+=(--evaluate-only); fi
  "$python" -m obson.babel.content_guard_run "${args[@]}" &
  child=$!
  wait "$child"
fi
