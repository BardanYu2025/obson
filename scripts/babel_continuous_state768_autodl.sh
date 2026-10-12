#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run="${BABEL_CONTINUOUS_STATE_RUN:-checkpoints/babel_continuous_state768}"
source_run="${BABEL_CONTINUOUS_STATE_SOURCE:-checkpoints/babel_history_state768}"
raw_root="${BABEL_RAW_ROOT:-/root/autodl-tmp/data/contracts}"
log="${BABEL_CONTINUOUS_STATE_LOG:-logs/babel_continuous_state768.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
python_bin="${PYTHON_BIN:-python}"
mode="${1:-all}"
case "$mode" in all|export) ;; *) echo 'Usage: bash scripts/babel_continuous_state768_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  code=$?
  trap - EXIT
  if ! "$python_bin" -m obson.babel.continuous_state_launch --export \
      --source "$source_run" --raw-root "$raw_root" --out "$run" --log "$log" \
      --download "$download" --mode "$mode" --command-exit-code "$code"; then
    echo "Report export failed; original command_exit_code=$code" >&2
    if [[ "$code" == 0 ]]; then code=1; fi
  fi
  exit "$code"
}
trap finish EXIT
if [[ "$mode" == all ]]; then
  "$python_bin" -m obson.babel.continuous_state_run --source "$source_run" --raw-root "$raw_root" --out "$run" &
  worker_pid=$!
  stop_worker() {
    trap '' INT TERM
    kill -TERM "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" || true
    exit 130
  }
  trap stop_worker INT TERM
  wait "$worker_pid"
fi
