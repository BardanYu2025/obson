#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python="${PYTHON_BIN:-python}"
mode="${1:-all}"
run="${BABEL_RESTORE_RUN:-checkpoints/babel_macro_restore_s01}"
data_root="${BABEL_DATA_ROOT:-/root/autodl-tmp/data}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
micro="${BABEL_RESTORE_MICRO:-16}"
case "$mode" in all|prepare|export) ;; *) echo 'Usage: [all|prepare|export]' >&2; exit 2;; esac
if [[ "$#" -gt 1 ]]; then echo 'No extra positional training arguments.' >&2; exit 2; fi
child=""
finish() {
  code=$?
  trap - EXIT
  set +e
  "$python" -m obson.babel.macro_restore_backup --out "$run" --download "$download" --exit-code "$code"
  export_code=$?
  if [[ "$code" == 0 && "$export_code" != 0 ]]; then code=$export_code; fi
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
  # Original raw CSVs live in the existing portable Git snapshot. Its fresh-study
  # cache is installed as an inert asset; Macro data is rebuilt from ORIGINAL rows.
  "$python" -m obson.babel.raw_snapshot install --bundle artifacts/babel_fresh_20261115 --target "$data_root"
  args=(--raw-root "$data_root/contracts" --out "$run" --download "$download" --micro "$micro")
  if [[ "$mode" == prepare ]]; then args+=(--prepare-only); fi
  "$python" -m obson.babel.macro_restore_run "${args[@]}" &
  child=$!
  wait "$child"
fi
