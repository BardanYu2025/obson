#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python="${PYTHON_BIN:-python}"
mode="${1:-all}"
run="${BABEL_FRESH_RUN:-checkpoints/babel_input_factorial_fresh768}"
data_root="${BABEL_DATA_ROOT:-/root/autodl-tmp/data}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
bundle="artifacts/babel_fresh_20261115"
case "$mode" in all|install-data|verify-data|export) ;; *) echo 'Usage: [all|install-data|verify-data|export]' >&2; exit 2;; esac
if [[ "$#" -gt 1 ]]; then echo 'No extra training arguments: the matched recipe is fixed.' >&2; exit 2; fi
if [[ "$mode" == install-data || "$mode" == verify-data ]]; then
  action=install
  if [[ "$mode" == verify-data ]]; then action=verify; fi
  exec "$python" -m obson.babel.raw_snapshot "$action" --bundle "$bundle" --target "$data_root"
fi
child=""
finish() {
  code=$?
  trap - EXIT
  set +e
  "$python" -m obson.babel.input_factorial_fresh_export --out "$run" --download "$download" --command-exit-code "$code"
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
if [[ "$mode" == all ]]; then
  "$python" -m obson.babel.raw_snapshot install --bundle "$bundle" --target "$data_root"
  "$python" -m obson.babel.input_factorial_fresh_run --bundle "$bundle" --data-root "$data_root" --out "$run" &
  child=$!
  wait "$child"
fi
