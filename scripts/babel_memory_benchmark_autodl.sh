#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
run="${BABEL_MEMORY_BENCHMARK_RUN:-checkpoints/babel_memory_benchmark_s42}"
log="${BABEL_MEMORY_BENCHMARK_LOG:-logs/babel_memory_benchmark_s42.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}/$(basename "$run")"
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  for name in manifest.json benchmark_metrics.json; do
    if [[ -f "$run/$name" ]]; then cp "$run/$name" "$download/$name"; fi
  done
  if [[ -f "$log" ]]; then cp "$log" "$download/benchmark.log"; fi
  cp docs/BABEL_MEMORY_BENCHMARK.md "$download/benchmark_notes.md"
  printf 'exit_code=%s\n' "$status" > "$download/run_status.txt"
  echo "Download: $download (exit_code=$status)"
  exit "$status"
}
case "${1:-all}" in all|export) ;; *) echo 'Usage: bash scripts/babel_memory_benchmark_autodl.sh [all|export]' >&2; exit 2 ;; esac
trap finish EXIT
if [[ "${1:-all}" == all ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.memory_benchmark \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" \
    --memory-run "${BABEL_MEMORY_RUN:-checkpoints/babel_memory_z256_s42}" --out "$run"
fi
