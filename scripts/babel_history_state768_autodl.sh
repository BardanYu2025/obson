#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run="${BABEL_HISTORY_STATE_RUN:-checkpoints/babel_history_state768}"
source_run="${BABEL_HISTORY_STATE_SOURCE:-checkpoints/babel_state_readability768}"
log="${BABEL_HISTORY_STATE_LOG:-logs/babel_history_state768.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-all}"
case "$mode" in all|export) ;; *) echo 'Usage: bash scripts/babel_history_state768_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  code=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.history_state_export.XXXXXX")
  name=$(basename "$run")
  mkdir -p "$stage/$name"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$stage/$name/$(dirname "$rel")"
      cp "$file" "$stage/$name/$rel"
    done < <(find "$run" -type f \( -name '*.json' -o -name '*.txt' -o -name '*.md' -o -name '*.html' -o \( -name '*.npz' ! -path "$run/cache/*" ! -path "$run/readouts/*" \) \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$stage/$name/run.log"; fi
  cp docs/BABEL_ROADMAP.md "$stage/$name/roadmap_at_export.md"
  run_status=not_started
  if [[ -f "$run/status.json" ]]; then
    run_status=$("${PYTHON_BIN:-python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run/status.json") || run_status=unreadable_status
  fi
  if [[ "$code" == 124 ]]; then run_status=timeout
  elif [[ "$code" == 130 ]]; then run_status=stopped
  elif [[ "$code" != 0 ]]; then run_status=failed; fi
  printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\noptimizer_updates=0\nreadout_fits=0\ntask_ids=F11\n' "$mode" "$code" "$run_status" > "$stage/$name/export_status.txt"
  archive="$download/${name}_reports.tar.gz"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$archive"
  rm -r "$stage"
  echo "Download archive: $archive (command_exit_code=$code, run_status=$run_status)"
  exit "$code"
}
trap finish EXIT
if [[ "$mode" == all ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.history_state_delivery --source "$source_run" --out "$run" &
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
