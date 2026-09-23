#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_TIME_RUN:-checkpoints/babel_time_confirmation}"
log="${BABEL_TIME_LOG:-logs/babel_time_confirmation.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-all}"
case "$mode" in all|export) ;; *) echo 'Usage: bash scripts/babel_time_confirmation_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.time_confirmation.XXXXXX")
  name=$(basename "$run")
  mkdir -p "$stage/$name"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$stage/$name/$(dirname "$rel")"
      cp "$file" "$stage/$name/$rel"
    done < <(find "$run" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.log' \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$stage/$name/run.log"; fi
  cp docs/BABEL_TIME_CONFIRMATION.md "$stage/$name/experiment_notes.md"
  run_status=partial
  if [[ -f "$run/run_state.json" ]]; then
    run_status=$("${PYTHON_BIN:-python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run/run_state.json")
  fi
  if [[ "$status" == 3 ]]; then run_status=blocked; elif [[ "$status" != 0 ]]; then run_status=failed; fi
  printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\n' "$mode" "$status" "$run_status" > "$stage/$name/run_status.txt"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$download/${name}_reports.tar.gz"
  rm -r "$stage"
  echo "Download archive: $download/${name}_reports.tar.gz (command_exit_code=$status, run_status=$run_status)"
  exit "$status"
}
trap finish EXIT
if [[ "$mode" == all ]]; then
  args=(--source "${BABEL_TIME_SOURCE:-checkpoints/babel_bar_alignment}"
        --registry "${BABEL_TIME_REGISTRY:-checkpoints}"
        --root "${BABEL_TIME_ROOT:-/root/autodl-tmp/data/contracts}" --out "$run"
        --batch "${BABEL_TIME_BATCH:-128}" --device "${BABEL_TIME_DEVICE:-cuda}")
  if [[ -n "${BABEL_TIME_ASOF:-}" ]]; then args+=(--asof "$BABEL_TIME_ASOF"); fi
  "${PYTHON_BIN:-python}" -m obson.babel.time_confirmation "${args[@]}"
fi
