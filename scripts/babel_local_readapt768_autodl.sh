#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_LOCAL_READAPT_RUN:-checkpoints/babel_local_readapt768}"
log="${BABEL_LOCAL_READAPT_LOG:-logs/babel_local_readapt768.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-all}"
case "$mode" in all|export) ;; *) echo 'Usage: bash scripts/babel_local_readapt768_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  command_code=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.local_readapt768.XXXXXX")
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
  cp docs/BABEL_LOCAL_READAPT768.md "$stage/$name/experiment_notes.md"
  run_status=partial
  if [[ -f "$run/completion.json" ]]; then run_status=complete;
  elif [[ -f "$run/failure.json" ]]; then run_status=failed; fi
  if [[ "$command_code" != 0 ]]; then run_status=failed; fi
  printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\n' "$mode" "$command_code" "$run_status" > "$stage/$name/run_status.txt"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$download/${name}_reports.tar.gz"
  rm -r "$stage"
  echo "Download archive: $download/${name}_reports.tar.gz (command_exit_code=$command_code, run_status=$run_status)"
  exit "$command_code"
}
trap finish EXIT
if [[ "$mode" == all ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.local_readapt_run all \
    --source "${BABEL_LOCAL_READAPT_SOURCE:-checkpoints/babel_growth768}" --out "$run" \
    --jobs "${BABEL_LOCAL_READAPT_JOBS:-2}"
fi
