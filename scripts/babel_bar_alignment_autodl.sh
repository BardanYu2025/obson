#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_ALIGN_RUN:-checkpoints/babel_bar_alignment}"
log="${BABEL_ALIGN_LOG:-logs/babel_bar_alignment.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-all}"
case "$mode" in all|evaluate|preflight|export) ;; *) echo 'Usage: bash scripts/babel_bar_alignment_autodl.sh [all|evaluate|preflight|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.alignment_export.XXXXXX")
  name=$(basename "$run")
  mkdir -p "$stage/$name"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$stage/$name/$(dirname "$rel")"
      cp "$file" "$stage/$name/$rel"
    done < <(find "$run" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.html' -o -name '*.log' \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$stage/$name/run.log"; fi
  cp docs/BABEL_BAR_ALIGNMENT.md "$stage/$name/experiment_notes.md"
  cp docs/BABEL_RESEARCH_GOAL.md "$stage/$name/research_goal.md"
  run_status=partial
  if [[ -f "$run/completion.json" ]]; then run_status=complete; fi
  if [[ -f "$run/audit_status.json" ]] && [[ ! -f "$run/completion.json" ]]; then
    run_status=$("${PYTHON_BIN:-python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run/audit_status.json")
  fi
  if [[ "$mode" == export && -f "$run/run_status.txt" ]]; then
    while IFS='=' read -r key value; do
      if [[ "$key" == run_status ]]; then
        case "$value" in complete|blocked|failed|partial|eligible) run_status="$value" ;; esac
      fi
    done < "$run/run_status.txt"
  fi
  if [[ "$status" == 3 ]]; then run_status=blocked; elif [[ "$status" != 0 ]]; then run_status=failed; fi
  if [[ "$mode" != export && -d "$run" ]]; then
    printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\n' "$mode" "$status" "$run_status" > "$run/run_status.txt"
  fi
  printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\n' "$mode" "$status" "$run_status" > "$stage/$name/run_status.txt"
  archive="$download/${name}_reports.tar.gz"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$archive"
  rm -r "$stage"
  echo "Download archive: $archive (command_exit_code=$status, run_status=$run_status)"
  exit "$status"
}
trap finish EXIT
if [[ "$mode" != export ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.bar_alignment_benchmark "$mode" \
    --source "${BABEL_ALIGN_SOURCE:-checkpoints/babel_shared_readout512}" --out "$run" \
    --jobs "${BABEL_ALIGN_JOBS:-2}"
fi
