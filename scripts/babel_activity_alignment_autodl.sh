#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_ALIGNMENT_RUN:-checkpoints/babel_activity_alignment512}"
log="${BABEL_ALIGNMENT_LOG:-logs/babel_activity_alignment512.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-all}"
case "$mode" in all|evaluate|export) ;; *) echo 'Usage: bash scripts/babel_activity_alignment_autodl.sh [all|evaluate|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.activity_export.XXXXXX")
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
  cp docs/BABEL_ACTIVITY_ALIGNMENT.md "$stage/$name/experiment_notes.md"
  cp docs/BABEL_RESEARCH_GOAL.md "$stage/$name/research_goal.md"
  run_status=partial
  if [[ -f "$run/completion.json" ]]; then run_status=complete; fi
  if [[ "$status" != 0 ]]; then run_status=failed; fi
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
  "${PYTHON_BIN:-python}" -m obson.babel.activity_alignment "$mode" \
    --activity-run "${BABEL_ACTIVITY_RUN:-checkpoints/babel_activity512}" \
    --fusion-run "${BABEL_FUSION_RUN:-checkpoints/babel_fusion512_s42}" --out "$run" \
    --epochs "${BABEL_ALIGNMENT_EPOCHS:-30}" --streams "${BABEL_ALIGNMENT_STREAMS:-64}" \
    --jobs "${BABEL_ALIGNMENT_JOBS:-2}" --probe-epochs "${BABEL_ALIGNMENT_PROBE_EPOCHS:-80}"
fi
