#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
mode="${1:-audit}"
case "$mode" in audit|short|short-evaluate) ;; *) echo 'Usage: bash scripts/babel_state_autodl.sh [audit|short|short-evaluate]' >&2; exit 2 ;; esac
source_run="${BABEL_LARGE_RUN:-checkpoints/babel_large512_s42}"
if [[ "$mode" == audit ]]; then
  run="${BABEL_AUDIT_RUN:-checkpoints/babel_large512_audit}"
else
  run="${BABEL_SHORT_RUN:-checkpoints/babel_short512_s42}"
fi
log="${BABEL_STATE_LOG:-logs/babel_${mode}.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}/$(basename "$run")"
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$download/$(dirname "$rel")"
      cp "$file" "$download/$rel"
    done < <(find "$run" -path "$run/teacher_cache" -prune -o -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.html' \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$download/run.log"; fi
  cp docs/BABEL_SHORT_STATE.md "$download/experiment_notes.md"
  printf 'exit_code=%s\n' "$status" > "$download/run_status.txt"
  echo "Download: $download (exit_code=$status)"
  exit "$status"
}
trap finish EXIT
common=(--root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" --source "$source_run" --out "$run" --local-batch "${BABEL_STATE_LOCAL_BATCH:-64}")
if [[ "$mode" == audit ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.stage_audit "${common[@]}" --batch "${BABEL_AUDIT_BATCH:-16}"
else
  extra=()
  if [[ "$mode" == short-evaluate ]]; then extra+=(--evaluate); fi
  "${PYTHON_BIN:-python}" -m obson.babel.short_state "${common[@]}" --batch "${BABEL_SHORT_BATCH:-32}" --epochs "${BABEL_SHORT_EPOCHS:-30}" "${extra[@]}"
fi
