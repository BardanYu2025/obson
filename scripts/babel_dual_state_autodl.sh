#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_DUAL_RUN:-checkpoints/babel_dual_state512}"
log="${BABEL_DUAL_LOG:-logs/babel_dual_state512.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-all}"
case "$mode" in all|evaluate|export) ;; *) echo 'Usage: bash scripts/babel_dual_state_autodl.sh [all|evaluate|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.dual_export.XXXXXX")
  name=$(basename "$run")
  mkdir -p "$stage/$name"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$stage/$name/$(dirname "$rel")"
      cp "$file" "$stage/$name/$rel"
    done < <(find "$run" -path "$run/targets" -prune -o -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.html' -o -name '*.log' -o -name '*.md' \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$stage/$name/run.log"; fi
  if [[ -f "$run/targets/statistics.json" ]]; then cp "$run/targets/statistics.json" "$stage/$name/training_statistics.json"; fi
  cp docs/BABEL_DUAL_STATE.md "$stage/$name/experiment_notes.md"
  training_status=partial
  if [[ -f "$run/completion.json" ]]; then training_status=complete; fi
  if [[ "$status" != 0 ]]; then training_status=failed; fi
  printf 'command=%s\ncommand_exit_code=%s\ntraining_status=%s\n' "$mode" "$status" "$training_status" > "$stage/$name/run_status.txt"
  archive="$download/${name}_reports.tar.gz"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$archive"
  rm -r "$stage"
  echo "Download archive: $archive (command_exit_code=$status, training_status=$training_status)"
  exit "$status"
}
trap finish EXIT
if [[ "$mode" != export ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.dual_state "$mode" \
    --source "${BABEL_RECON_RUN:-checkpoints/babel_recon_fusion512_s42}" \
    --long-run "${BABEL_LARGE_RUN:-checkpoints/babel_large512_s42}" \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" --out "$run" \
    --jobs "${BABEL_DUAL_JOBS:-2}" --epochs "${BABEL_DUAL_EPOCHS:-100}" \
    --batch "${BABEL_DUAL_BATCH:-128}" --micro "${BABEL_DUAL_MICRO:-128}" \
    --eval-batch "${BABEL_DUAL_EVAL_BATCH:-16}" --seeds 42 43
fi
