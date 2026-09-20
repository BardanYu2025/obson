#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_RECON_RUN:-checkpoints/babel_recon_fusion512_s42}"
log="${BABEL_RECON_LOG:-logs/babel_recon_fusion512.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}/$(basename "$run")"
case "${1:-all}" in all|export) ;; *) echo 'Usage: bash scripts/babel_reconstruction_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$download/$(dirname "$rel")"
      cp "$file" "$download/$rel"
    done < <(find "$run" -path "$run/target_cache" -prune -o -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.html' -o -name '*.log' \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$download/run.log"; fi
  cp docs/BABEL_RECONSTRUCTION_FUSION.md "$download/experiment_notes.md"
  printf 'exit_code=%s\n' "$status" > "$download/run_status.txt"
  # Bundle everything automatically; no second command needed after a successful run.
  archive="$(dirname "$download")/$(basename "$run")_reports.tar.gz"
  tar -czf "$archive" -C "$(dirname "$download")" "$(basename "$run")"
  echo "Download archive: $archive (exit_code=$status)"
  exit "$status"
}
trap finish EXIT
if [[ "${1:-all}" == all ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.reconstruction_fusion all \
    --source "${BABEL_FUSION_RUN:-checkpoints/babel_fusion512_s42}" \
    --long-run "${BABEL_LARGE_RUN:-checkpoints/babel_large512_s42}" \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" --out "$run" \
    --jobs "${BABEL_RECON_JOBS:-2}" --epochs "${BABEL_RECON_EPOCHS:-60}" \
    --batch "${BABEL_RECON_BATCH:-128}" --fusion-micro "${BABEL_RECON_MICRO:-8}" \
    --eval-batch "${BABEL_RECON_EVAL_BATCH:-16}"
fi
