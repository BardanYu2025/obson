#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
stage="${1:-all}"
run_dir="${BABEL_AE_RUN:-checkpoints/babel_ae_r1_s42}"
download_dir="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}/$(basename "$run_dir")"
log_file="${BABEL_AE_LOG:-logs/babel_ae_r1_s42.log}"

export_reports() {
  mkdir -p "$download_dir" || return
  for report in manifest.json history.jsonl ae_metrics.json reconstruction_examples.json reconstruction_examples.html; do
    if [[ -f "$run_dir/$report" ]]; then cp "$run_dir/$report" "$download_dir/$report" || return; fi
  done
  if [[ -f "$log_file" ]]; then cp "$log_file" "$download_dir/training.log" || return; fi
  cp docs/BABEL_HISTORY_AE.md "$download_dir/experiment_notes.md"
}

finish() {
  status=$?
  trap - EXIT
  export_reports || { echo 'Report export failed; check download path and disk space.' >&2; exit 1; }
  printf 'stage=%s\nexit_code=%s\n' "$stage" "$status" > "$download_dir/run_status.txt"
  echo "Download reports: $download_dir (exit_code=$status)"
  exit "$status"
}

case "$stage" in all|train|evaluate|export) ;; *) echo 'Usage: bash scripts/babel_ae_autodl.sh {all|train|evaluate|export}' >&2; exit 2 ;; esac
trap finish EXIT
run_stage() {
  "${PYTHON_BIN:-python}" -m obson.babel.history_autoencoder "$1" \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" \
    --reference "${BABEL_REFERENCE:-checkpoints/babel_r1_s42/manifest.json}" \
    --out "$run_dir" --epochs "${BABEL_EPOCHS:-30}" \
    --batch-size "${BABEL_BATCH_SIZE:-32}" --seed "${BABEL_SEED:-42}"
}
case "$stage" in
  all) run_stage train; run_stage evaluate ;;
  train|evaluate) run_stage "$stage" ;;
  export) : ;;
esac
