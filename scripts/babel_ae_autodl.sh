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
  for report in manifest.json history.jsonl warm_start_validation.json ae_metrics.json reconstruction_examples.json reconstruction_examples.html ae_diagnostics.json ae_diagnostics.md; do
    if [[ -f "$run_dir/$report" ]]; then cp "$run_dir/$report" "$download_dir/$report" || return; fi
  done
  if [[ -f "$log_file" ]]; then
    log_name="training.log"
    if [[ "$stage" == diagnose ]]; then log_name="diagnostics.log"; fi
    cp "$log_file" "$download_dir/$log_name" || return
  fi
  cp docs/BABEL_HISTORY_AE.md "$download_dir/experiment_notes.md" || return
  if [[ "${BABEL_AE_CONTEXT:-none}" == ema8_32 ]]; then cp docs/BABEL_STAGE2.md "$download_dir/stage2_notes.md" || return; fi
  if [[ "${BABEL_AE_EXTEND:-0}" == 1 ]]; then cp docs/BABEL_EXTEND.md "$download_dir/extension_notes.md" || return; fi
}

finish() {
  status=$?
  trap - EXIT
  export_reports || { echo 'Report export failed; check download path and disk space.' >&2; exit 1; }
  printf 'stage=%s\nexit_code=%s\n' "$stage" "$status" > "$download_dir/run_status.txt"
  echo "Download reports: $download_dir (exit_code=$status)"
  exit "$status"
}

case "$stage" in all|train|evaluate|diagnose|export) ;; *) echo 'Usage: bash scripts/babel_ae_autodl.sh {all|train|evaluate|diagnose|export}' >&2; exit 2 ;; esac
trap finish EXIT
run_stage() {
  extra=()
  if [[ -n "${BABEL_AE_CONTEXT:-}" ]]; then extra+=(--context "$BABEL_AE_CONTEXT"); fi
  if [[ -n "${BABEL_AE_BASELINE_RUN:-}" ]]; then extra+=(--baseline-run "$BABEL_AE_BASELINE_RUN"); fi
  if [[ "$1" == train ]]; then
    if [[ "${BABEL_AE_RESUME:-0}" == 1 ]]; then extra+=(--resume)
    elif [[ -n "${BABEL_AE_CONTINUE_FROM:-}" ]]; then extra+=(--continue-from "$BABEL_AE_CONTINUE_FROM")
    elif [[ -n "${BABEL_AE_WARM_START:-}" ]]; then extra+=(--warm-start "$BABEL_AE_WARM_START"); fi
  fi
  "${PYTHON_BIN:-python}" -m obson.babel.history_autoencoder "$1" \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" \
    --reference "${BABEL_REFERENCE:-checkpoints/babel_r1_s42/manifest.json}" \
    --out "$run_dir" --epochs "${BABEL_EPOCHS:-30}" \
    --batch-size "${BABEL_BATCH_SIZE:-32}" --seed "${BABEL_SEED:-42}" "${extra[@]}"
}
case "$stage" in
  all) run_stage train; run_stage evaluate; run_stage diagnose ;;
  train|evaluate|diagnose) run_stage "$stage" ;;
  export) : ;;
esac
