#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_READOUT_RUN:-checkpoints/babel_state_readout1024}"
log="${BABEL_READOUT_LOG:-logs/babel_state_readout1024.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-all}"
case "$mode" in all|export) ;; *) echo 'Usage: bash scripts/babel_readout_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.readout_export.XXXXXX")
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
  cp docs/BABEL_STATE_READOUT.md "$stage/$name/experiment_notes.md"
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
  "${PYTHON_BIN:-python}" -m obson.babel.state_readout \
    --bundle "${BABEL_STREAM_BUNDLE:-checkpoints/babel_stream1024_numeric/inference_bundle.pt}" \
    --fusion-run "${BABEL_FUSION_RUN:-checkpoints/babel_fusion512_s42}" \
    --long-run "${BABEL_LARGE_RUN:-checkpoints/babel_large512_s42}" \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" --out "$run" \
    --batch "${BABEL_READOUT_BATCH:-128}" --stream-batch "${BABEL_READOUT_STREAM_BATCH:-128}"
fi
