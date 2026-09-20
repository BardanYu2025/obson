#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
run="${BABEL_FUSION_RUN:-checkpoints/babel_fusion512_s42}"
log="${BABEL_FUSION_LOG:-logs/babel_fusion512.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}/$(basename "$run")"
case "${1:-all}" in all|export) ;; *) echo 'Usage: bash scripts/babel_fusion_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$download/$(dirname "$rel")"
      cp "$file" "$download/$rel"
    done < <(find "$run" -path "$run/feature_cache" -prune -o -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.html' \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$download/run.log"; fi
  cp docs/BABEL_RESIDUAL_FUSION.md "$download/experiment_notes.md"
  printf 'exit_code=%s\n' "$status" > "$download/run_status.txt"
  echo "Download: $download (exit_code=$status)"
  exit "$status"
}
trap finish EXIT
if [[ "${1:-all}" == all ]]; then
  "${PYTHON_BIN:-python}" -m obson.babel.residual_fusion \
    --root "${BABEL_DATA:-/root/autodl-tmp/data/contracts}" \
    --long-run "${BABEL_LARGE_RUN:-checkpoints/babel_large512_s42}" \
    --short-run "${BABEL_SHORT_RUN:-checkpoints/babel_short512_b128_s42}" \
    --out "$run" --batch "${BABEL_FUSION_BATCH:-512}" \
    --encode-batch "${BABEL_FUSION_ENCODE_BATCH:-16}" --stream-batch "${BABEL_FUSION_STREAM_BATCH:-128}" \
    --epochs "${BABEL_FUSION_EPOCHS:-100}" --preserve "${BABEL_FUSION_PRESERVE:-0.1}"
fi
