#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
# Fixed audit runtime; reject inherited malformed or different thread settings by replacing them before Python import.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run="${BABEL_V14_RUN:-checkpoints/babel_recovery_provenance}"
source_run="${BABEL_V14_SOURCE:-checkpoints/babel_recovery_interface}"
log="${BABEL_V14_LOG:-logs/babel_recovery_provenance.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
bundle_source="${BABEL_V14_BUNDLE_SOURCE:-checkpoints/babel_control600_v2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mode="${1:-all}"
case "$mode" in all|export) ;; *) echo 'Usage: bash scripts/babel_recovery_provenance_autodl.sh [all|export]' >&2; exit 2 ;; esac
finish() {
  code=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.v14_export.XXXXXX")
  name=$(basename "$run")
  mkdir -p "$stage/$name"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$stage/$name/$(dirname "$rel")"
      cp "$file" "$stage/$name/$rel"
    done < <(find "$run" -type f \( -name '*.json' -o -name '*.txt' -o -name '*.md' -o \( -name '*.npz' ! -path "$run/cache/*" \) \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$stage/$name/run.log"; fi
  if [[ -f "${run}.startup_receipt.json" ]]; then
    cp "${run}.startup_receipt.json" "$stage/$name/startup_receipt.json"
  fi
  if [[ -d "${run}.startup_failure_v1" ]]; then
    mkdir -p "$stage/$name/previous_startup"
    cp "${run}.startup_failure_v1/"*.json "$stage/$name/previous_startup/" 2>/dev/null || true
  fi
  cp docs/BABEL_ROADMAP.md "$stage/$name/roadmap_at_export.md"
  run_status=not_started
  if [[ -f "$run/status.json" ]]; then
    run_status=$("${PYTHON_BIN:-python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run/status.json")
  fi
  if [[ "$code" == 124 ]]; then run_status=timeout
  elif [[ "$code" == 130 || "$code" == 143 ]]; then run_status=stopped
  elif [[ "$code" != 0 ]]; then run_status=failed; fi
  printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\noptimizer_updates=see_worker_journals\ntask_ids=V14\n' "$mode" "$code" "$run_status" > "$stage/$name/export_status.txt"
  archive="$download/${name}_reports.tar.gz"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$archive"
  rm -r "$stage"
  echo "Download archive: $archive (command_exit_code=$code, run_status=$run_status)"
  exit "$code"
}
trap finish EXIT
if [[ "$mode" == all ]]; then
  if [[ -f "$download/$(basename "$run")_reports.tar.gz" ]]; then
    prior=$(mktemp "$download/$(basename "$run")_before_startup_fix.XXXXXX.tar.gz")
    cp "$download/$(basename "$run")_reports.tar.gz" "$prior"
    echo "Preserved previous archive: $prior"
  fi
  timeout --signal=TERM --kill-after=30s 7200s "${PYTHON_BIN:-python}" -m obson.babel.recovery_provenance_startup --source "$source_run" --bundle-source "$bundle_source" --out "$run"
fi
