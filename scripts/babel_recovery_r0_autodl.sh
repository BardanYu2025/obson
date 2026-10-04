#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_RECOVERY_RUN:-checkpoints/babel_recovery_r0_v2}"
source_run="${BABEL_RECOVERY_SOURCE:-checkpoints/babel_context_transfer768}"
log="${BABEL_RECOVERY_LOG:-logs/babel_recovery_r0_v2.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
prior="${BABEL_RECOVERY_PRIOR:-checkpoints/babel_recovery_r0}"
mode="${1:-audit}"
case "$mode" in audit|export) ;; *) echo 'Usage: bash scripts/babel_recovery_r0_autodl.sh [audit|export]' >&2; exit 2 ;; esac
finish() {
  status=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.recovery_export.XXXXXX")
  name=$(basename "$run")
  mkdir -p "$stage/$name"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$stage/$name/$(dirname "$rel")"
      cp "$file" "$stage/$name/$rel"
    done < <(find "$run" -type f \( -name '*.json' -o -name '*.txt' \) -print0)
  fi
  # Preserve the first failure alongside the corrected run; never use it as
  # an input, acceptance threshold or a reason to skip any checks.
  if [[ -d "$prior" ]]; then
    mkdir -p "$stage/$name/prior_failure"
    for rel in manifest.json audit_status.json macro_s42_original.json macro_s42_context.json; do
      if [[ -f "$prior/$rel" ]]; then cp "$prior/$rel" "$stage/$name/prior_failure/$rel"; fi
    done
  fi
  if [[ -f "$log" ]]; then cp "$log" "$stage/$name/run.log"; fi
  cp docs/BABEL_RECOVERY_R0.md "$stage/$name/protocol.md"
  cp docs/BABEL_RECOVERY_PLAN.md "$stage/$name/plan.md"
  run_status=not_started
  if [[ -f "$run/audit_status.json" ]]; then
    run_status=$("${PYTHON_BIN:-python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run/audit_status.json")
  fi
  if [[ "$status" == 124 ]]; then run_status=timeout; elif [[ "$status" != 0 ]]; then run_status=failed; fi
  printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\noptimizer_updates=0\n' "$mode" "$status" "$run_status" > "$stage/$name/export_status.txt"
  archive="$download/${name}_reports.tar.gz"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$archive"
  rm -r "$stage"
  echo "Download archive: $archive (command_exit_code=$status, run_status=$run_status)"
  exit "$status"
}
trap finish EXIT
if [[ "$mode" == audit ]]; then
  # Independent supervisor also covers blocking CUDA/library calls. No child trainers.
  "${PYTHON_BIN:-python}" - "$source_run" "$run" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

source, out = map(Path, sys.argv[1:])
# Prevent timeout handling from overwriting an existing run or source.
s, o = source.resolve(), out.resolve()
if s == o or s in o.parents or o in s.parents or (o.exists() and any(o.iterdir())):
    raise SystemExit('Use a new, separate audit output directory; existing evidence is protected')
try:
    result = subprocess.run([sys.executable, '-m', 'obson.babel.recovery_audit', '--source', str(source), '--out', str(out)], timeout=3600)
    raise SystemExit(result.returncode)
except subprocess.TimeoutExpired:
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'audit_status.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(status='timeout', timeout_seconds=3600, optimizer_updates=0, r1_authorized=False)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(previous, indent=2))
    temp.replace(path)
    raise SystemExit(124)
PY
fi
