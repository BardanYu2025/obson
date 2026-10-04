#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
run="${BABEL_QUAL_RUN:-checkpoints/babel_recovery_qualification}"
source_run="${BABEL_QUAL_SOURCE:-checkpoints/babel_context_transfer768}"
audit="${BABEL_QUAL_AUDIT:-checkpoints/babel_recovery_r0_v2}"
log="${BABEL_QUAL_LOG:-logs/babel_recovery_qualification.log}"
download="${BABEL_DOWNLOAD_DIR:-/root/autodl-tmp/download}"
mode="${1:-audit}"
case "$mode" in audit|export) ;; *) echo 'Usage: bash scripts/babel_recovery_qualification_autodl.sh [audit|export]' >&2; exit 2 ;; esac
finish() {
  code=$?
  trap - EXIT
  mkdir -p "$download"
  stage=$(mktemp -d "$download/.qualification_export.XXXXXX")
  name=$(basename "$run")
  mkdir -p "$stage/$name"
  if [[ -d "$run" ]]; then
    while IFS= read -r -d '' file; do
      rel="${file#"$run"/}"
      mkdir -p "$stage/$name/$(dirname "$rel")"
      cp "$file" "$stage/$name/$rel"
    done < <(find "$run" -type f \( -name '*.json' -o -name '*.txt' -o -name '*.md' -o -name '*.npz' \) -print0)
  fi
  if [[ -f "$log" ]]; then cp "$log" "$stage/$name/run.log"; fi
  if [[ ! -f "$stage/$name/protocol.md" ]]; then cp docs/BABEL_RECOVERY_QUALIFICATION.md "$stage/$name/protocol.md"; fi
  run_status=not_started
  if [[ -f "$run/status.json" ]]; then
    run_status=$("${PYTHON_BIN:-python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$run/status.json")
  fi
  if [[ "$code" == 124 ]]; then run_status=timeout
  elif [[ "$code" == 3 ]]; then run_status=blocked
  elif [[ "$code" != 0 ]]; then run_status=failed; fi
  printf 'command=%s\ncommand_exit_code=%s\nrun_status=%s\noptimizer_updates=0\ntask_ids=V02,V14\n' "$mode" "$code" "$run_status" > "$stage/$name/export_status.txt"
  archive="$download/${name}_reports.tar.gz"
  tar -czf "$stage/reports.tar.gz" -C "$stage" "$name"
  mv "$stage/reports.tar.gz" "$archive"
  rm -r "$stage"
  echo "Download archive: $archive (command_exit_code=$code, run_status=$run_status)"
  exit "$code"
}
trap finish EXIT
if [[ "$mode" == audit ]]; then
  "${PYTHON_BIN:-python}" - "$source_run" "$audit" "$run" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

source, audit, out = [Path(p).resolve() for p in sys.argv[1:]]
if any(out == p or p in out.parents or out in p.parents for p in (source, audit)) or out.exists():
    raise SystemExit('Use a new separate output; source, prior audit and existing evidence are protected')
try:
    result = subprocess.run([sys.executable, '-m', 'obson.babel.recovery_qualification', '--source', str(source), '--audit', str(audit), '--out', str(out)], timeout=1800)
    raise SystemExit(result.returncode)
except subprocess.TimeoutExpired:
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'status.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(status='timeout', timeout_seconds=1800, optimizer_updates=0, r1_authorized=False, rs_authorized=False, task_ids=['V02', 'V14'])
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)
    raise SystemExit(124)
PY
fi
