#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: audit.sh RAW_ROOT NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
experiment_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python3}" "$experiment_dir/src/tqdata_audit.py" --root "$1" --out "$2"
