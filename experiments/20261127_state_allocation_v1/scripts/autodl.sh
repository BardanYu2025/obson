#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd -- "$experiment_dir"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
if [[ -n "${BABEL_PYTHON:-}" ]]; then
  python_bin="$BABEL_PYTHON"
elif [[ -x "$experiment_dir/.venv/bin/python" ]]; then
  python_bin="$experiment_dir/.venv/bin/python"
else
  python_bin=python
fi
exec "$python_bin" -m src.cli "$@"
