#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
log_root="${BABEL_CAPACITY_LOG_ROOT:-logs}"
mkdir -p "$log_root"
for latent in 128 256; do
  echo "Starting capacity experiment z=$latent"
  # Sequential GPU jobs. Failure stops the pair; resume individual arms explicitly.
  bash scripts/babel_ae_capacity_autodl.sh "$latent" all > "$log_root/babel_ae_w256_l6_z${latent}_s42.log" 2>&1
  echo "Completed capacity experiment z=$latent"
done
