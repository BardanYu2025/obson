#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
latent="${1:?Usage: bash scripts/babel_ae_capacity_autodl.sh 128-or-256 [all|train|evaluate|diagnose|export]}"
case "$latent" in 128|256) ;; *) echo 'latent must be 128 or 256' >&2; exit 2 ;; esac
export BABEL_AE_CAPACITY_LATENT="$latent"
export BABEL_AE_CONTEXT=ema8_32
export BABEL_AE_BASELINE_RUN="${BABEL_CAPACITY_BASELINE:-checkpoints/babel_ae_ema_200_s42}"
export BABEL_AE_RUN="${BABEL_CAPACITY_ROOT:-checkpoints}/babel_ae_w256_l6_z${latent}_s42"
export BABEL_AE_LOG="${BABEL_CAPACITY_LOG_ROOT:-logs}/babel_ae_w256_l6_z${latent}_s42.log"
export BABEL_EPOCHS=200 BABEL_BATCH_SIZE=32 BABEL_SEED=42
export BABEL_AE_RESUME="${BABEL_CAPACITY_RESUME:-0}"
unset BABEL_AE_WARM_START BABEL_AE_CONTINUE_FROM BABEL_AE_EXTEND
exec bash scripts/babel_ae_autodl.sh "${2:-all}"
