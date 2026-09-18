#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BABEL_AE_CONTEXT=ema8_32
export BABEL_AE_BASELINE_RUN="${BABEL_AE_BASELINE_RUN:-checkpoints/babel_ae_r1_s42}"
export BABEL_AE_RUN="${BABEL_AE_RUN:-checkpoints/babel_ae_ema_r2_s42}"
export BABEL_AE_LOG="${BABEL_AE_LOG:-logs/babel_ae_ema_r2_s42.log}"
exec bash scripts/babel_ae_autodl.sh "${1:-all}"
