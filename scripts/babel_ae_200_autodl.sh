#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BABEL_AE_CONTINUE_FROM="${BABEL_AE_CONTINUE_FROM:-checkpoints/babel_ae_ema_long_s42/last.pt}"
export BABEL_AE_RUN="${BABEL_AE_RUN:-checkpoints/babel_ae_ema_200_s42}"
export BABEL_AE_LOG="${BABEL_AE_LOG:-logs/babel_ae_ema_200_s42.log}"
export BABEL_EPOCHS="${BABEL_EPOCHS:-200}"
exec bash scripts/babel_ae_extend_autodl.sh "${1:-all}"
