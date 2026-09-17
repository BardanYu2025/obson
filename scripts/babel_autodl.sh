#!/usr/bin/env bash
# Run from any directory. Train only on CUDA; never silently fall back to CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

stage="${1:-train}"
python_bin="${PYTHON_BIN:-python}"
run_dir="${BABEL_RUN:-checkpoints/babel_r1_s42}"
data_root="${BABEL_DATA:-data/contracts}"
index_file="${BABEL_INDEX:-data/babel/babel_r1_s42.npz}"
read -r -a symbols <<< "${BABEL_SYMBOLS:-rb hc i sr p j jm m y cu ag TA MA}"
read -r -a periods <<< "${BABEL_PERIODS:-60 30 15}"

case "$stage" in
  train|evaluate|index)
    "$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: use an AutoDL PyTorch GPU image; no CPU fallback"; print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)'
    ;;
  audit|benchmark|blind|blind-pairs) ;;
  *) echo "Usage: bash scripts/babel_autodl.sh {audit|train|evaluate|index|benchmark|blind|blind-pairs}" >&2; exit 2 ;;
esac

if [[ "$stage" == train || "$stage" == audit || "$stage" == index ]]; then
  for symbol in "${symbols[@]}"; do
    for period in "${periods[@]}"; do
      if ! compgen -G "$data_root/$symbol/*_${period}m.csv" >/dev/null; then
        echo "Missing contract data: $data_root/$symbol/*_${period}m.csv. Supply the files or explicitly set BABEL_PERIODS/BABEL_SYMBOLS." >&2
        exit 2
      fi
    done
  done
fi

case "$stage" in
  audit)
    "$python_bin" -m obson.babel audit --root "$data_root" --symbols "${symbols[@]}" --periods "${periods[@]}" --out "${run_dir}_audit.json"
    ;;
  train)
    "$python_bin" -m obson.babel train --root "$data_root" --symbols "${symbols[@]}" --periods "${periods[@]}" \
      --window 128 --warmup 32 --hidden 64 --layers 2 --epochs "${BABEL_EPOCHS:-30}" \
      --stride 16 --batch-size "${BABEL_BATCH_SIZE:-64}" --lr 3e-4 --seed "${BABEL_SEED:-42}" \
      --device cuda --out "$run_dir"
    ;;
  evaluate)
    "$python_bin" -m obson.babel evaluate --root "$data_root" --checkpoint "$run_dir/best.pt" \
      --split test --stride 1 --device cuda --out "$run_dir/test_metrics.json"
    ;;
  index)
    "$python_bin" -m obson.babel index --root "$data_root" --symbols "${symbols[@]}" --periods "${periods[@]}" \
      --checkpoint "$run_dir/best.pt" --window 128 --stride 16 --device cuda --out "$index_file"
    ;;
  benchmark)
    "$python_bin" -m obson.babel benchmark --root "$data_root" --index "$index_file" \
      --checkpoint "$run_dir/best.pt" --count 100 --out "$run_dir/retrieval_metrics.json"
    ;;
  blind)
    "$python_bin" -m obson.babel blind --root "$data_root" --index "$index_file" \
      --checkpoint "$run_dir/best.pt" --count 20 --out "$run_dir/blind"
    ;;
  blind-pairs)
    "$python_bin" -m obson.babel blind-pairs --root "$data_root" --index "$index_file" \
      --checkpoint "$run_dir/best.pt" --count 6 --repeats 3 --out "$run_dir/blind_pairs_v1"
    ;;
esac
