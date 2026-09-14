#!/bin/bash
# AutoDL 一键夜盘信号：增量补内盘K线 → 补外盘日线 → 同步 → GPU出信号
# 前置：tqsdk 已 pip 安装；模型默认 models/best.pt，
#       可用 CKPT 环境变量覆盖（支持逗号分隔多种子集成）
# 用法: ./night_signal_autodl.sh [品种...]   默认 rb sr p
#       CKPT=checkpoints/q90_softfix_s42/best.pt,checkpoints/q90_softfix_s7/best.pt ./night_signal_autodl.sh
# 注意：内盘增量只补指定品种；外盘更新是全局的（棕榈/铁矿/银），与品种参数无关
set -e
cd "$(dirname "$0")"
SYMS=${@:-"rb sr p"}
CKPT=${CKPT:-models/best.pt}
# 天勤账号从环境变量读，不写进仓库：export TQ_USER=xxx TQ_PASS=yyy
: "${TQ_USER:?请先 export TQ_USER=你的天勤账号}"
: "${TQ_PASS:?请先 export TQ_PASS=你的天勤密码}"

echo "==> 增量补天勤K线: $SYMS"
python -u scripts/download_tqsdk_v2.py --symbols $SYMS --periods 60 30 --incremental 2>&1 \
  | grep -E "增量合并|已是最新|已保存|跳过|异常|!!" || true

echo "==> 补外盘日线（棕榈/铁矿/银，当天行为进行中价）"
PYTHONPATH=src python -u scripts/update_foreign_sina.py 2>&1 | grep -E "^\[|⚠️" || true

echo "==> 同步到 data/"
for f in $SYMS; do
  for per in 60 30; do
    [ -f "data/raw_tqsdk/${f}_${per}m.csv" ] && cp "data/raw_tqsdk/${f}_${per}m.csv" "data/${f}_${per}m.csv"
  done
done

echo "==> 实盘信号（$(date '+%H:%M')）| 模型: $CKPT"
OUT=$(PYTHONPATH=src python scripts/signal_live.py --ckpt "$CKPT" --theta-q 0.90 --symbols $SYMS --periods 60 30 2>&1) || {
  echo "!! signal_live 出错，原始输出："; echo "$OUT" | tail -20; exit 1; }
echo "$OUT" | grep -E "^\[|实盘信号|观望|喊多|喊空|现价"
