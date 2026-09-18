# EMA 模型延长训练：先判断是否训练不足

阶段 2 综合测试损失从 0.12446 降至 0.09956，收盘误差从 33.62 降至
27.17 bp，但单根变化误差只下降约 1.3%。30 轮尚无明确平台证据。
本轮保持模型、128 维瓶颈、输入、目标、损失和学习率不变，不加入新损失。

从阶段 2 的 best.pt（第 29 轮）热启动。源实验已经消耗 30 轮，本实验
额外训练 70 轮，预算编号 31–100。权重实际路径为前 29 轮加后 70 轮，
不能说是无缝训练到第 100 轮：旧文件没有优化器状态，本次 AdamW 重新初始化。
因此实验结论是“热启动后继续训练是否改善”，不严格等同于从头连续训练 100 轮。
学习率仍 3e-4，weight_decay=.01，batch=32，seed=42，clip=1。
没有提前停止或学习率调度，避免同时改变多个变量。

每轮只在原验证集记录原综合损失、收盘误差、1/4/16 根变化 MAE、相关性、
标准差比及支持窗口数，沿用诊断的逐窗口等权口径。最优模型只按原综合验证
损失选取。初始热启动模型也参与最优候选；不会因为训练变差而丢失旧结果。
初始验证结果单独保存为 warm_start_validation.json，新 history.jsonl 只含 31–100。
综合损失下降而单根细节停滞，才支持下一步优先检验损失或容量限制。
测试集仅在全部训练完成后运行评价与诊断，不用于每轮模型选择。

last.pt 原子替换保存：当前权重、AdamW 状态、预算轮次、Python/NumPy/Torch/
CUDA 随机状态、历史记录和最佳权重。best.pt 是仅用于推理的最佳权重。
从 last.pt 恢复时重建报告，避免中断留下历史记录与 checkpoint 不一致。
中断在轮次中间时，恢复到上一个完整轮次，重新运行未完成轮次。
保存随机状态不代表跨硬件或非确定性 CUDA 内核可以逐位复现。
不改动源实验；新实验目录必须为空。恢复只能沿用声明过的相同预算和配置。

## AutoDL 启动

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_REFERENCE=checkpoints/babel_r1_s42/manifest.json
export BABEL_AE_WARM_START=checkpoints/babel_ae_ema_r2_s42/best.pt
export BABEL_AE_RUN=checkpoints/babel_ae_ema_long_s42
export BABEL_AE_LOG=logs/babel_ae_ema_long_s42.log
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
export BABEL_EPOCHS=100 BABEL_BATCH_SIZE=32 BABEL_SEED=42 BABEL_AE_RESUME=0
mkdir -p logs
nohup bash scripts/babel_ae_extend_autodl.sh all > "$BABEL_AE_LOG" 2>&1 &
tail -f "$BABEL_AE_LOG"
```

中断后保留上述环境，设 `export BABEL_AE_RESUME=1`，用同样的 nohup 命令启动，
但重定向改为 `>> "$BABEL_AE_LOG"` 追加日志。请先确认原进程已结束。
last.pt 留在 checkpoint 目录，无需下载。

训练、评价、诊断结束，报告自动复制到
`/root/autodl-tmp/download/babel_ae_ema_long_s42/`。
回传 manifest.json、history.jsonl、warm_start_validation.json、ae_metrics.json、
ae_diagnostics.json。原阶段 2 说明文件描述源实验，本文件描述延长实验。
