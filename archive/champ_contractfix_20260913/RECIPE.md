# 冠军模型档案 · contractfix（2026-09-13 扶正）

## 身份

- 代码快照：`code_champ_contractfix.tar.gz`（本目录，即 E3 补丁应用前的 src/scripts/tests/docs）
- 模型权重（在 AutoDL）：`checkpoints/contractfix_s42/best.pt`、`checkpoints/contractfix_s7/best.pt`
  - AutoDL 备份：`champ_contractfix_backup_20260913.tar.gz`、`models/champ_contractfix/{s42,s7}.pt`
  - 生产指针：`models/best.pt` = contractfix_s7

## 训练命令（复现用）

```bash
PYTHONPATH=src python -u scripts/train_multi_symbol.py --task classify \
  --label-anchor day_close --theta-mode dynamic --theta-q 0.90 --soft-label \
  --close-path --periods 60 30 --daily-bars 20 --foreign-bars 20 \
  --contract-mode \
  --batch-size 256 --lr 1e-3 --epochs 40 --patience 8 --seed 42 \
  --save-dir checkpoints/contractfix_s42
# seed 7 同式
```

## 关键口径

- 数据：13 品种（rb hc i sr p j jm m y cu ag TA MA）× 60/30m，合约模式（1/5/10 月轮转，自建连续序列）
- 输入：[日K×20][外盘日K×20][分钟主窗口一周]，close_path 锚定
- 标签：当日收盘锚定 first-passage 三分类，θ_q=0.90 动态轨，θ 校准按段独立（本版修复点）
- 损失：0.5×硬CE(类权) + 0.5×软CE(v2)；无 path_aux（E3 前）
- 选模：验证集 mean_edge，早停回滚最佳

## 判决成绩（2026-09-13）

- 验证 mean_edge：s42 +0.0905 / s7 +0.1281
- 单仓 v2：+7.30%（80 笔，胜率 51.2%）；成本加倍 +3.51%
- p 品种 +6.31%（胜率 59.1%）；已知短板 m 成本加倍 -0.45%

## 纪律（用户 2026-09-13 立）

任何后续改动不得丢失冠军的脚本与架构；仅靠参数开关无法实现的改动，
必须先把当前代码打包存档（放入 archive/）再继续修改。
