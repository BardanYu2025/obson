# obson · K 线形态理解骨干（E12）

训练"看懂 K 线"的表示模型：per-bar 因果 Transformer，稠密监督结构身份
（段方向/枢轴/坐标/成色），每根 bar 输出上下文 embedding，作为下游任务
（形态检索、gate、复盘工具）的预训练骨干。**不做未来 K 线的点预测**——
该范式已证伪封存（分支 `archive/e3v11-champion`，档案见 archive/）。

## 当前状态（2026-09-17）

- **E12-R1 骨干已毕业**：A1 枢轴容差 F1=0.630（margin +0.084）、
  A2 段方向 BA=0.792、U4-U6 全过（判决档案 docs/PRODUCTION.md）
- **M4 形态检索两轮未过门禁**（R2=0.662 < 0.70 线）→ 瓶颈在骨干容量，
  下一步 E12-R2 扩容（hidden 256 / 6 层 / +15m 数据）

## 快速开始

```bash
pip install torch pandas numpy tqsdk

# 1. 数据（天勤账号）→ 2. 打标签 → 3. 训练
export TQ_USER=<账号> TQ_PASS=<密码>
PYTHONPATH=src python -u scripts/download_tqsdk_v2.py --symbols rb --periods 60
PYTHONPATH=src python -u scripts/build_bar_labels.py --contract
PYTHONPATH=src python -u scripts/train_pattern.py --contract \
  --symbols rb hc i sr p j jm m y cu ag TA MA --periods 60 30 \
  --window 256 --stride 25 --batch-size 64 --lr 3e-4 \
  --epochs 30 --patience 6 --seed 42 --save-dir checkpoints/e12_s42

# 4. 检索：建库 → 查询
PYTHONPATH=src python -u scripts/e12_index.py --ckpt checkpoints/e12_s42/best.pt \
  --symbols rb hc i sr p j jm m y cu ag TA MA --periods 60 30
PYTHONPATH=src python -u scripts/e12_query.py --ckpt checkpoints/e12_s42/best.pt \
  --index data/index/e12_index.npz --code rb --period 60
```

## 文档地图

- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) —— 项目总览（架构/数据/训练/评价/证伪清单）
- [docs/STATUS_REPORT.md](docs/STATUS_REPORT.md) —— 阶段任务汇报
- [docs/E12_PATTERN_BACKBONE_DESIGN.md](docs/E12_PATTERN_BACKBONE_DESIGN.md) —— 骨干设计稿
- [docs/BAR_LABELS.md](docs/BAR_LABELS.md) —— 标签宪法
- [docs/E12_M4_RETRIEVAL_DESIGN.md](docs/E12_M4_RETRIEVAL_DESIGN.md) —— 检索设计稿
- [docs/PRODUCTION.md](docs/PRODUCTION.md) —— 判决档案

纪律：一次一变量；先承诺门禁线再跑数；改动前固化档案；archive/ 只增不删。
