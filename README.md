# obson · 期货 K 线 Transformer 交易信号模型

基于 Transformer 的商品期货盘中信号模型：13 品种 × 60/30m 混训，
当日收盘锚定 first-passage 三分类（先摸上轨=多 / 先破下轨=空 / 都没=无），
路径状态 + excursion 辅助监督，双种子集成，合约模式数据管线。

## 现役冠军：E3 v1.1（2026-09-14）

| 指标 | 成绩 |
|---|---|
| 验证 mean_edge | s42 +0.1003 / s7 +0.1262 |
| 单仓 v2 回测 | **+10.06%**（58 笔，胜率 58.6%，回撤 1.70%） |
| 成本加倍 | **+7.24%** |
| 路径辅助头 | 节点 BA 0.38~0.42（存活，非塌缩） |

模型权重不进本仓库（387MB）。获取方式：训练复现（命令见
[archive/champ_e3v11_20260914/RECIPE.md](archive/champ_e3v11_20260914/RECIPE.md)），
或从已有部署环境拷贝 `checkpoints/e3v11_s42/best.pt` + `checkpoints/e3v11_s7/best.pt`。

## 结构

```
src/obson/            模型与数据管线（dataset / transformer / mixed_trainer / playbook）
scripts/              训练 / 回测 / 实盘信号 / 数据下载
tests/                生产回归测试（26 条）
docs/                 生产文档（PRODUCTION.md 为总文档）
archive/              历代冠军档案（代码快照 + 配方 + 判决成绩）
```

## 快速开始

```bash
pip install torch pandas numpy tqsdk
python tests/test_production.py        # 26/26 应通过

# 夜盘一键信号（数据增量 + 外盘 + 出信号）
export TQ_USER=<天勤账号> TQ_PASS=<天勤密码>
./night_signal_autodl.sh rb sr p
```

训练、回测、判决口径详见 [docs/PRODUCTION.md](docs/PRODUCTION.md) 与
[archive/champ_e3v11_20260914/RECIPE.md](archive/champ_e3v11_20260914/RECIPE.md)。

## 实验族谱

```
q90_softfix → contract → contractfix → E3 v1 → E3 v1.1（现役）
→ E4 excursion ❌弃案（与主任务语义冲突，成本加倍转负）
→ E5 双向 encoder（进行中）
```

纪律：一次一变量；选模只看验证 mean_edge；判决看生产口径
（单仓 v2 / 成本加倍 / p 品种 / 信号不塌缩）；改动前先固化冠军档案。
