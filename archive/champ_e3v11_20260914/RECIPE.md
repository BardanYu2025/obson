# 冠军模型档案 · E3 v1.1（2026-09-14 扶正）

## 身份

- 代码快照：`code_champ_e3v11.tar.gz`（E4 改动前的 src/scripts/tests/docs）
- 模型权重（AutoDL + 用户本地已下载）：`checkpoints/e3v11_s42/best.pt`、`checkpoints/e3v11_s7/best.pt`
  - 备份包：`champ_e3v11_20260914.tar.gz`（用户已下载回本地 2026-09-14）
  - 生产指针：`models/best.pt` = e3v11_s7

## 训练命令

```bash
PYTHONPATH=src python -u scripts/train_multi_symbol.py --task classify \
  --label-anchor day_close --theta-mode dynamic --theta-q 0.90 --soft-label \
  --close-path --periods 60 30 --daily-bars 20 --foreign-bars 20 \
  --contract-mode --path-aux --path-aux-weight 0.1 \
  --batch-size 256 --lr 1e-3 --epochs 40 --patience 8 --seed 42 \
  --save-dir checkpoints/e3v11_s42
# seed 7 同式
```

## 与前任（contractfix）的唯一差异

E3 路径状态辅助头：pooled + node_emb(4) → 逐节点 3 态分类，
masked CE（同根双触整行 mask）+ **逐节点类别权重**（v1.1 修复塌缩），λ=0.1。

## 判决成绩（2026-09-14）

- 验证 mean_edge：s42 +0.1003 / s7 +0.1262
- 单仓 v2：+10.06%（58 笔，胜率 58.6%，单笔期望 +0.1735%，回撤 1.70%）
- 成本加倍：+7.24%（历代最佳）
- p 品种 +8.62%（胜率 65.6%）；sr 缩水至 5 笔 +0.76%；m 成本加倍 -0.37%（短板依旧）
- 路径头：节点 BA 0.38~0.42（>0.333 基线），单调违规 ≤1.1%

## 族谱

q90_softfix → contract → contractfix → E3 v1 → **E3 v1.1（现役）**
