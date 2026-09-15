# Teacher 训练线

本分支 `feature/teacher` 是在 E3 v1.1 有效组件上的增量设计，不覆盖现役冠军资产。

## 保留内容

- causal KLineTransformer 主干
- close-path、technical features、日线/外盘上下文
- dynamic theta 和 first-passage 三分类标签
- soft CE
- E3 路径状态头及逐节点类别权重
- contract mode、交易日切分、双 seed 和生产回测 harness

## 新增内容

Teacher 头预测两个生产方向的效用，顺序固定为 `[short, long]`：

- 命中方向：`+0.8`
- 反向先触：`-0.5`
- 未触轨：用已有 `fwd_ret / theta`，方向镜像后作为持有到锚点的效用

它只使用现有标签和数据集已经计算出的 `theta`、`fwd_ret`，没有新增标签，也没有预训练数据或未来信息。

训练目标为：

```text
分类损失 + 0.15 * utility_smooth_l1 + E3 路径损失
```

效用分数以很小的权重（默认 `0.10`）加入多/空 logit，只用于改善方向排序，不替代分类概率。

验证选模使用：

```text
mean_edge + 0.10 * top-5% utility_edge
```

旧配置不启用 `utility_head`，旧模型结构和旧训练命令保持兼容。

## 首轮训练命令

建议先用现役冠军配方，单独跑 `s42`，不要同时改标签、数据和结构：

```bash
PYTHONPATH=src python -u scripts/train_multi_symbol.py \
  --task classify --label-anchor day_close --theta-mode dynamic --theta-q 0.90 \
  --soft-label --close-path --periods 60 30 --daily-bars 20 --foreign-bars 20 \
  --contract-mode --path-aux --path-aux-weight 0.1 \
  --batch-size 256 --lr 1e-3 --epochs 40 --patience 8 --seed 42 \
  --teacher --teacher-loss-weight 0.15 \
  --teacher-logit-weight 0.10 --teacher-selection-weight 0.10 \
  --save-dir checkpoints/teacher_aug0_s42
```

通过首轮 smoke 后，再用同一命令只把 seed 改为 `7`。

## 判决与回退

Teacher 训练线不是自动替换冠军的机制。必须继续使用现有生产评测：

1. 验证集报告 mean_edge、utility_edge、覆盖率、多空分项和路径头指标。
2. 测试集方案冻结后只评估一次。
3. 使用 `backtest_playbook.py --playbook-v2 --single-position` 做生产口径比较。
4. 成本加倍、最大回撤、信号数量和 p 品种表现都要检查。

如果效用头导致验证或生产表现恶化，直接把 `--teacher` 关闭，现役冠军路径不受影响。

## 设计边界

这条线故意没有引入：

- 自监督预训练
- 下一根 bar 回归预训练
- 新的标签定义
- E4 excursion 辅助目标
- 双向 encoder 或 query decoder

原因是先验证“任务相关的生产效用监督”是否比脱钩预训练更有帮助，保持一次只改变一个实验因素。
