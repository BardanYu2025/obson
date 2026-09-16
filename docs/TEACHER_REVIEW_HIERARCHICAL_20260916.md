# 层级任务与方向分支实验汇报

> 供教师模型评审。日期：2026-09-16。代码分支：`feature/klm`。
> 本文记录 `eef419e` 至 `caa57ff` 的相关实现、诊断和实验判决。
> 现役生产冠军没有被替换，仍为 E3 v1.1。

## 1. 背景

当前生产模型 E3 v1.1 使用合约模式、13 品种、60m/30m 混训，输入为日 K、外盘日 K 和分钟窗口，标签为当日收盘锚定的 first-passage 三分类（空/无/多），动态 `theta=c*sigma_window*sqrt(remaining_bars)`，`theta_q=0.90`。主损失是硬 CE 与软 CE 的 0.5/0.5 混合，另有 E3 路径状态辅助头。

本轮疑问是：三分类中约 90% 为“无”，是否可以拆为：

```text
Task 1: 当前窗口是否存在值得交易的机会？
Task 2: 如果值得交易，方向是多还是空？
```

目标是检验机会识别与方向识别能否分开学习，也探索 gate 是否能作为冠军信号的过滤器。

## 2. 任务定义

### 机会 gate

根据生产止盈止损规则计算归一化效用：命中目标方向为 `+0.8`，先触反向止损为 `-0.5`，未触轨则保留到锚点的实际收益。定义：

```text
gate_target = 1[max(short_utility, long_utility) >= 0.80]
```

这个标签表示“至少存在一个方向达到完整止盈级别”，不表示冠军当前喊出的方向必然正确。

### 方向

试过两种标签：

```text
utility:      argmax(short_utility, long_utility)
close_return: 1[fwd_ret >= 0]
```

层级公开概率原本按以下方式组合：

```text
P(空)=P(gate)*P(空|gate)
P(无)=1-P(gate)
P(多)=P(gate)*P(多|gate)
```

后续证据表明，条件方向分支没有稳定的可泛化信号，不能这样接管生产输出。

## 3. 代码实现与诊断

共享 pooled 表示后增加两塔：

```text
pooled -> opportunity_tower -> gate_head
pooled -> direction_tower  -> direction_head
```

支持三种阶段：`joint` 联合、`gate` 只训练 gate、`direction` 只训练方向、`direction_ft` 训练方向并解冻最后 N 层 encoder。默认不启用，E3 冠军行为不变。

方向分支新增诊断开关：

- `--hier-direction-active-only`：训练 loader 仅保留 `gate=1` 样本；
- `--hier-direction-all`：对全部样本训练 close-return 方向，`gate=0` 行弱权重；
- `--hier-direction-none-weight`：全样本模式中 gate=0 行的默认权重为 0.2。

训练/评估修复：`direction` 阶段按方向 loss 选模；`gate` 阶段按 gate loss 选模；验证日志增加 gate accuracy、balanced accuracy、正例率、top10% precision；方向日志增加 pooled/logit 方差、梯度范数、参数首步更新；测试增加纯 gate 指标。

## 4. 数据审计

contract 数据集审计（p/sr）结果：

```text
p train/val/test gate = 12.77%/13.98%/12.42%
p active short/long  = 50.89/49.11, 47.34/52.66, 40.48/59.52%
sr train/val/test gate = 12.17%/11.90%/14.12%
sr active short/long  = 53.00/47.00, 42.50/57.50, 45.55/54.45%
```

`item_match=True`：数据集数组标签与 `__getitem__` 标签一致。当前没有证据表明失败来自标签错位、方向类别完全失衡或 split 错误。

## 5. 方向实验记录

### H1 联合层级训练

提交 `eef419e`，后续日志修复为 `debbd58`。方向 BA 约 0.500，预测趋向单一类别；gate 有弱排序迹象，但硬级联没有超过冠军。

### H2 冻结 encoder 的方向训练

`hier-stage=direction`，仅训练方向塔/头，方向 loss 只取 gate-positive 行。结果方向 loss 约 0.66~0.70，BA 约 0.500。

### H3 解冻最后两层

提交 `52880f3`，`direction_ft`、最后 2 层、constant `lr=1e-4`。仍然 `dirBA≈0.500`，测试方向接近单一类别。

### H4 换 close-return 标签

提交 `5dec686`。active short/long 约 50/50，但方向 BA 仍约 0.500，说明不是 utility 标签特有的问题。

### H5 代码诊断

提交 `008c8c3`。有效短跑的首 batch：

```text
active=3/64, target0=3, target1=0
pooled_std=1.00047529
logit_std=[0.0227423, 0.0255218]
pLong_std=0.01084827
grad_l2=4.38132461
param_delta_l2=0.03604919
```

因此 pooled 表示、方向前向、梯度和参数更新均正常，失败不是显然的代码断路。

### H6 active-only

提交 `637df4a`。训练样本从 p=6169 降为 788、sr=6169 降为 750；首 batch 为 `64/64` 方向样本，标签 `34/30` 或 `64/64`。20 epoch 后方向 loss 仍约 0.69，验证 `dirBA≈0.500`，测试在全空/全多间漂移。判决：样本稀疏不是唯一原因。

### H7 全样本方向

提交 `85e15e4`。对全部样本训练 close-return 方向，gate=0 权重 0.2，新增 `allBA` 和 `dirBA`。结果：`allBA≈0.500`、`dirBA≈0.500`、loss 约 0.689~0.697，测试 p=-0.042%、sr=-0.020%。判决：方向塔不进入生产。

## 6. Gate 全量训练

提交 `9e4a318` 修复 gate 阶段选模/验证指标，提交 `caa57ff` 增加纯 gate 测试输出。配置为 13 品种 × 60/30m、contract、day_close、dynamic theta、q=0.90、seed42、全量样本训练 gate，3 epoch 冒烟约 24 分钟。

验证 gate 正例率约 11%~17%，代表性结果：

| 组合 | gate 率 | balanced accuracy | top10 precision |
|---|---:|---:|---:|
| rb_60m | 13.9% | 0.601 | 24.3% |
| hc_60m | 12.9% | 0.599 | 27.1% |
| sr_60m | 11.9% | 0.623 | 23.7% |
| p_60m | 14.0% | 0.566 | 30.4% |
| m_60m | 17.2% | 0.618 | 26.7% |
| y_60m | 12.0% | 0.632 | 26.7% |
| MA_60m | 13.2% | 0.675 | 34.1% |

验证集显示 gate 可能学到机会排序，尤其是部分 60m 组合。但普通 accuracy 不应作为主要指标，因为类别不平衡。

### 测试结果的限制

3 epoch 日志的测试级联使用了未训练的随机方向头，所以其中多空数量与收益不能评价 gate。纯 gate 测试打印已在 `caa57ff` 加入；截至本文生成时，AutoDL 尚未返回这次复评结果。因此 gate 的最终测试泛化尚未判定。

## 7. 当前判决

已经可以确认：

1. 当前输入、标签和训练方式下，条件方向没有稳定信号；
2. active-only 不能挽救方向；
3. 全样本 close-return 方向也没有超过随机；
4. gate 在验证集具有弱到中等的排序信号，值得做一次纯 gate 测试复评；
5. gate 不应与随机方向头级联，也不能把附带的多空回测当成 gate 成绩；
6. E3 v1.1 仍是生产冠军，层级模型无资格替换它。

不能过度声称：验证 top10 不是独立测试成绩，gate 普通 accuracy 不是有效能力证明，也不能证明 gate 对所有品种有效。

## 8. 请老师重点审查

1. `max(short_utility,long_utility)>=0.8` 是否是稳定机会标签，还是主要在识别波动率/剩余时间？
2. gate 是否应采用 walk-forward 与按交易日 block bootstrap，而不是单一时间切分？
3. gate 测试是否应报告 PR-AUC、precision@k、收益 uplift、Brier/ECE 和跨品种汇总？
4. gate 是否只能作为冠军信号的风险过滤器，而不是生成三分类概率？
5. 若 gate 只对 p/m/y 或 60m 有效，是否按品种×周期白名单启用？
6. 是否先完成纯 gate 测试，再冻结 top10/20/30% 阈值做 E3 冠军过滤组合回测？

## 9. 提交索引与下一步

```text
b72d18f hierarchical dataset audit
debbd58 hierarchical loss logging
eef419e staged hierarchical training
48d7d04 direction stage selection/diagnostics
52880f3 fine tune encoder for direction
5dec686 close-return direction labels
008c8c3 direction collapse diagnostics
637df4a active-only direction diagnostic
85e15e4 all-sample direction probe
9e4a318 correct gate evaluation/selection
caa57ff pure gate test metrics
```

下一步只做：用 `caa57ff` 对 `checkpoints/hier_gate_full_s42/best.pt` 做纯 gate 测试；若测试仍有 uplift，再在验证集预注册 gate top10/20/30%，对 E3 冠军做过滤组合回测。组合至少报告交易数、覆盖率、命中率、单笔收益、单仓收益、最大回撤和成本加倍结果。组合通过前，生产继续使用 E3 v1.1。
