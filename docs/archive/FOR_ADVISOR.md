# Obson 项目汇报（教师版）

> 用途：向老师汇报本轮审查→修复→重训→判决的完整闭环。
> 系统全貌见 `docs/PRODUCTION.md`（5 分钟可读完）；实验流水在 `docs/EXPERIMENTS.md`。
> 本文最后更新：2026-09-11（判决封卷日）。

## 一句话

商品期货盘中信号系统：Transformer 读多品种多周期 K 线，预测"当日收盘前先摸上轨还是下轨"
三分类（开多/开空/按兵不动），盘中实时输出信号与置信度。
**本轮完成三轮代码审查的全部修复，并按修正后的口径完整重训验证——
结论：老模型在新口径下依然最优，判决封卷，生产模型不换。**

## 系统现状（截至 9/11）

- **架构**：KLineTransformer，13 品种 × 60/30m 混训，单向 causal attention，last 读出
- **输入**：`[日K×20][外盘日K×20][分钟主窗口(一周)]`，OHLC 锚定窗口前 3 根收盘均值、按窗口波动率缩放
- **标签**：当日收盘锚定 first-passage 三分类，θ 动态 = c×窗口σ×√剩余bar；软标签 v2 给部分奖励
- **生产模型**：q90_softfix 双种子集成（s42+s7 概率平均），`models/best.pt`
- **生产打法**：手册 v2 白名单（p_60m 多 [0.35,0.50)、p_60m 空 ≥0.50、sr_60m 多 [0.40,0.50)），
  0.8θ 止盈 / 0.5θ 止损 / 收盘兜底，单仓执行

## 三轮审查修复闭环（老师提的 7+ 个问题，全部落地）

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| 1 | 回测口径混淆（信号质量 vs 资金曲线） | `backtest_playbook.py --single-position` 单仓资金曲线口径成为判决标准；"+8.6%/96笔"已标注为信号质量口径 | 测试锁定 |
| 2 | playbook 规则两处实现不一致 | `scripts/playbook.py` 唯一权威，v1/v2 统一走 `signal_mask()` | `test_no_v1_reimport` 防回归 |
| 3 | 验证/测试集 warm-up 缺失（切分点头部样本历史窗口不完整） | val/test 建在含全部历史的完整 df 上按窗口末截样，训练集不动 | 测试锁定 |
| 4 | weekly_seq_len 日历/交易日口径 | 统一交易日口径，26 个品种×周期组合实测 seq_len 不变 | `tests/seq_len_check.csv` 锁死 |
| 5 | 同根双触（高低同破）语义含糊 | 两阶段语义：训练记"无"、执行记止损；实测发生率 0.00~0.02% | `test_double_touch_*` |
| 6 | ckpt 与数据管线参数脱节 | ckpt 内嵌 data_manifest（21 项），所有脚本自动恢复 | 回归测试 |
| 7 | horizon 标签越界 | `_restrict_samples` 防护 + 测试 | 回归测试 |

**回归测试 14/14 通过**：`.venv/bin/python tests/test_production.py`

## 关键实验：按新口径重训（re_s42/re_s7）与判决

用修正后的口径**原配方重训**（架构/数据/标签/损失不动），双种子，走冻结判决协议：

| 判决标准 | 要求 | re 系 | 老冠军（同口径补测） |
|---|---|---|---|
| 验证集 mean_edge | 双种子过冠军区间 | ✅ +0.0923/+0.0980 | （区间封存值 +0.084~+0.101） |
| 单仓 v2 回测 auto（生产口径） | 累计正 且 回撤≤4.9% | ❌ −0.48% / 回撤 6.57% | ✅ **+4.56% / 回撤 3.25%** |
| 成本加倍 | 不转负 | ❌ auto −3.52% | ✅ **auto +2.73%**，argmax +0.47% |
| 冻结规则独立测试 | 白名单规则有效 | p_60m多 E=+0.167%（胜）；sr_60m多 E≈0（败） | sr_60m多 摸轨率33.3% E=**+0.182%** |

**判决：挑战者不通过，老冠军留任。**

方法论结论（汇报时重点讲这三条）：

1. **验证集选模指标（mean_edge）过线 ≠ 生产能赢**。re 系 mean_edge 双过线、argmax 口径甚至改善，
   但生产口径（auto 阈值+白名单+止盈止损）收益为负、回撤超标、扛不住成本。
   机理：warm-up 修复改变 val 分布 → top-5/100 阈值标定漂移。判决协议的"生产回测关"抓住了它——协议本身被验证有效。
2. **配方级结论**：auto 模式 2 倍成本下仍 +2.73%（单笔期望 +0.0534%），生产打法本身稳健；
   re 系 auto 转负是其自身问题，非配方缺陷。
3. **训练端/评价端口径已全对齐**：冠军训练用旧口径选模（残留不对称，mean_edge 区间不可直接比），
   但测试端双方同 harness、同测试集、零调参，判决公平。

## 已证伪档案（10 个方案，勿重试）

双向 attention、RoPE、cross 全 attention、sector 板块 token、量仓 vol/OI、fine15 细粒度上下文、
margin loss、软标签 v1、clean30 干净路径标签、strat/strat07 策略对齐标签、re 口径重训。
明细在 `docs/PRODUCTION.md` §1 挑战者判决档案。

## 已知局限（诚实声明）

1. 测试期约 3 个月，确认集仍在积累
2. 选模指标 mean_edge 与生产目标存在脱钩（本轮已实锤），复合早停分的离线验证在 todo
3. 概率未做温度校准（需独立校准集）
4. 数据量仍是最大约束：训练集约 4000~6000 样本/品种/周期

## 复核入口

```bash
# 回归测试（14 条）
.venv/bin/python tests/test_production.py

# 生产模型信号（CPU 可跑）
PYTHONPATH=src python -u scripts/signal_live.py \
  --ckpt models/best.pt --symbols p sr rb --periods 60 30

# 生产口径回测（单仓资金曲线）
PYTHONPATH=src python -u scripts/backtest_playbook.py \
  --ckpt models/best.pt --symbols p sr y m rb --periods 60 30 \
  --playbook-v2 --single-position
```
