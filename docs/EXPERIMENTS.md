# Obson 实验日志 — 2026-09-01

## 项目状态速览（读完这一节就能接着干活）

- 模型：KLineTransformer v12，hidden 256，4 层 GQA(8Q/1KV)+ALiBi(时间感知、斜率可学)+因果 conv stem。
  参数量 892,132(3 指标版）/ 892,388(tech6 版，差 256=(7-3)×64，即 technical_proj 第一层)。
- 任务配置：`--task classify --label-anchor day_close --theta-mode dynamic`，
  三分类 first-passage 标签，θ = c×窗口σ×√剩余bar（训练段校准 c≈1.45~1.51），尾盘 30min 剔除。
- 数据：`data/{rb,hc,i}_{5,15,30,60}m.csv`，天勤主连，每个 csv 恰好 8964 根（**免费版单次请求上限，不是历史上限**）。
  起点：5m=2026-02(仅~6.5个月，最短板) / 15m=2025-01 / 30m=2023-07 / 60m=2021-05。
- 训练机在 AutoDL：`~/autodl-tmp/obson`；本地打包目录 `/Users/bardan/private/obson`。
- 包：`obson_20260901.zip`(3指标+新CLI) / `obson_20260901_tech6.zip`(7指标)。
  **tech6 与旧 checkpoint 不兼容**(technical_proj 3→7 维)。
- 新 CLI：`--lr --hidden --layers --heads --dropout --save-dir --eval-ckpt`(eval-ckpt=不重训直接评估任意存档)。
- val epoch 日志新增 `r+/r-` 列 = 该 epoch 模型喊多/喊空后持有到收盘的平均收益%，纸面经济价值。
- **加权 CE 的先验基线 ≈1.376**(类别权重 [2.08~2.31, 0.479, 2.08~2.31] 下按定义 p_c·w_c=1/3)。
  日志里"先验分布基线 CE≈0.65"是未加权熵，**不是**同一把尺，别拿 train/val CE 跟它比。

## 实验记录(均为 day_close、θ-q=0.70、time 切分 70/15/15)

| run | 代码包 | 配置 | best val BA(轮) | test BA | 关键诊断 |
|---|---|---|---|---|---|
| old | 3指标 | 12组合, 80ep@3e-3 | 0.4714(ep41) | 0.368 | 背题: 5m train CE 0.24, val CE 1.06→1.6 爆炸被 BA 掩盖 |
| C_v1 | tech6 | 30m-only, 18ep@1e-3/p6 | 0.3555(ep16) | 0.369 | 欠训练: 被 epoch cap 掐停(patience 才 2/6), OneCycle 1.8 轮就过峰 |
| C' | tech6 | 30m-only, 50ep@1.5e-3/p10 | 0.3805(ep19) | 0.379 | 欠训练修复验证 rb_30m: 0.374→0.411, r+ 转正 +0.050% |
| A' | tech6 | 12组合, 40ep@1e-3/p8 | 0.4355(ep25) | 0.356 | **ep25 又已背题**(5m train CE 0.41~0.49, val CE 2.1~2.3, avg 1.33 逼近 1.376 基线) |

test BA 组合明细:
old：rb 0.274/0.383/0.400/0.376, hc 0.290/0.442/0.386/0.369, i 0.366/0.357/0.371/0.397
A'：rb 0.318/0.354/0.427/0.359, hc 0.268/0.338/0.409/0.374, i 0.353/0.284/0.397/0.393

## 已坐实的结论

1. **裸 val BA 选模不可靠——两次选中的都是背题 checkpoint**(old ep41、A' ep25)。
   规律：val loss 见底(mean≈1.06，约 ep 7~12 @1e-3)后回升、而 mean BA 仍爬升 = 背诵进行中。
   以后选模看 val loss(或 BA∩loss 联合、或对 BA 序列做 EMA 再取 max)。
2. **混频迁移是真的**：同一三个 30m 组合，混训 test 0.411 vs 单练 0.379。放弃"只用 30m"路线。
3. 欠训练→多练的修复成立(C_v1→C')。
4. **唯一跨三个 run 存活的正期望信号：rb_30m 多头**
   (r+: +0.002% → +0.050% → +0.026%，覆盖率 37~58%)。
   rb/hc 空头在测试期全面是反指(r- 为正=喊空后上涨)；i 品种信号全线不值钱。
   注意测试期本身是上涨行情(负类仅 8~16%)，"空头不行"部分是 regime 假象。
5. 技术指标(7个: ma_position/vol_regime/range_position/rsi/macd_norm/boll_squeeze/roc20)
   在窗口内因果计算，前 ~26 根暖机失真；60m 窗口仅 35 根 → 74% 受影响。
   断点密度实测：60m ~65~76 处、平均段长 116~136 根(~18 交易日)，38~51 段不足 60 根。
   正确修法=断点分段+段内重置预计算.dataset 改 + transformer technical 分支变直通(未实施)。

## 待办(按优先级)

1. 【零成本】评估 A' 健康期 checkpoint：`checkpoints/full_tech6/epoch_10.pt`(服务器上存在),test 上若
   ≥ ep25 的 best → 坐实"BA 选模有害"，改用 val loss 选模。
   命令：`--eval-ckpt checkpoints/full_tech6/epoch_10.pt` + 训练同款 --task/--label-anchor/--theta-mode。
2. 选模指标改造：换成 avg val loss(或 BA×loss 约束、EMA-BA),改 mixed_trainer.py 的 selection_score。
3. `--theta-q 0.85` 稀疏信号：信号减半、单笔期望抬高，rb_30mrounded 为首要观察对象。
4. 只做多边 rb_30m 的策略草案(rb/hc 空头按反指处理或弃用)。
5. 数据侧：下载脚本改 end_dt 翻页，把 5m 从 6.5 个月拉到几年(所有频率一起重拉保持对齐)。
6. 段内重置的技术指标预计算(待 1~3 证明方向后再投入)。
7. 夜盘反转审计：hc_5m 夜盘 bal_acc 曾 0.140(≈8σ 反向),先排交易日映射 bug，再在 val 验证反指利用。
8. 若 θ 稀疏化后 r+ 仍上不了 0.08~0.1%(成本线),换任务：次日收盘 horizon 或回归+IC+尾部交易。

## 备忘

- OneCycle 峰值在总步数 10% 处：epochs=18 → 1.8 轮过峰(欠训练元凶);epochs 与 lr 要成对调。
- 每 10 轮自动存档 `epoch_N.pt` → 配合 `--eval-ckpt` 可免费回放任意训练阶段的测试表现。
- checkpoint 目录用 `--save-dir` 隔离各 run;unzip 新包不会覆盖 checkpoints/。
- 测试期(最后 15%)呈上涨趋势：负类 8~16%、正类 14~24%,所有"空头不行"的结论带 regime 折扣。
