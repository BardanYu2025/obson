# 冻结表示的当前状态读出与刷新对照

目标：判断512维短状态、512维长状态及1024维组合能否有效读出当前行情结构，
是否在价格/均线基线上提供额外信息，以及每128根刷新长状态是否造成损失。
不更新编码器或重建头，不做下一根价格预测。

## 固定实验协议

- 使用已验收的`babel_stream1024_numeric/inference_bundle.pt`和原fusion缓存；校验来源hash。
- 沿用原时间划分，验证/测试GRU独立从各自分区起点重置；同一合约按时间输入，不拼接换月。
- 原合约此前的因果归一化/EMA和规则标注状态可以预热，但神经记忆不跨分区。
- 以原缓存完整128根端点为基准，在年龄0、16、48、96、127采样；当前bar必须仍在同一分区、
  满足原滞后主力选择条件。不因后续采样点缺失而删除当前点，不用未来标签筛选。
- 年龄分组0、1–31、32–63、64–127；只代表这些固定采样年龄，不宣称全量逐bar。
  每组样本数与类别分布单列；组间难度可能不同，不能从组间BA直接推断刷新因果效应。
- 标签来自`structure.annotate`在当前bar可知的已确认拐点结构，三个固定ATR尺度0.75/1.5/3.0。
  四类为未形成、上升结构、下降结构、区间震荡；最后一类是既有规则名称，也包含高低点关系混合情况。
  不回填极值时点，不使用未来收益。这些是规则参考标签，并非市场真理或人工共识。
- 读出方法：同一类加权线性ridge分类器，训练集均值/标准差、训练类别权重；
  alpha固定1/10/100/1000，仅按验证集BA选择，各表示/尺度独立选择。截距不惩罚。
  全部模型都附加同一个可观测年龄标量。只衡量线性可读信息，不等于向量的信息上限。

## 一次运行的实验矩阵

| 表示 | 输入 |
|---|---|
| price_ema | 25维价格形态、因果EMA、4/16/32/64/128/256收益、16/64/256路径效率和区间位置 |
| short | 当前短状态512 |
| long_held | 最近完整128根网格端点的长状态512 |
| dual_held | 当前short + 保持不变的long，1024 |
| price_dual_held | 价格基线 + 双状态；衡量embedding的增量价值 |
| long_fresh | 使用截至当前的行情重新计算long |
| dual_fresh | 当前short + fresh long |
| price_dual_fresh | 价格基线 + fresh双状态 |

价格基线不包含标签、已确认拐点位置或规则状态。另报告训练多数类、
上一long更新时间的**精确规则状态保持不变**两个对照；后者在age0天然100%，
用于衡量更新之间规则状态本身的变化，不作为学习模型的输入。
完整规则计算器按定义能100%复现这些标签，因此本实验不能证明模型优于规则计算器。

fresh与held在同一时点配对，使用同一组冻结权重、相同块数n（4–16）、相同128根块长。
fresh将n个完整历史块重新对齐到当前bar，held保留原网格。这是**刷新与窗口对齐策略对照**，
不是把不足128根的局部块塞进原模型，也不是只改变一个时间戳。
age0直接复用原缓存；非零年龄的局部块按合约/结束行去重批量编码，再聚合。
测试另固定held读出头，替换fresh向量，不重新拟合；同时报告fresh独立训练读出结果。
两者可区分直接替换的分布变化与重新适配的收益。

## 评价与范围

- BA、各类召回/精确率/支持数、混淆矩阵；按年龄、品种、周期分组。
- 关键对照报告同样本BA差值及1000次**按日历周整体重采样**的区间，
  同周所有品种/周期一起抽取，避免将重叠bar当作独立证据。
  少于5周时不提供区间。区间只是研发描述性结果，不作多重比较校正后的显著性承诺。
- 测试期已被多轮研发查看，不是新holdout；本轮只在验证集挑选alpha。
- 对当前状态有用不代表预测未来有用，更不代表扣费后交易盈利。
- 初始路线决策以中尺度为主，并检查另外两尺度与品种一致性；不只挑最高的一格。
  若price_dual相对price没有稳定增量，应如实判断现有向量未体现这项任务的额外价值。
  若fresh改善有限，保留便宜的128根刷新；若paired改善明显，再单独设计更频繁的线上刷新实现。

## AutoDL一条任务

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_STREAM_BUNDLE=checkpoints/babel_stream1024_numeric/inference_bundle.pt
export BABEL_READOUT_RUN=checkpoints/babel_state_readout1024
export BABEL_READOUT_LOG=logs/babel_state_readout1024.log
export BABEL_READOUT_BATCH=128
export BABEL_READOUT_STREAM_BATCH=128
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_readout_autodl.sh all > "$BABEL_READOUT_LOG" 2>&1 &
tail -f "$BABEL_READOUT_LOG"
```

按顺序完成数据校验、一次共享特征提取、全部读出拟合与对照、报告打包。
两种batch都是**推理提取批量**，不是训练batch或epoch；无encoder optimizer。
局部Transformer在GPU批量推理，ridge在CPU做小矩阵求解；阶段性GPU空闲是预期行为。
按split持久化缓存并校验hash，原命令重跑可复用完整split；未完成split会重算。
不能同时用两个进程写同一输出目录。改变batch/实验配置须使用新的输出目录。

成功标志：`State readout experiment complete`，随后`run_status=complete`。
成功或失败都自动将报告复制、打包到：
`/root/autodl-tmp/download/babel_state_readout1024_reports.tar.gz`。
上传这一个包即可，内含state_metrics.json、summary.md、逐样本预测、配置、数据索引与日志。
排除.pt、.npz权重/向量缓存；源checkpoint不改动。

另开终端重新打包默认目录：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_readout_autodl.sh export
```

本地只做合成数据功能测试与无梯度推理；真实512模型的吞吐/显存与结果由AutoDL验证。
