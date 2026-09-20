# 阶段对照与独立短时状态实验

先运行 audit，再决定是否运行 short。不会自动开始新训练，不修改原大模型权重。

## 阶段对照

读取 large512 的 local/aggregate/joint 最佳权重。检查 aggregate 的局部权重与 local 逐张量完全相同，
所以 aggregate/local_metrics 就是第一阶段局部模型的评价，joint/local_metrics 是最终局部模型。
两个阶段用相同局部/层次窗口、相同指标、相同 teacher 和相同训练集归一化参数。
额外拆分验证与测试的全局重建、局部重建、teacher、价格锚点和历史结构损失。
拆分结果先在每个窗口内平均，再跨窗口平均，避免评价 batch 和有效片段数改变报告权重。
与旧训练日志的 micro-batch 内跨片段平均不完全同口径，不直接比较总值。
默认局部评价 batch64、层次评价 batch16；只影响推理，不改训练 manifest。
原最佳权重哈希在开始和结束都校验。

## 短时状态第一版

18维因果特征 → 输入投影 → 2层512宽度GRU → 每根bar一个512维状态。
位置查询解码器只接收末端状态，重建过去64根7通道；16/32/64三个历史尺度等权重。
目标价格相对64根窗口之前的收盘价，解码不读取原始输入或价格锚点。
完整价格恢复额外使用一个显式价格锚点。无未来监督。
512维状态大于64×7=448个目标标量；本实验检验近期信息保存与在线更新，
不能仅凭64根重建成功声称实现高压缩率或证明长期记忆容量。

batch32 表示32条不同的合约/周期序列同时运行。每条序列内部保持时间顺序。
每128根截断梯度，但传递隐藏状态值；只打乱训练序列顺序，不打乱序列内片段。
每个合约、周期、时间分区和 epoch 起点重置状态，不在不同合约之间串接。
padding 只在序列结束后出现，该 lane 不再产生任何有效评价端点，也不接入新合约。
沿用原128根历史数据集的全部端点；先在当前分区内累计至少128根才评分。
每个 step 的有效端点数不同，记录 batch_streams 而不把它冒充固定样本 batch。
训练默认30轮、AdamW lr3e-4、weight_decay=.01、clip1；验证选择最佳权重。
完整epoch末保存优化器、随机状态和权重，中断从上一个完整epoch恢复。
改变 batch/预算须新实验目录，不能在原 manifest 上伪装精确续训。

这是独立短时状态实验，尚未融合2048根长期记忆；不会宣称已经完成长短期联合模型。
冻结的第一阶段 local512 在同一端点重建最后64根并作状态读出对照。
GRU可读取分区内更早历史，local仅128根，因此不是纯架构消融；明确披露上下文不同。
结果输出16/32/64多尺度重建诊断和简单线性状态读出，不以测试集挑轮次。
测试结果已反复用于研究决策，未来预测实验须另外预留未参与研发的时间区间。

## AutoDL 命令

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_STATE_LOG=logs/babel_audit.log
mkdir -p logs
nohup bash scripts/babel_state_autodl.sh audit > "$BABEL_STATE_LOG" 2>&1 &
tail -f "$BABEL_STATE_LOG"
```

audit结果自动复制到 `/root/autodl-tmp/download/babel_large512_audit/`。
其中 aggregate 和 joint 子目录分别保存指标与图，顶层 stage_comparison.json 保存拆分损失。
不复制 teacher_cache 和权重。

阶段对照看完后，独立启动短时状态：

```bash
export BABEL_SHORT_BATCH=32
export BABEL_SHORT_RUN=checkpoints/babel_short512_s42
export BABEL_STATE_LOG=logs/babel_short.log
nohup bash scripts/babel_state_autodl.sh short > "$BABEL_STATE_LOG" 2>&1 &
tail -f "$BABEL_STATE_LOG"
```

报告复制到 download/babel_short512_s42。short-evaluate 仅评价已有短时状态最佳权重。
OOM 时根据实际日志调整 batch；没有在本地运行CUDA，不承诺默认批次在所有显卡都可用。

原大模型新实验也允许分别用 BABEL_LARGE_LOCAL_EFFECTIVE / AGG_EFFECTIVE / JOINT_EFFECTIVE
指定有效batch，默认32/16/8；对应MICRO默认仍16/2/1以兼容旧manifest。
若只减少梯度累积，新实验可用micro32/16/8而保持有效batch不变。
旧实验更改micro或effective会被一致性检查拒绝，需后续显式续训迁移，不能覆盖旧记录。
