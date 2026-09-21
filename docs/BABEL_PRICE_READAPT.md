# 固定量仓状态，重新适配价格解码器

## 阶段问题

量仓辅助权重0.20在固定第100轮已经跨种子提高历史信息读出，并超过当前输入对照，但收盘重建较同轮control退步3.0%～3.8%。本轮检验：冻结这些表示，给予共同来源解码器同样的适配预算，价格表达能否恢复？不改变编码器容量、特征或量仓目标，不把失败解释为信息不存在。

## 四组匹配实验

- 使用activity_alignment100的control/aux020各seed42/43，全部来自last.pt和其manifest声明的最终轮次（本次100）。要求已有完整last_diagnostic。
- 逐合约、逐分区顺序回放编码器，缓存原端点的512维状态；验证/测试分数核对原报告，保存replay.json和缓存hash。输入状态无梯度，训练进程不创建编码器或量仓辅助头。
- 四组同一价格解码器架构：512到256，两层8头Transformer解码64根历史，来源固定为上游reconstruction source/short/best.pt。这是共同的预训练解码器，非随机初始化；不能称为从头独立复现。
- 每组100个解码器训练epoch，batch256，默认2个并行worker。每轮完整遍历相同训练端点，端点顺序按seed+epoch确定；这是固定状态解码训练，不是打乱原始行情生成状态。
- 新建仅含解码器参数的AdamW，weight_decay0.01，梯度裁剪1。学习率按绝对适配epoch余弦下降，第1轮1e-4，第100轮1e-5。复跑恢复解码器、AdamW、RNG、完整history；不会重新开始。
- 沿用base+0.25×detail损失、训练集尺度。base仍含原7通道及原价格变化约束，没有偷偷改为纯价格或额外加入趋势标签/量仓辅助头损失。
- 预算固定不等于收敛。报告每轮训练/验证、学习率、显存和耗时，用于检查是否还在改善。

## 验证选模和完整报告

每轮（含epoch0）只用验证选模。相对同种子固定的父30轮control最佳验证结果，close<=1.02、base<=1.05、change16MSE<=1.05、detail<=1.02。合格者以detail、base依次最小为准。没有任何合格者时保留最低detail作诊断，明确qualified=false；epoch0或失败回退不能算训练成功。

同时评价四类结果：原第100轮编码器配自己的旧解码器、验证所选新解码器、固定末轮新解码器、固定父control参考。分别报告，不用测试在它们之间选赢家。所有组保存合格状态、每窗口指标、固定样例图以及按周配对的价格/detail差异。

额外记录打乱测试窗口状态后的所选解码器误差，用于检查恢复是否仍依赖对应状态，而不是仅输出平均曲线。训练/验证过程中不使用该打乱测试对照。

编码器没有改变，因此量仓和状态读出复用经hash校验的last_diagnostic证据，明确标注“继承证据”，不重复训练读出头冒充独立复现。最终筛选要求两种子都通过验证资格、所选解码器实际训练过、相对适配后control及固定父参考的价格/detail保留，且原量仓/状态保留成立。不会因control末轮本身退步而降低唯一参考线，也不会自动替换主模型。

若价格恢复，支持存在可恢复的解码适配因素；若不恢复，仅否定本轮方法和预算的恢复效果。当前用的是反复研究的测试时段，不能作为新留出泛化确认。

## AutoDL命令

需要保留完整上游checkpoint和缓存，仅有下载报告包不足以运行。使用独立输出目录。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_READAPT_PARENT=checkpoints/babel_activity_alignment100
export BABEL_READAPT_RUN=checkpoints/babel_price_readapt512
export BABEL_READAPT_LOG=logs/babel_price_readapt512.log
export BABEL_READAPT_EPOCHS=100
export BABEL_READAPT_BATCH=256
export BABEL_READAPT_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_price_readapt_autodl.sh all > "$BABEL_READAPT_LOG" 2>&1 &
tail -f "$BABEL_READAPT_LOG"
```

四组全部跑完后自动结束并导出：
`/root/autodl-tmp/download/babel_price_readapt512_reports.tar.gz`。

打包包括manifest、冻结回放审计、预检、四组全部history、初始化/选模记录、主报告、样例图和失败日志，不含.pt/.npy。失败也自动导出并标明非零退出码。重跑同一命令续训；不改epochs/batch，改动实验配置需要新目录。仅需重做评价时把all换成evaluate；手动打包用export。

此代码尚需AutoDL真实GPU运行。本地只执行合成回放、前向/反向、恢复及报告测试，不执行优化器更新。
