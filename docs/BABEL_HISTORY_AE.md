# 主线：可解码的因果历史行情压缩

本轮停止把未来 bar 预测作为主要训练目标。128 根已发生行情经因果编码器，
压缩成一个 128 维 z，再从这个 z 重建同一段历史。z 挂在窗口末端 bar 上，
也是该窗口的 seq embedding。schema=`babel-history-ae-v1`；不覆盖或加载旧权重。

## 模型和输入

4 层因果 Transformer、hidden=128、4 头。每根产生上下文向量，但重建损失
只作用于完整 128 根窗口的末端 z。窗口内较早位置的表示因上下文较短，
没有同等的完整窗口重建保证。实际提取每根的固定历史表示，应逐根使用
截至该根的最近 128 根窗口。stride=16 是训练采样间隔，不是只能每 16 根推理。

输入复用 AR 因果编码：相对上一根收盘价的对数价格几何、历史收益 RMS、
volume/OI、可用性、周期与时间间隔。历史 RMS 可能含窗口前的历史，是可用
的因果上下文。它不进入解码器，也不需要作为额外逐根尺度侧信息来还原价格。

解码器只有一个 z，经线性层和固定位置编码生成 128 个查询位置，经过两层
Transformer 输出序列。没有原始 K 线跳跃连接、整串编码器隐藏状态、真值
teacher forcing 或未来输入。解码器在历史位置之间双向交互是允许的：它解码
的是整个 z 已经编码的过去，不是用来生成各历史位置的因果 h。

## 重建目标与价格锚点

每个位置七个目标：相对窗口开始前收盘价的 open/close 对数百分比、
上下影的对数百分比、log1p(volume)/10、log1p(OI)/10、log(间隔/周期)。
首个窗口之前没有记录时，用首根开盘作锚点。所有目标均来自截至当前的历史。
窗口锚点永远不用于重新计算编码器的各根输入，避免窗口末端归一化泄漏。

解码器输出前两维自由数值，后五维非负值，确保 OHLC 几何关系合法。
绝对价格恢复需要一个明确保存的锚点，它不提供给神经解码器。归一化 OHLC
目标不依赖逐根 RMS；价格水平和波动尺度没有被悄悄作为整串旁路输入。

损失：七通道 SmoothL1，权重 [1,1,0.5,0.5,0.1,0.1,0.05]，按有效权重归一；
另加 0.5 倍相邻收盘路径变化 SmoothL1。缺失 OI 不计损失。没有未来预测或
规则标签损失。时间间隔在此是重建已发生的记录间隔，不是预测未来交易日历。

AdamW lr=3e-4、weight_decay=0.01、batch=32、30 epochs、梯度裁剪=1、seed=42。
按完整窗口的验证集重建损失选择 best.pt。训练、验证、测试的重建目标窗口
分别完整落在各自时间集合内，仍按单合约和滞后主力终点选样。数据指纹与旧
manifest 严格匹配。输出目录要求为空；没有断点恢复。

## 自动评价

- 神经自编码器、同 128 维 PCA、训练均值、清零 z 的消融对照。
- PCA 在固定随机抽取的最多 8192 个训练窗口拟合，使用训练统计标准化和
  特征分解，解码后非负通道投影到非负值。它压缩的是相同重建坐标，不是相同
  原始输入。其全局均值、尺度、基向量对应共享解码器参数，不能作为逐样本信息。
- 报告加权重建损失、各通道 MAE、收盘路径和相邻变化的 log-price 基点误差。
- 冻结编码器后，以当前中尺度规则结构为读出诊断，对比随机编码器与最近
  16 根原始编码。类别平衡岭回归，正则参数只在验证集选择。规则未用于预训练，
  但这仍不代表唯一的形态真值，也不是未知盈利信号的证明。
- 六个重建样例按测试索引等距抽取，保存 JSON 和本地可打开的 HTML。它们
  不是随机代表性统计样本；原始、模型和 PCA 使用同一价格坐标范围。

名义瓶颈：128×7=896 个目标数值 → 128 个潜变量 + 1 个绝对价格锚点。
不是文件压缩率，不计共享模型参数、元数据；保存所有重叠窗口的 z 未必省空间。
这是有损压缩，不是加密或可保证无损的编码。模型若丢失信息，解码器无法保证
恢复真实原值。当前只固定一个瓶颈维度，尚不是完整的压缩率—误差曲线。
单种子、重叠窗口和已多次查看的测试日期限制了结论强度。重建好不等于一定
适合分类、更不等于未来可预测；需要分别观察。

## AutoDL 与下载

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_REFERENCE=checkpoints/babel_r1_s42/manifest.json
export BABEL_AE_RUN=checkpoints/babel_ae_r1_s42
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
export BABEL_AE_LOG=logs/babel_ae_r1_s42.log
export BABEL_BATCH_SIZE=32
export BABEL_EPOCHS=30
export BABEL_SEED=42
mkdir -p logs
nohup bash scripts/babel_ae_autodl.sh all > "$BABEL_AE_LOG" 2>&1 &
tail -f "$BABEL_AE_LOG"
```

成功或普通报错退出时，脚本将已有报告和日志统一 cp 到：
`/root/autodl-tmp/download/babel_ae_r1_s42/`。没有复制大权重。
文件包括 manifest.json、history.jsonl、ae_metrics.json、reconstruction_examples.json、
reconstruction_examples.html、training.log、experiment_notes.md 和 run_status.txt。
如果被强制 kill -9 或机器关机，退出钩子无法运行，恢复后可以手工执行：
`bash scripts/babel_ae_autodl.sh export`。
评价失败时运行 `bash scripts/babel_ae_autodl.sh evaluate`，不必重训。
下载 HTML 可直接打开，无服务端、JavaScript 或网络依赖。

Python 推理：使用 schema/features 检查后的 `HistoryAE(**ck["config"])`，
加载 ck["model"] 并 `.eval()`。`encode(x)[:, -1]` 是 z；`decode(z)` 返回
归一化历史；`to_ohlc(reconstruction, anchor)` 还原 OHLC。不要用旧 index/serve
加载此权重。decoder 的 decode API 明确只接受二维单向量张量。
