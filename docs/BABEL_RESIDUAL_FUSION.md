# 长期表示 + 短时修正：冻结编码器实验

研究假设：把短时状态投影为长期综合向量的修正项，在保护长期信息的前提下，改善当前状态读出。
这次长期输入是 large512/joint/best.pt 的历史聚合向量，不是局部128根向量。
短时输入来自 short512_b128/best.pt，沿各合约/周期/时间分区按原始顺序更新。
两个编码器、历史解码器、历史结构头全部冻结。原权重不修改，实验开始/结束核对指纹。

## 对照

所有方案输出512维，使用同构 LayerNorm + Linear(512,4) 状态读出头：

| 模式 | 表示 |
|---|---|
| long | m |
| short | s |
| add | m + W·LN(s) |
| gated | m + sigmoid(G[LN(m);LN(s)]) ⊙ W·LN(s) |
| long_gated | 与 gated 相同参数量，但将短时输入替换为 m |

long_gated 是增加可训练参数的对照，用于区分短时信息与额外非线性容量。
投影权重与bias初始为0；门控权重初始为0、bias使sigmoid为0.1。
因此 add/gated/long_gated 初始严格输出 m，门控在投影开始学习后获得梯度。
readout 在各方案中用同seed初始化。add和gated本身参数量不同，不把差异完全归因于门控。
只在输出处融合，不把融合结果回写原GRU状态或长期记忆。

## 时间与端点

使用原 HierWindows 的完整4–16片段端点，每片段128根；所有被评分历史都在对应分区内。
长向量与短状态必须属于同一合约、周期和精确bar端点，不做最近邻时间拼接。
短状态从本分区起点开始更新，结束后padding不产生端点评分、不接入其他合约。
模型读入当前bar的已完成信息；只在bar收盘后可用。没有未来目标或未来输入。
本次只在完整片段端点验证融合，尚不是任意bar上刷新长期记忆的在线系统。
短状态可包含更早的分区内历史，长模型最多2048根，明确保留这一上下文差异。
预计端点为 train4789/val978/test984，以 coverage.json 为准；不能与8936窗口的短时报告直接混比。

## 训练与选择

先一次性提取冻结表示并缓存，缓存绑定原权重指纹与精确端点。
训练数据的缓存放GPU上，只训练小融合模块与读出头；默认batch512、100轮、AdamW lr3e-4、weight_decay=.01、clip1。
每个方案独立seed42。分类权重仅由训练标签频数决定。
loss = 训练类别加权交叉熵 + 0.1 × mean((z-m)^2)/mean(m^2)。
后项用于add/gated/long_gated；long/short不适用。它是软约束，不保证重建不变。
每个方案按验证集相同形式目标选择最佳轮次，包含初始epoch0候选；不在测试集上挑轮次。
所有方案选完权重后才评分测试集。报告类别召回率、BA、门控均值、相对修正幅度。
完整epoch断点保存模型、AdamW、随机状态、历史和最佳权重；同配置重跑自动恢复。
改变训练batch/预算/保护系数必须新目录；评价提取batch可独立调整。
训练是针对现有规则状态标签的监督融合，不能宣称获得未经监督验证的通用表示。
既有测试区间已反复参与研究判断，未来预测实验必须另留最终时间区间。

## 长期信息是否保留

在验证和测试中，把 long、add、gated、long_gated 输出送回原冻结的历史解码器和历史结构头。
比较历史重建损失、价格误差、1/4/16根变化相关性、最近64根重建、历史高低点/年龄误差。
这里测量与原解码器的兼容和可读信息，不等价于信息论上的全部信息保留。
short向量不属于原长期解码器的坐标空间，因此不强行给它计算长期解码指标，报告明确为null。
固定六个测试端点保存真值与重建图；不按误差挑图。

## AutoDL：运行与导出

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_SHORT_RUN=checkpoints/babel_short512_b128_s42
export BABEL_FUSION_RUN=checkpoints/babel_fusion512_s42
export BABEL_FUSION_BATCH=512
export BABEL_FUSION_LOG=logs/babel_fusion512.log
mkdir -p logs
nohup bash scripts/babel_fusion_autodl.sh all > "$BABEL_FUSION_LOG" 2>&1 &
tail -f "$BABEL_FUSION_LOG"
```

出现 `Fusion complete` 并看到 `Download:` 后，报告已自动复制到download，执行：

```bash
cd /root/autodl-tmp
tar -czf babel_fusion512_reports.tar.gz -C download babel_fusion512_s42
```

上传 `/root/autodl-tmp/babel_fusion512_reports.tar.gz`。
包内包含manifest、coverage、五组history/metrics、fusion_metrics、重建图、artifacts与日志，不含权重和表示缓存。
中断后确认旧进程退出，保持配置，将nohup中的 > 改为 >> 重跑即可。
本地仅运行合成前向/反向和只读覆盖测试，正式训练在AutoDL；没有本地CUDA验证。
这是轻量融合训练，显存低于端到端训练是预期行为，不以占满显存作为成功标准。
