# 长短双状态：统一表示的信息保留实验

目标：区分长短信息能否共同保留，以及压缩到512维的代价。复用冻结长时512与短时512状态，
不重训编码器，不使用未来标签，不加入随机噪声或注意力融合新机制。

## 实验矩阵与表示

默认一次运行6组，各100轮：concat_s42、compress_s42、concat_s43、compress_s43、long_s42、short_s42。
两次种子覆盖新投影和读出头初始化，不覆盖原编码器训练或市场区间的不确定性。
每组使用相同4789/978/984训练、验证、测试端点，按时间划分；端点来自原合约，未拼连续合约。
仍只在完整128根片段末端验证，长历史4–16片段。没有实现任意每根bar在线刷新全局记忆。

- concat：长短向量按训练集逐维均值/标准差标准化后拼接，1024维，无可学习编码变换。
- compress：同一拼接输入 → Linear(1024,512) → GELU → Linear(512,512) → LayerNorm，512维。
- long/short：各自训练集标准化后的原始512维状态。

标准差下限.01；统计量只用训练集并保存在权重中。原长短状态不被覆盖，融合不反馈到编码器。
1024是保留信息的参考而非等参数上界；压缩候选有额外投影参数。报告各组参数量。

## 三类读出任务

所有组仅从本组的z解码，不允许原始状态、真实bars、目标mask或结构标签输入解码器。
三类头各有一个 z→256 投影，后续结构相同。同种子共用头初始化（输入宽度不同的投影除外）。

1. 最近64根：2层256宽8头Transformer，固定64位置编码，输出7通道。
   保留原16/32/64多尺度重建损失，以及.25倍4/16根收盘变化损失。
2. 长历史：2层256宽8头Transformer，固定16个片段位置，每位置同时输出128×7通道。
   历史padding只在损失与评价时排除；不作为模型输入。输出跨度最大2048根，无2048²注意力。
   这是新训练的历史读出头，不是原冻结历史解码器，不能将两者结果混作相同解码器实验。
3. 历史结构：z→256→LayerNorm→GELU→3，回归历史高低点距离与高点所在片段的归一化年龄。
   延用原HierWindows定义：排除当前128根，按各窗口实际可用历史计算。目标用训练均值/标准差标准化。

目标的价格锚点和7通道定义沿用上一轮cache。实际价格还原使用显式窗口前收盘锚点。
训练loss = recent/训练常数基线recent + history/训练常数基线history + 三项标准化结构MSE均值。
recent常数基线是训练样本的逐位置均值；history常数基线是训练集有效历史bars的通道均值。
窗口内部先平均有效片段，再跨窗口平均；长窗口不因片段多而得到更多权重。
三类任务等权，不能保证所有指标同时优化，必须报告每项结果。

## 选择规则：不能用平均分掩盖退步

基准来自同源缓存：上一轮验证集选中的short近期头，以及原长时历史解码器/结构头。
参考头不在本轮重选；结构比较使用本轮训练统计量下的每项MSE。

合格要求同时满足：
- 近期验证loss <= 原short近期loss ×1.05；
- 历史验证loss与收盘MAE各 <= 原long对应值 ×1.02；
- 三项结构验证MSE，每项 <= 原long对应项 ×1.02。

先选择合格候选，再最小化六指标中最坏的阈值比，最后用平均比值打破平局。
若没有合格轮次，也保存最小最坏比的候选，但qualified=false，必须报告未通过，不能当成成功。
包含epoch0候选，若选中0说明训练未产生更好的候选。所有选择仅看验证集。
此外按种子报告compress相对concat的每项验证指标，是否全部在5%容差内。
这些百分比是预先固定的工程容差，不是统计显著性保证，也不是测试期保证。

## 统一最终评价

所有6组训练完成后才进行测试，输出：
- 新近期/历史头：全历史、16/32/64近期尺度的误差、变化相关性和波动比例；历史按片段数/周期分组；
- 新结构头：三项MAE/RMSE/R²；
- 新近期头与新历史头最后64根的收盘重建差异（先统一价格锚点），检查两种读出是否互相矛盾；
- 统一岭回归结构读出与当前状态分类读出：只在train拟合，alpha=1/10/100按val选；
- 适配后的旧读出能力：用train拟合z到原long坐标的仿射岭回归，按val向量恢复MSE选alpha，
  将恢复的long输入原冻结历史解码器及结构头，报告重建与结构指标。
  这是经过适配的旧读出检查，不是把新向量直接塞进旧头；它能帮助区分信息缺失和新头学习不足。
- 同一组固定6个测试例子的历史/近期HTML和JSON、逐窗口指标、训练历史、汇总Markdown/JSON。

当前状态分类标签不参与神经网络训练。历史结构是已知历史的辅助监督，不能称为完全无监督。
长期/近期重建不证明未来预测能力。研究测试期已被多轮使用，不能充当最终独立留出集。

## 效率、恢复与输出

复用上一轮target_cache文件，不复制大型缓存。额外结构目标只准备一次，并验证原数据指纹、
缓存所有文件哈希和端点顺序。冻结编码器不在训练进程中加载；新读出头与compress投影才更新。
默认并行2进程、有效batch128、实际micro128；不是靠梯度累积冒充实际大batch。
AdamW lr3e-4、weight_decay.01、梯度裁剪1，cosine衰减到3e-5；每组100轮。
同seed各组每epoch使用相同样本打乱顺序。并行单GPU不保证更快，可BABEL_DUAL_JOBS=1。
每轮保存权重、AdamW、调度器、RNG；中断从上一个完整epoch恢复。失败停止本次启动的子进程。
只改变jobs/eval-batch可原目录恢复；改训练batch/micro/epochs需新目录（避免悄悄改变实验）。

本地仅合成前向/反向、不含optimizer更新的测试；正式训练只允许CUDA，在AutoDL运行。
显存占用低可能正常：这一轮不重训大编码器，也不在每步反传旧历史解码器。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_RECON_RUN=checkpoints/babel_recon_fusion512_s42
export BABEL_DUAL_RUN=checkpoints/babel_dual_state512
export BABEL_DUAL_JOBS=2
export BABEL_DUAL_BATCH=128
export BABEL_DUAL_MICRO=128
export BABEL_DUAL_EPOCHS=100
export BABEL_DUAL_LOG=logs/babel_dual_state512.log
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_dual_state_autodl.sh all > "$BABEL_DUAL_LOG" 2>&1 &
tail -f "$BABEL_DUAL_LOG"
```

成功标志：`Dual-state matrix complete`，随后`Download archive:`且`training_status=complete`。
自动打包 `/root/autodl-tmp/download/babel_dual_state512_reports.tar.gz`，直接下载上传。
无需另行打包。失败也打包已有日志和部分报告，run_status.txt同时记录本次命令状态与训练完成状态。
每次用新临时目录打包，避免旧下载目录污染报告；不含权重和大型缓存。

重跑上述同一命令从断点恢复。若仅最终评价失败，可以同样环境下执行：
`bash scripts/babel_dual_state_autodl.sh evaluate`。
只重新打包：`bash scripts/babel_dual_state_autodl.sh export`，不会训练。
