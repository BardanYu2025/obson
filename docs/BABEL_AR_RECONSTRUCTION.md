# 冻结短状态：自回归历史解码实验

只训练新解码器，复用reconstruction_fusion已校验的短向量和64根目标缓存。
不加载或更新编码器，不改变18维输入、512维短状态、归一化、价格锚点或时间划分。
本轮只能判断解码方式是否改善重建；不能据此声称embedding已经变好。

## 一次运行的矩阵

每组种子42/43，各60轮，同样本顺序，同实际batch256，AdamW lr3e-4、weight_decay .01、clip1。
默认两个训练进程并行，一共八个新解码器。另独立评价原short最佳解码器。

| 组 | 结构 | 训练目标 |
|---|---|---|
| parallel | 原2层、8头、width256 Transformer整段解码 | 原recent_loss |
| ar_tf | 2层width352 GRU，每步注入同一个512维向量 | 仅teacher forcing |
| ar_mix | 同一GRU结构与初始化 | teacher forcing + 全64步自由重建 |
| ar_noz_mix | 同一GRU，向量输入恒为零 | 同ar_mix；独立训练的无向量对照 |

并行头1,713,159参数，条件AR头1,678,695参数（相差约2%）。无向量组保留相同模块，
但condition权重中依赖z的180,224个参数不影响输出；不把它描述成相同有效容量。
GRU和Transformer结构不同，因此这是近似参数预算下的实用解码方案对照，不是只改变因果mask的单因素实验。
同轮数不代表同FLOPs或耗时；mix每批有TF与自由两次前向，记录实际耗时和显存。

## 解码协议与对齐

目标仍为最近64根原七通道，开收盘是相对该窗口前收盘价的log-percent坐标。
AR每步输入为：向量投影 + 前一步七通道经asinh后的投影 + 固定位置编码，经LayerNorm送入GRU。
第一步使用学习的BOS。输出开收盘自由实值，其余通道softplus，保持原OHLC几何参数化。
不使用外部真实bar初始化生成，不输入原编码序列、真实目标mask、真实中间价格或真实终点价。
显式价格锚点仅在将生成坐标换回OHLC时使用，不是AR神经网络输入。

teacher forcing严格右移：生成位置i只输入目标位置<i的真实bar。
自由解码只接收z；后续输入都是自己的输出，隐藏状态与输出均不detach，完整64步反传。
mix前5轮只做TF，随后10轮将自由损失权重λ从0.1升到1，之后保持1。
loss=(TF_loss+λ×free_loss)/(1+λ)。两个loss均使用原16/32/64多尺度重建，
包括原1根及4/16根收盘变化项。它是自由重建辅助目标，不是随机scheduled sampling。
ar_tf保持λ=0，是训练/生成差异的诊断对照。

所有新模型（包括ar_tf）都按验证集**完整自由64根重建recent_loss**选轮，包含epoch0。
不按teacher loss选轮，不按测试表现挑种子。不同组各自最佳轮次，完整历史保留。
选中epoch0也如实报告，表示后续训练没有改善该自由重建标准。

## 评价

- 保留原最佳并行头，另从头重训并行头以匹配本轮预算；二者均单列。
- 全64根自由重建：收盘MAE bp、1/4/16根变化误差、相关性、波动标准差比例。
- 最近16/32根、从最早到最近的Q1–Q4分组，以及逐解码步MAE曲线，检查误差累积。
- AR同时报告teacher-forced误差；绝不与完整自由重建混为同一种能力。
- 固定无自映射乱序z，目标不变，评价自由重建变化；不跨评价batch临时打乱，避免batch相关。
  乱序有分布变化，只是依赖诊断；独立训练的ar_noz_mix是更强对照。
  验证no-z输出不依赖被传入的向量，数值比较atol/rtol各2e-5。
- 按品种/周期分组，固定6个例子输出可离线打开的HTML和OHLC JSON，样例不按误差选择。
- 同seed配对比较TF/parallel、mix/parallel、mix/TF、mix/no-z、mix/原最佳头；
  对窗口收盘MAE差值按日历周整体bootstrap1000次，负差值有利于前者。
- 测试仍是此前研究测试期，不是新holdout。两个种子反映解码头初始化，不覆盖原编码器不确定性。

优先看两个种子的完整自由重建、误差累积、变化相关性和波动保留，再看teacher指标。
只降低teacher loss不算成功；更平滑的曲线降低MAE但丢失变化，也不能称为全面改善。
若AR在相同向量上有效，再另开保留并行重建约束的编码器联合训练实验，本轮不会自动解冻。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_ARDEC_RUN=checkpoints/babel_ardec512
export BABEL_ARDEC_LOG=logs/babel_ardec512.log
export BABEL_ARDEC_BATCH=256
export BABEL_ARDEC_EPOCHS=60
export BABEL_ARDEC_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_ardec_autodl.sh all > "$BABEL_ARDEC_LOG" 2>&1 &
tail -f "$BABEL_ARDEC_LOG"
```

启动先校验旧缓存与数据指纹，再做一次性合成GPU前向/反向预检，预检模型丢弃、不做optimizer step。
预检不覆盖两个进程加AdamW的全部内存，因此不保证所有GPU设置都不OOM。
真实数据训练仅在AutoDL运行，本地只做合成功能/梯度/恢复/导出测试。
数据准备与末尾打包阶段GPU可能空闲。编码器不在训练进程中，解码头约170万参数，显存不必占满。
每轮记录free/teacher验证、秒数、单进程峰值显存和该组剩余时间估计（阶段切换/并发竞争会影响估计）。

last.pt保存模型、最佳候选、优化器和RNG，按完整epoch恢复。训练中断时确认旧进程已经停止，
用相同参数重跑all可续训，建议日志重定向改为`>>`。jobs/eval-batch可改，训练batch/预算不可改。
本次启动任一worker失败会停止同批其他worker，保留checkpoint并以失败状态导出。
同一目录不能同时启动两个主进程。

成功标志：`AR reconstruction matrix complete`，最后`run_status=complete`。
成功/失败都自动输出：`/root/autodl-tmp/download/babel_ardec512_reports.tar.gz`。
包内含总指标、各组训练历史/验证/自由与teacher指标、逐窗口指标、重建图、配置和日志；排除权重和大缓存。
上传这一个包即可。单独打包：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_ardec_autodl.sh export
```

只重做测试评价：保持原训练配置，运行`bash scripts/babel_ardec_autodl.sh evaluate`。
