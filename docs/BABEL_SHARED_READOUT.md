# 冻结共享读出与PCA容量曲线

运行结果已复核，见[BABEL_SHARED_READOUT_REVIEW.md](BABEL_SHARED_READOUT_REVIEW.md)：共享头在保留位置仍失败，768维PCA达到当前目标的数值重建精度。本文件保留运行前协议。

研究目标仍是每根bar对已观察历史的可复用压缩表示。本轮在冻结编码器上区分共享读出训练不足和潜在容量问题，不进行未来预测、不拟合趋势规则、不改编码器或自动晋升。

## 固定协议

- 来源为已完成的`checkpoints/babel_prefix_readout512`，递归核验sampling/plain、缓存、读出权重及源代码。plain与sampled两个种子均使用第200轮编码器，编码器更新0。所有旧源码、缓存和权重只读。
- 训练共享头的位置固定为32/64/96/128；48/80/112为未用于共享头拟合、输入统计或验证选模的位置。后者衡量同窗口内的位置插值，不是新市场/新时间留出。
- 每个状态仍读取当前bar之前16根历史，排除当前bar，收盘路径在局部重新锚定。保持旧七通道目标及其训练统计，旧四位置数据逐元素复现；新位置输入严格截断。
- 五种输入：plain42/43、sampled42/43、current28。每种拟合一个共享ridge和一个共享MLP，不给位置ID，也不在推理时切换头或重新拟合统计。current28包含EMA等历史摘要，不是无历史或等容量控制。
- 每轮原4789窗口×4位置=19156行；不是19156个独立市场窗口。共享MLP100轮、batch256、75步/轮，每头7500步，五头37500步。AdamW、lr1e-3、5轮warmup/原cosine、weight_decay1e-4、clip1。旧四个独立头总7600步，聚合行曝光相同，但步数、参数容量和共享约束不同，不宣称严格等算力。
- ridge固定alpha网格1e-4/1e-3/.01/.1，缺失目标从对应拟合中排除。共享输入尺度只由train四位置拟合；共享验证只用val的同四位置，等位置权重选alpha/MLP epoch，包含第0轮。plain/sample同种子的共享MLP初始化相同。
- 全部头完成预算、锁定选择后才读取研究集。单独导出epoch0验证、全部历史、最佳与末轮、输入尺度、权重摘要、预算和验证回放。

## 对照与评价

训练位置直接复用并回放旧独立ridge头，研究逐窗口误差与旧报告逐项比较。额外为48/80/112拟合独立ridge参考头：只用该位置train拟合、该位置val选alpha；这些参考头与共享模型隔离，不能拿它们调整共享头。所谓“未训练位置”仅指共享头，不能声称所有模型均未在这些位置拟合。

保留全部位置、两种子、共享ridge/MLP最佳/末轮、独立ridge及train均值；MLP相对独立ridge只是读出质量诊断，不是纯共享因素消融。未新增未训练位置的独立MLP，避免扩张诊断矩阵。分项含路径、实体、量仓、一步变化MSE/相关性/幅度、各通道R²。训练位置和未训练位置分别汇总；先在同一原窗口内平均位置，再对窗口做按原端点周配对区间，不把位置当独立样本。

源test984与cross453仍为反复使用的研究集，区间未经多重比较校正。局部16根主指标与旧128根重建主指标不同，不能直接比数值。共享头接近独立头支持统一读出的可行性；只在训练位置好或靠实体/量仓明显退化换取综合改善，不算通用表示通过。

窗口队列仍继承原128端点的资格筛选，早期位置是离线同窗口诊断，不能冒称在线逐bar无选择偏差。输入特征含截至当前时刻的EMA等历史摘要；严格截断保证不输入后续bar，不表示每个输入特征都只取自当前神经窗口。

## PCA维度—误差曲线

原固定train4789、原归一化/七通道目标不变，仅拟合一次896维PCA，前缀截取32/64/128/256/512/768/896。所有维度预先固定，在val/test/cross全部报告，不按研究结果挑维度。缺失标签按原缓存置归一化0，评分仍用原mask。训练解释方差不代替验证/研究误差；掩码评分与多尺度变化指标不保证逐维单调。

512维重建必须复现旧PCA控制；896维完整空间重建检查数值往返。896不是建议的神经状态维度，也不表示存在896个有效独立方向：原最后一根未评分、目标有冗余。PCA直接读取已观察的目标历史，未直接编码全部28维原始特征，不证明神经可训练性或未来任务充分性。该曲线只刻画目前目标和误差要求下的线性压缩参照。

## AutoDL运行

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_shared_readout_autodl.sh all \
  > logs/babel_shared_readout512.log 2>&1 &
tail -f logs/babel_shared_readout512.log
```

默认两个读出worker并行；CPU拟合PCA阶段GPU可能空闲。来源目录可用`BABEL_SHARED_SOURCE`指定，输出用`BABEL_SHARED_RUN`，并行数`BABEL_SHARED_JOBS=1/2`。模型/数据身份变化必须新建输出，不覆盖已有实验。相同命令支持恢复优化器、RNG和绝对轮次；不要同时启动两个相同输出控制器。

成功或失败都会自动导出：
`/root/autodl-tmp/download/babel_shared_readout512_reports.tar.gz`

手动重新打包：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_shared_readout_autodl.sh export
```

报告含shared_readout_metrics.json、capacity_metrics.json、逐窗口误差/库存、selection_lock、capacity_fit、validation_replay、preflight、manifest/runtime、全部读出history/initial_validation/training_summary、参考头选择和日志。排除.pt/.npy/.npz；不需下载权重。正式提取/拟合只在AutoDL CUDA运行，本地仅合成数学/前后向和无神经更新流程测试。
