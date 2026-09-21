# 影线精度取舍：三组联合训练对照

## 问题与固定条件

检验降低已经观测到的影线极值精度，能否改善单向量的实体/路径表示。
这不把所有影线认定为随机，也不是预测未来极值；粗粒度形态不等于交易信号。
保留EMA8/32共18输入、512维2层GRU、width256两层并行解码头、原七输出，无新头。
同一原short epoch21与reconstruction short epoch33初始化，不从测试成绩挑某个微调种子。
每组seed42/43，各30轮，共六组；默认两进程并行。来源权重与缓存只读。

## 三个目标

- original_loss：原recent_loss + .25×训练集RMS归一化1/4/16根收盘变化损失。
- wick_low：仅将原上下影线精确损失各自的分子权重从.5改为.1。
- wick_coarse：wick_low + .05×影线粗粒度区间损失。

三组全部保留原损失分母，避免降低影线权重时意外放大开收盘和其他通道系数。
16/32/64重建权重、收盘变化、量仓与时间项均不变。所有组从完全相同模型函数开始。
原目标的数值及对输出的梯度须与原实现一致，降权组非影线梯度系数须保持一致。

粗粒度不添加分类头：直接从重建OHLC派生四个描述量——
log1p(上影/前序波动率)、log1p(下影/前序波动率)、上影/整根振幅、下影/整根振幅。
波动率是每根bar输入中的前序收益EWMA RMS，转为log-percent，与影线目标同单位；不使用未来统计。
预测影线比例使用预测自身实体和振幅，不把真实实体喂给解码器；因此新增比例损失也会对实体产生梯度，这是辅助几何约束的一部分。

每个描述量的三分位边界和RMS归一化尺度仅从训练集拟合。重复或零边界去重；退化通道允许少于三档。
根据真实值所在档位，预测落入该区间即不罚，超出区间的距离归一化后做SmoothL1。
最后一档无硬上界，但仍有.1权重的精确影线损失约束。四个描述量等权，.05是预先固定的实验权重。

## 数据、选轮与诊断候选

原4789/978/984端点、最近64根目标；64并行时序、TBPTT128、原分区神经状态重置、同seed相同分组。
原始数据只处理一次，增加每个目标bar的历史波动率缓存并纳入hash检查。
编码器lr3e-5、head lr1e-4、AdamW decay .01、clip1。

三组统一按验证细节误差选轮（并列时收盘MAE），要求相对原模型：
- 收盘MAE<=1.02倍；
- 实体MAE<=1.05倍；
- 16根收盘变化MSE<=1.05倍；
- 粗粒度区间误差<=1.10倍。

原七通道精确损失仍报告，但不再用其作为保留门槛，否则会把本轮允许的影线精度取舍直接拒绝。
上述形态保留门槛是工程约束，不能替代完整的档位混淆矩阵与状态读出评价。
第0轮也参与合格候选选择，选回0表示没有合格改善。
另保存diagnostic_best.pt：忽略保留门槛，只按相同验证细节误差选轮，独立报告其验证/测试结果。
该诊断候选不是部署候选，不能拿其测试结果反过来选轮。两类候选均包含0轮，并明确记录轮次。

## 必须查看的结果

原收盘1/4/16变化误差、相关性、幅度比、分段误差、每品种周期、旧head兼容性、状态读出、打乱向量与固定图全部保留。
增加各通道精确MAE、实体MAE、实体方向/十字星准确率、振幅MAE、影线波动率/比例档位误差、混淆矩阵及平衡准确率。
十字星死区为训练实体绝对值的10%分位；这仅是固定评价规则，不是市场真值。
价格/影线/实体/振幅MAE以log-price bp报告，量仓时间保持原编码坐标；通道支持数另列。
按周配对bootstrap比较两个seed中：降权−原目标、粗粒度−原目标、粗粒度−降权。

保存原模型固定训练端点上的损失分量与dL/d embedding梯度范数/夹角。
这是解码器输入处的梯度诊断，不是完整编码器参数梯度，更不能单独证明影线是瓶颈。
各候选报告实际原损失贡献；不会把较低的修改后训练loss直接叫作模型提升。
研究测试期重复使用、两个种子共享预训练；只有在路径/实体改善且形态与状态保留时才考虑升级。
不自动替换stream bundle。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_SHORT_RUN=checkpoints/babel_short512_b128_s42
export BABEL_RECON_RUN=checkpoints/babel_recon_fusion512_s42
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_FUSION_RUN=checkpoints/babel_fusion512_s42
export BABEL_WICK_RUN=checkpoints/babel_wick512
export BABEL_WICK_LOG=logs/babel_wick512.log
export BABEL_WICK_STREAMS=64
export BABEL_WICK_EPOCHS=30
export BABEL_WICK_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_wick_autodl.sh all > "$BABEL_WICK_LOG" 2>&1 &
tail -f "$BABEL_WICK_LOG"
```

自动准备数据/对齐、拟合训练分档、做合成反向预检、执行六组训练、评价两类候选、导出。
本地仅做合成推理/梯度/导出测试，没有optimizer更新；真实训练强制CUDA。
成功或失败均自动打包到`/root/autodl-tmp/download/babel_wick512_reports.tar.gz`，上传这一个包即可。
包含指标、梯度诊断、分档、训练记录、图和日志，不含权重和大缓存。

手动导出：
```bash
cd /root/autodl-tmp/obson
bash scripts/babel_wick_autodl.sh export
```

相同配置重跑all可从完整epoch恢复optimizer/RNG；先确认旧进程停止，日志改为>>保留旧记录。
仅jobs可改变；改streams/epochs/目标必须换目录。失败保留非零退出码并停止本批其他worker。
Wick ablation complete表示整套实验跑完，improved_candidates另列验证达标且选中训练轮次的候选，不等于部署认可。
