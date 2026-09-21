# 价格与成交/持仓：输入特征和分支结构对照

## 六组实验

| 模式 | 输入 | 输入结构 |
|---|---|---|
| baseline | 原18维EMA8/32，新增10槽位为零 | 原线性投影→LayerNorm→GELU |
| features | 原18维＋10维活动描述量/有效性标记 | 同一线性投影→LayerNorm→GELU |
| branches | 与features完全相同 | 价格与活动分支分别处理，再融合 |

每组seed42/43、30轮，默认两进程并行。保持512维、2层GRU、原width256两层并行解码头。
原short epoch21与reconstruction short epoch33共同暖启动，原18输入权重保留、新10列零初始化。
不沿用影线粗粒度目标：全部使用原recent_loss + .25归一化1/4/16根收盘变化损失，原七输出及量仓权重不变。
没有新增训练头；检验输入信息和结构，不把目标变化混进来。

## 数据字段与特征

下载代码使用TqSdk Kline。官方定义volume为该bar内成交量，open_oi/close_oi为bar起止持仓量：
https://doc.shinnytech.com/tqsdk/latest/reference/tqsdk.api.html#tqsdk.api.TqApi.get_kline_serial
旧输入仅显式使用收盘OI；本轮利用原始CSV已有但未进入旧模型的open_oi，因此不全是对旧18维的代数改写。
下载器可能把缺失持仓字段补零，新特征保守地把非正持仓视为不可用；不把缺失当作减仓。

新增10维：
1. log1p(volume)减去前序log成交量EMA32；
2. 相邻bar的log1p(volume)差，首根为0；
3. asinh(100×持仓变化/起点持仓)；
4. asinh(持仓变化/本bar成交量)，成交量为0时屏蔽；
5. asinh(100×成交量/起点持仓)；
6. 持仓变化可用标记；
7. 持仓变化/成交量可用标记；
8. 成交量/持仓可用标记；
9. 持仓起点来自open_oi的标记；
10. 前一根成交量存在标记。

持仓变化优先close_oi−open_oi；open_oi不可用时，仅在两个bar时间恰好相邻、前后持仓有效时用收盘持仓差。
跨休市缺口不猜持仓变化；有有效open_oi时即使隔夜也能描述当前bar自身变化。
成交量相对变化按观察到的bar比较，保留原时间间隔特征，不声称已消除日内季节性。
特征按原合约/周期完整前序计算，再按原分区截取。首根、零成交、无OI与开盘OI缺失均显式处理。
不按某个阈值强行裁剪持仓/成交比；只记录abs(delta)>volume与>2volume异常比例，不能据此推断主动买卖。
这些是观测描述量，不是开多、开空、平多、平空真值，也不是已测得的流动性。

共享28列缓存只准备一次；每个合约周期的可用率、回退、零成交和比值异常写入cache/activity_audit.json，缓存hash覆盖所有新文件。
本地抽查rb2505/60、MA309/15、cu2503/30均能读取起止OI；完整966文件统计在AutoDL按训练来源指纹执行。

## 参数与暖启动对齐

flat两组输入维度均28，参数存储量一致；baseline的新增零输入列不贡献有效容量。
价格分支索引0..8、12..17，活动分支9..11、18..27；两组不重叠且覆盖全部输入。
分支模型复制同一线性权重的列并共享bias：

p=Wp*x_price, a=Wa*x_activity
u=p+a+b+tanh(gp)*(GELU(LN(p))-p)+tanh(ga)*(GELU(LN(a))-a)

分支LN无可学习affine参数，gp/ga是两个零初始化标量；融合后仍使用原共享LayerNorm/GELU和同一GRU。
因此分支模型仅比flat多2个参数，零门控时与flat初始函数等价（矩阵乘法分组允许数值舍入误差）。
这是输入分支实验，不是两个独立GRU，也不声称恢复真实订单流。
所有组初始化消耗相同随机数，包括flat组丢弃的split模块，保证同seed训练dropout随机流起点一致。
GPU训练前逐组真实验证集重放，要求原有max .002/RMS .0002边界及重建保留门槛；记录原严格比较，不隐藏差异。

## 训练与评价

原4789/978/984端点、最近64根目标、64条并行时序、TBPTT128、编码器lr3e-5/head1e-4、AdamW .01、clip1。
同seed各组时序分组/端点/优化步数一致，不打乱bar。数据分区和神经状态重置规则保持原样。
按验证细节误差选轮，同时要求原收盘MAE<=1.02、原loss<=1.05、16根变化MSE<=1.05。
包含epoch0，选回0表示未发现合格改善；另保存不受门槛约束的diagnostic_best.pt并报告验证/测试重建，避免把初始模型误称为训练后的表现。

原收盘变化、各品种周期、状态读出、旧头兼容、打乱向量和固定图保留。另报告实体/振幅/影线MAE、原编码坐标中的成交量/持仓及1/4/16根变化误差，逐项列支持数；不把编码坐标误差当成原始手数误差。
新增冻结线性读出：预测当前bar与最近16根平均的前5个活动描述量，共10个回归目标。
16根平均要求该描述量16根都有效；首根变化、无OI或零成交按独立mask排除，不补零充当真值。
每个目标的embedding标准化与目标尺度仅训练集拟合，ridge alpha在1/10/100/1000中按验证MSE选取。
报告测试R²、训练尺度归一化MSE、训练均值基线、支持数，常数或支持不足目标明确跳过。
这里预测的是已观测历史描述量，不是未来。没有对真实开平仓贴伪标签。
对features−baseline、branches−features、branches−baseline提供同seed按周配对区间；状态读出不能直接与其他端点集合比较。

分支门控最终数值写入报告。若门控近零，只能说明本次训练未明显采用该处理方式，不能证明所有分支架构无效。
两个seed共享预训练，测试期已经多轮使用；观察价形保留、状态与活动读出共同变化，不仅看新增读出分数。
本轮不自动更新stream bundle，使用新模型须携带新特征与架构版本。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_SHORT_RUN=checkpoints/babel_short512_b128_s42
export BABEL_RECON_RUN=checkpoints/babel_recon_fusion512_s42
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_FUSION_RUN=checkpoints/babel_fusion512_s42
export BABEL_ACTIVITY_RUN=checkpoints/babel_activity512
export BABEL_ACTIVITY_LOG=logs/babel_activity512.log
export BABEL_ACTIVITY_STREAMS=64
export BABEL_ACTIVITY_EPOCHS=30
export BABEL_ACTIVITY_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_activity_autodl.sh all > "$BABEL_ACTIVITY_LOG" 2>&1 &
tail -f "$BABEL_ACTIVITY_LOG"
```

自动准备数据/审计、暖启动对齐、合成反向预检、六组训练、全部评价和打包。
正式训练强制CUDA，本地仅合成测试和只读样本核对。结束后上传一个文件即可：
`/root/autodl-tmp/download/babel_activity512_reports.tar.gz`。
报告包含数据可用性审计、初始化/门控、指标、图、训练记录和日志，不含.pt/.npy。
手动导出：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_activity_autodl.sh export
```

同配置重跑all可恢复optimizer/RNG；先确认旧进程停止，日志改为>>。只有jobs可直接改变，其余设置变更需新输出目录。
失败时停止本批其余worker、保留非零退出码并自动导出。Activity ablation complete只表示运行完整；improved_candidates是验证候选，不是部署认可。
