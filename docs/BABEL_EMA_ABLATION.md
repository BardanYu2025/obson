# EMA输入尺度对照

上一轮联合微调改善整体重建，但单根变化MAE改善不足0.2%，幅度比仍约0.40。
本轮保持512维、2层GRU、2层width256并行解码头，检验显式价格EMA上下文的增益。
所有组均联合训练编码器/解码器，原recent_loss + 0.25×训练集RMS归一化1/4/16根变化损失。

## 八组实验与初始化

| 输入 | 上下文 | 初始化 |
|---|---|---|
| none | 五列零，无显式价格EMA | 五列上下文投影全零 |
| ema8_32 | 原四特征加一列零 | 五列上下文投影全零 |
| multiscale | price−EMA4、EMA4−8、EMA8−16、EMA16−32、EMA32−64 | 五列上下文投影全零 |
| retained_ema8_32 | 原四特征加一列零 | 保留原四列投影，新增一列零 |

每组种子42/43，各30轮，共八组，默认两个进程并行。前三组是主对照，第四组测重新适应输入的成本。
所有组使用相同19输入槽位，保持原14基础特征投影、bias、GRU、LayerNorm与并行头权重。
前三组只清零上下文投影，因此初始函数输出必须完全相同；随后EMA组可通过梯度学会使用上下文。
第四组与原18输入模型函数等价，启动审计比较真实验证集重放与旧缓存，采用已有max .002/RMS .0002边界和重建保留门槛。

来源仍是原short最佳21轮、原并行head最佳33轮，避免以本轮测试挑选上一轮某个种子来初始化。
这不是从零预训练比较：共同GRU曾用EMA训练，清零输入投影不会消除这种历史影响。
none仍保留基础特征里的波动率、前收盘信息，只去除显式价格EMA上下文。
参数存储量相同，但零输入列不贡献有效容量；相对旧模型仅增加512个输入权重，不增加隐维或层数。

## 特征与数据

先按原合约/周期全历史计算对数收盘价EMA，span4/8/16/32/64，adjust=False，首根初始化为首根log close。
每根衰减系数2/(span+1)，按bar数量递推，窗口、128chunk、交易日和训练分区不重置。
差值除以前序收益EWMA RMS波动率，再asinh；与原四EMA特征采用同一尺度。
14基础+4原EMA+5多尺度共23列共享缓存，每组按需映射到19列输入。
原四通道直接复用已审计encode_context，避免重算造成不必要数值差异。

复用原4789/978/984训练/验证/测试端点和最近64根目标。共享数据只读取/处理一次，缓存有hash核验。
同一seed各组采用相同时间序列分组、端点、更新次数与dropout随机序列起点。
64 streams是同时推进的合约周期序列数；不打乱bar，不拼接不同合约。神经状态在合约/周期/分区边界重置；TBPTT128。
编码器lr3e-5、head lr1e-4、AdamW decay .01、clip1。

## 选轮与报告

验证集收盘MAE<=原模型1.02倍、原损失<=1.05倍、16根变化MSE<=1.05倍，再最小化综合细节误差。
中性初始化可能暂时不满足原模型门槛，允许学习适应，但绝不放宽门槛。
若整轮没有任何合格候选，保存epoch0用于诊断，明确validation_retained=false，不称为成功升级。
首个合格候选可替换不合格初始状态，即使初始状态某一项细节误差更低。
completion只表示实验完整；accepted_candidates另列达到验证保留门槛的候选。

报告包含全部种子、原模型、按周配对bootstrap、1/4/16根变化误差/相关性/幅度、近期/分段/逐步误差、
各品种周期、同984端点状态读出、旧head兼容性、打乱向量控制、固定六张重建图。
比较EMA8/32−none、多尺度−none、多尺度−EMA8/32，以及两个EMA组相对保留权重控制。
不能只按标准差比挑选：更接近1而误差或相关性恶化，不算保留更多真实细节。
仅有两个微调种子且共享预训练，测试期已复用多轮；不宣称独立新holdout结果。

模型、特征版本与解码头成套保存，不自动替换stream bundle。本轮不改变长历史分支。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_SHORT_RUN=checkpoints/babel_short512_b128_s42
export BABEL_RECON_RUN=checkpoints/babel_recon_fusion512_s42
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_FUSION_RUN=checkpoints/babel_fusion512_s42
export BABEL_EMA_RUN=checkpoints/babel_ema512
export BABEL_EMA_LOG=logs/babel_ema512.log
export BABEL_EMA_STREAMS=64
export BABEL_EMA_EPOCHS=30
export BABEL_EMA_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_ema_autodl.sh all > "$BABEL_EMA_LOG" 2>&1 &
tail -f "$BABEL_EMA_LOG"
```

所有组均重放真实时序，预计比上一轮四组冻结/四组微调耗时更长。每轮记录耗时和剩余估计。
开始的数据准备在CPU完成，预检仅合成前向/反向，不做优化更新，也不能代替AutoDL真实训练验证。
本地不训练。正式执行强制CUDA。原权重和原目标缓存只读。

结束后自动打包成功或失败报告到：
`/root/autodl-tmp/download/babel_ema512_reports.tar.gz`。
只需上传这个包；包含指标、初始化审计、训练曲线、图、配置与日志，不含.pt或.npy。
手动导出：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_ema_autodl.sh export
```

同一配置重新执行all可从完整epoch恢复，optimizer/RNG同时恢复；确认旧进程已停止，日志用>>保留旧记录。
仅jobs可直接改变；调整streams/epochs/学习率/特征必须换输出目录。
all完成条件：EMA ablation complete以及run_status=complete；是否找到合格模型另看accepted_candidates。
evaluate重做评价仍需保持原配置环境变量。进程失败时停止本批其余worker，保持非零退出码并自动导出。
