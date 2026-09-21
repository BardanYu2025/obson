# 局部变化：编码器与重建损失2×2对照

上一轮固定embedding的AR解码未显著改善完整自由重建，本轮回到原并行头，
区分现有向量的可读取细节与编码器训练目标的影响。保留512维、2层GRU和2层width256并行解码器。

## 实验矩阵

| 模式 | 编码器 | 重建头训练目标 |
|---|---|---|
| frozen_base | 冻结 | 原recent_loss |
| frozen_detail | 冻结 | 原损失 + 0.25×归一化局部变化损失 |
| joint_base | 小学习率微调 | 原recent_loss |
| joint_detail | 小学习率微调 | 原损失 + 0.25×归一化局部变化损失 |

每组种子42/43，共八组，各30轮；另评价未经本轮更新的original。
全部从short512_b128第21轮编码器和recon_fusion short第33轮头开始，不从随机头开始。
种子影响组顺序和dropout，不代表独立预训练编码器。
编码器lr3e-5、头lr1e-4，AdamW weight_decay .01，clip1；不扩宽、不增层、不加自回归头。
原ShortState内部的小decoder保留在state_dict中但冻结且不使用。

## 数据与对齐

- 复用原target_cache的4789/978/984端点及最近64根七通道目标（以运行时coverage为准）。
- 数据只准备一次：按原分区开始行打包18维因果特征，逐合约/周期保存，直到本分区最后评分端点。
  特征仍由更早已知行情预热；GRU神经状态在分区开始重置。
- 64表示同时推进的序列数，每条每步128根；每步有效评分端点数量随序列长度变化。
  相近长度的序列成组，训练仅打乱整组顺序，不打乱bar，不将结束的lane接到其他合约。
- joint沿时间推进真实输入，隐藏值跨chunk保留、计算图每128根截断；标准TBPTT更新期间隐藏值
  可能来自上一步权重，这是训练近似。验证/测试重新从分区起点使用最终权重重放。
- frozen使用原缓存向量，但与joint采用相同的端点分组和优化器step；防止采样和更新次数成为额外差异。
- 启动先将原编码器按本轮分组重放验证集，对齐缓存；保留严格比较结果，同时要求
  max_abs<=.002、RMS<=.0002和原重建保留门槛。失败导出warm_start_validation.json并停止。
- 来源权重/目标缓存通过hash核验；新的时序缓存独立写入本实验目录，不改源数据或源权重。

## 局部损失和选轮

对h=1/4/16，比较真实与重建收盘变化，除以该h在**训练集**上的全局RMS尺度
（log-percent单位，下限.01），再做SmoothL1，三种h等权。
尺度不根据当前评价窗口、验证集或测试集重新计算。增加的是匹配真实变化的监督，
不是把标准差强行拉到1，也没有生成随机高频波动。
原recent_loss保持原有16/32/64多尺度七通道损失及收盘差分项。

四组统一选轮：在验证集满足以下三条后，最小化归一化局部变化损失：
- 收盘MAE不超过original的1.02倍；
- 原recent_loss不超过original的1.05倍；
- 16根变化MSE不超过original的1.05倍。

包含epoch0，若没有合格改进则保留原候选，明确输出best_epoch=0，不用失败候选替换原基线。
门槛是预先声明的工程保留标准，不保证测试集一定满足，也不是统计显著性门槛。
训练目标不同，但选择规则一致；original与各自继续训练的base控制额外训练预算影响。

## 评价和解释

- 重建MAE、1/4/16根变化相关性/误差/标准差比，16/32近期和Q1–Q4分段、逐步MAE。
- 同seed比较frozen_detail−frozen_base、joint_base−frozen_base、joint_detail−joint_base、joint_detail−frozen_detail。
  对归一化细节误差、MAE、1根变化误差/相关性/标准差比给出日历周配对bootstrap区间。
  标准差比不是越大越好，必须结合对应误差和相关性；两个种子都完整报告。
- 每组将新向量送入原冻结解码头，单列旧接口兼容性；新头适配后变好不代表旧头仍可直接使用。
- 同一984测试端点做训练拟合、验证选alpha的中尺度线性状态读出；joint重新提取三分区向量，
  frozen复用同一基准结果。不可与上一轮4581个中间时点的58.28%直接比较。
- 固定打乱z控制和固定6张重建图；不按图片好看程度选择样本。
- 若冻结编码器改loss即可改善，说明至少一部分细节原本可读；若joint进一步改善，说明编码过程
  可以学得更好。任何一组失败均不能证明向量中完全没有信息或512维达到理论容量上限。
- 状态读出与旧头兼容性是附加报告，不参与逐epoch选轮。它们退步时不自动部署新模型。
  新encoder/head作为独立候选保存，不自动替换原stream bundle；长历史分支保持原样。
- 研究测试期已多轮使用；不会把这轮结果声称为独立新holdout结论。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_DETAIL_RUN=checkpoints/babel_detail512
export BABEL_DETAIL_LOG=logs/babel_detail512.log
export BABEL_DETAIL_STREAMS=64
export BABEL_DETAIL_EPOCHS=30
export BABEL_DETAIL_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_detail_autodl.sh all > "$BABEL_DETAIL_LOG" 2>&1 &
tail -f "$BABEL_DETAIL_LOG"
```

自动进行共享数据准备、暖启动对齐、GPU合成joint前向/反向预检、八组训练、评价与打包。
预检模型丢弃，不做optimizer step，不保证两个worker的总显存；记录峰值用于诊断。
joint需重放原始时序，会明显慢于只训练缓存上的小解码头；日志每轮记录秒数和剩余估计。
源模型和大缓存只读，训练仅在CUDA环境执行；本地仅合成无optimizer更新测试。

完整epoch末保存optimizer/RNG，可用同配置重跑all恢复。仅jobs可以调整；
改变streams/epochs/学习率/目标须新目录。重新启动前确认旧进程停止，日志可改为`>>`。
任一worker失败会停止本批其他worker并以失败状态自动导出，不杀无关训练。
成功标志：`Detail alignment matrix complete`，随后`run_status=complete`。

成功/失败均自动打包到：`/root/autodl-tmp/download/babel_detail512_reports.tar.gz`。
上传这一个包即可，内含配置、对齐诊断、训练记录、总指标、状态读出、旧头兼容性、图片和日志；不含权重或.npy缓存。
手动重新导出：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_detail_autodl.sh export
```
