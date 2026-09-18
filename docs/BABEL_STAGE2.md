# 阶段 2：只加入因果均线上下文

## 假设与边界

阶段 1 的全测试集诊断显示，各位置、状态和周期的单根变化都明显被平滑。
本轮检验：显式给出局部趋势、偏离和斜率，是否能让同一个历史压缩模型保留更多
局部变化。指标没有新增独立市场信息，也不保证能学会头肩顶等复杂形态。
本轮不加入未来预测、形态标签、渐进课程、加噪或新损失，避免混淆结果。

## 唯一输入变化

保留原 14 个输入，加上 4 个上下文特征，分别为：

1. log(close) − EMA8(log(close))。
2. EMA8(log(close)) − EMA32(log(close))。
3. EMA8 当前值 − 上一根值。
4. EMA32 当前值 − 上一根值。

EMA 使用 adjust=False，从每个真实合约序列第一根初始化，span 分别为 8、32 根。
不做连续合约拼接，不跨合约传状态，不删除跳空。每一时点只用当根已完成 bar
和之前数据；如果在 bar 未收盘时使用，需要另行定义实时输入，不能用最终收盘价。
四项都除以原编码器的上一根 RMS 波动尺度，再做 asinh，不拟合全量数据统计量。
EMA 在完整合约历史上递推，因此窗口首部的指标可以携带窗口前的历史摘要；
它不是未来泄漏，但本实验不应被解释为严格仅从窗口内原始价格增加变换。
8/32 是 bar 数，15/30/60 分钟数据对应不同的实际时长。

新增无偏置 Linear(4,128)，与原输入投影相加，增加 512 参数。
这个投影零初始化，并保存/恢复初始化时的随机数状态，因此同 seed 的主干、
解码器权重和初始输出与基线一致；上下文投影能从第一次反向传播开始学习。
网络总参数为 1,226,759。所有指标必须经过编码器和 128 维瓶颈，不能直达解码器。

## 固定条件

128 根窗口、128 维单向量、4 层编码器、4 个头、2 层解码器保持不变。
原始 7 通道重建目标、损失权重、相邻收盘差分损失保持不变。
同一数据与时间切分、stride=16、seed=42、batch=32、30 epochs，从头训练。
最佳 checkpoint 仍按验证集历史重建损失选择，不按测试结果挑选。
启动前核对阶段 1 的数据、配置和预算，保存其报告和 best.pt 的 SHA256；
新实验目录必须为空，旧权重不覆盖。旧版无 context 字段的 checkpoint 仍可加载。

## 评价和解释

训练后自动运行原评价和阶段 1 全量诊断。比较同一测试窗口上的收盘误差、
1/4/16 根变化 MAE、相关性与标准差比，并检查四个位置、状态、跳空和周期分组。
不能只因输出幅度更大就判定改善：误差和相关性也必须一起检查。
固定 PCA 基线、原始最后 16 根×14 维探针和随机编码器对照；随机编码器的
上下文投影也为零，所以该对照在初始化时与阶段 1 等价。
状态探针仍是规则标签读出，不是未来收益证明。
重叠窗口不当作独立样本；单 seed 结果仅作为是否值得继续的实验线索。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_REFERENCE=checkpoints/babel_r1_s42/manifest.json
export BABEL_AE_BASELINE_RUN=checkpoints/babel_ae_r1_s42
export BABEL_AE_RUN=checkpoints/babel_ae_ema_r2_s42
export BABEL_AE_LOG=logs/babel_ae_ema_r2_s42.log
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
export BABEL_EPOCHS=30 BABEL_BATCH_SIZE=32 BABEL_SEED=42
mkdir -p logs
nohup bash scripts/babel_ae_stage2_autodl.sh all > "$BABEL_AE_LOG" 2>&1 &
tail -f "$BABEL_AE_LOG"
```

结束后报告、日志、HTML 自动复制到 `~/autodl-tmp/download/babel_ae_ema_r2_s42/`。
请回传 manifest.json、history.jsonl、ae_metrics.json、ae_diagnostics.json。
不需要下载大权重。正常报错退出也会导出已有报告；若进程被强制 kill，
可保留以上环境变量，手动执行 `bash scripts/babel_ae_stage2_autodl.sh export`。
诊断单独重跑用 `diagnose`，评价单独重跑用 `evaluate`，无需重新训练。
