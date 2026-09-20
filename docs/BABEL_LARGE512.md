# 512 维分层历史模型：完整分阶段训练

这是新模型，从头开始，不把旧 256 维向量线性放大冒充新增信息。
总参数 45,598,731，其中局部模块 32,065,543。
局部编码器 width=512、8 层、8 头，512 维瓶颈，2 层 512 宽度解码器。
历史聚合器 width=512、4 层、8 头，输入 4–16 个完整的局部片段及价格锚点关系，
输出单个 512 维综合状态。每片段仍为 128 根，对应最多 2048 根。

## 解码与因果边界

综合状态通过固定位置查询和非线性历史解码器产生 16 个局部向量，以及各片段
相对当前片段前收盘价的价格偏移。局部解码器将它们还原为 16×128×7 通道。
全局解码只接收一个向量和固定位置，不接收原输入、原局部向量或真实历史锚点。
恢复实际价格时，只额外使用当前片段前收盘价这个标量；其他片段的锚点由模型预测。
当前片段相对自身锚点的偏移固定为零。

聚合器可读取窗口内全部已发生片段，仅暴露当前最终综合状态；不将其内部较早
token 宣称为当时可用的 bar embedding。局部编码器仍有因果 attention mask。
这是历史重建，不包含下一根预测。价格位置输入为 asinh(log(片段锚点/当前锚点)*100)。

历史训练端点 stride=128，历史不足则取 4–16 个有效片段，左侧 padding 被遮蔽。
所有训练/验证/测试重建目标都必须完整位于各自时间分区，不允许评分窗口跨边界。
这比上一轮允许历史上下文跨分区的基准更严格，样本及任务定义不同，不直接比较分数。
coverage.json 报告各分区样本数、完整 16 片段数和各周期覆盖。
对现有 966 文件做过只读覆盖检查：层次训练/验证/测试为 4789/978/984 个窗口，
三种周期均保留；验证和测试各有 511 个完整 16 片段窗口。以运行时 coverage 为准。
高低点距离及高点年龄沿用原几何定义，但范围为实际有效的过去 n-1 个片段，
年龄归一化到 0–1。因此不冒充固定 1920 根基准上的直接提升。

## 三阶段

1. local：200 轮，原 128 根历史重建损失，micro=16，梯度累积有效 batch=32，lr=3e-4。
2. aggregate：固定最佳局部模型，提取不可变的局部 teacher 向量；30 轮，micro=2，
   有效 batch=16，lr=3e-4。训练历史聚合与全局解码。
3. joint：加载第二阶段最佳模型，解冻局部模块，20 轮，micro=1，有效 batch=8，
   lr=3e-5。固定 teacher 仍来自第一阶段，避免监督向量随联合优化一起漂移。

后两阶段损失为全局历史重建 + .1×teacher 向量 SmoothL1 + .1×锚点偏移 SmoothL1
+ .1×历史结构标准化 MSE。joint 再加 .25×局部直接重建约束。
历史结构标准化只使用训练集。局部训练维持已有 7 通道和原差分重建项；没有新增未来监督。
每阶段独立初始化 AdamW，weight_decay=.01、clip=1。阶段随机种子依次 42/43/44。
这是预先声明的学习过程，不是宣称不同阶段 loss 可以直接比较。

FP32 训练，局部及上层按层使用 activation checkpoint；保持随机状态，换取较小激活显存。
训练前在 AutoDL 做一次性 synthetic GPU forward/backward/optimizer-step 预检，预检模型丢弃，
不会更新真实训练权重。preflight.json 记录每阶段峰值显存。真实数据训练尚未在本地运行。
预检不是整个训练过程永不 OOM 的保证，实际缓存和保存最佳权重还有额外占用。

每阶段 best.pt 按该阶段验证综合损失选择，last.pt 保存模型、优化器、随机状态、
历史与最佳候选。阶段间切换加载前阶段 best.pt；中断恢复不重置当前优化器。
旧最佳文件字节在无改进时保留，避免恢复导致下游缓存指纹改变。
运行完三阶段后才运行测试评价、512 维 PCA、层次重建、多尺度诊断和冻结状态读出。
local_metrics.json 对应最终联合微调后的局部模块。global512 与 local512 状态读出
使用同一批层次窗口，不能与旧的 8936 窗口结果混比。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_REFERENCE=checkpoints/babel_r1_s42/manifest.json
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_LARGE_LOG=logs/babel_large512_s42.log
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_large_autodl.sh all > "$BABEL_LARGE_LOG" 2>&1 &
tail -f "$BABEL_LARGE_LOG"
```

all 自动依次执行预检、局部训练、teacher 缓存、聚合训练、联合微调和评价。
可单独用 preflight 先检查显存，完成后同配置运行 all 会复用预检记录。
micro 可以通过 BABEL_LARGE_LOCAL_MICRO/AGG_MICRO/JOINT_MICRO 在首次启动前设置，
需整除对应有效 batch。默认 joint 已为 1；若预检失败需保留日志诊断，不能保证 32GB 足够。
所有设置都冻结在根 manifest；改变设置应使用新目录，不能假装精确续训。
阶段预算可在首次启动前用 BABEL_LARGE_LOCAL_EPOCHS/AGG_EPOCHS/JOINT_EPOCHS 设置。

中断后确认原进程停止，保持相同环境，将 nohup 的 > 改成 >> 再运行，自动恢复。
完成后从 download/babel_large512_s42/ 下载 manifest.json、preflight.json、coverage.json、
local_history.jsonl、aggregate_history.jsonl、joint_history.jsonl、local_metrics.json、
hierarchical_metrics.json、artifacts.json。HTML 为本地可打开的重建图，JSON 含完整 OHLC。
权重和 teacher 缓存保留在 checkpoint 目录，不复制到 download。
这是容量与目标一起变化的新路线，不是严格单因素消融；更好的历史重建不等于交易预测能力。
