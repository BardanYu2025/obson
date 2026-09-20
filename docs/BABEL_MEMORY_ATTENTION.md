# 小型历史注意力聚合器与仅锚点对照

本轮开始训练上层模块，但保持局部编码器及记忆缓存不变。任务是已知历史的
高低点距离与高点新旧位置，不预测未来，不生成买卖建议。

## 输入、模型和对照

严格复用远期历史 v1 基准、切分和 511 维原始输入：当前局部向量 256 维，
15 个过去片段各有投影 16 维与相对当前价格的锚点偏移 1 维。
不改用完整历史 256 维向量，避免与线性基准同时改变输入量和模型。
固定时间位置以 1/15 到 15/15 表示片段新旧，加入 token 投影。

模型共 76,355 参数，hidden=64、4 头、1 层历史片段自注意力、1 层当前查询交叉注意力。
当前向量直接投影为 64 维，作为查询；读取历史后与当前表示残差相加并归一化，
再拼接当前表示，通过小 MLP 输出三个标准化回归值。dropout=.1。
历史自注意力可双向读取这 15 个片段，因为它们全部位于当前窗口之前，
不是允许当前 bar 读取未来。

full：使用全部输入。
anchors_only：保留当前向量、历史锚点与时间位置，只把历史投影坐标屏蔽为零。
两组架构、初始化、batch、训练轮次和目标一致。屏蔽在共享训练期标准化后执行。
这回答的是历史片段投影在锚点与当前向量之外是否有用；不声称仅锚点组完全无 embedding。
同时补充仅锚点线性 ridge，与原线性 ordered/masked/shuffled 结果一起保存。

## 训练与验证

两组从头顺序训练，各 50 轮、batch=256、seed=42、AdamW lr=3e-4、
weight_decay=.01、clip=1，无调度/早停。输入和目标标准化只用训练集拟合。
目标及综合选择指标沿用原协议：三项标准化回归误差的平均 MSE。
每轮验证，按验证 MSE 保存 best.pt；随机初始化也参与候选。
每组 last.pt 保存权重、优化器、随机状态、标准化参数、历史与最佳候选。
两组均完成后才评估测试。按相同元数据自动恢复中断的完整轮次；已完成组
跳过训练，仅重建可下载报告。不要同时启动两个进程写同一目录。

测试额外打乱片段与锚点的配对整体，固定时间槽位不变，用于干预已训练模型。
这与上一轮“单独训练 shuffled ridge”不同；干预也会改变输入分布，不单凭退步
宣称学到了因果机制。高低点距离本来对顺序不敏感，主要看高点位置的顺序依赖。

当前内部表示是这三个任务监督下的 128 维拼接，不是新通用 256 维 bar embedding。
结果好只能证明固定历史基准上的读取改善；后续仍需独立留出期和更多任务。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_MEMORY_RUN=checkpoints/babel_memory_z256_s42
export BABEL_MEMORY_BENCHMARK_RUN=checkpoints/babel_memory_benchmark_s42
export BABEL_ATTENTION_RUN=checkpoints/babel_memory_attention_s42
export BABEL_ATTENTION_LOG=logs/babel_memory_attention_s42.log
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_memory_attention_autodl.sh all > "$BABEL_ATTENTION_LOG" 2>&1 &
tail -f "$BABEL_ATTENTION_LOG"
```

复用上一轮的缓存和基准报告，GPU 只训练这个小模块，CPU 完成数据准备和线性对照。
源数据、缓存或协议指纹不一致会停止。参数在代码中固定，避免旧训练环境变量干扰。
中断后确认旧进程结束，再将同一 nohup 命令改成追加日志 `>>` 执行，自动恢复。
完成后复制到 download/babel_memory_attention_s42/。
回传 manifest.json、attention_metrics.json、full_history.jsonl、anchors_only_history.jsonl。
