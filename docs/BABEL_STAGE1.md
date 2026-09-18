# 阶段 1：冻结历史压缩模型，定位信息损失

基线为 `babel-history-ae-v1`，128 根 → 128 维，4 层编码器，seed 42。
用户回传的基线选择 epoch 30，测试 8936 窗口；AE 综合损失 0.1244617114，
PCA 0.1226296222；收盘 log-price 误差 33.6160 / 35.2535 bp。
当前结构读出 BA 0.563103；震荡召回 0.106667。这些是已有结果，不是验收阈值。

本阶段不改变输入、模型、损失、数据分割、阈值或权重，也不延长训练。
诊断命令先核对原来的数据指纹和权重 schema，再只执行 eval/no_grad。
读取 best.pt、manifest.json、history.jsonl、ae_metrics.json 的 SHA256，
运行结束再次核对，记录在诊断报告里。原始文件不覆盖。

## 固定诊断定义

- 位置：窗口四等分，Q1 最早，Q4 最近。
- 尺度：1、4、16 根收盘变化误差、相关系数和标准差保留比。
- 状态：终点的中尺度规则状态（上升、下降、区间、未形成）。不是声称整个
  窗口内每根都是这个状态。
- 跳空：绝对 log(open/previous_close) 超过窗口 log(high/low) 中位数的 3 倍。
  这是本阶段预先固定的描述分组，不是删数据标准，也不把真实跳空视为异常。
- 同时对 AE、同样设置的训练期 PCA 计算；额外按原始周期分组。
- 每组按窗口等权汇总均值和中位数。窗口重叠，不把样本数视为独立观测数。
  常数路径不计算相关系数或标准差比，单独报告有效样本数，不填零。
  四分区内的变化只在分区内计算；短分区不支持的跨度省略。

先根据位置差异判断是否存在明显历史遗忘，再判断小尺度变化是否在所有位置
都被压平，最后查看震荡/跳空分组与普通窗口的差别。不仅看误差数值，也要看
原始变化幅度和标准差保留比；组间难度不同，不能把绝对误差直接解释为偏好。
未经验证，不据六个展示样例替全测试集下结论。

## AutoDL：只诊断，不训练

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_REFERENCE=checkpoints/babel_r1_s42/manifest.json
export BABEL_AE_RUN=checkpoints/babel_ae_r1_s42
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
export BABEL_AE_LOG=logs/babel_ae_stage1.log
export BABEL_BATCH_SIZE=32
mkdir -p logs
nohup bash scripts/babel_ae_autodl.sh diagnose > "$BABEL_AE_LOG" 2>&1 &
tail -f "$BABEL_AE_LOG"
```

运行结束自动复制到 `~/autodl-tmp/download/babel_ae_r1_s42/`：
`ae_diagnostics.json`、`ae_diagnostics.md`、`diagnostics.log`，及已有实验报告。
此诊断不重新拟合神经网络，也不重跑结构分类探针；只重建训练期 PCA 基线。
收到全量报告后才决定阶段 2 的具体实验；本提交不启动阶段 2。
