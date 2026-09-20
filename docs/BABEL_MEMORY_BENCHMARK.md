# 远期历史评价基准 v1

上轮均值和拼接历史没有提高原状态规则的读出。本轮先固定明确依赖远期历史的
监督与控制基线，不训练新的神经网络，也不修改旧编码器、记忆缓存和状态标签。
使用已生成的 z256 记忆缓存；CPU 拟合轻量 ridge 回归。

## 样本与目标

当前窗口为 [t-127,t]；完整观察范围为 [t-2047,t]，仅使用有完整 16 个片段的样本。
远期目标取此前 [t-2047,t-128] 共 1920 根，不含当前 128 根。
所有分区复用原当前窗口划分。验证/测试读取更早训练期行情属于历史背景，
不是未来泄漏；样本重叠，不能当独立观测。报告各分区实际样本数。

三个目标：

1. distance_from_prior_high = asinh(log(C_t / H_past) / sigma_t)。
2. distance_from_prior_low = asinh(log(C_t / L_past) / sigma_t)。
3. prior_high_age：最高点所在的过去片段序号 / 14，0 最近，1 最远；
   相同高点优先取最近片段。不是首次触及时间，也不是预测标签。

H_past/L_past 是上述远期 1920 根的真实最高/最低价。
sigma_t 沿用编码器的因果 prior RMS，不用全量数据拟合；这些距离不是 bp 或概率。
age 的 MAE 乘 14 可以换算为历史片段序号误差（不是精确 bar 时间误差）。
回归值不做截断；报告 MAE、RMSE、R²，R² 可能为负。

## 可用输入与对照

当前向量保留 256 维。每个历史片段向量以 seed=42 的固定随机正交投影降至
16 维，另加 asinh(log(窗口前价格锚点 / C_t) / sigma_t)。15 个片段按近到远
拼成 255 维；合计 511 维。随机投影没有拟合任何数据，目的是控制线性基线成本。
这些锚点来自已发生行情，但不直接包含各片段真实最高/最低价。

- ordered：保留真实顺序、向量投影和价格相对位置。
- masked：当前向量相同，全部远期输入置零，独立拟合；避免只用测试期置零
  造成分布外输入而误判历史重要性。
- shuffled：每个样本独立、确定性打乱过去片段与其价格锚点的配对整体，
  保留当前向量，独立拟合；不跨样本、合约或时间引入别的历史。
- train_mean：仅用训练集目标均值预测。

高低点距离是集合信息，对片段置换本来应不敏感，因此不能要求它们在打乱后
一定下降；最高点的新旧位置才是顺序敏感目标。线性模型本身有限，打乱与有序
差异可能反映其表达能力，不是神经注意力机制的证据。
投影会丢失信息，失败不能直接解释为原 256 维编码器没有远期信息。

每种表示分别拟合训练集输入/目标标准化，alpha=1/10/100 根据验证集三个标准化
目标的平均 MSE 选择；测试只报告，不挑参数。保留完整训练窗口、缓存和权重指纹。
最近 128 根本身可能与旧高低点相关，所以 local 基线不必接近常数基线；
评价重点是完整历史是否提供额外可用信息。

这套基准描述历史几何关系，不证明这些价位有交易支撑阻力效应。
测试日期已反复用于开发讨论，后续独立泛化需要新留出期。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_MEMORY_RUN=checkpoints/babel_memory_z256_s42
export BABEL_MEMORY_BENCHMARK_RUN=checkpoints/babel_memory_benchmark_s42
export BABEL_MEMORY_BENCHMARK_LOG=logs/babel_memory_benchmark_s42.log
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_memory_benchmark_autodl.sh all > "$BABEL_MEMORY_BENCHMARK_LOG" 2>&1 &
tail -f "$BABEL_MEMORY_BENCHMARK_LOG"
```

不需要 GPU 推理，不需要新的 200 轮训练。旧缓存必须仍在原目录，不从 download
中读取；缓存或数据指纹不匹配会报错。中断后可用相同协议重新运行，追加日志。
结果复制到 download/babel_memory_benchmark_s42/，回传 manifest.json、benchmark_metrics.json。
以此作为后续带时间/价格信息的可学习历史模块的固定评价，不因成绩调整定义。
