# Babel AR R1：因果 bar 表示与下一根生成

本轮目标：把输入编码、上下文表示、下一根生成和自动评价连成完整链路。
它是新实验，不加载或覆盖 `babel-v1` / `babel-bar-representation-v1` 权重。
单个种子的可行性实验不代表已获得通用 embedding 或交易价值。

## 表示与编码

`ARModel(x)["h"]` 为 `[batch, bars, 128]`，每个 h_t 只看到截至 t 的输入。
`["seq"]` 直接等于最后一个 h_t，不另加聚合模块，也不把 seq 回灌到历史位置。
位置编码和三角注意力掩码确保窗口内所有位置因果；窗口切换会改变上下文。

每根 bar 以**上一根收盘价**为参考：

- g = log(O / previous_close)：跳空；b = log(C / O)：实体。
- u = log(H / max(O,C))；d = log(min(O,C) / L)：非负上下影。
- 波动状态为已观测 close-to-close 对数收益平方的 EMA，alpha=2/65，
  初始标准差 0.01，下限 0.0001。编码 t 时只用 t-1 的状态；消费 t 后
  更新状态，以供预测和编码下一根。这是收益 RMS，不是去均值标准差。
- 输入保留 asinh(100 × g,b,u,d) 与 asinh((g,b,u,d)/sigma)，并显式输入
  log(sigma)。asinh 是可逆压缩，不把极端行情裁剪成同一个值。
- 其余输入为 log1p(volume)/10、log1p(OI)/10、OI 可用性、log1p(period)/6、
  log(相邻 bar 起始时间间隔 / 周期)。总计 14 维。

正价格、OHLC 关系、有限数值是硬校验。首根以自身开盘初始化价格，所有训练
窗口有预热区。不做复权或消除真实跳空。数据仍按单合约，样本终点用滞后主力选择。
数据源指纹与旧运行清单完全一致才允许训练。尚未增加交易日历或流动性过滤。

## 下一根输出与训练

主干：4 层、128 维、4 头、FFN 512、dropout 0.1；窗口按批混用 128/256，
前 64 根只提供上下文。用 h_t 预测 y_(t+1)，明确右移一根；输入中没有未来目标。
训练与评价的监督位置和未来目标都限制在对应时间集合及同一合约内。

下一根的七个坐标：

1. asinh(g / sigma_t)
2. asinh(b / sigma_t)
3. sqrt(u / sigma_t)
4. sqrt(d / sigma_t)
5. log1p(volume)/10
6. log1p(OI)/10
7. log(时间间隔 / 周期)

分布头是 **5 分量混合分布**。前两维为正态，后五维为折叠正态；分量内部
条件独立，共享分量选择提供跨通道依赖。非负上下影解码后保证 OHLC 合法。
模型同时生成 volume、OI 和间隔，滚动过程不读取真实未来协变量。
这是按 bar 自回归、bar 内联合混合分布，不是每个价格字段再自回归。

损失为按有效坐标数归一的联合 NLL + 0.1 当前输入重建 SmoothL1；
缺失 OI 的概率因子被边缘化，重建 OI 项不计分。结构规则不参与本轮训练。
重建输入可能通过复制完成，不能独立证明模型理解了序列。
AdamW lr=3e-4、weight_decay=0.01、梯度裁剪=1、batch=32、stride=16、30 epochs。
验证集窗口终点 NLL 选 best.pt。输出目录必须为空，无断点恢复。

NLL 是**变换坐标中的连续密度分数**，可能为负，不是原始价格密度或准确率。
零跳空、零上下影与时间间隔会出现点质量，连续折叠正态只作近似；log_std
限制为 [-5,3] 防止方差无限收缩，仍须观察是否过度拟合这些重复值。
价格 tick 与成交量整数约束尚未编码，生成的是连续数值。

## 评价

1. 单步 NLL：对比训练期拟合的无条件分布，以及以当前 bar 几何与量仓为中心、
   训练期估计残差尺度的持续性分布。所有基线采用同一变换与折叠分布评价。
   同时报告按周成对 bootstrap 差异，负值有利于模型。
2. 自由滚动：固定抽取最多 32 个测试窗口，每个采样 8 条路径、滚动 16 根。
   每步只消费已生成的 OHLC、volume、OI、间隔；价格基准和波动状态随之更新。
   输入不使用真实未来价格、ATR 或时间戳。输出 1/4/16 根的 log-return 经验
   CRPS、路径中位数误差、路径离散程度，与价格不变基线对比。前三个窗口保存
   生成 OHLC 和真实未来，供事后检查；真实数据在生成结束后才读取。
3. 冻结表示：预训练末端 h、同架构随机 h、最近 16 根完整原始编码的平坦向量
   对照。类别平衡岭回归，alpha=[1,10,100] 仅在验证集选择。
   两个任务分别为当前中尺度结构状态、未来 8 根价格方向。前者检验规则结构
   是否可解码，不把规则当唯一形态真值；后者是独立的预测用途。两者分开报告。

生成溢出明确计作失败，不用截断伪装正常价格。CRPS 只对完成的路径计算，
可能有生存者偏差，必须一起看失败数；八条路径、32 个窗口的估计噪声较大。
预测的时间间隔不受交易所日历约束，滚动按“未来第几根记录”而不是指定时刻评价。
长期价格/波动失控、路径塌缩、极少量样本的好成绩均不能通过验收。

旧测试日期已被多轮查看，不能称全新盲测。此轮同时改变了编码、目标与输出头，
不用于归因“增加层数有用”。未来需要新的时间留出、多个种子与容量对照。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_REFERENCE=checkpoints/babel_r1_s42/manifest.json
export BABEL_AR_RUN=checkpoints/babel_ar_r1_s42
export BABEL_BATCH_SIZE=32
export BABEL_EPOCHS=30
export BABEL_SEED=42
mkdir -p logs
nohup bash -c 'bash scripts/babel_ar_autodl.sh train && bash scripts/babel_ar_autodl.sh evaluate' > logs/babel_ar_r1_s42.log 2>&1 &
tail -f logs/babel_ar_r1_s42.log
```

全程 CUDA runner；前期文件读取、因果编码在 CPU 上完成，日志分阶段显示。
本地测试只运行合成数据的前向、反向与评价，不执行正式训练。
训练成功后自动评价，返回新目录的 `manifest.json`、`history.jsonl`、`ar_metrics.json`。
评价中途失败可以单独运行 `bash scripts/babel_ar_autodl.sh evaluate`。
权重 schema 为 `babel-ar-v1`，不可交给旧版 index/serve 命令。

Python 中读取权重使用 `torch.load(..., weights_only=True)`，核对 schema/features，
构造 `ARModel(**checkpoint["config"])` 后加载 `checkpoint["model"]`。
调用 `.eval()`，在 `torch.no_grad()` 下编码已完成的 bar。历史与生成必须使用
同一 `ar_codec` 与相同状态初始化；不要用窗口末端价格重算前面各根的编码。
