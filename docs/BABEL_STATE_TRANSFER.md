# 冻结行情表示的探索性迁移评价

## 目标和证据边界

本轮回答：在不修改编码器的条件下，量仓增强状态是否更容易读出未直接加入原辅助损失的历史状态，并在简单因果统计之外提供信息？原20目标辅助头不再训练，也不把新任务标签反向传给编码器。

“未直接监督的新描述量”不等于未见原始行情。这些指标仍由已观察的价格/量仓历史派生；测试端点已经反复用于研究。本轮明确为探索性迁移评价，不能称为独立留出确认或市场机制理解。原价格重建seed43未通过2%保留门槛的结果随报告继承，不因新读出改善自动撤销。

## 固定12个任务

每个任务采用32/64/128根已观察bar窗口。仅用窗口内部h−1次收盘变化，不借用窗口前一根价格；窗口不跨缓存中的合约/训练验证测试分区。

从现有无裁剪编码坐标恢复log收益：`r=(sinh(x_gap_log_percent)+sinh(x_body_log_percent))/100`。成交量变化为窗口内部`diff(10*x_log_volume)`，持仓描述量沿用可观察的`asinh(100*ΔOI/baseOI)`。

1. 方向持续性：`sum(r)/sum(abs(r))`，全零路径定义为0。它是已观察路径的有符号效率，不是未来上涨概率，也不是交易标签。
2. 波动变化：窗口收益分为前后两半，`log((mean(r后²)+1e-8)/(mean(r前²)+1e-8))`。奇数个收益按前floor(n/2)、后其余拆分。描述波动变化，不等同于不可观测的“真实市场状态”。
3. 价量联动：`corr(r, Δlog(1+V))`。
4. 价仓联动：有效记录上的`corr(r, asinh持仓变化描述量)`。

常数序列导致相关系数不可定义时置为缺失监督，而非0标签。价仓至少80%的内部收益对应有效持仓变化；延续零成交量却非零持仓变化、`abs(ΔOI/V)>1`的目标掩码。保留原输入，不自动删除异常行情。所有掩码、各分区支持数和训练归一化统计都会导出。

这些是预先声明的测量任务，不把规则输出重新训练成编码器的唯一目的。

## 10种输入表示与公平性

- current：当前28通道，包含已有因果EMA上下文，不能称为没有历史的原始单bar。
- statistics：当前28通道，加三个尺度的11项通用统计，共61维。每个尺度包括收益、绝对收益、log成交量、log成交量变化的均值/标准差，及有效持仓变化的均值/标准差和有效率。不直接把上述任务标签/交叉相关系数作为输入。
- control/aux020各两个编码器种子，共4组512维冻结状态，来自price_readapt的已校验端点缓存。
- 上述4组分别拼接statistics，共4组573维输入，检验在简单统计之外的增量价值。

简单统计可以表达某些目标的组成量，这是强而透明的对照，不刻意削弱它。所有任务都能由完整历史按公式直接计算，所以本轮评价的是压缩状态的可读性和信息覆盖，不能证明神经网络比直接算指标更必要。

全历史统计使用同一个128bar观察窗口；循环状态可携带更早前缀。输入维度/参数量会报告。MLP统一128隐藏维和搜索预算，但不同输入维度意味着不同参数量，不能宣称逐参数等容量；control/aux020的对应比较维度相同。

## 读出训练与选模

每种输入使用：

- Ridge：逐目标，仅用训练拟合输入/目标归一化，验证选择alpha=1/10/100/1000。
- MLP：输入→128/GELU→12，AdamW LR1e-3，batch256；读出种子1701/1702分别跑weight_decay0.001/0.01，每个配置100轮。仅训练读出参数，梯度裁剪1。
- MLP目标标准化只拟合有效训练目标；优化按有效目标等权SmoothL1，验证以等目标权重标准化MSE选epoch和decay。两种读出种子分别报告，不用测试挑一个。
- 选择过程只接收训练/验证数组；选定decay后才对对应模型执行测试预测。checkpoint保存模型、优化器、RNG、完整history和归一化信息。中断重跑恢复未完成trial，已完成representation校验hash后跳过。
- epoch0若被选中明确标注trained_selection=false，不能触发正向研究信号。100轮是预算，不证明所有读出已充分收敛。

## 汇总和对照

主汇总为12任务等权标准化MSE，只在12项都有效的同一测试端点集计算，至少50端点才可触发研究信号。分任务/任务族同时报告支持数、误差和R²；不能用总分掩盖某个任务族退步。

按自然周配对bootstrap比较：强约束对control、当前输入、统计特征；embedding+statistics对statistics；两种embedding拼接统计后的相互比较。正向信号要求两个编码器种子、线性和两组MLP读出都支持主汇总改善。任务族区间仅作探索性分析，不冒称多重比较校正后的独立发现，也不自动升级模型。

保留所有100轮history、选模参数、逐端点误差（JSON中的缺失值为null）、完整对照及旧价格筛选结果。主包不含.pt/.npy。

## 独立留出审计

coverage_audit.json记录已有测试时间范围、上游预训练源及分区边界。可选扫描BABEL_ROOT内CSV文件名与尾部时间，将未在该源清单或晚于源结束时间的文件列为“未经验证的候选”。该扫描不会加载候选进入实验，不证明其未出现在其他预训练/研究中，也不证明数据质量或足够覆盖。当前报告始终independent_holdout=false。

不能把缓存中没选作端点的合约/时间直接当新留出，因为其bar可能已作为前缀或早期预训练输入。新的独立评估需要单独审计完整来源和样本覆盖。

## AutoDL

保留完整price_readapt512缓存及上游产物。此次训练只有小读出头，默认两种输入表示并行；低显存和较快epoch是正常现象。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_TRANSFER_PARENT=checkpoints/babel_price_readapt512
export BABEL_TRANSFER_RUN=checkpoints/babel_state_transfer512
export BABEL_TRANSFER_LOG=logs/babel_state_transfer512.log
export BABEL_TRANSFER_EPOCHS=100
export BABEL_TRANSFER_BATCH=256
export BABEL_TRANSFER_JOBS=2
export BABEL_ROOT=/root/autodl-tmp/data/contracts
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_state_transfer_autodl.sh all > "$BABEL_TRANSFER_LOG" 2>&1 &
tail -f "$BABEL_TRANSFER_LOG"
```

BABEL_ROOT仅用于候选留出清单，不改变冻结缓存或训练数据；目录不存在会明确标记清单不可用，本轮缓存评价仍可运行。

全部结束自动导出：`/root/autodl-tmp/download/babel_state_transfer512_reports.tar.gz`。失败也打包日志并标明非零退出码。手动打包：`bash scripts/babel_state_transfer_autodl.sh export`。重跑all自动恢复；只重汇总用evaluate。配置变更需新目录。

本地只做合成目标/掩码/恢复/报告测试和无优化器更新的检查，正式MLP训练在AutoDL。完成代码不等于已有迁移成功证据。
