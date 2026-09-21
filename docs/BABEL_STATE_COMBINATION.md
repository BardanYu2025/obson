# 冻结价格／量仓状态组合：匹配读出对照

## 本轮目标

跨品种报告确认量仓增强状态保留更多历史量仓信息，但价格有代价，新描述任务增益不全面。本轮使用现成control与aux020固定状态，检查组合是否值得增加一条推理路径。先验证互补信息的可读性，不训练编码器，不预设需要更大模型或重新压缩。

原test和已打开的RM/au/c/cs/ru五品种都属于研究评价；后者命名cross_research，不再声称是新的独立留出。拟合和选模只用原train/val。规则描述任务仅用于诊断压缩信息，不是行情表示的最终目标、市场机制识别或未来预测。

## 18种表示与两类任务

沿用10种输入：current28、statistics61、control/aux两个种子的512维状态、四种状态分别拼接statistics后的573维状态。

新增8组：

| 种子 | dual（1024维） | price_pair（1024维） |
|---|---|---|
| 42 | control42 + aux42 | control42 + control43 |
| 43 | control43 + aux43 | control43 + control42 |

这四组再各拼接statistics，得到四组1085维输入。拼接顺序固定，第一段始终是对应control。price_pair与dual具有相同输入维数、读出参数量及两路编码成本，用来区分新信息与增加维数／第二个编码器。两个price_pair只是同一对control不同排列，不能当独立编码器重复；全部种子共享预训练。

两类任务分别训练读出头，不混合选模：

1. transfer：既有4类历史描述量×32/64/128根，共12项。主指标是12项完整支持端点的等权标准化MSE。
2. activity_past：过去1～16根统计、lag4、lag16各5项量仓，共15项。排除当前5项目标，以免当前输入复制主导评分。此类目标参与过aux监督，不冒称未监督任务迁移。

任务和缺失掩码直接复用已校验缓存，不改标签定义。current28比之前量仓评价的current10包含更多价格/EMA信息，statistics包含历史通用统计，因此本轮量仓基线和旧报告不同；所有18组在新协议下重新拟合同预算读出，不混用旧量仓MLP80轮的分数。

## 冻结来源与价格路径

- 来自activity_alignment100固定第100轮编码器与price_readapt原选中价格头。old train/val/test直接复用state_transfer缓存。
- 五品种缓存上重新回放四个编码器，先核对原holdout价格objective，确认回放对应原候选；将状态存入新缓存。无需重新读取原始CSV，也不修改旧报告/权重。
- 价格解码调用固定为`control_head(joined_state[:512])`。在两个研究集逐批核对直接control输出、组合路由输出以及大幅扰动aux部分后的输出完全相等。
- 价格不变是保留原路径的设计结果，不能计为本轮训练带来的价格提升，也不撤销单独aux在旧实验中的价格保留失败。
- 1024维需要两个512维编码器，1085还包括统计特征，尚未蒸馏成单一512维状态。实际部署价值需与维数、计算和状态存储成本一起判断。本轮不增加长历史Transformer分支。

## 读出预算、选择与恢复

每类任务、每个表示均使用：

- Ridge：训练拟合归一化，验证逐目标选择alpha=1/10/100/1000；相同有效行掩码的目标共享矩阵求解，保留各自最优alpha及系数。拟合接口只接收train/val。
- MLP：输入→128/GELU→目标数，AdamW LR1e-3，batch256，weight_decay0.001/0.01，各100轮，种子1701/1702分别选epoch与decay。训练目标是掩码等权SmoothL1，选择指标为验证标准化MSE。
- 共18表示×2任务=36个worker，144条MLP训练轨迹。默认四个worker并行，复用同一来源缓存，报告参数量、完整轨迹和所选轮次。
- 100轮是匹配预算，不表示全部收敛；选末轮会显式标记selected_at_budget_end。不同输入维数的MLP参数量不同，只有相应dual/price_pair维数和参数量严格一致。
- 选择过程不读取test/cross_research数组。全部选择完成后先写selection_lock，再读取两个评价集。读出权重、归一化、Ridge系数持久化。
- epoch0若被选中，trained_selection=false，不触发正向信号。保留优化器/RNG实现trial续跑；完成worker校验文件hash后跳过。配置变化必须使用新目录。

## 评价和阶段判断

两份研究集单独报告、每类任务单独报告，不拼接出一个综合好看分数。

每个种子比较dual对control/aux/price_pair，三组拼接statistics后也相互比较；另比较dual+statistics对statistics、对dual。不按测试挑种子、任务或品种。主要关注加入量仓状态是否超过control+statistics与等维price_pair+statistics，同时相对单独aux+statistics是否有收益。

保留逐目标R²/支持数、任务族、品种/周期/月分组、共同支持集周配对bootstrap。至少50端点且5周才标记区间有足够支持；正向探索信号要求Ridge和两个已训练MLP选择均支持改善。多重任务族/分组区间未校正，不称独立发现，不自动替换主模型。

若收益只来自维度增加、复制统计或个别种子，不将组合升级为通用表示。即使价格路径不变，也不能用它掩盖读出无增量。组合价值充分后，再考虑是否需要压缩、联合训练或新数据确认。

## AutoDL

保留state_transfer512、state_holdout512及其完整上游缓存/权重。正式神经读出训练只在AutoDL。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_COMBINATION_TRANSFER=checkpoints/babel_state_transfer512
export BABEL_COMBINATION_CROSS=checkpoints/babel_state_holdout512
export BABEL_COMBINATION_RUN=checkpoints/babel_state_combination1024
export BABEL_COMBINATION_LOG=logs/babel_state_combination1024.log
export BABEL_COMBINATION_EPOCHS=100
export BABEL_COMBINATION_BATCH=256
export BABEL_COMBINATION_STREAMS=32
export BABEL_COMBINATION_JOBS=4
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_state_combination_autodl.sh all > "$BABEL_COMBINATION_LOG" 2>&1 &
tail -f "$BABEL_COMBINATION_LOG"
```

结束或失败均自动导出：`/root/autodl-tmp/download/babel_state_combination1024_reports.tar.gz`。
包括准备审计、两个数据集全部对照、选择参数、144条history、逐窗口误差和日志，不含.pt/.npy。
手动打包：`bash scripts/babel_state_combination_autodl.sh export`。
只重新汇总已完成worker：`bash scripts/babel_state_combination_autodl.sh evaluate`。

本地仅合成数据和无优化器更新的前向／恢复／报告测试。代码测试通过不代表本轮实验完成或组合有用。
