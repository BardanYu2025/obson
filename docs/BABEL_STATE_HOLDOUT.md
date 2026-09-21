# 冻结行情状态的跨品种留出审计与评价

## 研究目的与范围

上一轮未直接监督的12描述任务出现跨种子迁移与统计增量，但仍重复使用研究端点。本轮先核查RM、au、c、cs、ru是否在可追溯记录内未使用，再对合格品种一次评价冻结候选。主变量是评价数据，不变更模型、损失、归一化或读出预算。

这是跨品种评估，沿用原test时间范围，不是新的未来时段。注册表外、已删除或没有记录的人为研究无法由程序证明不存在；报告明确限制在可审计来源，不出具“绝对从未见过”的证明。旧价格seed43超过2%保留门槛的事实继续显示，不自动替换模型。

## 评分前资格审计

- 扫描指定registry中全部manifest.json，排除当前输出；沿根迁移实验的显式来源路径和manifest SHA追溯依赖。缺少声明的祖先、无法解析的manifest SHA、无来源的叶节点或不可读注册清单均阻止资格确认。源数据快照同时计入训练、验证和旧研究测试，不能只排除旧训练端点。
- 所有注册清单的源品种都列为已使用，即使不是当前权重的直接祖先；其他同类留出运行一旦写入评分锁，也将其合格品种计为已使用。候选品种大小写归一后匹配。这是保守排除策略。
- 校验历史源CSV在其记录起止区间的内容hash。缺失历史CSV或内容不符，不能证明旧新样本无重叠，流程阻止评价。
- 核查候选完整OHLCV/OI、正价格、唯一递增时间及bar间隔；同品种任何受支持周期文件不合格，整个品种排除，避免改变此前一日成交量选合约逻辑。
- 排除已记录合约别名、匹配数据hash，以及至少128行时间/OHLCV完全相同的疑似复制文件。候选品种之间发现同类重复时双方均排除。行hash是内容去重筛查，不证明经改写、缩放或另行聚合的数据绝无重复。
- 候选资格只根据来源/质量，不能看到模型分数后筛选。无合格品种或无合格窗口时退出码3，状态blocked并自动打包；运行错误则failed。不得手改audit_status绕过。

large512的根manifest记录from-scratch训练及完整数据快照；其reference_sha256是数据协议来源，不是继承权重。实际权重祖先由short/source/fusion等manifest引用及现有来源验证链核对。未声明的外部依赖仍无法自动发现。

## 模型与读出锁定

- control/aux020×seed42/43，均使用activity_alignment100固定第100轮编码器；价格头使用price_readapt512原验证选中best。
- 迁移10种表示：current、statistics、四组embedding、四组embedding+statistics。MLP1701/1702直接加载上一轮选定权重及训练归一化，先重放旧研究集逐窗口误差，再锁定。
- Ridge此前未存系数：使用原训练分区、原已选alpha恢复闭式解，核对旧测试误差；不重新用留出选alpha。固定权重与归一化保存在本轮frozen目录，不覆盖父产物。
- 原20量仓目标使用同样恢复并核验的Ridge，含当前输入对照。旧量仓MLP未持久化权重，本轮明确不评价它，不把重新拟合冒充原权重。迁移MLP完整保留两个种子。
- 本轮没有神经网络优化器更新，不补统计MLP预算。沿用上一轮100轮预算的条件内结论，不声称统计基线收敛。本轮也不评价长编码器、趋势分类头或未来预测。
- 在读取新样本评分前写入evaluation_lock.json，绑定来源身份、资格审计与冻结读出hash。恢复运行要求配置、来源、缓存均一致；读出/缓存已完成部分校验后复用。

## 输入与端点

保持按原始合约编码，不拼连续合约。每根bar的价格与量仓28通道预处理沿用训练代码；EMA/波动估计从合约历史前缀因果递推。神经GRU在原test分区起点重置，合约/周期之间完全独立，按128bar分块保留状态。

端点沿用long_run记录的test会话边界；只有末128根均在该分区且端点满足前一交易日成交量主合约选择才计入。每合约固定stride128，每品种周期按时间排序等距最多256端点，避免高度重叠窗口和单品种主导。cap在运行前锁定，不按结果调整。评分前导出实际端点、各任务掩码、异常量仓覆盖。

价格重建目标为末64根，锚定其前一根收盘。观察信息与监督均不包含端点之后行情。神经状态可以包含超过128根的已观察历史；statistics只含128根，此差异延续前一阶段，不能把增量完全归因于架构。

## 评价与解释

- 迁移主指标：同一完整12任务支持集的标准化MSE（训练尺度），四任务族及逐目标R²/支持数同时报告。
- 量仓：原20目标逐项、过去15项及clean子集，当前输入对照保留。clean只审计末17根目标跨度，不证明整个前缀完全无异常。
- 价格：close bp、detail、变化相关性/标准差比；对照aux与同种子control，沿用2%价格保留门槛作解释，旧失败不抹去。
- 分品种/周期/月报告。周配对bootstrap在共同支持集计算，少于50端点不构成正向支持，少于5周不输出区间。只有五个候选品种，窗口数不能当独立品种样本数。
- 总体、所有种子、失败与缺失都保留。任务族/分组多重比较属于探索性解释，无自动promotion。

所有指标一起计算，四个编码器依次用批量合约流回放，避免复制完整原始特征银行。GPU占用不高并非训练卡死；CPU来源核验和因果特征生成阶段GPU可以空闲。

## AutoDL命令

需保留state_transfer512及其完整上游产物和原始合约CSV；不需要上传或复制checkpoint。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_HOLDOUT_TRANSFER=checkpoints/babel_state_transfer512
export BABEL_HOLDOUT_REGISTRY=checkpoints
export BABEL_HOLDOUT_RUN=checkpoints/babel_state_holdout512
export BABEL_HOLDOUT_LOG=logs/babel_state_holdout512.log
export BABEL_ROOT=/root/autodl-tmp/data/contracts
export BABEL_HOLDOUT_STREAMS=32
export BABEL_HOLDOUT_CAP=256
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_state_holdout_autodl.sh all > "$BABEL_HOLDOUT_LOG" 2>&1 &
tail -f "$BABEL_HOLDOUT_LOG"
```

无论成功、来源阻断还是运行失败，自动导出：
`/root/autodl-tmp/download/babel_state_holdout512_reports.tar.gz`。
包含来源/质量审计、锁定协议、旧读出复核、实际覆盖、指标、逐窗口误差、日志与状态，不含.pt/.npy。来源阻断时没有新模型指标是预期行为，传回同一包即可。

只做审计：`bash scripts/babel_state_holdout_autodl.sh audit`。
手动重打包：`bash scripts/babel_state_holdout_autodl.sh export`。

## 本地验证范围

仅合成数据的来源失败保护、别名重复、目标掩码、Ridge恢复、合约流与未来后缀不变性、完整冻结前向及报告导出；不做优化器更新。真实AutoDL权重、全量原始数据和CUDA数值兼容性由实际运行核验，不将本地通过视作跨品种结果。
