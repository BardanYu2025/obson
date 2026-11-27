# Babel 当前研究资料

第一次读，先看已经做到哪一步，再看每个研究问题的含义和下一步的交付物。[已完成的工作](BABEL_ROADMAP.md#current-review) · [研究编号速查](BABEL_ROADMAP.md#research-questions) · [实际工作顺序](BABEL_ROADMAP.md#work-sequence)

P0、P1 是上一版的模型方案，P2、P3 是当前要比较的两套方案。[四个模型方案](BABEL_ROADMAP.md#model-recipes) · [首批要交付的能力](BABEL_ROADMAP.md#requirement-fit)

- 想顺着一个样本读，可以依次看[输入哪些 K 线](BABEL_ROADMAP.md#input-views)、[模型总图](BABEL_ROADMAP.md#model-architecture)、[每次更新怎样混合三种任务](BABEL_ROADMAP.md#training-tasks)和[怎样验收](BABEL_ROADMAP.md#delivery-criteria)。
- 想了解已经下载的行情能做什么，看[月份合约资格报告](../../experiments/20261127_history_qualification_v1/reports/completed_months_20261127/qualification.md)和[各品种可用数量](../../experiments/20261127_history_qualification_v1/reports/completed_months_20261127/by_product.md)。价格、方向分类、自然顺序和完整行情描述各有自己的覆盖范围。[各用途的范围初表与下一步](BABEL_ROADMAP.md#next-deliverable)
- 想查具体用途，看[价格能重建到哪里](BABEL_ROADMAP.md#reconstruction-evidence)、[重建后哪些问题可以直接计算](BABEL_ROADMAP.md#reconstruction-implies)、[缺失行情的价格锚点](BABEL_ROADMAP.md#historical-price-anchors)或[行情分类与技术形态](BABEL_ROADMAP.md#technical-patterns)。

路线图第 0～9 节和 [JSON](BABEL_ROADMAP.json) 的 `current_plan` 记录 `codex/babel-next` 的现行方案。第 10 节保留带日期的历史审计。

- 怎样做公平比较、解释负结果、判断证据够不够，见[实验分析原则](BABEL_EXPERIMENT_WORKFLOW.md#analysis-principles)。实验或路线的结论如何向你说明，见[回复原则](BABEL_EXPERIMENT_WORKFLOW.md#reply-principles)。发布、恢复和备份见[完整工作流程](BABEL_EXPERIMENT_WORKFLOW.md)。
- [实验目录原则](../../experiments/README.md)：模型/训练/评测/配置/说明同目录，发布后冻结，改变进入新版本。
- [新分钟数据审计](../../experiments/20261124_minute_audit_v1/README.md)：原8份证据与独立CPU审计，不等于训练资格完成。
- [本分支日志](BABEL_PROGRESS_LOG.jsonl)：从迁移事件开始追加；不把旧队列带入。
- [分支迁移清单](BRANCH_BASELINE.json)：旧Git身份、保留文件与原证据哈希。

只保留当前方案与必要经验。旧实验、完整旧日志、失败记录和暂未执行的历史假设可以从迁移清单的固定Git链接追溯，不在新分支复制历史源码或恢复入口。

新模型尚未实现/训练；当前没有AutoDL训练命令。
