# Babel 当前研究资料

第一次读，从路线图首页的[四个模型方案与两次比较](BABEL_ROADMAP.md#model-recipes)开始。P0、P1 是上一版，P2、P3 是当前要比较的两套方案。

- 想顺着一个样本读，可以依次看[输入哪些 K 线](BABEL_ROADMAP.md#input-views)、[模型总图](BABEL_ROADMAP.md#model-architecture)、[怎样训练](BABEL_ROADMAP.md#training-tasks)和[怎样验收](BABEL_ROADMAP.md#delivery-criteria)。
- 想了解当前进度，看[已有证据和下一步](BABEL_ROADMAP.md#current-review)。当前下载完成后，下一份交付是[数据与样本资格包](BABEL_ROADMAP.md#sample-qualification)。
- 想查具体用途，看[价格能重建到哪里](BABEL_ROADMAP.md#reconstruction-evidence)、[缺失行情的价格锚点](BABEL_ROADMAP.md#historical-price-anchors)或[行情分类与技术形态](BABEL_ROADMAP.md#technical-patterns)。

路线图第 0～9 节和 [JSON](BABEL_ROADMAP.json) 的 `current_plan` 记录 `codex/babel-next` 的现行方案。第 10 节保留带日期的历史审计。

- [工作流程](BABEL_EXPERIMENT_WORKFLOW.md)：实验决策、报告、备份与人类可读汇报。
- [实验目录原则](../../experiments/README.md)：模型/训练/评测/配置/说明同目录，发布后冻结，改变进入新版本。
- [新分钟数据审计](../../experiments/20261124_minute_audit_v1/README.md)：原8份证据与独立CPU审计，不等于训练资格完成。
- [本分支日志](BABEL_PROGRESS_LOG.jsonl)：从迁移事件开始追加；不把旧队列带入。
- [分支迁移清单](BRANCH_BASELINE.json)：旧Git身份、保留文件与原证据哈希。

只保留当前方案与必要经验。旧实验、完整旧日志、失败记录和暂未执行的历史假设可以从迁移清单的固定Git链接追溯，不在新分支复制历史源码或恢复入口。

新模型尚未实现/训练；当前没有AutoDL训练命令。
