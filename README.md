# Babel：新分钟数据上的行情状态研究

当前分支：`codex/babel-next`。目标是把过去行情编码成可复用状态，支持历史回读、当前行情描述与待验证的概率未来。

从[当前路线图](docs/babel/BABEL_ROADMAP.md)开始。现在只做方案，尚无新模型或可开跑训练命令。

```text
docs/babel/           当前方案、流程、任务状态与本分支日志
experiments/          每个实验独立、自包含、发布后冻结
  _template/         新实验目录模板（不能运行）
  20261124_minute_audit_v1/  唯一迁入的CPU分钟数据审计及原证据
tools/               目录/版本治理工具；不承载模型、训练或科学指标
.github/workflows/   实验隔离检查
```

[实验目录原则](experiments/README.md)与[研究工作流程](docs/babel/BABEL_EXPERIMENT_WORKFLOW.md)适用于后续全部实验。

旧模型/脚本、旧数据包及庞大报告不在本分支工作目录。旧研究保留于`features/babel`及[固定来源清单](docs/babel/BRANCH_BASELINE.json)。保留Git历史便于追溯，因此这次精简不是Git对象库瘦身，也没有删除旧电脑目录或外部原始数据。

本地原始分钟数据仍为`/Users/bardan/private/tqdata_2015`，不复制进每个实验。以后由固定数据清单和哈希绑定输入。

仅检查目录规范（不训练、不读取行情）：

```bash
python3 tools/experiment_guard.py --base HEAD
python3 -m unittest discover -s tests -v
```
