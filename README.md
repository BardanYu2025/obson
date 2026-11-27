# Babel：新分钟数据上的行情状态研究

固定工作目录：`/Users/bardan/Documents/Codex/来自旧电脑-20261031/obson`，与PyCharm一致。

当前分支：`codex/babel-next`。当前优先把过去行情编码成可复用状态，验证历史保留和当前行情描述。未来预测作为后续用途，可与当时可得的舆情/市场信息融合；纯K线预测不是当前编码器成功的必要条件。

从[当前路线图](docs/babel/BABEL_ROADMAP.md)第 0 节开始。完整月份合约已做数据资格核对，可查每个品种有多少行情能还原原价、多少收盘时刻可验证分类与顺序。新模型处于设计阶段。[本批资格报告](experiments/20261127_history_qualification_v1/reports/completed_months_20261127/qualification.md) · [44 个品种覆盖](experiments/20261127_history_qualification_v1/reports/completed_months_20261127/by_product.md)

```text
docs/babel/           当前方案、流程、任务状态与本分支日志
experiments/          每个实验独立、自包含、发布后冻结
  _template/         新实验目录模板（不能运行）
  20261124_minute_audit_v1/  唯一迁入的CPU分钟数据审计及原证据
  20261127_history_qualification_v1/  已下载月份合约的原价、分类与顺序资格
tools/               目录/版本治理工具；不承载模型、训练或科学指标
.github/workflows/   实验隔离检查
```

[实验目录原则](experiments/README.md)与[研究工作流程](docs/babel/BABEL_EXPERIMENT_WORKFLOW.md)适用于后续全部实验。

旧模型/脚本、旧数据包及庞大报告不在本分支的Git跟踪文件中。原目录的本地数据、缓存及历史产物保留并忽略，不作为新实验依赖。旧研究保留于`features/babel`及[固定来源清单](docs/babel/BRANCH_BASELINE.json)。保留Git历史便于追溯，因此这次精简不是Git对象库瘦身，也没有删除旧电脑目录或外部原始数据。

本地原始分钟数据仍为`/Users/bardan/private/tqdata_2015`，不复制进每个实验。以后由固定数据清单和哈希绑定输入。

仅检查目录规范（不训练、不读取行情）：

```bash
python3 tools/experiment_guard.py --base HEAD
python3 -m unittest discover -s tests -v
```
