# Babel 当前研究资料

**直接阅读[当前路线图](BABEL_ROADMAP.md)。** 正文完整说明新方案，不需要先读旧实验：

| 想了解什么 | 入口 |
|---|---|
| 目标、当前状态、为什么重新设计 | [第1节](BABEL_ROADMAP.md#1-我们要做什么为什么要重新设计) |
| 新数据有哪些、怎样聚合与标准化、为什么这样处理 | [第2节](BABEL_ROADMAP.md#2-数据这次究竟多了什么怎样使用才成立) |
| 状态、注意力、历史解码和概率未来怎样配合 | [第3节](BABEL_ROADMAP.md#3-架构模型怎样把多段历史整理成一个状态) |
| 每根bar怎样训练、损失和梯度怎样计算 | [第4节](BABEL_ROADMAP.md#4-训练每一步究竟在教模型什么) |
| 怎样公平评价，什么时候结束这一阶段 | [第5节](BABEL_ROADMAP.md#5-评价怎样知道它真的有用怎样算这一阶段结束) |
| 同批A/B/C对照，各自回答什么 | [第6节](BABEL_ROADMAP.md#6-第一批实验一次回答两个核心决定) |
| 哪些沿用旧经验，哪些数学方向条件保留 | [第7节](BABEL_ROADMAP.md#7-借鉴与保留旧经验怎么进入新数学方法何时进入) |
| 下一步、交付、固定任务编号及六方面复查 | [第8节](BABEL_ROADMAP.md#8-从现在到阶段成果当前顺序与交付物)、[第9节](BABEL_ROADMAP.md#9-执行原则任务索引与本次复查) |

当前只做方案，没有新训练任务或可执行命令。详细研究/报告/备份/汇报规则在[BABEL_EXPERIMENT_WORKFLOW.md](BABEL_EXPERIMENT_WORKFLOW.md)。[JSON](BABEL_ROADMAP.json)是当前结构化镜像，[进度日志](BABEL_PROGRESS_LOG.jsonl)按次追加。

原始数据证据保存在[data_audit_20261124](data_audit_20261124/summary.json)。旧路线、旧决策和全部历史任务存于[归档MD](archive/BABEL_ROADMAP_BEFORE_REWRITE_20261125.md)与[归档JSON](archive/BABEL_ROADMAP_BEFORE_REWRITE_20261125.json)，只作追溯，不能恢复其中的旧队列。原协议/报告仍在上级`docs/`，证据没有删改。

```text
docs/babel/
├── README.md
├── BABEL_ROADMAP.md                 ← 从头可读的当前方案
├── BABEL_ROADMAP.json               ← 当前方案与任务状态镜像
├── BABEL_EXPERIMENT_WORKFLOW.md     ← 执行、报告、备份、回复规则
├── BABEL_PROGRESS_LOG.jsonl         ← 只追加的工作记录
├── data_audit_20261124/             ← 原始数据审计证据
└── archive/                        ← 有版本身份的历史，只读追溯
```

原`docs/BABEL_ROADMAP.*`、工作流程、日志及`artifacts/babel_tqdata_20261124`保留兼容链接，不另存一套当前内容。历史快照中的相对Markdown链接已按归档目录调整；其来源提交、原文件SHA和未执行标记均写在归档入口。
