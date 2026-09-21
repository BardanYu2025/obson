# 量仓实验：同预算续训与双选模核验

## 本阶段问题与纠偏

总目标见BABEL_RESEARCH_GOAL.md：因果状态压缩观察历史。此前30轮实验在同预算下的观察仍然有效，但不足以排除充分训练后出现更好权衡。先检验训练预算，不增加结构，也不把所有旧实验全部重跑。

本轮三组control/aux005/aux020、两个种子全部从各自第30轮last.pt续训到总计100轮，共新增6×70轮。不是重新从预训练起点跑100轮，不从各组不同轮次best.pt恢复。相同512维GRU、28输入、损失权重、64流、TBPTT128、encoder LR3e-5、头LR1e-4、优化器/随机状态/完整history都保留；不新增学习率调度以免混淆训练预算的效果。

100轮是固定预算，不是收敛证明。最终同时报告验证曲线和最近两个10轮区间的中位数，辅助读出网格的变化也保留。若仍有稳定改善再判断是否继续，而不因达到某个整数轮数宣布结构无效。

## 独立目录与完整恢复

默认父目录checkpoints/babel_activity_alignment512，新目录checkpoints/babel_activity_alignment100。父目录必须有完整6组last.pt/best.pt、缓存及报告，只有下载的reports包不足以续训。程序校验每组达到父manifest声明轮数、history连续、last内部最佳权重与best.pt一致、文件hash及全部来源不变。

复制last完整状态，仅更新新实验的来源元数据；保留权重、AdamW累积量、步数、学习率、torch/CUDA/Python/NumPy RNG。评分时回放分区序列，核对第30轮验证结果能复现。导出resume_validation.json。导入后、正式第31轮前先保存恢复状态，初始化审计中断可安全重试。

本轮使用独立缓存索引，复制小目标数组，大输入数组仍链接原activity512特征缓存；只读使用父目录。不能删掉上游activity512或alignment512目录。中断重跑相同命令从新目录最后完整epoch恢复。jobs可调整，其他设置改变必须用另一目录。

## 两套预先声明的选模规则

### A：价格最佳（主结果）

保持上一轮规则不变：验证价格detail最小，原close/base/change16保留门槛不变。继承旧best，即比较覆盖原0～30轮和新增31～100轮。生成根目录alignment_metrics.json。它直接回答多训练70轮后，原评价协议下是否更好。

### B：价格保留条件下量仓历史最佳（次结果）

不能拿辅助头loss公平选模，因为control的辅助头没有训练。三组统一采用冻结embedding的线性Ridge选择器，只接收训练/验证状态与目标，不接触测试集。

- 目标为预先定义的15个past-only量仓描述量；只用这些目标均有效的共同窗口。
- 目标归一化沿用训练集统计；embedding归一化仅拟合该候选的训练集共同窗口。
- 多输出Ridge的alpha候选1/10/100/1000，按验证标准化MSE选择，同一个alpha服务15目标。
- 价格门槛固定相对同种子父实验control的价格最佳验证结果：close<=1.02、base<=1.05、change16MSE<=1.05、detail<=1.02。
- 开始检查可获得的父last（30轮）和父price best，然后统一检查40/50/60/70/80/90/100轮。若以后续训已有本模块产物，也保留可获得的父history best候选。诊断保存/恢复RNG，不改变主训练随机流。
- 未保存的旧轮次无法补选，报告明确限制；不伪称该选择器覆盖所有旧轮次。
- 没有合格候选时明示selection_qualified=false，价格候选仅作为诊断fallback，强制阻止自动证据筛选通过。

次结果在activity_selection/alignment_metrics.json，使用同一价格、状态、线性/MLP读出、当前输入对照、按周配对与原验收规则。主次结果分别报告，不以测试分数在两种规则之间挑赢家。新的选择器也是方法变化，因此“新增选模的收益”和“单纯延长训练的收益”分开解释。

## 报告

budget_comparison.json和budget_summary.md汇总父30轮价格最佳、本轮100轮价格最佳、本轮量仓历史最佳。保留六组全部1～100轮history、selector网格、恢复审计、目标统计、两套完整评价和成功/失败状态。

这些仍是共享预训练、两个微调种子、反复使用的研究测试集证据。辅助监督目标的读出改善不等于新任务迁移或市场机制理解。不会自动替换主模型。

## 固定末轮诊断（100轮结果复核后新增）

历史选择器的价格门槛会排除信息读出仍改善的晚期候选。为完整展示这种取舍，新增diagnose-last：读取manifest中固定总轮数，对全部六组last.pt统一评价，包括未通过价格门槛的状态。不会运行编码器训练循环；冻结编码器后仍按原协议拟合线性/MLP读出头（MLP在AutoDL）。不增加测试选模，不改原门槛。

last_diagnostic目录保存完整重建、状态、量仓读出、当前输入对照、周配对结果及diagnostic_protocol.json。明确diagnostic_only=true、automatic_promotion=false，阶段筛选仅名为diagnostic_stage_screen；同种子验证价格资格也纳入筛选。共享评价器使用该目录私有best.pt副本，但其内容固定来自末轮last，绝非验证选出的best。根目录和activity_selection产物通过hash核对保持不变，来源与缓存继续校验。

新增代码的all/evaluate结束后也包含此诊断，避免只看到通过选择器的候选。已有完整100轮结果只执行下列命令，无需重跑all：

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_EXTEND_RUN=checkpoints/babel_activity_alignment100
export BABEL_EXTEND_LOG=logs/babel_activity_alignment100_last.log
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_activity_extend_autodl.sh diagnose-last > "$BABEL_EXTEND_LOG" 2>&1 &
tail -f "$BABEL_EXTEND_LOG"
```

结束自动更新同名download/babel_activity_alignment100_reports.tar.gz，包含原报告及新增last_diagnostic。源权重/优化器不变，不打包.pt/.npy。此前两套选择器的结论不会被这份固定末轮诊断覆盖。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_ALIGNMENT_PARENT=checkpoints/babel_activity_alignment512
export BABEL_EXTEND_RUN=checkpoints/babel_activity_alignment100
export BABEL_EXTEND_LOG=logs/babel_activity_alignment100.log
export BABEL_EXTEND_EPOCHS=100
export BABEL_EXTEND_SELECTION_EVERY=10
export BABEL_EXTEND_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_activity_extend_autodl.sh all > "$BABEL_EXTEND_LOG" 2>&1 &
tail -f "$BABEL_EXTEND_LOG"
```

结束自动导出/root/autodl-tmp/download/babel_activity_alignment100_reports.tar.gz。手动：bash scripts/babel_activity_extend_autodl.sh export。不要把BABEL_EXTEND_EPOCHS理解成新增轮数，它是总轮数。
