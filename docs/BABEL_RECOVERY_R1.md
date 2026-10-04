# V03 / R1：固定五轮学习率稳定性筛查

2026-10-04。对应N01/N03/N04/N07/N08/N09、V03及旧T1/R1。前置V02/V14已经真实复核通过。本轮只降低适配学习率，检验五轮内是否保留源能力；不训练RS，不扩架构，不更换主模型。

## 来源、信息与唯一变量

- Context源固定`checkpoints/babel_context_transfer768`及其原Macro control s42/s43 best97/94（768维、4层、8头），原训练输入适配器和历史查询头保持。
- 已审资格目录`checkpoints/babel_recovery_qualification`，completion SHA固定`2e0fe27289d2000db90c0ac4e80d67cc175545f07370c85975373a859cc47c86`。校验其全部报告、109个绑定模块及源身份；不再全量运行R0/梯度资格。
- 原train4789/val978个窗口，255缓存只用于取得真实上下文。训练只输入末128，抽五个位置，其早期状态仍只有此前缀历史；不冒称每个位置均看128。单合约/周期、train-only统计和分区沿用已审来源，不重新拟合或读取新原始文件。
- 编码器、原query更新；原其它读头不参与训练，原局部损失仍不回传。价量目标保持Q+S：Q=.4path+.2change1+.1body+.3activity，额外远端structure权重1。mask、段平均、delta跨边界定义、目标坐标均由绑定旧代码执行。
- **直接调用原`context_transfer_run.train_epoch`**。prefix单视图、batch128/micro8、确定性五位置采样、实际尾batch权重、合并梯度裁剪1.0、eval模式训练与原来一致。
- 新AdamW，encoder decay=.01、query decay=.0001，betas=.9/.999、eps=1e-8；源没有选中时刻Adam，称适配而非精确延续。只将固定LR改为原选中轮记录：

| 种子 | encoder LR | query LR |
|---|---:|---:|
| 42 | 3.066380815642346e-6 | 3.0663808156423456e-5 |
| 43 | 3.2648704606900738e-6 | 3.2648704606900735e-5 |

高LR参照复用原prefix前5轮，检查原采样/前向目标源码/初始化/步数后导出原逐轮24值。没有新高LR任务。来源绑定失败就停止，不自动补组。运行要求与资格相同的3080Ti及PyTorch2.8.0+cu128，MHA fastpath/TF32关闭、线程4；换环境需重新说明比较范围，不能静默续用单因素结论。

## 预算、评价和停止

双种子串行，各5轮、190更新，共10轮/380更新。每轮原24项：full/native×endpoint/interior×price/near/mid/far/activity/structure；不读取研究集评价，不新增拟合头或选参。

实际第5轮所有24值均不超过源的110%，两个种子均满足且模型实际发生更新，才通过R1。不给best或epoch0回退过关机会，保留全部逐轮值、原高LR参照和低/高LR逐项比率。通过只说明五轮稳定，不说明长期稳定、结构问题解决或最终用途升级；失败不扫第三LR、不补轮。

额外保存源和实际第5轮978窗口×4位置的各项误差数组及ID，复算其聚合与原验证一致；这不是原始价格预测或研究集效果。第5轮权重执行原因果检查。s42旧局部头末端较弱仍记录于V14；本轮不通过改变它来掩盖问题。

R1独立监督累计最多3600秒，作为既有10小时总恢复训练会话预算中的一部分；后续RS须扣除R1实际累计用时，不能再另给完整10小时。本轮未增加任务/轮数/更新上限。一小时是保守停止上限，不是预计耗时。至少2GiB可用磁盘，不复制源/cache，保留两个含优化器的last.pt并为原子替换留空间；不删除源文件。

## 恢复与交付

- 每个完整epoch原子保存模型、Adam、RNG、全历史与签名；导出history和receipt可由该检查点恢复。仅在完整epoch边界恢复，逐项验证优化器配方、状态步数、采样hash、初始化和全部24字段。
- 每轮开始前写pending记录。若进程在整轮检查点提交前中断，**不自动重放不确定轮次**，失败导出后复核，不能假装没有发生过额外更新。若提交后、派生JSON写完前中断，恢复已提交轮，避免重复更新。
- 独立监督在启动前预记剩余时间额度；正常退出按实际用时结算。监督进程本身被杀时不自动清零预算。预算耗尽、不明pending或数值错误停止并导出；不删除状态后偷偷续跑。
- 返回0：两种子实际第5轮都通过，`r1_complete_requires_review`；返回3：完成训练但至少一组保护失败。其他非零为执行失败，124为超时。所有情况都导出。`rs_authorized`始终false，需报告复核，不自动启动下一矩阵。
- 当前只交付R1，RS完整训练入口未实现。正式CUDA训练待AutoDL；本地只运行合成测试（小合成模型的优化器/恢复测试不属于真实行情训练）。

## AutoDL命令

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_recovery_r1_autodl.sh all > logs/babel_recovery_r1.log 2>&1 &
tail -f logs/babel_recovery_r1.log
```

默认输出`checkpoints/babel_recovery_r1`；同命令可继续已提交的完整epoch。继续时保留旧日志，使用追加：

```bash
nohup bash scripts/babel_recovery_r1_autodl.sh all >> logs/babel_recovery_r1.log 2>&1 &
tail -f logs/babel_recovery_r1.log
```

成功失败都自动打包到`/root/autodl-tmp/download/babel_recovery_r1_reports.tar.gz`（不含权重）。单独导出：

```bash
bash scripts/babel_recovery_r1_autodl.sh export
```

需要定位不同路径时使用`BABEL_R1_SOURCE`、`BABEL_R1_QUALIFICATION`、`BABEL_R1_RUN`和`BABEL_R1_LOG`；内容绑定不因改路径放宽。不可用新目录绕过已消耗预算或重跑失败的实际训练。
