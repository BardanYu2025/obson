# Babel R0：零更新核查入口

2026-10-04。落实[BABEL_RECOVERY_PLAN.md](BABEL_RECOVERY_PLAN.md)的第一阶段。这里只核查，不执行R1或R2，不实例化优化器，不新建训练缓存，不改动旧实现及其代码哈希。保持原Macro作研究参照，所有新训练仍暂停。

## 本次检查

1. Context完成记录、绑定源码、报告及四个输入缓存；递归验证原Macro来源，列出Macro与Shared直接权重边，区分Uniform输入缓存引用与权重继承。缺文件不等于需要重训。
2. 从原始合约重新生成全部候选端点的255根因果特征，确认整个输入跨度在分区内，复核排除项及缓存中的原始行身份。每个合约做确定位置的未来截断检查；不是穷举每个时间点。
3. 逐端点比较原128输入，独立的旧目标生成管线与新目标管线；检查A/B共同标签和mask完全一致，输出不同历史支持的年龄段权重。
4. 用两个Macro best、两个Shared control best、两个rolling update_last，共六份真实权重，按各自原验证接口复放；Macro另复放Context的24项初始指标。检查未来后缀扰动不改变当前状态和读出、单年龄查询不依赖其他查询。
5. 固定train小批次分别计算路径、单步变化、实体、活动、远处结构的编码器/查询头梯度、任务梯度夹角及假设整体裁剪系数，另按原损失精确拆分近/中/远价格梯度（包括跨段的单步变化，不重复计入总梯度）；检验实际局部头的detach路径。执行前后哈希相同才算无更新。
6. 重建旧prefix前5轮采样哈希和更新账目，确认高LR参考的初始化与当前Macro一致；记录选中snapshot是否包含optimizer，不将新Adam称为精确续训。

数值口径预先固定：输入/指标`atol=1e-6, rtol=2e-5`；独立float32目标管线`atol=1e-5, rtol=2e-5`；状态/读出因果复放沿Context原预检`atol=1e-4, rtol=2e-4`。失败先落盘差异，不在运行时放宽阈值。

## 完成意味着什么

成功状态为`audit_complete_requires_review`，**不是R1准入通过**，更不是旧实验全部有效。失败记录最早失败阶段和具体文件/指标；不删除或修复源文件，不自动重试训练。原统计量的train-only来源、绑定代码与数值一致性在本次核查，原始scaler拟合的独立完整再现未包括；审计报告显式保留此限制。梯度仅代表固定train小批次，不能单独证明训练全程近端误差的原因。结果回传后还需结合这些限制判定R1是否可开。

默认输出独立目录`checkpoints/babel_recovery_r0`，目录非空会拒绝运行，保护失败证据。R0默认最多3600秒，独立父进程超时终止并打包；超时后不得声称核查通过。预计无需新的GiB级空间，但原权重、原数据与缓存仍须存在。CPU原始特征阶段GPU可能空闲。

## AutoDL一次启动及自动导出

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash scripts/babel_recovery_r0_autodl.sh audit > logs/babel_recovery_r0.log 2>&1 &
tail -f logs/babel_recovery_r0.log
```

无论成功或失败，报告自动汇总至：

```text
/root/autodl-tmp/download/babel_recovery_r0_reports.tar.gz
```

仅重新打包已有结果：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_recovery_r0_autodl.sh export
```

源目录非默认时指定`BABEL_RECOVERY_SOURCE`；重新核查需新的`BABEL_RECOVERY_RUN`，相应设置`BABEL_RECOVERY_LOG`并将输出重定向到同一路径。不要用旧训练命令执行R1/R2，它们尚未实现。

## 本地交付验证

新增R0及原Context/Uniform相关测试共44项通过，涵盖合成六检查点复放、真实detach分支、精确年龄损失/梯度分解、非因果反例、原始输入和独立标签复建、采样篡改检测、超时和失败导出。源码静态检查及shell语法检查通过。另直接对照用户已下载归档确认旧106份源码哈希未变、源97/94轮学习率与归档记录一致。没有在本地训练或复放真实神经模型；CUDA显存、耗时和真实数值复放待AutoDL执行。
