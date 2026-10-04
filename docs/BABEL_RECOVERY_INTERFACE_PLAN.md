# V07第一步：固定权重的编码器／查询头交叉回读

2026-10-04。由[V03正式失败](BABEL_RECOVERY_R1_REVIEW.md)触发。对应N02/N03/N07/N09和V07／S6；不是换一个名字续训R1，也不解除V04/V05前置门槛。运行入口已实现，合成测试通过；真实CUDA尚未运行。

## 问题与边界

原Macro编码器E0及query D0，在R1各5轮后变成E5和D5。两者同时更新，当前只知道组合E5+D5的结构读取存在退化。先检验固定模块替换的误差变化，再决定有没有理由做等预算读头适配；不把头交换失败解释成永久信息丢失。

## 固定八个读取组合，零拟合

每个种子42/43分别评价：E0+D0、E5+D0、E0+D5、E5+D5。共8次组合评价，E/D全部eval及冻结，无optimizer、无新随机头、无学习率搜索；不按得分选新主模型。

来源保持原Macro97/94、R1实际第5轮（completion SHA `d6ef283d86910327686b1dd5a3dce8bc4e47df8da3f31af154ccc9389ea24456`）及原资格；严格校验架构、normalizer、buffer、权重和源统计。768维/4层/8头及原query结构不变。交换query时必须完整加载其绑定参数/buffer，不偷偷重拟合状态尺度或新增校准。

数据只用原978个val端点、相同255缓存、同四位置32/64/96/128；分别native短前缀与各位置完整rolling128，保持物理目标、价格坐标和mask。跨模块不新增原始历史或标签旁路。val已经用于诊断，不称独立泛化。

## 预先规定报告

1. E0+D0与E5+D5必须按原数值门槛复放本轮源／第5轮全部96个验证值；来源或复放不符即失败留证，先处理工程，不解释混合组合。
2. 原24项全部导出，不只挑结构项；另按固定prefix导出近/中/远、活动、结构。结构展开level/trend及4/16分块误差，验证与原primary聚合关系；新增分解为诊断，不更改训练目标/保护门槛。
3. 保存相同窗口的逐行误差和ID，分别报告source、encoder-only swap、query-only swap、joint及源比率。对每项固定定义：encoder替换变化=L(E5,D0)−L(E0,D0)，query替换变化=L(E0,D5)−L(E0,D0)，交互=L(E5,D5)−L(E5,D0)−L(E0,D5)+L(E0,D0)。加和只能解释这些冻结组合的读取误差，不是优化过程唯一机制。
4. 两种子、全部位置、均值/中位数/P95都披露，缺支持位置显式标记；不把native/interior平均通过说成所有位置通过。

## 决策规则

- 混合组合比joint好：说明该固定接口替换能减轻某项误差，不能自动断定信息增加、更新编码器无价值或直接部署混合模型。
- 两个混合组合都差：可能存在坐标/协同变化，不能据此判定两个模块均损坏或信息消失。
- 若只是接口歧义，再单独锁定V07后续等目标等预算的读出验证；本计划不预授权新头训练。
- 无论哪种结果，R1原失败记录保留；是否重设RS对照需另写清新证据、必要变化和门槛，不自动放行。

## 资源与六方面检查

只读8个组合，串行GPU，独立累计30分钟上限；计入原10小时余量，不增加训练任务/轮数/更新预算。源码入口须验证源不变、设备/dtype/因果、复放、失败导出和权重加载路径，检查实际磁盘需求后再交付命令；报告统一download，不复制优化器或全缓存。

计划检查：目的限定接口敏感性；架构与可见历史明确；0 loss更新；同目标/坐标且复放原结果；数据沿用已审缓存、统计不重拟合；实现/CUDA均待完成。最重要限制是：**模块交换只能定位固定解码的敏感性，不能独自回答状态信息是否丢失。**

| 复查维度 | 状态 | 证据与未完成项 |
|---|---|---|
| purpose | pass_for_scope | V07限定固定模块替换的读取敏感性；不增加训练或选新主模型。 不能独立判定状态信息是否丢失。 |
| architecture | pass_for_scope | 双种子E0/E5与完整D0/D5含buffer交叉；768/4层/8头与可见历史不变。 实际加载路径、接口与因果检查待实现。 |
| training | not_applicable | 全部冻结，零更新，无新头、校准或统计拟合。 若后续需要读头适配必须单独锁协议。 |
| validation | pass_for_scope | 先复放96源与末轮值，再披露全部组合/位置/分项/逐窗口；不改原门槛。 实现与CUDA尚未验证；val为反复使用研究诊断数据。 |
| data | pass_for_scope | 沿用978端点、255缓存、物理目标/价格坐标/mask与固定统计。 源码须验证来源不变；不扩张原scaler审查结论。 |
| engineering | open | 规定1800秒累计上限计入原预算，串行与失败导出。 运行入口、磁盘预检、加载/数值测试和CUDA均未完成，暂无运行命令。 |

## 工程交付与运行

模块`obson.babel.recovery_interface`，脚本`scripts/babel_recovery_interface_autodl.sh`。R1及其绑定历史代码、脚本、协议均未修改。本入口只增加诊断模块；原R1失败结论不变。

使用本次R1同一环境：RTX3080Ti、PyTorch2.8.0+cu128；保持micro8、原MHA/TF32/线程配置，不改容差。先校验R1全部绑定文件（含两个last.pt）和资格，再加载原Macro与第5轮；每种子串行读取四组合。没有权重复制和优化器恢复，需至少512MiB剩余磁盘用于报告及打包，检查失败不删源文件。

累计1800秒监督覆盖来源检查与评价；重启沿用同一目录budget，余额不重置。未完成的零更新诊断可以在余额内重新读取，不会增加训练更新；已完成报告重启只校验并导出。监督进程异常消失时保留已预留时间，避免隐性扩时。输出目录不能复用其他实验；不要通过换目录绕过预算。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash scripts/babel_recovery_interface_autodl.sh all > logs/babel_recovery_interface.log 2>&1 &
tail -f logs/babel_recovery_interface.log
```

成功、报错或超时均自动打包到`/root/autodl-tmp/download/babel_recovery_interface_reports.tar.gz`。打包包含状态、绑定信息、逐窗口数组、支持mask、复现检查、分位置指标、冻结模块签名、交叉差值与日志，不含任何.pt或输入缓存。单独重新导出（不会运行模型）：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_recovery_interface_autodl.sh export
```

`interface_complete_requires_review`只表示诊断完整执行，不代表模型改进或RS放行。`failed`／`timeout`只审查已完成证据，不给不完整八组排胜负。下一步是收到报告按V07复核，再决定是否需要等预算读取验证；不自动训练新头、追加轮数或启动V04/V05。

## 实现后六方面复查（取代上文计划阶段的待实现状态）

详见[BABEL_RECOVERY_INTERFACE_VALIDATION.json](BABEL_RECOVERY_INTERFACE_VALIDATION.json)。

| 维度 | 状态 | 检查与证据缺口 |
|---|---|---|
| purpose | pass_for_scope | V07八组固定模块替换仅用于读取敏感性定位；零新主模型选择，RS仍被R1失败阻断。 交叉回读不能单独判定信息消失，科学结论待CUDA报告。 |
| architecture | pass_for_scope | 复用原构造器与预测路径；完整state_dict含归一化buffer；逐组合签名、冻结及eval、因果检查。合成四组合验证。 真实768维/4层/8头源权重的CUDA加载与因果复现待运行。 |
| training | not_applicable | 没有优化器构造、拟合、校准、重新拟合统计或模型选择；测试禁止AdamW；仅读取R1已完成第5轮权重。 序列化checkpoint含旧Adam张量，加载后立即释放；没有恢复或更新优化器。 |
| validation | pass_for_scope | 原24均值复算；E0D0/E5D5沿用原容差；支持mask及逐位置均值/中位数/P95；结构按4/16分解并核对原聚合；复现失败先留证停止。 真实96值复现待AutoDL；val反复用于诊断，不是独立留出，混合组合不选赢家。 |
| data | pass_for_scope | 绑定R1完成SHA及全部文件、原资格、原身份链、978端点ID；沿用255缓存与原目标生成；运行后复核源不变。 不独立重做scaler拟合，不扩大原始数据审查范围。 |
| engineering | pass_for_scope | 新增14项合成测试，关联73项通过；含四组合执行、错误源/epoch/复现失败、无更新、超时累计和失败导出；ruff与shell语法检查通过。 无本地真实权重运行；CUDA和实际资源用量待报告。累计30分钟、512MiB可用磁盘预检；不复制权重/优化器/缓存。 |
