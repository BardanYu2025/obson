# V02＋V14：零更新梯度与新入口资格

2026-10-04。对应总计划V02（原E2）及V14；目的为后续V03/R1和V04/V05/RS补足准入证据。**本轮不训练、不改变架构或损失、不晋升任何模型。** 代码和合成验证交付不等于真实CUDA资格通过。

## 固定来源和问题

- 保留Macro control s42/s43 best97/94，768维、4层、8头，原输入适配器、原历史查询头和原局部头。
- 入口使用已复核Context Transfer源和R0v2报告目录；源manifest/completion、历史代码及缓存按原SHA核对，R0v2 completion固定为`aeb30e1d35240ce0d551df7362046f42ada8d89ab70fff12738d9bd158086a0d`。
- 两种子均只读，无optimizer、scheduler或参数更新。Context原MHA fastpath关闭、TF32关闭、线程4；保持原数值阈值。
- 正式运行仅AutoDL CUDA。本地合成夹具不加载真实权重。此次只读取train/val做数值/资格检查；既有研究缓存仅核验绑定，不重新做研究集模型评分。

V02回答：实际总反向是否等于分项反向之和？同一图重复反向是否稳定？精确保留跨年龄边界变化后，近/中/远分解是否一致？此前约0.1084%的平方范数汇总差异不能直接回答这些问题。

V14回答：当前新入口是否仍加载相同的模型和训练参考；255条缓存与原128输入、两种目标生成及局部目标是否一致；冻结源局部头是否具备最低读取资格？这不是重新验证全部祖先训练。

## V02的直接检验

每种子固定训练清单前2个窗口，prefix32/64/96/128；分别native前缀和rolling完整历史，共4个案例。每个案例只构建一次前向与目标图，核对全部可训练encoder/query参数、逐坐标检查：

1. 同一总loss连续两次反向。
2. 实际Q＋S总反向与0.4path、0.2change1、0.1body、0.3activity、1.0structure五项梯度之和。
3. 直接价格loss反向与三价格族梯度之和。
4. 直接价格loss反向与近/中/远梯度之和；保留16→17、64→65边界变化，不能简单遮掉段外mask来分解。

CUDA反向仍是原FP32，**只把CPU诊断求和、平方范数和残差累积改为float64**。门槛保持`atol=1e-6, rtol=2e-5`，不是扩大容差。另保存各族原float32范数与float64平方范数，辅助定位数值汇总差异。

完整梯度逐元素核对；报告保存每个参数的名字、形状、dtype、实际/预期平方范数、残差平方范数、超差个数和最大8个残差坐标的双方实际值。**不把数GB全梯度打包**，因此tar能复算导出的坐标和汇总，不足以独立重建全部梯度；完整复核仍需绑定源码和原权重。这是导出体积细化，不改变检验范围或阈值。

报告裁剪因子仅是该2窗小批的诊断，没有真正裁剪、优化或证明全训练分布中的任务主导关系。无论通过与否，都不直接调损失权重。

## V14的入口资格

1. 复用已复核R0的原始输入/分区/因果审计：核验R0 completion全部绑定报告、历史代码及当前来源身份相同。当前不重建7190个原始窗口，也不独立重拟合源scaler。新入口消费的是这些绑定缓存；原始CSV的新版本不被静默纳入。
2. train/val共4789/978个缓存窗口，通过原始index及合约/终点清单映射原数据，逐张量要求原128输入等于255缓存末128。
3. 在固定四位置比较旧query目标与255特征重建的native目标/mask，再把后者转换为原过去16根局部坐标，与独立`local_targets`比较。缺失通道按相同mask归零，不能拿缺失值当观察量。沿R0目标门槛`atol=1e-5, rtol=2e-5`，报告失败批次而不放宽。
4. 核验原prefix前5轮样本顺序、五位置和更新记录；恢复两种子原初始化签名，复放原24项full/native验证值。源码/起点/缓存不符立即停止。
5. 局部头原权重冻结，不新建/预热头。基线仅从train按固定prefix×lag×channel的可用值求均值。两个种子的native验证端点128和中途32/64/96，局部primary使用原SmoothL1、价格/活动定义，**各类均须严格小于训练均值基线**；验证出现训练均值无覆盖的有效通道也不通过。保存每窗口×四位置的头/均值误差及ID，以便独立复算。此为最低资格，不等于充分拟合或通用用途成功。
6. 在真实源上执行四个新损失开关，与原255缓存目标路径逐窗口核对query/structure和各组优化值，并核对各组local诊断值相同，门槛沿用1e-6/2e-5；再检查local.detach开关及实际局部损失到encoder的有限非零梯度（损失恰为0时允许零梯度）、到query的零梯度；局部头仍不更新。执行原因果检查，所有模型参数/缓冲区前后签名相同、`.grad`无累积。
7. 结束重新核验全部来源身份。没有选模、拟合或优化器恢复；源缺选中时刻Adam的既有事实不改变。

## 状态与停止

- `qualification_complete_requires_review`/退出0：全部数值与读头资格通过，等待报告复核，**不自动启动R1或RS**。
- `blocked`/退出3：数值或读头资格未过；写清`r1_preflight_passed`和`rs_reader_qualified`。局部头读取、局部目标契约或局部梯度路线单独失败只阻断RS资格；共同query目标、原验证复放或梯度核对失败则阻断R1预检。均不能据此判定整个表示无效。
- `failed`/非零：来源、设备、执行或因果检查异常，保留已写证据；不能当作模型科学结论。
- `timeout`/退出124：独立监督1800秒上限，杀停子进程、记录未完成并导出；上限不是保证耗时。
- 输出必须新建、与来源和旧审计分离；拒绝覆盖。正式运行至少512MiB可用空间，只保存小报告和误差数组，不复制模型、全梯度或缓存。CPU暂存梯度仍有内存开销，不是只用数MB内存。

## AutoDL命令

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_recovery_qualification_autodl.sh audit > logs/babel_recovery_qualification.log 2>&1 &
tail -f logs/babel_recovery_qualification.log
```

默认源`checkpoints/babel_context_transfer768`，已复核R0目录`checkpoints/babel_recovery_r0_v2`，新输出`checkpoints/babel_recovery_qualification`。需要不同现存路径时用`BABEL_QUAL_SOURCE`/`BABEL_QUAL_AUDIT`显式指定，不更换来源内容；再次执行用新`BABEL_QUAL_RUN`和相同对应`BABEL_QUAL_LOG`，保留旧失败。

成功、阻断、失败、超时均自动导出：

```
/root/autodl-tmp/download/babel_recovery_qualification_reports.tar.gz
```

需要单独重新打包时：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_recovery_qualification_autodl.sh export
```

提交本轮包后，按V02/V14分别复核和更新总计划；V03的完整训练入口及后续RS仍是下一阶段工作。
