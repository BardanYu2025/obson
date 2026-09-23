# 冻结末端读出诊断：同一个h128，两条现有路径

本轮不训练、不拟合统计、不改变已有选模，不扩大模型。上一轮联合监督在中间位置改善明显，但h128局部读取较差，所有新路线均未通过完整晋升门槛。本轮只判断：近期信息是否能通过已有的全局解码路径更好地读取。

## 固定协议

- 只读完整`checkpoints/babel_bar_alignment`及其上游来源。保留八组、两个种子、best/last共16个检查点，不凭新研究分数挑权重。
- 每个检查点的同一个h128分别输入原局部头和原固定PCA全局解码器。目标为第112–127根，共16根，排除当前第128根。全局预测的第111根作为其自己的价格锚点（零索引110），再转换到原局部归一化尺度；绝不使用真实收盘锚点修正预测。真值使用同一段真实历史重新锚定，掩码仅用于评分。
- 原尺度、原研究端点、原七通道及局部三族等权指标不变。报告primary/path/body/activity、一步误差/相关性/幅度比、closeMAE、通道R²、逐窗口误差、分组与配对周95%区间。配对为global_crop减local_head，负数表示裁切路径较好。重新核对旧全局指标和旧p128局部指标的逐窗口误差。
- 导出全局锚点误差，仅作诊断，不参与预测校正；固定等距四样例/研究集，全部检查点均保留JSON与HTML。不是预测未来，也不凭样例挑优。
- 每个已训练检查点先复现原验证，再在最多8个固定等距验证窗口测试严格前缀32/48/64/80/96/112/127/128、修改后续输入、追加16根伪未来、独立调用清空上下文与批次独立性。状态和局部头都比较，h128追加未来还比较全局头。不是仅检查初始化模型。
- FP32逐元素容差预设atol5e-5/rtol2e-4；失败后使用同权重CPU FP64重放，容差1e-9/1e-8。FP64通过则明确记录`passed_with_fp32_drift`，不会暗中放宽FP32门槛；两者失败停止并打包全部诊断。有限固定样本的测试不冒称所有数据上的形式化证明。
- 另报告128上下文末端和重置64根后末端的状态差异。两者历史与位置原点不同，不要求相等，不宣称模型具有流式缓存等价性。

## 收敛边界

源代码、源模型、源选模、历史与数组hash只读核验；不调用旧模块会改写来源锁的函数。每个检查点报告完成后写校验清单，同命令重跑跳过已核验完成的检查点；中断中的检查点从头重做推理。没有优化器，也没有PCA/标准化/读出头拟合。

输出保留原失败晋升决策，`automatic_promotion=false`。某条现有读出更好，能证明至少部分信息可读；两条都不好也不能证明向量完全丢失信息。本轮不证明监督失衡是唯一根因，不修复模型、不启动下一训练矩阵。先完成这次定位，再决定主线接口如何使用。

## AutoDL运行与导出

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_endpoint_readout_autodl.sh all \
  > logs/babel_endpoint_readout.log 2>&1 &
tail -f logs/babel_endpoint_readout.log
```

默认来源`checkpoints/babel_bar_alignment`，新输出`checkpoints/babel_endpoint_readout`。可通过`BABEL_ENDPOINT_SOURCE`和`BABEL_ENDPOINT_RUN`指定对应目录。串行推理16个检查点，不需要启动并行训练。预计数分钟，实际取决于来源文件hash核验和是否触发FP64复查，日志逐检查点/数据集报告进度。

成功/失败自动打包到`/root/autodl-tmp/download/babel_endpoint_readout_reports.tar.gz`。提交这一个包即可，包含完整指标、逐窗口误差、因果诊断、验证复现、固定样例和日志；排除.pt/.npy/.npz。手工重新打包：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_endpoint_readout_autodl.sh export
```
