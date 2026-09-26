# 跨周期实验：只补评价与父模型数值回放

目标仍是检验跨周期共同历史约束能否改善可复用bar状态。本次只修复评价执行和数值诊断，不改变研究目标、模型、训练预算、选模、用途读出或升级判据。

## 已知事实与未确认项

本轮日志已经到达8份检查点研究状态提取以及读出校准之后，失败于冻结父模型重叠稳定性的`path_signed_offset_a_bps/p50`回放。原growth manifest的评价batch为128，本轮是64，这是明确的执行口径差异。换GPU也可能影响浮点结果，但不能在缺少实际数值时直接认定偏差仅是浮点误差，更不能将失败当作通过。

旧比较只给出字段名，没有实际值。新的独立入口保留原55个绑定模块及原实验目录，完整核验四组100轮、权重、来源、原始数据、划分、8份选模和520个既有Ridge候选。只做冻结模型推理及评分，禁止重新训练或拟合读出。缺任何已完成锁定资料则停止，不自动补训练。

## 回放规则

1. 研究评分继续使用本轮固定batch64，所有模型、各指标、原判据完全相同。
2. 父模型重叠摘要仍先按原容差`atol=1e-6, rtol=2e-5`检查。记录所有不匹配字段的实际值、参考值、绝对差与允许误差；结构、计数、支持掩码不放宽。
3. 若失败且父报告batch不同，额外以父报告的batch128回放，全部摘要必须通过上述原容差。
4. 同时对所有配对窗口及0/1/16/64四种视图，比较batch64与128的状态、归一化解码输出，沿用`overlap_diagnostic_run.evaluate_cell`已有的同窗口FP32数值对照容差`5e-5/2e-4`。这是张量级数值对照，不是放宽研究指标或父摘要容差。
5. 只有父报告原batch回放及全部张量对照通过，才接受该执行差异。仍保留batch64研究评分，不替换成更有利的父报告数值；否则停止并导出诊断。未验证通过之前不声称已解决实际GPU上的回放失败。

重建、用途、raw/PCA/current基线仍使用原严格回放。成功只能表示评价完成，最终候选结论仍由原判据决定。研究集已复用，不宣称独立最终留出。

评价循环从原`cross_period_evaluate.evaluate`单独版本化，目标函数、数据准备、评分和判定复用原模块；独立模块是为了不改写训练来源指纹。测试覆盖原/新评价循环的相同合成结果及原输入不变；本地不执行真实模型推理、训练或真实读出拟合。

## 命令

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash scripts/babel_cross_period_recheck_autodl.sh all \
  > logs/babel_cross_period768_v2_recheck.log 2>&1 &
tail -f logs/babel_cross_period768_v2_recheck.log
```

默认读取`checkpoints/babel_cross_period768_v2`，新结果写入`checkpoints/babel_cross_period768_v2_recheck`。不要重新运行训练脚本，不删除或改名原训练目录。同一补评价命令可重跑；已成功完成的输出核验后直接复用，失败时仅重做评价。

无论成功或失败，自动导出：

`/root/autodl-tmp/download/babel_cross_period768_v2_recheck_reports.tar.gz`

报告主目录为本次补评价结果，`training_source/`保留上次训练报告（其中的失败状态是历史记录）；大权重及数组不打包。诊断文件为`test_parent_s42_numeric_replay.json`等，逐个父模型保存实际差异；若校验仍失败，发整个包即可，不必另外找日志。

手动打包：

```bash
bash scripts/babel_cross_period_recheck_autodl.sh export
```

源目录非默认时使用`BABEL_CROSS_PERIOD_RECHECK_SOURCE`；输出、日志分别用`BABEL_CROSS_PERIOD_RECHECK_RUN`与`BABEL_CROSS_PERIOD_RECHECK_LOG`指定。代码或输入指纹变化必须新评价目录，原来源校验不绕过。
