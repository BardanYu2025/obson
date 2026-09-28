# Causal Pool768：同一bar状态的编码端历史汇聚

2026-09-28：本轮已完成并[正式复核](BABEL_CAUSAL_POOL768_REVIEW.md)，三项架构比较均未达预定共同增益要求，按计划收束；以下命令保留用于复现，不是要求重新训练。

本轮检验不同历史距离的直接汇聚是否改善近远信息的共同保留。[事前设计和结论边界](BABEL_CAUSAL_POOL768_DESIGN.md)保持。plain/full_pool/scale_pool × seed42/43，共六组各100轮；原4层/8头/768维主干和解码器不变，三组同用uniform查询＋1.0远程结构目标。相同种子的来源都是上轮History Structure control best，输出残差零初始化，第0轮函数相同。两个pool同参数同初始化，仅范围mask不同。

full_pool每路看完整可见历史，scale_pool四路分别直接看最近16/32/64/128个隐藏状态；隐藏状态已经包含更早历史，不能称为纯尺度解耦。四路总768维；解码器仍只见一个768维向量。新增模块2,364,416参数，plain没有这部分算力，不能称三组等参数。所有新模块参与联合训练，不复用两个冻结专家拼接。

验证只用64/128同末端过去16根、原生查询及最远65..127粗结构，第0轮保护和minimax选模；80及原留出48/112不参与选模。新增近远两条改善路径分别报告，两个pool对plain、scale对full共三个正式比较；近远均保留5%、至少一项改善5%，并保留用途、各任务族、共同历史和当前字段。必须同一改善路径覆盖两种子、两研究集合、best/last，不能各处挑较好的指标。完整raw/PCA/current用途协议独立保留，绝不自动晋升。固定矩阵完成即收束。

来源为checkpoints/babel_history_structure768及其既有祖先，不需要已清理的局部适配缓存或旧last。新输出不复制训练数组，不生成全位置embedding缓存。磁盘预检根据实际模型state、Adam矩、六组best/last、并行原子保存临时文件测算，另预留25%及1GiB报告空间；不足时在训练前退出，无自动删除。默认两worker/micro16，有效batch128，38更新/轮；同采样流401..500。保留来源期望的Torch/NumPy环境和原评分batch/容差。

启动前回放六份固定来源的原评分、验证来源状态因果性及实际micro零更新反向；全部训练完成后锁定best/last及780用途候选，再统一评价。可恢复完整epoch、优化器和RNG；失败也自动打包。训练/评价仅CUDA，本地检查仅合成数据、不进行神经优化器更新。

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash scripts/babel_causal_pool768_autodl.sh all > logs/babel_causal_pool768.log 2>&1 &
tail -f logs/babel_causal_pool768.log
```

报告自动导出 `/root/autodl-tmp/download/babel_causal_pool768_reports.tar.gz`。训练失败的包不能当作完成报告；检查command_exit_code与run_status。正式对照预算完成后无自动加训。已知上轮同机约6小时，新汇聚和新增验证推理有额外成本，首次暂按8–12小时安排，实际以逐worker ETA为准，最终评价另需时间。

中断后确认旧进程退出，原命令重跑即可恢复；用`>> logs/babel_causal_pool768.log`保留旧日志。只能降低并发`BABEL_CAUSAL_POOL_JOBS=1`后在原目录恢复；更改micro或源需新输出目录。环境变量BABEL_CAUSAL_POOL_SOURCE/RUN/LOG可更改路径，下载目录统一用BABEL_DOWNLOAD_DIR。

训练全部完成、仅评价失败时（不会补跑任何训练）：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_causal_pool768_autodl.sh evaluate
```

仅重新打包：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_causal_pool768_autodl.sh export
```

本地核验：100项合成与回归检查通过，覆盖新汇聚四路范围/有效通路、初始函数和共同参数、非零输出因果前缀与追加未来、相同目标梯度、micro累积、三组预算与中断恢复、第0轮回退、同末端验证不查询80、六组best/last到780候选和1224判据的完整流程、固定来源回放和篡改失败、按state大小测算磁盘、失败导出不含权重。原71份绑定代码与实际上传报告哈希一致，新运行绑定74份模块；Ruff和shell语法检查通过。本地没有真实模型推理、真实数据拟合或神经优化器更新；这些检查不能代替AutoDL数值路径和实际效果确认。
