# V04/V05：RS-v2受控损失干预协议

2026-10-04。对应N01～N09中的因果、行情信息保留、匹配评价和工程要求；只执行V04/S1与V05/S2。依据R1/V07已复核结果，前瞻性取代旧RS的执行准入，**不修改R1失败判定或原保护门槛**。V07模块交换不是单独训练模块，信息损失归因仍开放。

## 准入修订和边界

旧RS要求R1双种子24项全部保留；该条件未过。此版允许在已知同期基线可能退化的情况下，研究“开/关损失项在这套有限适配配方中的影响”。这是新版本研究准入，不是宣称R1通过或存在稳定适配基线。

工程准入不变：绑定已复核V02/V14、R1和V07完成文件及所有依赖，旧局部头资格/梯度路线/目标资格均须通过。原Macro初始化及24值先复现。baseline前5轮还须复现R1低LR轨迹（原1e-6绝对/2e-5相对容差）；不符就停止并导出。共同起点、数值与目标资格失败不能被新准入豁免。

## 固定模型、数据和矩阵

- 原Macro control s42/s43 best97/94，768维、4层、8头；完整加载原输入适配器、原query参数和buffer。原局部16根头冻结；其源坐标偏好是解释限制，不等于信息增益。
- 以R1的Context源身份绑定原单合约/周期数据、train-only统计及255条缓存。train4789、val978；训练只看缓存最后128，四个随机前缀加128共五位置，保持原Context采样流。短前缀并不具有完整128历史。
- 主查询仍预测已发生历史1～127年龄的7字段；native mask只允许该前缀已见过去。局部目标为原16根价格与活动联合目标，原源价格坐标/统计不变。标签不进编码器或查询头。
- 四组×两种子，每组从同种子原Macro重新开始，不继承R1额外5轮。编码器与query共同更新，其他头冻结，不增层或宽度。

| 组 | 损失 | V04/V05对照 |
|---|---|---|
| baseline | Q + S | 同期适配对照 |
| without_remote | Q | 无局部项时移除S |
| with_local | Q + S + 0.25L | 有S时打开L到编码器的梯度 |
| without_remote_with_local | Q + 0.25L | 补全2×2，分别评价另一背景下的S/L效应 |

Q为原0.4路径＋0.2一步历史变化＋0.1实体＋0.3活动，按原可用年龄段/字段mask平均；S为原17～127根、4/16分块的结构Huber项，权重1；L为原局部头Huber联合目标，权重0.25。未开启L时仅计算detach后的诊断值，不进入优化目标；开启时只向编码器回传，局部头参数始终冻结。Q/S进入编码器和query。每个窗口五位置的原均值、尾batch按实际样本数归一化，禁止按micro数量平均。

每组固定30轮、每轮38步，共240 worker-epoch/9120更新。原batch128/micro8、eval-mode（有梯度）、新AdamW、原weight_decay、全encoder/query合并clip1.0、原FP32后端。s42编码器LR=3.066380815642346e-6、query=3.0663808156423456e-5；s43分别3.2648704606900738e-6、3.2648704606900735e-5。保持常数，不再扫LR。记录Q/S/L、有效目标、实际更新、采样hash、裁剪前范数和裁剪比例；损失改变导致的梯度/裁剪变化属于该配方效应，不能单归容量竞争。

## 选模、锁定和评价

1. 每轮val保留原24项full/native×endpoint/interior×price/near/mid/far/activity/structure。候选须全部≤源110%；按原full endpoint/interior price最差比最小选择，严格改善才替换，含epoch0回退。实际last30与best分开保存。只有epoch0合格不叫修复。
2. 同预算last30是因子比较主口径；best是独立val选择的候选口径，不混作同训练轮数。四组训练全部完成并锁定权重后才拟合用途头；用途头只用train拟合/val选原5档alpha。13原目标、原raw3584/current28/PCA768基线相同机会；合计21表示×13目标×5档=1365候选拟合机会，不增加编码器训练任务。
3. 读头锁定后才评分test与cross_research。两者是重复使用的研究集，不称最终独立留出。所有组、两种子、best/last都报告full/native四位置、近中远、活动、结构和物理log价格误差；另报告误差分布P50/P90/P99及分合约/周期结果，缺年龄支持不当作0误差改善。
4. 区间使用相同窗口成对误差，2000次固定种子bootstrap；日历月成组为主，合约×周期×月成组为敏感性分析，两者都要支持；少于6组或100窗口为证据不足。组内保留全部重叠窗口；不声称组间完全独立或已解决所有时间相关。事先声明四个因子对比及交互，不挑研究最优组合，不将多组未校正区间叫全局显著发现。
5. 每个候选（后三组）分别相对同期baseline和原Macro：四个near口径的误差均须至少降低5%，其他原24项均不超过110%，配对区间上界必须满足相应因子界限。两种子、两研究集、best与实际last均满足且best非epoch0，才报告“本配方近端修复获支持”。best有收益但last不稳只能记录有限候选/取舍。只比坏对照好不算修复。
6. 表示进展还要求冻结用途utility至少改善5%，direction/volatility不损伤超过10%，同样对同期baseline和原Macro、同区间/种子/研究集/best-last范围保护。当前字段单独报告，不能混入utility凑成绩。即使通过也仅为重复研究集上的有限进展，绝不自动提升主模型或声称预测成功。
7. 四个因素对比：without_remote−baseline；without_remote_with_local−with_local；with_local−baseline；without_remote_with_local−without_remote。逐原24项报告配对差；交互为both−without_remote−with_local+baseline。固定头兼容性与一般信息增益分开，不以组合好就称两个猜测都成立。

## 执行、停止与资源

单worker串行，固定两baseline先跑再其余六组；源/cache只读。原10小时上限已用758.652147969231秒，本入口累计上限35241秒，包含预检、训练、选模、用途拟合与研究评价；余下小数不另分配。总训练上限仍10任务（已完成R1两任务＋本次八任务），不增加轮数/更新。

按R1源checkpoint实际大小测算八组恢复/最佳权重、一次原子替换及约1.2GiB状态/报告余量；磁盘不够则停止，不删源。运行中显存/耗时有差异，不称等FLOPs。只从完整原子epoch恢复；若发现未提交epoch已开始，拒绝静默重跑，保留证据。累计超时预算不重置；所有组未完整结束时只报告incomplete，不排胜负。来源/数值/因果失败或非有限损失立即停止；所有组固定30轮不按结果追加。全部未达标即收束本配方，保留原Macro和V06/F01～F12。

入口、测试和六方面实施检查随本次代码交付；真实CUDA未运行，详见BABEL_RECOVERY_RS_VALIDATION.json。成功/失败/超时均导出到`/root/autodl-tmp/download`。只有入口交付完毕后的下方命令可执行，不复用旧RS组件命令。

## AutoDL运行与导出

使用R1/V07原RTX3080Ti、PyTorch2.8.0+cu128环境及其未删来源；改变设备/后端会拒绝，不能以调整容差绕过。启动：

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash scripts/babel_recovery_rs_autodl.sh all > logs/babel_recovery_rs.log 2>&1 &
tail -f logs/babel_recovery_rs.log
```

八任务串行；前两组结束后继续其余六组，随后自动锁权重、拟合用途头和评价。上限35241秒约9小时47分，是累计停止上限而非耗时预测。输出目录固定`checkpoints/babel_recovery_rs`。正常/异常结束均自动打包`/root/autodl-tmp/download/babel_recovery_rs_reports.tar.gz`。需要单独重新导出时：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_recovery_rs_autodl.sh export
```

原子整轮检查点允许在剩余预算内用同一启动命令恢复；覆盖日志前可另存旧log，恢复时可改用`>> logs/babel_recovery_rs.log`。未提交轮次不会自动重跑；遇到该错误只导出报告，不能删除pending文件假装没有更新。时间耗尽也不能换输出目录获得新预算。

报告含全部行身份、验证轨迹、分窗误差、用途预测/目标/读头及两例重建。P50/P90/P99是分窗/位置聚合误差的分位数，并非每个bar的误差分布。状态缓存与模型权重留在远端，导出的缓存索引列出省略文件的hash；没有这些资产不能在本地独立复跑模型。六方面实施检查：[验证记录](BABEL_RECOVERY_RS_VALIDATION.json)。
