# PCA教师早期退火：预声明协议

## 阶段目标

学习已观察历史的因果压缩状态。上一轮固定PCA解码的plain组已有综合、逐根变化及量仓收益；额外坐标监督早期验证有帮助，长期效果种子分歧。本轮只检验：首次训练早期逐步退出辅助监督是否保留早期收益，避免长期目标冲突。不新增架构，不改变特征，不预测未来。

## 两个新实验，四个只读对照

- 新组：anneal_s42、anneal_s43，从相同种子随机初始化，各200轮。
- 只读对照：已完成babel_pca_teacher512的plain_s42/s43、teacher_s42/s43，各200轮。从来源读取best/last权重并复跑验证及研究指标，不重新训练，不纳入300轮续训组。
- λ：epoch0到50均0.25；51到100线性下降，第100轮恰为0；101到200保持0。第75轮0.125，第99轮0.005。
- 原模型保持不变：28输入，128bar，512维Attention两层8头，FFN512；512维坐标投影，训练集PCA固定逆变换。残差关闭且冻结。
- 原训练保持不变：batch128、micro64，LR3e-4，原5轮warmup/cosine，AdamW0.01，clip1，FP32且TF32关闭。连续优化器，无阶段重置；数据顺序seed+epoch×1009。两worker并行，各200×38=7600步（4789训练窗口），总计15200步。
- 全部固定配置。已选择同batch的理由是隔离教师日程，不能同时用增加batch改变优化轨迹。
- 原始PCA、目标标准化、教师坐标均复用训练集拟合结果。没有重新拟合；原始数据缓存和旧模型不修改。最后一行只作为输入，不计重建损失；目标仍为127根已观察历史。

## 来源、恢复与评价

先检查旧实验完成状态、代码/文件指纹、四组历史LR/λ/预算、验证最小值选模与权重锁。新组epoch0验证需复现同种子教师对照的初始验证。正式运行要求与旧对照相同Torch/NumPy版本；GPU型号及运行时间另行记录。

恢复加载last模型、AdamW状态、绝对epoch和RNG，重放验证。原验证primary MSE是唯一选模指标；坐标辅助损失不参与选模，epoch0依然允许。两个新组全部结束并锁定后才打开研究数据做推理。报告所有六个神经组的best/固定末轮，明确选中epoch及该轮λ，不把λ仍大于0的best宣称为完全退出监督。

每个种子同时对plain和constant-teacher做周分组配对区间，报告路径/多尺度变化/实体/量仓/收盘MAE，提供逐窗口误差和固定等距例图。探索性验收：相对plain的综合改善在两个种子、两个研究集均有支持，同时没有上述分项的受支持退化。没检测到退化不等于等效；区间未作多重比较校正，不自动晋升模型。这仍是反复使用的test与cross_research，不是新留出集。

若退火仍不稳定，保留plain路线，不自动继续搜索权重或残差组合。200轮仅为预算，若最优在末端仍应承认未充分收敛。

## 极端路径诊断

预先固定两个既有端点：ag/60/SHFE.ag2606，2026-02-11 14:00；au/60/SHFE.au2606，2026-02-25 00:00。在选模锁定后的只读审计中，核对CSV时间/正价格/OHLCV合法性、前窗口锚点、log/asinh累计路径与缓存特征和目标。没有删除困难窗口，没有重算更有利排名。

本地CSV确有大幅变化，编码往返最大误差约2.24e-6/5.72e-7对数百分点；这只能排查数值编码，不证明原始行情真实。AutoDL还需与实际缓存对齐。原始文件缺失、不同快照、缓存目标不一致分别报告；缓存目标自身无法复现时报错。审计不提供交易所逐笔真实性核验。

## AutoDL

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_pca_anneal_autodl.sh all > logs/babel_pca_anneal512.log 2>&1 &
tail -f logs/babel_pca_anneal512.log
```

默认源checkpoints/babel_pca_teacher512，输出checkpoints/babel_pca_anneal512，原始CSV根/root/autodl-tmp/data/contracts。若实际目录不同，用BABEL_ANNEAL_SOURCE、BABEL_DATA_ROOT明确设置；不改源数据内容。可用BABEL_ANNEAL_JOBS=1串行运行；不改变训练批量。

正常结束或失败都会自动导出/root/autodl-tmp/download/babel_pca_anneal512_reports.tar.gz，含manifest、来源锁、初始化验证、训练历史、best/last指标、配对区间、逐窗口误差、路径审计、例图、日志及完成/失败状态；不包含.pt/.npy/.npz。手动重新导出：

```bash
bash scripts/babel_pca_anneal_autodl.sh export
```

同配置中断后重跑all恢复；evaluate只复跑评价和审计。源验证在开跑前完成；若原环境依赖或文件变化，报错而不是静默混用控制组。

## 提交前验证与边界

36项相关测试通过：日程边界、来源/坐标损坏拒绝、步数与曝光保护、epoch0选模、只读旧对照、研究集延后读取、完整合成流水线、带人工构造动量的AdamW恢复、失败自动导出及正确模块调度。512维原配置合成forward/backward通过，固定解码器无参数梯度。上述本地检查没有执行神经优化器更新；真实AutoDL CUDA训练/推理仍待运行。保留原teacher/capacity/architecture源码，避免破坏已经完成实验的来源指纹。

另行按结论→分项指标→选模→数据/代码来源自查，区分研究验收与训练完成、完全释放与选中时仍有λ、复制当前输入与127根历史重建；这是自查，不是独立同行评审。
