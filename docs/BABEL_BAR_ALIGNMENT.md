# 逐bar表示收敛实验：512/768 × 末端/多位置监督

主线是对已观察历史的因果压缩表示：保留全窗口信息，同时使不同位置的bar状态可用同一个头读取。上一轮已确认固定512PCA空间的细节上限，以及冻结共享头的保留位置失败；本轮回到编码器训练，只改这两个因素，不再扩层、扩头、改EMA或追逐趋势规则。

## 固定矩阵与公平控制

八格：w512_endpoint/joint、w768_endpoint/joint，分别seed42/43。全部从头200轮；原4789窗口/轮、batch128/micro64、7600次编码器更新。复用原train38549候选池及每轮采样/打乱计划，两种子内部各格使用相同窗口。两层、八头、FFN512、原28输入及七通道全局目标保持不变；768同时改变状态宽度和固定PCA rank，不宣称单独隔离embedding维度。

每格都增加同结构的共享历史头：状态→256→GELU→112，读取当前bar之前16根路径/实体/量仓。endpoint格仅切断局部目标到编码器的梯度，读出头照常训练；joint格允许局部梯度更新编码器。这样不会把“是否有训练过读出头”与编码器监督混淆。旧512模型仅作为来源/初始化参照，不混入新矩阵：读出头需要在编码器从头学习的过程中接受同预算训练，故八格全部重跑。

全局训练保持原SmoothL1。局部目标用原prefix训练尺度、三族等权SmoothL1，固定权重0.25，不再搜索权重。总训练损失为G_smooth+0.25L_smooth。原编码器AdamW lr3e-4、weight_decay.01；读出头独立AdamW lr1e-3、weight_decay1e-4；均使用200轮的5轮warmup/cosine日程，分别clip1。必须独立裁剪，避免对照头的梯度改变原编码器的裁剪系数。预检验证endpoint核心梯度与仅全局目标逐元素相同；joint局部梯度确实进入编码器。

## 位置与因果目标

每窗口每轮独立均匀抽4个不重复位置，从32..128中排除48/80/112，共94个位置。采样RNG与窗口/模型RNG分离，所有同种子格位置计划相同，保存指纹和逐位置曝光。单格957800窗口曝光、3831200局部目标曝光、7600次头更新；跨宽度或joint/endpoint计算量不同，不把等更新称为等FLOPs。记录参数量、训练时间、峰值显存，不推算未经测量的FLOPs。

48/80/112不接受直接局部监督，也不参与epoch选择，最后评价其插值。它们仍处于全局历史重建和其他局部片段内，不能称为完全未见数据。当前范围只覆盖32..128，不声称最开始几根bar已经获得通用表示。

每个局部目标排除当前bar，价格相对16根之前的收盘重新锚定，掩码与旧目标一致。目标构造使用与旧缓存相同的double计算/float32转换步骤。编码器全序列执行时使用因果掩码，通过严格截断前缀对照验证；早期状态不能读取后续bar。EMA等原输入特征仍可包含窗口之前的因果历史摘要。

## 选模、验收与停止

每格统一以原验证集32/64/96/128四位置的`G_MSE + 0.25 L_MSE`选epoch，包含第0轮。不用保留位置/研究结果挑epoch。全部八格完成预算并锁定后，复跑best/last验证，再评价原test984/cross453。导出全局与各位置的分项、一步相关性/幅度、训练/保留位置聚合以及固定样例；聚合先在同一原窗口内平均，再按原128端点周配对，位置不当作独立样本。

预先声明以下工程取舍门槛，均须两个种子、两个研究集、best和last通过：

- 相对本轮512_endpoint，保留位置局部primary至少下降10%。检验配对差`candidate - 0.90 * reference`的周95%区间上界≤0。
- 全局primary允许最多5%退化；path/changes/body/activity/closeMAE分别最多10%。同样对差`candidate - (1+margin) * reference`要求区间上界≤0，而不是用“没有检出显著退化”冒充非劣。
- 优先保留通过的512_joint。768只有相对较小方案额外至少5%保留位置局部改善、且全局保持上述边界时才升级；两个768方案均通过时，优先endpoint，joint也必须有额外5%收益。该选择顺序及每项失败原因写入decision.json。
- 若无新方案通过，保留512_endpoint并停止这条容量/监督扩展。若有方案通过，下一阶段转为状态接口的实际用途验证，不自动再加层、加维、加权重或延长预算。

这些是为研究收敛设置的探索性工程门槛，不是理论最优性证明。研究集已多次查看，区间未经多重比较校正；阶段路线筛选不冒称新的独立泛化证明。还应阅读一步幅度/各位置分项，不能把阶段通过解释为交易可用或自动部署。窗口继承原128端点资格筛选，不等于线上逐bar无选择偏差。

## 来源与恢复

依赖完整`checkpoints/babel_shared_readout512`及其prefix/sampling/teacher/architecture来源。原数据、权重、源码只读；PCA512使用原基与坐标尺度，PCA768从已完成的train-only完整PCA取前768维，坐标尺度仅在原固定train拟合。所有数组/采样计划/统计和源码指纹绑定manifest与检查点。续跑恢复两个优化器、RNG、绝对epoch及确定的位置计划；不重置日程。

## AutoDL命令

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_bar_alignment_autodl.sh all \
  > logs/babel_bar_alignment.log 2>&1 &
tail -f logs/babel_bar_alignment.log
```

默认两个worker并行，全部八格完成后自动评价，不会只跑两格就停。可设`BABEL_ALIGN_JOBS=1`降低并发；`BABEL_ALIGN_SOURCE`改来源，`BABEL_ALIGN_RUN`改新输出目录。重复相同命令恢复已完成epoch，不允许同输出并发启动。

成功或失败自动输出`/root/autodl-tmp/download/babel_bar_alignment_reports.tar.gz`。手工重新打包：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_bar_alignment_autodl.sh export
```

提交该压缩包即可，包含alignment_metrics、decision、examples、summary、验证回放、每格初始验证/历史/选模/预算/位置曝光及日志；排除.pt/.npy/.npz。正式训练只在AutoDL进行，本地仅合成前后向、数学检查及禁止神经优化器更新的完整流程测试。
