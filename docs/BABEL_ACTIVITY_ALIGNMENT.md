# Babel：量仓历史信息保留实验

## 不变的总目标

学习每根已收盘 bar 对应的因果状态 embedding，压缩此前可观察的价格形态、交易活动及其历史关系，供历史解码和后续独立读出任务使用。长状态与短状态的整合方向不变。本阶段仅修改短状态的训练约束，不修改长状态、不扩大编码器，不训练未来预测，不模拟主动买卖或给潜变量赋予开平仓含义。

不能把单个指标下降、当前输入的复制、训练 loss 下降或测试集挑出的最好种子当作阶段完成。

## 上一阶段结论与本阶段问题

activity512 的6组30轮实验全部完成。两个种子的收盘 MAE 平均：原特征43.394bp、新特征43.281bp、输入分支43.580bp；新特征差异的周配对区间跨0。当前持仓变化率的线性读出 R² 约-0.13，输入分支门控系数约0.001。新特征/保守分支微调尚无稳定收益；这不等于交易活动没有价值，也不证明信息从 embedding 完全消失。

本阶段只问：显式要求状态保留已观察的量仓变化，能否提升历史信息的可读性，同时保住价格表示？

## 受控实验

control、aux005、aux020，辅助权重0/0.05/0.20，各种子42/43、30轮，共6组。固定28输入（原18+活动10）、512维两层GRU、原并行价格重建头、相同历史流和原始预训练起点。并非从上轮挑出的某个测试最佳模型继续。所有组新增同样的 LayerNorm+Linear(512,20)辅助头，相同初始化/RNG；control的辅助权重为0，其辅助头不作为有效读出报告。

原价格损失保持 recent_loss +0.25 detail。辅助项为训练集逐目标标准化后的 SmoothL1，每组5目标、共4组等权平均。原编码器LR3e-5、价格头/辅助头LR1e-4，TBPTT128，默认64流、同时2任务。辅助头只接收 endpoint embedding，不能直接读取原始输入。

20个目标为5种观察描述量 ×4个时间位置：

- 当前bar；
- 前1～16根bar的描述量均值（严格不含当前bar）；
- 前第4根bar；
- 前第16根bar。

5描述量：相对先前EMA32的log成交量、log成交量变化、asinh持仓变化率、asinh持仓变化/成交量、asinh成交量/持仓规模。均为变换后坐标，均值不是原始比值的均值。它们描述已发生的行情，不是未来监督。历史目标仅取同一合约同一数据分区内的观察值。

当零成交量却有持仓变化，或abs(持仓变化)/成交量>1，标记为待审计记录，屏蔽相应持仓变化两项目标；不声称它们违反普适市场规律，不删除原始输入。均值目标要求16条均有效。另报告整个17bar目标跨度无此标记的clean子集。缺失OI/零分母仍按上一阶段掩码处理。

## 选模与评价

所有训练组使用相同的验证集价格detail选模和原价格保留门槛，含epoch0。辅助/测试读出不参与选择编码器。这是一个刻意保守的“改善历史信息且保留价格”实验，不能排除被价格选模舍弃的其他权衡点。

- 保留价格重建、实体/影线/范围、量仓水平与变化、状态分类、旧头兼容、打乱embedding、分品种周期、固定图片和周配对评价。
- 每个冻结encoder重新拟合线性Ridge和小MLP（128隐藏维、固定seed1701、80轮、weight_decay0.001/0.01）。标准化仅使用训练集，正则/轮次由验证集选择。MLP训练仅允许CUDA。固定一个读出种子是本轮限制；主训练两个种子独立。
- 增加只看当前10维量仓输入的线性/非线性对照，检验过去信息是否超越当前量仓状态的持续性。该对照范围仅为量仓输入，不是所有当前价格输入。
- 当前目标单独报告。预先指定的主要终点：15个past-only目标的平均标准化MSE，在全部15目标有效的共同窗口上进行按周配对bootstrap。
- 同时报告all和clean的目标支持数和敏感性；统计量不把掩码0当成目标0。

自动研究证据筛选要求：两个种子均有非epoch0候选；相对本轮control，线性和MLP过去目标误差的周配对95%区间上界<0，all和clean均成立；收盘误差和detail不超过control的1.02倍，状态BA下降不超过1个百分点。同样要求相对当前量仓输入对照的过去目标误差配对区间上界<0，才能宣称超越当前输入的历史记忆改善。筛选只是判断进一步研究的证据，绝不自动替换主模型。

辅助目标直接监督的描述量更容易读出，只能证明这些信息的保留改善，不能当作未训练任务的迁移能力或市场机制理解。

同一历史测试时段已经被多轮实验反复使用，结果属于研究集证据。后续定型需要新的留出评估，当前不能宣称最终泛化或交易收益。

## 数据与复现

直接验证并复用已完成的checkpoints/babel_activity512/cache，免重读966份原始文件。大npy通过只读使用的符号链接复用，新目标另存本轮cache；上游manifest、cache哈希、原始encoder/head来源都校验。不要删除或移动上轮目录。新旧权重不覆盖。

相同参数重跑all可恢复每个主训练任务的optimizer和RNG，完成的主任务不重训；读出评价重跑。只有jobs可以改变，其他配置变化必须换输出目录。日志包含每轮验证价格与活动损失；活动loss跨辅助权重不能单独决定选模。训练量仓归一化参数、实际轮次、支持数、目标与阶段界限一起导出。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_ACTIVITY_RUN=checkpoints/babel_activity512
export BABEL_ALIGNMENT_RUN=checkpoints/babel_activity_alignment512
export BABEL_ALIGNMENT_LOG=logs/babel_activity_alignment512.log
export BABEL_ALIGNMENT_STREAMS=64
export BABEL_ALIGNMENT_JOBS=2
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_activity_alignment_autodl.sh all > "$BABEL_ALIGNMENT_LOG" 2>&1 &
tail -f "$BABEL_ALIGNMENT_LOG"
```

自动导出`/root/autodl-tmp/download/babel_activity_alignment512_reports.tar.gz`，包含成功/失败状态、目标、manifest、历史、线性/非线性读出、敏感性和配对结果；不包含权重或npy。手动：`bash scripts/babel_activity_alignment_autodl.sh export`。
