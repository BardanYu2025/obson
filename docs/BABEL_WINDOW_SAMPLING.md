# 固定预算的窗口位置采样实验

完成更新：两种子200轮及控制复现已完成，通过预定探索性门槛；末轮结论一致，但一步变化幅度仍偏低。完整结果、复核范围和下一阶段判断见[BABEL_WINDOW_SAMPLING_REVIEW.md](BABEL_WINDOW_SAMPLING_REVIEW.md)。下文保留预先固定的实验协议。

## 目标与对照

在已观察历史的因果压缩任务中，检验更丰富的训练窗口切片是否有助于表示学习。采用通过严格覆盖审计的原训练分区，保持512根最小分区内历史要求，仅将候选端点步长128改为16。候选池38549窗口、684320根唯一合约周期bar，包含原4789窗口；新增唯一bar为11.64%，不能宣称新增八倍独立行情。

保持原plain配置：原生512维attention编码器（2层、8头）、28维因果输入、固定训练PCA逆变换、原归一化、SmoothL1训练目标；坐标辅助权重0，残差关闭。窗口仍为128根，监督前127根历史，最后输入bar不计入重建目标。PCA和坐标统计都从旧实验复制，不重新拟合。输入不包含教师重建。

新组sampled_s42/s43均从头训练，各200轮；每轮从38549候选均匀无放回抽4789，跨轮可重复。成员抽样使用独立NumPy RNG，由种子、绝对epoch、常数20260923确定，索引排序后使用原训练器的数据打乱规则。每轮38次更新、总7600次更新/957800次窗口曝光/种子；batch128、micro64、lr3e-4、5轮warmup及原cosine、AdamW/裁剪与原plain一致。剩余53个窗口仍组成最后一个有效batch，不丢样本。

旧babel_pca_teacher512中的plain_s42/s43为只读控制，预算相同。新训练不是从旧权重续训；每个种子的初始验证输出必须复现其原plain的epoch0。更换输入视角可能改变合约/时间/周期曝光分布，也包含少量新增原历史；不是纯位置的独立因果归因。相同更新数不等于相同实际耗时或充分收敛。

## 数据与恢复保护

- 先验证已完成的babel_coverage512严格审计、原始来源、全部缓存与PCA身份、旧代码及控制组预算。保留旧训练模块文件指纹，不修改旧缓存或源运行。
- CPU准备候选时再次核验原始CSV规范化内容指纹，只将原train且有512根分区内历史的端点加入新缓存。候选中全部4789旧端点的x/y/mask与原数组逐元素比较，超出原1e-6绝对/2e-5相对容差立即停止；掩码要求完全相等。
- 候选目标从各自窗口重新构造，按该窗口之前的价格锚点定义。不能截取另一窗口的累积路径目标直接使用。
- 新缓存放在独立candidates目录并使用只读mmap取小批量；每轮不将整个候选池复制到显存。每个种子的采样计划记录各轮成员SHA、分层曝光、累计不同窗口和唯一bar覆盖；最终每候选曝光次数随报告导出。
- 保存优化器、RNG、绝对轮次和验证选模状态。恢复加载原状态，核对过去采样记录并复现最近验证结果，再执行下一轮。采样不会消耗全局NumPy或torch RNG，恢复不重新开始日程。

## 评价与停止规则

两新组完成后锁定原验证主指标最小的检查点（epoch0有资格），再打开研究集进行模型评价。同时导出best和固定第200轮的各目标族/收盘误差、逐窗口误差、按周配对区间、旧plain控制best/last复现及PCA256/512基线。固定示例按均匀索引选取，不按成绩挑图。

探索性标准为两个种子、两个研究集主指标均改善，且未出现有统计支持的价格/实体/量仓/收盘MAE退化；未检出退化不证明等效。不挑选好种子、不以均值掩盖反向结果。test984及cross_research453已经多次使用，不是新的独立留出。无自动主模型晋升；若收益不稳定，保留原plain路线，不自动延长预算或追加采样配置。

## AutoDL运行

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_sampling_autodl.sh all > logs/babel_sampling512.log 2>&1 &
tail -f logs/babel_sampling512.log
```

默认并行跑两种子（BABEL_SAMPLING_JOBS=2），有效batch固定128，不调大以改变对照。初始CPU缓存准备时GPU闲置正常；随后会出现每组epoch、训练/验证主误差、最佳轮、耗时和剩余时间。保持旧实验torch/NumPy版本，脚本不自动升级环境。可设BABEL_SAMPLING_JOBS=1顺序运行，两种子矩阵和各自预算不变。

默认来源：checkpoints/babel_pca_teacher512与checkpoints/babel_coverage512，原始数据/root/autodl-tmp/data/contracts，新输出checkpoints/babel_sampling512。旧依赖路径需完整保留。可用BABEL_SAMPLING_SOURCE、BABEL_SAMPLING_COVERAGE、BABEL_DATA_ROOT、BABEL_SAMPLING_RUN指定路径，不得混入另一实验来源。

正常结束或失败自动导出：

```text
/root/autodl-tmp/download/babel_sampling512_reports.tar.gz
```

包含manifest、候选核验/库存/采样计划、两组历史/选模、best和last指标/逐窗口误差/区间、固定重建示例、日志和退出状态；不含.pt/.npy/.npz。将此文件下载发送即可。需要重新导出时：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_sampling_autodl.sh export
```

中断后重新执行all会核验来源并从last.pt接着跑。只补评价用evaluate；单独预检用preflight（会准备候选缓存，但无训练更新）。preflight成功的导出标为partial，因为200轮实验尚未完成。

## 本地验证边界

本地仅执行合成前后向、无神经优化器更新的流水线、采样/缓存/恢复/评价锁/导出测试及CPU原始数据准备检查。正式神经训练、CUDA预检、控制组真实权重推理在AutoDL执行。

55项相关检查通过，含13项新增测试；另以正式512维/2层/8头配置完成合成前后向及因果前缀检查，输出形状2×128×7、可训练参数3433472，无优化器更新。新候选缓存指纹写入检查点，即使重新计算候选文件索引哈希，也不能把变更后的数据接到已有检查点继续训练。旧实验源码没有修改。

真实966份原始合约周期文件、3370650行的内容指纹复现；在真实38549候选上预演两种子各200轮，最终均覆盖全部候选、684320根唯一合约周期bar，总曝光957800。seed42每候选被抽7–45次，seed43为9–44次。首轮唯一bar为412688/409840，低于旧非重叠窗口每轮612992：候选重叠使单轮信息重复增加，累计观察位置更多。固定窗口曝光并不匹配每轮唯一bar数量，这一差别必须与最终结果一起解释。本地预演不依赖模型权重，没有拟合新PCA；真实PCA及原二进制缓存的候选重放由AutoDL准备阶段执行。
