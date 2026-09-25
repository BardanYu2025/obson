# 固定768状态的前馈宽度与深度对照

2026-09-25，用户批准。当前一致性模型及新读出已保留有限用途，本轮只问：相同768状态和固定解码器下，更宽FFN/更多编码层是否增加保真并保留稳定性与用途。不是再次扩展状态维度或搜索一致性权重。

## 固定六组

| 分支 | 状态宽度 | Transformer层 | 头数 | FFN宽度 |
|---|---:|---:|---:|---:|
| base | 768 | 2 | 8 | 512 |
| wide | 768 | 2 | 8 | 3072 |
| deep | 768 | 4 | 8 | 3072 |

各seed42/43，共六组。固定原128根×28特征、单合约边界、训练归一化、PCA768解码器和768→256局部头。FFN512不是整个表示的512维瓶颈，因为残差主干仍768。wide/base回答FFN容量，deep/wide回答较宽FFN下的深度增量；不声称测完所有结构交互或证明最优层数。

## 共同起点与公平预算

六组从对应种子的consistency768/consistent_s*/best.pt总400轮开始；其best等于last400，来源须通过consistency_readout768完整恢复验收。保留旧单位，扩宽FFN新增第二线性层列为0、第一层新增单位随机初始化；新增两层为prenorm残差块，注意力out_proj与FFN第二线性输出置0，使初始化整体功能保持到FP32容差。新参数不是永久冻结，检查新增输出投影初始梯度及激活后内部梯度。迁移后回放原验证和因果性，训练后对全部best/last再查因果性。

**统一重置两个AdamW**：base/wide/deep均重置encoder和local-head动量，不将旧形状动量塞入新参数，不单独给某组重启优势。保留父权重但这是新微调阶段，绝不称完整优化器续训。所有组相同5轮预热至encoder1e-5/head3e-5，余弦下降到峰值0.1；weight decay沿用0.01/0.0001，分别clip1。新增神经参数无额外更大学习率，没有LR网格。实际中断恢复则完整恢复本阶段模型/AdamW/RNG与绝对轮次。

每组固定100轮，绝对401～500，每轮同一36106资格池中无放回抽4789对、9578窗口曝光，effective batch128对、默认micro64对、38次encoder与head更新。每组共478900对/957800视图/3800步，两种子各三组相同ID、shift1/16、局部前缀及顺序。shift64不参与一致性训练。数据资格和池来自前轮全量回放并绑定原始bank哈希，无重划分、无新增样本。保持原双视图global+0.25local监督＋0.10共同历史一致性。

相同曝光与更新数不是相同算力。记录参数量、每轮耗时、峰值显存和总时长；默认两个GPU worker并行。100轮为本次固定适应预算，不证明充分收敛；报告尾段曲线，不看研究分数后自动延长或扩大搜索。可能因优化策略/预算未充分释放容量而无收益，只能据此结束当前配置，不能断言加深普遍无效。

## 锁定与一次评价

只用原完整验证集global MSE+0.25local MSE选权重，epoch0可回退。六组全部完成后锁定best和last，best主结果、last确认，二者都必须过关，不能二选一。然后为12份冻结状态各13目标×原五档Ridge拟合，共780个闭式候选、156头；仍用原4789训练/978验证，全部头锁定后才提取研究集。父模型400轮配套的新读出冻结复用，同训练集和选参机会；原PCA/raw/current基线冻结复用并核验回放，不能把旧头不适配当作新增结构损失。

一次输出：原984/453研究窗口的全局、局部held/近期回读、路径偏移分解及样例；984/443共同队列shift1/16/64差异与A/B真实误差；13目标新读出、分品种/周期分组、预测真值mask和逐窗口误差；原raw/PCA/current完整用途条件并列报告。所有数据仍为反复使用研究集。

升级条件事前锁定，两个种子、两个研究集、best和last全部满足。交易周配对bootstrap1000次seed42，至少50窗口/5周，上界≤0；未作多重校正：

- wide相对base全局综合误差至少下降5%；deep相对wide及base均至少下降5%。
- 每个比较中，候选相对比较组和旧冻结400轮父模型，global/held/recent主误差最多增加5%，全局path/changes/body/activity分项最多增加10%。
- 每个位移共同历史差异、形状差、A/B综合真实误差及原锚点MAE相对两参考最多增加5%。不能用共同变坏换来稳定，也不能悄悄丢掉已取得的一致性收益。
- 校准后主用途最多增加5%，方向/波动族最多增加10%，比较双方四目标R²≥0.5；当前七字段R²≥0.8且标准化NMSE周上界≤0.1。

deep两个比较都通过则标deeper_candidate；否则wide通过则wider_candidate；否则no_capacity_upgrade。候选资格不自动改写原完整用途失败、不自动替换已交付512。为“更深”付出更多计算必须有明确增量，本次不追加头数/层数/损失网格。

## AutoDL启动与导出

保留原checkpoints/babel_consistency_readout768、babel_consistency768、babel_path768及全部原始依赖，不需要重跑之前阶段。原Torch2.8.0+cu128/NumPy2.3.2环境。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_growth768_autodl.sh all \
  > logs/babel_growth768.log 2>&1 &
tail -f logs/babel_growth768.log
```

总日志显示各worker逐轮进度、最好轮次、耗时及完成状态，详细逐轮日志位于checkpoints/babel_growth768/{base,wide,deep}_s{42,43}/run.log。默认jobs2、micro64对；32GB显存预计可运行，但正式峰值以实测为准。若需较小micro，首次运行可设BABEL_GROWTH_MICRO=32；改micro必须使用新的BABEL_GROWTH_RUN，不能静默覆盖已锁定配置。控制并发可设BABEL_GROWTH_JOBS=1，不改有效batch。

本轮是真实六组×100轮训练，比上一轮分钟级线性校准长。尚无该扩容结构的GPU实测，预计小时量级，以首批每轮耗时估计，不能承诺准确总时长。小显存占用不等于未训练；检查worker的epoch、验证值与耗时。

成功或失败自动打包`/root/autodl-tmp/download/babel_growth768_reports.tar.gz`，含六组history/选模/结构与预算、迁移及因果检查、拟合与评价、日志和状态，不含.pt/.npy/.npz。只发回这一份即可。

手动导出：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_growth768_autodl.sh export
```

支持BABEL_GROWTH_SOURCE/RUN/LOG/JOBS/MICRO、BABEL_DOWNLOAD_DIR。部分完成恢复未完成worker，已完成训练核验后跳过；读出未锁定时整批重新拟合，研究评分中断后重算，完整完成后仅核验不重复训练。所有旧绑定Python源码保持不变。本地只做合成检查，不执行真实模型训练、推理或拟合。

## 提交前核验

33项相关合成/回归测试通过，覆盖扩宽与加层的初始功能保持、新参数梯度、因果掩码、六组相同抽样与预算、epoch0回退、中断恢复、十二检查点读出锁定、十四份状态完整评价与原参考回放、worker进度/失败清理及失败导出。测试不执行神经优化器更新；闭式Ridge只使用合成数据。实际上传清单的45个旧绑定Python源码哈希全部未变，真实清单祖先路径和新实验清单只读核验通过。

本次自查从升级条件回查选模及数据来源；不冒称独立评审。正式CUDA数值、速度、峰值显存和容量收益尚待AutoDL，合成测试通过不等于扩容有效。训练摘要保留末20轮验证值和每轮斜率以诊断是否仍在改善，但不据研究集分数自动延长预算。
