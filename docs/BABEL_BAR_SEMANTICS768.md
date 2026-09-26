# Bar Semantics768：一致的历史读出能否改善表示

## 目的与边界

已有control600能把128根历史压缩到末端768维状态，但原全局PCA头与原局部头在不同位置存在明显分工。decoder parity的同目标对照不能支持“80根不能重建”，也不能单独归因于位置编码。本轮检验：移除旧的编码器监督组合，只保留按bar相对距离定义的共享历史目标，是否能改善不同位置的读出一致性，并带来可用的信息增量。

这次不增加层数、头数或新的解码器结构：仍是768维、4层、8头、FFN3072；共享历史读出沿用History Query的4个192维记忆token、2层cross-attention。读出只接收一个bar状态和历史距离，不接收原行情、其他bar状态或真实价格锚点。编码器始终因果；重建过去，不训练下一根价格预测。

这里移除的是旧global/local/overlap编码器监督的**组合**，因此不能把差异归功于某一个损失。不同长度上下文还同时改变位置及直接历史，继承的因果EMA保留；结果不是纯位置编码因果实验。跨分辨率共同宏观内容仍保留为研究方向，本轮没有新增跨周期对齐约束，也不强行拉近完整embedding。

## 三组公平对照

|组|编码器接收的梯度|共享查询头|原局部头|
|---|---|---|---|
|control|原global + 0.25 local + 0.10 overlap|在断开梯度的状态上训练|照原方式训练|
|additive|原组合 + 1.0共享历史目标|正常训练|照原方式训练|
|uniform|仅1.0共享历史目标|正常训练|在断开梯度的状态上以0.25权重训练，仅作旧接口诊断|

每组种子42/43，从同种子的相同control600编码器/局部头和相同查询头出发。复用已经完成的50轮查询头预热best（两种子均选中第50轮），核对原编码器/局部头逐张量不变、预热来源与文件签名。不重做预热，不读取旧warm/frozen/control/joint的last.pt；保留的best.pt及manifest/completion/history/training_summary必须完整。

六组各100轮，每轮相同4789组三视图、14367个视图、38次编码器/局部头/查询头更新；有效batch128，micro16。每个视图四个随机非留出位置加末端128；留出位置48/80/112。三组使用同一采样、优化器重置及学习率日程。编码器1e-5、原局部头3e-5、查询头3e-4。所有组原PCA解码器固定。更新与曝光预算相同，不声称计算量相同。本轮additive的查询编码器梯度为1.0，不能直接当成上轮0.25 joint的复现。

目标使用当前已收盘价的相对历史坐标，历史距离1..127，沿用训练集尺度及sqrt(age)路径缩放；不可见历史严格mask，1..16、17..64、65..127三个可见距离段等权。查询头不获得真实当前价格旁路。训练同周期重叠视图及已有较粗周期视图，各自监督；uniform没有旧overlap编码器梯度。

## 选模、评价及停止

所有组用相同原生查询验证分数（32/64/96/128位置）选best，第0轮可回退；旧PCA/局部验证分数仅诊断。必须完成六组全部100轮后锁定best和last，再统一拟合12份状态各13目标×5档Ridge，共780个闭式候选；读出锁定后才评价原test984与cross453。两者均是重复使用的研究集，不能宣称新的独立泛化证据。

主要比较uniform与同预算additive：

- 固定相同结束时刻和过去16根真实目标，分别用32/64/80/128根直接上下文。预定80与128的输出差异至少下降10%，两者真实误差均值至少下降5%；防止一致地答错。记录每个长度实际误差和固定样例。
- 保留原生查询中途/末端表现：综合代价不超过5%，路径/逐根变化/实体/活动分别不超过10%。末端最远65..127根单独验收，防止只优化近处。
- 查询共同历史shift1/16/64差异不超过additive的5%代价。
- 校准后历史方向/波动用途对control、additive及原parent均保护（综合5%、任务族10%），当前bar七字段沿用NMSE≤0.1且R²≥0.8。
- 另要求用途相对control与additive均至少改善5%，才标为representation_candidate。仅一致性改善、用途保留但无增量，标为interface_gain_only。其他状态区分interface_gain_with_tradeoffs与no_uniform_interface_gain。

304项判据按周分组配对区间检查，两个种子、两个研究集、best与last都需支持，不能用总体均值掩盖失败。完整raw/PCA/current绝对用途协议仍单独报告，不因相对改善改写。原固定PCA/局部接口、旧共同历史表现全部报告迁移成本，但原PCA槽位坐标正是被替换的训练职责，不作为本轮新表示的硬性否决条件；长历史及实际用途仍设保护。

一次100轮矩阵完成即停止。候选不自动替换control600；只有接口改善不算通用表示突破，不因loss仍下降自动加训或扩网格。本轮不承诺逐点100%还原，也不把规则读出成功等同理解市场机制。

## AutoDL运行与空间

读取已有control600来源链及历史查询预热best；原始行情来源/root/autodl-tmp/data/contracts，既有样本缓存babel_architecture512/cache、候选池babel_sampling512/candidates。不会使用已清理的babel_local_readapt768/cache和babel_local_warmstart768/cache。严格检查完整来源链，不能继续删除受保护祖先。

不生成全位置embedding缓存。默认并行2组、micro16，首次预留9GiB可用空间（单组8GiB），含六组best/可恢复last及写入余量；这是保守预算，不是实测峰值保证。启动前检查空间。小批读取旧数据，评估临时状态不落成巨大缓存。上轮同架构、micro16的control/joint实际峰值分配约2.29GiB/进程，作为默认双并行的依据；不等于本轮实测。实机显存/速度以日志为准，本地未执行真实训练或CUDA推理。

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash scripts/babel_bar_semantics768_autodl.sh all > logs/babel_bar_semantics768.log 2>&1 &
tail -f logs/babel_bar_semantics768.log
```

每个worker打印轮数、query_val、old_val、best、每轮耗时和自身剩余时间；自身ETA不包含等待中的组及最终评价。默认两组并行，共三批；共享同一GPU并不意味着耗时减半。若确有显存不足，待本次进程退出后可将并行数改为1继续，保持micro与其他配置不变：

```bash
BABEL_BAR_SEMANTICS_JOBS=1 nohup bash scripts/babel_bar_semantics768_autodl.sh all > logs/babel_bar_semantics768.log 2>&1 &
```

同一目录重跑会校验并恢复未完成组，完成组不重复训练；改变实验配置须用新目录。不要同时启动两次。失败也自动打包诊断并保留非零退出码，完成会自动导出报告（不含权重）：

`/root/autodl-tmp/download/babel_bar_semantics768_reports.tar.gz`

手动重新打包：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_bar_semantics768_autodl.sh export
```

下载上述一个tar.gz即可复核。模型及断点留在checkpoints/babel_bar_semantics768，本轮结束前不要清理。

## 本地验证

70项相关合成/回归检查通过（bar_semantics、history_query、decoder_parity、research_state、local_warmstart、bar_alignment），包括三组梯度方向及微批累积等价、采样和预算、断点恢复/第0轮回退、预热best复用而不依赖last、780个合成读出候选至304项决策的完整流程、文件篡改拒绝、空间预检以及失败导出。神经网络优化器step在本地测试中均为no-op；闭式拟合仅使用小规模合成数据，不是本地真实训练。Ruff及shell语法检查通过。已与上传报告核对原history_query的65份和decoder_parity的66份源码签名完全一致；真实模型检查及效果仍须AutoDL完成。
