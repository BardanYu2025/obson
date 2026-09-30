# Grounded Readout768：同状态、同区间的零训练定位

2026-10-01。接Grounded History768资格失败，先补齐既有解码器基线和完整可复算导出。只定位现有读取链条；不增加训练矩阵，不改变此前24项资格判据或blocked结论。

## 这次具体比较什么

两种子仍取原Shared History control各自best（100/99）。直接读取资格阶段缓存的三种状态A/B/C，及同一1424组验证窗口内两份各含两个变化的请求。复用以下已选中的读取器：

- 原历史查询解码器：由原768维状态恢复历史价格位置，然后取两个指定端点之差；
- 上次40轮中按验证选中的正常辅助头；
- 上次已选中的仅状态、仅查询头及训练均值对照。

所有预测对应相同物理区间、相同标签、相同训练均值与尺度。既报告全部1424组，也报告原先按真实内容差异筛选的1266／1257组；不能根据本次预测重新筛选。错查询对照在同一个状态中交换两份请求，真实目标不动。

原查询头的价格输出按`年龄平方根×原delta_scale`逐端点还原，随后计算“较晚价格−较早价格”。共同的当前价格锚点自然抵消，不把真实价格额外提供给解码器。不拿两个不同年龄的标准化输出直接相减。

缓存已经执行了原查询头固定的状态归一化，因此直接调用原memory、query、blocks和output权重，避免反归一化再归一化造成额外舍入；合成测试验证与原forward逐值一致。加载原模型仅用于严格恢复来源，编码器随即丢弃，不执行其前向。

## 锁定与完整复算

原89份源码保持不变，新审计模块额外绑定。检查原control来源、资格completion、cache及selected权重哈希，验证缓存窗口身份、查询坐标和训练标签标准化。既有state_only/query_only的best保存在resume文件；这些文件未列入旧completion，所以本次先锁定它们的哈希，再逐项验证其40轮历史／选模与旧报告一致，并强制复放旧best的MSE及完整资格报告。不能把本次新增哈希误称为旧completion早已绑定；数值复放不过就停止，不悄悄接受替换。

本次补充导出完整窗口列表、行号、informative IDs、每个视图的年龄端点、训练目标尺度、所有真实目标、所有读取器预测，以及计算dtype。保留筛选前全体数据，能够在本地复算筛选、误差、合约bootstrap和原24项资格检查。旧筛选／资格计算使用float32；新增匹配误差计算用float64，原头物理坐标亦为float64。单位可由`normalized×scale+mean`还原为log-percent；绝对误差乘scale×100为log-bps。

对原头／新头、原头／仅状态、原头／仅查询、原头／均值及新头／仅状态分别做同合约配对MSE差值，bootstrap2000次、95%区间，factor=1。另各自比较正确／错误查询。报告点估计、MAE和区间，不设置新的晋升筛选。这里重用的是选过best的验证集，区间未校正多重比较，结果用于定位，不能写成独立泛化证据。

## 如何解释以及何时结束

若原头在相同状态和片段上明显更好，说明这些信息至少有一条现成读取路径，当前新辅助读取／训练配方没有充分利用；不能归因为状态容量不足。两个头训练经历和目标不同，因此这不是等预算架构竞赛，不能据此宣布原头架构普遍更优。

若两者都弱，只能说目前两条读取路径在这些短区间价差上不足，仍不能证明状态内不存在相关信息。无论结果如何，都不能把本检查等同于宏观形态检验，也不会自动恢复上一轮联合训练、扩层或追加头训练。两个种子完成一次定位即结束，保持原主线状态。

## AutoDL命令

默认读取`checkpoints/babel_grounded_history768`及它引用的原模型源，输出到新目录`checkpoints/babel_grounded_readout768`；不修改旧目录。需要保留两个qualification目录的cache.pt、selected.pt、state_only_resume.pt、query_only_resume.pt及原control来源。缺文件会明确失败，不补训、不重建缓存。

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_grounded_readout768_autodl.sh all \
  > logs/babel_grounded_readout768.log 2>&1 &
tail -f logs/babel_grounded_readout768.log
```

0次优化器更新、0次拟合、0次编码器前向。默认CUDA读取已有状态；预计比训练短得多，以实际日志为准，不承诺固定耗时。源Torch／NumPy版本必须相同。入口持有源的共享锁，防止与源任务写入同时运行；新输出另有互斥锁。同命令可重做中断的诊断，完成后只核验，不重复计算。

成功和失败均自动导出`/root/autodl-tmp/download/babel_grounded_readout768_reports.tar.gz`，上传这一个文件即可。补导出命令：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_grounded_readout768_autodl.sh export
```

本次的complete仅表示诊断完成，不表示上一轮资格通过或新模型胜出。CUDA原权重运行尚待AutoDL；本地只用合成张量验证坐标、完整流程及来源变更拦截。

本地验证：40项合成／回归检查通过，覆盖原头标准化接口逐值一致、年龄尺度／端点符号／锚点抵消、完整缓存与24项资格复放、JSON独立重算、0编码器调用、源／选模／权重变更拦截、完成后幂等及失败手工导出。Ruff与shell语法通过；原89份绑定源码逐哈希一致。未加载真实模型做本地训练，也未把合成通过视为真实CUDA回放已通过。
