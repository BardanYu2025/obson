# Frozen State Rollout768：有限未来状态实验

2026-10-03。实现协议；结果尚未产生。总分支见[BABEL_STATE_ROLLOUT768_PLAN.md](BABEL_STATE_ROLLOUT768_PLAN.md)。原监督密度/全位置训练分支结论不因本任务而改变；从头全位置训练和完整上下文职责仍未被本轮回答。

## 问题与固定来源

固定Macro History control原validation best（s42 epoch97，s43 epoch94），编码器768维/4层/8头及原历史查询头均冻结。从真实滚动128窗口末端状态预测下一滚动128状态；真实目标129是E(2..129)，不是E(2..128)。每个端点独立重新编码，不能截取长窗口中间状态代替。因果EMA允许包含窗口前已知历史。

预测只接受初始状态，然后自由运行；所有未来bar、特征和状态只在标签/评分通路。主读价：`R_hat(h)=-D_price(z_hat(t+h),age=h)`恢复实际age尺度，`C_hat=C_t*exp(R_hat/100)`，全程只锚定已知起点收盘。另报age1积分。真实未来状态+D仅为oracle诊断，不是预测器/严格误差下界。只评价未来收盘，未实现未来OHLC/活动生成。

## 有限矩阵

5组×2既有来源种子×2LR=20次拟合，各60轮。L1仿射残差、N1和N8同768→256→768残差MLP；全从恒等状态开始。L1/N1用真实前状态做8对单步监督，N8从起点自由8步并完整反传；三者每个origin相同8个真实标签。Z-direct/Raw-direct分别从状态/原128×28输入，经256宽MLP直接预测8个累计收益，零输出起点。所有选模/评价是自由预测。

AdamW LR3e-4/1e-3、weight_decay0.01、clip1、有效batch256；micro由零优化CUDA预检确定，最多两个独立worker。同种子同epoch同样本顺序、相同更新数，不声称相同参数/FLOPs。源预训练成本不包含在小模型训练计时内。N1/N8初始权重相同，256仅为修正分支，768状态有直连。

状态损失是按训练真实相邻变化RMS标准化后的8步状态MSE，状态mean/std及增量尺度均只来自训练去重端点/相邻对。Direct用起点历史64收益RMS×sqrt(h)标准化价格MSE；RMS下限为训练同周期5%分位数和1e-6的较大值。无新增EMA/活动权重、解码器搜索或编码器更新。

## 样本与公平性

固定原train/val/test/cross清单，输入128及未来32必须完整处于同一合约/分区。仅按标签可用性排除，不按未来涨跌、波动或主力状态筛选。训练状态缓存9个端点，非训练33个；每周期步长为下一条有效bar，跨交易时段另报。对每个原始输入回放、每合约做因果特征截断检查、每来源做编码器因果检查，按端点去重存float32状态，禁止保存展开的未来×128输入。

训练简单均值基线拟合32个未来价格标签；学习组只优化前8个。统一32步资格防止各组样本差异，但包含完整未来标签资格偏差，应在报告解释。Direct只有8步输出，不伪造9..32比较。

## 选模、评价与退出

每轮val自由预测1..8标准化累计价格MSE选best，纳入epoch0。同分更早epoch，再更小LR。全部20组完成并锁定LR、best、简单基线后才提取研究集状态。选择简单基线为val上的flat、过去8/32平均收益趋势、训练同周期均值中的最优；每个对照都展示。

预定主候选N8：两种子×两个研究集分别至少5%价格MSE增益；新增收益MSE和绝对误差95分位不恶化超过10%。2000次配对自然月bootstrap，区间上界≤0且至少6月/100origin才算支持。比较Raw-direct的5%保留、N1/L1/Z-direct增量分别报告，不合并成“架构全面成功”。

逐h1..32报告MAE近似bp、normalized RMSE、增量误差、尾部、相对基线skill及32点同时bootstrap带，报告从第1步连续正skill的范围及非连续优势。1..8训练内，9..32压力测试。波动分层阈值来自训练；分周期/品种/跨时段支持不足只作描述。向量误差、余弦、范数和固定首256样本有效秩只作辅助，127根共享历史可能解释相似度。

固定6个按品种/周期排序的样例，另画预定规则的最大标准化误差失败样例。已多次使用的研究集不冒充新留出；图像像行情不等于预测收益。本轮结束不自动增轮、扩网格或晋升模型。

## 工程与导出

零优化预检→train/val状态缓存→val oracle/identity诊断→20次训练→选择锁→研究缓存→best/last评价→PNG/HTML→自动报告。每epoch原子保存模型/Adam/RNG/epoch，确定性epoch permutation支持中断恢复；恢复校验来源/代码/缓存/变换/资源配置。已完成任务不重训；并发失败结束同轮其它worker，保留最近完整epoch。

磁盘预检按原始端点数量给无去重上界、origin输入、20组检查点和导出余量；不删除来源。CUDA预检仅前向/反向，零optimizer step；实际GPU吞吐和峰值仍由AutoDL确认。默认提取batch16，有效训练batch256；最多双worker，micro默认128并按预检OOM减半。训练OOM/非有限输出不会标为成功。

报告包含所有小模型best/last、逐行价格预测/真实目标/尺度、状态逐行误差/范数/余弦、选择锁、历史曲线、排除清单与PNG。为控制磁盘，完整预测768维轨迹和旧大模型不重复打包；状态诊断可用远程绑定缓存和小权重重算，价格主指标可仅靠报告重算。原始缓存留在实验目录。未来state oracle明确标识使用未来标签。

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs
BABEL_STATE_ROLLOUT_JOBS=2 nohup bash scripts/babel_state_rollout768_autodl.sh all > logs/babel_state_rollout768.log 2>&1 &
tail -f logs/babel_state_rollout768.log
```

完成或报错均导出`/root/autodl-tmp/download/babel_state_rollout768_reports.tar.gz`。同命令可恢复同一环境下的中断；只重打包用`bash scripts/babel_state_rollout768_autodl.sh export`。不需要用户逐组回来索取命令。

## 提交前核验

56项合成/回归检查通过（18项本轮新增），AdamW.step均禁用；Ruff、shell语法及diff检查通过。97个继承源码模块相对779966e未改变；合成PNG已渲染检查，不是模型成绩。另作结论→指标→选模→来源的自查，详见[BABEL_STATE_ROLLOUT768_VALIDATION.json](BABEL_STATE_ROLLOUT768_VALIDATION.json)。真实源权重/GPU数值、资源和学习效果仍需AutoDL验证。
