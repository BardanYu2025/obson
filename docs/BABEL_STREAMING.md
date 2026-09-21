# 长短双状态因果推理接口

本轮不训练。不使用上一轮新读出头的共同选轮结果替代原专用头，保留：
- large512 joint最佳编码器、历史解码器、历史结构头；
- short512 b128最佳GRU编码器（原训练第21轮）；
- reconstruction融合实验的short专用近期头（验证集选中第33轮）。

## 一根bar如何更新

`ClosedBar`包含合约key、分钟周期、bar起始时间、OHLC、成交量、持仓量及可用标记。
只接收已经收盘的bar；调用者必须确认closed=True。起始时间不是信息可用时间，
输出`available_at=bar.datetime+period`。不读取后续bars，不拼接合约，不按夜盘/交易日自动清空状态。
缺bar/隔夜间隔保留实际时间差，沿用训练特征，未改变归一化定义。

1. `CausalFeatures`从当前bar更新前序收盘、历史收益方差和EMA8/32，生成与encode_context一致的18维输入。
2. GRU每根bar更新隐藏状态，输出short512。至少128根后才开放近期64根解码，与已验证暖启动范围一致。
3. 从该合约原始第0行开始，每128根一个固定片段；完成后只编码这个片段，缓存其局部向量。
4. 累积4个完整片段后，每完成新片段刷新long512，最多保留16片段。测试分区起点处的不完整片段丢弃，
   使用与原HierWindows相同的绝对行号网格。一般生产新合约从第0行开始，最早第512根输出综合向量。
5. 使用训练集均值/标准差分别标准化long与short，拼成1024维。没有可学习的新投影，不回写长短状态。

`long_age_bars`明确说明长状态距今多少根，范围0–127；两次刷新之间long保持不变。
近期解码对应当前bar；历史解码和历史结构读出对应`long_as_of`，不能把它们当作当前bar的结构预测。
首次准备好之前embedding=None，不用零向量冒充有效长期记忆。

综合1024维向量包含两个不同更新频率的状态，是截至当前的因果状态组合。
现有质量证据主要来自128根片段末端；逐bar接口正确性与中间bar表示质量是两件事。
本轮不会宣称已验证中间bar趋势分类、未来预测、交易收益，亦不保证长短解码头的曲线一致。

## 状态和重置

每个合约/周期一个DualStream。遇到另一合约或周期直接拒绝，调用者须显式reset_contract，
或建立独立实例。重置清除特征历史、EMA、GRU、块缓存、锚点和行号，避免换月混入价格跳变。
同一合约重复时间、逆序、短于周期的时间间隔、未收盘或非法OHLCV均拒绝。

为了复现原时间划分：从合约第0行用warm_features只预热特征，到分区起点调用start_partition，
只重置神经状态，保留由过去计算的方差/EMA。不能只从测试期第一行重新初始化特征并期望与缓存一致。
生产若从中途冷启动，它是一个新的起点，不能保证等同于读过全部历史的状态。

snapshot/restore包含方差、EMA、上一根价格/时间、GRU隐藏状态、片段缓存、价格锚点与模型标识。
可用atomic_save和torch.load(weights_only=True)保存恢复。恢复后输入必须从下一根开始，不得重复最后一根。
每个实例应由单一串行消费者按时间喂入；不是跨线程可重入服务。GPU异常中断后应从最近持久化快照重播。

## Python接口

```python
from pathlib import Path
from obson.babel.streaming import load_bundle, ClosedBar

stream = load_bundle(Path('checkpoints/babel_stream1024/inference_bundle.pt'),
                     key='rb/15/SHFE.rb2605', period=15, device='cuda')
# 依次喂入真实、已收盘的数据。以下数值仅示意接口。
bar = ClosedBar(key='rb/15/SHFE.rb2605', period=15,
                datetime='2026-09-21 09:00:00', open=3100, high=3105,
                low=3098, close=3103, volume=120, oi=10000, oi_available=True)
result = stream.push(bar)
# result: metadata、short、long、embedding、long_refreshed。张量均为快照，不会被下一根原地覆盖。
decoded = stream.decode()
# recent/history的coordinates为模型坐标，anchor是还原OHLC所需的显式价格锚点。
# structure.values依次是原定义的历史高点距离、低点距离、归一化高点片段年龄。
```

`decode_embedding(embedding, recent_anchor, history_anchor, blocks, metadata)`只用单个1024向量，
恢复两份原状态并调用原专用头；无需真实bars或GRU隐藏状态。锚点、跨度、时间元数据需随向量保存；push输出的metadata已经包含recent_anchor/history_anchor。
尺度逆变换和分片是确定操作，不是训练出的新解码器。

CLI可读取与原数据同格式的原始合约CSV（datetime/OHLC/volume，持仓字段close_oi或open_interest）：

```bash
PYTHONPATH=src python -m obson.babel.streaming \
  --bundle checkpoints/babel_stream1024/inference_bundle.pt \
  --csv /path/to/one_contract_15m.csv --key rb/15/SHFE.rb2605 --period 15 \
  --device cuda --emit-every 128 --out /root/autodl-tmp/download/stream_embeddings.jsonl \
  --state-out checkpoints/babel_stream1024/stream_state.pt
```

所有bar都更新；emit-every只控制写盘频率，设1可逐bar输出。已有输出文件会拒绝覆盖。
后续CSV只包含新bar时可加--state-in恢复，另用新的out文件。默认不会导出逐bar大型向量数据。

## AutoDL一次验收、自动打包

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_SHORT_RUN=checkpoints/babel_short512_b128_s42
export BABEL_RECON_RUN=checkpoints/babel_recon_fusion512_s42
export BABEL_FUSION_RUN=checkpoints/babel_fusion512_s42
export BABEL_STREAM_RUN=checkpoints/babel_stream1024
export BABEL_STREAM_LOG=logs/babel_stream1024.log
export BABEL_STREAM_BATCH=16
export BABEL_STREAM_REPLAYS=6
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
mkdir -p logs
nohup bash scripts/babel_stream_autodl.sh all > "$BABEL_STREAM_LOG" 2>&1 &
tail -f "$BABEL_STREAM_LOG"
```

依次执行：
1. 校验原数据、来源权重、融合缓存及近期头，生成本地inference_bundle.pt。
2. 全部984个缓存测试端点，通过1024向量还原原专用头，评价近期/历史重建和历史结构。
3. 固定均匀选6个测试端点，从原始合约起点预热特征，从测试分区起点逐bar运行完整接口。
   检查18维特征、long/short向量和解码结果与原批处理缓存一致，检查快照恢复及下一根bar的long年龄。
4. 输出重建JSON/HTML、逐次刷新时间线、数值容差和耗时；无optimizer，无新训练。

数值检查分为两级，均记录原始差异，不修改模型输出：
- 原严格cache比较仍记录atol5e-4、rtol2e-4与strict_cache_match。长状态和历史解码仍须通过它。
- 短状态新增同一缓存输入的独立单步与128根分块重放。流状态与独立单步必须满足atol/rtol均2e-5；
  单步/分块、分块/旧cache、流/旧cache及近期解码所有通道差异，均须最大绝对值<=.002且RMS<=.0002。
- 近期解码的开收盘价格坐标差额外要求：最大<=0.1 bp、开/收盘各自平均<=0.02 bp。
- 原始特征最大误差<=2e-5；1024向量拼接逆变换<=1e-5。记录Torch/CUDA/cuDNN版本与TF32设置。

这是预先明示的工程数值等价标准，不是位级一致性，也不是未来行情预测误差门槛。
选择价格上限是为了限制实际解码影响，不能因为向量误差小就自动宣布无影响。
同输入单步对照用于发现流状态、特征或重置实现错误；分块对照用于定位执行布局引入的数值差异，
但不能单凭此实验认定具体CUDA内核原因。六个端点全部写入诊断，检查失败也保留结果，最后统一失败退出。
旧版在第二个端点short maxdiff约.000666/RMS约.000074时直接中断；新版不会仅凭放宽allclose就通过。

本地用合成数据测试功能；真实512模型与GPU一致性由这条AutoDL命令验证。

成功标志：`Stream inference audit complete`，随后`Download archive:`且`audit_status=complete`。
下载并上传：`/root/autodl-tmp/download/babel_stream1024_reports.tar.gz`。
权重bundle保留在checkpoints/babel_stream1024/inference_bundle.pt，不放入报告压缩包。
失败也自动打包日志与部分结果，可直接发送。重跑从头做只读验收，不会更新源权重。
单独重新导出：`bash scripts/babel_stream_autodl.sh export`。
