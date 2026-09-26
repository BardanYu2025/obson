# 固定 control600：可调用的研究状态与可视化

2026-09-26。目的：把已经取得的结果变成可调用、可检查的固定版本，不再启动训练。512既有交付基线保留；此次工程检查通过不意味着control600通过全部用途门槛。

## 固定哪些内容

- 来源：`checkpoints/babel_cross_period768_v2_recheck`，严格核验完整训练来源、选模锁、读出锁、文件与代码哈希。
- 两个种子42/43的control最佳100轮，即累计600轮；不按研究结果挑种子、不混合向量。
- 768维、4层、8头、FFN3072；28个原因果输入通道；原固定PCA解码器和原768→256→112局部头。
- 原13个校准读出及各自训练标准化。既不拟合新的头，也不使用warm-start100轮局部头。
- 独立模型包包含模型参数、PCA基底、全部标准化、原局部头与读出；推理不再需要旧checkpoint目录。仍须本仓库匹配源码及依赖，包内绑定运行代码哈希。

## 接口语义

每一根**已经收盘**的bar，用截至它的滚动128根计算末端向量；单合约因果EMA/波动/成交持仓特征连续更新，观察512根后才输出。换合约必须重置，不能拼接主连；时间戳沿用训练源bar开始时间，`as_of`必须至少达到开始时间加周期。

输出：

| 字段 | 意义 |
|---|---|
| embedding | 当前滚动窗口的768维末端状态 |
| history | 从该状态并行解码的前127根历史；当前bar排除；还原绝对价格只用窗口前已观察收盘作坐标锚点 |
| recent | 同一全局输出裁取前16根历史，用模型自己的第111根预测收盘作近期锚点 |
| original_local | 原局部头的前16根历史，相对路径保留原尺度，不额外借真实锚点拼接绝对价格 |
| readouts | 13个冻结线性读出：历史16/64根方向效率、趋势拟合度、波动及当前字段；属于历史描述估计 |
| metadata | 合约、周期、可用时间、预热、来源及研究用途限制 |

不把一次窗口里的早期状态当作也已经看过128根。这里不提供未来价格、开仓建议或已经验收的趋势交易器。读出未通过完整raw/PCA/current用途协议的结论原样保留。

## 一次运行做什么

1. 从来源提取原始权重，独立构造架构并逐参数签名核验。
2. 保持原batch64，复现两种子、两研究集全部逐窗口重建误差（含全部原局部位置和近期裁切）与13读出预测。原严格数值容差保持；差异写入独立报告后失败。
3. 固定每研究集首尾两个不同合约，各9个连续滚动端点，两种子共72次：核对28通道增量特征、原缓存、独立批量状态、未来后缀不改特征前缀和快照恢复。
4. 全部通过且来源不变后，签发加载凭证。未通过不能通过公开加载接口读取包。
5. 导出脱离网络可打开的HTML。16个图例来自原报告等距样本，不挑好图；同时展示完整路径、逐根变化与活动，可切换近期16根。显示全研究集评分，避免用样例代表整体。

本地仅做合成权重/数据测试、已有报告的反标准化与页面检查；真实权重回放在AutoDL进行。没有优化器、拟合或新的选模。

## AutoDL命令

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_control600_autodl.sh all > logs/babel_control600_v2.log 2>&1 &
tail -f logs/babel_control600_v2.log
```

来源目录不同时，设置`BABEL_CONTROL600_SOURCE`。输出默认`checkpoints/babel_control600_v2`。原Torch2.8.0+cu128、NumPy2.3.2环境需保持；CPU数据核验时GPU空闲属于正常阶段。这不是训练，不需要跑100轮。

成功自动导出：

- `/root/autodl-tmp/download/babel_control600_v2_reports.tar.gz`：发回审查，排除权重与大型数组。
- `/root/autodl-tmp/download/babel_control600_v2_bundle.tar.gz`：自用模型包，含权重；仍需仓库源码，暂不需要上传。
- `/root/autodl-tmp/download/babel_control600_v2_review.html`：下载后本地浏览器打开，无需AutoDL打开HTML。

失败也自动打包报告并保留非零退出码，不导出本次模型包。需要重打包时：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_control600_autodl.sh export
```

## 调用示例

在成功检查后，Python可逐根调用：

```python
from obson.babel.research_state import load_bundle
from obson.babel.window_state import MarketBar

engine = load_bundle("checkpoints/babel_control600_v2/bundle", 42,
                     "MA/15/CZCE.MA601", 15, device="cuda")
# 使用真实单合约已收盘历史，先warm预热，再push逐根输出。
# bar = MarketBar(key, period, datetime, open, high, low, close,
#                 volume, oi, oi_available, closed, open_oi)
# engine.warm(bar, as_of=bar_close_time)
# result = engine.push(bar, as_of=bar_close_time)
# snapshot = engine.snapshot(); another_engine.restore(snapshot)
```

CSV调用（示例合约文件需真实存在，截止时间按自己的数据设置）：

```bash
PYTHONPATH=src python -m obson.babel.research_state_cli \
  --bundle checkpoints/babel_control600_v2/bundle --seed 42 \
  --key MA/15/CZCE.MA601 --period 15 \
  --csv /root/autodl-tmp/data/contracts/MA/CZCE.MA601_15m.csv \
  --as-of '2025-10-09 22:30:00' --device cuda --last-only \
  --out /root/autodl-tmp/download/control600_example.jsonl \
  --snapshot-out /root/autodl-tmp/download/control600_snapshot.json
```

去掉`--last-only`可逐根输出；快照续跑要求只提供新的bar。文件输出拒绝覆盖已有文件。推理API支持CPU，但正式数值回放使用原CUDA环境。实际运行时所需原始数据列遵循`data.validate_frame`，尤其持仓通常为`close_oi`，不要假定任意CSV字段名都可接受。


## 2026-09-26 输入结构恢复修复

首次打包在strict load报错，原因是control600继承了`PathInput`包装器，构造便携模型时却使用普通Linear输入。检查点没有损坏；失败发生在打包中，不涉及训练。即使control的路径开关为0，也必须恢复包装器及其original/projection权重、float64归一化buffer和enabled状态，不能删除字段、忽略strict检查或擅自打开开关。

模型包现在显式记录input_adapter，恢复时先构造相同结构，再严格加载全部状态并保留参数签名/原结果回放。合成测试覆盖普通输入、关闭/打开路径包装器的四层模型完整打包，以及768维4层8头FFN3072真实规格的合成参数前向一致性、float64保留和残缺状态拒绝。没有在本地运行真实模型。

修复后默认输出切换为`checkpoints/babel_control600_v2`与同名日志/下载文件，保留第一次失败目录，避免旧manifest源码哈希冲突；继续使用原训练来源，不重新训练。
