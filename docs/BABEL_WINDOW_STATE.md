# 冻结窗口状态接口

本轮收束实验，交付原 `512_endpoint` 的两个 best 检查点（种子42/43），不训练、不拟合统计、不重新选模。原候选晋升失败决策保持不变。全局解码后的近期裁切已在上一轮复核；这里验证接口是否忠实复现已有模型。

## 状态定义

- 每个已收盘bar对应一个新滚动128根窗口，取因果编码器末端 `h128`，返回512维向量。当前bar包含在编码输入中。
- 编码器沿用原两层、八头结构与原固定PCA坐标解码器。这里没有自回归生成，也没有新增神经状态机。
- 原28通道特征、训练集标准化统计不变。EMA8/32、滞后成交量EMA和波动尺度沿单合约历史持续更新，不在滚动窗口边界重置。
- 每个实例固定 `品种/周期分钟/合约`；只接受来源包含的周期。换合约必须显式重置；不自动拼接连续合约、挑主力、用未来交易日信息。交易时段间隔保留。
- 首次输出要求512根已观测bar。这是保守的接口预热策略，神经上下文仍为128。复现原数据必须从原合约历史起点回放；任意截取512根并不保证EMA初始化完全相同。
- `as_of` 是调用方已观测到的时间；bar起始时间加周期不得晚于它。时间统一上海时区。CSV命令先按收盘时间筛选，再验证已收盘数据，未收盘价格不参与状态计算。
- 每根bar重新计算128根编码，不复用位置已变化的旧神经KV。`warm()` 仅更新历史特征、不调用编码器；`push()` 输出状态。

## 返回值

`metadata` 包含模型/权重指纹、种子、原选模轮次、合约周期、窗口开始、当前bar起始/可用时间、预热计数和能力边界。预热不足返回 `ready=false`，embedding/history/recent均为null。

`embedding` 是512维末端表示。`history` 返回前127根历史；`recent` 返回输入第112–127根，共16根。当前第128根没有被训练为重建目标，因此两路均排除当前bar。七个通道为收盘相对路径、实体、成交量与持仓变换量，不是原始OHLCV，也不包含上下影线。

全局收盘路径是相对窗口之前真实收盘的log百分比；绝对价格仅用该观测锚点还原。近期路径用模型自己预测的第111根重新锚定，再裁取第112–127根，绝不使用真实局部锚点修正预测。近期的绝对显示价格因此与全局对应片段一致。`valid_mask` 说明原目标可用性，不用于修改重建输出，也不是置信度。

`snapshot()` 保存因果特征累积量、最近128行、时间和模型身份。恢复必须同模型/合约/周期；CLI续跑CSV只能包含新增bar，重叠或倒序会报错。历史行情修订应从历史起点重放。

## AutoDL交付校验与导出

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download
nohup bash scripts/babel_window_state_autodl.sh all > logs/babel_window_state.log 2>&1 &
tail -f logs/babel_window_state.log
```

不训练；冻结回放原两研究集的全局/近期逐窗口误差，并为每集固定首尾两个不同合约，各种子逐根检查9个连续状态。核验28通道增量与批量特征、追加未来不改变前缀、原缓存窗口、末端表示和快照恢复。原检查点/来源hash前后保持不变。正式检查仅在AutoDL CUDA进行，Torch/NumPy版本必须与原审计一致。输出冲突或来源变更直接拒绝；重复执行保留同一打包权重并重新进行只读检查。

成功或失败自动打包，手动补导出：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_window_state_autodl.sh export
```

- `/root/autodl-tmp/download/babel_window_state_reports.tar.gz`：发回审阅，包含清单、校验、原始回放样例和日志，不含权重/数组。
- `/root/autodl-tmp/download/babel_window_state_bundle.tar.gz`：成功校验后另行导出的模型包，包含两个冻结权重、固定统计、来源、指纹与通过凭证，留存用于接口调用。未通过则不会导出新的模型包。

模型包不依赖旧checkpoint目录，但仍需本仓库匹配的实现和依赖，不是独立可执行程序。不要只更新旧源码后继续用旧包；加载会核验代码hash。42作为显式默认种子并非基于新的结果挑选，不自动平均两个embedding。可用环境变量 `BABEL_STATE_SOURCE`、`BABEL_STATE_ROOT`、`BABEL_STATE_RUN` 修改源审计、原始CSV根目录和独立输出路径。

## 调用

以打包回放计划中的固定合约为例，以下命令自动读取第一条计划，不需要猜合约名和结束时间：

```bash
cd /root/autodl-tmp/obson
PYTHONPATH=src python - <<'PY'
import json
import pandas as pd
from obson.babel.window_state_cli import run
p=json.load(open('checkpoints/babel_window_state/raw_plan.json'))[0]
r=p['inventory']
print(run('checkpoints/babel_window_state/bundle',42,r['key'],int(r['period']),p['path'],
          pd.Timestamp(r['end'])+pd.Timedelta(minutes=int(r['period'])),
          '/root/autodl-tmp/download/window_state_example.jsonl',device='cuda',last_only=True,
          snapshot_out='/root/autodl-tmp/download/window_state_snapshot.json'))
PY
```

该示例需要新输出文件名，防止覆盖已有结果。通用CLI为 `PYTHONPATH=src python -m obson.babel.window_state_cli --help`，支持 `--last-only`、`--snapshot-in`、`--snapshot-out`。去掉last-only后逐根输出，512根之前仅返回预热状态。

## 本阶段边界

基线质量证据来自原选定研究端点；这次原始回放检查的是一致性，不能把少量数值检查当作全市场逐bar质量验证。接口不输出未经检验的涨跌趋势标签、预测概率或开仓信号。全历史通用表示目标仍未完成。本阶段产物为可追溯的固定窗口表示接口，后续用途检验必须冻结表示、事先定义任务与比较基线，避免重新开启无边界容量搜索。
