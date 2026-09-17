# Babel：结构看盘与表示学习

分支：`features/babel`。训练在 AutoDL CUDA 环境，本地用于开发、回放和查看结果。

## 需求与实现

目标是盘中辅助人工判断，积累可复用的结构标签与 per-bar embedding。
首版包含可工作的规则基线、独立的因果表示训练通路，以及结构回放与检索工作台。
规则证据和模型估计分开展示，模型指标不被解释成交易胜率。

| 需求 | 实现 |
|---|---|
| 看清当前结构 | 三尺度确认枢轴、摆动方向、高低点趋势、区间状态、确认高低点参考位 |
| 识别形态 | 双顶/双底、头肩顶/底候选及当前颈线突破状态；可审计的规则，不宣称是独立模型能力 |
| 量仓作为输入 | 成交量、持仓变化与水平；`close_oi` 优先，缺失显式标记 |
| 盘中回放 | 按收盘可用时刻逐根查询，只显示当时已经确认的结构 |
| 历史参照 | 结构描述符、原始价格路径、学习表示三种检索；时间隔离、独立案例去重 |
| 可迁移表示 | 因果 Transformer 输出每根 bar 的 embedding；训练长度128/64混合，其他长度可通过新配置训练 |
| 独立验收 | 末根规则拟合评测、同查询检索对照、按周分块区间、隐藏方法的人工形态盲评 |

## 数据和时间契约

- 直接读取 `data/contracts/{code}/{contract}_{period}m.csv`，不需要预生成旧 E12 标签。
- 必需列：`datetime,open,high,low,close,volume`；持仓支持 `close_oi` 或 `open_interest`。
- 时间按上海时区解释。CSV 时间是 bar 起点，保守地以 `起点+周期` 作为可用时刻；跨休市的短 bar 可能延迟可用，不会提前使用。
- 每个合约单独计算 ATR、因果特征、标签和后续路径，永不跨合约拼价格或收益。
- 每根输入使用当时价格和此前统计量。修改未来不会改变历史输入、标签或模型因果前缀。
- 枢轴事件和确认时刻分开；标签更新发生在确认时刻，不回填过去。不确定同根高低顺序时，更新极值的 bar 不同时确认它。
- 幅度为带符号 `(close - 已确认极值价格)/当根ATR`。
- 主力选择使用**上一交易时段**的成交量领先合约，不读取旧的同日最终成交量日历。
- 交易时段通过所提供合约的日盘日期并集推导，夜盘映射到下一个已观察日盘日期。此数据日历不是交易所日历服务；没有下一日盘日期的末尾夜盘暂不作为查询锚点。
- 训练/验证/测试共享全品种、全周期的日历切分；验证历史上下文允许来自训练段，但验证/测试标签不会进入训练损失。
- 检索候选结束时间早于整个查询窗口开始，跨品种、跨周期同样适用。返回案例间也不共享时间区间，可能不足 K 个。
- 后续走势仅在结果结束时间不晚于查询时间时展示。不同周期的后N根不合并计算方向概率。
- 模型的均值中心化只拟合训练期向量；模型历史查询必须晚于用于选模的验证期。
- 数据哈希、特征 schema、切分、配置写入 checkpoint / index。数据或权重不匹配时要求重建索引。

主力选择只在已提供合约集合内有效；缺少合约会影响覆盖。单合约格式也可用于其他资产的研究，当前交付没有声称完成跨资产验证。

## AutoDL：安装和训练

使用带 CUDA PyTorch 的 AutoDL 镜像。Babel 不依赖项目旧的 transformers / tqsdk 训练链路；不要为训练重新安装整份项目依赖。

```bash
cd ~/autodl-tmp/obson
git fetch origin
git switch --track origin/features/babel  # 第一次；已有本地分支用 git switch features/babel
git pull --ff-only origin features/babel
python -m pip install -r requirements-babel.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'

# 默认13品种×60/30/15m；只有60/30m数据时，在所有后续命令前保持这个设置：
# export BABEL_PERIODS="60 30"

bash scripts/babel_autodl.sh audit
mkdir -p logs
nohup bash scripts/babel_autodl.sh train > logs/babel_r1_s42.log 2>&1 &
tail -f logs/babel_r1_s42.log
```

脚本在 CUDA 不可用或指定品种/周期缺文件时直接退出，不会悄悄改用 CPU 或少训某个组合。
已有 `best.pt` 的目录不会覆盖；新实验设置新目录：

```bash
export BABEL_RUN=checkpoints/babel_r1_s7
export BABEL_SEED=7
export BABEL_INDEX=data/babel/babel_r1_s7.npz
bash scripts/babel_autodl.sh train
```

参数默认：hidden64、2层、128/64混合上下文、warmup32、stride16、batch64、AdamW lr3e-4、30轮。
显存不足可设置 `BABEL_BATCH_SIZE=32`；`BABEL_EPOCHS` 控制固定训练预算，修改即视为新实验。
其他环境变量：`PYTHON_BIN` 指定解释器，`BABEL_DATA` 指定合约目录，`BABEL_SYMBOLS` 指定品种。
训练按验证末根任务 BA 选择 checkpoint，仅保存实验状态，不自动宣称毕业。没有自动断点续训；中断后保留已有 best.pt，重跑请选择新目录。

## 训练结束后

确认模型与实验方案冻结后才评估测试段：

```bash
bash scripts/babel_autodl.sh evaluate
bash scripts/babel_autodl.sh index
bash scripts/babel_autodl.sh benchmark
bash scripts/babel_autodl.sh blind
```

产物：

- `checkpoints/babel_r1_s42/best.pt`：schema、配置、权重、验证成绩、数据清单。
- `manifest.json` / `history.jsonl`：完整实验配置、各任务训练损失和逐轮验证指标。
- `test_metrics.json`：逐根末端评测，不做跨样本事件匹配；每个预测只匹配一个事件。
- `retrieval_metrics.json`：相同测试查询下三种方法的对照及配对差值，按查询周分块。
- `blind/blind.html`：离线人工盲评页面；评分后下载 `ratings.csv`。不要把 `answer_key.json` 给评分者。

```bash
PYTHONPATH=src python -m obson.babel score-blind \
  --ratings /path/to/ratings.csv \
  --answer-key checkpoints/babel_r1_s42/blind/answer_key.json \
  --out checkpoints/babel_r1_s42/human_metrics.json
```

数据、权重、索引、运行报告不进入 Git。保留与索引完全一致的合约数据快照，回到本地查看时同步必要产物。

## 查看工作台

```bash
PYTHONPATH=src python -m obson.babel serve \
  --index data/babel/babel_r1_s42.npz \
  --checkpoint checkpoints/babel_r1_s42/best.pt --port 8765
```

打开 `http://127.0.0.1:8765`。服务只监听回环地址；远程查看使用 SSH 隧道，不必开放公网端口。
训练前也可不用模型：执行 `index` 子命令不传 checkpoint，再以同样方式启动 `serve`。

## 验收边界

1. 工程正确性：前缀因果性、确认不回填、量仓、交易时段、切分、事件匹配、历史隔离、后续可知性、文件版本匹配和训练往返测试全部通过。
2. 规则拟合：只回答模型有没有学到这些已定义的结构任务。确定规则在其自身标签上的精度天然可达100%，模型不能借此宣称超越规则。
3. 检索段方向一致率仅为诊断代理。模型对两条基线的配对差值及不确定性必须报告，不能以一个绝对阈值替代。
4. 独立形态能力由冻结后的人工盲评检验；还需跨时间/品种、少样本迁移验证。没有人工评分时保持“未验证”，不会生成虚构评分。
5. 不从单轮失败推断容量不足，不从有限实验推断市场不可预测。交易辅助价值仍需真实使用反馈和独立实验。

## 本地开发检查

```bash
PYTHONPATH=src python -m unittest discover -s tests/babel -v
```

旧 `tests/test_*.py` 仍对应已归档模块，本分支未改写它们；Babel 使用独立测试目录。
旧 E12 代码和历史判决保留作档案，新特征与 checkpoint 不兼容旧 E12；不要混用旧训练、标签或检索脚本。
