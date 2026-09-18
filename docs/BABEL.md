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

## 两图对照评估（推荐的下一轮）

第一轮20例单人0/1/2盲评中，模型未展示人工感知优势，评分者也报告不确定。
因此保留原权重与第一轮结果，新增评估工具，不据此扩容或改训练目标。

```bash
cd ~/autodl-tmp/obson
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_RUN=checkpoints/babel_r1_s42
export BABEL_INDEX=data/babel/babel_r1_s42.npz
bash scripts/babel_autodl.sh blind-pairs
```

直接使用已有权重和索引，无需训练或重建索引。生成目录为
`checkpoints/babel_r1_s42/blind_pairs_v1/`，已有目录不会被覆盖。
下载其中的 `review.html` 到本地浏览器打开，无需联网或运行Python。

- 默认抽6个不同测试周的查询，比较模型与两条基线，共12组主要对照。
- 加3组隐藏重复，共15组；重复置于后段、左右互换，与原例隔开至少6组。
- 每组分别选**整体走势、转折顺序、右端位置**：左图更像／右图更像／差不多／无法判断。
- 不要求选择总冠军；不强制填满，不把空白和“无法判断”当作平局。
- 页面尝试在浏览器保存进度，失败时提示及时下载；全空文件不能导出。
- 下载按钮导出 `pair_ratings_<packet_id>.csv`，不会生成一个容易与真实评分混淆的空白CSV模板。
- 方法映射只在同目录 `answer_key.json` 中；不交给评分者。每份评分携带packet ID，防止文件混用。

```bash
PYTHONPATH=src python -m obson.babel score-pairs \
  --ratings /path/to/pair_ratings_PACKET.csv \
  --answer-key checkpoints/babel_r1_s42/blind_pairs_v1/answer_key.json \
  --out checkpoints/babel_r1_s42/pair_metrics.json
```

统计按三个维度分别报告胜／平／负、未判断数量、按查询周分块的偏好差值区间。
重复只检查同一评分者的稳定性，不计入方法成绩两次；比较时校正左右互换。
一致性高不等于正确，一致性低也不是评分者“做错了”；它帮助识别含糊题目。
6个查询与3个重复属于低负担试点，不足以支持强显著性、跨评分者泛化或毕业。
这是在已有测试历史上开展的新探索性评估，不能重新包装成一次完全未使用过的最终确认集。
统计区间反映查询抽样变化，不包含人工评分误差；不要把三个维度合成未经定义的总分。

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
# 盲测图形数据追溯

新的因果 bar 编码与自回归实验见 [BABEL_AR.md](BABEL_AR.md)，与下述旧版规则/检索流程分开运行。

完成评分后，可运行 `bash scripts/babel_autodl.sh audit-pairs`，输出当前
`BABEL_RUN/pair_quality.json`。沿用训练时的 `BABEL_DATA`、`BABEL_RUN`、
`BABEL_INDEX`。此步骤只使用 CPU，不加载模型、不重新训练、不修改数据或题包。

审计通过题包编号、模型指纹、源文件指纹及逐根 OHLC 匹配，将盲测图追溯到
具体合约与时间。输出每个窗口最大的五次价格跳空、相邻记录时间间隔、成交量、
零成交量和一字 K 线数量。超过一个周期的间隔可能来自正常休市或节假日，
不能直接当成缺失 K 线；跳空/窗口振幅中位数仅作描述，不是删除数据的阈值。
该报告也不替代交易日历校验或成为主力前的流动性审计。
