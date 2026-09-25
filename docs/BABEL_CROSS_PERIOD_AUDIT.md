# 跨周期共同历史：先核对原始数据

本轮只做CPU原始数据审计，不训练、不运行模型、不拟合读出，也不修改既有哈希绑定的训练模块。

用户的新问题是：同一合约、同一段已经结束的历史，在15/30/60分钟下提供的不同观察尺度，能否帮助学习共享行情信息？先证明时间与原始数值确实对应，才值得设计训练损失。原局部头暖启动适配暂列备选。

## 固定核对规则

- 按目录中的品种、文件名中的完整合约分别匹配15→30、15→60、30→60，不跨合约拼接。
- 沿用当前管线的bar开始时间解释，名义结束时间为开始＋周期。带时区数据转上海本地时间；无时区按原始本地时间解释。
- 用时间区间定位细周期bar，不依赖行号倍数。完整区间须严格具有2/4个预期网格点，且没有跨边界bar。计数不足、错位和空区间单列。
- 原始O/H/L/C分别取首开、最高、最低、末收；成交量求和；开盘持仓取首值、收盘持仓取末值，持仓不求和。
- 数值匹配固定绝对容差1e-6、相对容差0。价格、成交量、持仓的差异分别统计；持仓未知不当成0。
- `full_price_match`是完整区间且OHLC一致；`full_ohlcv_match`再要求成交量一致；`full_ohlcv_and_oi_match`再要求两个持仓端点可用且一致。这些是原始数据条件，**不是最终训练样本认证**。
- 部分区间即使数值完全相等也不纳入完整匹配。不用观测到的钟点分布宣称已经区分休市与数据缺失；本轮没有权威交易日历。
- 同时记录两端各至少128行历史的匹配数量，仅表示行数可用。它不证明128行日历连续，也未施加原训练划分、主力筛选、特征预热、来源指纹等条件。
- 非法OHLCV或时间顺序使对应文件无效，继续汇总后整次运行以失败退出并导出诊断，不静默跳过成成功。

输入包括全量原始文件，可能覆盖既有研究时段；本轮只核对原始等式，不根据研究误差选择模型配置。保存逐文件SHA256、代码SHA256、逐合约统计及失败例子。时间戳语义得到数值支持仍不等于供应商来源证书，当前快照也不能证明没有事后修订。

## 后续训练设计边界

只有严格匹配子集才可进入下一步可行性筛选。128根15分钟与128根60分钟拥有不同历史长度，不强迫整个embedding相等，也不直接相加已标准化的活动特征或EMA。先考虑共同已收盘区间的价格变化一致性，使用共同价格参考与时间边界，保留原历史监督和同周期重叠约束。

正式训练前仍须固定同预算control/辅助损失对照、种子和选模协议；所有成对样本须属于同一原划分，同一端点只看当时已经结束的粗细bar；排除当前bar的原监督边界不能悄悄改变。当前4层768维8头候选及原局部头保持不变。本文件没有宣告跨周期训练已经实现或取得收益。

## 运行与下载

AutoDL（无需GPU）：

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash scripts/babel_cross_period_audit_autodl.sh all > logs/babel_cross_period_audit.log 2>&1 &
tail -f logs/babel_cross_period_audit.log
```

脚本默认读`/root/autodl-tmp/data/contracts`，输出`checkpoints/babel_cross_period_audit`，成功或失败都自动导出：

`/root/autodl-tmp/download/babel_cross_period_audit_reports.tar.gz`

手动重新打包（无需重跑）：

```bash
bash scripts/babel_cross_period_audit_autodl.sh export
```

可用`BABEL_DATA_ROOT`、`BABEL_CROSS_PERIOD_RUN`、`BABEL_CROSS_PERIOD_LOG`、`BABEL_DOWNLOAD_DIR`覆盖路径。输出目录必须为空；再次运行请指定新目录，不覆盖旧审计。

本地仅CPU审计：

```bash
PYTHONPATH=src .venv/bin/python -m obson.babel.cross_period_audit --root data/contracts --out reports/babel_cross_period_audit
```
