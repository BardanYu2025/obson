# 512/768冻结模型的新时间段确认

这轮不训练。固定512_endpoint、768_joint，各seed42/43及原best/last，共八个检查点；同时使用旧训练集拟合的PCA512/768参考。只回答新时间段上历史回读的综合收益和价格路径代价是否保持，不补训768趋势读出，不改变原晋升失败结论。

## 数据资格先于评分

1. 验证完整`babel_bar_alignment`及上游来源，扫描checkpoint注册清单和既往跨品种raw_audit。取已记录原始行情最晚**收盘时间**为全局边界，不用旧训练截止日冒充研究截止日。
2. 新原始目录须保留已登记单合约历史及其逐内容指纹，允许追加后续行情和同品种新合约。缺文件、改旧行情或来源链未解析则阻断。不自动下载、不拼接连续合约、不覆盖旧数据快照。
3. 所有128根输入的开始时间必须严格晚于边界；每合约至少512根历史才能输出。EMA等预热允许使用过去旧数据，但不计入新窗口回读目标。stride128、沿用前一交易时段成交量主力筛选，完整评价所有合格窗口，不按收益或分数抽样。
4. `asof`固定为首次运行时刻（上海时间），只读届时已经收盘的bar；可提前指定更早时刻，不能指定未来。源CSV、数据清单、模型选择、缓存、代码在评分前锁定；期间变化即失败。历史来源目录必须静止，不与其他写checkpoint注册表的实验并行。
5. 注册范围之外的删除记录、其他机器及人工研究不能靠程序证明未见，报告保留这一限制。新时间段仍需用户确认此前没有用于其他实验。原始CSV快照整体登记为已检查，后续不得将同快照改名作为另一份独立确认。

本地检查：1251份15/30/60分钟合约文件最晚开始时间为2026-09-12 02:15，与已有来源最晚开始时间一致，没有更晚数据。新资格逻辑进一步核验全部旧历史指纹一致，全局最后收盘边界为2026-09-12 03:00，之后0行。这不证明AutoDL没有更新；若AutoDL数据相同，程序退出码3并导出`no_later_raw_history`，不会调用GPU推理。尚未获得真实新时间段结果。

## 固定评价

- 只从末端向量通过原固定PCA逆变换重建之前127根；近期16根使用模型自身预测锚点裁切，不能拿真实锚点修正。
- 全局沿用原路径/变化/实体/活动加权评分，近期沿用原三族评分。保留分项、一步相关性/幅度、逐窗口误差、品种/周期/月的周配对区间，以及固定等距样例。两种PCA都是目标历史PCA，不是之前趋势探针的输入PCA。
- 每个原检查点先复现旧验证分数，之后评分新时间段；不在新数据上重拟合统计、PCA、解码器或选择best/last。两个模型串行推理，避免为了显存占用而竞争GPU；默认推理batch128，可设置256，结果仍需通过原验证回放。
- 单独检验768相对512全局及近期综合误差的配对改善；另外原全局primary最多5%、path/changes/body/activity/closeMAE最多10%的保留界限不变。两个种子和best/last均报告，至少50个窗口、5个周分组才给有支持的判断；样本不足输出`insufficient_support`，不当作模型失败。
- 这是新时间段的末端重建确认，不是原多位置完整晋升实验。即使`gain_and_retention_confirmed`也不自动替换模型；原研究集的路径失败照常保留。区间未做多重比较校正。

## AutoDL命令

已有合格的新数据快照后运行。新目录必须含旧历史加新行情，结构仍为`品种/交易所.合约_15m.csv`等。不要用追加数据覆盖原实验依赖的数据目录。

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
mkdir -p logs /root/autodl-tmp/download

# 修改为实际的新完整数据快照路径；不要创建空目录冒充新数据。
export BABEL_TIME_ROOT=/root/autodl-tmp/data/contracts_updated
nohup bash scripts/babel_time_confirmation_autodl.sh all \
  > logs/babel_time_confirmation.log 2>&1 &
tail -f logs/babel_time_confirmation.log
```

AutoDL脚本默认数据目录为`/root/autodl-tmp/data/contracts`，与之前窗口交付脚本一致，位于仓库外。省略`BABEL_TIME_ROOT`即检查该目录；若此前设置过其他值，应显式覆盖或`unset BABEL_TIME_ROOT`。命令行Python模块自身的相对路径默认值仍供本地使用。与本地数据相同就会阻断，没有必要为已知未更新数据启动GPU实例。

2026-09-24修复：首版AutoDL脚本误用仓库内`data/contracts`，可能在加载模型前报`Invalid batch, raw root or registry/source relationship`。已纠正脚本默认目录，启动时打印全部解析后路径，并将缺目录、batch、来源范围和缺来源文件分别报错。该入口错误发生在创建运行manifest前，修正后可直接使用原输出目录；原模型和数据均未修改。

成功、无新数据或失败都会导出：

`/root/autodl-tmp/download/babel_time_confirmation_reports.tar.gz`

只提交这个包，包含来源/资格检查、数据边界、固定样例、分项误差和判定；不含权重和原始数组。随时手工打包：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_time_confirmation_autodl.sh export
```

中断后同命令仅恢复原manifest、原数据快照和原asof，可重做冻结推理，不移动评分边界。已完成运行仅核验后返回。无manifest的阻断/失败需要新输出目录重新开始，例如设`BABEL_TIME_RUN=checkpoints/babel_time_confirmation_v2`；新数据快照也必须用新输出目录，之前评分过的历史会被注册表排除。手动export时保留相同`BABEL_TIME_RUN`，归档名随目录名改变。并发同输出被文件锁拒绝。

新时间跨度不足五周可能只得到描述结果；不预报未经测量的运行时长。现有源码和权重均只读，新增入口不改变旧模型包指纹。
