# 2026-11-24分钟行情审计：独立迁入版本

这是只读CPU基础盘点，不是模型实验，不是完整训练资格检查。来源提交和SHA见`experiment.json`。审计核心代码与旧版本逐字节相同；只把测试import改为本目录模块，补充独立启动包装。没有重跑完整原始行情审计，没有重新声称数据已到齐。

`reports/20261124/`保存8份原始证据，字节与原提交相同。其receipt里的旧路径是当时的路径，不能当作新分支入口；`compare_old_minutes.py`仅为原比较的证据快照，依赖旧数据路径，不是当前命令。新分支不依赖原`obson`包。

输入：外部`/Users/bardan/private/tqdata_2015`，按main/index/month families读取Parquet；时间/数值/覆盖问题原样报告，不训练、不加载权重、不改变行情。仍未完成历史交易日历、as-of选样、量仓语义和缺段训练资格。

可选的只读重新盘点（只在需要新快照时运行，输出必须是新目录）：

```bash
python3 -m venv experiments/20261124_minute_audit_v1/.venv
experiments/20261124_minute_audit_v1/.venv/bin/python -m pip install -r experiments/20261124_minute_audit_v1/requirements.lock
PYTHON=experiments/20261124_minute_audit_v1/.venv/bin/python bash experiments/20261124_minute_audit_v1/scripts/audit.sh /Users/bardan/private/tqdata_2015 /tmp/babel_minute_audit_new_snapshot
```

本地合成检查：

```bash
PYTHONPATH=experiments/20261124_minute_audit_v1/src python3 -m pytest experiments/20261124_minute_audit_v1/tests -q
```

依赖文件固定原审计使用的直接包；它不是新CUDA训练环境的完整锁文件。历史报告没有模型/优化器，备份内容仅为原始证据、脚本身份与原行情清单。原始数据本体不在本目录。以后修改审计逻辑也新建v2，不覆盖本目录或旧报告。
