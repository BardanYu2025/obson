# 新时间段确认报告：来源扫描阻断，尚未评分

收到`babel_time_confirmation_reports.tar.gz`，SHA256：`85d7e04b9b485ed82ce6a11f0ab0f0f66be8985a05ad8faa74ef81fa4755d412`。报告仅含运行状态、来源图、日志和说明；退出码3，`blocked / unresolved_registry_lineage`。未生成raw_audit、coverage、模型评分或decision，不能据此断言没有新数据或判断512/768优劣。

## 原因与修复

扫描登记了49份manifest，追溯17个祖先；4个错误都是同一个未解析哈希`9e6f19ffd5c7eec083c0742af475c3703c79600a6b8d5e2a86a84b7e7ccdb220`。从先前coverage和sampling归档读取原文件字节，确认它是`babel_coverage512/audit_manifest.json`；采样报告还保存了相同字节的副本。

旧holdout扫描器只枚举`manifest.json`，但递归哈希引用规则会把`audit_manifest.json`也视为祖先，所以它报告了一份实际存在的清单无法解析。复用旧扫描器时没有覆盖这一后续新增格式，是本轮入口的兼容性遗漏。不是用户遗漏检查点，也不是模型失败。

新增独立`time_lineage.py`用于本轮：同时枚举两种清单，并验证覆盖审计的路径/哈希祖先确实到达原始数据来源；注册表锁也包含audit清单。旧holdout与其他已绑定源码不改。未忽略任何未知哈希，不跳过损坏清单、不补造源文件。

## 验证范围

- 12项相关测试通过，新增覆盖具名审计的哈希解析、注册表锁定、篡改后继续拒绝、无数据祖先/损坏JSON拒绝。原冻结执行、恢复、来源修改、输入目录和成功/失败导出检查保持通过。
- 从历史归档及已有manifest取回哈希一致的40份原登记条目（原报告49份），其中包含全部17个可达祖先；再补入确证的audit清单，共41份可用条目。用原AutoDL绝对路径离线回放图，得到18个可达祖先、0问题。其余9份非当前可达祖先的登记清单没有本地原文件；未声称完整复刻AutoDL注册表。
- 没有真实权重推理、训练或新时间段数据评价。AutoDL仍须对实际完整注册表及数据快照重新执行资格检查。

## 下一步

更新代码后使用`BABEL_TIME_RUN=checkpoints/babel_time_confirmation_v2`与对应日志，避免旧无manifest的blocked状态直接返回。既有blocked归档保留。完整命令和导出路径见[BABEL_TIME_CONFIRMATION.md](BABEL_TIME_CONFIRMATION.md)。若数据仍截至旧边界，之后应返回`no_later_raw_history`；目前尚未到该步骤，不能提前写成已确认无新数据。
