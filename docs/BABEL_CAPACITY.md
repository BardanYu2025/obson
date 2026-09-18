# 固定窗口的容量对照

目标：区分网络处理能力与单向量容量对历史重建的影响，为长期历史模块选择局部编码器。
本轮不实现长历史记忆、变长窗口、预测或新损失，也不要求单根波动标准差比强行到 1。

两组均为 hidden=256、encoder layers=6、heads=8、decoder layers=2、window=128，
EMA 18 维输入、原 7 通道目标和原损失。仅 latent 分别为 128、256。
两组参数量分别为 6,391,943 和 6,457,863；旧模型为 1,226,759。
解码器深度不变，但其宽度也从旧模型的 128 增至 256；与小模型比较不能将收益
仅归于编码器。两组大模型的解码器宽度相同。

两组从头训练 200 轮，batch=32、seed=42、AdamW lr=3e-4、weight_decay=.01、
clip=1、dropout=.1、stride=16。无学习率调度、早停或测试集选优。
共有的 input/encoder/decoder/output/context_input 使用相同初始权重；压缩层和
展开层随 latent 改变形状，分别初始化。初始化后恢复相同随机数状态。
不要把此匹配初始化解释为两种模型后续每个参数更新都能一一对应。

冻结参考实验 `checkpoints/babel_ae_ema_200_s42`。启动前核对数据、切分、预算、
batch、seed、损失与架构，保存原报告和 best.pt 指纹；不修改源文件。
小模型曾在预算第 31 轮重置 AdamW，新两组从头连续训练；因此大模型之间是
更严格的对照，小模型是历史参考，不是完全匹配优化轨迹的容量消融。

每轮记录同样的完整验证集多尺度指标；按原验证综合损失保存 best.pt。
last.pt 保存优化器、随机状态、历史和最佳候选，支持完整轮次恢复。
initial_validation.json 为随机初始化的验证结果，也作为最优候选（epoch=0）。
每轮 train_seconds、validation_seconds、train_validation_seconds 记录计算耗时，
不含 checkpoint I/O、启动预处理和最终评价；manifest 记录运行设备和 PyTorch 版本。
训练耗时应连同质量一起比较，200 轮相同并不意味着算力消耗相同。

训练结束后才执行测试评价、PCA 与结构探针和全量诊断。PCA 使用对应 latent 秩，
所以 z256 的 PCA 基线不同于 z128；不能声称两组 PCA 数字应该相同。
raw_last16 对照仍固定为最后 16 根原始 14 维；随机模型使用各自匹配初始化。
优先比较多尺度价格路径及状态读出；单根标准差比只作为辅助诊断，不能单独选优。
当前尚无专门的拐点/复杂形态验收指标，不从重建结果推断已学会头肩顶。
单 seed、重叠窗口和已多次检查的测试日期，不能证明新的独立泛化能力。

## AutoDL 顺序运行两组

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_DATA=/root/autodl-tmp/data/contracts
export BABEL_REFERENCE=checkpoints/babel_r1_s42/manifest.json
export BABEL_DOWNLOAD_DIR=/root/autodl-tmp/download
export BABEL_CAPACITY_BASELINE=checkpoints/babel_ae_ema_200_s42
export BABEL_CAPACITY_ROOT=checkpoints
export BABEL_CAPACITY_LOG_ROOT=logs
export BABEL_CAPACITY_RESUME=0
mkdir -p logs
nohup bash scripts/babel_ae_capacity_pair_autodl.sh > logs/babel_ae_capacity_pair.log 2>&1 &
tail -f logs/babel_ae_w256_l6_z128_s42.log
```

先完成 z128 的训练/评价/诊断，再开始 z256，避免两组争抢显存。
若 tail 提示文件尚未建立，稍后重试；总进度在 logs/babel_ae_capacity_pair.log。
第二组日志为 logs/babel_ae_w256_l6_z256_s42.log。
默认目录分别为 checkpoints/babel_ae_w256_l6_z128_s42 与 z256_s42。
新运行要求空目录，不能重复执行首次启动命令覆盖已有实验。

中断后确认旧进程停止。只恢复对应组，例如 z128：

```bash
export BABEL_CAPACITY_RESUME=1
nohup bash scripts/babel_ae_capacity_autodl.sh 128 all >> logs/babel_ae_w256_l6_z128_s42.log 2>&1 &
```

恢复单组不会自动开始另一组。若 z256 尚未开始，等待 z128 完成后设置
BABEL_CAPACITY_RESUME=0，用相同命令替换 128 为 256 并改用 z256 日志。
若只需重新评价或诊断，第二个参数用 evaluate/diagnose；export 只复制报告。
所有报告自动复制到 download 下各自同名目录，大权重不复制。
分别回传两组的 manifest.json、history.jsonl、ae_metrics.json、ae_diagnostics.json，
保留所属实验目录或打包，避免混淆。
