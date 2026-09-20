# 一次启动的并行重建融合实验

复用上一轮冻结long512/short512向量缓存；不重训编码器，不使用分类标签训练融合。
三组共享同一批4789/978/984层次窗口：long、short、add（m + W LayerNorm(s)）。
add投影权重和bias为0，初始严格保留m；长短两份状态仍分别维护，输出不回写。
本实验仅覆盖完整128根片段端点；没有宣称完成逐bar刷新长期记忆的在线接口。

## 相同近期解码头

每组都从同seed初始化 Linear(512,256) + 固定64位置编码 + 2层256宽8头Transformer + 7通道输出。
近期解码器只接收该组单个512维向量，没有原始输入或短时向量旁路。
目标为最近64根行情，价格相对窗口之前的收盘价；实际价格还原只另外使用一个显式锚点。
损失：16/32/64三个尺度的原重建损失等权平均，再加0.25倍4/16根收盘变化SmoothL1均值。
long和short只训练同构近期解码头；add还训练短时投影。
add前5轮先冻结零初始化的投影，只训练近期解码头；之后再释放投影。
这使长期保护条件下的保底候选不只有随机解码器，也避免初始随机解码头立即扰乱长期表示。

## 直接约束长期解码

原分层历史解码器与结构头完全冻结并保持eval模式。
add训练loss = 近期loss + 原历史重建loss + 5×与原解码结果的一致性loss。
一致性使用同样的价格/通道/差分重建度量；原解码结果仅为损失目标，不作为融合输入。
历史loss先在每个窗口的有效片段内平均，再跨窗口平均，padding不参与。
冻结解码器仍允许梯度传到融合向量；用activation checkpoint降低激活显存，不打开dropout。

事先固定选择规则：验证集历史重建loss、历史收盘价MAE都不高于原模型的1.02倍，
从符合条件的轮次里选择近期验证loss最低者。epoch0恒等映射是保底候选；若最后选中0，
必须报告实验没找到合格改进，不能只挑更好看的训练轮次。
long/short按各自近期验证loss选择。分类标签只在所有模型选完以后用于统一岭回归读出。
测试历史误差可以超过2%，报告如实展示；2%是验证选择条件，不是测试性能保证。

## 并行与恢复

一次准备共享target_cache：原始历史目标、原解码结果、近期目标、mask、冻结表示、端点。
缓存校验数据/来源指纹；文件不放入下载包。准备阶段是串行，完成后并行启动独立进程。
默认jobs2：add和long先同时跑；空出位置后启动short。三组各自60轮。
每组有效batch128；add micro8梯度累积，两个近期解码基线micro128。
每epoch用相同种子的独立shuffle generator保持端点顺序一致；不同micro可能产生不同dropout采样。
AdamW lr3e-4、weight_decay=.01、clip1。没有未来预测监督或分类训练损失。
每组last.pt包含AdamW、RNG、当前/最佳权重和历史；恢复从上一个完整epoch开始。
主进程等待三组完成才统一测试。任一子进程失败会停止本次启动的其他子进程，保留日志和断点。
SIGTERM/键盘中断主进程也会清理它的训练子进程；不会终止无关GPU进程。
降低jobs可恢复同一实验；改变训练batch/micro/轮数需新目录。

单GPU并行不保证比串行快，默认2个进程限制竞争；各history记录实际epoch时间和进程峰值显存。
本地只进行合成前向/反向、调度/导出测试，真实CUDA训练在AutoDL执行。
原研究测试集已经用于多轮研发，不能作为最终独立交易预测验证集。

## AutoDL：一次启动、自动打包

```bash
cd /root/autodl-tmp/obson
git switch features/babel
git pull --ff-only origin features/babel
export BABEL_LARGE_RUN=checkpoints/babel_large512_s42
export BABEL_FUSION_RUN=checkpoints/babel_fusion512_s42
export BABEL_RECON_RUN=checkpoints/babel_recon_fusion512_s42
export BABEL_RECON_JOBS=2
export BABEL_RECON_BATCH=128
export BABEL_RECON_MICRO=8
export BABEL_RECON_EPOCHS=60
export BABEL_RECON_LOG=logs/babel_recon_fusion512.log
mkdir -p logs
nohup bash scripts/babel_reconstruction_autodl.sh all > "$BABEL_RECON_LOG" 2>&1 &
tail -f "$BABEL_RECON_LOG"
```

成功标志为 `Reconstruction matrix complete`，随后出现 `Download archive:`。
自动生成：`/root/autodl-tmp/download/babel_recon_fusion512_s42_reports.tar.gz`，直接下载上传。
包内包含三组history、metrics、原历史/近期重建图、总报告、覆盖、配置、各子进程日志，
不含权重或目标缓存。失败也会打包已有报告与日志，run_status.txt记录非零退出码。

如果只想重新导出，保持实验环境变量，执行：

```bash
cd /root/autodl-tmp/obson
bash scripts/babel_reconstruction_autodl.sh export
```
