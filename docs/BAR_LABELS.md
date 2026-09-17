# Per-Bar 结构标签 Schema（E12 标签地基 v1）

> 生成：`scripts/build_bar_labels.py`（引擎 `src/obson/pivot_labels.py`）
> 产物：`data/labels/{code}_{period}m_labels.pkl`，每根 bar 一行，与行情 csv 按行对齐。
> 审计：`scripts/audit_pivots.py` → `reports/pivot_audit/*.png`

## 定义

枢轴 = 价格从候选极值**反向运行 ≥ k×ATR(14)** 后确认的极值点。
纯几何定义；ATR 仅作波动率尺子，不作形态判定（标签宪法：指标不参与定义本体）。

尺度：k ∈ {0.75, 1.5, 3.0}（短/中/长，后缀 s0/s1/s2）。

## 字段（每 bar × 3 尺度 = 18 列 + datetime）

| 字段 | 类型 | 含义 |
|---|---|---|
| `seg_dir_s*` | int8 | 当前段方向：+1 上升段（自枢轴低点向上）/ -1 下降段 / 0 起始未定 |
| `bars_since_piv_s*` | float32 | 距上一枢轴的 bar 数（时间坐标） |
| `amp_since_piv_s*` | float32 | 段内从枢轴到当根的幅度 ÷ 当根 ATR，带符号（空间坐标） |
| `is_piv_high_s*` | bool | 本 bar 是该尺度枢轴高点（事件时刻标签） |
| `is_piv_low_s*` | bool | 本 bar 是该尺度枢轴低点 |
| `piv_amp_s*` | float32 | 若本 bar 是枢轴：前一条腿幅度 ÷ ATR（枢轴成色） |
| `confirm_delay_s*` | float32 | 若本 bar 是枢轴：确认延迟 bar 数 |

## 关键语义（不可违反）

1. **事件时刻 vs 确认时刻分离**：`is_piv_*` 打在事件时刻 t（事后全知，供训练）；
   实盘口径"当时可知性"由 `confirm_delay` 重建（t 时刻的枢轴在 t+delay 才确认）。
2. **段身份标签是事后口径**：`seg_dir`/`amp_since_piv` 用事件时刻枢轴切段，
   用于训练监督；实盘评估必须用确认时刻重算，差异由 confirm_delay 携带。
3. **幅度全部 ATR 归一**：跨品种、跨波动 regime 可比。
4. 标签是规则族 A（zigzag/ATR 阈值）。跨规则族泛化测试（族 B）是 E12 的毕业门禁之一。

## 全量生成快照（2026-09-17，本地 36 个文件，每文件 8964~8965 bar）

- 枢轴高低点数量完全对称（方向无偏置）
- 确认延迟：短尺度中位 1 bar，长尺度中位 6~7 bar、p90 12~15 bar
- 腿幅度中位：短 ≈1.7×ATR / 中 ≈2.6×ATR / 长 ≈4.9×ATR
- 本地 csv 为近期子集（tqsdk 单次上限）；完整历史标签需在 AutoDL 全量数据上重跑本脚本
- 跨品种审计图（rb 趋势市、sr 低波、i_5m 噪声市）均通过人眼一致性检查
