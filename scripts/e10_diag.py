"""E10-0 离线诊断（教师批准的第一步）：不训练，只回答三个问题

  Q1 未来表示有可预测结构吗？  → 恒等/线性 baseline vs 预测误差
  Q2 未来表示和主标签相关吗？  → 按 first-passage 标签分组的 z_b 相似度
  Q3 哪个视野更合理？          → 当日收盘锚 vs 一周对称窗口的可预测性对比

口径（教师 E10-0 条款）：
- 用冠军 encoder（models/champ_e3v11/s42.pt）纯前向，不改任何权重
- z_a = 样本输入窗口的 norm 输出 last 位置（与生产 readout 一致）
- z_b = 窗口末移到锚 bar 后的 norm 输出；mean-pool 覆盖未来段（主口径），
        last-token（对照口径）；a/b 两侧都只用 kline_seq（无日K/外盘上下文），
        保证表示空间同构（教师 §三.2：不同构则对齐无意义）
- 线性 baseline：train 段学 W: z_a→z_b（最小二乘），val 段测 MSE/cos
- 坍缩检查：z_a/z_b 逐维 std 分位数
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from obson.model.dataset import build_datasets, trading_day_ids, weekly_seq_len
from obson.model.transformer import KLineTransformer
from train_multi_symbol import _theta_c_dynamic

CKPT = "models/champ_e3v11/s42.pt"
SYMBOLS = ["rb", "sr", "p"]
PERIOD = 60
DEVICE = "cpu"


def norm_out(model, x: torch.Tensor) -> torch.Tensor:
    """抓主干 norm 输出 [B,T,H]。"""
    grabbed = {}
    h = model.norm.register_forward_hook(lambda m, i, o: grabbed.__setitem__("x", o))
    with torch.no_grad():
        model(kline_seq=x)
    h.remove()
    return grabbed["x"]


@torch.no_grad()
def collect(model, ds, df_raw: np.ndarray, day_ids: np.ndarray, n: int, horizon: str):
    """对每个样本算 (z_a, z_b)。horizon: 'close'=当日收盘锚 / 'week'=未来 N 根对称窗。"""
    seq_len = ds.seq_len
    vi = ds.valid_indices
    base = vi + seq_len - 1
    day_last = {}
    for idx, d in enumerate(day_ids):
        day_last[int(d)] = idx  # 递增覆盖，最后值=该日末根
    Za, Zb_mean, Zb_last, labels = [], [], [], []
    for s in range(len(vi)):
        j = int(base[s])
        if horizon == "close":
            a = day_last[int(day_ids[j])]
        else:
            a = j + seq_len
        if a - j < 2 or a - seq_len + 1 < 0 or a >= len(df_raw):
            continue
        x_a = torch.from_numpy(df_raw[j - seq_len + 1: j + 1]).unsqueeze(0).float()
        x_b = torch.from_numpy(df_raw[a - seq_len + 1: a + 1]).unsqueeze(0).float()
        ha = norm_out(model, x_a)[0]          # [T,H]
        hb = norm_out(model, x_b)[0]
        fut = hb[-(a - j):]                    # 未来段（j+1..a 对应的末位）
        Za.append(ha[-1].numpy())
        Zb_mean.append(fut.mean(dim=0).numpy())
        Zb_last.append(hb[-1].numpy())
        labels.append(int(ds.labels[s]))
        if len(Za) >= n:
            break
    return (np.asarray(Za), np.asarray(Zb_mean), np.asarray(Zb_last),
            np.asarray(labels))


def cos(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (an * bn).sum(1)


def mse(a, b):
    return float(((a - b) ** 2).mean())


def lin_fit(Za_tr, Zb_tr, Za_va):
    """岭回归 z_a→z_b，train 拟合 val 预测。"""
    X = np.concatenate([Za_tr, np.ones((len(Za_tr), 1), np.float32)], 1)
    W = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Zb_tr)
    Xv = np.concatenate([Za_va, np.ones((len(Za_va), 1), np.float32)], 1)
    return Xv @ W


def group_report(Zb, labels):
    """按标签分组：组内 cos 均值 vs 跨组 cos 均值（质心距离）。"""
    out = {}
    cents = {}
    for c in (0, 1, 2):
        m = labels == c
        if m.sum() >= 5:
            cents[c] = Zb[m].mean(0)
    for c, v in cents.items():
        out[f"centroid_std_c{c}"] = float(np.linalg.norm(v))
    pairs = [(0, 1), (1, 2), (0, 2)]
    for c1, c2 in pairs:
        if c1 in cents and c2 in cents:
            a, b = cents[c1], cents[c2]
            out[f"cos_c{c1}_c{c2}"] = float(
                a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return out


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = KLineTransformer(ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()
    report = {"ckpt": CKPT, "period": PERIOD, "combos": {}}

    for code in SYMBOLS:
        f = f"data/{code}_{PERIOD}m.csv"
        if not Path(f).exists():
            print(f"[{code}] 缺数据，跳过")
            continue
        df = pd.read_csv(f, parse_dates=["datetime"])
        raw = df[["open", "high", "low", "close"]].to_numpy(np.float32)
        seq_len = weekly_seq_len(df)
        n_train = int(len(df) * 0.7)
        theta = _theta_c_dynamic(df.iloc[:n_train], seq_len, 0.90)
        train_ds, val_ds, _ = build_datasets(
            df, seq_len=seq_len, target_offset=1, train_ratio=0.7, val_ratio=0.15,
            label_mode="day_close", label_threshold=theta,
            theta_mode="dynamic", soft_label=True)
        day_ids = trading_day_ids(df["datetime"])
        tag = f"{code}_{PERIOD}m"
        combo = {}
        for hz in ("close", "week"):
            Za_tr, Zbm_tr, Zbl_tr, y_tr = collect(model, train_ds, raw, day_ids, 3000, hz)
            Za_va, Zbm_va, Zbl_va, y_va = collect(model, val_ds, raw, day_ids, 1500, hz)
            for zb_name, Zb_tr, Zb_va in (("mean", Zbm_tr, Zbm_va), ("last", Zbl_tr, Zbl_va)):
                pred = lin_fit(Za_tr, Zb_tr, Za_va)
                combo[hz] = combo.get(hz, {})
                combo[hz][zb_name] = {
                    "n_val": len(Za_va),
                    "cos_za_zb": float(cos(Za_va, Zb_va).mean()),
                    "mse_identity": mse(Za_va, Zb_va),
                    "mse_linear": mse(pred, Zb_va),
                    "lin_improve": 1 - mse(pred, Zb_va) / max(mse(Za_va, Zb_va), 1e-12),
                    "std_za_p50": float(np.median(Za_va.std(0))),
                    "std_zb_p50": float(np.median(Zb_va.std(0))),
                }
            combo[hz]["label_groups_mean"] = group_report(Zbm_va, y_va)
        report["combos"][tag] = combo
        print(f"\n== {tag} (seq_len={seq_len}) ==")
        for hz in ("close", "week"):
            for zb in ("mean", "last"):
                r = combo[hz][zb]
                print(f"  [{hz}/{zb}] cos(z_a,z_b)={r['cos_za_zb']:.3f} "
                      f"恒等MSE={r['mse_identity']:.4f} 线性MSE={r['mse_linear']:.4f} "
                      f"线性提升={r['lin_improve']:+.1%} std(z_a)~{r['std_za_p50']:.2f}")
        print(f"  标签分组(mean): {combo['close']['label_groups_mean']}")

    out = Path("reports/e10_diag.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()
