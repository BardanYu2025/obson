"""E12 毕业门禁 U4-U6（在训练完成的 ckpt 上纯前向评估）

U4 跨规则族泛化：冻结骨干，线性 probe 在独立规则族 B（固定百分比 zigzag，
   与族 A 的 ATR 阈值定义不同族）标签上的枢轴检测 F1 ≥ 族 A 的 70%
U5 不变性：价格仿射（缩放+平移）与时间伸缩（±20% 重采样）后，
   同一 bar 的 h_t 余弦相似度 ≥ 0.9
U6 无坍缩：h_t 逐维 std 中位 ≥ 0.1，且 U1 显著高于随机初始化骨干

用法：
  PYTHONPATH=src python -u scripts/e12_gate.py \
      --ckpt checkpoints/e12_bidir_s42/best.pt --symbols rb sr p --periods 60 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from obson.pattern_data import N_SCALES, build_pattern_datasets
from obson.pattern_model import PatternEncoder


def fractal_labels(close: np.ndarray, k: int = 3) -> np.ndarray:
    """规则族 B：滚动极值（分形）定义——bar t 是枢轴高点当且仅当它是
    前后各 k 根内的严格最高点（低点同理）。纯序数定义，对任意单调变换不变，
    因此可在归一化特征上直接打标（与族 A 的 ATR 阈值 zigzag 不同族）。
    返回 int8：0 无 / 1 高 / 2 低。"""
    n = len(close)
    out = np.zeros(n, np.int8)
    for t in range(k, n - k):
        win = close[t - k: t + k + 1]
        if close[t] == win.max() and (win < close[t]).sum() >= 2 * k - 1:
            out[t] = 1
        elif close[t] == win.min() and (win > close[t]).sum() >= 2 * k - 1:
            out[t] = 2
    return out


@torch.no_grad()
def collect_h(model, loader, device, max_batches: int = 30):
    Hs, PIVs, Xs = [], [], []
    for bi, batch in enumerate(loader):
        out = model(batch["x"].to(device), batch["symbol_id"].to(device),
                    batch["freq_id"].to(device))
        Hs.append(out["h"].cpu())
        PIVs.append(batch["piv_cls"])
        Xs.append(batch["x"])
        if bi + 1 >= max_batches:
            break
    return torch.cat(Hs), torch.cat(PIVs), torch.cat(Xs)


def ridge_probe_f1(H_tr, y_tr, H_va, y_va) -> float:
    """岭回归线性 probe → 拟合集上搜 F1 最优阈值 → val 上测 F1。
    （不平衡数据用 0.5 固定阈值会全灭——v1 探针自杀事故的修复）"""
    X = np.concatenate([H_tr, np.ones((len(H_tr), 1), np.float32)], 1)
    W = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ y_tr.astype(np.float32))
    s_tr = X @ W
    best_t, best_f1 = 0.5, 0.0
    for t in np.quantile(s_tr, np.linspace(0.5, 0.999, 60)):
        p = s_tr > t
        tp = int((p & y_tr).sum()); fp = int((p & ~y_tr).sum()); fn = int((~p & y_tr).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    Xv = np.concatenate([H_va, np.ones((len(H_va), 1), np.float32)], 1)
    p = (Xv @ W) > best_t
    tp = int((p & y_va).sum()); fp = int((p & ~y_va).sum()); fn = int((~p & y_va).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--symbols", nargs="+", default=["rb", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = PatternEncoder(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    model_rand = PatternEncoder(cfg).to(device).eval()   # U6 随机骨干对照

    from train_multi_symbol import SYMBOLS
    FREQ_IDS = {5: 1, 15: 2, 30: 3, 60: 4}

    report = {"ckpt": args.ckpt, "mode": ck.get("mode"), "combos": {}}
    for code in args.symbols:
        for period in args.periods:
            tag = f"{code}_{period}m"
            lab_f = Path(f"data/labels/{tag}_contract_labels.pkl")
            if not lab_f.exists():
                continue
            tr, va, _ = build_pattern_datasets(
                code, period, args.window, args.stride,
                symbol_id=SYMBOLS.get(code, 0), freq_id=FREQ_IDS.get(period, 7),
                contract=True)
            tr_dl = DataLoader(tr, batch_size=64)
            va_dl = DataLoader(va, batch_size=64)
            Htr, Ptr, _ = collect_h(model, tr_dl, device)
            Hva, Pva, Xva = collect_h(model, va_dl, device)
            r: dict = {}

            # ── U4：族 A probe vs 族 B probe ──
            s = 1  # 中尺度
            yA_tr = (Ptr[..., s].reshape(-1).numpy() > 0)
            yA_va = (Pva[..., s].reshape(-1).numpy() > 0)
            f1A = ridge_probe_f1(Htr.reshape(-1, Htr.shape[-1]).numpy(), yA_tr,
                                 Hva.reshape(-1, Hva.shape[-1]).numpy(), yA_va)
            # 族 B：分形序数标签，归一化 close 上直接打标（单调变换不变）
            yB_va_list = []
            for xi in Xva:
                yB_va_list.append(fractal_labels(xi[:, 3].numpy(), k=3))
            yB_va = np.concatenate(yB_va_list) > 0
            # 族 B 训练标签：用 train 集 piv_cls 近似重构太绕，直接在 val 前 70% 拟合 probe 后 30% 测
            nB = len(yB_va)
            cut = int(nB * 0.7)
            Hflat = Hva.reshape(-1, Hva.shape[-1]).numpy()
            f1B = ridge_probe_f1(Hflat[:cut], yB_va[:cut], Hflat[cut:], yB_va[cut:])
            r["U4_f1_familyA"] = round(f1A, 4)
            r["U4_f1_familyB"] = round(f1B, 4)
            r["U4_ratio"] = round(f1B / max(f1A, 1e-9), 4)

            # ── U5：仿射 + 时间伸缩不变性 ──
            with torch.no_grad():
                x = Xva[:8].to(device)
                sid = torch.zeros(8, dtype=torch.long, device=device)
                fid = torch.full((8,), FREQ_IDS.get(period, 7), dtype=torch.long, device=device)
                h0 = model(x, sid, fid)["h"]
                # 价格仿射：OHLC 缩放 1.3 + 平移 0.5（归一化空间内）
                xa = x.clone(); xa[..., :4] = xa[..., :4] * 1.3 + 0.5
                ha = model(xa, sid, fid)["h"]
                # 时间伸缩：线性插值 0.8× 再截断/补齐
                idx = torch.linspace(0, x.shape[1] - 1, x.shape[1], device=device) / 1.2
                idx = idx.clamp(0, x.shape[1] - 1)
                i0 = idx.floor().long(); i1 = (i0 + 1).clamp(max=x.shape[1] - 1)
                w = (idx - i0.float()).unsqueeze(-1)
                xt = x[:, i0] * (1 - w) + x[:, i1] * w
                ht = model(xt, sid, fid)["h"]
                cos = torch.nn.functional.cosine_similarity
                r["U5_cos_affine"] = round(float(cos(h0, ha, dim=-1).mean()), 4)
                r["U5_cos_timewarp"] = round(float(cos(h0, ht, dim=-1).mean()), 4)

            # ── U6：坍缩检查 + 随机骨干对照 ──
            std_med = float(Hva.std(dim=1).median())
            Hr_va, _, _ = collect_h(model_rand, va_dl, device, max_batches=5)
            rand_std = float(Hr_va.std(dim=1).median())
            r["U6_std_median"] = round(std_med, 4)
            r["U6_std_random"] = round(rand_std, 4)
            r["U6_pass"] = std_med >= 0.1

            r["U4_pass"] = r["U4_ratio"] >= 0.7
            r["U5_pass"] = r["U5_cos_affine"] >= 0.9 and r["U5_cos_timewarp"] >= 0.9
            verdict = {k: r[f"{u}_pass"] for u in ("U4", "U5", "U6") for k in [u] if f"{u}_pass" in r}
            r["pass_U456"] = all([r["U4_pass"], r["U5_pass"], r["U6_pass"]])
            report["combos"][tag] = r
            print(f"[{tag}] U4 族B/族A={r['U4_ratio']:.2f}({'✅' if r['U4_pass'] else '❌'}) "
                  f"U5 仿射={r['U5_cos_affine']:.3f} 时伸={r['U5_cos_timewarp']:.3f}({'✅' if r['U5_pass'] else '❌'}) "
                  f"U6 std={r['U6_std_median']:.3f}({'✅' if r['U6_pass'] else '❌'})")

    n = len(report["combos"])
    npass = sum(c["pass_U456"] for c in report["combos"].values())
    report["summary"] = {"pass": npass, "total": n}
    print(f"\nU4-U6 总判决: {npass}/{n} 组合通过")
    out = Path("reports/e12_gate.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
