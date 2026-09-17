"""E12 死因定位诊断（训练后跑）：三个疑点一次查清

  Q1 A1 margin：因果版枢轴容差 F1 vs 三条基线
     （随机骨干 / 按枢轴率乱猜 / 只看过去的滚动极值规则检测器）
  Q2 双向倒挂：双向版与因果版同 val 集同口径对比 + 分量 loss 对照
  Q3 坐标躺平：时间/幅度头的预测方差 vs 目标方差（R²≈0 的解剖）

用法：
  PYTHONPATH=src python -u scripts/e12_diag.py \
      --ckpt checkpoints/e12_causal_s42/best.pt \
      [--ckpt-bidir checkpoints/e12_bidir_s42/best.pt] \
      --symbols rb sr p --periods 60 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from obson.pattern_data import N_SCALES, build_pattern_datasets
from obson.pattern_model import PatternEncoder, gate_metrics

FREQ_IDS = {5: 1, 15: 2, 30: 3, 60: 4}
TOL = 3


def tol_f1(piv_pred: torch.Tensor, piv_gt: torch.Tensor) -> float:
    """全尺度全类合并的容差事件 F1（±TOL bar）。"""
    vals = []
    for s in range(N_SCALES):
        for c in (1, 2):
            p = (piv_pred[..., s] == c).reshape(-1)
            g = (piv_gt[..., s] == c).reshape(-1)
            gi = g.nonzero().flatten(); pi = p.nonzero().flatten()
            if len(gi) == 0:
                continue
            hit_g = torch.zeros(len(gi), dtype=torch.bool)
            hit_p = torch.zeros(len(pi), dtype=torch.bool)
            for ii, x in enumerate(gi):
                m = (pi >= x - TOL) & (pi <= x + TOL)
                if m.any():
                    hit_g[ii] = True
                    hit_p |= m
            tp = hit_g.sum().item()
            vals.append(2 * tp / max(2 * tp + (~hit_p).sum().item() + (len(gi) - tp), 1))
    return float(np.mean(vals))


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, gts, t_pred, t_true, a_pred, a_true = [], [], [], [], [], []
    ms = []
    for batch in loader:
        out = model(batch["x"].to(device), batch["symbol_id"].to(device),
                    batch["freq_id"].to(device))
        ms.append(gate_metrics(out, batch))
        preds.append(out["piv_cls"].argmax(-1).cpu())
        gts.append(batch["piv_cls"])
        t_pred.append(torch.expm1(out["bars_since"].cpu().clamp(min=0)))
        t_true.append(batch["bars_since"])
        a_pred.append(out["amp_since"].cpu())
        a_true.append(batch["amp_since"])
    agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    agg["A1_tolF1"] = tol_f1(torch.cat(preds), torch.cat(gts))
    tp_, tt = torch.cat(t_pred), torch.cat(t_true)
    ap_, at = torch.cat(a_pred), torch.cat(a_true)
    agg["time_pred_std"] = float(tp_.std()); agg["time_true_std"] = float(tt.std())
    agg["amp_pred_std"] = float(ap_.std());  agg["amp_true_std"] = float(at.std())
    return agg


def rule_baseline(loader) -> float:
    """只看过去的滚动极值检测器：bar t 是最近 k=4 根的最高/最低 → 预测枢轴。"""
    preds, gts = [], []
    k = 4
    for batch in loader:
        cl = batch["x"][..., 3]  # 归一化 close
        B, T = cl.shape
        pred = torch.zeros(B, T, N_SCALES, dtype=torch.long)
        for t in range(k, T):
            win = cl[:, t - k:t + 1]
            is_hi = cl[:, t] >= win.max(dim=1).values
            is_lo = cl[:, t] <= win.min(dim=1).values
            pred[:, t, :] = torch.where(is_hi, torch.tensor(1),
                                        torch.where(is_lo, torch.tensor(2), torch.tensor(0)))[:, None]
        preds.append(pred); gts.append(batch["piv_cls"])
    return tol_f1(torch.cat(preds), torch.cat(gts))


def trivial_baseline(loader) -> float:
    """按枢轴经验频率随机猜。"""
    gts = []
    for batch in loader:
        gts.append(batch["piv_cls"])
    gt = torch.cat(gts)
    rng = torch.Generator().manual_seed(0)
    rate = (gt > 0).float().mean().item()
    pred = (torch.rand(gt.shape, generator=rng) < rate).long()
    return tol_f1(pred, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-bidir", default=None)
    ap.add_argument("--symbols", nargs="+", default=["rb", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from train_multi_symbol import SYMBOLS

    vals = []
    for code in args.symbols:
        for period in args.periods:
            if not Path(f"data/labels/{code}_{period}m_contract_labels.pkl").exists():
                continue
            _, va, _ = build_pattern_datasets(
                code, period, args.window, args.stride,
                symbol_id=SYMBOLS.get(code, 0), freq_id=FREQ_IDS.get(period, 7),
                contract=True)
            vals.append(va)
    val_dl = DataLoader(ConcatDataset(vals), batch_size=64)

    report = {}
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = PatternEncoder(ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    report["causal"] = eval_model(model, val_dl, device)

    rand = PatternEncoder(ck["config"]).to(device)
    report["random_init"] = eval_model(rand, val_dl, device)

    report["baseline_rule"] = {"A1_tolF1": rule_baseline(val_dl)}
    report["baseline_trivial"] = {"A1_tolF1": trivial_baseline(val_dl)}

    if args.ckpt_bidir and Path(args.ckpt_bidir).exists():
        ckb = torch.load(args.ckpt_bidir, map_location=device, weights_only=False)
        mb = PatternEncoder(ckb["config"]).to(device)
        mb.load_state_dict(ckb["model"])
        report["bidir"] = eval_model(mb, val_dl, device)

    print("\n== Q1 A1 margin（因果版枢轴容差F1 vs 基线）==")
    print(f"  模型      : {report['causal']['A1_tolF1']:.3f}")
    print(f"  随机骨干  : {report['random_init']['A1_tolF1']:.3f}")
    print(f"  规则检测器: {report['baseline_rule']['A1_tolF1']:.3f}")
    print(f"  乱猜基线  : {report['baseline_trivial']['A1_tolF1']:.3f}")
    m = report['causal']['A1_tolF1']
    print(f"  margin vs 规则={m - report['baseline_rule']['A1_tolF1']:+.3f} "
          f"vs 随机={m - report['random_init']['A1_tolF1']:+.3f}")

    print("\n== Q2 双向倒挂对照 ==")
    if "bidir" in report:
        for k in ("A1_tolF1", "A2_segBA_conf", "U1_pivF1", "U2_segBA"):
            print(f"  {k:16s} 因果={report['causal'][k]:.3f}  双向={report['bidir'][k]:.3f}")
    else:
        print("  （未提供 --ckpt-bidir，跳过）")

    print("\n== Q3 坐标头解剖（预测std / 真实std，≈0 说明只学均值）==")
    c = report["causal"]
    print(f"  时间坐标: {c['time_pred_std']:.2f} / {c['time_true_std']:.2f}")
    print(f"  幅度坐标: {c['amp_pred_std']:.2f} / {c['amp_true_std']:.2f}")

    out = Path("reports/e12_diag.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float))
    print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()
