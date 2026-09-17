"""E11-A 机制门禁评估（设计稿 docs/E11_PATH_QUANTILE_DESIGN.md 第一层）

纯前向、不改权重，回答：分位数头（头①）学到的幅度分布可信吗？

六项门禁（判决线预先承诺）：
  G1 覆盖率校准：每节点×方向×τ，经验覆盖率 P(target ≤ q_τ) 与名义 τ 的
     平均绝对偏差 ≤ 0.10
  G2 分位数交叉率 < 1%（同节点同方向 q_τ 应随 τ 单调）
  G3 时间排序违规率 < 1%（同方向 50% 节点的 q50 ≤ 100% 节点的 q50）
  G4 pinball 损失 ≤ 经验分位数基线（train 段每组合×节点×方向×τ 的经验
     分位数当常数预测）—— 模型要比"无条件分布"更强
  G5 多空分离：按三分类标签分组，多头组 q50(m_up) 中位数 > 无方向组，
     空头组 q50(m_dn) 中位数 > 无方向组（方向正确且组样本 ≥ 50）
  G6 分布未收缩：预测 q90−q10 的均值 ≥ 目标 q90−q10 均值的 50%

用法（与训练同口径：合约模式 + 日K/外盘上下文）：
  PYTHONPATH=src python -u scripts/e11_gate.py \
      --ckpt checkpoints/e11a_s42/best.pt [--symbols rb sr p] [--periods 60 30]

结果写 reports/e11_gate.json 并打印判决。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from obson.contract_series import build_contract_frame
from obson.model.dataset import build_datasets_contract, weekly_seq_len
from obson.model.transformer import KLineTransformer
from train_multi_symbol import _theta_c_dynamic_ms, load_foreign

TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
DEVICE = "cpu"


@torch.no_grad()
def collect(model, ds, batch_size: int = 512):
    """对 ds 全量前向（与训练评估同接线），返回 quant_preds/quant_targets/labels。"""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, tgts, labels = [], [], []
    for batch in loader:
        out = model(
            batch["seq"].to(DEVICE),
            temporal_feat=batch["temporal_feat"].to(DEVICE),
            time_pos=batch["time_pos"].to(DEVICE),
            freq_feat=batch["freq_feat"].to(DEVICE),
            symbol_id=batch["symbol_id"].to(DEVICE),
            daily_ctx=batch.get("daily_ctx"),
            foreign_ctx=batch.get("foreign_ctx"),
        )
        if "quant_preds" not in out:
            raise RuntimeError("该 checkpoint 没有 quant 头（训练时未开 --quant-aux）")
        preds.append(out["quant_preds"].cpu().numpy())
        tgts.append(batch["quant_targets"].numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(preds), np.concatenate(tgts), np.concatenate(labels)


def pinball(pred: np.ndarray, tgt: np.ndarray, tau: float) -> float:
    d = tgt - pred
    return float(np.maximum(tau * d, (tau - 1) * d).mean())


def gate_combo(model, train_ds, val_ds, tag: str) -> dict:
    P, T, Y = collect(model, val_ds)
    _, T_tr, _ = collect(model, train_ds)
    r: dict = {"n_val": int(len(Y))}

    # ── G1 覆盖率校准 + G4 pinball vs 基线（逐节点×方向×τ）────────────
    cov_rows, cov_dev = [], []
    pin_model, pin_base = [], []
    for k in range(2):
        for d in range(2):
            for ti, tau in enumerate(TAUS):
                p, t = P[:, k, d, ti], T[:, k, d]
                cov = float((t <= p).mean())
                cov_rows.append({"node": k, "dir": d, "tau": tau,
                                 "coverage": round(cov, 4), "nominal": tau})
                cov_dev.append(abs(cov - tau))
                base_q = float(np.quantile(T_tr[:, k, d], tau))
                pin_model.append(pinball(p, t, tau))
                pin_base.append(pinball(np.full_like(p, base_q), t, tau))
    r["G1_coverage"] = cov_rows
    r["G1_mean_abs_dev"] = round(float(np.mean(cov_dev)), 4)
    r["G4_pinball_model"] = round(float(np.mean(pin_model)), 5)
    r["G4_pinball_baseline"] = round(float(np.mean(pin_base)), 5)

    # ── G2 交叉率 / G3 时间排序违规率 ─────────────────────────────────
    r["G2_cross_rate"] = round(float((np.diff(P, axis=3) < 0).mean()), 5)
    r["G3_time_order_violation"] = round(float((P[:, 0, :, 2] > P[:, 1, :, 2]).mean()), 5)

    # ── G5 多空分离（100% 节点 q50，按三分类标签分组）──────────────────
    sep, ok = {}, True
    for d, sig_c, name in ((1, 2, "up"), (0, 0, "dn")):
        sig = P[:, 1, d, 2]
        med_sig = float(np.median(sig[Y == sig_c]))
        med_non = float(np.median(sig[Y == 1]))
        sep[name] = {"n_signal": int((Y == sig_c).sum()),
                     "median_q50_signal": round(med_sig, 4),
                     "median_q50_none": round(med_non, 4),
                     "diff": round(med_sig - med_non, 4)}
        if (Y == sig_c).sum() < 50 or med_sig <= med_non:
            ok = False
    r["G5_separation"] = sep

    # ── G6 分布未收缩：预测 IQR vs 目标 IQR ───────────────────────────
    w_pred = float((P[:, :, :, 4] - P[:, :, :, 0]).mean())
    w_tgt = float((np.quantile(T, 0.9, axis=0) - np.quantile(T, 0.1, axis=0)).mean())
    r["G6_pred_iqr"] = round(w_pred, 4)
    r["G6_tgt_iqr"] = round(w_tgt, 4)
    r["G6_ratio"] = round(w_pred / max(w_tgt, 1e-9), 4)

    verdict = {
        "G1_校准(≤0.10)": r["G1_mean_abs_dev"] <= 0.10,
        "G2_交叉(<1%)": r["G2_cross_rate"] < 0.01,
        "G3_时序(<1%)": r["G3_time_order_violation"] < 0.01,
        "G4_优于基线": r["G4_pinball_model"] <= r["G4_pinball_baseline"],
        "G5_多空分离": ok,
        "G6_未收缩(≥50%)": r["G6_ratio"] >= 0.5,
    }
    r["verdict"] = verdict
    r["pass"] = all(verdict.values())

    print(f"\n== {tag} ==")
    print(f"  G1 覆盖率平均偏差 {r['G1_mean_abs_dev']:.4f} (≤0.10)")
    print(f"  G2 交叉率 {r['G2_cross_rate']:.4%} (<1%) | G3 时序违规 {r['G3_time_order_violation']:.4%} (<1%)")
    print(f"  G4 pinball 模型 {r['G4_pinball_model']:.5f} vs 基线 {r['G4_pinball_baseline']:.5f}")
    for name, s in sep.items():
        print(f"  G5 {name}: 信号组 q50 中位 {s['median_q50_signal']:.3f} vs 无方向 {s['median_q50_none']:.3f}"
              f" (Δ={s['diff']:+.3f}, n={s['n_signal']})")
    print(f"  G6 IQR 比 {r['G6_ratio']:.2%} (≥50%)")
    print(f"  判决: {'✅ 过' if r['pass'] else '❌ 未过'} {verdict}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--symbols", nargs="+", default=["rb", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--theta-q", type=float, default=0.90)
    ap.add_argument("--daily-bars", type=int, default=20)
    ap.add_argument("--foreign-bars", type=int, default=20)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = KLineTransformer(ck["config"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    # 品种 → symbol_id：从 ckpt config 的品种数按训练顺序映射（与训练脚本一致）
    from train_multi_symbol import SYMBOLS  # 训练脚本的固定品种→id映射
    symbols = {c: SYMBOLS[c] for c in args.symbols if c in SYMBOLS}

    report = {"ckpt": args.ckpt, "combos": {}}
    n_pass = n_tot = 0
    for code, sym_id in symbols.items():
        for period in args.periods:
            tag = f"{code}_{period}m"
            try:
                cframe, cbreaks, cmask, cday_ids = build_contract_frame(code, period)
            except (FileNotFoundError, ValueError) as e:
                print(f"  [{tag}] 合约段帧构建失败: {e}，跳过")
                continue
            last_seg = cframe[cframe["seg"] == cframe["seg"].max()].reset_index(drop=True)
            seq_len = weekly_seq_len(last_seg, trading_days=5)
            main_days = np.unique(cday_ids[cmask])
            n_tr_days = int(len(main_days) * 0.7)
            tr_rows = cmask & (cday_ids <= main_days[n_tr_days - 1])
            ms_all = []
            for _, gdf in cframe[tr_rows].groupby("seg"):
                gdf = gdf.reset_index(drop=True)
                if len(gdf) < seq_len + 20:
                    continue
                ms_all.extend(_theta_c_dynamic_ms(gdf, seq_len))
            if not ms_all:
                print(f"  [{tag}] θ 校准无样本，跳过")
                continue
            theta = float(np.quantile(ms_all, args.theta_q))
            try:
                train_ds, val_ds, _ = build_datasets_contract(
                    code, period, seq_len=seq_len,
                    train_ratio=0.7, val_ratio=0.15,
                    target_offset=1, symbol_id=sym_id,
                    daily_bars=args.daily_bars,
                    foreign_close=load_foreign(code) if args.foreign_bars > 0 else None,
                    foreign_bars=args.foreign_bars,
                    label_mode="day_close", label_threshold=theta,
                    theta_mode="dynamic", soft_label=True,
                    base_period_min=period,
                )
            except ValueError as e:
                print(f"  [{tag}] 构造失败: {e}，跳过")
                continue
            report["combos"][tag] = gate_combo(model, train_ds, val_ds, tag)
            n_tot += 1
            n_pass += int(report["combos"][tag]["pass"])

    report["summary"] = {"pass": n_pass, "total": n_tot,
                         "gate_pass": n_tot > 0 and n_pass == n_tot}
    print(f"\n{'=' * 50}\n总判决: {n_pass}/{n_tot} 组合过机制门禁 → "
          f"{'✅ 可进信号门禁' if report['summary']['gate_pass'] else '❌ 机制未过，不进第二层'}")

    out = Path("reports/e11_gate.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
