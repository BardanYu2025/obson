"""E12 形态骨干训练（M2）

用法：
  PYTHONPATH=src python -u scripts/train_pattern.py \
    --symbols rb sr p --periods 60 30 --window 256 \
    --epochs 30 --batch-size 64 --lr 3e-4 --seed 42 \
    --bidir            # 默认因果版；--bidir 训双向对照版
    --save-dir checkpoints/e12_causal_s42

门禁（毕业线，预先承诺，详见 docs/E12_PATTERN_BACKBONE_DESIGN.md §五）：
  G1 枢轴F1≥0.6 | G2 段方向BA≥0.7 | G3 坐标R²>0.5
  （G4 跨规则族/G5 不变性/G6 坍缩复查由 scripts/e12_gate.py 独立执行）
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

sys_path_added = False
import sys
sys.path.insert(0, "src")

from obson.pattern_data import N_SCALES, build_pattern_datasets
from obson.pattern_model import PatternConfig, PatternEncoder, gate_metrics, pattern_losses

FREQ_IDS = {1: 0, 5: 1, 15: 2, 30: 3, 60: 4, 120: 5, 240: 6}


def make_mask(bsz: int, T: int, ratio: float, device) -> torch.Tensor:
    return torch.rand(bsz, T, device=device) < ratio


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ms, losses = [], []
    for batch in loader:
        out = model(batch["x"].to(device), batch["symbol_id"].to(device),
                    batch["freq_id"].to(device))
        losses.append(sum(pattern_losses(out, batch, None).values()).item())
        ms.append(gate_metrics(out, batch))
    agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    agg["val_loss"] = float(np.mean(losses))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["rb", "sr", "p"])
    ap.add_argument("--periods", nargs="+", type=int, default=[60, 30])
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bidir", action="store_true", help="双向版（仅离线研究用）")
    ap.add_argument("--mask-ratio", type=float, default=0.15)
    ap.add_argument("--contract", action="store_true", help="合约段帧模式（全量历史，需先跑 build_bar_labels.py --contract）")
    ap.add_argument("--warmup-epochs", type=int, default=0, help="线性 warmup 轮数（双向版建议 2）")
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from obson.symbols import SYMBOLS
    trains, vals, tests = [], [], []
    for code in args.symbols:
        for period in args.periods:
            ok = Path(f"data/labels/{code}_{period}m_contract_labels.pkl").exists() if args.contract \
                else Path(f"data/{code}_{period}m.csv").exists()
            if not ok:
                print(f"  [{code}_{period}m] 缺{'合约标签' if args.contract else '数据'}，跳过")
                continue
            tr, va, te = build_pattern_datasets(
                code, period, args.window, args.stride,
                symbol_id=SYMBOLS.get(code, 0), freq_id=FREQ_IDS.get(period, 7),
                contract=args.contract)
            trains.append(tr); vals.append(va); tests.append(te)
    if not trains:
        raise SystemExit("无可用数据")

    cfg = PatternConfig(hidden_size=args.hidden, num_layers=args.layers,
                        num_heads=args.heads, causal=not args.bidir,
                        mask_ratio=args.mask_ratio)
    model = PatternEncoder(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    mode = "bidir" if args.bidir else "causal"
    print(f"E12 [{mode}] 参数量 {n_param:,} | window={args.window} | device={device}")

    train_dl = DataLoader(ConcatDataset(trains), batch_size=args.batch_size,
                          shuffle=True, drop_last=True)
    val_dl = DataLoader(ConcatDataset(vals), batch_size=args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.warmup_epochs > 0:
        w = args.warmup_epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda ep: (ep + 1) / w if ep < w else
            0.5 * (1 + math.cos(math.pi * (ep - w) / max(args.epochs - w, 1))))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    best, bad = math.inf, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, comp_sum, nb = time.time(), {}, 0
        for batch in train_dl:
            mask = make_mask(batch["x"].shape[0], args.window, args.mask_ratio, device)
            out = model(batch["x"].to(device), batch["symbol_id"].to(device),
                        batch["freq_id"].to(device), mask=mask)
            comp = pattern_losses(out, batch, mask)
            loss = sum(comp.values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in comp.items():
                comp_sum[k] = comp_sum.get(k, 0) + v.item()
            nb += 1
        sched.step()
        m = evaluate(model, val_dl, device)
        comp_str = " ".join(f"{k}={v/nb:.3f}" for k, v in comp_sum.items())
        print(f"Ep{ep:02d} | {comp_str} | val_loss={m['val_loss']:.4f} "
              f"U1={m['U1_pivF1']:.3f} U2={m['U2_segBA']:.3f} U3={m['U3_coordBA']:.3f} "
              f"A1(tolF1)={m['A1_pivF1_tol']:.3f} A2conf(BA)={m['A2_segBA_conf']:.3f} "
              f"A2conf(coordBA)={m['A2_coordBA_conf']:.3f} "
              f"| {time.time()-t0:.0f}s")
        if math.isfinite(m["val_loss"]) and m["val_loss"] < best - 1e-4:
            best, bad = m["val_loss"], 0
            torch.save({"config": cfg, "model": model.state_dict(),
                        "metrics": m, "epoch": ep, "mode": mode}, save / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early Stopping @ep{ep}")
                break

    # 测试段 + 门禁判决（v1.1 双轨：bidir 走 U 线，causal 走 A 线仅报告不判死刑）
    if not (save / "best.pt").exists():
        print("⚠️ 训练全程 val_loss 非有限，无 best.pt；请先看分量日志定位 NaN 源")
        return
    ck = torch.load(save / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test_dl = DataLoader(ConcatDataset(tests), batch_size=args.batch_size)
    tm = evaluate(model, test_dl, device)
    if args.bidir:
        gate = {"U1": tm["U1_pivF1"] >= 0.6, "U2": tm["U2_segBA"] >= 0.7, "U3": tm["U3_coordBA"] >= 0.4}
        print(f"\n== 理解轨门禁（双向版）== U1(F1)={tm['U1_pivF1']:.3f}({'✅' if gate['U1'] else '❌'}) "
              f"U2(BA)={tm['U2_segBA']:.3f}({'✅' if gate['U2'] else '❌'}) "
              f"U3(坐标桶BA)={tm['U3_coordBA']:.3f}({'✅' if gate['U3'] else '❌'}，7桶乱猜≈0.14)")
        print(f"判决: {'✅ U1-U3 全过，可跑 U4-U6' if all(gate.values()) else '❌ 未过，按分量定位死因'}")
    else:
        print(f"\n== 前瞻轨指标（因果版，只报告；margin 待预注册）== "
              f"A1(±3bar tolF1)={tm['A1_pivF1_tol']:.3f} "
              f"A2确认段(BA)={tm['A2_segBA_conf']:.3f} A2确认段(坐标桶BA)={tm['A2_coordBA_conf']:.3f}")
        print("对照指标（理解轨口径，因果版预期低，不作判决）: "
              f"U1={tm['U1_pivF1']:.3f} U2={tm['U2_segBA']:.3f}")
        gate = {}
    (save / "gate_result.json").write_text(json.dumps(
        {"test": tm, "gate": gate, "mode": mode}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
