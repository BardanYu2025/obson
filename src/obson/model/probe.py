"""E7-A 表征探针门禁（协议 v3 §6.2）

冻结 encoder，用线性 probe 检验表示里还剩什么：
  - label probe：三分类标签（balanced acc / macro-F1 / CE）
  - path probe ：路径状态 4 节点（balanced acc）
  - symbol probe：品种身份（accuracy；高不是罪，配合其他指标看挤占）

三方对照（相同数据划分、相同 probe 预算、相同种子）：
  - random   ：随机初始化 encoder（下界/噪声参考）
  - pretrained：E7-A 预训练 encoder（待门禁）
  - supervised ：冠军 best.pt（上界参考，若存在）

判决（协议）：label probe 双 seed 同向下降 + bootstrap CI 不重叠才暂停；
5% 只是告警线。测试段不参与 probe（协议 §6.1）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _balanced_acc(y, pred, n_classes):
    recs = []
    for c in range(n_classes):
        m = y == c
        if m.sum() > 0:
            recs.append((pred[m] == c).mean())
    return float(np.mean(recs)) if recs else float("nan")


def _macro_f1(y, pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def collect_reprs(model, loaders: dict, device: str, max_per_loader: int = 4000):
    """抓 frozen encoder 表示 + 标签。返回 dict key -> (X, y_label, y_path, y_sym)"""
    store = {}
    model.norm.register_forward_hook(lambda m, i, o: store.__setitem__("r", o))
    model.eval()
    model.to(device)
    out = {}
    for key, ld in loaders.items():
        Xs, Ys, Ps, Ss = [], [], [], []
        for batch in ld:
            store.pop("r", None)  # 清掉上一 batch 的残留，防止抓到过期表示
            kw = {}
            for k in ("temporal_feat", "time_pos", "freq_feat", "symbol_id",
                      "daily_ctx", "foreign_ctx", "fine_ctx", "cross_ctx", "cross_mask"):
                v = batch.get(k)
                if v is not None:
                    kw[k] = v.to(device)
            model(kline_seq=batch["seq"].to(device), **kw)
            if "r" not in store:
                raise RuntimeError(f"[probe debug] {key}: forward 后 hook 未捕获表示")
            rep = store.pop("r")
            pooled = rep[:, 0, :] if getattr(model, "readout", "last") == "cls" else rep[:, -1, :]
            if pooled.shape[0] != len(batch["label"]):
                raise RuntimeError(f"[probe debug] {key}: 表示行数 {pooled.shape[0]} != "
                                   f"标签数 {len(batch['label'])}（hook 抓到了别的 batch）")
            Xs.append(pooled.cpu().numpy())
            Ys.append(batch["label"].numpy())
            ps = batch.get("path_states")
            Ps.append(ps.numpy() if ps is not None else
                      np.full((len(batch["label"]), 4), -1, dtype=np.int64))
            Ss.append(batch["symbol_id"].numpy())
            if sum(len(x) for x in Xs) >= max_per_loader:
                break
        X = np.concatenate(Xs)[:max_per_loader]
        out[key] = (X,
                    np.concatenate(Ys)[:max_per_loader],
                    np.concatenate(Ps)[:max_per_loader],
                    np.concatenate(Ss)[:max_per_loader])
    return out


def _fit_probe(Xtr, ytr, Xva, yva, seed=0, classes=(0, 1, 2)):
    """torch 线性 probe（softmax 回归，固定预算 500 epoch full-batch）。
    与协议一致：相同划分/预算/种子对照。
    v2 修复：特征按训练集统计标准化（不标准化时 512 维表示尺度混乱，
    线性 probe 不收敛，出现冠军=随机的仪器失灵）。"""
    mask_tr = np.isin(ytr, list(classes))
    mask_va = np.isin(yva, list(classes))
    if mask_tr.sum() < 50 or mask_va.sum() < 30:
        return None
    mu = Xtr[mask_tr].mean(axis=0)
    sd = Xtr[mask_tr].std(axis=0) + 1e-8
    Xtr_n = (Xtr[mask_tr] - mu) / sd
    Xva_n = (Xva[mask_va] - mu) / sd
    cls_list = sorted(set(ytr[mask_tr].tolist()))
    n_classes = len(cls_list)
    remap = {c: i for i, c in enumerate(cls_list)}
    ytr_r = np.vectorize(remap.get)(ytr[mask_tr])
    yva_r = np.vectorize(remap.get)(yva[mask_va])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xtr_n, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr_r, dtype=torch.long, device=device)
    Xv = torch.tensor(Xva_n, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    probe = nn.Linear(Xt.shape[1], n_classes).to(device)
    # 类别加权：90% 多数类会把不加权的 probe 压成全猜"无"（BA=0.33 地板）
    cnt = np.bincount(ytr_r, minlength=n_classes).astype(np.float32)
    w = torch.tensor(cnt.sum() / (n_classes * np.clip(cnt, 1, None)), device=device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    for _ in range(500):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(Xt), yt, weight=w)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        logits = probe(Xv)
        pred = logits.argmax(-1).cpu().numpy()
        ce = nn.functional.cross_entropy(logits, torch.tensor(yva_r, device=device)).item()
        # 训练集拟合度：连训练集都拟合不了 = 优化失败（仪器检查）
        tr_pred = probe(Xt).argmax(-1).cpu().numpy()
        tr_ba = _balanced_acc(ytr_r, tr_pred, n_classes)
    return dict(
        bal_acc=_balanced_acc(yva_r, pred, n_classes),
        macro_f1=_macro_f1(yva_r, pred, n_classes),
        ce=float(ce),
        train_ba=tr_ba,
        n_val=int(mask_va.sum()),
    )


def run_probes(reprs: dict, tag: str, seed: int = 0):
    """对一组表示跑三个 probe。reprs[key]=(X, y, path, sym)，
    key 命名约定 xxx_train / xxx_val 配对。"""
    results = {}
    train_keys = [k for k in reprs if k.endswith("_train")]
    for ktr in train_keys:
        kva = ktr.replace("_train", "_val")
        if kva not in reprs:
            continue
        Xtr, ytr, ptr, _ = reprs[ktr]
        Xva, yva, pva, sva = reprs[kva]
        _, _, _, str_ = reprs[ktr]
        r = {}
        r["label"] = _fit_probe(Xtr, ytr, Xva, yva, seed)
        # path probe：逐节点（只报 BA 均值），忽略 -1
        node_bas = []
        for k in range(4):
            pr = _fit_probe(Xtr, ptr[:, k], Xva, pva[:, k], seed, classes=(0, 1, 2))
            if pr:
                node_bas.append(pr["bal_acc"])
        r["path"] = {"bal_acc": float(np.mean(node_bas))} if node_bas else None
        r["symbol"] = _fit_probe(Xtr, str_, Xva, sva, seed,
                                 classes=tuple(np.unique(str_).tolist()))
        results[ktr.replace("_train", "")] = r
    # 汇总
    lab = [v["label"]["bal_acc"] for v in results.values() if v["label"]]
    f1 = [v["label"]["macro_f1"] for v in results.values() if v["label"]]
    ce = [v["label"]["ce"] for v in results.values() if v["label"]]
    trba = [v["label"]["train_ba"] for v in results.values() if v["label"]]
    path = [v["path"]["bal_acc"] for v in results.values() if v["path"]]
    sym = [v["symbol"]["bal_acc"] for v in results.values() if v["symbol"]]
    print(f"\n── probe [{tag}] (seed={seed}) ──")
    print(f"  label : BA={np.mean(lab):.4f}  macroF1={np.mean(f1):.4f}  CE={np.mean(ce):.4f}"
          f"  (训练集BA={np.mean(trba):.3f}，≈0.33=优化失败仪器警报)")
    if path:
        print(f"  path  : BA={np.mean(path):.4f}")
    if sym:
        print(f"  symbol: BA={np.mean(sym):.4f}（高≠有罪，需结合 label 看挤占）")
    return dict(label_ba=float(np.mean(lab)), label_f1=float(np.mean(f1)),
                label_ce=float(np.mean(ce)),
                path_ba=float(np.mean(path)) if path else None,
                symbol_ba=float(np.mean(sym)) if sym else None,
                per_combo=results)
