"""Read-only stage-1 reconstruction diagnostics; no updates or test-set tuning."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .progress import progress


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(truth, pred):
    """Inputs are existing AE coordinates; price errors expressed in log-price bp."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    result = {"close_mae_bps": float(np.abs(pred[:, 1] - truth[:, 1]).mean() * 100)}
    for h in (1, 4, 16):
        if len(truth) <= h:
            continue
        a, b = truth[h:, 1] - truth[:-h, 1], pred[h:, 1] - pred[:-h, 1]
        result[f"change_{h}_mae_bps"] = float(np.abs(a - b).mean() * 100)
        result[f"change_{h}_truth_std_bps"] = float(a.std() * 100)
        result[f"change_{h}_std_ratio"] = float(b.std() / a.std()) if a.std() > 1e-8 else None
        result[f"change_{h}_correlation"] = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-8 else None
    return result


def gap_mask(truth):
    # True log(high/low)*100 = |open-close| + upper + lower.
    typical_range = np.median(np.abs(truth[:, 0] - truth[:, 1]) + truth[:, 2] + truth[:, 3])
    gap = truth[:, 0] - np.r_[0., truth[:-1, 1]]
    return np.abs(gap) > max(3 * typical_range, 1e-6)


def group_metrics(truth, pred, state):
    truth, pred = np.asarray(truth), np.asarray(pred)
    groups = {"all": metrics(truth, pred), f"endpoint_state/{state}": metrics(truth, pred)}
    for k, indices in enumerate(np.array_split(np.arange(len(truth)), 4), 1):
        groups[f"position/Q{k}"] = metrics(truth[indices], pred[indices])
    flagged = gap_mask(truth)
    groups["window/with_large_gap" if flagged.any() else "window/without_large_gap"] = metrics(truth, pred)
    true_gaps = truth[:, 0] - np.r_[0., truth[:-1, 1]]
    pred_gaps = pred[:, 0] - np.r_[0., pred[:-1, 1]]
    for name, mask in (("large_gap", flagged), ("other", ~flagged)):
        if mask.any():
            groups[f"bar/{name}"] = {"close_mae_bps": float(np.abs(pred[mask, 1] - truth[mask, 1]).mean() * 100),
                                      "gap_mae_bps": float(np.abs(pred_gaps[mask] - true_gaps[mask]).mean() * 100)}
    return groups


def summarize(records):
    result = {}
    for group, rows in records.items():
        keys = sorted({k for row in rows for k in row})
        result[group] = {"windows": len(rows), "metrics": {}}
        for key in keys:
            values = [r[key] for r in rows if r.get(key) is not None]
            result[group]["metrics"][key] = {"mean": float(np.mean(values)) if values else None,
                                             "median": float(np.median(values)) if values else None,
                                             "supported_windows": len(values)}
    return result


@torch.no_grad()
def diagnose(model, datasets, device, batch_size, seed, directory):
    from .history_autoencoder import fit_pca, pca_reconstruct, write

    directory = Path(directory)
    tracked = ["best.pt", "manifest.json", "history.jsonl", "ae_metrics.json"]
    before = {name: fingerprint(directory / name) for name in tracked if (directory / name).exists()}
    tr, _, te = datasets
    model.eval()
    progress("Stage 1: fitting the unchanged training-only PCA baseline")
    pca = fit_pca(tr, model.config["latent"], seed)
    records = {name: {} for name in ("autoencoder", "pca")}
    inventory = []
    labels = ("unformed", "up", "down", "range")
    loader = DataLoader(te, batch_size=batch_size)
    for step, b in enumerate(loader, 1):
        y = b["y"].numpy()
        reconstructed = model(b["x"].to(device))["reconstruction"].cpu().numpy()
        alternatives = {"autoencoder": reconstructed, "pca": pca_reconstruct(y, pca)}
        for j, (i, row) in enumerate(zip(b["series"].tolist(), b["row"].tolist(), strict=True)):
            s = te.series[i]
            state = labels[int(s.labels["state"][row, 1])]
            inventory.append({"source": s.key, "row": row, "endpoint_state": state,
                              "large_gap_bars": int(gap_mask(y[j]).sum())})
            for method, predictions in alternatives.items():
                groups = group_metrics(y[j], predictions[j], state)
                groups[f"period/{s.period}"] = metrics(y[j], predictions[j])
                for group, values in groups.items():
                    records[method].setdefault(group, []).append(values)
        if step == 1 or step % 50 == 0 or step == len(loader):
            progress(f"Stage 1 diagnostic inference {step}/{len(loader)}")
    after = {name: fingerprint(directory / name) for name in before}
    if before != after:
        raise ValueError("Frozen experiment files changed during diagnosis")
    report = {
        "schema": "babel-ae-diagnostics-v1", "frozen_artifacts_sha256": before,
        "config": model.config, "test_windows": len(te), "pca_training_windows": pca["samples"],
        "definitions": {
            "positions": "Q1 oldest to Q4 newest; four equal parts; differences computed within each part",
            "state": "Medium-scale rule state at window endpoint; does not mean every bar shares that state",
            "large_gap": "abs(log open/previous close) > 3 * window median log(high/low); descriptive only, not a data-error verdict",
            "aggregation": "Equal window means, including bar subgroups; overlapping windows are not independent. Correlations/std ratios exclude constant paths; supports reported.",
            "scope": "Reconstruction of already-observed history only. No optimizer, no checkpoint changes, no threshold tuning or new model training.",
        },
        "methods": {name: summarize(groups) for name, groups in records.items()},
        "window_inventory": inventory,
    }
    write(directory / "ae_diagnostics.json", report)
    lines = ["# 阶段 1：冻结模型重建诊断", "", f"测试窗口：{len(te)}。权重、原配置和原指标的指纹未变化。", "",
             "以下是窗口等权均值；重叠窗口不独立，差异不是显著性结论。Q1 最早，Q4 最近。", "",
             "| 分组 | 方法 | 窗口数 | 收盘误差 bp | 1 根变化误差 bp | 1 根波动保留比 |", "|---|---|---:|---:|---:|---:|"]
    for group in sorted(report["methods"]["autoencoder"]):
        for method in records:
            row = report["methods"][method][group]
            values = []
            for key in ("close_mae_bps", "change_1_mae_bps", "change_1_std_ratio"):
                value = row["metrics"].get(key, {}).get("mean")
                values.append(f"{value:.4f}" if value is not None else "—")
            lines.append(f"| {group} | {method} | {row['windows']} | {' | '.join(values)} |")
    (directory / "ae_diagnostics.md").write_text("\n".join(lines) + "\n")
    progress(f"Stage 1 reports saved in {directory}; no training performed")
    return report
