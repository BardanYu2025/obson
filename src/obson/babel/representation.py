"""Causal per-bar pretraining, isolated from the Babel v1 rule checkpoints.

Future targets are supervision only. All horizons must stay inside the split.
Quantiles describe marginal feature distributions, not a joint OHLC generator.
"""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import FEATURES, load_series, manifest, split_boundaries, split_mask
from .model import Config, Encoder, targets, task_loss
from .metrics import balanced_accuracy
from .progress import progress

SCHEMA = "babel-bar-representation-v1"
HORIZONS = (1, 4, 16)
QUANTILES = (0.1, 0.5, 0.9)


def time_mask(s, bounds, split):
    days = s.sessions
    if split == "train":
        return (days <= np.datetime64(bounds["train_until"])) & ~np.isnat(days)
    lo = bounds["train_until"] if split == "val" else bounds["val_until"]
    hi = bounds["val_until"] if split == "val" else bounds["test_until"]
    return (days > np.datetime64(lo)) & (days <= np.datetime64(hi)) & ~np.isnat(days)


class BarWindows(Dataset):
    def __init__(self, series, bounds, split, cfg, stride=16):
        self.series, self.cfg = series, cfg
        self.labels = [targets(s) for s in series]
        self.valid = []
        self.items = []
        for i, s in enumerate(series):
            valid = split_mask(s, bounds, split).copy()
            same = time_mask(s, bounds, split)
            for h in HORIZONS:
                future = np.zeros(len(s.frame), bool)
                future[:-h] = same[h:]
                valid &= future
            self.valid.append(valid)
            self.items.extend((i, j) for j in range(cfg.window - 1, len(s.frame), stride) if valid[j])
        if not self.items:
            raise ValueError(f"No eligible {split} windows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, end = self.items[index]
        s = self.series[i]
        rows = np.arange(end - self.cfg.window + 1, end + 1)
        valid = self.valid[i][rows].copy()
        valid[:self.cfg.warmup] = False
        # Out-of-range target slots are masked, never wrapped into another contract.
        future_rows = np.minimum(rows[:, None] + np.array(HORIZONS), len(s.x) - 1)
        future = s.x[future_rows, :7]
        available = np.ones_like(future, dtype=bool)
        available[:, :, 5:7] = s.x[future_rows, 7, None] > 0
        result = {"x": s.x[rows], "future": future, "available": available,
                  "valid": valid, "series": i, "row": end}
        result.update({k: v[rows].astype(np.int64) for k, v in self.labels[i].items()})
        return result


class BarEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Encoder(cfg)
        self.forecast = nn.Linear(cfg.hidden, len(HORIZONS) * 7 * 3)
        self.reconstruct = nn.Linear(cfg.hidden, 7)

    def forward(self, x):
        out = self.backbone(x)
        h = out["h"]
        q = self.forecast(h).reshape(*h.shape[:2], len(HORIZONS), 7, 3)
        # Ordered quantiles, with a free median and positive distances.
        median = q[..., 1]
        out["quantiles"] = torch.stack((median - nn.functional.softplus(q[..., 0]),
                                        median, median + nn.functional.softplus(q[..., 2])), -1)
        out["reconstruction"] = self.reconstruct(h)
        return out


def pinball(pred, target):
    err = target.unsqueeze(-1) - pred
    q = pred.new_tensor(QUANTILES)
    return torch.maximum(q * err, (q - 1) * err).mean(-1)


def losses(out, batch, device):
    valid = batch["valid"].to(device)
    available = batch["available"].to(device) & valid[..., None, None]
    forecast = pinball(out["quantiles"], batch["future"].to(device))[available].mean()
    x = batch["x"].to(device)
    keep = valid[..., None].expand(*valid.shape, 7).clone()
    keep[..., 5:7] &= x[..., 7, None] > 0
    reconstruction = nn.functional.smooth_l1_loss(out["reconstruction"], x[..., :7], reduction="none")[keep].mean()
    auxiliary = torch.stack(list(task_loss(out, batch, device).values())).mean()
    return forecast, reconstruction, auxiliary


@torch.no_grad()
def forecast_eval(model, ds, device, batch_size, baseline=None):
    model.eval()
    total = np.zeros((len(HORIZONS), 7))
    base_total, counts = total.copy(), total.copy()
    loader = DataLoader(ds, batch_size=batch_size)
    for step, batch in enumerate(loader, 1):
        out = model(batch["x"].to(device))
        target = batch["future"][:, -1].to(device)
        mask = batch["available"][:, -1].numpy()
        value = pinball(out["quantiles"][:, -1], target).cpu().numpy()
        total += (value * mask).sum(0)
        counts += mask.sum(0)
        if baseline is not None:
            value = pinball(torch.tensor(baseline, device=device, dtype=target.dtype), target).cpu().numpy()
            base_total += (value * mask).sum(0)
        if step == 1 or step % 100 == 0 or step == len(loader):
            progress(f"forecast evaluation: batch={step}/{len(loader)}")
    supported = counts > 0
    mean = np.divide(total, counts, out=np.zeros_like(total), where=supported)
    result = {"pinball": float(mean[supported].mean()), "by_horizon_feature": mean.tolist(),
              "support": counts.astype(int).tolist(), "samples": len(ds)}
    if baseline is not None:
        b = np.divide(base_total, counts, out=np.zeros_like(total), where=supported)
        result["train_constant_quantiles_pinball"] = float(b[supported].mean())
    return result


def constant_quantiles(ds):
    values = []
    for i, row in ds.items:
        s = ds.series[i]
        v = s.x[row + np.array(HORIZONS), :7].copy()
        v[:, 5:7] = np.where(s.x[row + np.array(HORIZONS), 7, None] > 0, v[:, 5:7], np.nan)
        values.append(v)
    a = np.stack(values)
    result = np.zeros((len(HORIZONS), 7, 3), np.float32)
    for h in range(len(HORIZONS)):
        for f in range(7):
            v = a[:, h, f]
            v = v[np.isfinite(v)]
            if len(v):
                result[h, f] = np.quantile(v, QUANTILES)
    return result


def train_representation(series, out, cfg, epochs, batch_size, device, bounds):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Run directory must be empty; preserve earlier experiments")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train = BarWindows(series, bounds, "train", cfg)
    val = BarWindows(series, bounds, "val", cfg)
    model = BarEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    baseline = constant_quantiles(train)
    metadata = {"schema": SCHEMA, "config": asdict(cfg), "features": list(FEATURES),
                "horizons": HORIZONS, "quantiles": QUANTILES, "seed": 42,
                "epochs": epochs, "batch_size": batch_size, "stride": 16,
                "loss_weights": {"forecast": 1., "current_reconstruction": .2, "structure": .1},
                "manifest": manifest(series, bounds), "baseline": baseline.tolist(),
                "selection": "validation endpoint mean marginal quantile loss",
                "parameters": sum(p.numel() for p in model.parameters())}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    progress(f"bar pretrain: parameters={metadata['parameters']:,}, train={len(train):,}, val={len(val):,}; device={device}")
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running = np.zeros(3)
        loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        for step, batch in enumerate(loader, 1):
            length = random.choice(cfg.trained_windows)
            for k in ("x", "future", "available", "valid", *train.labels[0]):
                batch[k] = batch[k][:, -length:]
            batch["valid"][:, :cfg.warmup] = False
            out_batch = model(batch["x"].to(device))
            terms = losses(out_batch, batch, device)
            loss = terms[0] + .2 * terms[1] + .1 * terms[2]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running += [v.item() for v in terms]
            if step == 1 or step % 25 == 0 or step == len(loader):
                progress(f"epoch={epoch}/{epochs} batch={step}/{len(loader)} losses={(running / step).round(5).tolist()}")
        metrics = forecast_eval(model, val, device, batch_size, baseline)
        record = {"epoch": epoch, "train_losses": (running / step).tolist(), "validation": metrics}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        progress(f"epoch={epoch} validation_pinball={metrics['pinball']:.6f} constant={metrics['train_constant_quantiles_pinball']:.6f}")
        if metrics["pinball"] < best:
            best = metrics["pinball"]
            torch.save({**metadata, "model": model.state_dict(), "epoch": epoch}, out / "best.pt")


@torch.no_grad()
def probe_features(model, ds, device, batch_size):
    learned, raw, labels = [], [], []
    loader = DataLoader(ds, batch_size=batch_size)
    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device)
        learned.append(model(x)["h"][:, -1].cpu().numpy())
        raw.append(torch.cat((x[:, -1], x[:, -32:].mean(1)), -1).cpu().numpy())
        for i, row in zip(batch["series"].tolist(), batch["row"].tolist(), strict=True):
            s = ds.series[i]
            # An 8-bar accumulated-return classification task, absent from the heads.
            change = (s.frame.close.iloc[row + 8] - s.frame.close.iloc[row]) / s.labels["atr"][row]
            labels.append(int(np.digitize(change, [-.5, .5])))
        if step == 1 or step % 100 == 0 or step == len(loader):
            progress(f"probe extraction: batch={step}/{len(loader)}")
    return np.concatenate(learned), np.concatenate(raw), np.array(labels)


def ridge_probe(train_x, train_y, test_x, test_y):
    mean, std = train_x.mean(0), train_x.std(0).clip(.01)
    a = np.column_stack(((train_x - mean) / std, np.ones(len(train_x))))
    b = np.column_stack(((test_x - mean) / std, np.ones(len(test_x))))
    penalty = np.eye(a.shape[1]) * 10
    penalty[-1, -1] = 0
    weights = np.linalg.solve(a.T @ a + penalty, a.T @ np.eye(3)[train_y])
    return balanced_accuracy((b @ weights).argmax(1), test_y, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "evaluate"])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", required=True, help="Existing Babel manifest.json; freezes data and time cuts")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for this runner; no local CPU training fallback")
    ref = json.loads(Path(args.reference).read_text())["manifest"]
    keys = [s["key"].split("/") for s in ref["sources"]]
    series, _ = load_series(args.root, sorted({k[0] for k in keys}), sorted({int(k[1]) for k in keys}))
    if manifest(series)["sources"] != ref["sources"]:
        raise ValueError("Source data changed since reference run")
    bounds = ref.get("boundaries") or split_boundaries(series)
    cfg = Config(hidden=128, layers=4, heads=4, window=256, warmup=64)
    if args.stage == "train":
        if args.epochs < 1 or args.batch_size < 1:
            raise ValueError("Epochs and batch size must be positive")
        train_representation(series, args.out, cfg, args.epochs, args.batch_size, "cuda", bounds)
        return
    ck = torch.load(Path(args.out) / "best.pt", map_location="cpu", weights_only=True)
    if ck["schema"] != SCHEMA or ck["manifest"] != manifest(series, bounds):
        raise ValueError("Representation checkpoint/data mismatch")
    cfg = Config(**ck["config"])
    model = BarEncoder(cfg).to("cuda")
    model.load_state_dict(ck["model"])
    model.eval()
    tr = BarWindows(series, bounds, "train", cfg)
    te = BarWindows(series, bounds, "test", cfg)
    report = {"schema": SCHEMA, "selected_epoch": ck["epoch"],
              "forecast": forecast_eval(model, te, "cuda", args.batch_size, ck["baseline"])}
    progress("extracting frozen embeddings for fixed 8-bar return probe")
    a, raw_a, y = probe_features(model, tr, "cuda", args.batch_size)
    b, raw_b, z = probe_features(model, te, "cuda", args.batch_size)
    report["frozen_pretrained_probe"] = ridge_probe(a, y, b, z)
    report["raw_feature_probe"] = ridge_probe(raw_a, y, raw_b, z)
    torch.manual_seed(42)
    random_model = BarEncoder(cfg).to("cuda").eval()
    a, _, _ = probe_features(random_model, tr, "cuda", args.batch_size)
    b, _, _ = probe_features(random_model, te, "cuda", args.batch_size)
    report["frozen_random_probe"] = ridge_probe(a, y, b, z)
    report["interpretation"] = "Single-seed pilot, stride-16 correlated endpoints. Fixed ridge=10; no test tuning. Forecast is marginal quantile loss on clipped causal features, not joint generation. Probe is a related future-return task, not proof of universal embeddings or trading value. Previously inspected test period is not a pristine holdout."
    (Path(args.out) / "representation_metrics.json").write_text(json.dumps(report, indent=2))
    progress(json.dumps(report))


if __name__ == "__main__":
    main()
