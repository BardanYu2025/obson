"""A single causal bottleneck reconstructs a bounded history, never the future."""

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .ar_codec import FEATURES, encode_frame
from .autoregressive import probe
from .data import load_series, manifest, split_mask
from .progress import progress
from .representation import time_mask

SCHEMA = "babel-history-ae-v1"
WINDOW = 128
LATENT = 128
CHANNELS = ("open_log_percent", "close_log_percent", "upper_log_percent", "lower_log_percent",
            "log_volume", "log_oi", "log_interval_ratio")
WEIGHTS = (1., 1., .5, .5, .1, .1, .05)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def raw_targets(frame, period):
    """Per-record arrays; the preceding-window price anchor is applied in Dataset."""
    o, h, l, c = np.log(frame[["open", "high", "low", "close"]].to_numpy()).T
    elapsed = frame.datetime.diff().dt.total_seconds().fillna(period * 60).to_numpy() / (period * 60)
    return np.column_stack((o * 100, c * 100, (h - np.maximum(o, c)) * 100,
                            (np.minimum(o, c) - l) * 100, np.log1p(frame.volume) / 10,
                            np.log1p(frame.oi) / 10, np.log(elapsed)))


class HistoryWindows(Dataset):
    def __init__(self, series, encoded, bounds, split, stride=16, window=WINDOW):
        self.series, self.encoded, self.window = series, encoded, window
        self.targets = [raw_targets(s.frame, s.period) for s in series]
        self.items = []
        for i, s in enumerate(series):
            same = time_mask(s, bounds, split)
            bad = np.r_[0, np.cumsum(~same)]
            main = split_mask(s, bounds, split)
            self.items.extend((i, j) for j in range(window - 1, len(s.x), stride)
                              if main[j] and bad[j + 1] == bad[j + 1 - window])
        if not self.items:
            raise ValueError(f"No complete {split} history windows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, end = self.items[idx]
        lo = end - self.window + 1
        s, e = self.series[i], self.encoded[i]
        anchor = float(s.frame.close.iloc[lo - 1] if lo else s.frame.open.iloc[0])
        y = self.targets[i][lo:end + 1].copy()
        y[:, :2] -= np.log(anchor) * 100
        mask = np.ones_like(y, dtype=bool)
        mask[:, 5] = e["available"][lo:end + 1]
        y[:, 5] = np.where(mask[:, 5], y[:, 5], 0.)
        return {"x": e["x"][lo:end + 1], "y": y.astype(np.float32), "mask": mask,
                "anchor": anchor, "series": i, "row": end}


def positions(t, hidden, device, dtype):
    pos = torch.arange(t, device=device, dtype=dtype)[:, None]
    freq = torch.exp(torch.arange(0, hidden, 2, device=device, dtype=dtype) * (-math.log(10000) / hidden))
    pe = torch.zeros(t, hidden, device=device, dtype=dtype)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe


class HistoryAE(nn.Module):
    def __init__(self, hidden=128, layers=4, heads=4, latent=LATENT, window=WINDOW, dropout=.1):
        super().__init__()
        self.config = dict(hidden=hidden, layers=layers, heads=heads, latent=latent, window=window, dropout=dropout)
        self.input = nn.Linear(len(FEATURES), hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.compress = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, latent), nn.LayerNorm(latent))
        self.expand = nn.Linear(latent, hidden)
        decoder_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, 2, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 7))

    def encode(self, x):
        t = x.shape[1]
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        h = self.encoder(self.input(x) + positions(t, self.config["hidden"], x.device, x.dtype), mask=mask)
        return self.compress(h)

    def decode(self, z):
        """Only [batch, latent] accepted; no original bars, encoder sequence or anchor."""
        if z.ndim != 2 or z.shape[-1] != self.config["latent"]:
            raise ValueError("Decoder accepts one bottleneck vector per history")
        pe = positions(self.config["window"], self.config["hidden"], z.device, z.dtype)
        tokens = self.expand(z)[:, None] + pe
        raw = self.output(self.decoder(tokens))
        return torch.cat((raw[..., :2], nn.functional.softplus(raw[..., 2:])), -1)

    def forward(self, x):
        h = self.encode(x)
        z = h[:, -1]
        return {"h": h, "z": z, "reconstruction": self.decode(z)}


def reconstruction_loss(pred, truth, mask):
    weights = truth.new_tensor(WEIGHTS) * mask
    error = nn.functional.smooth_l1_loss(pred, truth, reduction="none")
    value = (error * weights).sum((1, 2)) / weights.sum((1, 2))
    # Preserve adjacent close-path changes as well as overall position.
    changes = nn.functional.smooth_l1_loss(pred[:, 1:, 1] - pred[:, :-1, 1],
                                         truth[:, 1:, 1] - truth[:, :-1, 1], reduction="none").mean(1)
    return value + .5 * changes


def to_ohlc(y, anchor):
    """Physical units require one explicit preceding-window close, not future data."""
    y = np.asarray(y, float)
    logs = np.column_stack((y[:, 0], np.maximum(y[:, 0], y[:, 1]) + y[:, 2],
                            np.minimum(y[:, 0], y[:, 1]) - y[:, 3], y[:, 1])) / 100 + np.log(anchor)
    if not np.isfinite(logs).all() or np.max(np.abs(logs)) > 600:
        raise ValueError("Reconstructed price overflow")
    return np.exp(logs)


@torch.no_grad()
def validate(model, ds, device, batch_size):
    model.eval()
    values = []
    for b in DataLoader(ds, batch_size=batch_size):
        pred = model(b["x"].to(device))["reconstruction"]
        values.extend(reconstruction_loss(pred, b["y"].to(device), b["mask"].to(device)).cpu().tolist())
    return float(np.mean(values))


def train(series, encoded, bounds, directory, epochs, batch_size, seed, device):
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Run directory must be empty; preserve previous experiments")
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va = (HistoryWindows(series, encoded, bounds, name) for name in ("train", "val"))
    model = HistoryAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    metadata = {"schema": SCHEMA, "config": model.config, "features": FEATURES, "channels": CHANNELS,
                "weights": WEIGHTS, "delta_loss_weight": .5, "manifest": manifest(series, bounds),
                "seed": seed, "epochs": epochs, "batch_size": batch_size, "stride": 16,
                "parameters": sum(p.numel() for p in model.parameters()),
                "train_windows": len(tr), "val_windows": len(va),
                "selection": "validation history reconstruction loss; no future prediction loss",
                "side_information": "one preceding-window price anchor for restoring absolute prices; not supplied to decoder"}
    write(directory / "manifest.json", metadata)
    progress(f"History AE: {metadata['parameters']:,} parameters; windows={len(tr):,}/{len(va):,}")
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
        for step, b in enumerate(loader, 1):
            pred = model(b["x"].to(device))["reconstruction"]
            loss = reconstruction_loss(pred, b["y"].to(device), b["mask"].to(device)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += loss.item()
            if step == 1 or step % 25 == 0 or step == len(loader):
                progress(f"epoch={epoch}/{epochs} batch={step}/{len(loader)} reconstruction={total / step:.6f}")
        score = validate(model, va, device, batch_size)
        with (directory / "history.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": total / step, "validation_loss": score}) + "\n")
        progress(f"epoch={epoch} validation_reconstruction={score:.6f}")
        if score < best:
            best = score
            torch.save({**metadata, "epoch": epoch, "model": model.state_dict()}, directory / "best.pt")


def fit_pca(ds, latent=LATENT, seed=42, limit=8192):
    """Matched floating-point bottleneck, trained only on a fixed training subsample."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), min(limit, len(ds)), replace=False)
    a = np.stack([ds[int(i)]["y"].reshape(-1) for i in indices]).astype(np.float64)
    mean, scale = a.mean(0), a.std(0).clip(.01)
    a = (a - mean) / scale
    eigenvalues, vectors = np.linalg.eigh(a.T @ a / max(1, len(a) - 1))
    rank = min(latent, len(a) - 1, a.shape[1])
    return {"mean": mean, "scale": scale, "vectors": vectors[:, -rank:],
            "samples": len(a), "rank": rank}


def pca_reconstruct(y, pca):
    shape = y.shape
    a = (y.reshape(len(y), -1) - pca["mean"]) / pca["scale"]
    result = ((a @ pca["vectors"]) @ pca["vectors"].T * pca["scale"] + pca["mean"]).reshape(shape)
    result[..., 2:] = np.maximum(result[..., 2:], 0)
    return result.astype(np.float32)


@torch.no_grad()
def evaluate(model, datasets, device, batch_size, seed):
    tr, va, te = datasets
    progress("Fitting training-only PCA compression baseline")
    pca = fit_pca(tr, model.config["latent"], seed)
    report = {"test_windows": len(te), "pca_training_windows": pca["samples"], "pca_rank": pca["rank"],
              "nominal_ratio": {"target_floats": te.window * 7, "latent_floats": model.config["latent"], "price_anchor_scalars": 1},
              "reconstruction": {}}
    sums = {name: {"loss": [], "mae_channels": [], "close_mae_bps": [], "change_mae_bps": []}
            for name in ("autoencoder", "pca", "train_mean", "zero_latent")}
    examples = []
    example_indices = set(np.linspace(0, len(te) - 1, min(6, len(te)), dtype=int).tolist())
    offset = 0
    loader = DataLoader(te, batch_size=batch_size)
    for step, b in enumerate(loader, 1):
        y = b["y"].numpy()
        out = model(b["x"].to(device))
        alternatives = {
            "autoencoder": out["reconstruction"].cpu(),
            "pca": torch.from_numpy(pca_reconstruct(y, pca)),
            "train_mean": torch.tensor(np.broadcast_to(pca["mean"].reshape(1, te.window, 7), y.shape).copy(), dtype=torch.float32),
            "zero_latent": model.decode(torch.zeros_like(out["z"])).cpu(),
        }
        for name, pred in alternatives.items():
            sums[name]["loss"].extend(reconstruction_loss(pred, b["y"], b["mask"]).tolist())
            abs_err = (pred - b["y"]).abs()
            mask = b["mask"]
            mae = (abs_err * mask).sum(1) / mask.sum(1).clamp_min(1)
            # Store mask counts separately so absent OI is never reported as perfect reconstruction.
            sums[name]["mae_channels"].extend(mae.tolist())
            sums[name]["close_mae_bps"].extend((abs_err[..., 1].mean(1) * 100).tolist())
            change = (pred[:, 1:, 1] - pred[:, :-1, 1] - b["y"][:, 1:, 1] + b["y"][:, :-1, 1]).abs().mean(1) * 100
            sums[name]["change_mae_bps"].extend(change.tolist())
        for j in range(len(y)):
            if offset + j in example_indices:
                i, end = int(b["series"][j]), int(b["row"][j])
                s = te.series[i]
                anchor = float(b["anchor"][j])
                examples.append({"source": s.key, "end": str(s.frame.datetime.iloc[end]), "anchor": anchor,
                    "truth_ohlc": to_ohlc(y[j], anchor).tolist(),
                    "reconstructed_ohlc": to_ohlc(alternatives["autoencoder"][j].numpy(), anchor).tolist(),
                    "pca_ohlc": to_ohlc(alternatives["pca"][j].numpy(), anchor).tolist(),
                    "truth_channels": y[j].tolist(), "reconstructed_channels": alternatives["autoencoder"][j].tolist()})
        offset += len(y)
        if step == 1 or step % 100 == 0 or step == len(loader):
            progress(f"History reconstruction evaluation {step}/{len(loader)}")
    oi_present = np.array([te.encoded[i]["available"][row] for i, row in te.items])
    for name, metrics in sums.items():
        result = {k: float(np.mean(v)) for k, v in metrics.items() if k != "mae_channels"}
        channels = np.array(metrics["mae_channels"])
        result["mae_channels"] = {k: float(channels[:, j].mean()) if j != 5 else
                                   (float(channels[oi_present, j].mean()) if oi_present.any() else None)
                                   for j, k in enumerate(CHANNELS)}
        report["reconstruction"][name] = result
    report["oi_supported_windows"] = int(oi_present.sum())
    return report, examples


def write_review(path, examples):
    """Standalone HTML/SVG; no server, JS, fonts or external network required."""
    sections = []
    for example in examples:
        arrays = [np.array(example[k]) for k in ("truth_ohlc", "reconstructed_ohlc", "pca_ohlc")]
        low = min(a[:, 2].min() for a in arrays)
        high = max(a[:, 1].max() for a in arrays)
        panels = []
        for label, bars in zip(("原始行情", "单向量重建", "同维 PCA 重建"), arrays, strict=True):
            def sy(price):
                return 190 - (price - low) / max(high - low, 1e-12) * 170
            shapes = []
            for j, (o, h, l, c) in enumerate(bars):
                x = 20 + j * 760 / len(bars)
                color = "#bd3e39" if c >= o else "#16806a"
                shapes.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{sy(h):.2f}" y2="{sy(l):.2f}" stroke="{color}"/>')
                shapes.append(f'<rect x="{x - 1.5:.2f}" y="{min(sy(o), sy(c)):.2f}" width="3" height="{max(abs(sy(o) - sy(c)), 1):.2f}" fill="{color}"/>')
            panels.append(f'<h3>{label}</h3><svg viewBox="0 0 800 210" role="img" aria-label="{label}">{"".join(shapes)}</svg>')
        sections.append(f'<section><h2>{html.escape(example["source"])} · 截至 {html.escape(example["end"])}</h2>{"".join(panels)}</section>')
    Path(path).write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>历史行情重建对照</title>'
                         '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#222;background:#f5f5f2}section{background:white;padding:20px;margin:24px 0}svg{width:100%;background:#fafafa}h3{margin-bottom:0}</style>'
                         '<h1>历史行情重建对照</h1><p>每组使用相同价格坐标范围。全部是已发生的历史重建，不是未来预测。样例按测试索引等距选取，不按效果挑选。</p>'
                         + ''.join(sections) + '</html>')


@torch.no_grad()
def frozen_features(model, ds, device, batch_size):
    latent, raw, labels = [], [], []
    loader = DataLoader(ds, batch_size=batch_size)
    for step, b in enumerate(loader, 1):
        x = b["x"].to(device)
        latent.append(model.encode(x)[:, -1].cpu().numpy())
        raw.append(x[:, -16:].flatten(1).cpu().numpy())
        for i, row in zip(b["series"].tolist(), b["row"].tolist(), strict=True):
            labels.append(int(ds.series[i].labels["state"][row, 1]))
        if step == 1 or step % 100 == 0 or step == len(loader):
            progress(f"Frozen current-structure probe {step}/{len(loader)}")
    return np.concatenate(latent), np.concatenate(raw), np.array(labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "evaluate", "diagnose"))
    p.add_argument("--root", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; no local training fallback")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Positive epochs/batch size required")
    directory = Path(args.out)
    ref = json.loads(Path(args.reference).read_text())["manifest"]
    keys = [s["key"].split("/") for s in ref["sources"]]
    series, _ = load_series(args.root, sorted({k[0] for k in keys}), sorted({int(k[1]) for k in keys}))
    if manifest(series)["sources"] != ref["sources"]:
        raise ValueError("Reference/source mismatch")
    encoded = [encode_frame(s.frame, s.period) for s in series]
    bounds = ref["boundaries"]
    if args.stage == "train":
        train(series, encoded, bounds, directory, args.epochs, args.batch_size, args.seed, "cuda")
        return
    ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    if ck["schema"] != SCHEMA or ck["manifest"] != manifest(series, bounds) or list(ck["features"]) != list(FEATURES):
        raise ValueError("History checkpoint/data mismatch")
    model = HistoryAE(**ck["config"]).to("cuda").eval()
    model.load_state_dict(ck["model"])
    datasets = [HistoryWindows(series, encoded, bounds, split, window=ck["config"]["window"]) for split in ("train", "val", "test")]
    if args.stage == "diagnose":
        from .ae_diagnostics import diagnose

        diagnose(model, datasets, "cuda", args.batch_size, ck["seed"], directory)
        return
    report, examples = evaluate(model, datasets, "cuda", args.batch_size, ck["seed"])
    report.update(schema=SCHEMA, selected_epoch=ck["epoch"])
    values = [frozen_features(model, ds, "cuda", args.batch_size) for ds in datasets]
    report["current_structure_probe"] = {}
    for name, column in (("pretrained", 0), ("raw_last16", 1)):
        report["current_structure_probe"][name] = probe(*(v for row in values for v in (row[column], row[2])), 4)
    torch.manual_seed(ck["seed"])
    random_model = HistoryAE(**ck["config"]).to("cuda").eval()
    values = [frozen_features(random_model, ds, "cuda", args.batch_size) for ds in datasets]
    report["current_structure_probe"]["random"] = probe(*(v for row in values for v in (row[0], row[2])), 4)
    report["interpretation"] = "Causal bounded-history lossy compression, not encryption or forecasting. Decoder only receives last z and fixed position queries. PCA has the same nominal latent width but is fitted to reconstruction coordinates. Test windows overlap; one seed and previously inspected dates do not establish generality. Current structure is a rule-readout diagnostic, not semantic truth. Zero-latent is out-of-distribution ablation, not a competitive compression baseline."
    write(directory / "ae_metrics.json", report)
    write(directory / "reconstruction_examples.json", examples)
    write_review(directory / "reconstruction_examples.html", examples)
    progress(f"Saved {directory / 'ae_metrics.json'} and reconstruction_examples.json")


if __name__ == "__main__":
    main()
