"""Babel AR pilot: train, evaluate, and free-run without future price inputs."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ar_codec import FEATURES, TARGETS, State, consume, decode, encode_frame
from .ar_model import ARModel, nll, sample
from .data import load_series, manifest, split_mask
from .metrics import balanced_accuracy, block_interval
from .progress import progress
from .representation import time_mask

SCHEMA = "babel-ar-v1"
WINDOW, WARMUP = 256, 64


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


class Windows(Dataset):
    def __init__(self, series, encoded, bounds, split, stride=16, horizon=1, window=WINDOW):
        self.series, self.encoded, self.window = series, encoded, window
        self.items, self.valid = [], []
        for i, s in enumerate(series):
            valid = split_mask(s, bounds, split).copy()
            same = time_mask(s, bounds, split)
            for h in range(1, horizon + 1):
                shifted = np.zeros(len(s.x), bool)
                shifted[:-h] = same[h:]
                valid &= shifted
            self.valid.append(valid)
            self.items.extend((i, j) for j in range(window - 1, len(s.x), stride) if valid[j])
        if not self.items:
            raise ValueError(f"No {split} windows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        i, end = self.items[idx]
        e = self.encoded[i]
        rows = np.arange(end - self.window + 1, end + 1)
        future = np.minimum(rows + 1, len(e["x"]) - 1)
        mask = self.valid[i][rows].copy()
        mask[:min(WARMUP, self.window // 2)] = False
        available = np.ones((len(rows), 7), bool)
        available[:, 5] = e["available"][future]
        return {"x": e["x"][rows], "y": e["y"][future], "valid": mask,
                "available": available, "persistence": e["persistence"][rows], "series": i, "row": end}


def fit_baselines(ds):
    y, p, masks = [], [], []
    for i, row in ds.items:
        e = ds.encoded[i]
        y.append(e["y"][row + 1])
        p.append(e["persistence"][row])
        a = np.ones(7, bool)
        a[5] = e["available"][row + 1]
        masks.append(a)
    y, p, masks = np.array(y), np.array(p), np.array(masks)
    means, std, residual = np.zeros(7), np.ones(7), np.ones(7)
    for f in range(7):
        v = y[masks[:, f], f]
        if len(v):
            means[f], std[f] = v.mean(), float(np.clip(v.std(), np.exp(-5), np.exp(3)))
            residual[f] = float(np.clip(np.sqrt(np.mean((v - p[masks[:, f], f]) ** 2)), np.exp(-5), np.exp(3)))
    return {"mean": means.tolist(), "std": std.tolist(), "persistence_std": residual.tolist()}


def baseline_output(mean, std):
    return {"mean": mean.unsqueeze(-2), "log_std": std.log().expand_as(mean).unsqueeze(-2),
            "logits": torch.zeros(*mean.shape[:-1], 1, device=mean.device)}


@torch.no_grad()
def evaluate_one(model, ds, device, batch_size, baseline):
    model.eval()
    scores = {k: [] for k in ("model", "constant", "persistence")}
    blocks = []
    loader = DataLoader(ds, batch_size=batch_size)
    for step, b in enumerate(loader, 1):
        out = model(b["x"].to(device))
        out = {k: v[:, -1] for k, v in out.items() if k in ("mean", "log_std", "logits")}
        y, a = b["y"][:, -1].to(device), b["available"][:, -1].to(device)
        constant = baseline_output(y.new_tensor(baseline["mean"]).expand_as(y), y.new_tensor(baseline["std"]))
        persist = baseline_output(b["persistence"][:, -1].to(device), y.new_tensor(baseline["persistence_std"]))
        for k, prediction in (("model", out), ("constant", constant), ("persistence", persist)):
            scores[k].extend((nll(prediction, y, a) / a.sum(-1)).cpu().tolist())
        for i, row in zip(b["series"].tolist(), b["row"].tolist(), strict=True):
            blocks.append(str(ds.series[i].frame.datetime.iloc[row].to_period("W")))
        if step == 1 or step % 100 == 0 or step == len(loader):
            progress(f"AR evaluation {step}/{len(loader)}")
    return {"samples": len(blocks), "mean_nll_per_observed_coordinate": {k: float(np.mean(v)) for k, v in scores.items()},
            "paired_weekly_model_minus_baseline": {k: block_interval(np.array(scores["model"]) - scores[k], blocks)
                                                   for k in ("constant", "persistence")}}


def train(series, encoded, bounds, directory, epochs, batch_size, device, seed):
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use an empty run directory; no overwrite/resume")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tr, va = (Windows(series, encoded, bounds, split) for split in ("train", "val"))
    model = ARModel().to(device)
    baseline = fit_baselines(tr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    meta = {"schema": SCHEMA, "config": model.config, "features": FEATURES, "targets": TARGETS,
            "window": WINDOW, "warmup": WARMUP, "manifest": manifest(series, bounds),
            "baseline": baseline, "epochs": epochs, "batch_size": batch_size, "seed": seed,
            "train_samples": len(tr), "validation_samples": len(va),
            "parameters": sum(p.numel() for p in model.parameters()),
            "selection": "validation endpoint NLL per observed coordinate",
            "loss": "next-bar transformed-coordinate mixture NLL + 0.1 current feature reconstruction"}
    write(directory / "manifest.json", meta)
    progress(f"AR train: parameters={meta['parameters']:,}; windows={len(tr):,}/{len(va):,}")
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(2)
        loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
        for step, b in enumerate(loader, 1):
            length = random.choice((128, 256))
            for k in ("x", "y", "available", "valid"):
                b[k] = b[k][:, -length:]
            b["valid"][:, :WARMUP] = False
            x, y, a, mask = (b[k].to(device) for k in ("x", "y", "available", "valid"))
            out = model(x)
            forecast = (nll(out, y, a) / a.sum(-1))[mask].mean()
            keep = mask.unsqueeze(-1).expand_as(x).clone()
            keep[..., 10] &= x[..., 11] > 0
            reconstruction = torch.nn.functional.smooth_l1_loss(out["reconstruction"], x, reduction="none")[keep].mean()
            loss = forecast + .1 * reconstruction
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite AR loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            totals += [forecast.item(), reconstruction.item()]
            if step == 1 or step % 25 == 0 or step == len(loader):
                progress(f"epoch={epoch}/{epochs} batch={step}/{len(loader)} NLL/reconstruction={(totals / step).round(5).tolist()}")
        metrics = evaluate_one(model, va, device, batch_size, baseline)
        with (directory / "history.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train_losses": (totals / step).tolist(), "validation": metrics}) + "\n")
        value = metrics["mean_nll_per_observed_coordinate"]["model"]
        progress(f"epoch={epoch} validation_NLL={value:.6f}")
        if value < best:
            best = value
            torch.save({**meta, "epoch": epoch, "model": model.state_dict()}, directory / "best.pt")


@torch.no_grad()
def rollout(model, ds, device, seed=123, count=32, paths=8, steps=16):
    """Only prefix data enters generated paths; truth is read afterwards for scoring."""
    model.eval()
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    cases = []
    for number, idx in enumerate(rng.choice(len(ds), min(count, len(ds)), replace=False), 1):
        i, end = ds.items[int(idx)]
        s, e = ds.series[i], ds.encoded[i]
        contexts = np.repeat(e["x"][None, end - ds.window + 1:end + 1], paths, axis=0)
        states = [State(float(s.frame.close.iloc[end]), float(e["variance"][end])) for _ in range(paths)]
        alive = np.ones(paths, bool)
        generated = np.full((paths, steps, 4), np.nan)
        for step in range(steps):
            ids = np.flatnonzero(alive)
            if not len(ids):
                break
            out = model(torch.tensor(contexts[ids], device=device))
            prediction = {k: v[:, -1] for k, v in out.items() if k in ("mean", "log_std", "logits")}
            draws = sample(prediction, generator).cpu().numpy()
            for j, y in zip(ids, draws, strict=True):
                try:
                    bar, volume, oi, elapsed = decode(states[j], y, e["available"][end])
                    x, _, _ = consume(states[j], bar, volume, oi, e["available"][end], s.period, elapsed)
                    if not np.isfinite(x).all():
                        raise ValueError("Nonfinite generated input")
                    generated[j, step] = bar
                    contexts[j] = np.concatenate((contexts[j, 1:], x[None]), axis=0)
                except ValueError:
                    alive[j] = False
        # Targets are accessed only after the entire free-running generation.
        truth = s.frame.close.to_numpy()[end + 1:end + steps + 1]
        start = float(s.frame.close.iloc[end])
        complete = generated[alive, :, 3]
        result = {"source": s.key, "anchor": str(s.frame.datetime.iloc[end]),
                  "paths": paths, "failed_paths": int((~alive).sum()), "horizons": {}}
        if number <= 3:
            result["trace"] = {
                "prefix_ohlc": s.frame[["open", "high", "low", "close"]].iloc[end - 15:end + 1].to_numpy().tolist(),
                "truth_ohlc": s.frame[["open", "high", "low", "close"]].iloc[end + 1:end + steps + 1].to_numpy().tolist(),
                "generated_ohlc": [[bar.tolist() if np.isfinite(bar).all() else None for bar in path] for path in generated],
            }
        if len(complete):
            for h in (1, 4, 16):
                pred = np.log(complete[:, h - 1]) - np.log(start)
                actual = float(np.log(truth[h - 1]) - np.log(start))
                crps = np.abs(pred - actual).mean() - .5 * np.abs(pred[:, None] - pred[None, :]).mean()
                result["horizons"][str(h)] = {"empirical_crps_log_return": float(crps),
                    "last_close_crps": abs(actual), "median_abs_error": float(abs(np.median(pred) - actual)),
                    "sample_std_log_return": float(pred.std()), "truth_log_return": actual}
        cases.append(result)
        progress(f"free rollout {number}/{min(count, len(ds))}, failed_paths={result['failed_paths']}/{paths}")
    summary = {}
    for h in ("1", "4", "16"):
        eligible = [c["horizons"][h] for c in cases if h in c["horizons"]]
        summary[h] = {"scored_anchors": len(eligible), **{
            k: float(np.mean([c[k] for c in eligible])) if eligible else None
            for k in ("empirical_crps_log_return", "last_close_crps", "median_abs_error", "sample_std_log_return")}}
    return {"anchors": len(cases), "paths_per_anchor": paths, "failed_paths": sum(c["failed_paths"] for c in cases),
            "summary": summary, "cases": cases,
            "note": "Stochastic free rollout; no true future prices, volume, OI, sigma or timestamps are fed back. Scores use surviving complete paths only, so failures must be read alongside scores. Eight paths give a noisy empirical CRPS. Not an economic backtest."}


@torch.no_grad()
def extract(model, ds, device, batch_size):
    features = {"embedding": [], "raw": []}
    labels = {"current_structure": [], "future8_direction": []}
    loader = DataLoader(ds, batch_size=batch_size)
    for step, b in enumerate(loader, 1):
        x = b["x"].to(device)
        features["embedding"].append(model(x)["seq"].cpu().numpy())
        # Stronger raw baseline preserves each of the last sixteen bars.
        features["raw"].append(x[:, -16:].flatten(1).cpu().numpy())
        for i, row in zip(b["series"].tolist(), b["row"].tolist(), strict=True):
            s = ds.series[i]
            labels["current_structure"].append(int(s.labels["state"][row, 1]))
            change = (s.frame.close.iloc[row + 8] - s.frame.close.iloc[row]) / s.labels["atr"][row]
            labels["future8_direction"].append(int(np.digitize(change, [-.5, .5])))
        if step == 1 or step % 100 == 0 or step == len(loader):
            progress(f"AR frozen probe extraction {step}/{len(loader)}")
    return {k: np.concatenate(v) for k, v in features.items()}, {k: np.array(v) for k, v in labels.items()}


def probe(a, y, b, v, c, z, classes):
    """Balanced ridge, fixed validation-only regularization grid."""
    mean, scale = a.mean(0), a.std(0).clip(.01)
    a, b, c = [np.column_stack(((x - mean) / scale, np.ones(len(x)))).astype(np.float64) for x in (a, b, c)]
    counts = np.bincount(y, minlength=classes)
    w = len(y) / (max(1, (counts > 0).sum()) * np.maximum(counts[y], 1))
    lhs, rhs = a.T @ (a * w[:, None]), a.T @ (np.eye(classes)[y] * w[:, None])
    best = None
    for alpha in (1., 10., 100.):
        penalty = np.eye(a.shape[1]) * alpha
        penalty[-1, -1] = 0
        weight = np.linalg.solve(lhs + penalty, rhs)
        score = balanced_accuracy((b @ weight).argmax(1), v, classes)["ba"]
        if best is None or score > best[0]:
            best = score, alpha, weight
    return {"validation_ba": best[0], "alpha": best[1], "test": balanced_accuracy((c @ best[2]).argmax(1), z, classes)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "evaluate"))
    p.add_argument("--root", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; no CPU training fallback")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Positive epochs and batch size required")
    directory = Path(args.out)
    reference = json.loads(Path(args.reference).read_text())["manifest"]
    keys = [s["key"].split("/") for s in reference["sources"]]
    series, _ = load_series(args.root, sorted({k[0] for k in keys}), sorted({int(k[1]) for k in keys}))
    if manifest(series)["sources"] != reference["sources"]:
        raise ValueError("Source fingerprint mismatch")
    bounds = reference["boundaries"]
    encoded = []
    for i, s in enumerate(series, 1):
        encoded.append(encode_frame(s.frame, s.period))
        if i == 1 or i % 50 == 0 or i == len(series):
            progress(f"AR causal codec {i}/{len(series)}")
    if args.stage == "train":
        train(series, encoded, bounds, directory, args.epochs, args.batch_size, "cuda", args.seed)
        return
    ck = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    if ck["schema"] != SCHEMA or ck["manifest"] != manifest(series, bounds) or list(ck["features"]) != list(FEATURES):
        raise ValueError("Checkpoint schema/data mismatch")
    model = ARModel(**ck["config"]).to("cuda")
    model.load_state_dict(ck["model"])
    model.eval()
    te = Windows(series, encoded, bounds, "test")
    report = {"schema": SCHEMA, "selected_epoch": ck["epoch"],
              "one_step": evaluate_one(model, te, "cuda", args.batch_size, ck["baseline"])}
    datasets = [Windows(series, encoded, bounds, split, horizon=16) for split in ("train", "val", "test")]
    report["rollout"] = rollout(model, datasets[-1], "cuda")
    saved = []
    for split, ds in zip(("train", "val", "test"), datasets, strict=True):
        progress(f"frozen representation extraction: {split}")
        saved.append(extract(model, ds, "cuda", args.batch_size))
    report["probes"] = {}
    for name in ("embedding", "raw"):
        report["probes"][name] = {task: probe(*(value for pair in [(f[name], y[task]) for f, y in saved] for value in pair), classes)
                                  for task, classes in (("current_structure", 4), ("future8_direction", 3))}
    torch.manual_seed(ck["seed"])
    random_model = ARModel(**ck["config"]).to("cuda").eval()
    random_saved = [extract(random_model, ds, "cuda", args.batch_size) for ds in datasets]
    report["probes"]["random_embedding"] = {task: probe(*(value for pair in [(f["embedding"], y[task]) for f, y in random_saved] for value in pair), classes)
                                                for task, classes in (("current_structure", 4), ("future8_direction", 3))}
    report["interpretation"] = "Single-seed pilot. Previously inspected test dates are not a pristine holdout. Current structure probe uses rules not used in AR training; it is a rule-readout diagnostic, not universal shape truth. Future direction is a separate task. Architecture/codec/objective changed together: not a capacity ablation."
    write(directory / "ar_metrics.json", report)
    progress(f"Evaluation saved: {directory / 'ar_metrics.json'}")


if __name__ == "__main__":
    main()
