"""Trainable causal per-bar backbone. Teacher fidelity is NOT transfer evidence."""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import SCHEMA
from .data import FEATURES, manifest, split_boundaries, split_mask
from .metrics import balanced_accuracy, event_f1
from .structure import AMP_EDGES, TIME_EDGES

HEADS = {
    "direction": (3, 3),
    "age": (3, 7),
    "amplitude": (3, 7),
    "event": (3, 3),
    "state": (3, 4),
    "compression": (1, 2),
}


@dataclass
class Config:
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    window: int = 128
    warmup: int = 32

    def __post_init__(self):
        if (
            self.hidden <= 0
            or self.heads <= 0
            or self.layers < 1
            or self.window < 16
            or self.hidden % self.heads
            or self.hidden % 2
            or not 0 <= self.warmup < self.window
        ):
            raise ValueError("Invalid hidden/heads/window/warmup configuration")

    @property
    def trained_windows(self):
        return sorted({w for w in (self.window // 2, self.window) if w > self.warmup and w >= 16})


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(len(FEATURES), cfg.hidden)
        layer = nn.TransformerEncoderLayer(
            cfg.hidden,
            cfg.heads,
            4 * cfg.hidden,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.hidden)
        self.heads = nn.ModuleDict({k: nn.Linear(cfg.hidden, n * c) for k, (n, c) in HEADS.items()})

    def forward(self, x):
        b, t, _ = x.shape
        h = self.input(x)
        pos = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(
            torch.arange(0, self.cfg.hidden, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000) / self.cfg.hidden)
        )
        pe = torch.zeros(t, self.cfg.hidden, device=x.device, dtype=x.dtype)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        h = self.norm(self.encoder(h + pe, mask=mask))
        out = {k: head(h).reshape(b, t, *HEADS[k]) for k, head in self.heads.items()}
        out["h"] = h
        # Explicit retrieval readout; not assumed optimal, tested against baselines.
        out["embedding"] = torch.cat([h[:, -1], h[:, -min(32, t) :].mean(1)], -1)
        return out


def targets(s):
    labels = s.labels
    return {
        "direction": labels["direction"],
        "age": np.digitize(labels["age"], TIME_EDGES),
        "amplitude": np.digitize(labels["amplitude"], AMP_EDGES),
        "event": labels["event"],
        "state": labels["state"],
        "compression": labels["compression"].astype(np.int64)[:, None],
    }


class Windows(Dataset):
    def __init__(self, series, bounds, split, cfg, stride=16):
        self.series, self.cfg = series, cfg
        self.targets = [targets(s) for s in series]
        self.allowed = [split_mask(s, bounds, split) for s in series]
        self.items = [
            (i, int(j))
            for i, s in enumerate(series)
            for j in range(cfg.window - 1, len(s.frame), stride)
            if self.allowed[i][j]
        ]
        if not self.items:
            raise ValueError(f"No {split} windows; reduce window/stride or supply more data")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, j = self.items[index]
        lo = j - self.cfg.window + 1
        valid = self.allowed[i][lo : j + 1].copy()
        valid[: self.cfg.warmup] = False
        return dict(
            x=torch.from_numpy(self.series[i].x[lo : j + 1]),
            valid=torch.from_numpy(valid),
            series=i,
            row=j,
            **{
                k: torch.from_numpy(v[lo : j + 1].astype(np.int64))
                for k, v in self.targets[i].items()
            },
        )


def task_loss(out, batch, device):
    mask = batch["valid"].to(device)
    losses = {}
    for k, (_, classes) in HEADS.items():
        weight = torch.tensor([1.0, 5.0, 5.0], device=device) if k == "event" else None
        losses[k] = nn.functional.cross_entropy(
            out[k][mask].reshape(-1, classes), batch[k].to(device)[mask].reshape(-1), weight=weight
        )
    return losses


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@torch.no_grad()
def evaluate(model, ds, device, batch_size=64):
    """Only score the window endpoint: same position/context as the online query."""
    model.eval()
    preds, truths = {k: [] for k in HEADS}, {k: [] for k in HEADS}
    ids, rows = [], []
    for batch in DataLoader(ds, batch_size=batch_size):
        out = model(batch["x"].to(device))
        for k in HEADS:
            preds[k].append(out[k][:, -1].argmax(-1).cpu().numpy())
            truths[k].append(batch[k][:, -1].numpy())
        ids.extend(batch["series"].tolist())
        rows.extend(batch["row"].tolist())
    p, g = (
        {k: np.concatenate(v) for k, v in preds.items()},
        {k: np.concatenate(v) for k, v in truths.items()},
    )
    scores = {k: balanced_accuracy(p[k], g[k], c) for k, (_, c) in HEADS.items()}
    groups = []
    ids, rows = np.array(ids), np.array(rows)
    for i in np.unique(ids):
        for s in range(3):
            for c in (1, 2):
                groups.append(
                    (
                        rows[(ids == i) & (p["event"][:, s] == c)],
                        rows[(ids == i) & (g["event"][:, s] == c)],
                    )
                )
    scores["confirmation_events"] = event_f1(groups)
    scores["samples"] = len(ids)
    scores["interpretation"] = (
        "Endpoint fidelity to observable rules; not proof of shape transfer or trading value."
    )
    return scores


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("schema") != SCHEMA or ck.get("features") != list(FEATURES):
        raise ValueError("Incompatible checkpoint: Babel requires its own causal feature schema")
    model = Encoder(Config(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def train(
    series,
    directory,
    cfg=None,
    epochs=10,
    stride=16,
    batch_size=64,
    lr=3e-4,
    seed=42,
    device="auto",
    boundaries=None,
):
    cfg = cfg or Config()
    if epochs < 1 or stride < 1:
        raise ValueError("epochs and stride must be positive")
    directory = Path(directory)
    if (directory / "best.pt").exists():
        raise ValueError("Output already contains a checkpoint; choose a new run directory")
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device_for(device)
    bounds = boundaries or split_boundaries(series)
    tr = Windows(series, bounds, "train", cfg, stride)
    va = Windows(series, bounds, "val", cfg, stride)
    model = Encoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    run = {
        "schema": SCHEMA,
        "config": asdict(cfg),
        "seed": seed,
        "epochs": epochs,
        "stride": stride,
        "batch_size": batch_size,
        "lr": lr,
        "manifest": manifest(series, bounds),
        "selection": "mean endpoint validation BA across rule tasks, excluding events",
        "status": "experimental; no independent transfer graduation",
        "features": list(FEATURES),
    }
    (directory / "manifest.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
    best = -1
    for epoch in range(1, epochs + 1):
        model.train()
        sums, count = dict.fromkeys(HEADS, 0.0), 0
        for batch in DataLoader(tr, batch_size=batch_size, shuffle=True):
            length = random.choice(cfg.trained_windows)
            for k in ("x", "valid", *HEADS):
                batch[k] = batch[k][:, -length:]
            batch["valid"][:, : cfg.warmup] = False
            out = model(batch["x"].to(device))
            parts = task_loss(out, batch, device)
            optimizer.zero_grad()
            sum(parts.values()).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, loss in parts.items():
                sums[k] += loss.item()
            count += 1
        metrics = evaluate(model, va, device, batch_size)
        score = float(np.mean([metrics[k]["ba"] for k in HEADS if k != "event"]))
        record = {
            "epoch": epoch,
            "losses": {k: v / count for k, v in sums.items()},
            "validation": metrics,
            "score": score,
        }
        with (directory / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch={epoch} val_endpoint_BA={score:.4f} losses={record['losses']}", flush=True)
        if score > best:
            best = score
            torch.save(
                dict(**run, model=model.state_dict(), epoch=epoch, validation=metrics),
                directory / "best.pt",
            )
    return {
        "checkpoint": str(directory / "best.pt"),
        "train_windows": len(tr),
        "validation_windows": len(va),
        "best_validation_score": best,
        "boundaries": bounds,
    }
