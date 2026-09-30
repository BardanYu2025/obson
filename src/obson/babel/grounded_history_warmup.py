"""Frozen-state semantic qualification; no encoder update before this gate passes."""

import copy

import numpy as np
import torch

from . import grounded_history as core
from . import grounded_history_data as gd
from .ae_extend import atomic_json, atomic_save
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

ABLATIONS = ("none", "state_only", "query_only")


@torch.no_grad()
def cache(meta, out, seed, device):
    from . import grounded_history_run as run

    path = out / f"qualification_s{seed}"
    path.mkdir(exist_ok=True)
    identity = {
        "manifest": sha256(out / "manifest.json"),
        "seed": seed,
        "data_lock": sha256(out / "grounded_data_lock.json"),
    }
    if (path / "cache_lock.json").exists():
        lock = read_json(path / "cache_lock.json")
        if lock["identity"] != identity:
            raise ValueError("Warmup cache identity differs")
        verify_files(path, lock["files"])
        return torch.load(path / "cache.pt", map_location="cpu", weights_only=True)
    model, query = run.construct(meta, seed, device)
    model.requires_grad_(False)
    builder = run.xr.xd.Builder(run.tm(meta), run.root(meta))
    schedule = run.plan(meta, seed, 0, len(builder.plan))
    result = {}
    for split in ("train", "val"):
        rows = (
            [builder.plan[i] for i in schedule[0]]
            if split == "train"
            else read_json(out / "val_grounded_plan.json")["rows"]
        )
        parts = []
        for start in range(0, len(rows), meta["evaluation_batch"]):
            stop = min(start + meta["evaluation_batch"], len(rows))
            if split == "train":
                b, _, _, _, ages, mask, _ = run.make_batch(builder, schedule, start, stop, device)
                coords = core.queries(ages, mask)
                target = core.targets(b, coords, run.stats(meta)[0])
            else:
                b, coords, _, target = gd.evaluation_batch(
                    meta, rows[start:stop], "val", 16, device
                )
            z = model.core.encoder(b["x"])[:, -1]
            z = ((z - query.state_mean) / query.state_scale).reshape(3, stop - start, -1)
            parts.append((z.cpu(), coords.cpu(), target.cpu()))
        result[split] = {
            k: torch.cat([p[i] for p in parts], 1) for i, k in enumerate(("z", "q", "y"))
        }
        result[split]["rows"] = rows
    y = result["train"]["y"].double()
    mean, scale = float(y.mean()), float(y.std(unbiased=False))
    if not np.isfinite([mean, scale]).all() or scale <= 1e-8:
        raise ValueError("Degenerate training-only target scale")
    result["target_stats"] = {
        "mean": mean,
        "scale": scale,
        "unit": "original log-percent close change",
        "fit": "only fixed qualification training exposures, both ordered slots",
    }
    for split in ("train", "val"):
        result[split]["y"] = (result[split]["y"] - mean) / scale
    atomic_save(result, path / "cache.pt")
    atomic_json(result["target_stats"], path / "target_stats.json")
    atomic_json(
        {
            "identity": identity,
            "files": {n: sha256(path / n) for n in ("cache.pt", "target_stats.json")},
        },
        path / "cache_lock.json",
    )
    return result


def predict(head, dataset, device, ablation="none", batch=128):
    pieces = []
    for start in range(0, dataset["z"].shape[1], batch):
        z = dataset["z"][:, start : start + batch].to(device)
        q = dataset["q"][:, start : start + batch].to(device)
        expanded = z[:, :, None].expand(-1, -1, 2, -1)
        _, pred = head(expanded.reshape(-1, z.shape[-1]), q.reshape(-1, core.SLOTS, 2), ablation)
        pieces.append(pred.reshape(3, z.shape[1], 2, core.SLOTS).cpu())
    return torch.cat(pieces, 1)


def interval(errors, reference, rows, factor=0.9):
    values = np.asarray(errors) - factor * np.asarray(reference)
    groups = [
        values[[i for i, r in enumerate(rows) if r["key"] == key]]
        for key in sorted({r["key"] for r in rows})
    ]
    if not groups:
        return {"supported": False, "high": None}
    sums, counts = np.array([v.sum() for v in groups]), np.array([len(v) for v in groups])
    rng = np.random.default_rng(20261001)
    ix = rng.integers(len(groups), size=(2000, len(groups)))
    draws = sums[ix].sum(1) / counts[ix].sum(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "supported": len(rows) >= 50 and len(groups) >= 5,
        "rows": len(rows),
        "contracts": len(groups),
        "delta": float(values.mean()),
        "low": float(lo),
        "high": float(hi),
        "factor": factor,
    }


def qualify(predictions, target, rows):
    """Same-state wrong-query counterfactual; exclude genuinely similar target pairs."""
    target = np.asarray(target)
    if target.shape[0] != 3 or target.shape[2:] != (2, core.SLOTS):
        raise ValueError("Three views with two two-slot content queries required")
    if set(predictions) != set(ABLATIONS) or any(
        np.shape(v) != target.shape for v in predictions.values()
    ):
        raise ValueError("All equally trained ablations required")
    if not np.isfinite(target).all() or any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("Nonfinite qualification")
    informative = np.square(target[1, :, 0] - target[1, :, 1]).mean(-1) >= 0.05
    rs = [r for r, keep in zip(rows, informative, strict=True) if keep]
    report = {"informative_rows": len(rs), "total_rows": len(rows), "checks": [], "views": {}}
    for view, name in enumerate(("earlier_fine", "fine", "coarse")):
        y = target[view, informative]
        pred = np.asarray(predictions["none"])[view, informative]
        errors = {
            "correct": np.square(pred - y).mean((1, 2)),
            "wrong_query": np.square(pred[:, ::-1] - y).mean((1, 2)),
            "train_mean": np.square(y).mean((1, 2)),
        }
        for ablation in ABLATIONS[1:]:
            errors[ablation] = np.square(
                np.asarray(predictions[ablation])[view, informative] - y
            ).mean((1, 2))
        report["views"][name] = {k: v.tolist() for k, v in errors.items()}
        for ref in ("wrong_query", "state_only", "query_only", "train_mean"):
            ci = interval(errors["correct"], errors[ref], rs)
            report["checks"].append(
                {
                    "view": name,
                    "reference": ref,
                    "interval": ci,
                    "passed": bool(ci["supported"] and ci["high"] < 0),
                }
            )
    report["passed"] = all(c["passed"] for c in report["checks"])
    report["scope"] = (
        "Validation-only admission gate, contract bootstrap,10% MSE margin; not independent research evidence."
    )
    return report


def fit(meta, out, seed, device):
    from . import grounded_history_run as run

    path = out / f"qualification_s{seed}"
    d = cache(meta, out, seed, device)
    identity = {
        "manifest": sha256(out / "manifest.json"),
        "cache": sha256(path / "cache_lock.json"),
    }
    if (path / "completion.json").exists():
        done = read_json(path / "completion.json")
        if done["identity"] != identity:
            raise ValueError("Qualification identity differs")
        verify_files(path, done["files"])
        return read_json(path / "qualification.json")
    predictions = {}
    best_head = None
    for ablation in ABLATIONS:
        head = core.GroundedReader(d["train"]["z"].shape[-1], seed, meta["shared"]["width"]).to(
            device
        )
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
        statefile = path / f"{ablation}_resume.pt"
        state = {
            "identity": identity,
            "epoch": 0,
            "history": [],
            "best_loss": float("inf"),
            "best_epoch": 0,
        }
        if statefile.exists():
            state = torch.load(statefile, map_location="cpu", weights_only=True)
            if state["identity"] != identity or len(state["history"]) != state["epoch"]:
                raise ValueError("Warmup resume differs")
            head.load_state_dict(state["head"], strict=True)
            opt.load_state_dict(state["optimizer"])
        for epoch in range(state["epoch"] + 1, meta["qualification_epochs"] + 1):
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 7001])).permutation(
                d["train"]["z"].shape[1]
            )
            for start in range(0, len(order), meta["batch"]):
                ids = order[start : start + meta["batch"]]
                z = d["train"]["z"][:, ids].to(device)
                q = d["train"]["q"][:, ids].to(device)
                y = d["train"]["y"][:, ids].to(device)
                z = z[:, :, None].expand(-1, -1, 2, -1)
                _, pred = head(z.reshape(-1, z.shape[-1]), q.reshape(-1, core.SLOTS, 2), ablation)
                loss = (pred.reshape_as(y) - y).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
                opt.step()
            with torch.no_grad():
                val = float(
                    (predict(head, d["val"], device, ablation) - d["val"]["y"]).square().mean()
                )
            if not np.isfinite(val):
                raise ValueError("Nonfinite warmup loss")
            if val < state["best_loss"]:
                state.update(
                    best_loss=val, best_epoch=epoch, best_head=copy.deepcopy(run.parent.cpu(head))
                )
            state["history"].append(
                {
                    "epoch": epoch,
                    "val_mse": val,
                    "steps": (len(order) + meta["batch"] - 1) // meta["batch"],
                }
            )
            state.update(epoch=epoch, head=run.parent.cpu(head), optimizer=opt.state_dict())
            atomic_save(state, statefile)
            atomic_json(
                {k: state[k] for k in ("epoch", "history", "best_loss", "best_epoch")},
                path / f"{ablation}_history.json",
            )
            progress(
                f"qualification/s{seed}/{ablation}: {epoch}/{meta['qualification_epochs']}, val_mse={val:.6f}, best={state['best_epoch']}"
            )
        atomic_json(
            {k: state[k] for k in ("epoch", "history", "best_loss", "best_epoch")},
            path / f"{ablation}_history.json",
        )
        head.load_state_dict(state["best_head"], strict=True)
        with torch.no_grad():
            predictions[ablation] = predict(head, d["val"], device, ablation).numpy()
        if ablation == "none":
            best_head = run.parent.cpu(head)
    report = qualify(predictions, d["val"]["y"].numpy(), d["val"]["rows"])
    atomic_json(report, path / "qualification.json")
    atomic_save(
        {
            "identity": identity,
            "head": best_head,
            "target_stats": d["target_stats"],
            "passed": report["passed"],
        },
        path / "selected.pt",
    )
    names = ["selected.pt", "qualification.json", "target_stats.json", "cache_lock.json"] + [
        f"{a}_history.json" for a in ABLATIONS
    ]
    atomic_json(
        {"identity": identity, "files": {n: sha256(path / n) for n in names}},
        path / "completion.json",
    )
    return report


def require_passed(out):
    for seed in (42, 43):
        path = out / f"qualification_s{seed}"
        done = read_json(path / "completion.json")
        if done["identity"]["manifest"] != sha256(out / "manifest.json"):
            raise ValueError("Foreign qualification")
        verify_files(path, done["files"])
        if not read_json(path / "qualification.json")["passed"]:
            raise ValueError("Frozen semantic qualification failed; joint training prohibited")


def load_head(query, out, seed):
    require_passed(out)
    value = torch.load(
        out / f"qualification_s{seed}/selected.pt", map_location="cpu", weights_only=True
    )
    if not value["passed"]:
        raise ValueError("Unqualified auxiliary head")
    query.shared.load_state_dict(value["head"], strict=True)
    return value["target_stats"]
