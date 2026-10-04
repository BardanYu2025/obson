"""Locked reconstruction and frozen-utility evaluation for RS-v2."""

import gc
from functools import lru_cache

import numpy as np
import torch

from . import recovery_rs as training
from . import utility_probe as up
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

old, r0 = training.old, training.r0
SPLITS = ("test", "cross_research")


def variants():
    return [
        {"name": f"source_s{s}", "seed": s, "job": None, "kind": "source"} for s in (42, 43)
    ] + [
        {"name": j["name"] + "_" + k, "seed": j["seed"], "job": j, "kind": k}
        for j in training.jobs()
        for k in ("best", "last")
    ]


def check_models(out):
    lock = read_json(out / "model_selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Model selection binding changed")
    for n, h in lock["files"].items():
        r0.required_file(out, n, h)
    return lock


def load(meta, out, v, device):
    e, q = old.construct(meta, v["seed"], device)
    if v["job"]:
        ck = torch.load(
            out / v["job"]["name"] / (v["kind"] + ".pt"), map_location="cpu", weights_only=True
        )
        if ck["binding"] != training.binding(out, v["job"]):
            raise ValueError("Evaluation snapshot binding changed")
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
    return e.eval().requires_grad_(False), q.eval().requires_grad_(False)


def summaries(arrays):
    result = {}
    for k, a in arrays.items():
        result[k] = {
            "mean": float(a.mean()),
            "p50": float(np.quantile(a, 0.5)),
            "p90": float(np.quantile(a, 0.9)),
            "p99": float(np.quantile(a, 0.99)),
            "positions": [
                {
                    "prefix": p,
                    "mean": float(a[:, j].mean()),
                    "p50": float(np.quantile(a[:, j], 0.5)),
                    "p90": float(np.quantile(a[:, j], 0.9)),
                    "p99": float(np.quantile(a[:, j], 0.99)),
                }
                for j, p in enumerate(old.core.PREFIXES)
            ],
        }
    return result


def physical_rows(pred, target, mask, stats):
    p, y = (
        training.objective.hq.physical(pred, stats),
        training.objective.hq.physical(target, stats),
    )
    ok = mask[..., 0]
    error = (p[..., 0] - y[..., 0]) * 100  # log-percent -> log-price basis points
    both = ok[..., 1:] & ok[..., :-1]
    delta = error[..., 1:] - error[..., :-1]
    return {
        "close_mae_logbps": torch.where(ok, error.abs(), 0).sum(-1) / ok.sum(-1).clamp_min(1),
        "change_rmse_logbps": (
            torch.where(both, delta.square(), 0).sum(-1) / both.sum(-1).clamp_min(1)
        ).sqrt(),
    }


def check_readouts(out):
    lock = read_json(out / "readout_lock.json")
    if lock["model_lock_sha256"] != sha256(out / "model_selection_lock.json"):
        raise ValueError("Readout model lock changed")
    for n, h in lock["files"].items():
        r0.required_file(out, n, h)
    for split, h in lock["states_indexes"].items():
        if sha256(out / f"cache/{split}_index.json") != h:
            raise ValueError("Readout input states changed")
    return lock


@torch.no_grad()
def extract(meta, context, out, split, device):
    check_models(out)
    research = split in SPLITS
    if research:
        check_readouts(out)
    folder = out / "cache"
    folder.mkdir(exist_ok=True)
    x, rows = old.data.cache(context, split)
    bind = {
        "model_lock_sha256": sha256(out / "model_selection_lock.json"),
        "source_index_sha256": sha256(context / f"cache/{split}_index.json"),
    }
    index = folder / f"{split}_index.json"
    if index.exists():
        d = read_json(index)
        if d["binding"] != bind:
            raise ValueError("State extraction binding changed")
        for n, h in d["files"].items():
            r0.required_file(out, n, h)
        with np.load(folder / f"{split}_states.npz", allow_pickle=False) as f:
            return dict(f), x, rows
    stats = old.parent.stats(meta)[0]
    result = {}
    summary = {}
    examples = {}
    support = {}
    diagnostics = out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for v in variants():
        e, q = load(meta, out, v, device)
        name = v["name"]
        states = []
        errors = {}
        before = training.lr.model_signature(e, q)
        if split == "train":
            r0.causality(e, q, torch.as_tensor(x[:2], device=device), diagnostics, name)
        for left in range(0, len(x), meta["micro"]):
            b = torch.as_tensor(x[left : left + meta["micro"]], device=device)
            ps = torch.tensor(old.core.PREFIXES if research else (128,), device=device).expand(
                len(b), -1
            )
            if not research:
                states.append(old.core.states(e, b, ps, False)[:, -1].cpu().numpy())
                continue
            for view, native in (("full", False), ("native", True)):
                z, pred = old.core.predict(e, q, b, ps, native)
                if not native:
                    states.append(z[:, -1].cpu().numpy())
                y, mask = old.core.original.targets(b, ps, stats, native)
                raw = old.core.original.band_rows(pred, y, mask, stats)
                values = {k: raw[k] for k in old.core.METRICS}
                values.update(physical_rows(pred, y, mask, stats))
                for k, t in values.items():
                    errors.setdefault(view + "/" + k, []).append(t.cpu().numpy())
                if v == variants()[0]:
                    for metric, (lo, hi) in zip(
                        ("near", "mid", "far"), training.objective.hq.BANDS, strict=True
                    ):
                        support.setdefault(view + "/" + metric, []).append(
                            mask[..., lo - 1 : hi, 0].any(-1).cpu().numpy()
                        )
                if left == 0:
                    examples[f"{name}/{view}/prediction"] = pred[:2].cpu().numpy()
                    examples[f"{view}/target"] = y[:2].cpu().numpy()
                    examples[f"{view}/mask"] = mask[:2].cpu().numpy()
        result[name] = np.concatenate(states)
        if research:
            arrays = {k: np.concatenate(a) for k, a in errors.items()}
            summary[name] = summaries(arrays)
            result.update({f"error/{name}/{k}": a for k, a in arrays.items()})
        if training.lr.model_signature(e, q) != before:
            raise ValueError("Frozen evaluation changed weights/buffers")
        del e, q
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        progress(f"RS {split}: {name}, {len(x)} matched endpoints")
    if any(not np.isfinite(a).all() for a in result.values()):
        raise ValueError("Nonfinite extracted states or errors")
    training.interface.save_arrays(folder / f"{split}_states.npz", result)
    atomic_json(rows, out / f"{split}_rows.json")
    files = [f"cache/{split}_states.npz", f"{split}_rows.json"]
    if research:
        # Support is independent of the chosen model. Unsupported prefixes are explicitly marked.
        sup = {k: np.concatenate(a) for k, a in support.items()}
        for metrics in summary.values():
            for key, available in sup.items():
                for j, p in enumerate(metrics[key]["positions"]):
                    p["supported_windows"] = int(available[:, j].sum())
                    if not available[:, j].any():
                        for f in ("mean", "p50", "p90", "p99"):
                            p[f] = None
        atomic_json(summary, out / f"{split}_reconstruction.json")
        training.interface.save_arrays(out / f"{split}_examples.npz", examples)
        training.interface.save_arrays(out / f"{split}_support.npz", sup)
        files += [f"{split}_reconstruction.json", f"{split}_examples.npz", f"{split}_support.npz"]
    if split == "train":
        files += [f"diagnostics/{v['name']}_causality.json" for v in variants()]
    atomic_json({"binding": bind, "files": {n: sha256(out / n) for n in files}}, index)
    return result, x, rows


def representation(name, x, states, pca, scales):
    if name == "raw3584":
        return x[:, -128:].reshape(len(x), -1)
    if name == "current28":
        return x[:, -1]
    if name == "pca768":
        return up.probe.normalize(x[:, -128:].reshape(len(x), -1), scales) @ pca
    return states[name]


def fit(meta, context, out, device):
    if (out / "readout_lock.json").exists():
        check_readouts(out)
        return
    ts, train, _ = extract(meta, context, out, "train", device)
    vs, val, _ = extract(meta, context, out, "val", device)
    stats = old.parent.stats(meta)[0]
    y, mask = up.targets(up.restore_raw(train[:, -128:], stats))
    vy, vm = up.targets(up.restore_raw(val[:, -128:], stats))
    ys = up.target_scales(y, mask)
    raw = train[:, -128:].reshape(len(train), -1)
    xs = up.probe.scales(raw)
    if min(raw.shape) <= 768:
        raise ValueError("Insufficient rows for PCA768")
    eigen = up.probe.eigensystem(up.probe.normalize(raw, xs), device)
    pca = eigen[1][:, -768:].flip(1).cpu().numpy()
    del eigen
    heads = {}
    weights = {"pca": pca}
    for name in ["raw3584", "current28", "pca768"] + [v["name"] for v in variants()]:
        a = representation(name, train, ts, pca, xs)
        b = representation(name, val, vs, pca, xs)
        head, w, intercept = up.fit_heads(a, y, mask, b, vy, vm, ys, device)
        heads[name] = head
        weights[name + "/weights"] = w
        weights[name + "/intercepts"] = intercept
        progress(f"RS {name}:13 readouts, original5 alphas; train fit/val selection")
    training.interface.save_arrays(out / "readout_weights.npz", weights)
    atomic_json({"target_stats": ys, "raw_stats": xs, "heads": heads}, out / "readout_fit.json")
    atomic_json(
        {
            "model_lock_sha256": sha256(out / "model_selection_lock.json"),
            "files": {n: sha256(out / n) for n in ("readout_weights.npz", "readout_fit.json")},
            "states_indexes": {s: sha256(out / f"cache/{s}_index.json") for s in ("train", "val")},
        },
        out / "readout_lock.json",
    )


@lru_cache(maxsize=4)
def bootstrap_groups(keys):
    names = sorted(set(keys))
    lookup = {k: i for i, k in enumerate(names)}
    inverse = np.array([lookup[k] for k in keys])
    counts = np.bincount(inverse)
    ids = np.random.default_rng(20261004).integers(0, len(names), (2000, len(names)))
    weights = np.stack([np.bincount(a, minlength=len(names)) for a in ids]).astype(float)
    weights /= (weights @ counts)[:, None]
    return inverse, weights, len(names)


def paired(a, b, rows):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.shape != b.shape
        or len(a) != len(rows)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Finite matched paired arrays required")
    delta = (a - b).reshape(len(rows), -1).mean(1)
    result = {}
    for scheme, keys in (
        ("month", tuple(r["month"] for r in rows)),
        ("contract_month", tuple(r["key"] + "/" + r["month"] for r in rows)),
    ):
        inverse, w, groups = bootstrap_groups(keys)
        sums = np.bincount(inverse, weights=delta)
        low, high = np.quantile(w @ sums, [0.025, 0.975])
        result[scheme] = {
            "delta": float(delta.mean()),
            "low": float(low),
            "high": float(high),
            "groups": groups,
            "windows": len(rows),
            "supported": groups >= 6 and len(rows) >= 100,
        }
    return result


def check(a, b, rows, factor):
    ci = paired(a, factor * np.asarray(b), rows)
    return {
        "factor": factor,
        "intervals": ci,
        "passed": all(c["supported"] and c["high"] <= 0 for c in ci.values()),
    }


def metrics24(states, name):
    return {
        f"{view}/{context}/{metric}": states[f"error/{name}/{view}/{metric}"][:, sl]
        for view in ("full", "native")
        for context, sl in (("endpoint", slice(-1, None)), ("interior", slice(None, -1)))
        for metric in old.core.METRICS
    }


def gate(states, utility, rows, candidate, baseline, source, updated):
    a = metrics24(states, candidate)
    checks = {}
    utilities = {}
    for ref in (baseline, source):
        b = metrics24(states, ref)
        checks[ref] = {k: check(a[k], b[k], rows, 0.95 if k.endswith("/near") else 1.10) for k in a}
        utilities[ref] = {
            k: check(utility[candidate][k], utility[ref][k], rows, 0.95 if k == "utility" else 1.10)
            for k in ("utility", "direction", "volatility")
        }
    reconstruction = bool(updated and all(c["passed"] for v in checks.values() for c in v.values()))
    return {
        "actually_updated": updated,
        "reconstruction_progress": reconstruction,
        "representation_progress": bool(
            reconstruction and all(c["passed"] for v in utilities.values() for c in v.values())
        ),
        "reconstruction_checks": checks,
        "utility_checks": utilities,
    }


def effects(states, rows, seed, kind):
    names = {a: f"{a}_s{seed}_{kind}" for a in training.ARMS}
    vals = {a: metrics24(states, n) for a, n in names.items()}
    pairs = (
        ("remove_S_without_L", "without_remote", "baseline"),
        ("remove_S_with_L", "without_remote_with_local", "with_local"),
        ("add_L_with_S", "with_local", "baseline"),
        ("add_L_without_S", "without_remote_with_local", "without_remote"),
    )
    result = {
        label: {k: paired(vals[a][k], vals[b][k], rows) for k in vals[a]} for label, a, b in pairs
    }
    result["interaction"] = {
        k: paired(
            vals["without_remote_with_local"][k]
            - vals["without_remote"][k]
            - vals["with_local"][k]
            + vals["baseline"][k],
            np.zeros_like(vals["baseline"][k]),
            rows,
        )
        for k in vals["baseline"]
    }
    return result


def run(meta, context, out, device):
    check_models(out)
    fit(meta, context, out, device)
    check_readouts(out)
    fitted = read_json(out / "readout_fit.json")
    lock = read_json(out / "model_selection_lock.json")
    with np.load(out / "readout_weights.npz", allow_pickle=False) as f:
        weights = dict(f)
    reports = {}
    for split in SPLITS:
        states, x, rows = extract(meta, context, out, split, device)
        y, mask = up.targets(up.restore_raw(x[:, -128:], old.parent.stats(meta)[0]))
        scores = {}
        utilities = {}
        arrays = {"targets": y, "mask": mask}
        for name, head in fitted["heads"].items():
            xx = representation(name, x, states, weights["pca"], fitted["raw_stats"])
            pred = up.predict(
                head,
                weights[name + "/weights"],
                weights[name + "/intercepts"],
                xx,
                fitted["target_stats"],
            )
            scores[name], utilities[name] = up.measure(pred, y, mask, fitted["target_stats"])
            arrays[name + "/utility_predictions"] = pred
        arrays.update({k: v for k, v in states.items() if k.startswith("error/")})
        gates = {}
        factor = {}
        for seed in (42, 43):
            for kind in ("best", "last"):
                factor[f"s{seed}/{kind}"] = effects(states, rows, seed, kind)
                for arm in training.ARMS[1:]:
                    name = f"{arm}_s{seed}_{kind}"
                    updated = kind == "last" or lock["workers"][f"{arm}_s{seed}"]["best_epoch"] > 0
                    gates[name] = gate(
                        states,
                        utilities,
                        rows,
                        name,
                        f"baseline_s{seed}_{kind}",
                        f"source_s{seed}",
                        updated,
                    )
        cohorts = {}
        for field in ("key", "period"):
            for value in sorted({str(r[field]) for r in rows}):
                ids = np.array([str(r[field]) == value for r in rows])
                cohorts[field + "/" + value] = {
                    "windows": int(ids.sum()),
                    "models": {
                        v["name"]: {
                            "near": float(states[f"error/{v['name']}/full/near"][ids].mean()),
                            "utility": float(utilities[v["name"]]["utility"][ids].mean()),
                        }
                        for v in variants()
                    },
                }
        reports[split] = {
            "utility": scores,
            "gates": gates,
            "factor_contrasts": factor,
            "cohorts": cohorts,
        }
        training.interface.save_arrays(out / f"{split}_predictions.npz", arrays)
    # No research-picked winner; all predeclared arms retain their own verdict.
    decision = {
        arm: {
            k: all(
                reports[s]["gates"][f"{arm}_s{seed}_{kind}"][k]
                for s in SPLITS
                for seed in (42, 43)
                for kind in ("best", "last")
            )
            for k in ("reconstruction_progress", "representation_progress")
        }
        for arm in ARMS_FOR_REPORT
    }
    atomic_json(reports, out / "rs_metrics.json")
    atomic_json(
        {
            "arms": decision,
            "promoted": False,
            "scope": "Repeated research cohorts, finite matched adaptation; no global causal attribution or final independent acceptance",
        },
        out / "decision.json",
    )


# All intervention arms are declared before research evaluation.
ARMS_FOR_REPORT = tuple(training.ARMS[1:])
