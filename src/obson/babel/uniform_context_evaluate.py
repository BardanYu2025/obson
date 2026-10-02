"""Locked historical reconstruction and frozen-state utility, with source references."""

import gc
import html

import numpy as np
import torch

from . import linear_history_run as parent
from . import uniform_context as core
from . import uniform_context_data as data
from . import utility_probe as up
from .ae_extend import atomic_json
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress


def variants(meta):
    from .uniform_context_run import jobs

    return [f"{j['name']}_{kind}" for j in jobs(meta) for kind in ("best", "last")] + [
        f"macro_s{s}" for s in meta["seeds"]
    ]


def load_model(meta, out, name, device):
    if name.startswith("macro_"):
        model, query = parent.construct(meta, int(name.split("_s")[1]), device)
        return model.core.encoder.eval().requires_grad_(False), query.eval().requires_grad_(False)
    mode, seed, kind = name.split("_")
    encoder, query = core.make_models(mode, int(seed[1:]), device, meta["encoder"])
    ck = torch.load(out / f"{mode}_{seed}/{kind}.pt", map_location="cpu", weights_only=True)
    from .uniform_context_run import binding

    if ck["metadata"] != binding(
        out, {"name": f"{mode}_{seed}", "mode": mode, "seed": int(seed[1:])}
    ):
        raise ValueError("Wrong evaluation checkpoint")
    encoder.load_state_dict(ck["encoder"], strict=True)
    query.load_state_dict(ck["query"], strict=True)
    return encoder.eval().requires_grad_(False), query.eval().requires_grad_(False)


def summary(errors):
    return {
        k: {
            "all": float(v.mean()),
            "endpoint": float(v[:, -1].mean()),
            "interior": float(v[:, :-1].mean()),
            "by_prefix": [float(a) for a in v.mean(0)],
        }
        for k, v in errors.items()
    }


@torch.no_grad()
@torch.no_grad()
def chunk_audit(encoder, x):
    ps = torch.tensor([32, 64, 96, 128], device=x.device).expand(len(x), -1)
    actual = core.read_states(encoder, x, ps)
    rolling = encoder(core.rolling_windows(x, ps).flatten(0, 1))[:, -1].reshape_as(actual)
    changed = x.clone()
    changed[:, 190:] += 3.0
    # p64 ends at190: index190 itself is current; test p32 endpoint158 only.
    before = encoder(x)[:, 158]
    after = encoder(changed)[:, 158]
    diff = (actual - rolling).abs()
    return {
        "chunk_max_abs": float(diff.max()),
        "chunk_rms": float(diff.square().mean().sqrt()),
        "future_max_abs": float((before - after).abs().max()),
        "passed": bool(
            torch.allclose(actual, rolling, atol=1e-4, rtol=2e-4)
            and torch.allclose(before, after, atol=1e-5, rtol=2e-5)
        ),
    }


def extracted(out, split):
    folder = out / "cache"
    index = read_json(folder / f"{split}_states_index.json")
    if index["model_lock_sha256"] != sha256(out / "model_selection_lock.json") or index[
        "input_index_sha256"
    ] != sha256(folder / f"{split}_index.json"):
        raise ValueError("Extracted states changed lineage")
    verify_files(folder, index["files"])
    with np.load(folder / f"{split}_states.npz", allow_pickle=False) as f:
        return dict(f)


@torch.no_grad()
def extract(meta, out, split, device):
    from .uniform_context_run import check_selection

    check_selection(out)
    if split in data.SPLITS[2:]:
        check_readouts(out)
    if (out / f"cache/{split}_states_index.json").exists():
        return extracted(out, split)
    x, rows = data.cache(out, split)
    stats = parent.stats(meta)[0]
    research = split in data.SPLITS[2:]
    prefixes = [32, 64, 96, 128] if research else [128]
    result = {}
    reports = {}
    examples = {}
    audits = {}
    for name in variants(meta):
        encoder, query = load_model(meta, out, name, device)
        states = []
        errors = {}
        example_predictions = []
        for start in range(0, len(x), meta["micro"]):
            b = torch.tensor(x[start : start + meta["micro"]], device=device)
            ps = torch.tensor(prefixes, device=device).expand(len(b), -1)
            if isinstance(encoder, core.Encoder):
                z = core.read_states(encoder, b, ps)
            else:
                z = encoder(core.rolling_windows(b, ps).flatten(0, 1))[:, -1].reshape(
                    len(b), len(prefixes), -1
                )
            states.append(z[:, -1].cpu().numpy())
            if research:
                pred = query(z.flatten(0, 1)).reshape(len(b), len(prefixes), 127, 7)
                y, mask = core.targets(b, ps, stats)
                for k, v in core.band_rows(pred, y, mask, stats).items():
                    errors.setdefault(k, []).append(v.cpu().numpy())
                if start == 0:
                    example_predictions = pred[:3].cpu().numpy()
                    examples["target"] = y[:3].cpu().numpy()
                    examples["mask"] = mask[:3].cpu().numpy()
            if start == 0 and name.startswith("relative_"):
                audits[name] = chunk_audit(encoder, b[:2])
                atomic_json(audits, out / f"{split}_chunk_audit.json")
                if not audits[name]["passed"]:
                    raise ValueError(
                        "Relative attention chunk/future invariance failed; numeric diagnostic saved"
                    )
        result[name] = np.concatenate(states)
        if research:
            values = {k: np.concatenate(v) for k, v in errors.items()}
            reports[name] = summary(values)
            for k, v in values.items():
                result[f"error/{name}/{k}"] = v
            examples[name] = example_predictions
        del encoder, query, z, b, ps
        gc.collect()
        torch.cuda.empty_cache()
        progress(f"{split}: {name} extracted on {len(x)} common endpoints")
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite extracted state/error")
    folder = out / "cache"
    np.savez(folder / f"{split}_states.tmp.npz", **result)
    (folder / f"{split}_states.tmp.npz").replace(folder / f"{split}_states.npz")
    names = [f"{split}_states.npz"]
    if research:
        atomic_json(reports, out / f"{split}_reconstruction.json")
        np.savez_compressed(out / f"{split}_examples.npz", **examples)
    atomic_json(
        {
            "model_lock_sha256": sha256(out / "model_selection_lock.json"),
            "input_index_sha256": sha256(folder / f"{split}_index.json"),
            "files": {n: sha256(folder / n) for n in names},
        },
        folder / f"{split}_states_index.json",
    )
    return result


def check_readouts(out):
    from .uniform_context_run import check_selection

    check_selection(out)
    lock = read_json(out / "readout_lock.json")
    if lock["model_lock_sha256"] != sha256(out / "model_selection_lock.json"):
        raise ValueError("Readout model identity changed")
    verify_files(out, lock["files"])
    for split, h in lock["states_indexes"].items():
        if sha256(out / f"cache/{split}_states_index.json") != h:
            raise ValueError("Readout state inputs changed")
    return lock


def representation(name, x, states, pca, raw_stats):
    if name == "raw3584":
        return x[:, -128:].reshape(len(x), -1)
    if name == "current28":
        return x[:, -1]
    if name == "pca768":
        return up.probe.normalize(x[:, -128:].reshape(len(x), -1), raw_stats) @ pca
    return states[name]


def fit(meta, out, device):
    if (out / "readout_lock.json").exists():
        check_readouts(out)
        return
    train, val = data.cache(out, "train")[0], data.cache(out, "val")[0]
    ts, vs = extract(meta, out, "train", device), extract(meta, out, "val", device)
    stats = parent.stats(meta)[0]
    y, mask = up.targets(up.restore_raw(train[:, -128:], stats))
    vy, vm = up.targets(up.restore_raw(val[:, -128:], stats))
    ys = up.target_scales(y, mask)
    raw = train[:, -128:].reshape(len(train), -1)
    xs = up.probe.scales(raw)
    if min(raw.shape) <= 768:
        raise ValueError("Insufficient rows for fixed PCA768 reference")
    eigen = up.probe.eigensystem(up.probe.normalize(raw, xs), device)
    pca = eigen[1][:, -768:].flip(1).cpu().numpy()
    del eigen
    heads = {}
    weights = {"pca": pca}
    for name in ["raw3584", "current28", "pca768"] + variants(meta):
        a = representation(name, train, ts, pca, xs)
        b = representation(name, val, vs, pca, xs)
        head, w, intercept = up.fit_heads(a, y, mask, b, vy, vm, ys, device)
        heads[name] = head
        weights[name + "/weights"] = w
        weights[name + "/intercepts"] = intercept
        progress(f"{name}:13 masked readouts fitted on train, five alphas selected on val")
    np.savez(out / "readout_weights.npz", **weights)
    atomic_json({"target_stats": ys, "raw_stats": xs, "heads": heads}, out / "readout_fit.json")
    atomic_json(
        {
            "model_lock_sha256": sha256(out / "model_selection_lock.json"),
            "files": {n: sha256(out / n) for n in ("readout_weights.npz", "readout_fit.json")},
            "states_indexes": {
                s: sha256(out / f"cache/{s}_states_index.json") for s in ("train", "val")
            },
        },
        out / "readout_lock.json",
    )


def paired(a, b, rows, draws=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.shape != b.shape
        or len(a) != len(rows)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Matched finite paired errors required")
    d = (a - b).reshape(len(rows), -1).mean(1)
    months = np.array([r["month"] for r in rows])
    groups = [np.flatnonzero(months == v) for v in sorted(set(months))]
    ids = np.random.default_rng(20261004).integers(0, len(groups), (draws, len(groups)))
    sums = np.array([d[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    low, high = np.quantile(sums[ids].sum(1) / counts[ids].sum(1), [0.025, 0.975])
    return {
        "delta": float(d.mean()),
        "low": float(low),
        "high": float(high),
        "months": len(groups),
        "windows": len(rows),
        "supported": bool(len(groups) >= 6 and len(rows) >= 100),
        "scope": "Paired calendar months; reused research cohorts; no multiple-model winner search",
    }


def gates(meta, states, utility, rows):
    result = {}
    for seed in meta["seeds"]:
        candidate = f"relative_s{seed}_best"
        reference = f"rolling_s{seed}_best"
        checks = []
        for metric, factor in [
            ("price", 0.95),
            ("near", 1.10),
            ("mid", 1.10),
            ("far", 1.10),
            ("activity", 1.10),
        ]:
            ci = paired(
                states[f"error/{candidate}/{metric}"],
                factor * states[f"error/{reference}/{metric}"],
                rows,
            )
            checks.append(
                {
                    "metric": metric,
                    "factor": factor,
                    "interval": ci,
                    "passed": bool(ci["supported"] and ci["high"] <= 0),
                }
            )
        for context, index in (("endpoint", slice(-1, None)), ("interior", slice(None, -1))):
            for metric in ("price", "near", "mid", "far", "activity"):
                ci = paired(
                    states[f"error/{candidate}/{metric}"][:, index],
                    1.10 * states[f"error/{reference}/{metric}"][:, index],
                    rows,
                )
                checks.append(
                    {
                        "metric": f"{context}/{metric}",
                        "factor": 1.10,
                        "interval": ci,
                        "passed": bool(ci["supported"] and ci["high"] <= 0),
                    }
                )
        for group, factor in (("utility", 1.05), ("direction", 1.10), ("volatility", 1.10)):
            ci = paired(utility[candidate][group], factor * utility[reference][group], rows)
            checks.append(
                {
                    "metric": "frozen " + group,
                    "factor": factor,
                    "interval": ci,
                    "passed": bool(ci["supported"] and ci["high"] <= 0),
                }
            )
        result[str(seed)] = {
            "reference": reference,
            "candidate": candidate,
            "checks": checks,
            "passed": all(c["passed"] for c in checks),
        }
    return result


def visualize(out, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = ["test/42", "test/43", "cross/42", "cross/43"]
    for ax, metric in zip(axes.flat, ("price", "near", "far", "utility"), strict=True):
        points = []
        for split in ("test", "cross_research"):
            for seed in (42, 43):
                a = f"relative_s{seed}_best"
                b = f"rolling_s{seed}_best"
                if metric == "utility":
                    va = report[split]["utility"][a]["groups"][metric]
                    vb = report[split]["utility"][b]["groups"][metric]
                else:
                    va = report[split]["reconstruction"][a][metric]["all"]
                    vb = report[split]["reconstruction"][b][metric]["all"]
                points.append(100 * (va / max(vb, 1e-12) - 1))
        ax.bar(labels, points, color=["#176b87" if p < 0 else "#c4503e" for p in points])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(metric + " error vs matched rolling control")
        ax.set_ylabel("% change (lower is better)")
    fig.suptitle("Uniform-context study: reconstruction and frozen-state readout")
    fig.tight_layout()
    fig.savefig(out / "uniform_context.png", dpi=160)
    plt.close(fig)
    text = (
        '<!doctype html><meta charset="utf-8"><title>Uniform context study</title><h1>Uniform context768</h1><p>Below zero means lower error. These are historical representations, not future forecasts. Macro references have more pretraining. See JSON for uncertainty, endpoint/age breakdowns and all best/last checkpoints.</p><img style="max-width:100%" src="uniform_context.png"><pre>'
        + html.escape(str({s: report[s]["gates"] for s in report}))
        + "</pre>"
    )
    (out / "report.html").write_text(text)


def evaluate(meta, out, device):
    check_readouts(out)
    fit = read_json(out / "readout_fit.json")
    with np.load(out / "readout_weights.npz", allow_pickle=False) as f:
        weights = dict(f)
    report = {}
    for split in ("test", "cross_research"):
        states = extract(meta, out, split, device)
        x, rows = data.cache(out, split)
        y, mask = up.targets(up.restore_raw(x[:, -128:], parent.stats(meta)[0]))
        scores = {}
        errors = {}
        arrays = {"targets": y, "mask": mask}
        for name, heads in fit["heads"].items():
            xx = representation(name, x, states, weights["pca"], fit["raw_stats"])
            pred = up.predict(
                heads,
                weights[name + "/weights"],
                weights[name + "/intercepts"],
                xx,
                fit["target_stats"],
            )
            scores[name], errors[name] = up.measure(pred, y, mask, fit["target_stats"])
            arrays[name + "/predictions"] = pred
        for name in variants(meta):
            for metric in ("price", "near", "mid", "far", "activity", "structure"):
                arrays[f"error/{name}/{metric}"] = states[f"error/{name}/{metric}"]
        cohorts = {}
        for field in ("symbol", "period"):
            for value in sorted({str(r[field]) for r in rows}):
                ids = np.array([str(r[field]) == value for r in rows])
                cohorts[f"{field}/{value}"] = {
                    "windows": int(ids.sum()),
                    "reconstruction": {
                        name: float(states[f"error/{name}/price"][ids].mean())
                        for name in variants(meta)
                    },
                    "utility": {
                        name: float(err["utility"][ids].mean()) for name, err in errors.items()
                    },
                }
        comparisons = {}
        for seed in meta["seeds"]:
            for a, b in (
                (f"rolling_s{seed}_best", f"prefix_s{seed}_best"),
                (f"relative_s{seed}_best", f"macro_s{seed}"),
            ):
                comparisons[f"{a} vs {b}"] = {
                    "price": paired(states[f"error/{a}/price"], states[f"error/{b}/price"], rows),
                    "utility": paired(errors[a]["utility"], errors[b]["utility"], rows),
                }
        report[split] = {
            "reconstruction": read_json(out / f"{split}_reconstruction.json"),
            "utility": scores,
            "cohorts": cohorts,
            "gates": gates(meta, states, errors, rows),
            "secondary_comparisons": comparisons,
        }
        np.savez_compressed(out / f"{split}_predictions.npz", **arrays)
    atomic_json(report, out / "uniform_context_metrics.json")
    atomic_json(
        {
            "research_direction_supported": all(
                g["passed"] for r in report.values() for g in r["gates"].values()
            ),
            "promoted": False,
            "interpretation": "Fixed recipe screen only; unchanged Macro source remains reference. If unsupported stop this finite recipe, do not close all-position or cross-resolution branches.",
        },
        out / "decision.json",
    )
    visualize(out, report)
