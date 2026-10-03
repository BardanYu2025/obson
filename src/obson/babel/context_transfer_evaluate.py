"""Warm-start update/time controls, original-interface retention and frozen utility."""

import gc
import html

import numpy as np
import torch

from . import context_transfer as core
from . import context_transfer_data as data
from . import linear_history_run as parent
from . import uniform_context_evaluate as previous
from . import utility_probe as up
from .ae_extend import atomic_json
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress

paired = previous.paired
summary = previous.summary
representation = previous.representation


def variants(meta):
    return [
        f"{arm}_s{seed}_{budget}_{kind}"
        for seed in meta["seeds"]
        for arm, budget in (("rolling", "update"), ("prefix", "update"), ("prefix", "time"))
        for kind in ("best", "last")
    ] + [f"macro_s{s}" for s in meta["seeds"]]


def load_model(meta, out, name, device):
    from .context_transfer_run import binding, construct

    seed = int(name.split("_")[1][1:])
    encoder, query = construct(meta, seed, device)
    if not name.startswith("macro_"):
        mode, _, budget, kind = name.split("_")
        job = {"name": f"{mode}_s{seed}", "mode": mode, "seed": seed}
        ck = torch.load(
            out / job["name"] / f"{budget}_{kind}.pt", map_location="cpu", weights_only=True
        )
        if ck["metadata"] != binding(out, job):
            raise ValueError("Wrong evaluation checkpoint")
        encoder.load_state_dict(ck["encoder"], strict=True)
        query.load_state_dict(ck["query"], strict=True)
    return encoder.eval().requires_grad_(False), query.eval().requires_grad_(False)


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
    from .context_transfer_run import check_selection

    check_selection(out)
    if split in data.SPLITS[2:]:
        check_readouts(out)
    if (out / f"cache/{split}_states_index.json").exists():
        return extracted(out, split)
    x, rows = data.cache(out, split)
    stats = parent.stats(meta)[0]
    research = split in data.SPLITS[2:]
    prefixes = core.PREFIXES if research else (128,)
    result = {}
    reports = {}
    examples = {}
    for name in variants(meta):
        e, q = load_model(meta, out, name, device)
        states = []
        errors = {}
        for start in range(0, len(x), meta["micro"]):
            b = torch.tensor(x[start : start + meta["micro"]], device=device)
            ps = torch.tensor(prefixes, device=device).expand(len(b), -1)
            for view, native in (
                (("full", False), ("native", True)) if research else (("full", False),)
            ):
                z, pred = core.predict(e, q, b, ps, native)
                if not native:
                    states.append(z[:, -1].cpu().numpy())
                if research:
                    y, mask = core.original.targets(b, ps, stats, native)
                    for k, v in core.original.band_rows(pred, y, mask, stats).items():
                        if k in core.METRICS:
                            errors.setdefault(view + "/" + k, []).append(v.cpu().numpy())
                    if start == 0:
                        examples[f"{view}/target"] = y[:3].cpu().numpy()
                        examples[f"{view}/mask"] = mask[:3].cpu().numpy()
                        examples[f"{name}/{view}"] = pred[:3].cpu().numpy()
        result[name] = np.concatenate(states)
        if research:
            values = {k: np.concatenate(v) for k, v in errors.items()}
            reports[name] = summary(values)
            result.update({f"error/{name}/{k}": v for k, v in values.items()})
        del e, q, z, pred, b, ps
        gc.collect()
        torch.cuda.empty_cache()
        progress(f"{split}: {name}, {len(x)} common endpoints")
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite extracted state or error")
    folder = out / "cache"
    np.savez(folder / f"{split}_states.tmp.npz", **result)
    (folder / f"{split}_states.tmp.npz").replace(folder / f"{split}_states.npz")
    if research:
        atomic_json(reports, out / f"{split}_reconstruction.json")
        np.savez_compressed(out / f"{split}_examples.npz", **examples)
    atomic_json(
        {
            "model_lock_sha256": sha256(out / "model_selection_lock.json"),
            "input_index_sha256": sha256(folder / f"{split}_index.json"),
            "files": {f"{split}_states.npz": sha256(folder / f"{split}_states.npz")},
        },
        folder / f"{split}_states_index.json",
    )
    return result


def check_readouts(out):
    from .context_transfer_run import check_selection

    check_selection(out)
    lock = read_json(out / "readout_lock.json")
    if lock["model_lock_sha256"] != sha256(out / "model_selection_lock.json"):
        raise ValueError("Readout model identity changed")
    verify_files(out, lock["files"])
    for split, h in lock["states_indexes"].items():
        if sha256(out / f"cache/{split}_states_index.json") != h:
            raise ValueError("Readout state inputs changed")
    return lock


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


def gates(meta, states, utility, rows, time_controls):
    result = {}
    for seed in meta["seeds"]:
        candidate = f"rolling_s{seed}_update_best"
        checks = []
        utility_checks = []
        refs = (f"prefix_s{seed}_update_best", f"prefix_s{seed}_time_best", f"macro_s{seed}")

        def check(a, b, metric, reference, factor):
            ci = paired(a, factor * b, rows)
            return {
                "metric": metric,
                "reference": reference,
                "factor": factor,
                "interval": ci,
                "passed": bool(ci["supported"] and ci["high"] <= 0),
            }

        for ref in refs:
            checks.append(
                check(
                    states[f"error/{candidate}/full/price"],
                    states[f"error/{ref}/full/price"],
                    "full/price",
                    ref,
                    0.95,
                )
            )
            for view in ("full", "native"):
                for context, index in (
                    ("endpoint", slice(-1, None)),
                    ("interior", slice(None, -1)),
                ):
                    for metric in core.METRICS:
                        k = view + "/" + metric
                        checks.append(
                            check(
                                states[f"error/{candidate}/{k}"][:, index],
                                states[f"error/{ref}/{k}"][:, index],
                                view + "/" + context + "/" + metric,
                                ref,
                                1.10,
                            )
                        )
            for group, factor in (("utility", 0.95), ("direction", 1.10), ("volatility", 1.10)):
                utility_checks.append(
                    check(
                        utility[candidate][group],
                        utility[ref][group],
                        "frozen/" + group,
                        ref,
                        factor,
                    )
                )
        matched = time_controls[str(seed)]["matched"]
        reconstruction = matched and all(c["passed"] for c in checks)
        result[str(seed)] = {
            "candidate": candidate,
            "time_control": time_controls[str(seed)],
            "reconstruction_progress": reconstruction,
            "representation_upgrade": bool(
                reconstruction and all(c["passed"] for c in utility_checks)
            ),
            "reconstruction_checks": checks,
            "utility_checks": utility_checks,
        }
    return result


def visualize(out, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["test/42", "test/43", "cross/42", "cross/43"]
    for ax, metric in zip(axes, ("full/price", "utility"), strict=True):
        for offset, budget in ((-0.2, "update"), (0.2, "time")):
            values = []
            for split in ("test", "cross_research"):
                for seed in (42, 43):
                    a = f"rolling_s{seed}_update_best"
                    b = f"prefix_s{seed}_{budget}_best"
                    if metric == "utility":
                        va = report[split]["utility"][a]["groups"][metric]
                        vb = report[split]["utility"][b]["groups"][metric]
                    else:
                        va = report[split]["reconstruction"][a][metric]["all"]
                        vb = report[split]["reconstruction"][b][metric]["all"]
                    values.append(100 * (va / max(vb, 1e-12) - 1))
            ax.bar(np.arange(4) + offset, values, width=0.38, label="vs prefix " + budget)
        ax.set_xticks(np.arange(4), labels)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Error % change (lower is better)")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(
        "Warm full-context study; inspect timing qualification and original-interface retention"
    )
    fig.tight_layout()
    fig.savefig(out / "context_transfer.png", dpi=160)
    plt.close(fig)
    (out / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Context transfer768</h1><p>Historical representation, not future prediction. A failed time-match check prevents compute-efficiency claims.</p><img style="max-width:100%" src="context_transfer.png"><pre>'
        + html.escape(str({s: report[s]["gates"] for s in report}))
        + "</pre>"
    )


def evaluate(meta, out, device):
    check_readouts(out)
    fit = read_json(out / "readout_fit.json")
    time_controls = read_json(out / "model_selection_lock.json")["time_controls"]
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
            for view in ("full", "native"):
                for metric in core.METRICS:
                    arrays[f"error/{name}/{view}/{metric}"] = states[
                        f"error/{name}/{view}/{metric}"
                    ]
        cohorts = {}
        for field in ("symbol", "period"):
            for value in sorted({str(r[field]) for r in rows}):
                ids = np.array([str(r[field]) == value for r in rows])
                cohorts[f"{field}/{value}"] = {
                    "windows": int(ids.sum()),
                    "reconstruction": {
                        name: float(states[f"error/{name}/full/price"][ids].mean())
                        for name in variants(meta)
                    },
                    "utility": {
                        name: float(err["utility"][ids].mean()) for name, err in errors.items()
                    },
                }
        report[split] = {
            "reconstruction": read_json(out / f"{split}_reconstruction.json"),
            "utility": scores,
            "cohorts": cohorts,
            "gates": gates(meta, states, errors, rows, time_controls),
        }
        np.savez_compressed(out / f"{split}_predictions.npz", **arrays)
    atomic_json(report, out / "context_transfer_metrics.json")
    atomic_json(
        {
            "reconstruction_progress": all(
                g["reconstruction_progress"] for r in report.values() for g in r["gates"].values()
            ),
            "representation_upgrade": all(
                g["representation_upgrade"] for r in report.values() for g in r["gates"].values()
            ),
            "promoted": False,
            "interpretation": "No automatic promotion. Time-budget mismatch or failed original-interface retention is not a supported upgrade. Dense-all-position and future branches remain open.",
        },
        out / "decision.json",
    )
    visualize(out, report)
