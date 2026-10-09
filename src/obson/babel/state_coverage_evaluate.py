"""F11: fixed-last evaluation, same readout opportunity, no research winner selection."""

import gc
import html

import numpy as np
import torch

from . import state_coverage as core
from . import state_coverage_data as data
from . import utility_probe as up
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress


def variants():
    return [f"{a}_s{s}" for a in core.ARMS for s in (42, 43)] + ["macro_s42", "macro_s43"]


def check_selection(out):
    from . import state_coverage_run as run

    lock = read_json(out / "model_selection_lock.json")
    expected = {f"{j['name']}/last.pt" for j in run.jobs()}
    if (
        lock["manifest_sha256"] != sha256(out / "manifest.json")
        or set(lock["files"]) != expected
        or set(lock["workers"]) != {j["name"] for j in run.jobs()}
    ):
        raise ValueError("Incomplete/changed fixed-last lock")
    for worker in lock["workers"].values():
        if worker["epochs"] != run.EPOCHS or worker["optimizer_updates"] != 38 * run.EPOCHS:
            raise ValueError("Unmatched final budgets")
    data.source.old.verify_files(out, lock["files"])
    return lock


def check_readouts(out):
    check_selection(out)
    lock = read_json(out / "readout_lock.json")
    if lock["model_lock_sha256"] != sha256(out / "model_selection_lock.json"):
        raise ValueError("Readout models changed")
    data.source.old.verify_files(out, lock["files"])
    for s, h in lock["states_indexes"].items():
        if sha256(out / f"cache/{s}_states_index.json") != h:
            raise ValueError("Readout extraction changed")
    return lock


def load(meta, out, name, device):
    from . import state_coverage_run as run

    seed = int(name.split("_s")[1])
    if name.startswith("macro_"):
        e, q = data.source.old.construct(meta["source_meta"], seed, device)
    else:
        arm = name.split("_")[0]
        e, q = core.make_models("rolling", seed, device, meta["encoder"])
        ck = torch.load(out / name / "last.pt", map_location="cpu", weights_only=True)
        if (
            ck["binding"] != run.binding(out, {"name": name, "arm": arm, "seed": seed})
            or ck["epoch"] != run.EPOCHS
        ):
            raise ValueError("Wrong final checkpoint")
        e.load_state_dict(ck["encoder"], strict=True)
        q.load_state_dict(ck["query"], strict=True)
    return e.eval().requires_grad_(False), q.eval().requires_grad_(False)


def save_arrays(path, bank):
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp, **bank)
    temp.replace(path)


@torch.no_grad()
def extract(meta, out, split, device):
    from . import state_coverage_run as run

    check_selection(out)
    if split in data.SPLITS[2:]:
        check_readouts(out)
    x, rows = data.cache(meta, out, split)
    folder = out / "cache"
    folder.mkdir(exist_ok=True)
    bind = {
        "model_lock_sha256": sha256(out / "model_selection_lock.json"),
        "input_index_sha256": meta["source_indexes"][split],
    }
    index = folder / f"{split}_states_index.json"
    if index.exists():
        report = read_json(index)
        if report["binding"] != bind:
            raise ValueError("Extraction binding changed")
        data.source.old.verify_files(folder, report["files"])
        with np.load(folder / f"{split}_states.npz", allow_pickle=False) as f:
            return dict(f)
    research = split in data.SPLITS[2:]
    bank = {}
    summaries = {}
    examples = {}
    for name in variants():
        model_path = folder / f"{split}_{name}.npz"
        receipt = folder / f"{split}_{name}.json"
        if receipt.exists():
            saved = read_json(receipt)
            if saved["binding"] != bind or saved["sha256"] != sha256(model_path):
                raise ValueError("Per-model extraction changed")
            with np.load(model_path, allow_pickle=False) as f:
                part = dict(f)
        else:
            e, q = load(meta, out, name, device)
            before = run.signature(e, q)
            audit = core.audit(e, q, torch.as_tensor(x[:2], device=device), meta["statistics"])
            atomic_json(audit, out / f"{split}_{name}_audit.json")
            if not audit["passed"]:
                raise ValueError("Frozen trained-state audit failed")
            zs = []
            errors = {}
            model_examples = {}
            for left in range(0, len(x), meta["micro"]):
                b = torch.as_tensor(x[left : left + meta["micro"]], device=device)
                per = {}
                positions = list(range(1, 129)) if research else [128]
                for start in range(0, len(positions), 5):
                    ps = torch.tensor(positions[start : start + 5], device=device).expand(
                        len(b), -1
                    )
                    z, pred = core.predict(e, q, b, ps)
                    if int(ps[0, -1]) == 128:
                        zs.append(z[:, -1].cpu().numpy())
                    if research:
                        y, m = core.targets(b, ps, meta["statistics"])
                        for k, v in core.band_rows(pred, y, m, meta["statistics"]).items():
                            per.setdefault(k, []).append(v.cpu().numpy())
                        if left == 0 and int(ps[0, -1]) == 128:
                            model_examples["example/prediction"] = pred[:2, -1].cpu().numpy()
                            model_examples["example/target"] = y[:2, -1].cpu().numpy()
                            model_examples["example/mask"] = m[:2, -1].cpu().numpy()
                for k, v in per.items():
                    errors.setdefault(k, []).append(np.concatenate(v, axis=1))
            if run.signature(e, q) != before:
                raise ValueError("Frozen weights/buffers changed")
            part = (
                {"states": np.concatenate(zs)}
                | {"error/" + k: np.concatenate(v) for k, v in errors.items()}
                | model_examples
            )
            if not all(np.isfinite(v).all() for v in part.values()):
                raise ValueError("Nonfinite extracted result")
            save_arrays(model_path, part)
            atomic_json({"binding": bind, "sha256": sha256(model_path)}, receipt)
            del e, q
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        bank[name] = part["states"]
        if research:
            examples[f"{name}/prediction"] = part["example/prediction"]
            examples["target"] = part["example/target"]
            examples["mask"] = part["example/mask"]
        for k, v in part.items():
            if k.startswith("error/"):
                metric = k[6:]
                bank[f"error/{name}/{metric}"] = v
                summaries.setdefault(name, {})[metric] = {
                    "mean": float(v.mean()),
                    "endpoint": float(v[:, -1].mean()),
                    "interior": float(v[:, :-1].mean()),
                    "by_position": v.mean(0).tolist(),
                    "window_p50": float(np.quantile(v.mean(1), 0.5)),
                    "window_p95": float(np.quantile(v.mean(1), 0.95)),
                }
        progress(
            f"F11 {split}: {name}, {len(rows)} parents, "
            + ("all128 history endpoints" if research else "p128 frozen states")
        )
    if research:
        save_arrays(out / f"{split}_examples.npz", examples)
        atomic_json(summaries, out / f"{split}_reconstruction.json")
    atomic_json(rows, out / f"{split}_rows.json")
    save_arrays(folder / f"{split}_states.npz", bank)
    atomic_json(
        {"binding": bind, "files": {f"{split}_states.npz": sha256(folder / f"{split}_states.npz")}},
        index,
    )
    return bank


def representation(name, x, states, pca, stats):
    raw = x[:, -128:].reshape(len(x), -1)
    if name == "raw3584":
        return raw
    if name == "current28":
        return x[:, -1]
    if name == "pca768":
        return up.probe.normalize(raw, stats) @ pca
    return states[name]


def fit(meta, out, device):
    if (out / "readout_lock.json").exists():
        check_readouts(out)
        return
    train = data.cache(meta, out, "train")[0]
    val = data.cache(meta, out, "val")[0]
    ts = extract(meta, out, "train", device)
    vs = extract(meta, out, "val", device)
    y, m = up.targets(up.restore_raw(train[:, -128:], meta["statistics"]))
    vy, vm = up.targets(up.restore_raw(val[:, -128:], meta["statistics"]))
    ys = up.target_scales(y, m)
    xs = up.probe.scales(train[:, -128:].reshape(len(train), -1))
    folder = out / "readouts"
    folder.mkdir(exist_ok=True)
    bind = {
        "model_lock": sha256(out / "model_selection_lock.json"),
        "train": sha256(out / "cache/train_states_index.json"),
        "val": sha256(out / "cache/val_states_index.json"),
    }
    if (folder / "pca.json").exists():
        saved = read_json(folder / "pca.json")
        if saved["binding"] != bind or saved["sha256"] != sha256(folder / "pca.npz"):
            raise ValueError("PCA lineage changed")
        with np.load(folder / "pca.npz") as f:
            pca = f["pca"]
    else:
        eigen = up.probe.eigensystem(
            up.probe.normalize(train[:, -128:].reshape(len(train), -1), xs), device
        )
        pca = eigen[1][:, -768:].flip(1).cpu().numpy()
        del eigen
        save_arrays(folder / "pca.npz", {"pca": pca})
        atomic_json({"binding": bind, "sha256": sha256(folder / "pca.npz")}, folder / "pca.json")
    heads = {}
    weights = {"pca": pca}
    vp = {"targets": vy, "mask": vm}
    for name in ["raw3584", "pca768", "current28"] + variants():
        path = folder / f"{name}.json"
        wpath = folder / f"{name}.npz"
        a = representation(name, train, ts, pca, xs)
        b = representation(name, val, vs, pca, xs)
        if path.exists():
            saved = read_json(path)
            if saved["binding"] != bind or saved["sha256"] != sha256(wpath):
                raise ValueError("Completed readout changed")
            head = saved["head"]
            with np.load(wpath) as f:
                w = f["weights"]
                intercept = f["intercepts"]
        else:
            # Pending fitting is never silently repeated under the finite search budget.
            pending = folder / f"{name}.pending"
            if pending.exists():
                raise ValueError("Interrupted uncommitted readout fit; export for review")
            pending.write_text("five alphas per each of thirteen targets")
            head, w, intercept = up.fit_heads(a, y, m, b, vy, vm, ys, device)
            save_arrays(wpath, {"weights": w, "intercepts": intercept})
            atomic_json({"binding": bind, "sha256": sha256(wpath), "head": head}, path)
            pending.unlink()
        if len(head) != 13 or any(len(h["candidates"]) != 5 for h in head):
            raise ValueError("Readout selection budget changed")
        heads[name] = head
        weights[name + "/weights"] = w
        weights[name + "/intercepts"] = intercept
        vp[name] = up.predict(head, w, intercept, b, ys)
        progress(f"F11 {name}: thirteen readouts, original five alpha opportunities")
    save_arrays(out / "readout_weights.npz", weights)
    save_arrays(out / "validation_predictions.npz", vp)
    atomic_json(
        {"heads": heads, "target_stats": ys, "raw_stats": xs, "fits": 845}, out / "readout_fit.json"
    )
    atomic_json(
        {
            "model_lock_sha256": sha256(out / "model_selection_lock.json"),
            "files": {
                p: sha256(out / p)
                for p in ("readout_weights.npz", "readout_fit.json", "validation_predictions.npz")
            },
            "states_indexes": {
                s: sha256(out / f"cache/{s}_states_index.json") for s in ("train", "val")
            },
        },
        out / "readout_lock.json",
    )


def paired(a, b, rows, quantile=None, draws=2000):
    a = np.asarray(a, dtype=float).reshape(len(rows), -1).mean(1)
    b = np.asarray(b, dtype=float).reshape(len(rows), -1).mean(1)
    if not len(rows) or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite matched parent rows required")
    result = {}
    for kind, labels in (
        ("month", [r["month"] for r in rows]),
        ("contract_month", [r["key"] + "/" + r["month"] for r in rows]),
    ):
        _, codes = np.unique(labels, return_inverse=True)
        n = int(codes.max()) + 1
        samples = np.random.default_rng(20261009).integers(0, n, (draws, n))
        counts = np.bincount(codes)
        if quantile is None:
            sums = np.bincount(codes, weights=a - b)
            values = sums[samples].sum(1) / counts[samples].sum(1)
            delta = float((a - b).mean())
        else:
            values = []
            for sample in samples:
                multiplicity = np.bincount(sample, minlength=n)[codes]
                values.append(
                    np.quantile(np.repeat(a, multiplicity), quantile)
                    - np.quantile(np.repeat(b, multiplicity), quantile)
                )
            values = np.asarray(values)
            delta = float(np.quantile(a, quantile) - np.quantile(b, quantile))
        low, high = np.quantile(values, [0.025, 0.975])
        result[kind] = {
            "delta": delta,
            "low": float(low),
            "high": float(high),
            "groups": n,
            "windows": len(rows),
            "supported": bool(n >= 6 and len(rows) >= 100),
        }
    return result


def gate(a, b, rows, factor=1.0, quantile=None):
    ci = paired(a, factor * np.asarray(b), rows, quantile)
    ref = float(np.mean(b))
    cand = float(np.mean(a))
    passed = all(v["supported"] and v["high"] <= 0 for v in ci.values())
    # Relative gain cannot be claimed against a zero/near-zero error reference.
    if factor < 1 and ref <= 1e-12:
        passed = False
    return {
        "factor": factor,
        "intervals": ci,
        "candidate_mean": cand,
        "reference_mean": ref,
        "relative_mean_change": cand / ref - 1 if ref > 1e-12 else None,
        "passed": bool(passed),
    }


def history_gates(bank, a, b, rows):
    checks = {}
    for view, sl in (("endpoint", slice(-1, None)), ("interior", slice(None, -1))):
        for metric in (
            "near",
            "mid",
            "far",
            "activity",
            "near_activity",
            "mid_activity",
            "far_activity",
        ):
            checks[view + "/" + metric] = gate(
                bank[f"error/{a}/{metric}"][:, sl], bank[f"error/{b}/{metric}"][:, sl], rows, 1.1
            )
    checks["window_price_p95"] = gate(
        bank[f"error/{a}/price"].mean(1), bank[f"error/{b}/price"].mean(1), rows, 1.1, quantile=0.95
    )
    return checks


def absolute(scores, errors, mask, rows, name):
    readable = all(
        scores[n]["targets"][i]["r2"] is not None and scores[n]["targets"][i]["r2"] >= 0.5
        for n in (name, "raw3584")
        for i in up.GROUPS["utility"]
    )
    checks = {}
    for metric, ref, factor in (
        ("utility", "raw3584", 1.05),
        ("direction", "raw3584", 1.1),
        ("volatility", "raw3584", 1.1),
        ("utility", "pca768", 0.95),
        ("utility", "current28", 0.95),
    ):
        c = gate(errors[name][metric], errors[ref][metric], rows, factor)
        c["passed"] &= readable
        checks[metric + "/" + ref] = c
    for i in range(6, 13):
        ids = np.flatnonzero(mask[:, i])
        rs = [rows[k] for k in ids]
        r2 = scores[name]["targets"][i]["r2"]
        raw_r2 = scores["raw3584"]["targets"][i]["r2"]
        if len(ids):
            c = gate(errors[name][up.NAMES[i]][ids], np.full(len(ids), 0.1), rs)
        else:
            c = {"passed": False, "reason": "no_observed_targets"}
        c["passed"] &= bool(
            len(ids) >= 50
            and len({r["week"] for r in rs}) >= 5
            and r2 is not None
            and r2 >= 0.8
            and raw_r2 is not None
            and raw_r2 > 0
        )
        c.update(r2=r2, raw_r2=raw_r2, support=len(ids))
        checks[up.NAMES[i]] = c
    return {
        "utility_readable": readable,
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
    }


def decisions(bank, scores, errors, mask, rows):
    result = {}
    for seed in (42, 43):
        a = f"A_s{seed}"
        d = f"D_s{seed}"
        macro = f"macro_s{seed}"
        progress_checks = history_gates(bank, d, a, rows)
        for metric, factor in (("utility", 0.95), ("direction", 1.1), ("volatility", 1.1)):
            progress_checks["frozen/" + metric] = gate(
                errors[d][metric], errors[a][metric], rows, factor
            )
        reference_checks = history_gates(bank, d, macro, rows)
        reference_checks["frozen/utility"] = gate(
            errors[d]["utility"], errors[macro]["utility"], rows, 1.05
        )
        ab = absolute(scores, errors, mask, rows, d)
        progress_ok = all(c["passed"] for c in progress_checks.values())
        result[str(seed)] = {
            "research_progress": progress_ok,
            "progress_checks": progress_checks,
            "macro_checks": reference_checks,
            "absolute": ab,
            "limited_candidate": progress_ok
            and ab["passed"]
            and all(c["passed"] for c in reference_checks.values()),
        }
    return result


def contrasts(bank, errors, rows):
    result = {}
    for seed in (42, 43):
        values = {
            a: {"utility": errors[f"{a}_s{seed}"]["utility"]}
            | {
                m: bank[f"error/{a}_s{seed}/{m}"].mean(1)
                for m in ("price", "near", "mid", "far", "activity", "structure")
            }
            for a in core.ARMS
        }
        for left, right in (("B", "A"), ("D", "C"), ("C", "A"), ("D", "B")):
            result[f"s{seed}/{left}-{right}"] = {
                m: paired(v, values[right][m], rows) for m, v in values[left].items()
            }
        result[f"s{seed}/interaction"] = {
            m: paired(
                values["D"][m] - values["B"][m] - values["C"][m] + values["A"][m],
                np.zeros(len(rows)),
                rows,
            )
            for m in values["A"]
        }
    return result


def evaluate(meta, out, device):
    from . import utility_probe_run as legacy_utility

    check_readouts(out)
    fitted = read_json(out / "readout_fit.json")
    with np.load(out / "readout_weights.npz", allow_pickle=False) as f:
        weights = dict(f)
    report = {}
    for split in ("test", "cross_research"):
        bank = extract(meta, out, split, device)
        x, rows = data.cache(meta, out, split)
        y, m = up.targets(up.restore_raw(x[:, -128:], meta["statistics"]))
        scores = {}
        errors = {}
        export = {"targets": y, "mask": m}
        for name, head in fitted["heads"].items():
            xx = representation(name, x, bank, weights["pca"], fitted["raw_stats"])
            pred = up.predict(
                head,
                weights[name + "/weights"],
                weights[name + "/intercepts"],
                xx,
                fitted["target_stats"],
            )
            scores[name], errors[name] = up.measure(pred, y, m, fitted["target_stats"])
            export[name + "/predictions"] = pred
        scores["train_mean"], _ = up.measure(
            np.tile(fitted["target_stats"]["mean"], (len(y), 1)), y, m, fitted["target_stats"]
        )
        for key, value in bank.items():
            if key.startswith("error/"):
                export[key] = value
        save_arrays(out / f"{split}_predictions.npz", export)
        cohorts = {}
        for field in ("key", "period"):
            for value in sorted({str(r[field]) for r in rows}):
                ids = np.array([str(r[field]) == value for r in rows])
                cohorts[field + "/" + value] = {
                    "windows": int(ids.sum()),
                    "utility": {
                        n: up.measure(
                            export[n + "/predictions"][ids], y[ids], m[ids], fitted["target_stats"]
                        )[0]
                        for n in fitted["heads"]
                    },
                    "price": {n: float(bank[f"error/{n}/price"][ids].mean()) for n in variants()},
                }
        # Old weekly rule is reported separately; it cannot replace the new monthly gate.
        old_config = {
            "models": [f"D_s{s}" for s in (42, 43)],
            "pca_rank": 768,
            "decision": {
                "minimum_utility_r2": 0.5,
                "raw_retention": 0.05,
                "family_retention": 0.1,
                "baseline_gain": 0.05,
                "current_nmse": 0.1,
                "current_r2": 0.8,
                "min_windows": 50,
                "min_weeks": 5,
            },
        }
        report[split] = {
            "scores": scores,
            "cohorts": cohorts,
            "decisions": decisions(bank, scores, errors, m, rows),
            "factor_contrasts": contrasts(bank, errors, rows),
            "legacy_weekly_diagnostic": legacy_utility.decide(old_config, scores, errors, m, rows),
        }
        atomic_json(report[split], out / f"{split}_metrics.json")
    decision = {
        "research_progress_supported": all(
            v["research_progress"] for r in report.values() for v in r["decisions"].values()
        ),
        "limited_candidate": all(
            v["limited_candidate"] for r in report.values() for v in r["decisions"].values()
        ),
        "promoted": False,
        "stop": "Finite matrix finished. No automatic extensions. V13 independent validation remains open.",
    }
    atomic_json(decision, out / "decision.json")
    (out / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>F01/F11</title><h1>F01/F11 fixed-last research report</h1><p>Historical information and frozen readouts; not a future prediction or final independent acceptance.</p><pre>'
        + html.escape(str(decision))
        + "</pre>"
    )
