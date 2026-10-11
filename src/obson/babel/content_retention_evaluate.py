"""Locked-last F03 evaluation; raw physical truth, paired controls, no fitting."""

import gc
from pathlib import Path

import numpy as np
import torch

from . import content_retention as core
from . import content_retention_data as data
from . import history_price_truth as truth
from . import overlap_content_metrics as overlap
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress


def variants():
    return [f"{a}_s{s}" for s in core.SEEDS for a in ("source", *core.ARMS)]


def verify_lock(out):
    from . import content_retention_run as run

    lock = read_json(out / "model_lock.json")
    if lock["manifest"] != sha256(out / "manifest.json") or set(lock["records"]) != {
        j["name"] for j in run.jobs()
    }:
        raise ValueError("Complete matched model lock required before evaluation")
    data.source.source.old.verify_files(out, lock["files"])
    return lock


def load(meta, out, name, device):
    from . import content_retention_run as run

    seed = int(name.split("_s")[1])
    if name.startswith("source_"):
        return data.load_source(meta, seed, device)
    job = next(j for j in run.jobs() if j["name"] == name)
    ck = torch.load(out / name / "last.pt", map_location="cpu", weights_only=True)
    if ck["binding"] != run.binding(out, job) or ck["epoch"] != run.EPOCHS:
        raise ValueError("Expected fixed final checkpoint, not best/partial")
    e, q = core.source.make_models("rolling", seed, device, meta["coverage_meta"]["encoder"])
    e.load_state_dict(ck["encoder"], strict=True)
    q.load_state_dict(ck["query"], strict=True)
    return e.eval().requires_grad_(False), q.eval().requires_grad_(False)


@torch.no_grad()
def extract(meta, out, name, split, x, device):
    from . import content_retention_run as run

    folder = out / "predictions"
    folder.mkdir(exist_ok=True)
    bind = {
        "model_lock": sha256(out / "model_lock.json"),
        "source_index": meta["coverage_meta"]["source_indexes"][split],
        "name": name,
        "split": split,
    }
    path = folder / f"{split}_{name}.npz"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        saved = read_json(receipt)
        if saved["binding"] != bind or saved["sha256"] != sha256(path):
            raise ValueError("Evaluation output binding changed")
        with np.load(path, allow_pickle=False) as f:
            return dict(f)
    ledger_path = out / "evaluation_attempts.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {}
    # Reserve one whole model/cohort attempt before any forward. Interrupted
    # uncommitted extraction is export-only; no hidden extra inference loops.
    key = f"{split}/{name}"
    if key in ledger:
        raise ValueError("Uncommitted evaluation attempt; export only")
    ledger[key] = {"endpoints": len(x) * len(core.POSITIONS), "complete": False}
    atomic_json(ledger, ledger_path)
    e, q = load(meta, out, name, device)
    before = run.base.signature(e, q)
    queries, states = [], []
    for left in range(0, len(x), meta["micro"]):
        b = torch.as_tensor(x[left : left + meta["micro"]], device=device)
        ps = torch.tensor(core.POSITIONS, device=device).expand(len(b), -1)
        preds, zs = [], []
        for j in range(0, len(core.POSITIONS), 2):
            z, pred = core.source.predict(e, q, b, ps[:, j : j + 2])
            preds.append(pred.cpu().numpy())
            zs.append(z.cpu().numpy())
        queries.append(np.concatenate(preds, axis=1))
        states.append(np.concatenate(zs, axis=1)[:, -1])
    result = {"query": np.concatenate(queries), "endpoint_state": np.concatenate(states)}
    if not all(np.isfinite(v).all() for v in result.values()) or run.base.signature(e, q) != before:
        raise ValueError("Nonfinite output or changed frozen model")
    if name.startswith("source_"):
        with np.load(
            Path(meta["source"]) / f"{split}_D_s{name.split('_s')[1]}.npz", allow_pickle=False
        ) as old:
            ix = [core.POSITIONS.index(p) for p in overlap.POSITIONS]
            checks = {
                "query": run.r0.compare(result["query"][:, ix], old["query"], 1e-4, 2e-4),
                "state": run.r0.compare(
                    result["endpoint_state"], old["endpoint_state"], 1e-4, 2e-4
                ),
            }
        run.r0.save_checks(out, f"{split}_{name}_replay", checks)
    data.save_arrays(path, result)
    atomic_json({"binding": bind, "sha256": sha256(path), "frozen_signature": before}, receipt)
    ledger[key]["complete"] = True
    atomic_json(ledger, ledger_path)
    del e, q
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def targets(x, raw, stats):
    ps = torch.tensor(core.POSITIONS).expand(len(x), -1)
    y, m = core.source.targets(torch.as_tensor(x), ps, stats)
    prices = np.stack([raw[:, p - 1 : p + 127] for p in core.POSITIONS], axis=1)
    return {"target": y.numpy(), "mask": m.numpy(), "raw_close": prices}


def errors(pred, label, stats):
    n = len(pred)
    ix = [core.POSITIONS.index(p) for p in overlap.POSITIONS]
    y, m, raw = (label[k] for k in ("target", "mask", "raw_close"))
    bank = overlap.paired_rows(pred[:, ix], y[:, ix], m[:, ix], raw[:, ix], stats)
    original = core.source.band_rows(
        torch.from_numpy(pred), torch.from_numpy(y), torch.from_numpy(m), stats
    )
    for k, v in original.items():
        for j, pos in enumerate(core.POSITIONS):
            bank[f"original/p{pos}/{k}"] = v[:, j].numpy()
    paths = overlap.close_path(pred.reshape(-1, 127, 7), stats)
    estimated = truth.summarize_path(paths)["descriptors"].reshape(n, len(core.POSITIONS), 6)
    observed = truth.from_raw_closes(raw.reshape(-1, 128))["descriptors"].reshape(
        n, len(core.POSITIONS), 6
    )
    for j, pos in enumerate(core.POSITIONS):
        for k, name in ((0, "efficiency16"), (2, "logrms16"), (3, "efficiency64"), (5, "logrms64")):
            bank[f"descriptor/p{pos}/{name}"] = (estimated[:, j, k] - observed[:, j, k]) ** 2
    return bank


def protection_keys():
    keys = []
    for p in core.POSITIONS:
        keys.extend(
            f"original/p{p}/{k}"
            for k in (
                "price",
                "activity",
                "near",
                "mid",
                "far",
                "near_activity",
                "mid_activity",
                "far_activity",
            )
        )
        keys.extend(
            f"descriptor/p{p}/{k}" for k in ("efficiency16", "logrms16", "efficiency64", "logrms64")
        )
    for d in (1, 16, 64):
        keys.extend(
            f"shift{d}/price/{k}_mse"
            for k in (
                "shape_a",
                "shape_b",
                "common_anchored_a",
                "common_anchored_b",
                "entered_b",
                "leaving_a",
                "disagreement",
            )
        )
        keys.extend(
            f"shift{d}/{f}/{r}_mse" for f in overlap.FIELDS[1:] for r in ("entered_b", "leaving_a")
        )
    return keys


def comparison(candidate, reference, rows, primary, gain):
    guards = protection_keys()
    specifications = [("gain/" + k, k, 1 - gain) for k in primary] + [
        ("retain/" + k, k, 1.1) for k in guards
    ]
    for _, key, _ in specifications:
        a, b = np.asarray(candidate[key]), np.asarray(reference[key])
        if (
            a.shape != (len(rows),)
            or b.shape != a.shape
            or not np.array_equal(np.isfinite(a), np.isfinite(b))
        ):
            raise ValueError("Paired comparison population/support changed: " + key)
    matrix = np.column_stack(
        [candidate[k] - factor * reference[k] for _, k, factor in specifications]
    )
    intervals = overlap.intervals(matrix, rows)
    checks = {}
    for i, (label, key, factor) in enumerate(specifications):
        values = {kind: ci[i] for kind, ci in intervals.items()}
        valid = np.isfinite(candidate[key]) & np.isfinite(reference[key])
        a, b = np.asarray(candidate[key])[valid], np.asarray(reference[key])[valid]
        checks[label] = {
            "ratio_of_means": float(a.mean() / max(b.mean(), 1e-20)) if len(a) else None,
            "threshold": factor,
            "intervals": values,
            "passed": all(v["supported"] and v["high"] <= 0 for v in values.values()),
        }
    tails = {}
    for name in ("price", "activity"):
        a = np.column_stack([candidate[f"original/p{p}/{name}"] for p in core.POSITIONS]).mean(1)
        b = np.column_stack([reference[f"original/p{p}/{name}"] for p in core.POSITIONS]).mean(1)
        ratio = float(np.quantile(a, 0.95) / max(np.quantile(b, 0.95), 1e-20))
        tails[name] = {
            "parent_mean_p95_ratio": ratio,
            "passed": ratio <= 1.1,
            "scope": "descriptive point guard, not tail significance",
        }
    return {
        "checks": checks,
        "tails": tails,
        "gain_passed": all(v["passed"] for k, v in checks.items() if k.startswith("gain/")),
        "protection_passed": all(v["passed"] for k, v in checks.items() if k.startswith("retain/"))
        and all(v["passed"] for v in tails.values()),
        "passed": all(v["passed"] for v in checks.values())
        and all(v["passed"] for v in tails.values()),
    }


def decisions(all_banks, row_bank):
    results = {}
    price_primary = [f"shift{d}/price/shape_b_mse" for d in (16, 64)]
    agreement_primary = [f"shift{d}/price/disagreement_mse" for d in (1, 16, 64)]
    for split, banks in all_banks.items():
        if split == "val":
            continue
        rows = row_bank[split]
        for seed in core.SEEDS:
            for arm, references, primary, gain in (
                ("content", ("control", "source"), price_primary, 0.05),
                ("aligned", ("content",), agreement_primary, 0.10),
                ("aligned", ("control", "source"), price_primary, 0.05),
            ):
                for ref in references:
                    name = f"{split}/{arm}_vs_{ref}/s{seed}"
                    results[name] = comparison(
                        banks[f"{arm}_s{seed}"], banks[f"{ref}_s{seed}"], rows, primary, gain
                    )
    content_ok = all(v["passed"] for k, v in results.items() if "/content_vs_" in k)
    aligned_ok = content_ok and all(v["passed"] for k, v in results.items() if "/aligned_vs_" in k)
    return {
        "comparisons": results,
        "content_supported": content_ok,
        "alignment_increment_supported": aligned_ok,
        "conclusion": "limited_content_candidate" if content_ok else "no_balanced_content_gain",
        "scope": "fixed finite recipe, repeated research cohorts, eight endpoint positions; no model promotion or full F11/current-field/all128 utility acceptance",
        "promoted": False,
        "fit_count": 0,
    }


def evaluate(meta, out, device):
    verify_lock(out)
    all_banks, row_bank = {}, {}
    for split in ("val", "test", "cross_research"):
        x, rows = data.cache(meta, split)
        row_bank[split] = rows
        labels = targets(x, data.closes(out, split), meta["statistics"])
        data.save_arrays(out / f"{split}_targets.npz", labels)
        banks = {}
        for name in variants():
            extracted = extract(meta, out, name, split, x, device)
            bank = errors(extracted["query"], labels, meta["statistics"])
            data.save_arrays(out / f"{split}_{name}_errors.npz", bank)
            atomic_json(overlap.summarize(bank, rows), out / f"{split}_{name}_metrics.json")
            banks[name] = bank
            progress(
                f"F03 {split}/{name}: eight causal endpoints, three overlaps, physical truth; fixed last"
            )
        all_banks[split] = banks
    result = decisions(all_banks, row_bank)
    atomic_json(result, out / "decision.json")
    return result
