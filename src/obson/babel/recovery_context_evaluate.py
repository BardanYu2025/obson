"""Locked V06 same-target contrasts; no fitted heads and no research selection."""

import gc

import numpy as np
import torch

from . import recovery_context_run as run
from . import recovery_rs_evaluate as paired_tools
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SPLITS = ("test", "cross_research")


def variants():
    return [
        {"name": f"source_s{s}", "seed": s, "job": None, "kind": "source"} for s in (42, 43)
    ] + [
        {"name": j["name"] + "_" + k, "seed": j["seed"], "job": j, "kind": k}
        for j in run.jobs()
        for k in ("best", "last")
    ]


def check_lock(out):
    lock = read_json(out / "model_selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Model selection manifest changed")
    expected = {f"{j['name']}/{k}.pt" for j in run.jobs() for k in ("best", "last")}
    if set(lock["files"]) != expected or set(lock["workers"]) != {j["name"] for j in run.jobs()}:
        raise ValueError("Incomplete six-arm model lock")
    for name, digest in lock["files"].items():
        run.r0.required_file(out, name, digest)
    return lock


def load(meta, out, variant, device):
    encoder, query = run.old.construct(meta, variant["seed"], device)
    if variant["job"] is not None:
        checkpoint = torch.load(
            out / variant["job"]["name"] / (variant["kind"] + ".pt"),
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint["binding"] != run.binding(out, variant["job"]):
            raise ValueError("Readout model binding changed")
        encoder.load_state_dict(checkpoint["encoder"], strict=True)
        query.load_state_dict(checkpoint["query"], strict=True)
    return encoder.eval().requires_grad_(False), query.eval().requires_grad_(False)


def summaries(values, support):
    result = {}
    for k, a in values.items():
        if a.shape != support[k].shape or not np.isfinite(a).all():
            raise ValueError("Invalid evaluation values or masks")
        result[k] = []
        for i, prefix in enumerate(run.old.core.PREFIXES):
            v = a[:, i][support[k][:, i]]
            result[k].append(
                {
                    "prefix": prefix,
                    "supported_windows": len(v),
                    "mean": float(v.mean()) if len(v) else None,
                    "p50": float(np.quantile(v, 0.5)) if len(v) else None,
                    "p95": float(np.quantile(v, 0.95)) if len(v) else None,
                }
            )
    return result


def extract(meta, context, out, split, device):
    check_lock(out)  # Before opening any research input.
    x, rows = run.old.data.cache(context, split)
    folder = out / "evaluation"
    folder.mkdir(exist_ok=True)
    binding = {
        "model_lock_sha256": sha256(out / "model_selection_lock.json"),
        "input_index_sha256": sha256(context / f"cache/{split}_index.json"),
    }
    stats = run.old.parent.stats(meta)[0]
    bank, common_support, description = {}, None, {}
    for variant in variants():
        stem = split + "_" + variant["name"]
        receipt, path = folder / (stem + ".json"), folder / (stem + ".npz")
        if receipt.exists():
            saved = read_json(receipt)
            if saved["binding"] != binding or saved["variant"] != variant:
                raise ValueError("Saved evaluation binding changed")
            run.r0.required_file(folder, path.name, saved["sha256"])
            with np.load(path, allow_pickle=False) as data:
                values = {k[6:]: data[k] for k in data.files if k.startswith("error/")}
                support = {k[8:]: data[k] for k in data.files if k.startswith("support/")}
        else:
            e, q = load(meta, out, variant, device)
            values, support = run.core.evaluate(e, q, x, stats, meta["micro"], device)
            run.interface.save_arrays(
                path,
                {"error/" + k: v for k, v in values.items()}
                | {"support/" + k: v for k, v in support.items()},
            )
            atomic_json({"binding": binding, "variant": variant, "sha256": sha256(path)}, receipt)
            del e, q
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        if common_support is not None and (
            support.keys() != common_support.keys()
            or any(not np.array_equal(v, common_support[k]) for k, v in support.items())
        ):
            raise ValueError("Models were scored on different physical target supports")
        common_support = support
        bank[variant["name"]] = values
        description[variant["name"]] = summaries(values, support)
        progress(f"V06 {split}: {variant['name']} evaluated on three policies")
    atomic_json(rows, out / f"{split}_rows.json")
    atomic_json(description, out / f"{split}_summary.json")
    return bank, common_support, rows


def contrast(a, b, support_a, support_b, rows):
    if not np.array_equal(support_a, support_b):
        raise ValueError("A paired claim requires identical supported physical targets")
    keep = support_a.all(1)
    selected = [r for r, ok in zip(rows, keep, strict=True) if ok]
    if not selected:
        return {"status": "no_common_support", "windows": 0}
    av, bv = a[keep], b[keep]
    ci = paired_tools.paired(av, bv, selected)
    five = paired_tools.paired(av, 0.95 * bv, selected)
    # Inferential intervals are paired clusters, not independent overlapping windows.
    return {
        "status": "estimated",
        "windows": len(selected),
        "candidate_mean": float(av.mean()),
        "control_mean": float(bv.mean()),
        "relative_change": float(av.mean() / max(float(bv.mean()), 1e-12) - 1),
        "intervals": ci,
        "five_percent_intervals": five,
        "supported_gain_5pct": bool(
            bv.mean() > 1e-12
            and av.mean() <= 0.95 * bv.mean()
            and all(c["supported"] and c["high"] <= 0 for c in five.values())
        ),
    }


def contrasts(bank, support, rows, seed, kind):
    a, b, c = [bank[f"{arm}_s{seed}_{kind}"] for arm in run.ARMS]
    source = bank[f"source_s{seed}"]
    names = list(run.ARMS)
    result = {}
    # All fixed-interface comparisons use the exact same input and score support.
    for view in names:
        for metric in run.old.core.METRICS:
            key = view + "/" + metric
            for pos, sl in [(str(p), slice(i, i + 1)) for i, p in enumerate(run.old.core.PREFIXES)]:
                for label, candidate, baseline in (("B_minus_A", b, a), ("C_minus_B", c, b)):
                    result[f"fixed/{view}/{pos}/{metric}/{label}"] = contrast(
                        candidate[key][:, sl],
                        baseline[key][:, sl],
                        support[key][:, sl],
                        support[key][:, sl],
                        rows,
                    )
    # Intended deployment: same physical targets, each uses its own training input policy.
    # Position128 is excluded from the primary: A and B inputs coincide there.
    for metric in run.old.core.METRICS:
        ka, kb = names[0] + "/" + metric, names[1] + "/" + metric
        sa, sb = support[ka][:, :3], support[kb][:, :3]
        result[f"deployed_common/interior/{metric}"] = contrast(
            b[kb][:, :3], a[ka][:, :3], sb, sa, rows
        )
        result[f"source_policy_gap/interior/{metric}"] = contrast(
            source[kb][:, :3], source[ka][:, :3], sb, sa, rows
        )
        result[f"difference_in_differences/interior/{metric}"] = contrast(
            b[kb][:, :3] - source[kb][:, :3], a[ka][:, :3] - source[ka][:, :3], sb, sa, rows
        )
        # Relative percentages of signed changes are not interpretable; report deltas only.
        did = result[f"difference_in_differences/interior/{metric}"]
        did.pop("relative_change", None)
        did.pop("five_percent_intervals", None)
        did.pop("supported_gain_5pct", None)
    return result


def retention(candidate, source, support):
    ratios = {}
    # Original24 interpretation: native/common and full/extended, endpoint/interior,6metrics.
    for view in (run.ARMS[0], run.ARMS[2]):
        for metric in run.old.core.METRICS:
            key = view + "/" + metric
            for context, sl in (("endpoint", slice(3, 4)), ("interior", slice(0, 3))):
                ok = support[key][:, sl]
                cv, sv = candidate[key][:, sl][ok], source[key][:, sl][ok]
                ratios[f"{view}/{context}/{metric}"] = (
                    float(cv.mean() / max(float(sv.mean()), 1e-12)) if len(cv) else None
                )
    return {"ratios": ratios, "passed": all(v is not None and v <= 1.10 for v in ratios.values())}


def run_evaluation(meta, context, out, device):
    lock = check_lock(out)
    report, decisions = {}, []
    for split in SPLITS:
        bank, support, rows = extract(meta, context, out, split, device)
        effects, preservation = {}, {}
        for seed in (42, 43):
            for kind in ("best", "last"):
                label = f"s{seed}_{kind}"
                effects[label] = contrasts(bank, support, rows, seed, kind)
                preservation[label] = {
                    arm: retention(bank[f"{arm}_s{seed}_{kind}"], bank[f"source_s{seed}"], support)
                    for arm in run.ARMS
                }
                if kind == "last":
                    primary = effects[label]["deployed_common/interior/price"]
                    common = run.ARMS[1] + "/price"
                    vs_source = contrast(
                        bank[f"{run.ARMS[1]}_s{seed}_last"][common][:, :3],
                        bank[f"source_s{seed}"][common][:, :3],
                        support[common][:, :3],
                        support[common][:, :3],
                        rows,
                    )
                    decisions.append(
                        {
                            "split": split,
                            "seed": seed,
                            "finite_B_benefit": bool(
                                primary.get("supported_gain_5pct")
                                and vs_source.get("supported_gain_5pct")
                                and preservation[label][run.ARMS[1]]["passed"]
                            ),
                            "source_common_comparison": vs_source,
                        }
                    )
        report[split] = {"contrasts": effects, "source_retention": preservation}
    check_lock(out)
    atomic_json(report, out / "context_metrics.json")
    atomic_json(
        {
            "task_id": "V06",
            "status": "controlled_study_requires_review",
            "promoted": False,
            "fixed_last_epoch": run.EPOCHS,
            "best_epochs": {k: v["best_epoch"] for k, v in lock["workers"].items()},
            "finite_B_benefit": all(d["finite_B_benefit"] for d in decisions),
            "strata": decisions,
            "scope": "Historical reconstruction, reused research cohorts, no new reader fits or future prediction. No claim that unknown information loss is resolved; C-B includes expanded support AND changed effective weights.",
            "conclusion_repair_complete": False,
        },
        out / "decision.json",
    )
