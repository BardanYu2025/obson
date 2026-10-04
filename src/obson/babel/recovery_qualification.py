"""V02/V14: read-only gradient and source-reader qualification; no training."""

import argparse
import gc
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import recovery_audit as r0
from . import recovery_objectives as objectives
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

previous, hq = r0.previous, r0.hq
SCHEMA = "babel-recovery-qualification-v1"
# Completion from the reviewed R0v2 archive, not a mutable most recent audit.
AUDIT_COMPLETION = "aeb30e1d35240ce0d551df7362046f42ada8d89ab70fff12738d9bd158086a0d"
PREFIXES = (32, 64, 96, 128)


def evidence_binding(audit, source):
    done = read_json(audit / "completion.json")
    if sha256(audit / "completion.json") != AUDIT_COMPLETION:
        raise ValueError("Expected the reviewed R0v2 completion; do not silently substitute a run")
    if done["status"] != "audit_complete_requires_review" or done["optimizer_updates"] != 0:
        raise ValueError("Reviewed zero-update R0 required")
    for rel, digest in done["files"].items():
        r0.required_file(audit, rel, digest)
    meta = read_json(audit / "manifest.json")
    identity = read_json(audit / "identity.json")
    if (
        Path(meta["source"]).resolve() != source.resolve()
        or identity["manifest_sha256"] != sha256(source / "manifest.json")
        or identity["completion_sha256"] != sha256(source / "completion.json")
        or meta["historical_code"] != previous.code_identity()
        or meta["code_sha256"] != {Path(r0.__file__).name: sha256(r0.__file__)}
    ):
        raise ValueError("R0 source/code identity changed")
    return {"completion_sha256": sha256(audit / "completion.json"), "files": done["files"]}


def named_parameters(encoder, query):
    named = [
        (f"{group}.{n}", p)
        for group, module in (("encoder", encoder), ("query", query))
        for n, p in module.named_parameters()
        if p.requires_grad
    ]
    if not named or len({id(p) for _, p in named}) != len(named):
        raise ValueError("Nonempty unaliased encoder/query parameters required")
    return named


def gradients(loss, named):
    values = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    return [
        torch.zeros_like(p, device="cpu") if g is None else g.detach().cpu().clone()
        for (_, p), g in zip(named, values, strict=True)
    ]


def vector_comparison(actual, expected, named):
    """Check every coordinate; save per-parameter residuals and worst-coordinate witnesses.

    Accumulation/norm arithmetic is CPU float64. No change to FP32 CUDA backward.
    Full vectors are NOT exported; witnesses do not imply independent full replay.
    """
    if len(actual) != len(expected) or len(actual) != len(named):
        raise ValueError("Gradient parameter lists differ")
    records, sums, failures, elements = [], [0.0, 0.0, 0.0], 0, 0
    for a, b, (name, param) in zip(actual, expected, named, strict=True):
        if a.shape != b.shape or a.shape != param.shape:
            raise ValueError("Gradient shapes differ: " + name)
        a, b = a.double().flatten(), b.double().flatten()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        diff = a - b
        ok = torch.isfinite(a) & torch.isfinite(b) & (diff.abs() <= r0.ATOL + r0.RTOL * b.abs())
        bad = int((~ok).sum())
        failures += bad
        elements += a.numel()
        squares = [float(v.square().sum()) if finite else None for v in (a, b, diff)]
        rank = torch.nan_to_num(diff.abs(), nan=float("inf"), posinf=float("inf"))
        indices = torch.topk(rank, min(8, a.numel()), sorted=True).indices.tolist()

        def scalar(v):
            return float(v) if torch.isfinite(v) else None

        records.append(
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "count": a.numel(),
                "failed_coordinates": bad,
                "finite": finite,
                "actual_squared_norm": squares[0],
                "expected_squared_norm": squares[1],
                "residual_squared_norm": squares[2],
                "max_abs": scalar(rank.max()) if a.numel() else 0.0,
                "witnesses": [
                    {
                        "flat_index": i,
                        "actual": scalar(a[i]),
                        "expected": scalar(b[i]),
                        "residual": scalar(diff[i]),
                        "passed": bool(ok[i]),
                    }
                    for i in indices
                ],
            }
        )
        if finite:
            sums = [s + v for s, v in zip(sums, squares, strict=True)]
    finite = all(r["finite"] for r in records)
    return {
        "passed": failures == 0,
        "atol": r0.ATOL,
        "rtol": r0.RTOL,
        "elements": elements,
        "failed_coordinates": failures,
        "finite": finite,
        "actual_norm": sums[0] ** 0.5 if finite else None,
        "expected_norm": sums[1] ** 0.5 if finite else None,
        "residual_norm": sums[2] ** 0.5 if finite else None,
        "parameters": records,
    }


def gradient_diagnostic(encoder, query, x, stats, native):
    """All backward calls share ONE forward/target graph for this case."""
    ps = torch.tensor(PREFIXES, device=x.device).expand(len(x), -1)
    _, pred = previous.core.predict(encoder, query, x, ps, native)
    y, mask = previous.core.original.targets(x, ps, stats, native)
    parts = previous.core.original.loss_rows(pred, y, mask, stats, True)
    families = {
        k: w * parts[k].mean()
        for k, w in (
            ("path", 0.4),
            ("change1", 0.2),
            ("body", 0.1),
            ("activity", 0.3),
            ("structure", 1.0),
        )
    }
    bands = r0.price_band_losses(pred, y, mask, stats)
    named = named_parameters(encoder, query)
    total = gradients(parts["objective"].mean(), named)
    repeat = gradients(parts["objective"].mean(), named)
    checks = {"repeated_joint_backward": vector_comparison(total, repeat, named)}
    del repeat
    combined = [torch.zeros_like(g, dtype=torch.float64) for g in total]
    price = [torch.zeros_like(g, dtype=torch.float64) for g in total]
    summaries = {}
    for name, loss in families.items():
        values = gradients(loss, named)
        summaries[name] = {"weighted_loss": float(loss.detach()), "groups": {}}
        for group in ("encoder", "query"):
            group_values = [
                g for (n, _), g in zip(named, values, strict=True) if n.startswith(group + ".")
            ]
            summaries[name]["groups"][group] = {
                "squared_norm_float64": sum(float(g.double().square().sum()) for g in group_values),
                "squared_norm_legacy_float32": float(
                    torch.cat([g.flatten() for g in group_values]).norm().square()
                ),
            }
        for a, b, g in zip(combined, price, values, strict=True):
            a.add_(g.double())
            if name in ("path", "change1", "body"):
                b.add_(g.double())
        del values
    checks["joint_vs_sum_families"] = vector_comparison(total, combined, named)
    direct_price = gradients(families["path"] + families["change1"] + families["body"], named)
    checks["direct_price_vs_sum_families"] = vector_comparison(direct_price, price, named)
    del direct_price, price
    for value in combined:
        value.zero_()
    for loss in bands.values():
        values = gradients(loss, named)
        for a, g in zip(combined, values, strict=True):
            a.add_(g.double())
        del values
    direct_price = gradients(families["path"] + families["change1"] + families["body"], named)
    checks["direct_price_vs_sum_age_bands"] = vector_comparison(direct_price, combined, named)
    norm = checks["joint_vs_sum_families"]["actual_norm"]
    return {
        "passed": all(v["passed"] for v in checks.values()),
        "checks": checks,
        "components": summaries,
        "joint_loss": float(parts["objective"].mean().detach()),
        "joint_clip_factor_diagnostic": min(1.0, 1.0 / (norm + 1e-6)) if norm is not None else None,
        "scope": "First2 train windows, fixed4 prefixes; no optimizer, no actual clipping/update. Per-parameter summaries and8 largest residual coordinates, NOT full exported gradient vectors.",
        "same_forward": True,
        "runtime": r0.runtime_state(),
    }


def ps_for(n, device="cpu"):
    return torch.tensor(PREFIXES, device=device).expand(n, -1)


def objective_comparison(terms, historical):
    """Check new per-window objectives against the bound historical target path."""
    checks = {}
    for arm, mode in objectives.ARMS.items():
        parts = terms[arm]
        expected = historical["query"].mean(1)
        if mode.remote_structure:
            expected = expected + objectives.hs.STRUCTURE_WEIGHT * historical["structure"].mean(1)
        if mode.local_to_encoder:
            expected = expected + objectives.ba.LOCAL_WEIGHT * terms["baseline"]["local_diagnostic"]
        for name, actual, reference in (
            ("query", parts["query"], historical["query"].mean(1)),
            ("structure", parts["remote_structure"], historical["structure"].mean(1)),
            ("local", parts["local_diagnostic"], terms["baseline"]["local_diagnostic"]),
            ("optimization", parts["optimization_value"], expected),
        ):
            checks[f"{arm}/{name}"] = r0.compare(
                actual.detach().cpu().numpy(), reference.detach().cpu().numpy()
            )
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def aligned_data(source, meta, split):
    x, rows = previous.data.cache(source, split)
    old = previous.parent.data(meta, split)
    inventory = read_json(previous.parent.root(meta) / f"{split}_inventory.json")
    indices = [r["original_index"] for r in rows]
    if (
        len(set(indices)) != len(rows)
        or x.shape != (len(rows), 255, 28)
        or any(i < 0 or i >= len(inventory) for i in indices)
    ):
        raise ValueError("Invalid original-index mapping")
    for r, i in zip(rows, indices, strict=True):
        if {k: r[k] for k in inventory[i]} != inventory[i]:
            raise ValueError("Original endpoint identity differs")
    batch = {k: old[k][indices] for k in ("x", "y", "mask")}
    if not np.array_equal(batch["x"], x[:, -128:]):
        raise ValueError("Original native input differs from verified255 cache")
    return x, batch, rows


def target_audit(x, old, stats, local):
    maxima = {"query": 0.0, "local": 0.0}
    passed = {"query": True, "local": True}
    mask_equal = True
    failures = []
    for left in range(0, len(x), 64):
        b = {k: torch.as_tensor(v[left : left + 64]) for k, v in old.items()}
        ps = ps_for(len(b["x"]))
        y, m = hq.targets(b, ps, stats)
        z, zm = previous.core.original.targets(
            torch.as_tensor(x[left : left + 64]), ps, stats, True
        )
        ly, lm = hq.ba.local_targets(b["y"], b["mask"], ps, stats, local)
        derived_local = torch.where(lm, hq.recent_prediction(z, stats, local), 0)
        tests = {
            "query": r0.compare(z.numpy(), y.numpy(), r0.TARGET_ATOL),
            "local": r0.compare(derived_local.numpy(), ly.numpy(), r0.TARGET_ATOL),
        }
        masks = {
            "query": bool(torch.equal(m, zm)),
            "local": bool(torch.equal(lm, zm[..., :16, :].flip(-2))),
        }
        mask_equal &= all(masks.values())
        for name, check in tests.items():
            maxima[name] = max(maxima[name], check["max_abs"] or 0.0)
            passed[name] &= check["passed"] and masks[name]
        if not all(masks.values()) or not all(c["passed"] for c in tests.values()):
            failures.append({"first_row": left, "checks": tests, "masks_equal": masks})
    return {
        "passed": all(passed.values()),
        "query_passed": passed["query"],
        "local_passed": passed["local"],
        "rows": len(x),
        "prefixes": list(PREFIXES),
        "max_abs": maxima,
        "masks_equal": mask_equal,
        "failed_batches": failures,
        "atol": r0.TARGET_ATOL,
        "rtol": r0.RTOL,
    }


def train_mean(train, stats, local):
    sums = torch.zeros((4, 16, 7), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for left in range(0, len(train["x"]), 64):
        y, m = hq.ba.local_targets(
            torch.as_tensor(train["y"][left : left + 64]),
            torch.as_tensor(train["mask"][left : left + 64]),
            ps_for(len(train["x"][left : left + 64])),
            stats,
            local,
        )
        sums += torch.where(m, y.double(), 0).sum(0)
        counts += m.sum(0)
    return (sums / counts.clamp_min(1)).float(), counts


@torch.no_grad()
def local_reader(model, val, stats, local, mean, counts, device, micro=8):
    model.eval().requires_grad_(False)
    predictions, baselines = [], []
    coverage = True
    for left in range(0, len(val["x"]), micro):
        batch = {k: torch.as_tensor(v[left : left + micro], device=device) for k, v in val.items()}
        ps = ps_for(len(batch["x"]), device)
        z = model.core.encoder(batch["x"])
        picked = z[torch.arange(len(z), device=device)[:, None], ps - 1]
        pred = model.local_head(picked).reshape(len(z), 4, 16, 7)
        y, mask = hq.ba.local_targets(batch["y"], batch["mask"], ps, stats, local)
        baseline = mean.to(device).expand(len(z), -1, -1, -1)
        coverage &= not bool((mask & (counts.to(device)[None] == 0)).any())
        # One prefix at a time: local_rows itself averages its prefix dimension.
        predictions.append(
            torch.stack(
                [
                    hq.ba.local_rows(pred[:, j : j + 1], y[:, j : j + 1], mask[:, j : j + 1], True)[
                        "primary"
                    ]
                    for j in range(4)
                ],
                1,
            ).cpu()
        )
        baselines.append(
            torch.stack(
                [
                    hq.ba.local_rows(
                        baseline[:, j : j + 1], y[:, j : j + 1], mask[:, j : j + 1], True
                    )["primary"]
                    for j in range(4)
                ],
                1,
            ).cpu()
        )
    a, b = torch.cat(predictions).numpy(), torch.cat(baselines).numpy()
    values = {}
    for context, indices in (("endpoint", slice(3, 4)), ("interior", slice(0, 3))):
        actual, expected = float(a[:, indices].mean()), float(b[:, indices].mean())
        values[context] = {
            "reader": actual,
            "train_mean": expected,
            "passed": bool(np.isfinite([actual, expected]).all() and actual < expected),
        }
    return (
        {
            "passed": coverage and all(v["passed"] for v in values.values()),
            "train_baseline_covers_validation": coverage,
            "values": values,
        },
        a,
        b,
    )


def preflight(source, audit, out, device="cuda"):
    progress("V14: verifying reviewed R0 report, immutable source and cache hashes")
    audit_evidence = evidence_binding(audit, source)
    meta, _, _, stats = r0.identities(source, out)
    identity_before = read_json(out / "identity.json")
    if identity_before != read_json(audit / "identity.json"):
        raise ValueError("Current source identity differs from reviewed R0 evidence")
    local = previous.parent.stats(meta)[1]
    for key in ("mean", "scale"):
        value = np.asarray(local[key])
        if (
            value.shape != (7,)
            or not np.isfinite(value).all()
            or (key == "scale" and (value <= 0).any())
        ):
            raise ValueError("Invalid inherited local target scaling")
    datasets, checks = {}, {}
    for split in ("train", "val"):
        progress(f"V14 {split}: checking native inputs, row identities and both target pipelines")
        x, old, rows = aligned_data(source, meta, split)
        datasets[split] = x, old, rows
        checks[split] = target_audit(x, old, stats, local)
        atomic_json(checks[split], out / f"targets_{split}.json")
    if not all(v["query_passed"] for v in checks.values()):
        return {
            "r1_preflight_passed": False,
            "rs_reader_qualified": False,
            "reason": "target_contract_failed",
        }
    r0.schedule_audit(source, meta, len(datasets["train"][0]), out)
    atomic_json(datasets["val"][2], out / "local_validation_rows.json")
    mean, counts = train_mean(datasets["train"][1], stats, local)
    atomic_json(
        {
            "scope": "Train only; fixed prefix, lag, channel means; masked observed values",
            "mean": mean.tolist(),
            "counts": counts.tolist(),
        },
        out / "local_mean.json",
    )
    qualifications, gradients_ok, replay_ok, routes_ok = {}, [], [], []
    for seed in (42, 43):
        progress(f"V02/V14 s{seed}: original validation and local reader; zero updates")
        model, query = previous.parent.construct(meta, seed, device)
        signature = previous.parent.ur.bb.state_signature
        before = {"model": signature(model), "query": signature(query)}
        model.core.encoder.eval().requires_grad_(True)
        query.eval().requires_grad_(True)
        expected = read_json(source / f"prefix_s{seed}/completion.json")
        if (
            signature(model.core.encoder) != expected["initial_encoder_hash"]
            or signature(query) != expected["initial_query_hash"]
        ):
            raise ValueError("Original prefix initialization differs")
        actual = previous.validation(meta, model.core.encoder, query, datasets["val"][0], device)
        replay = r0.nested_compare(actual, expected["initial_validation"])
        atomic_json(
            {"checks": replay, "actual": actual, "expected": expected["initial_validation"]},
            out / f"replay_s{seed}.json",
        )
        replay_ok.append(all(v["passed"] for v in replay.values()))
        reader, a, b = local_reader(model, datasets["val"][1], stats, local, mean, counts, device)
        qualifications[str(seed)] = reader
        np.savez_compressed(out / f"local_s{seed}.npz", reader=a, train_mean=b)
        atomic_json(reader, out / f"local_s{seed}.json")
        model.core.encoder.requires_grad_(True)
        query.requires_grad_(True)
        model.local_head.eval().requires_grad_(False)
        fixed = torch.as_tensor(datasets["train"][0][:2], device=device)
        r0.causality(model.core.encoder, query, fixed, out, f"s{seed}")
        for mode in ("prefix", "rolling"):
            progress(f"V02 s{seed}/{mode}: same-forward gradient identity, no optimizer")
            report = gradient_diagnostic(model.core.encoder, query, fixed, stats, mode == "prefix")
            atomic_json(report, out / f"gradients_{mode}_s{seed}.json")
            gradients_ok.append(report["passed"])
        # Exercise all four NEW objective routes on actual source states/labels.
        old = {k: torch.as_tensor(v[:2], device=device) for k, v in datasets["train"][1].items()}
        terms = {
            arm: objectives.prefix_loss_rows(
                model, query, old, ps_for(2, device), stats, local, arm
            )
            for arm in objectives.ARMS
        }
        with torch.no_grad():
            _, original_pred = previous.core.predict(
                model.core.encoder, query, fixed, ps_for(2, device), True
            )
            original_y, original_mask = previous.core.original.targets(
                fixed, ps_for(2, device), stats, True
            )
            historical = previous.core.original.loss_rows(
                original_pred, original_y, original_mask, stats, True
            )
        route = {"historical_objective_parity": objective_comparison(terms, historical)}
        for arm, parts in terms.items():
            local_grad = parts["local_diagnostic"].requires_grad
            route[arm] = {
                "passed": local_grad == objectives.ARMS[arm].local_to_encoder,
                "local_requires_grad": local_grad,
                "loss": float(parts["optimization_value"].mean().detach()),
            }
        named = named_parameters(model.core.encoder, query)
        routed = gradients(terms["with_local"]["local_diagnostic"].mean(), named)
        norms = {
            group: sum(
                float(g.double().square().sum())
                for (name, _), g in zip(named, routed, strict=True)
                if name.startswith(group + ".")
            )
            for group in ("encoder", "query")
        }
        route["actual_local_gradient"] = {
            "passed": bool(np.isfinite(list(norms.values())).all())
            and (
                norms["encoder"] > 0
                or float(terms["with_local"]["local_diagnostic"].mean().detach()) == 0
            )
            and norms["query"] == 0,
            "squared_norms": norms,
            "zero_loss_allows_zero_gradient": True,
        }
        atomic_json(route, out / f"routes_s{seed}.json")
        routes_ok.append(all(r["passed"] for r in route.values()))
        after = {"model": signature(model), "query": signature(query)}
        unchanged = before == after and all(
            p.grad is None for p in list(model.parameters()) + list(query.parameters())
        )
        atomic_json(
            {"before": before, "after": after, "passed": unchanged}, out / f"unchanged_s{seed}.json"
        )
        if not unchanged:
            raise ValueError(
                "Read-only qualification changed model or accumulated parameter gradients"
            )
        del model, query, terms, parts, old, fixed, named, routed
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    r0.identities(source, out)
    if identity_before != read_json(out / "identity.json") or audit_evidence != evidence_binding(
        audit, source
    ):
        raise ValueError("Source or reviewed audit changed during qualification")
    return {
        "r1_preflight_passed": all(gradients_ok) and all(replay_ok),
        "rs_reader_qualified": all(q["passed"] for q in qualifications.values())
        and all(routes_ok)
        and all(v["local_passed"] for v in checks.values()),
        "rs_reader_quality_passed": all(q["passed"] for q in qualifications.values()),
        "rs_routes_passed": all(routes_ok),
        "rs_target_contract_passed": all(v["local_passed"] for v in checks.values()),
        "gradient_checks_passed": all(gradients_ok),
        "replays_passed": all(replay_ok),
        "local_reader": qualifications,
        "source_unchanged": True,
        "scope": "R0 raw reconstruction reused by identical bound caches; no raw/scaler refit or research-set model evaluation. Qualification is not a new training result.",
    }


def run(source, audit, out):
    source, out = r0.safe_output(source, out)
    audit, _ = r0.safe_output(audit, out)
    # Atomic directory creation also refuses a concurrent run at the same path.
    out.mkdir(parents=True)
    started = time.monotonic()
    status = {
        "schema": SCHEMA,
        "task_ids": ["V02", "V14"],
        "status": "running",
        "optimizer_updates": 0,
        "r1_authorized": False,
        "rs_authorized": False,
    }
    atomic_json(status, out / "status.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("AutoDL CUDA-only; local tests use synthetic fixtures")
        if shutil.disk_usage(out).free < 512 * 2**20:
            raise ValueError(
                "Need512MiB free for bounded reports; source files will not be deleted"
            )
        modules = (r0, objectives)
        repo = Path(__file__).resolve().parents[3]
        protocol = repo / "docs/BABEL_RECOVERY_QUALIFICATION.md"
        entrypoint = repo / "scripts/babel_recovery_qualification_autodl.sh"
        shutil.copyfile(protocol, out / "protocol.md")
        atomic_json(
            {
                "schema": SCHEMA,
                "task_ids": status["task_ids"],
                "source": str(source),
                "reviewed_audit": str(audit),
                "audit_completion_sha256": AUDIT_COMPLETION,
                "code_sha256": {Path(m.__file__).name: sha256(m.__file__) for m in modules}
                | {Path(__file__).name: sha256(__file__)},
                "historical_code": previous.code_identity(),
                "protocol_sha256": sha256(protocol),
                "entrypoint_sha256": sha256(entrypoint),
                "prefixes": list(PREFIXES),
                "atol": r0.ATOL,
                "rtol": r0.RTOL,
                "target_atol": r0.TARGET_ATOL,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "runtime_profile": r0.RUNTIME_PROFILES["context"],
                "optimizer_updates": 0,
            },
            out / "manifest.json",
        )
        with r0.replay_runtime("context"):
            result = preflight(source, audit, out)
        manifest = read_json(out / "manifest.json")
        current_code = {Path(m.__file__).name: sha256(m.__file__) for m in modules} | {
            Path(__file__).name: sha256(__file__)
        }
        if (
            current_code != manifest["code_sha256"]
            or sha256(protocol) != manifest["protocol_sha256"]
            or sha256(entrypoint) != manifest["entrypoint_sha256"]
        ):
            raise ValueError("Qualification implementation/protocol changed during execution")
        accepted = result["r1_preflight_passed"] and result["rs_reader_qualified"]
        status.update(
            result,
            status="qualification_complete_requires_review" if accepted else "blocked",
            elapsed_seconds=time.monotonic() - started,
        )
        atomic_json(status, out / "status.json")
        atomic_json(
            {
                **status,
                "files": {
                    str(p.relative_to(out)): sha256(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                },
            },
            out / "completion.json",
        )
        return 0 if accepted else 3
    except Exception as error:
        status.update(
            status="failed",
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
            elapsed_seconds=time.monotonic() - started,
        )
        atomic_json(status, out / "status.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    raise SystemExit(run(a.source, a.audit, a.out))


if __name__ == "__main__":
    main()
