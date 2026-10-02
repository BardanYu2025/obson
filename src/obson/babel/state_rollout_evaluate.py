"""Future information skill, explicit oracle diagnostics, paired calendar uncertainty."""

import html

import numpy as np
import torch

from . import state_rollout as core
from . import state_rollout_data as data
from . import state_rollout_run as run
from .ae_extend import atomic_json
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress


def bootstrap(rows, draws=2000):
    months = np.array([r["month"] for r in rows])
    unique = sorted(set(months))
    groups = [np.flatnonzero(months == m) for m in unique]
    ids = np.random.default_rng(20261003).integers(0, len(groups), (draws, len(groups)))
    return groups, ids


def paired(a, b, rows, tail=False, draws=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (
        a.shape != b.shape
        or a.shape[0] != len(rows)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Matched finite cohort required")
    groups, ids = bootstrap(rows, draws)
    if tail:
        delta = float(np.quantile(a, 0.95) - np.quantile(b, 0.95))
        values = [
            np.quantile(a[ix], 0.95) - np.quantile(b[ix], 0.95)
            for ix in (np.concatenate([groups[i] for i in draw]) for draw in ids)
        ]
    else:
        d = (a - b).reshape(len(rows), -1).mean(1)
        sums, counts = np.array([d[g].sum() for g in groups]), np.array([len(g) for g in groups])
        values = sums[ids].sum(1) / counts[ids].sum(1)
        delta = float(d.mean())
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {
        "delta": delta,
        "low": float(lo),
        "high": float(hi),
        "months": len(groups),
        "windows": len(rows),
        "supported": len(groups) >= 6 and len(rows) >= 100,
        "scope": "Paired calendar-month research interval; reused cohorts; not independent new-data confirmation",
    }


def skill_band(error, reference, rows, draws=2000):
    """Familywise32-horizon bootstrap band: max absolute centered skill deviation."""
    h = error.shape[1]
    reference = reference[:, :h]
    groups, ids = bootstrap(rows, draws)
    counts = np.array([len(g) for g in groups])
    a = np.array([error[g].sum(0) for g in groups])
    b = np.array([reference[g].sum(0) for g in groups])
    mean_a, mean_b = error.mean(0), reference.mean(0)
    if (mean_b <= 0).any():
        return {
            "supported": False,
            "reason": "zero baseline error; relative skill undefined",
            "contiguous_horizon": 0,
        }
    aa = a[ids].sum(1) / counts[ids].sum(1)[:, None]
    bb = b[ids].sum(1) / counts[ids].sum(1)[:, None]
    if (bb <= 0).any():
        return {
            "supported": False,
            "reason": "zero bootstrap baseline error",
            "contiguous_horizon": 0,
        }
    point = 1 - mean_a / mean_b
    boots = 1 - aa / bb
    spread = boots.std(0).clip(1e-12)
    critical = float(np.quantile(np.max(np.abs((boots - point) / spread), axis=1), 0.95))
    low, high = point - critical * spread, point + critical * spread
    supported = len(groups) >= 6 and len(rows) >= 100
    end = 0
    if supported:
        for i, v in enumerate(low):
            if v <= 0:
                break
            end = i + 1
    return {
        "skill": point.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "supported": supported,
        "contiguous_horizon": end,
        "other_positive_horizons": [
            i + 1 for i, v in enumerate(low) if supported and v > 0 and i + 1 > end
        ],
        "scope": "95% simultaneous max-standardized-deviation bootstrap band across displayed horizons;1..8 trained,9..32 stress",
    }


def summarize(pred, bank, scale, reference, rows):
    e = core.errors(pred, bank["targets"], scale)
    ref = core.errors(reference, bank["targets"], scale)
    h = pred.shape[1]
    normalized_abs = np.sqrt(e["price"])
    return {
        "windows": len(rows),
        "primary": float(e["price"][:, :8].mean()),
        "curve": {
            "horizon": list(range(1, h + 1)),
            "mae_bps": e["absolute"].mean(0).tolist(),
            "normalized_rmse": np.sqrt(e["price"].mean(0)).tolist(),
            "tail95_bps": np.quantile(e["absolute"], 0.95, axis=0).tolist(),
            "incremental_mse": e["incremental"].mean(0).tolist(),
            "predicted_rms": np.sqrt(np.mean(pred**2, axis=0)).tolist(),
            "actual_rms": np.sqrt(np.mean(bank["targets"][:, :h] ** 2, axis=0)).tolist(),
        },
        "band": skill_band(e["price"], ref["price"], rows),
        "tolerance_fraction": {
            str(t): (normalized_abs <= t).mean(0).tolist() for t in (0.5, 1.0, 2.0)
        },
    }, e


def gates(n8, refs, rows, baseline):
    checks = []
    for metric, reference, factor, tail in (
        ("price", baseline, 0.95, False),
        ("incremental", baseline, 1.10, False),
        ("absolute", baseline, 1.10, True),
        ("price", "Raw-direct", 1.05, False),
        ("price", "N1", 1.0, False),
        ("price", "L1", 1.0, False),
        ("price", "Z-direct", 1.0, False),
    ):
        ci = paired(n8[metric][:, :8], factor * refs[reference][metric][:, :8], rows, tail=tail)
        checks.append(
            {
                "metric": metric,
                "reference": reference,
                "factor": factor,
                "tail95": tail,
                "interval": ci,
                "passed": bool(ci["supported"] and ci["high"] <= 0),
            }
        )
    return {
        "future_price_increment_confirmed": all(c["passed"] for c in checks[:3]),
        "raw_retention": checks[3]["passed"],
        "multistep_training_gain": checks[4]["passed"],
        "checks": checks,
    }


@torch.no_grad()
def oracle(query, bank, seed, delta, device, batch=64):
    values = []
    for start in range(0, len(bank["mapping"]), batch):
        z = torch.as_tensor(
            bank[f"z{seed}"][bank["mapping"][start : start + batch, 1:]], device=device
        )
        values.append(core.read_prices(query, z, delta).cpu().numpy())
    return np.concatenate(values)


def latent_diagnostics(pred, actual, initial, scaling):
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite state trajectory")
    err = np.mean(((pred - actual) / np.array(scaling["increment"])) ** 2, axis=-1)
    norm = np.linalg.norm(pred, axis=-1)
    cosine = (pred * actual).sum(-1) / np.maximum(norm * np.linalg.norm(actual, axis=-1), 1e-12)
    movement = np.mean((pred - initial[:, None]) ** 2, axis=-1)
    rank = {}
    # Fixed first256 origins: bounded diagnostic, not population rank or a success gate.
    for h in (1, 8, 16, 32):
        x = pred[:256, h - 1].astype(float)
        spectrum = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
        if spectrum.sum() > 0:
            p = spectrum / spectrum.sum()
            rank[str(h)] = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
        else:
            rank[str(h)] = 0.0
    return {"increment_scaled_mse": err, "cosine": cosine, "norm": norm, "movement": movement}, {
        "increment_scaled_mse": err.mean(0).tolist(),
        "cosine": cosine.mean(0).tolist(),
        "norm": norm.mean(0).tolist(),
        "movement": movement.mean(0).tolist(),
        "effective_rank_first256": rank,
        "note": "Auxiliary state diagnostics; overlap127 can explain high similarity. Rank uses fixed first256 origins, not all768 independent dimensions.",
    }


def pretrain_audit(meta, out, device):
    if (out / "pretrain_lock.json").exists():
        lock = read_json(out / "pretrain_lock.json")
        if lock["manifest_sha256"] != sha256(out / "manifest.json"):
            raise ValueError("Pretrain audit binding changed")
        verify_files(out, lock["files"])
        return
    fit = data.transformations(out)
    bank, rows = data.cache(out, "val")
    scale = core.price_scale(bank, fit["price"])
    baselines = core.baselines(bank, fit["price"])
    result = {
        "scope": "Validation only, zero optimization. Oracle sees real future states; not a predictor or strict error lower bound.",
        "simple": {
            k: float(core.errors(v[:, :8], bank["targets"], scale)["price"].mean())
            for k, v in baselines.items()
        },
        "seeds": {},
    }
    exports = dict(truth=bank["targets"], scale=scale, **baselines)
    delta = run.parent.stats(meta)[0]["delta_scale"][0]
    for seed in (42, 43):
        model, query = run.parent.construct(meta, seed, device)
        del model
        query.eval().requires_grad_(False)
        signature = run.parent.ur.bb.state_signature(query)
        actual = oracle(query, bank, seed, delta, device)
        p = run.prepared(bank, fit, seed)
        identity = run.forecast(None, "identity", p, fit, seed, query, delta, device, 32)["price"]
        exports.update({f"oracle_s{seed}": actual, f"identity_s{seed}": identity})
        result["seeds"][str(seed)] = {
            "oracle_price_mse": float(
                core.errors(actual[:, :8], bank["targets"], scale)["price"].mean()
            ),
            "identity_price_mse": float(
                core.errors(identity[:, :8], bank["targets"], scale)["price"].mean()
            ),
            "decoder_signature": signature,
        }
        if signature != run.parent.ur.bb.state_signature(query):
            raise ValueError("Frozen query mutated")
    atomic_json(result, out / "pretrain_audit.json")
    np.savez_compressed(out / "pretrain_predictions.npz", **exports)
    atomic_json(rows, out / "pretrain_rows.json")
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "files": {
                n: sha256(out / n)
                for n in (
                    "pretrain_audit.json",
                    "pretrain_predictions.npz",
                    "pretrain_rows.json",
                    "transforms_lock.json",
                    "cache/val_index.json",
                )
            },
        },
        out / "pretrain_lock.json",
    )
    progress(
        "Validation oracle/identity diagnostics saved; engineering checks passed; starting fixed20-fit matrix"
    )


def subgroups(bank, rows, fit):
    groups = {f"period{p}": bank["period"] == p for p in (15, 30, 60)}
    for symbol in sorted({r["symbol"] for r in rows}):
        groups[f"symbol_{symbol}"] = np.array([r["symbol"] == symbol for r in rows])
    tertile = np.array(
        [
            np.searchsorted(fit["price"][str(int(p))]["rms_tertiles"], v)
            for p, v in zip(bank["period"], bank["past"][:, 0], strict=False)
        ]
    )
    for k in range(3):
        groups[f"past_volatility{k}"] = tertile == k
    groups["cross_session_1to8"] = bank["gaps"][:, :8].any(1)
    groups["no_session_gap_1to8"] = ~bank["gaps"][:, :8].any(1)
    return {k: v for k, v in groups.items() if v.any()}


def plot(out, split, seed, bank, rows, predictions, metrics, baseline):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in (
        "N8_best",
        "N1_best",
        "L1_best",
        "Z-direct_best",
        "Raw-direct_best",
        baseline,
        "identity",
    ):
        m = metrics[name]
        axes[0].plot(m["curve"]["horizon"], m["curve"]["normalized_rmse"], label=name)
        b = m["band"]
        if "skill" in b:
            axes[1].plot(m["curve"]["horizon"], b["skill"], label=name)
            if name == "N8_best":
                axes[1].fill_between(m["curve"]["horizon"], b["low"], b["high"], alpha=0.18)
    for ax in axes:
        ax.axvline(8.5, color="gray", linestyle=":")
        ax.set_xlabel("Future valid bars (9..32 beyond training length)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized cumulative-return RMSE")
    axes[1].set_ylabel("Skill vs validation-chosen simple baseline")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{split} / seed{seed}: free rollout; no new observed bars")
    fig.tight_layout()
    fig.savefig(folder / f"{split}_s{seed}_curves.png", dpi=150)
    plt.close(fig)
    # Fixed examples chosen by sorted symbol/period identities before reading errors.
    fixed = []
    for group in sorted({(r["symbol"], r["period"]) for r in rows}):
        fixed.append(next(i for i, r in enumerate(rows) if (r["symbol"], r["period"]) == group))
    fixed = fixed[:6]
    err = core.errors(
        predictions["N8_best"],
        bank["targets"],
        core.price_scale(bank, read_json(out / "transforms.json")["price"]),
    )["price"][:, :8].mean(1)
    selected = [(i, "fixed") for i in fixed] + [
        (int(np.argmax(err)), "worst normalized1..8 error (selected failure)")
    ]
    atomic_json(
        [dict(index=i, selection=why, **rows[i]) for i, why in selected],
        out / f"{split}_s{seed}_examples.json",
    )
    fig, axes = plt.subplots(len(selected), 1, figsize=(10, 2.4 * len(selected)))
    for ax, (i, why) in zip(np.atleast_1d(axes), selected, strict=False):
        c = bank["close"][i]
        ax.plot(
            np.arange(-127, 1), bank["history_close"][i], color="black", label="Observed history"
        )
        ax.plot(
            np.arange(1, 33),
            c * np.exp(bank["targets"][i] / 100),
            color="gray",
            label="Actual future (scoring only)",
        )
        for name in ("N8_best", "Z-direct_best", baseline):
            p = predictions[name][i]
            ax.plot(np.arange(1, len(p) + 1), c * np.exp(p / 100), label=name)
        ax.axvline(0, color="black", linestyle=":")
        ax.set_title(f"{rows[i]['key']} / {rows[i]['end']} / {why}", fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / f"{split}_s{seed}_examples.png", dpi=120)
    plt.close(fig)


def evaluate(meta, out, device):
    lock = run.check_selection(meta, out)
    fit = data.transformations(out)
    delta = run.parent.stats(meta)[0]["delta_scale"][0]
    results = {
        "scope": "Frozen-state exploratory forecast; reused research sets. No automatic promotion or trading-profit claim.",
        "cohorts": {},
    }
    validation_exports = {}
    for split in ("val", "test", "cross_research"):
        bank, rows = data.cache(out, split)
        scale = core.price_scale(bank, fit["price"])
        baseline_predictions = core.baselines(bank, fit["price"])
        baseline = lock["baseline"]
        cohort = {}
        atomic_json(rows, out / f"{split}_rows.json")
        for seed in (42, 43):
            progress(
                f"Evaluating {split}/s{seed}: locked best/last, future1..32, simple controls and oracle"
            )
            p = run.prepared(bank, fit, seed)
            model, query = run.parent.construct(meta, seed, device)
            del model
            query.eval().requires_grad_(False)
            signature = run.parent.ur.bb.state_signature(query)
            predictions = dict(baseline_predictions)
            predictions["oracle_future_state"] = oracle(query, bank, seed, delta, device)
            state_metrics, aux = {}, {}
            for arm in (*core.ARMS, "identity"):
                tags = ("best", "last") if arm != "identity" else ("fixed",)
                for tag in tags:
                    name = f"{arm}_{tag}" if arm != "identity" else "identity"
                    if arm != "identity":
                        jobname = lock["chosen"][f"{arm}_s{seed}"]
                        ck = torch.load(
                            out / jobname / f"{tag}.pt", map_location=device, weights_only=True
                        )
                        if ck["binding"] != run.binding(out, ck["binding"]["job"]):
                            raise ValueError("Selected checkpoint binding differs")
                        model = core.make_model(arm, p["u"].shape[1], p["raw"].shape[1]).to(device)
                        model.load_state_dict(ck["model"], strict=True)
                        model.eval()
                    else:
                        model = None
                    horizon = 32 if arm in core.ARMS[:3] or arm == "identity" else 8
                    result = run.forecast(
                        model, arm, p, fit, seed, query, delta, device, horizon, diagnostics=True
                    )
                    predictions[name] = result["price"]
                    if "states" in result:
                        raw, stat = latent_diagnostics(
                            result["states"],
                            p["u"][p["mapping"][:, 1:]],
                            p["u"][p["mapping"][:, 0]],
                            fit["states"][str(seed)],
                        )
                        aux.update({f"{name}_{k}": v for k, v in raw.items()})
                        aux[f"{name}_integrated_price"] = result["integrated"]
                        state_metrics[name] = stat
                    if split == "val" and tag == "best":
                        value = float(
                            core.errors(result["price"][:, :8], bank["targets"], scale)[
                                "price"
                            ].mean()
                        )
                        expected = lock["candidates"][jobname]["validation"]
                        if not np.isclose(value, expected, atol=1e-8, rtol=1e-5):
                            raise ValueError(
                                f"Free validation replay differs: {jobname}: {value}/{expected}"
                            )
                        validation_exports[jobname] = value
            metrics, errors = {}, {}
            for name, pred in predictions.items():
                metrics[name], errors[name] = summarize(
                    pred, bank, scale, baseline_predictions[baseline], rows
                )
            refs = {arm: errors[f"{arm}_best"] for arm in core.ARMS}
            refs[baseline] = errors[baseline]
            decision = gates(errors["N8_best"], refs, rows, baseline)
            groups = {}
            for name, ix in subgroups(bank, rows, fit).items():
                group_rows = [r for r, keep in zip(rows, ix, strict=False) if keep]
                ci = paired(
                    errors["N8_best"]["price"][ix, :8],
                    errors[baseline]["price"][ix, :8],
                    group_rows,
                )
                groups[name] = {
                    "windows": int(ix.sum()),
                    "price_difference": ci,
                    "n8_rmse": np.sqrt(errors["N8_best"]["price"][ix].mean(0)).tolist(),
                    "baseline_rmse": np.sqrt(errors[baseline]["price"][ix].mean(0)).tolist(),
                }
            cohort[str(seed)] = {
                "metrics": metrics,
                "latent": state_metrics,
                "decision": decision,
                "subgroups": groups,
            }
            np.savez_compressed(
                out / f"{split}_s{seed}_predictions.npz",
                truth=bank["targets"],
                scale=scale,
                close=bank["close"],
                history_close=bank["history_close"],
                gaps=bank["gaps"],
                **predictions,
                **aux,
            )
            if signature != run.parent.ur.bb.state_signature(query):
                raise ValueError("Decoder changed during evaluation")
            if split != "val":
                plot(out, split, seed, bank, rows, predictions, metrics, baseline)
        results["cohorts"][split] = cohort
    results["primary_passed"] = all(
        results["cohorts"][split][str(s)]["decision"]["future_price_increment_confirmed"]
        for split in ("test", "cross_research")
        for s in (42, 43)
    )
    results["selected_validation_replay"] = validation_exports
    results["decision"] = (
        "candidate_requires_new_time_confirmation"
        if results["primary_passed"]
        else "no_confirmed_future_price_increment_under_fixed_protocol"
    )
    atomic_json(results, out / "rollout_metrics.json")
    atomic_json(
        {
            "forecast_nonfinite": 0,
            "encoder_updates": 0,
            "decoder_updates": 0,
            "selection_lock": sha256(out / "selection_lock.json"),
            "predicted_states_export": "Full state trajectories omitted from small report; per-row state errors/norm/cosine and selected small weights included. Recompute from bound remote caches/checkpoints.",
            "price_predictions_export": "All selected best/last, baselines, identity and future-state oracle, every horizon and row.",
            "inference_future_access": False,
        },
        out / "evaluation_audit.json",
    )
    figures = sorted((out / "figures").glob("*.png"))
    body = (
        "<h1>Frozen state rollout768</h1><p>"
        + html.escape(results["decision"])
        + "</p><p>1–8 trained horizons;9–32 stress. Gray future paths are labels only. Oracle is not a forecast. Test cohorts reused.</p>"
    )
    body += "".join(
        f'<h2>{html.escape(p.stem)}</h2><img style="max-width:100%" src="figures/{p.name}">'
        for p in figures
    )
    (out / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>State rollout768</title>' + body
    )
