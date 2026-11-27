"""Dense held-out history reads, fixed masked tests and per-endpoint evidence."""

from __future__ import annotations

import numpy as np
import torch

from .objective import query_weights, readings
from .views import U, sample, spans_for_mask


def prices(values, anchor):
    close = anchor * np.exp(values[:, 0])
    op = close * np.exp(-values[:, 2])
    return np.stack(
        (
            op,
            np.maximum(op, close) * np.exp(values[:, 3]),
            np.minimum(op, close) * np.exp(-values[:, 4]),
            close,
        ),
        1,
    )


def average(x, w):
    ok = np.isfinite(x) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else None


def physical_desc(y, stats):
    out = y.reshape(3, 6).copy()
    out[:, 4] = np.expm1(out[:, 4].clip(-32, 32))
    out[:, 5] = np.sinh(out[:, 5].clip(-32, 32)) * stats["desc_delta"]
    return out.ravel()


def direction(log_prices):
    travel = np.abs(np.diff(log_prices)).sum()
    net = log_prices[-1] - log_prices[0]
    if travel > 0 and 2 * net >= travel:
        return 1
    if travel > 0 and -2 * net >= travel:
        return -1
    return 0


def history_evidence(s, post, prior, stats):
    w = query_weights(s)
    sd = stats["y_sd"][s.period]
    truth = s.target
    levels, failed = {}, {}
    for name, pred in (("deployed", post), ("before_correction", prior)):
        sq = ((pred - truth) / sd) ** 2
        failed[name] = [
            int(np.sum((w > 0) & np.isfinite(truth[:, k]) & ~np.isfinite(pred[:, k])))
            for k in (0, 1)
        ]
        levels[name] = [
            None if failed[name][k] else average(sq[:, k], w) for k in (0, 1)
        ]
    idx = np.flatnonzero(s.facts[:, 4])
    perpendicular = {}
    missing_blocks = 0
    for g in idx:
        take = np.flatnonzero(s.block == g)
        p = int(s.period[take[0]])
        residual = (post[take, 0] - truth[take, 0]) / stats["y_sd"][p, 0]
        if not np.isfinite(residual).all():
            missing_blocks += 1
            continue
        perpendicular.setdefault(p, []).append(
            float(np.sum((residual - U @ (U.T @ residual)) ** 2) / 14)
        )
    err = np.abs(prices(post, s.anchor) - s.physical[:, :4])
    err[s.current, 3] = np.nan
    strata = {}
    for p in np.unique(s.period):
        for age, mask in (
            ("current", s.current),
            ("1_16", (s.age > 0) & (s.age <= 16)),
            ("17_64", (s.age > 16) & (s.age <= 64)),
            ("65_plus", s.age > 64),
        ):
            ix = np.flatnonzero((s.period == p) & mask & np.isfinite(truth[:, 0]))
            if not len(ix):
                continue
            ww = np.ones(len(ix)) if age == "current" else w[ix]
            strata[f"{p}/{age}"] = {
                "count": len(ix),
                "reachable": int(s.reachable[ix].sum()),
                "ohlc_mae": [average(err[ix, k], ww) for k in range(4)],
                "ohlc_p95": [
                    float(np.quantile(err[ix, k], 0.95))
                    if np.isfinite(err[ix, k]).all()
                    else None
                    for k in range(4)
                ],
                "shape_mae": [
                    average(abs(post[ix, k] - truth[ix, k]), ww) for k in (2, 3, 4)
                ],
                "quantity_mae": [
                    average(abs(np.expm1(post[ix, k]) - np.expm1(truth[ix, k])), ww)
                    for k in (5, 6, 7)
                ],
            }
    last65 = np.flatnonzero(s.period == 0)[-65:]
    classification = None
    if len(last65) == 65 and np.array_equal(s.previous[last65[1:]], last65[:-1]):
        true_path = truth[last65, 0]
        read_path = post[last65, 0]
        if np.isfinite(true_path).all():
            classification = {
                "truth": direction(true_path),
                "reconstructed": direction(read_path)
                if np.isfinite(read_path).all()
                else None,
            }
    # No learned parameters. Subsample visible anchored prices by public calendar phase.
    interpolated = np.full(len(w), np.nan)
    for p in np.unique(s.period):
        ix = np.flatnonzero(s.period == p)
        keep = ix[s.reachable[ix] & ((s.slot[ix] == 15) | s.current[ix])]
        if len(keep):
            interpolated[ix] = np.interp(
                -s.age[ix], -s.age[keep], s.visible_close[keep]
            )
    baseline = {}
    for name, estimate in (
        ("zero", np.zeros(len(w))),
        ("calendar_subsample", interpolated),
    ):
        dif = np.full(len(w), np.nan)
        linked = s.previous >= 0
        dif[linked] = estimate[linked] - estimate[s.previous[linked]]
        baseline[name] = [
            average(((val - truth[:, f]) / sd[:, f]) ** 2, w)
            for f, val in enumerate((estimate, dif))
        ]
    scope = {}
    for name, mask in (("anchored", s.reachable), ("unproven", ~s.reachable)):
        eligible = mask & np.isfinite(truth[:, 0]) & (w > 0)
        scope[name] = {
            "past_queries": int(eligible.sum()),
            "close_bp_mae": average(
                np.abs(post[:, 0] - truth[:, 0]) * 1e4, w * eligible
            ),
        }
    return {
        "R_terms": levels,
        "analytic_price_controls": baseline,
        "price_connection": scope,
        "direction64": classification,
        "R_perp": float(np.mean([np.mean(v) for v in perpendicular.values()]))
        if perpendicular and not missing_blocks
        else None,
        "valid_blocks": len(idx),
        "failed_blocks": missing_blocks,
        "failed_price_queries": failed,
        "strata": strata,
    }


@torch.no_grad()
def evaluate_endpoint(model, corpus, stats, sid, row, task, rng):
    shard = corpus.shard(sid)
    hidden = spans_for_mask(shard, row, rng) if task == 1 else ()
    s = sample(shard, row, stats, hidden)
    z = model.encode(s)
    if task == 2:
        prediction = (
            model.describe(model.condition(z))[0].double().cpu().numpy() * stats["d_sd"]
            + stats["d_mu"]
        )
        truth = s.desc
        return {
            "A_standardized_sq": (((prediction - truth) / stats["d_sd"]) ** 2).tolist(),
            "A_physical_abs": np.abs(
                physical_desc(prediction, stats) - physical_desc(truth, stats)
            ).tolist(),
        }
    if task == 0:
        ids = np.arange(len(s.current))
        post, prior, _ = readings(model, z, s, ids)
        return history_evidence(s, post.cpu().numpy(), prior.cpu().numpy(), stats)
    ids = np.flatnonzero(s.hidden)
    post, _, _ = readings(model, z, s, ids)
    err = abs(post.cpu().numpy() - s.target[ids])
    known = (s.period == 0) & s.reachable & ~s.hidden & (s.x[:, 10] > 0)
    ref = None
    if known.any():
        ix = np.flatnonzero(known)
        interpolated = np.interp(-s.age[ids], -s.age[ix], s.visible_close[ix])
        ref = float(np.mean(abs(interpolated - s.target[ids, 0])))
    return {
        "hidden": list(map(int, hidden)),
        "absolute": [average(err[:, k], np.ones(len(ids))) for k in range(9)],
        "interpolation_close_mae": ref,
    }


def paired_evaluation(model, corpus, stats, split, n, emit):
    model.eval()
    sums = {k: [] for k in ("Rpath", "Radj", "Rperp", "A_trained_mse")}
    unavailable = []
    price_failures = np.zeros(2, dtype=int)
    try:
        for task in range(3):
            try:
                corpus.distribution(split, task)
            except ValueError:
                unavailable.append(task)
                continue
            for i in range(n):
                rng = np.random.default_rng([727, split, task, i])
                sid, row = corpus.draw(split, task, rng)
                result = evaluate_endpoint(model, corpus, stats, sid, row, task, rng)
                minute = int(corpus.shard(sid).time[row])
                emit(
                    {
                        "split": split,
                        "task": task,
                        "source": sid,
                        "row": row,
                        "minute_start_shanghai": minute,
                        "market_week": (minute // 1440 + 3) // 7,
                        **result,
                    }
                )
                if task == 0:
                    price_failures += result["failed_price_queries"]["deployed"]
                    for k, x in zip(
                        ("Rpath", "Radj", "Rperp"),
                        [*result["R_terms"]["deployed"], result["R_perp"]],
                    ):
                        if x is not None:
                            sums[k].append(x)
                if task == 2:
                    sums["A_trained_mse"].append(
                        float(np.mean(result["A_standardized_sq"]))
                    )
    finally:
        model.train()
    out = {k: float(np.mean(v)) if v else None for k, v in sums.items()}
    for i, key in enumerate(("Rpath", "Radj")):
        if price_failures[i]:
            out[key] = None
    out["failed_price_queries"] = price_failures.tolist()
    out["R"] = (
        (out["Rpath"] + out["Radj"]) / 2
        if out["Rpath"] is not None and out["Radj"] is not None
        else None
    )
    out["unavailable_tasks"] = unavailable
    out["counts"] = {k: len(v) for k, v in sums.items()}
    return out
