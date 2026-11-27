"""Predeclared fixed-capacity controls and fresh probes after all four last-N fits."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .assets import export_run, load_checkpoint, snapshot_context, verify_package
from .common import CAPS, atomic, exclusive, identity, json_file, read_json
from .metrics import history_evidence, physical_desc
from .model import calibrate, physical_output
from .objective import rate
from .run import device_setup, lock_study, new_model, ready_data
from .views import U, sample

NEURAL = ("P2-42", "P3-42", "P2-43", "P3-43")
PCA = ("pca768", "summary_pca608")
DESC = (*NEURAL, *PCA, "raw")
ORDER = (*NEURAL, "summary_pca608", "summary")
DIRECT = [0, 2, 3, 4, 5, 6, 7]


def slots(s):
    result = np.empty(len(s.x), int)
    offset = 0
    for p, cap in enumerate(CAPS):
        take = np.flatnonzero(s.period == p)
        result[take] = np.arange(offset + cap - len(take), offset + cap)
        offset += cap
    return result


def flattened(s):
    out = np.zeros((sum(CAPS), 37), dtype="f4")
    out[slots(s)] = np.column_stack((s.x, s.context, s.valid))
    return out.ravel()


def standard(x):
    mu = x.mean(axis=0, dtype="f8")
    sd = np.maximum(x.std(axis=0, dtype="f8"), 1e-6)
    return mu, sd


def principal(x, n=768):
    mu, sd = standard(x)
    scaled = (x - mu) / sd
    gram = scaled @ scaled.T
    eig, vec = np.linalg.eigh(gram)
    if len(eig) <= n or eig[-n] <= max(1e-10, eig[-1] * 1e-12):
        raise ValueError(f"Insufficient train-only rank for PCA{n}")
    take = np.arange(len(eig) - 1, len(eig) - 1 - n, -1)
    basis = (scaled.T @ vec[:, take]) / np.sqrt(eig[take])
    return mu, sd, basis


def ridge(x, y, w, penalty=0.01):
    """Weights sum to one; center X/Y to leave the intercept unpenalized."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mx, my = w @ x, w @ y
    a, b = np.sqrt(w)[:, None] * (x - mx), np.sqrt(w)[:, None] * (y - my)
    if a.shape[0] < a.shape[1]:
        mat = a @ a.T
        mat.flat[:: len(mat) + 1] += penalty
        coef = a.T @ np.linalg.solve(mat, b)
    else:
        mat = a.T @ a
        mat.flat[:: len(mat) + 1] += penalty
        coef = np.linalg.solve(mat, a.T @ b)
    return coef, my - mx @ coef


def historical_ridge(z, y):
    # One standardized coefficient system per distinct valid training support.
    mu, sd = standard(z)
    x = (z - mu) / sd
    y = y.reshape(len(x), -1)
    coeff = np.zeros((x.shape[1], y.shape[1]))
    bias = np.full(y.shape[1], np.nan)
    groups = {}
    for k in range(y.shape[1]):
        mask = np.isfinite(y[:, k])
        if mask.any():
            groups.setdefault(np.packbits(mask).tobytes(), []).append(k)
    for packed, cols in groups.items():
        ok = np.unpackbits(np.frombuffer(packed, "u1"))[: len(x)].astype(bool)
        coeff[:, cols], bias[cols] = ridge(
            x[ok], y[np.ix_(ok, cols)], np.ones(ok.sum())
        )
    return mu, sd, coeff, bias, len(groups)


def addresses(corpus, split, task, count):
    return [
        corpus.draw(split, task, np.random.default_rng([727, split, task, i]))
        for i in range(count)
    ]


def source_sample(corpus, stats, pair):
    sid, row = pair
    return sample(corpus.shard(sid), row, stats)


def raw_features(corpus, stats, pairs):
    raw, summary = [], []
    for pair in pairs:
        s = source_sample(corpus, stats, pair)
        raw.append(flattened(s))
        summary.append(s.facts.ravel())
    return np.array(raw), np.array(summary)


def fixed_features(raw, summary, pca):
    mu, sd, basis = pca
    coords = ((raw - mu) / sd @ basis).astype("f4")
    return {
        "raw": raw,
        "pca768": coords,
        "summary_pca608": np.column_stack((summary, coords[:, :608])),
        "summary": np.pad(summary, ((0, 0), (0, 608))),
    }


def order_truth(corpus, pair):
    sid, row = pair
    shard = corpus.shard(sid)
    if not shard.flags[row, 4]:
        raise ValueError("Ineligible natural order endpoint")
    c = shard.bars[0]["v"][row - 36 : row + 1, 3]
    net = float(np.log(c[-1] / c[0]))
    return (1 if np.argmax(c) < np.argmin(c) else -1), net


def standardized_history_target(s, stats):
    arr = np.full((sum(CAPS), 7), np.nan, dtype="f4")
    vals = ((s.target - stats["y_mu"][s.period]) / stats["y_sd"][s.period])[:, DIRECT]
    vals[s.current, 0] = np.nan
    arr[slots(s)] = vals
    return arr


def read_linear(s, z, fitted, stats, calibrated):
    mu, sd, coef, bias, _ = fitted
    tentative = (((z - mu) / sd) @ coef + bias).reshape(sum(CAPS), 7)[slots(s)]
    args = [
        torch.as_tensor(x)
        for x in (tentative, s.period, s.current, stats["y_mu"], stats["y_sd"])
    ]
    before = physical_output(*args)
    after = before.clone()
    if calibrated:
        for g in np.flatnonzero(s.facts[:, 4]):
            ix = np.flatnonzero(s.block == g)
            p = s.period[ix[0]]
            after[ix] = calibrate(
                before[ix][None],
                torch.tensor(s.facts[g : g + 1], dtype=torch.float64),
                torch.tensor(stats["block_delta"][p : p + 1]),
                torch.tensor(U),
            )[0]

    def terms(d):
        d = d.numpy()
        delta = np.full(len(d), np.nan)
        ok = s.previous >= 0
        delta[ok] = d[ok, 0] - d[s.previous[ok], 0]
        oi = np.arcsinh(
            (np.expm1(d[:, 6]) - np.expm1(d[:, 5])) / stats["delta"][s.period]
        )
        return np.column_stack((d[:, 0], delta, d[:, 1:], oi))

    return terms(after), terms(before)


def fit_description(x, y, n, device, checkpoint=None):
    mu, sd = standard(x)
    scaled = torch.as_tensor(((x - mu) / sd).astype("f4"), device=device)
    labels = torch.as_tensor(y.astype("f4"), device=device)
    torch.manual_seed(314)
    head = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, 18)).to(
        device
    )
    opt = torch.optim.AdamW(
        [
            {"params": [head[0].weight, head[2].weight], "weight_decay": 0.01},
            {"params": [head[0].bias, head[2].bias], "weight_decay": 0.0},
        ],
        lr=0.001,
        betas=(0.9, 0.95),
        foreach=False,
    )
    begin = 0
    if checkpoint is not None and checkpoint.exists():
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not np.array_equal(stored["mean"].numpy(), mu) or not np.array_equal(
            stored["sd"].numpy(), sd
        ):
            raise ValueError("Frozen probe inputs changed on resume")
        head.load_state_dict(stored["state"])
        opt.load_state_dict(stored["optimizer"])
        begin = stored["step"]
    for step in range(begin + 1, n + 1):
        ids = np.random.default_rng([314, step]).integers(len(x), size=64)
        ix = torch.as_tensor(ids, device=device)
        opt.param_groups[0]["lr"] = opt.param_groups[1]["lr"] = rate(step, n, 0.001)
        opt.zero_grad(set_to_none=True)
        F.smooth_l1_loss(head(scaled[ix]), labels[ix]).backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if checkpoint is not None and (step % 25 == 0 or step == n):
            with atomic(checkpoint) as tmp:
                torch.save(
                    {
                        "state": head.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step,
                        "mean": torch.tensor(mu),
                        "sd": torch.tensor(sd),
                    },
                    tmp,
                )
    return head, mu, sd, opt


def persist_np(path, **kwargs):
    with atomic(path) as tmp, tmp.open("wb") as out:
        np.savez(out, **kwargs)


def run_frozen(cache, download, release):
    cache = Path(cache)
    device = device_setup()
    card = lock_study(cache, cache / "study.lock.json", release)
    corpus, stats = ready_data(cache)
    design = identity(card)
    models = {}
    for key in NEURAL:
        arm, seed = key.split("-")
        rid = identity([design, arm, int(seed)])
        path = cache / "runs" / f"{key}-{rid[:12]}"
        if read_json(path / "completed.json")["step"] != card["updates"]:
            raise ValueError("All four primary last-N fits must finish first")
        models[key] = (arm, int(seed), rid, path)
    folder = cache / "frozen" / design[:16]
    with exclusive(cache / ".frozen.lock"):
        sources = {
            k: read_json(v[3] / "checkpoints.json")["last"] for k, v in models.items()
        }
        if (folder / "run.json").exists() and read_json(folder / "run.json").get(
            "sources"
        ) != sources:
            raise ValueError("Frozen stage source checkpoints changed")
        if (folder / "complete.json").exists():
            if not (folder / "export-receipt.json").exists():
                bundle = export_run(folder, download)
                json_file(
                    folder / "export-receipt.json",
                    {"file": str(bundle), **verify_package(bundle)},
                )
            return folder
        snapshot_context(
            folder,
            cache,
            release,
            stats,
            {"stage": "frozen", "study": card, "primary": "last", "sources": sources},
        )
        from shutil import copyfile

        copyfile(cache / "scales.json", folder / "data_receipt.json")
        budget_path = folder / "runtime.json"
        past = read_json(budget_path)["seconds"] if budget_path.exists() else 0.0
        start = time.monotonic()

        def check_budget():
            if past + time.monotonic() - start > 3600 * card["frozen_gpu_hours"]:
                raise TimeoutError("Frozen-stage time cap")
            if (
                sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
                > card["frozen_disk_gib"] * 2**30
            ):
                raise OSError("Frozen-stage disk cap")

        try:
            pairs = {}
            for split in (0, 2):
                for task in (0, 2, 3):
                    n = (
                        card["frozen_train_endpoints"]
                        if split == 0
                        else card["validation_endpoints"]
                    )
                    try:
                        pairs[split, task] = addresses(corpus, split, task, n)
                    except ValueError:
                        if split == 0:
                            raise
                        pairs[split, task] = []
            feature = {}
            pca = None
            for k, items in pairs.items():
                check_budget()
                if not items:
                    continue
                raw, summary = raw_features(corpus, stats, items)
                if k == (0, 0):
                    pca_path = folder / "pca.npz"
                    if pca_path.exists():
                        with np.load(pca_path, allow_pickle=False) as archive:
                            pca = tuple(archive[k] for k in ("mean", "sd", "basis"))
                    else:
                        pca = principal(raw)
                        persist_np(pca_path, mean=pca[0], sd=pca[1], basis=pca[2])
                feature[k] = fixed_features(raw, summary, pca)
                if k[1] != 2:
                    del feature[k]["raw"]
            for key, (arm, seed, rid, path) in models.items():
                check_budget()
                model = new_model(arm, stats, seed, device)
                load_checkpoint(path, "last", model, None, rid)
                model.eval()
                with torch.no_grad():
                    for k, items in pairs.items():
                        if k[1] != 0 and items:
                            feature[k][key] = np.vstack(
                                [
                                    model.encode(source_sample(corpus, stats, p))
                                    .cpu()
                                    .numpy()
                                    for p in items
                                ]
                            )
                del model
            outcomes = {}

            def standardized(k):
                return np.vstack(
                    [
                        (source_sample(corpus, stats, p).desc - stats["d_mu"])
                        / stats["d_sd"]
                        for p in pairs[k]
                    ]
                )

            y = standardized((0, 2))
            for name in DESC:
                check_budget()
                head, mu, sd, _ = fit_description(
                    feature[0, 2][name],
                    y,
                    card["description_updates"],
                    device,
                    folder / ("description-" + name + ".pt"),
                )
                if pairs[2, 2]:
                    x = torch.as_tensor(
                        ((feature[2, 2][name] - mu) / sd).astype("f4"), device=device
                    )
                    yh = head(x).detach().cpu().numpy()
                    yy = standardized((2, 2))
                    outcomes[name + "/U"] = float(np.mean((yh - yy) ** 2))
                    physical = [
                        np.abs(
                            physical_desc(a * stats["d_sd"] + stats["d_mu"], stats)
                            - physical_desc(b * stats["d_sd"] + stats["d_mu"], stats)
                        )
                        for a, b in zip(yh, yy)
                    ]
                    persist_np(
                        folder / ("description-development-" + name + ".npz"),
                        endpoints=pairs[2, 2],
                        prediction=yh,
                        truth=yy,
                        absolute_physical=physical,
                    )
            labels, nets = np.array([order_truth(corpus, p) for p in pairs[0, 3]]).T
            classes, counts = np.unique(labels, return_counts=True)
            if len(classes) != 2:
                raise ValueError(
                    "Natural order source supports only one training class"
                )
            weights = np.where(labels == 1, 0.5 / counts[1], 0.5 / counts[0])
            if pairs[2, 3]:
                test, net = np.array([order_truth(corpus, p) for p in pairs[2, 3]]).T
                rule = np.where(
                    net > 0,
                    -1,
                    np.where(net < 0, 1, (1 if counts[1] >= counts[0] else -1)),
                )
                persist_np(
                    folder / "order-direction-rule.npz",
                    prediction=rule,
                    truth=test,
                    endpoints=pairs[2, 3],
                )
            for name in ORDER:
                check_budget()
                x = feature[0, 3][name]
                mu, sd = standard(x)
                coef, bias = ridge((x - mu) / sd, labels[:, None], weights)
                persist_np(
                    folder / ("order-" + name + ".npz"),
                    mean=mu,
                    sd=sd,
                    coef=coef,
                    bias=bias,
                )
                if pairs[2, 3]:
                    val = (((feature[2, 3][name] - mu) / sd) @ coef + bias).ravel()
                    pred = np.where(val >= 0, 1, -1)
                    outcomes[name + "/order_BA"] = (
                        float(np.mean([np.mean(pred[test == k] == k) for k in (-1, 1)]))
                        if len(np.unique(test)) == 2
                        else None
                    )
                    persist_np(
                        folder / ("order-development-" + name + ".npz"),
                        score=val,
                        truth=test,
                        endpoints=pairs[2, 3],
                    )
            yy = np.stack(
                [
                    standardized_history_target(source_sample(corpus, stats, p), stats)
                    for p in pairs[0, 0]
                ]
            )
            for name in PCA:
                check_budget()
                path = folder / ("history-" + name + ".npz")
                if path.exists():
                    with np.load(path, allow_pickle=False) as archive:
                        fit = (
                            *[archive[k] for k in ("mu", "sd", "coef", "bias")],
                            int(archive["systems"]),
                        )
                else:
                    fit = historical_ridge(feature[0, 0][name], yy)
                    persist_np(
                        path,
                        mu=fit[0],
                        sd=fit[1],
                        coef=fit[2],
                        bias=fit[3],
                        systems=fit[4],
                    )
                outcomes[name + "/ridge_systems"] = fit[-1]
                with (
                    atomic(folder / ("history-development-" + name + ".jsonl")) as tmp,
                    tmp.open("w") as out,
                ):
                    for i, p in enumerate(pairs[2, 0]):
                        s = source_sample(corpus, stats, p)
                        post, prior = read_linear(
                            s, feature[2, 0][name][i], fit, stats, name == PCA[1]
                        )
                        report = history_evidence(s, post, prior, stats)
                        out.write(
                            json.dumps({"endpoint": p, **report}, allow_nan=False)
                            + "\n"
                        )
            json_file(
                folder / "complete.json",
                {
                    "results": outcomes,
                    "seconds": time.monotonic() - start,
                    "eligible": {str(k): len(v) for k, v in pairs.items()},
                    "scope": "Development readouts; independent confirmation open",
                },
            )
        except BaseException as exc:
            from uuid import uuid4

            json_file(
                folder / ("failure-" + uuid4().hex[:10] + ".json"), {"error": repr(exc)}
            )
            raise
        finally:
            json_file(budget_path, {"seconds": past + time.monotonic() - start})
            bundle = export_run(folder, download)
            json_file(
                folder / "export-receipt.json",
                {"file": str(bundle), **verify_package(bundle)},
            )
    return folder
