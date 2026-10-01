"""Finite, resumable frozen-encoder transfer; future prices are labels only."""

import argparse
import copy
import fcntl
import gc
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import activity_ablation as aa
from . import ae_context
from . import endpoint_readout as er
from . import linear_history_run as parent
from . import representation as rp
from . import volatility_transfer as core
from .ae_extend import atomic_json, atomic_save
from .dual_state import sha256
from .history_query_run import verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-volatility-transfer768-v1"
SPLITS = ("train", "val", "test", "cross_research")


def code_identity():
    return parent.code_identity() | {Path(p).name: sha256(p) for p in (__file__, core.__file__)}


def make_manifest(source, identity, batch):
    if batch < 1:
        raise ValueError("Positive extraction batch required")
    old = read_json(source / "manifest.json")
    meta = {k: copy.deepcopy(old[k]) for k in ("identity", "architecture")}
    meta.update(
        schema=SCHEMA,
        task_source=identity,
        code_sha256=code_identity(),
        extraction_batch=batch,
        reader_batch=512,
        reader_epochs=80,
        reader_lrs=list(core.LRS),
        fractions=list(core.FRACTIONS),
        reader_seeds=[42, 43],
        pca_rank=768,
        transfer_protocol={
            "target": "log(max(RMS of next16/64 same-contract close returns,1e-8)); no annualization; nominal-bar horizons, not equal physical durations",
            "quantiles": list(core.QUANTILES),
            "checkpoint": "Macro History control validation best97/94; fixed before this task. All encoder and decoder parameters frozen.",
            "inputs": "Inherited128x28 only; period one-hot given equally to all readers. Past EMA features can summarize pre-window history.",
            "labels": "All128 input bars and64 future bars stay in original session-date partition; no future-main-contract selection",
            "readers": "Affine6 outputs or128-wide GELU MLP6 outputs. Independent quantile pinball; sort quantiles only for scoring/inference.",
            "budget": "4 representations x2 families x2 seeds x2 label fractions x2 LR =64 fits,80 epochs each. Equal sample order/updates within budget; unequal parameter counts/compute disclosed.",
            "selection": "Each fit best validation pinball in original log-RMS units including epoch0; LR selected by same validation only. All locks precede research extraction.",
            "few_labels": "Stratified10% downstream TRAIN labels; same full validation labels for selection in both budgets; full unlabeled TRAIN used for feature scaling/PCA and original encoder pretraining. Not10% total-data learning.",
            "baseline": "Per-period empirical quantiles and calibrated historical-volatility persistence; stronger validation choice among these and simple-feature reader, per seed/fraction/family.",
            "primary": "Full-label MLP. Both encoder/reader paired seeds and both reused research sets: >=5% pinball gain vs validation-chosen simple baseline and PCA; <=5% mean and<=10% each horizon degradation vs raw. Paired monthly interval upper bound<=0.",
            "secondary": "Linear readers and10% labels reported independently; no choosing a family/fraction after research scores. Coverage80 and widths are diagnostics, not a guaranteed calibrated interval.",
            "scope": "Exploratory transfer, research sets repeatedly inspected. No trading-profit/general representation/independent holdout claim; no automatic promotion.",
            "stop": "One finite matrix; no encoder updates, extra epochs, grid expansion or test-driven checkpoint changes.",
        },
    )
    return meta


def cache(out, split):
    index = read_json(out / f"cache/{split}_index.json")
    if index["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Cache manifest changed")
    verify_files(out / "cache", index["files"])
    with np.load(out / f"cache/{split}.npz", allow_pickle=False) as f:
        bank = dict(f)
    return bank, read_json(out / f"cache/{split}_rows.json")


def encode(frame, period):
    return np.column_stack(
        (
            ae_context.encode_context(frame, period, "ema8_32")["x"],
            aa.activity_features(frame, period)[0],
        )
    )


@torch.inference_mode()
def prepare(meta, out, splits, cross, device):
    if any(s not in SPLITS or (s == "cross_research") != cross for s in splits):
        raise ValueError("Invalid preparation partition")
    if any(s in SPLITS[2:] for s in splits):
        check_selection(meta, out)
    pending = []
    for split in splits:
        if (out / f"cache/{split}_index.json").exists():
            cache(out, split)
        else:
            pending.append(split)
    if not pending:
        return
    tm = parent.tm(meta)
    series, bounds = parent.xr.xd.load_raw(tm, cross=cross)
    lookup = {s.key: s for s in series}
    statistics = parent.stats(meta)[0]
    for split in pending:
        progress(f"Preparing {split}: exact input replay and split-contained future labels")
        rows = read_json(parent.root(meta) / f"{split}_inventory.json")
        d = parent.data(meta, split)
        x = np.asarray(d["x"])
        if x.shape != (len(rows), 128, 28):
            raise ValueError("Inherited input/inventory shape differs")
        grouped = {}
        for i, row in enumerate(rows):
            grouped.setdefault(row["key"], []).append((i, row))
        selected, targets, past, kept, excluded = [], [], [], [], []
        maxdiff = 0.0
        for key, items in grouped.items():
            s = lookup[key]
            features = encode(s.frame, s.period)
            same = rp.time_mask(s, bounds, "test" if cross else split)
            close = s.frame.close.to_numpy(float)
            for i, row in items:
                end = int(row["row"])
                if (
                    end < 127
                    or end >= len(s.frame)
                    or str(s.frame.datetime.iloc[end]) != row["end"]
                ):
                    raise ValueError("Raw endpoint identity differs")
                reconstructed = (
                    (features[end - 127 : end + 1] - statistics["x_mean"]) / statistics["x_scale"]
                ).astype(np.float32)
                diff = float(np.max(np.abs(reconstructed - x[i])))
                maxdiff = max(maxdiff, diff)
                if not np.allclose(reconstructed, x[i], atol=1e-6, rtol=2e-5):
                    raise ValueError(f"Raw input replay differs at {split}/{key}/{end}: {diff}")
                y, reason = core.future_target(close, end, same)
                if reason:
                    excluded.append({"index": i, "key": key, "row": end, "reason": reason})
                    continue
                # Boundary/identity eligibility only. No filtering by label values or future main status.
                selected.append(i)
                targets.append(y)
                past.append(core.past_features(close, end))
                kept.append(
                    dict(
                        row,
                        original_index=i,
                        input_start=str(s.frame.datetime.iloc[end - 127]),
                        available_at=str(
                            s.frame.datetime.iloc[end] + np.timedelta64(s.period, "m")
                        ),
                        target_end=str(
                            s.frame.datetime.iloc[end + 64] + np.timedelta64(s.period, "m")
                        ),
                        target_session_end=str(s.sessions[end + 64]),
                    )
                )
            # Real-feature causality check on one fixed endpoint per contract, never labels into x.
            end = int(items[0][1]["row"])
            if not np.array_equal(encode(s.frame.iloc[: end + 1], s.period), features[: end + 1]):
                raise ValueError("Causal feature prefix replay differs")
        if len(selected) < 100:
            raise ValueError(f"Insufficient {split} eligible rows: {len(selected)}")
        order = np.argsort(selected)
        ids = np.asarray(selected)[order]
        kept = [kept[i] for i in order]
        bank = {
            "x": np.asarray(x[ids], np.float32),
            "targets": np.asarray(targets)[order],
            "simple": np.asarray(past)[order],
            "period": core.period_context(kept),
        }
        audits = {}
        for seed in (42, 43):
            model, query = parent.construct(meta, seed, device)
            del query
            model.eval().requires_grad_(False)
            before = parent.ur.bb.state_signature(model)
            causal = er.trained_causality(model, torch.as_tensor(bank["x"][:2], device=device))
            if causal["status"] == "failed":
                raise ValueError("Trained encoder causality check failed")
            bank[f"embedding_s{seed}"] = parent.de.states(
                model, bank, meta["extraction_batch"], device
            )
            unchanged = before == parent.ur.bb.state_signature(model)
            if not unchanged or any(p.requires_grad for p in model.parameters()):
                raise ValueError("Frozen encoder changed")
            if bank[f"embedding_s{seed}"].shape != (len(ids), 768):
                raise ValueError("Unexpected state width")
            audits[str(seed)] = {"signature": before, "unchanged": unchanged, "causality": causal}
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        if not all(np.isfinite(v).all() for v in bank.values()):
            raise ValueError("Nonfinite transfer cache")
        folder = out / "cache"
        folder.mkdir(exist_ok=True)
        tmp = folder / f"{split}.tmp.npz"
        np.savez(tmp, **bank)
        tmp.replace(folder / f"{split}.npz")
        atomic_json(kept, folder / f"{split}_rows.json")
        atomic_json(
            {
                "input_rows": len(rows),
                "kept": len(kept),
                "excluded": excluded,
                "counts": dict(Counter(r["reason"] for r in excluded)),
                "input_replay_max_abs": maxdiff,
                "bounds": bounds,
                "encoder": audits,
                "future_labels_enter_encoder": False,
            },
            folder / f"{split}_audit.json",
        )
        names = [f"{split}.npz", f"{split}_rows.json", f"{split}_audit.json"]
        atomic_json(
            {
                "manifest_sha256": sha256(out / "manifest.json"),
                "files": {n: sha256(folder / n) for n in names},
            },
            folder / f"{split}_index.json",
        )
        progress(f"{split}: {len(kept)}/{len(rows)} rows kept; frozen768 extraction complete")
    del series
    gc.collect()


def transformations(meta, out, train, device):
    if (out / "transforms_lock.json").exists():
        lock = read_json(out / "transforms_lock.json")
        if lock["manifest_sha256"] != sha256(out / "manifest.json"):
            raise ValueError("Transform identity changed")
        verify_files(out, lock["files"])
    else:
        raw = train["x"].reshape(len(train["x"]), -1)
        raw_stats = core.scaler(raw)
        x = core.normalize(raw, raw_stats)
        if min(x.shape) <= meta["pca_rank"]:
            raise ValueError("Insufficient train samples for PCA768")
        _, vectors = parent.ur.up.probe.eigensystem(x, device)
        components = vectors[:, -meta["pca_rank"] :].flip(1).cpu().numpy()
        np.savez(out / "pca.npz", components=components)
        tx = dict(
            raw=raw,
            pca=x @ components,
            simple=train["simple"],
            **{f"embedding_s{s}": train[f"embedding_s{s}"] for s in (42, 43)},
        )
        fit = {
            "raw_for_pca": raw_stats,
            "scalers": {k: core.scaler(v) for k, v in tx.items()},
            "scope": "All eligible unlabeled TRAIN only, also for10% label arms; no validation/research fitting",
        }
        atomic_json(fit, out / "transforms.json")
        atomic_json(
            {
                "manifest_sha256": sha256(out / "manifest.json"),
                "files": {n: sha256(out / n) for n in ("transforms.json", "pca.npz")},
            },
            out / "transforms_lock.json",
        )
    with np.load(out / "pca.npz", allow_pickle=False) as f:
        components = f["components"]
    return read_json(out / "transforms.json"), components


def inputs(bank, fit, components, representation, seed):
    name = f"embedding_s{seed}" if representation == "embedding" else representation
    if representation in ("raw", "pca"):
        x = bank["x"].reshape(len(bank["x"]), -1)
        if representation == "pca":
            x = core.normalize(x, fit["raw_for_pca"]) @ components
    else:
        x = bank[name]
    return np.column_stack((core.normalize(x, fit["scalers"][name]), bank["period"])).astype(
        np.float32
    )


def jobs(meta):
    return [
        {
            "name": f"{rep}_{family}_s{seed}_f{int(100 * fraction)}_lr{li}",
            "representation": rep,
            "family": family,
            "seed": seed,
            "fraction": fraction,
            "lr": lr,
        }
        for seed in meta["reader_seeds"]
        for fraction in meta["fractions"]
        for family in core.READERS
        for rep in core.REPRESENTATIONS
        for li, lr in enumerate(meta["reader_lrs"])
    ]


@torch.no_grad()
def prediction(model, x, target_stats, batch=512):
    normalized = torch.cat([model(v).sort(dim=-1).values for v in x.split(batch)]).cpu().numpy()
    return (
        normalized * np.array(target_stats["scale"])[None, :, None]
        + np.array(target_stats["mean"])[None, :, None]
    )


def fit_reader(meta, out, job, x, v, y, vy, target_stats, ids, device):
    folder = out / job["name"]
    folder.mkdir(exist_ok=True)
    binding = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "job": job,
        "train_index": sha256(out / "cache/train_index.json"),
        "val_index": sha256(out / "cache/val_index.json"),
        "transforms": sha256(out / "transforms_lock.json"),
        "label_ids": ids.tolist(),
        "target_stats": target_stats,
    }
    if (folder / "completion.json").exists():
        done = read_json(folder / "completion.json")
        if done["binding"] != binding:
            raise ValueError("Completed reader source changed")
        verify_files(folder, done["files"])
        return read_json(folder / "summary.json")
    torch.manual_seed(job["seed"])
    model = core.Reader(x.shape[1], job["family"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=job["lr"], weight_decay=0.01)
    train_x = torch.as_tensor(x[ids], device=device)
    train_y = torch.as_tensor(core.normalize(y[ids], target_stats), device=device)
    val_x = torch.as_tensor(v, device=device)
    if (folder / "resume.pt").exists():
        state = torch.load(folder / "resume.pt", map_location=device, weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Reader resume binding changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
    else:
        pred = prediction(model, val_x, target_stats)
        value = core.score(pred, vy, core.SCORE_SCALE)[0]["primary"]
        state = {
            "binding": binding,
            "epoch": 0,
            "best_epoch": 0,
            "best_value": value,
            "initial_validation": value,
            "best": copy.deepcopy(model.state_dict()),
            "history": [],
            "steps": 0,
        }
    for epoch in range(state["epoch"] + 1, meta["reader_epochs"] + 1):
        started = time.monotonic()
        # Reconstructed per-epoch permutation makes interruption recovery deterministic;
        # model has no dropout or other stochastic forward operations.
        order = np.random.default_rng(job["seed"] * 100000 + epoch).permutation(len(ids))
        total, steps = 0.0, 0
        model.train()
        for start in range(0, len(ids), meta["reader_batch"]):
            ix = torch.as_tensor(order[start : start + meta["reader_batch"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = core.pinball(model(train_x[ix]), train_y[ix]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reader loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
            steps += 1
        model.eval()
        value = core.score(prediction(model, val_x, target_stats), vy, core.SCORE_SCALE)[0][
            "primary"
        ]
        if value < state["best_value"]:
            state.update(best_value=value, best_epoch=epoch, best=copy.deepcopy(model.state_dict()))
        state["steps"] += steps
        state["history"].append(
            {
                "epoch": epoch,
                "train_pinball_unsorted": total / len(ids),
                "validation_pinball_sorted": value,
                "steps": steps,
                "samples": len(ids),
                "seconds": time.monotonic() - started,
            }
        )
        state.update(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict())
        atomic_save(state, folder / "resume.pt")
        if epoch == 1 or epoch % 10 == 0:
            progress(
                f"{job['name']}: {epoch}/{meta['reader_epochs']}, val={value:.5f}, best={state['best_epoch']}"
            )
    summary = {
        "job": job,
        "selected_epoch": state["best_epoch"],
        "validation_pinball": state["best_value"],
        "epochs": state["epoch"],
        "updates": state["steps"],
        "labeled_windows": len(ids),
        "input_width": x.shape[1],
        "parameters": sum(p.numel() for p in model.parameters()),
        "epoch0_fallback": state["best_epoch"] == 0,
        "initial_validation": state["initial_validation"],
        "target_stats": target_stats,
    }
    atomic_save(
        {"binding": binding, "model": state["best"], "summary": summary}, folder / "selected.pt"
    )
    atomic_json(state["history"], folder / "history.json")
    atomic_json(summary, folder / "summary.json")
    atomic_json(
        {
            "binding": binding,
            "files": {
                n: sha256(folder / n) for n in ("selected.pt", "summary.json", "history.json")
            },
        },
        folder / "completion.json",
    )
    return summary


def check_selection(meta, out):
    lock = read_json(out / "selection_lock.json")
    if lock["manifest_sha256"] != sha256(out / "manifest.json"):
        raise ValueError("Selection manifest differs")
    verify_files(out, lock["files"])
    if set(lock["candidates"]) != {j["name"] for j in jobs(meta)}:
        raise ValueError("Incomplete fixed reader matrix")
    for job in jobs(meta):
        name = job["name"]
        summary = read_json(out / name / "summary.json")
        history = read_json(out / name / "history.json")
        done = read_json(out / name / "completion.json")
        verify_files(out / name, done["files"])
        if summary != lock["candidates"][name] or summary["job"] != job:
            raise ValueError("Candidate summary differs")
        if (
            done["binding"]["manifest_sha256"] != lock["manifest_sha256"]
            or done["binding"]["job"] != job
        ):
            raise ValueError("Candidate manifest differs")
        n = len(done["binding"]["label_ids"])
        steps = int(np.ceil(n / meta["reader_batch"]))
        if (
            [r["epoch"] for r in history] != list(range(1, meta["reader_epochs"] + 1))
            or any(r["steps"] != steps or r["samples"] != n for r in history)
            or summary["updates"] != steps * meta["reader_epochs"]
            or summary["labeled_windows"] != n
            or summary["epochs"] != meta["reader_epochs"]
        ):
            raise ValueError("Reader training budget differs")
        winner = min(
            [(0, summary["initial_validation"])]
            + [(r["epoch"], r["validation_pinball_sorted"]) for r in history],
            key=lambda v: v[1],
        )
        if winner != (summary["selected_epoch"], summary["validation_pinball"]):
            raise ValueError("Best epoch differs from validation history")
    expected_groups = {
        f"{f}_s{s}_f{int(100 * p)}"
        for f in core.READERS
        for s in meta["reader_seeds"]
        for p in meta["fractions"]
    }
    if set(lock["chosen"]) != expected_groups or set(lock["baselines"]) != expected_groups:
        raise ValueError("Incomplete validation selection groups")
    for group, selected in lock["chosen"].items():
        if set(selected) != set(core.REPRESENTATIONS):
            raise ValueError("Missing representation")
        for rep, name in selected.items():
            choices = [
                v for k, v in lock["candidates"].items() if k.startswith(rep + "_" + group + "_lr")
            ]
            winner = min(choices, key=lambda v: (v["validation_pinball"], -v["job"]["lr"]))
            if name != winner["job"]["name"]:
                raise ValueError("Learning rate differs from validation winner")
        b = lock["baselines"][group]
        if b["chosen"] != min(b["validation_scores"], key=lambda k: (b["validation_scores"][k], k)):
            raise ValueError("Baseline selection differs")
    return lock


def fit_all(meta, out, device):
    if (out / "selection_lock.json").exists():
        return check_selection(meta, out)
    train, rows = cache(out, "train")
    val, val_rows = cache(out, "val")
    fit, components = transformations(meta, out, train, device)
    matrices = {
        (split, rep, seed): inputs(bank, fit, components, rep, seed)
        for split, bank in (("train", train), ("val", val))
        for rep in core.REPRESENTATIONS
        for seed in (42, 43)
    }
    candidates, chosen, baseline = {}, {}, {}
    validation_predictions = {}
    for job in jobs(meta):
        seed, fraction, rep = job["seed"], job["fraction"], job["representation"]
        ids = core.subset(rows, fraction, seed)
        stats = core.scaler(train["targets"][ids])
        summary = fit_reader(
            meta,
            out,
            job,
            matrices["train", rep, seed],
            matrices["val", rep, seed],
            train["targets"],
            val["targets"],
            stats,
            ids,
            device,
        )
        candidates[job["name"]] = summary
        ck = torch.load(out / job["name"] / "selected.pt", map_location=device, weights_only=True)
        reader = core.Reader(summary["input_width"], job["family"]).to(device)
        reader.load_state_dict(ck["model"], strict=True)
        reader.eval().requires_grad_(False)
        p = prediction(reader, torch.as_tensor(matrices["val", rep, seed], device=device), stats)
        value = core.score(p, val["targets"], core.SCORE_SCALE)[0]["primary"]
        if not np.isclose(value, summary["validation_pinball"], atol=1e-7, rtol=2e-5):
            atomic_json(
                {"job": job, "actual": value, "expected": summary["validation_pinball"]},
                out / "reader_replay_failure.json",
            )
            raise ValueError("Selected reader validation replay differs")
        validation_predictions[job["name"]] = p.tolist()
    for seed in (42, 43):
        for fraction in meta["fractions"]:
            ids = core.subset(rows, fraction, seed)
            bf = core.baseline_fit(
                train["targets"][ids],
                train["simple"][ids],
                np.array([r["period"] for r in rows])[ids],
            )
            stats = core.scaler(train["targets"][ids])
            for family in core.READERS:
                group = f"{family}_s{seed}_f{int(100 * fraction)}"
                chosen[group] = {}
                for rep in core.REPRESENTATIONS:
                    eligible = [
                        v
                        for v in candidates.values()
                        if v["job"]["seed"] == seed
                        and v["job"]["fraction"] == fraction
                        and v["job"]["family"] == family
                        and v["job"]["representation"] == rep
                    ]
                    best = min(eligible, key=lambda v: (v["validation_pinball"], -v["job"]["lr"]))
                    chosen[group][rep] = best["job"]["name"]
                scores = {"simple": candidates[chosen[group]["simple"]]["validation_pinball"]}
                for kind in ("constant", "persistence"):
                    pred = core.baseline_predict(
                        bf, val["simple"], [r["period"] for r in val_rows], kind
                    )
                    scores[kind] = core.score(pred, val["targets"], core.SCORE_SCALE)[0]["primary"]
                baseline[group] = {
                    "fit": bf,
                    "validation_scores": scores,
                    "chosen": min(scores, key=lambda k: (scores[k], k)),
                    "target_stats": stats,
                }
    atomic_json(
        {
            "rows": val_rows,
            "targets": val["targets"].tolist(),
            "predictions": validation_predictions,
        },
        out / "validation_predictions.json",
    )
    atomic_json(
        {
            "rows": rows,
            "targets": train["targets"].tolist(),
            "past_features": train["simple"].tolist(),
            "label_ids": {
                f"s{s}_f{int(100 * f)}": core.subset(rows, f, s).tolist()
                for s in meta["reader_seeds"]
                for f in meta["fractions"]
            },
        },
        out / "label_fit_audit.json",
    )
    files = {
        str(p.relative_to(out)): sha256(p)
        for job in jobs(meta)
        for p in (
            (out / job["name"] / "selected.pt"),
            (out / job["name"] / "summary.json"),
            (out / job["name"] / "history.json"),
            (out / job["name"] / "completion.json"),
        )
    }
    files.update(
        {
            n: sha256(out / n)
            for n in (
                "transforms.json",
                "validation_predictions.json",
                "label_fit_audit.json",
                "pca.npz",
                "transforms_lock.json",
                "cache/train_index.json",
                "cache/val_index.json",
            )
        }
    )
    atomic_json(
        {
            "manifest_sha256": sha256(out / "manifest.json"),
            "candidates": candidates,
            "chosen": chosen,
            "baselines": baseline,
            "files": files,
            "all_locked_before_research_extraction": True,
        },
        out / "selection_lock.json",
    )
    return check_selection(meta, out)


@torch.inference_mode()
def evaluate(meta, out, device):
    lock = check_selection(meta, out)
    train, _ = cache(out, "train")
    fit, components = transformations(meta, out, train, device)
    result = {}
    for split in SPLITS[2:]:
        bank, rows = cache(out, split)
        predictions, summaries, decisions = {}, {}, {}
        for group, chosen in lock["chosen"].items():
            reference = lock["baselines"][group]
            stats = reference["target_stats"]
            pred, errors, metrics = {}, {}, {}
            for rep, name in chosen.items():
                summary = lock["candidates"][name]
                job = summary["job"]
                x = inputs(bank, fit, components, rep, job["seed"])
                ck = torch.load(out / name / "selected.pt", map_location=device, weights_only=True)
                if ck["summary"] != summary:
                    raise ValueError("Selected reader summary changed")
                model = core.Reader(x.shape[1], job["family"]).to(device)
                model.load_state_dict(ck["model"], strict=True)
                model.eval().requires_grad_(False)
                pred[rep] = prediction(model, torch.as_tensor(x, device=device), stats)
            for kind in ("constant", "persistence"):
                pred[kind] = core.baseline_predict(
                    reference["fit"], bank["simple"], [r["period"] for r in rows], kind
                )
            strata = {}
            for rep, p in pred.items():
                metrics[rep], errors[rep] = core.score(p, bank["targets"], core.SCORE_SCALE)
                strata[rep] = {}
                for field in ("period", "symbol"):
                    for value in sorted({str(r[field]) for r in rows}):
                        ids = np.array([str(r[field]) == value for r in rows])
                        strata[rep][f"{field}/{value}"] = core.score(
                            p[ids], bank["targets"][ids], core.SCORE_SCALE
                        )[0]
            decisions[group] = core.comparisons(errors, rows, reference["chosen"])
            summaries[group] = {
                "scores": metrics,
                "strata": strata,
                "baseline": reference["chosen"],
            }
            predictions[group] = {k: v.tolist() for k, v in pred.items()}
        atomic_json(
            {"rows": rows, "targets": bank["targets"].tolist(), "predictions": predictions},
            out / f"{split}_predictions.json",
        )
        result[split] = {"groups": summaries, "decisions": decisions}
    primary = all(
        result[s]["decisions"][f"mlp_s{seed}_f100"]["passed"]
        for s in SPLITS[2:]
        for seed in (42, 43)
    )
    atomic_json(
        {
            "status": "exploratory_transfer_gain" if primary else "no_confirmed_transfer_gain",
            "primary_passed": primary,
            "encoder_updates": 0,
            "results": result,
            "uncertainty": "0.1/0.5/0.9 marginal forecasts;80% coverage and width reported, never claimed calibrated by construction",
            "scope": meta["transfer_protocol"]["scope"],
        },
        out / "transfer_metrics.json",
    )
    progress(f"Transfer matrix complete: primary_passed={primary}; no automatic encoder promotion")


def run(source, out, batch, device):
    started = time.monotonic()
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Use a separate output directory")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA required for real training; run on AutoDL")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress("Verifying frozen Macro control best checkpoints and original raw lineage")
        identity = parent.source_identity(source)
        meta = make_manifest(source, identity, batch)
        meta["runtime"] = {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name() if device == "cuda" else device,
        }
        if (out / "manifest.json").exists():
            if read_json(out / "manifest.json") != meta:
                raise ValueError("Run manifest changed; use a new output directory")
        else:
            if any(p.name != ".run.lock" for p in out.iterdir()):
                raise ValueError("Nonempty output without matching manifest")
            atomic_json(meta, out / "manifest.json")
        if (out / "completion.json").exists():
            verify_files(out, read_json(out / "completion.json")["files"])
            progress("Existing complete run verified; no fitting repeated")
            return
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise ValueError(
                "Need2GiB free for caches and reader checkpoints; source files untouched"
            )
        tm = parent.tm(meta)
        if (
            parent.xr.xd.audit_identity(Path(tm["raw_audit"]), Path(tm["raw_root"]))
            != tm["raw_audit_identity"]
        ):
            raise ValueError("Immutable raw lineage differs")
        prepare(meta, out, ("train", "val"), False, device)
        fit_all(meta, out, device)
        prepare(meta, out, ("test",), False, device)
        prepare(meta, out, ("cross_research",), True, device)
        evaluate(meta, out, device)
        if parent.source_identity(source) != identity:
            raise ValueError("Frozen source changed during run")
        check_selection(meta, out)
        if (
            parent.xr.xd.audit_identity(Path(tm["raw_audit"]), Path(tm["raw_root"]))
            != tm["raw_audit_identity"]
        ):
            raise ValueError("Raw source changed during transfer experiment")
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in out.rglob("*")
            if p.is_file()
            and p.name
            not in (
                "resume.pt",
                ".run.lock",
                "audit_status.json",
                "run_status.txt",
                "completion.json",
            )
        }
        atomic_json(
            {
                "status": "complete",
                "source_unchanged": True,
                "encoder_updates": 0,
                "fits": len(jobs(meta)),
                "files": files,
                "seconds_this_invocation": time.monotonic() - started,
            },
            out / "completion.json",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    try:
        run(args.source, args.out, args.batch, "cuda")
    except Exception as error:
        if (
            not isinstance(error, BlockingIOError)
            and args.out.is_dir()
            and (args.out / "manifest.json").is_file()
            and read_json(args.out / "manifest.json").get("schema") == SCHEMA
            and not (args.out / "completion.json").exists()
        ):
            atomic_json({"status": "failed", "error": str(error)}, args.out / "audit_status.json")
        raise


if __name__ == "__main__":
    main()
