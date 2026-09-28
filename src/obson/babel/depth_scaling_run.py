"""Controlled depth scaling with symmetric LR search and longer shallow controls."""

import argparse
import fcntl
import gc
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import bar_semantics_run as previous
from . import causal_pool_run as source_run
from . import depth_scaling as semantics
from . import history_query as hq
from . import research_state_delivery as delivery
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-depth768-v1"
warm, xr = delivery.warm, delivery.xr
ur = warm.ur
rs = previous.rs


def code_identity():
    from . import depth_scaling_evaluate as ev

    return (
        source_run.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (semantics, ev)}
        | {Path(__file__).name: sha256(__file__)}
    )


verify_files = previous.verify_files
source_identity = previous.source_identity


def tm(meta):
    return meta["identity"]["manifest"]["identity"]["training_manifest"]


def root(meta):
    return Path(meta["identity"]["manifest"]["identity"]["training_source"])


def stats(meta):
    return xr.statistics(tm(meta))


def data(meta, split):
    return delivery.data_for(meta["identity"]["manifest"], split)


def start_identity(source, identity):
    source = source.resolve()
    meta, done = read_json(source / "manifest.json"), read_json(source / "completion.json")
    if (
        meta["schema"] != source_run.SCHEMA
        or meta["code_sha256"] != source_run.code_identity()
        or meta["identity"] != identity
        or done["status"] != "complete"
        or not done["source_unchanged"]
        or meta["epochs"] != 100
    ):
        raise ValueError("Completed unchanged causal-pooling100 source required")
    if source_run.start_identity(Path(meta["start"]["source"]), identity) != meta["start"]:
        raise ValueError("Ancestral History Structure source changed")
    keep = {f"plain_s{seed}/best.pt" for seed in (42, 43)}
    files = {
        n: h
        for n, h in done["files"].items()
        if n in keep or Path(n).suffix in (".json", ".jsonl", ".md")
    }
    if not keep.issubset(files):
        raise ValueError("Retained plain bests missing")
    verify_files(source, files)
    trials = read_json(source / "model_selection_lock.json")["trials"]
    for rel in sorted(keep):
        ck = torch.load(source / rel, map_location="cpu", weights_only=True)
        name = rel.split("/")[0]
        seed = int(name.rsplit("_s", 1)[1])
        expected = {
            "manifest_sha256": sha256(source / "manifest.json"),
            "job": {"name": name, "mode": "plain", "seed": seed},
            "phase": "main",
        }
        if (
            ck["metadata"] != expected
            or ck["epoch"] != trials[name]["selected_epoch"]
            or trials[name]["epochs"] != 100
            or ck["validation"] != trials[name]["validation"]
        ):
            raise ValueError("Source identity/selection/budget changed")
    return {
        "source": str(source),
        "files": files,
        "completion_sha256": sha256(source / "completion.json"),
    }


def make_manifest(source, identity, warmup, start, micro=8):
    meta = previous.make_manifest(source, identity, warmup, micro)
    meta.update(
        schema=SCHEMA,
        code_sha256=code_identity(),
        start=start,
        budgets=semantics.budgets(),
        experiments=[
            {
                "name": f"d{depth}_lr{lr}_s{seed}",
                "mode": f"d{depth}",
                "depth": depth,
                "lr_scale": lr,
                "seed": seed,
            }
            for seed in (42, 43)
            for lr in semantics.LR_SCALES
            for depth in semantics.DEPTHS
        ],
        initialization="Same per-seed completed pool-study plain best; append identity prenorm blocks; preserve all inherited states and frozen query scaler. Reset all optimizers equally.",
        architecture={
            "width": 768,
            "heads": 8,
            "ff": 3072,
            "depths": list(semantics.DEPTHS),
            "pooling": False,
        },
        compute={
            "training_matmul_proxy_per_view": {
                str(d): semantics.matmul_proxy(d) for d in semantics.DEPTHS
            },
            "scope": "Analytical forward+backward dense/attention MAC proxy, not exact FLOPs or GPU time; excludes validation, optimizer, nonlinear ops, IO. Ceil shallow epochs. LR search cost separately included in histories.",
        },
        objective="All: uniform query +1.0 remote structure +.25 detached local diagnostic; original query decoder. No new pooling, future targets or consistency objective.",
        training="Two encoder peaks1e-5/3e-5 per depth and seed; both fully trained100. Four-layer trials additionally continue to fixed matmul-matched8/12 budgets. Common sampling prefix, reset-free Adam continuation and warmup/cosine restart at each shallow budget segment. Query/local rates fixed across arms.",
        selection="Original validation-only near/far minimax with epoch0 protection; choose LR by best eligible validation, ties lr1. Same selected trial supplies best AND last; never select on research. Each budget has its own immutable snapshot.",
        decision="Depth8/12 vs4 at100 exposures;12 vs8 secondary; depth8/12 vs longer4 at matched matmul proxy. Near/far gains and retention/utility gates separate; no global architecture superiority claim without both budget axes. Original absolute utility protocol retained.",
        stop="Fixed matrix; no automatic promotion, extra epochs or LR grid. Negative outcome applies to this continuation/init/two-LR budget only, not depth in general.",
    )
    return meta


def learning_rates(meta, job, epoch):
    start = 0
    for end in semantics.stages(meta, job["depth"]):
        if start < epoch <= end:
            return {
                key: xr.growth.learning_rate(
                    epoch - start,
                    end - start,
                    meta[key] * (job["lr_scale"] if key == "encoder_lr" else 1),
                )
                for key in ("encoder_lr", "head_lr", "query_lr")
            }
        start = end
    raise ValueError("Epoch outside prespecified depth budget")


def construct(meta, seed, device, mode="d4"):
    model, query = previous.construct(meta, seed, device)
    rel = f"plain_s{seed}/best.pt"
    path = Path(meta["start"]["source"]) / rel
    if sha256(path) != meta["start"]["files"][rel]:
        raise ValueError("Control starting checkpoint changed")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_mean", "state_scale", "age_encoding"):
        if not torch.equal(query.state_dict()[key].cpu(), ck["query"][key]):
            raise ValueError("Frozen query coordinate/scaler changed")
    model.load_state_dict(ck["model"], strict=True)
    query.load_state_dict(ck["query"], strict=True)
    return semantics.install(model, int(mode[1:]), seed), query


def plan(meta, seed, epoch, n, phase):
    stream = 500 + epoch
    ids, shifts, prefixes, info = xr.plan(
        dict(tm(meta), budget=meta["budget"]), {"seed": seed}, stream, n
    )
    qps = hq.sample_prefixes(len(ids), seed, 500 + stream)
    return (
        ids,
        shifts,
        prefixes,
        qps,
        dict(phase=phase, **info, query_prefixes=ur.bb.cov.ndarray_hash(qps)),
    )


def batch_tensors(d, device):
    return {k: torch.tensor(np.asarray(v), device=device) for k, v in d.items()}


def train_epoch(
    model,
    query,
    builder,
    schedule,
    statistics,
    local,
    batch,
    micro,
    device,
    optimizers,
    mode,
):
    ids, shifts, prefixes, qps, _ = schedule
    enc, head, dec = optimizers
    model.eval()
    query.train()  # All inherited and new dropout are exactly zero.
    total = np.zeros(4)
    steps = 0
    grad_norms = []
    train_started = time.monotonic()
    for left in range(0, len(ids), batch):
        size = min(batch, len(ids) - left)
        for opt in optimizers:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
        for start in range(left, left + size, micro):
            stop = min(start + micro, left + size)
            a, b, c, _, _ = builder(ids[start:stop], shifts[start:stop])
            joined = batch_tensors({k: np.concatenate([a[k], b[k], c[k]]) for k in a}, device)
            ps = torch.tensor(np.tile(prefixes[start:stop], (3, 1)), device=device)
            qp = torch.tensor(np.tile(qps[start:stop], (3, 1)), device=device)
            parts = semantics.losses(
                model,
                query,
                joined,
                ps,
                qp,
                torch.tensor(shifts[start:stop], device=device),
                statistics,
                local,
                mode,
            )
            base, within, qloss, loss = (
                parts[k] for k in ("structure", "consistency", "query", "optimization_value")
            )
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite query training loss")
            (loss.sum() / size).backward()
            total += [float(v.detach().sum()) for v in (base, within, qloss, loss)]
        norms = []
        for parameters in semantics.gradient_groups(model, query):
            norms.append(
                float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            )
        grad_norms.append(norms)
        for opt in optimizers:
            opt.step()
        steps += 1
    result = dict(
        zip(
            ("structure", "consistency", "query", "optimization_value"),
            (total / len(ids)).tolist(),
            strict=True,
        )
    )
    result["gradient_norm_mean"] = np.mean(grad_norms, axis=0).tolist()
    result["gradient_clip_fraction"] = np.mean(np.asarray(grad_norms) > 1, axis=0).tolist()
    result["training_seconds_including_data"] = time.monotonic() - train_started
    return result, steps


@torch.inference_mode()
def validation(meta, model, query, device, seed=42):
    statistics, local = stats(meta)
    d = data(meta, "val")
    total = 0.0
    protected = {"recent": 0.0, "far_primary": 0.0, "far_level": 0.0, "far_trend": 0.0}
    # Original validation selection is exactly the source's declared function.
    old = xr.validation(tm(meta), {"seed": seed}, model, device)
    for start in range(0, len(d["x"]), meta["evaluation_batch"]):
        b = batch_tensors(
            {
                k: v[start : start + meta["evaluation_batch"]]
                for k, v in d.items()
                if k in ("x", "y", "mask")
            },
            device,
        )
        ps = torch.tensor([hq.ba.VAL_PREFIXES] * len(b["x"]), device=device)
        _, p = hq.predict_states(model, query, b["x"], ps)
        y, m = hq.targets(b, ps, statistics)
        total += float(hq.metrics(p, y, m, statistics)["primary"].mean(1).sum())
        near_mask = m.clone()
        near_mask[..., 16:, :] = False
        protected["recent"] += float(
            hq.metrics(p, y, near_mask, statistics)["primary"].mean(1).sum()
        )
        far = semantics.metrics(p[:, -1:], y[:, -1:], m[:, -1:], statistics, lo=65)
        for key in ("primary", "level", "trend"):
            protected["far_" + key] += float(far[key].sum())
    from . import depth_scaling_evaluate as ev

    matched = ev.matched_scores(meta, model, query, d, device, lengths=(64, 128), pair=(64, 128))
    protected["recent"] = float(np.mean(matched["paired64_128"]["actual_error"]["primary"])) * len(
        d["x"]
    )
    protected["gap"] = float(np.mean(matched["paired64_128"]["gap"]["primary"])) * len(d["x"])
    result = {
        "original": old,
        "query": total / len(d["x"]),
        **{k: v / len(d["x"]) for k, v in protected.items()},
    }
    if not np.isfinite(
        [old["selection"], *[v for k, v in result.items() if k != "original"]]
    ).all():
        raise ValueError("Nonfinite validation cannot select a checkpoint")
    return result


def selected_value(result, initial):
    keys = ("query", "recent", "gap", "far_primary", "far_level", "far_trend")
    if any(
        not np.isfinite(initial[k])
        or initial[k] <= 0
        or not np.isfinite(result[k])
        or result[k] < 0
        for k in keys
    ):
        raise ValueError("Finite strictly positive initial selection metrics required")
    for key in keys:
        tolerance = 1.10 if key in ("gap", "far_level", "far_trend") else 1.05
        if result[key] > initial[key] * tolerance:
            return (float("inf"), float("inf"))
    near, far = result["recent"] / initial["recent"], result["far_primary"] / initial["far_primary"]
    return max(near, far), (near + far) / 2


def cpu(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def publish(state, path):
    atomic_save(state, path / "last.pt")
    atomic_save(
        {
            "metadata": state["metadata"],
            "epoch": state["best_epoch"],
            "model": state["best_model"],
            "query": state["best_query"],
            "validation": state["best_validation"],
        },
        path / "best.pt",
    )
    atomic_json(state["history"], path / "history.json")


def verify_history(meta, job, state, n, phase):
    epochs = semantics.stages(meta, job["depth"])[-1]
    if state["epoch"] > epochs or [h["epoch"] for h in state["history"]] != list(
        range(1, state["epoch"] + 1)
    ):
        raise ValueError("Incomplete or excessive training budget")
    updates = (meta["budget"] + meta["batch"] - 1) // meta["batch"]
    active = True
    for h in state["history"]:
        if (
            h["sampling"] != plan(meta, job["seed"], h["epoch"], n, phase)[4]
            or h["views"] != 3 * meta["budget"]
            or h["query_steps"] != updates
            or h["encoder_steps"] != updates * active
            or h["local_steps"] != updates * active
        ):
            raise ValueError("Sampling or optimizer budget mismatch")
        for key in ("encoder_lr", "head_lr", "query_lr"):
            expected = learning_rates(meta, job, h["epoch"])[key]
            if not np.isclose(h[key], expected, rtol=1e-14, atol=0):
                raise ValueError("Learning rate schedule changed")
    candidates = [(0, state["initial_validation"])] + [
        (h["epoch"], h["validation"]) for h in state["history"]
    ]
    chosen = min(candidates, key=lambda v: selected_value(v[1], state["initial_validation"]))
    if (state["best_epoch"], state["best_validation"]) != chosen:
        raise ValueError("Validation-only selection changed")


def worker(out, name, phase, device="cuda"):
    if phase != "main":
        raise ValueError("This study reuses warm bests; no warm training phase")
    meta = read_json(out / "manifest.json")
    if meta["code_sha256"] != code_identity():
        raise ValueError("Worker source code changed")
    job = next(j for j in meta["experiments"] if j["name"] == name)
    path = out / name
    path.mkdir(exist_ok=True)
    metadata = {
        "manifest_sha256": sha256(out / "manifest.json"),
        "job": job if phase == "main" else {"seed": job["seed"]},
        "phase": phase,
    }
    if (path / "completion.json").exists():
        done = read_json(path / "completion.json")
        if done["metadata"] != metadata or done["status"] != "complete":
            raise ValueError("Worker completion changed")
        verify_files(path, done["files"])
        cleanup_completed(path)
        return
    model, query = construct(meta, job["seed"], device, job["mode"])
    fork = meta["start"]["files"][f"plain_s{job['seed']}/best.pt"]
    active = True
    model.requires_grad_(False)
    model.core.encoder.requires_grad_(active)
    model.local_head.requires_grad_(active)
    query.requires_grad_(True)
    enc = (
        torch.optim.AdamW(model.core.encoder.parameters(), lr=meta["encoder_lr"], weight_decay=0.01)
        if active
        else None
    )
    head = (
        torch.optim.AdamW(model.local_head.parameters(), lr=meta["head_lr"], weight_decay=1e-4)
        if active
        else None
    )
    dec = torch.optim.AdamW(query.parameters(), lr=meta["query_lr"], weight_decay=1e-4)
    opts = (enc, head, dec)
    builder = xr.xd.Builder(tm(meta), root(meta))
    statistics, local = stats(meta)
    epochs = semantics.stages(meta, job["depth"])[-1]
    original_decoder = cpu(model.core.decoder)
    if (path / "last.pt").exists():
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if state["metadata"] != metadata or state["fork_sha256"] != fork:
            raise ValueError("Resume identity changed")
        verify_history(meta, job, state, len(builder.plan), phase)
        model.load_state_dict(state["model"])
        query.load_state_dict(state["query"])
        for opt, value in zip(opts, state["optimizers"], strict=True):
            if opt is not None:
                opt.load_state_dict(value)
            elif value is not None:
                raise ValueError("Unexpected resumed encoder optimizer")
        restore_rng(state["rng"])
        expected = (
            state["history"][-1]["validation"] if state["history"] else state["initial_validation"]
        )
        xr.ce.require_nested(
            validation(meta, model, query, device, job["seed"]), expected, "query resume"
        )
    else:
        initial = validation(meta, model, query, device, job["seed"])
        torch.manual_seed(job["seed"] + 20261020)
        state = {
            "metadata": metadata,
            "fork_sha256": fork,
            "epoch": 0,
            "initial_validation": initial,
            "best_epoch": 0,
            "best_validation": initial,
            "best_model": cpu(model),
            "best_query": cpu(query),
            "history": [],
        }

    def save():
        state.update(
            model=cpu(model),
            query=cpu(query),
            optimizers=[None if o is None else o.state_dict() for o in opts],
            rng=rng_state(),
        )
        publish(state, path)

    save()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if state["epoch"] in semantics.stages(meta, job["depth"]):
        snapshot(state, path, state["epoch"])
    for epoch in range(state["epoch"] + 1, epochs + 1):
        started = time.monotonic()
        schedule = plan(meta, job["seed"], epoch, len(builder.plan), phase)
        rates = learning_rates(meta, job, epoch)
        for opt, key in zip(opts, rates, strict=True):
            if opt is not None:
                for group in opt.param_groups:
                    group["lr"] = rates[key]
        train, steps = train_epoch(
            model,
            query,
            builder,
            schedule,
            statistics,
            local,
            meta["batch"],
            meta["micro"],
            device,
            opts,
            job["mode"] if phase == "main" else "warm",
        )
        model.eval()
        query.eval()
        val = validation(meta, model, query, device, job["seed"])
        if selected_value(val, state["initial_validation"]) < selected_value(
            state["best_validation"], state["initial_validation"]
        ):
            state.update(
                best_epoch=epoch, best_validation=val, best_model=cpu(model), best_query=cpu(query)
            )
        state["history"].append(
            dict(
                epoch=epoch,
                sampling=schedule[4],
                views=3 * len(schedule[0]),
                encoder_steps=steps * active,
                local_steps=steps * active,
                query_steps=steps,
                train=train,
                validation=val,
                **rates,
                seconds=time.monotonic() - started,
            )
        )
        state["epoch"] = epoch
        save()
        if epoch in semantics.stages(meta, job["depth"]):
            snapshot(state, path, epoch)
        elapsed = state["history"][-1]["seconds"]
        remaining = elapsed * (epochs - epoch) / 60
        progress(
            f"{path.name}: {epoch}/{epochs}, query_val={val['query']:.6f}, old_val={val['original']['selection']:.6f}, best={state['best_epoch']}, {elapsed:.1f}s/epoch, this_worker_eta~{remaining:.0f}min"
        )
    verify_history(meta, job, state, len(builder.plan), phase)
    if any(
        not torch.equal(v, model.core.decoder.state_dict()[k].cpu())
        for k, v in original_decoder.items()
    ):
        raise ValueError("Fixed PCA decoder changed")
    summary = dict(
        selected_epoch=state["best_epoch"],
        epochs=epochs,
        phase=phase,
        fork_sha256=fork,
        query_parameters=sum(p.numel() for p in query.parameters()),
        encoder_parameters=sum(p.numel() for p in model.core.encoder.parameters()),
        depth=job["depth"],
        lr_scale=job["lr_scale"],
        training_matmul_proxy=semantics.matmul_proxy(job["depth"])
        * sum(h["views"] for h in state["history"]),
        validation=state["best_validation"],
        initial_validation=state["initial_validation"],
        decoder_unchanged=True,
        frozen_parent_unchanged=not active,
        **{
            k: sum(h[k] for h in state["history"])
            for k in ("views", "encoder_steps", "local_steps", "query_steps", "seconds")
        },
        peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
        tail=state["history"][-20:],
    )
    atomic_json(summary, path / "training_summary.json")
    atomic_json(
        {
            k: state[k]
            for k in (
                "metadata",
                "fork_sha256",
                "epoch",
                "initial_validation",
                "best_epoch",
                "best_validation",
                "history",
            )
        },
        path / "training_state.json",
    )
    atomic_json(
        {
            "status": "complete",
            "metadata": metadata,
            "files": {
                n: sha256(path / n)
                for n in ["training_state.json", "history.json", "training_summary.json"]
                + [
                    f"e{e}_{k}.pt"
                    for e in semantics.stages(meta, job["depth"])
                    for k in ("best", "last")
                ]
            },
        },
        path / "completion.json",
    )
    cleanup_completed(path)


def cleanup_completed(path):
    """Remove only this worker's redundant resume files AFTER durable completion.

    All budget best/last weights and histories remain. A crash before completion
    keeps Adam/RNG; a crash after completion repeats this verified cleanup.
    """
    done = read_json(path / "completion.json")
    if done["status"] != "complete":
        raise ValueError("Cannot compact incomplete worker")
    verify_files(path, done["files"])
    for name in ("last.pt", "best.pt"):
        (path / name).unlink(missing_ok=True)


def snapshot(state, path, epoch):
    """Budget endpoints carry weights only; last.pt retains the resumable Adam/RNG."""
    for kind in ("best", "last"):
        best = kind == "best"
        value = {
            "metadata": state["metadata"],
            "budget_epoch": epoch,
            "epoch": state["best_epoch"] if best else epoch,
            "validation": state["best_validation"] if best else state["history"][-1]["validation"],
            "model": state["best_model"] if best else state["model"],
            "query": state["best_query"] if best else state["query"],
        }
        file = path / f"e{epoch}_{kind}.pt"
        if file.exists():
            old = torch.load(file, map_location="cpu", weights_only=True)
            if any(old[k] != value[k] for k in ("metadata", "budget_epoch", "epoch", "validation")):
                raise ValueError("Immutable budget snapshot metadata changed")
            for field in ("model", "query"):
                if set(old[field]) != set(value[field]) or any(
                    not torch.equal(v, old[field][k]) for k, v in value[field].items()
                ):
                    raise ValueError("Immutable budget snapshot weights changed")
        else:
            atomic_save(value, file)


def variants(meta):
    return [
        {
            "name": f"{mode}_s{seed}",
            "mode": mode,
            "seed": seed,
            "depth": int(mode.split("_")[0][1:]),
            "budget": budget,
        }
        for seed in (42, 43)
        for mode, budget in meta["budgets"].items()
    ]


def choose_learning_rate(choices):
    if len(choices) != 2 or {c["lr_scale"] for c in choices} != set(semantics.LR_SCALES):
        raise ValueError("Exactly the prespecified two LR opportunities required")
    if any(len(c["score"]) != 2 or not np.isfinite(c["score"]).all() for c in choices):
        raise ValueError("Finite protected validation scores required")
    return min(choices, key=lambda c: (*c["score"], c["lr_scale"]))


def lock_models(meta, out):
    trials, weights, selections = {}, {}, {}
    mh = sha256(out / "manifest.json")
    for job in meta["experiments"]:
        path = out / job["name"]
        done = read_json(path / "completion.json")
        verify_files(path, done["files"])
        expected = {"manifest_sha256": mh, "job": job, "phase": "main"}
        if done["status"] != "complete" or done["metadata"] != expected:
            raise ValueError("All twelve complete budgets required")
        last = read_json(path / "training_state.json")
        if (
            last["metadata"] != expected
            or last["epoch"] != semantics.stages(meta, job["depth"])[-1]
        ):
            raise ValueError("Incomplete worker or changed identity")
        verify_history(
            meta, job, last, read_json(root(meta) / "train_cross_plan.json")["eligible"], "main"
        )
        h = read_json(path / "history.json")
        if h != last["history"]:
            raise ValueError("Training history differs from checkpoint")
        summary = read_json(path / "training_summary.json")
        for key in ("views", "encoder_steps", "query_steps", "local_steps", "seconds"):
            if summary[key] != sum(r[key] for r in h):
                raise ValueError("Summary budget mismatch")
        if (
            summary["selected_epoch"] != last["best_epoch"]
            or summary["validation"] != last["best_validation"]
            or summary["initial_validation"] != last["initial_validation"]
        ):
            raise ValueError("Summary selection mismatch")
        fork = meta["start"]["files"][f"plain_s{job['seed']}/best.pt"]
        if summary["fork_sha256"] != fork or last["fork_sha256"] != fork:
            raise ValueError("Shared fork mismatch")
        trials[job["name"]] = summary
        for budget in semantics.stages(meta, job["depth"]):
            eligible = [(0, last["initial_validation"])] + [
                (x["epoch"], x["validation"]) for x in h[:budget]
            ]
            chosen = min(eligible, key=lambda x: selected_value(x[1], last["initial_validation"]))
            for kind in ("best", "last"):
                ck = torch.load(
                    path / f"e{budget}_{kind}.pt", map_location="cpu", weights_only=True
                )
                target = chosen if kind == "best" else (budget, h[budget - 1]["validation"])
                if (
                    ck["metadata"] != expected
                    or ck["budget_epoch"] != budget
                    or (ck["epoch"], ck["validation"]) != target
                ):
                    raise ValueError("Budget-specific checkpoint selection changed")
    for seed in (42, 43):
        initial = trials[f"d4_lr1_s{seed}"]["initial_validation"]
        for job in meta["experiments"]:
            if job["seed"] == seed:
                xr.ce.require_nested(
                    trials[job["name"]]["initial_validation"], initial, "same-seed epoch0"
                )
    for v in variants(meta):
        choices = []
        for lr in semantics.LR_SCALES:
            trial = f"d{v['depth']}_lr{lr}_s{v['seed']}"
            p = out / trial / f"e{v['budget']}_best.pt"
            ck = torch.load(p, map_location="cpu", weights_only=True)
            score = selected_value(ck["validation"], trials[trial]["initial_validation"])
            choices.append(
                {
                    "trial": trial,
                    "lr_scale": lr,
                    "selected_epoch": ck["epoch"],
                    "validation": ck["validation"],
                    "score": list(score),
                }
            )
        choice = choose_learning_rate(choices)
        selections[v["name"]] = dict(
            choice, budget=v["budget"], depth=v["depth"], candidates=choices
        )
        weights[v["name"]] = {
            kind: {
                "path": f"{choice['trial']}/e{v['budget']}_{kind}.pt",
                "sha256": sha256(out / choice["trial"] / f"e{v['budget']}_{kind}.pt"),
            }
            for kind in ("best", "last")
        }
    lock = {"manifest_sha256": mh, "trials": trials, "selections": selections, "weights": weights}
    if (out / "model_selection_lock.json").exists() and read_json(
        out / "model_selection_lock.json"
    ) != lock:
        raise ValueError("Locked LR/model selection changed")
    atomic_json(lock, out / "model_selection_lock.json")
    return lock


def dispatch(out, names, phase, jobs):
    pending = list(names)
    running = []
    try:
        while pending or running:
            while pending and len(running) < jobs:
                name = pending.pop(0)
                p = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "obson.babel.depth_scaling_run",
                        "--out",
                        str(out),
                        "--worker",
                        name,
                        "--phase",
                        phase,
                    ]
                )
                running.append((name, p))
            for name, p in list(running):
                code = p.poll()
                if code is None:
                    continue
                running.remove((name, p))
                if code:
                    raise RuntimeError(
                        f"Worker {name}/{phase} failed with exit{code}; traceback is in this log"
                    )
            if running:
                time.sleep(0.5)
    finally:
        for _, p in running:
            p.terminate()
        for _, p in running:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


@torch.no_grad()
def preflight(meta, out, device, jobs=2):
    d = data(meta, "val")
    results = []
    for seed in (42, 43):
        model, query = construct(meta, seed, device)
        model.eval()
        query.eval()
        expected = read_json(Path(meta["start"]["source"]) / "model_selection_lock.json")["trials"][
            f"plain_s{seed}"
        ]["validation"]["original"]
        actual = xr.validation(tm(meta), {"seed": seed}, model, device)
        xr.ce.require_nested(actual, expected, "retained uniform validation")
        x = torch.tensor(np.asarray(d["x"][:4]), device=device)
        causal = ur.pf.er.trained_causality(model, x)
        if causal["status"] == "failed":
            raise ValueError("Noncausal source encoder")
        z = model.core.encoder(x)[:, 63]
        subset = query(z, torch.tensor([1, 16, 63], device=device))
        whole = query(z)[:, [0, 15, 62]]
        numerical = warm.recheck.array_replay(subset.cpu().numpy(), whole.cpu().numpy())
        if not numerical["passed"]:
            raise ValueError("Query set changes individual answers")
        results.append(
            {"seed": seed, "validation": actual, "causality": causal, "query_subset": numerical}
        )
    atomic_json(results, out / "preflight.json")
    builder = xr.xd.Builder(tm(meta), root(meta))
    schedule = plan(meta, 42, 1, len(builder.plan), "main")
    ids, shifts, ps, qp, _ = schedule
    n = min(meta["micro"], len(ids))
    a, b, c, _, _ = builder(ids[:n], shifts[:n])
    batch = batch_tensors({k: np.concatenate([a[k], b[k], c[k]]) for k in a}, device)
    prefixes = torch.tensor(np.tile(ps[:n], (3, 1)), device=device)
    queries = torch.tensor(np.tile(qp[:n], (3, 1)), device=device)
    gradients = []
    reference, reference_query = construct(meta, 42, device)
    reference.eval()
    reference_query.eval()
    expected = reference.core.encoder(batch["x"])
    for mode in ("d4", "d8", "d12"):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        model, query = construct(meta, 42, device, mode)
        model.eval().requires_grad_(False)
        model.core.encoder.requires_grad_(True)
        model.local_head.requires_grad_(True)
        query.train().requires_grad_(True)
        if not torch.equal(model.core.encoder(batch["x"]), expected):
            raise ValueError("Depth expansion epoch0 does not preserve source function")
        before = (ur.bb.state_signature(model), ur.bb.state_signature(query))
        with torch.enable_grad():
            parts = semantics.losses(
                model,
                query,
                batch,
                prefixes,
                queries,
                torch.tensor(shifts[:n], device=device),
                *stats(meta),
                mode,
            )
            loss = parts["optimization_value"].mean()
            if not torch.isfinite(loss):
                raise ValueError("Preflight loss is nonfinite")
            loss.backward()
        for params in semantics.gradient_groups(model, query):
            grads = [p.grad for p in params if p.requires_grad and p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise ValueError("Preflight gradient missing or nonfinite")
        if before != (ur.bb.state_signature(model), ur.bb.state_signature(query)):
            raise ValueError("Zero-update preflight changed source state")
        appended = list(model.core.encoder.backbone.layers)[4:]
        if appended and any(
            layer.self_attn.out_proj.weight.grad is None
            or layer.linear2.weight.grad is None
            or not (
                layer.self_attn.out_proj.weight.grad.abs().sum()
                + layer.linear2.weight.grad.abs().sum()
                > 0
            )
            for layer in appended
        ):
            raise ValueError("New identity layers have no learning signal")
        memory = None
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            peak_increment = torch.cuda.max_memory_allocated() - baseline_bytes
            moments = 8 * sum(
                p.numel()
                for module in (model.core.encoder, model.local_head, query)
                for p in module.parameters()
            )
            memory = int(1.25 * (peak_increment + moments)) + 384 * 1024**2
        gradients.append(
            {
                "mode": mode,
                "loss": float(loss),
                "finite": True,
                "function_preserved": True,
                "new_layers_receive_gradient": True,
                "estimated_worker_bytes": memory,
            }
        )
        del model, query, parts, loss, params, grads, appended
    report = {
        "micro": n,
        "views": 3 * n,
        "optimizer_steps": 0,
        "gradients": gradients,
        "jobs": jobs,
    }
    if str(device).startswith("cuda"):
        gc.collect()
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        # Driver releases references/cache before workers; using current free is conservative.
        need = jobs * max(g["estimated_worker_bytes"] for g in gradients)
        report.update(
            estimated_workers_bytes=need, available_bytes=free, memory_passed=need <= free
        )
    atomic_json(report, out / "gradient_preflight.json")
    if report.get("memory_passed") is False:
        raise ValueError(
            "Parallel depth memory preflight failed; use BABEL_DEPTH_JOBS=1 or a new output with smaller micro. No optimizer updates performed."
        )


def run(start_source, out, jobs=2, micro=8, device="cuda", evaluate_only=False):
    from . import depth_scaling_evaluate as ev

    start_source, out = start_source.resolve(), out.resolve()
    source_meta = read_json(start_source / "manifest.json")
    source = Path(source_meta["source"])
    progress("Verifying original lineage and completed pool-study plain starting checkpoints")
    identity = source_identity(source)
    warm.check_output(source, out, identity["manifest"]["identity"])
    warm.check_output(start_source, out)
    warm_source = Path(source_meta["warmup"]["source"])
    warm.check_output(warm_source, out)
    warmup = previous.warm_identity(warm_source, identity)
    start = start_identity(start_source, identity)
    meta = make_manifest(source, identity, warmup, start, micro)
    source_runtime = read_json(Path(identity["manifest"]["source"]) / "runtime.json")
    if (
        source_runtime["torch"] != str(torch.__version__)
        or source_runtime["numpy"] != np.__version__
    ):
        raise ValueError("Retain original Torch/NumPy environment")
    if out.exists() and any(out.iterdir()) and read_json(out / "manifest.json") != meta:
        raise ValueError("Use new output for different source/config")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(meta, out / "manifest.json")
    if (out / "completion.json").exists():
        done = read_json(out / "completion.json")
        if done["status"] != "complete" or not done["source_unchanged"]:
            raise ValueError("Invalid experiment completion")
        verify_files(out, done["files"])
        progress("Completed query experiment verified; no repeated training or fitting")
        return
    if evaluate_only and any(
        not (out / j["name"] / "completion.json").exists() for j in meta["experiments"]
    ):
        raise ValueError("Evaluation-only requires all twelve completed training workers")
    only_evaluation_left = all(
        (out / j["name"] / "completion.json").exists() for j in meta["experiments"]
    )
    check_disk(out, jobs, meta, evaluate_only or only_evaluation_left)
    started = time.monotonic()
    atomic_json(
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
            "jobs": jobs,
            "micro": micro,
        },
        out / "runtime.json",
    )
    xr.xd.verify_data(tm(meta), root(meta))
    progress("Replaying original validation and checking causal query interface before training")
    preflight(meta, out, device, jobs)
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    ev.audit_sources(meta, out, device)
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    if not evaluate_only:
        dispatch(out, [j["name"] for j in meta["experiments"]], "main", jobs)
    lock_models(meta, out)
    ev.fit_readouts(meta, out, device)
    ev.evaluate(meta, out, device)
    if (
        source_identity(source) != identity
        or previous.warm_identity(warm_source, identity) != warmup
        or start_identity(start_source, identity) != start
    ):
        raise ValueError("Original source changed")
    atomic_json(
        {
            "status": "complete",
            "source_unchanged": True,
            "seconds_this_invocation": time.monotonic() - started,
            "files": {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file()
                and p != out / "completion.json"
                and p.suffix not in (".tmp", ".log")
                and p.name != "run_status.txt"
            },
        },
        out / "completion.json",
    )


def storage_budget(meta, jobs):
    estimates = []
    for job in meta["experiments"]:
        model, query = construct(meta, job["seed"], "cpu", job["mode"])
        state_bytes = sum(
            t.numel() * t.element_size() for m in (model, query) for t in m.state_dict().values()
        )
        trained = sum(
            p.numel() for m in (model.core.encoder, model.local_head, query) for p in m.parameters()
        )
        # Last has current+best states and two FP32 Adam moments. Best is one state.
        last = 2 * state_bytes + 8 * trained
        estimates.append(
            {
                "name": job["name"],
                "state_bytes": state_bytes,
                "adam_parameters": trained,
                "last_bytes": last,
                "best_bytes": state_bytes,
                "milestone_bytes": 2 * len(semantics.stages(meta, job["depth"])) * state_bytes,
            }
        )
        del model, query
    files = sum(v["milestone_bytes"] for v in estimates)
    # Completed trials retain milestones only. Each active worker can hold live
    # last+best plus an atomic replacement of last concurrently with milestones.
    atomic_temporary = jobs * max(2 * v["last_bytes"] + v["best_bytes"] for v in estimates)
    return {
        "checkpoints": estimates,
        "atomic_temporary_bytes": atomic_temporary,
        "total_bytes": int(1.25 * (files + atomic_temporary)) + 1024**3,
    }


def check_disk(out, jobs, meta, evaluate_only=False):
    estimate = storage_budget(meta, jobs)
    existing = sum(
        p.stat().st_size for p in out.rglob("*.pt") if p.is_file() and not p.is_symlink()
    )
    required = 1024**3 if evaluate_only else max(1024**3, estimate["total_bytes"] - existing)
    free = shutil.disk_usage(out).free
    report = {
        "free_bytes": free,
        "additional_bytes_required": required,
        "existing_checkpoint_bytes": existing,
        "estimate": estimate,
        "jobs": jobs,
        "passed": free >= required,
    }
    atomic_json(report, out / "disk_preflight.json")
    if free < required:
        raise ValueError(
            f"Need {required / 1024**3:.1f}GiB free; have {free / 1024**3:.1f}GiB. Source files were not deleted."
        )
    progress(
        f"Disk budget from actual state/optimizer sizes: need {required / 1024**3:.1f}GiB, free {free / 1024**3:.1f}GiB"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="checkpoints/babel_causal_pool768")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    p.add_argument("--micro", type=int, choices=(8, 16, 32, 64), default=8)
    p.add_argument("--worker")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--phase", choices=("main",), default="main")
    a = p.parse_args()
    ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError("Real training/evaluation runs on AutoDL CUDA")
    out = Path(a.out).resolve()
    if a.worker:
        worker(out, a.worker, a.phase)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ("." + out.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Query experiment already running") from None
        run(Path(a.source), out, a.jobs, a.micro, evaluate_only=a.evaluate_only)


if __name__ == "__main__":
    main()
