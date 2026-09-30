"""Shared macro paths through fixed original readouts; finite controlled encoder study."""

import argparse
import copy
import fcntl
import gc
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import depth_scaling_evaluate as de
from . import depth_scaling_run as parent
from . import grounded_history_run as grounded
from . import grounded_readout_audit as audit
from . import macro_history as core
from . import macro_history_data as macro_data
from . import shared_history_run as prior
from .ae_extend import atomic_json, restore_rng, rng_state
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-macro-history768-v1"
KEYS = (*core.FAMILIES, "query_fixed", "structure", "local", "optimization_value")
tm, root, stats, data = parent.tm, parent.root, parent.stats, parent.data
xr, ur = parent.xr, parent.ur


def code_identity():
    from . import macro_history_evaluate as ev

    return (
        audit.gr.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (audit, core, macro_data, ev)}
        | {Path(__file__).name: sha256(__file__)}
    )


def source_identity(source):
    source = source.resolve()
    meta = read_json(source / "manifest.json")
    done = read_json(source / "completion.json")
    if (
        meta["schema"] != audit.SCHEMA
        or done["status"] != "complete"
        or not done["source_unchanged"]
    ):
        raise ValueError("Completed immutable matched-readout audit required")
    parent.verify_files(source, done["files"])
    expected = audit.gr.code_identity() | {Path(audit.__file__).name: sha256(audit.__file__)}
    if meta["code_sha256"] != expected:
        raise ValueError("Original audit code differs")
    ground = Path(meta["source"]["source"])
    if audit.source_identity(ground) != meta["source"]:
        raise ValueError("Readout audit/grounded source changed")
    task = meta["source"]["task_source"]
    if grounded.source_identity(Path(task["source"])) != task:
        raise ValueError("Control source lineage changed")
    return dict(
        task,
        readout_audit=str(source),
        audit_files=done["files"],
        audit_completion_sha256=sha256(source / "completion.json"),
        grounded_source=str(ground),
        grounded_identity=meta["source"],
    )


def make_manifest(source, identity, micro=16):
    meta = copy.deepcopy(read_json(Path(identity["source"]) / "manifest.json"))
    meta.update(
        schema=SCHEMA,
        code_sha256=code_identity(),
        task_source=identity,
        epochs=100,
        budget=4800,
        micro=micro,
        encoder_lr=3e-5,
        head_lr=3e-5,
        query_lr=3e-4,
        experiments=[
            {"name": f"{mode}_s{seed}", "mode": mode, "seed": seed}
            for seed in (42, 43)
            for mode in core.MODES
        ],
        sampling_stream_start=1091,
        initialization="Shared History control best100/99; strict restore and remove unused free auxiliary head. Independent immutable clone of original query for macro reading; no new trainable head.",
        objective="Original query0.3+structure1+detached local0.25; content adds0.10 fixed-reader physical macro reconstruction; aligned adds0.01 same-content consistency.",
        training="Six100-epoch arms,4800 triplets/epoch,batch128,38 updates; identical samples/LR; macro reader frozen in all arms.",
        selection="Original protected price minimax with epoch0 fallback; no macro/research selection.",
        macro={
            "train_bands": [list(b) for b in core.TRAIN_BANDS],
            "held_bands": [list(b) for b in core.HELD_BANDS],
            "bins": 4,
            "min_nodes_per_bin": 2,
            "binning": "Four chronological groups of audited nodes (array_split); shared geometry across views, not equal-duration bins or interpolation",
            "content_weight": core.CONTENT_WEIGHT,
            "alignment_weight": core.ALIGN_WEIGHT,
            "cross_training": "15->30",
            "units": "Common oldest-node-relative log-price quarter means and three differences; training delta_scale times sqrt(shared fine-bar duration). No sample volatility normalization.",
        },
        decision="content/control and aligned/content; original information and utility gates plus physical content retained. No automatic promotion.",
        stop="Fixed six100-epoch arms; no extension, weight grid or automatic promotion; validation-only zero-fit fixed-reader gate precedes training.",
    )
    meta.pop("shared", None)
    return meta


def construct(meta, seed, device):
    source = Path(meta["task_source"]["source"])
    if sha256(source / "manifest.json") != meta["task_source"]["files"]["manifest.json"]:
        raise ValueError("Source manifest changed")
    model, query = prior.construct(read_json(source / "manifest.json"), seed, device)
    item = meta["task_source"]["chosen"][str(seed)]
    path = source / item["path"]
    if sha256(path) != item["sha256"]:
        raise ValueError("Chosen source checkpoint changed")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["epoch"] != item["selected_epoch"] or ck["budget_epoch"] != item["budget"]:
        raise ValueError("Wrong source epoch")
    for k in ("state_mean", "state_scale", "age_encoding"):
        if not torch.equal(query.state_dict()[k].cpu(), ck["query"][k]):
            raise ValueError("Frozen query coordinates changed")
    model.load_state_dict(ck["model"], strict=True)
    query.load_state_dict(ck["query"], strict=True)
    del query.shared
    return model.eval(), query.eval()


@lru_cache(maxsize=8)
def read_plan(path):
    return read_json(Path(path) / "train_cross_plan.json")["rows"]


def plan_rows(meta):
    return read_plan(str(root(meta)))


@lru_cache(maxsize=8)
def training_ids(path):
    return tuple(macro_data.eligible(read_plan(path)))


def plan(meta, seed, epoch, n):
    rows = plan_rows(meta)
    if len(rows) != n:
        raise ValueError("Training pool changed")
    stream = meta["sampling_stream_start"] + epoch
    ids, shifts = macro_data.schedule(
        rows, seed, stream, meta["budget"], training_ids(str(root(meta)))
    )
    ps = core.hq.ba.position_plan(len(ids), seed, stream)
    qs = core.hq.sample_prefixes(len(ids), seed, stream)
    info = {
        "phase": "main",
        "stream": stream,
        "ids": ur.bb.cov.ndarray_hash(ids),
        "shifts": ur.bb.cov.ndarray_hash(shifts),
        "prefixes": ur.bb.cov.ndarray_hash(ps),
        "query_prefixes": ur.bb.cov.ndarray_hash(qs),
        "unique_triplets": len(np.unique(ids)),
    }
    return ids, shifts, ps, qs, info


def rates(meta, epoch):
    return {
        k: xr.growth.learning_rate(epoch, meta["epochs"], meta[k])
        for k in ("encoder_lr", "head_lr", "query_lr")
    }


def make_batch(builder, schedule, start, stop, device):
    ids, shifts, prefixes, qps, _ = schedule
    a, b, c, _, _ = builder(ids[start:stop], shifts[start:stop])
    batch = parent.batch_tensors({k: np.concatenate([a[k], b[k], c[k]]) for k in a}, device)
    ps = torch.tensor(np.tile(prefixes[start:stop], (3, 1)), device=device)
    qs = torch.tensor(np.tile(qps[start:stop], (3, 1)), device=device)
    selected = [builder.plan[i] for i in ids[start:stop]]
    geometry, active = macro_data.tensors(selected, shifts[start:stop], device)
    return batch, ps, qs, torch.tensor(shifts[start:stop], device=device), geometry, active


def train_epoch(
    meta, job, model, query, reader, builder, schedule, optimizers, device, max_steps=None
):
    model.eval()
    query.eval()
    totals = dict.fromkeys(KEYS, 0.0)
    aux_totals = {"content": 0.0, "alignment": 0.0}
    norms = []
    steps = 0
    seen = 0
    first_update_l2 = None
    statistics, local = stats(meta)
    for left in range(0, len(schedule[0]), meta["batch"]):
        size = min(meta["batch"], len(schedule[0]) - left)
        model.zero_grad(set_to_none=True)
        query.zero_grad(set_to_none=True)
        for start in range(left, left + size, meta["micro"]):
            stop = min(start + meta["micro"], left + size)
            b, ps, qs, sh, geometry, active = make_batch(builder, schedule, start, stop, device)
            target = core.targets(b, geometry, statistics)
            parts, z = core.losses(model, query, b, ps, qs, sh, statistics, local)
            aux, components, _ = core.auxiliary(
                reader, z, geometry, target, active, job["mode"], statistics
            )
            loss = (parts["optimization_value"] + aux).sum() / size
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite macro objective")
            loss.backward()
            for k in KEYS:
                totals[k] += float(
                    (parts[k] + (aux if k == "optimization_value" else 0)).detach().sum()
                )
            for k in aux_totals:
                aux_totals[k] += float(components[k].detach().sum())
        norms.append(
            [
                float(torch.nn.utils.clip_grad_norm_(g, 1.0, error_if_nonfinite=True))
                for g in core.groups(model, query)
            ]
        )
        if optimizers is not None:
            parameters = list(model.core.encoder.parameters())
            before = [p.detach().clone() for p in parameters] if steps == 0 else None
            for opt in optimizers:
                opt.step()
            if before is not None:
                first_update_l2 = float(
                    torch.stack(
                        [
                            (p.detach() - v).square().sum()
                            for p, v in zip(parameters, before, strict=True)
                        ]
                    )
                    .sum()
                    .sqrt()
                )
                del before
        steps += 1
        seen += size
        if max_steps is not None and steps >= max_steps:
            break
    return dict(
        **{k: v / seen for k, v in totals.items()},
        auxiliary={k: v / seen for k, v in aux_totals.items()},
        gradient_norm_mean=np.mean(norms, axis=0).tolist(),
        gradient_clip_fraction=np.mean(np.asarray(norms) > 1, axis=0).tolist(),
        encoder_first_update_l2=first_update_l2,
    ), steps


@torch.inference_mode()
def validation(meta, model, query, device):
    model.eval()
    query.eval()
    d = data(meta, "val")
    statistics, _ = stats(meta)
    parts = dict.fromkeys((*core.FAMILIES, "primary"), 0.0)
    far = 0.0
    for start in range(0, len(d["x"]), meta["evaluation_batch"]):
        b = parent.batch_tensors(
            {
                k: v[start : start + meta["evaluation_batch"]]
                for k, v in d.items()
                if k in ("x", "y", "mask")
            },
            device,
        )
        ps = torch.tensor([core.hq.ba.VAL_PREFIXES] * len(b["x"]), device=device)
        _, pred = core.hq.predict_states(model, query, b["x"], ps)
        y, m = core.hq.targets(b, ps, statistics)
        sc = core.hq.metrics(pred, y, m, statistics)
        for k in parts:
            parts[k] += float(sc[k].mean(1).sum())
        far += float(
            core.structure.metrics(pred[:, -1:], y[:, -1:], m[:, -1:], statistics, lo=65)[
                "primary"
            ].sum()
        )
    parts = {k: v / len(d["x"]) for k, v in parts.items()}
    matched = de.matched_scores(meta, model, query, d, device, lengths=(64, 128), pair=(64, 128))
    pair = matched["paired64_128"]
    near = float(np.mean(np.asarray(pair["actual_error"]["path"])))
    gap = float(np.mean(np.asarray(pair["gap"]["path"])))
    result = {
        "query_price": float(core.price(parts)),
        "near_price": near,
        "far": far / len(d["x"]),
        "gap_price": gap,
        "query_fixed": parts["primary"],
        "activity": parts["activity"],
        "families": parts,
    }
    if not np.isfinite([v for k, v in result.items() if k != "families"]).all():
        raise ValueError("Nonfinite validation")
    return result


def selected_value(v, initial):
    for k in ("query_price", "near_price", "far", "gap_price"):
        if initial[k] <= 0 or not np.isfinite(initial[k]) or not np.isfinite(v[k]) or v[k] < 0:
            raise ValueError("Invalid selection metric")
        if v[k] > initial[k] * (1.10 if k == "gap_price" else 1.05):
            return (float("inf"), float("inf"))
    a, b = v["near_price"] / initial["near_price"], v["far"] / initial["far"]
    return max(a, b), (a + b) / 2


def verify_history(meta, job, state, n):
    h = state["history"]
    if state["epoch"] != len(h) or len(h) > meta["epochs"]:
        raise ValueError("Invalid history length")
    steps = (meta["budget"] + meta["batch"] - 1) // meta["batch"]
    for epoch, row in enumerate(h, 1):
        if (
            row["epoch"] != epoch
            or row["sampling"] != plan(meta, job["seed"], epoch, n)[4]
            or row["views"] != 3 * meta["budget"]
            or row["steps"] != steps
        ):
            raise ValueError("Sampling or update budget changed")
        if any(
            not np.isclose(row[k], v, rtol=1e-14, atol=0) for k, v in rates(meta, epoch).items()
        ):
            raise ValueError("LR changed")
    best = min(
        [(0, state["initial_validation"])] + [(x["epoch"], x["validation"]) for x in h],
        key=lambda x: selected_value(x[1], state["initial_validation"]),
    )
    if (state["best_epoch"], state["best_validation"]) != best:
        raise ValueError("Validation selection changed")


def component_gradients(meta, job, model, query, reader, builder, device):
    schedule = plan(meta, job["seed"], 1, len(builder.plan))
    b, ps, qs, sh, geometry, active = make_batch(builder, schedule, 0, meta["micro"], device)
    statistics, local = stats(meta)
    target = core.targets(b, geometry, statistics)
    parts, z = core.losses(model, query, b, ps, qs, sh, statistics, local)
    _, aux, _ = core.auxiliary(reader, z, geometry, target, active, job["mode"], statistics)
    values = {
        "price": 0.7 * core.price(parts).mean(),
        "activity": 0.3 * parts["activity"].mean(),
        "structure": parts["structure"].mean(),
        "content": core.CONTENT_WEIGHT * aux["content"].mean()
        if job["mode"] != "control"
        else parts["structure"].mean() * 0,
        "alignment": core.ALIGN_WEIGHT * aux["alignment"].mean()
        if job["mode"] == "aligned"
        else parts["structure"].mean() * 0,
    }
    parameters = [p for p in model.core.encoder.parameters() if p.requires_grad]
    gradients = {}
    for name, value in values.items():
        gradients[name] = torch.cat(
            [
                torch.zeros_like(p).flatten().cpu() if g is None else g.detach().flatten().cpu()
                for p, g in zip(
                    parameters,
                    torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True),
                    strict=True,
                )
            ]
        )
    norms = {k: float(v.norm()) for k, v in gradients.items()}
    cosines = {
        f"{a}/{b}": float(torch.dot(gradients[a], gradients[b]) / (norms[a] * norms[b]))
        if norms[a] * norms[b] > 1e-20
        else None
        for a in gradients
        for b in gradients
        if a < b
    }
    return {
        "norms": norms,
        "cosines": cosines,
        "scope": "One fixed training microbatch, raw weighted encoder gradients before clipping/Adam; not a no-regression guarantee.",
    }


def gradient_diagnostic(meta, job, model, query, reader, builder, device):
    before = (ur.bb.state_signature(model), ur.bb.state_signature(query))
    result, _ = train_epoch(
        meta,
        job,
        model,
        query,
        reader,
        builder,
        plan(meta, job["seed"], 1, len(builder.plan)),
        None,
        device,
        max_steps=1,
    )
    if before != (ur.bb.state_signature(model), ur.bb.state_signature(query)):
        raise ValueError("Diagnostic updated weights")
    return {
        "optimizer_steps": 0,
        "effective_batch": meta["batch"],
        "mode": job["mode"],
        "metrics": result,
        "components": component_gradients(meta, job, model, query, reader, builder, device),
        "scope": "One fixed training effective batch; fixed-reader macro content and reconstruction, no update.",
    }


def require_gate(out):
    status = read_json(out / "audit_status.json")
    if status["status"] != "eligible" or not status["qualified"]:
        raise ValueError("Zero-fit macro reader gate did not pass")
    for seed in (42, 43):
        r = read_json(out / f"s{seed}_macro_gate.json")
        if r["manifest_sha256"] != sha256(out / "manifest.json") or not r["passed"]:
            raise ValueError("Macro gate binding changed")
        parent.verify_files(out, r["files"])
        from . import macro_history_evaluate as ev

        name = f"s{seed}_macro_baseline.json"
        if set(r["files"]) != {name}:
            raise ValueError("Macro gate prediction binding differs")
        actual = ev.gate_metrics(read_json(out / name))
        if any(actual[k] != r[k] for k in actual):
            raise ValueError("Macro gate arithmetic or selection changed")


def worker(out, name, device="cuda"):
    meta = read_json(out / "manifest.json")
    job = next(j for j in meta["experiments"] if j["name"] == name)
    if meta["code_sha256"] != code_identity():
        raise ValueError("Bound code changed")
    require_gate(out)
    macro_data.verify(meta, out)
    path = out / name
    path.mkdir(exist_ok=True)
    metadata = {"manifest_sha256": sha256(out / "manifest.json"), "job": job, "phase": "main"}
    if (path / "completion.json").exists():
        done = read_json(path / "completion.json")
        if done["status"] != "complete" or done["metadata"] != metadata:
            raise ValueError("Worker identity changed")
        parent.cleanup_completed(path)
        return
    model, query = construct(meta, job["seed"], device)
    reader = copy.deepcopy(query).eval().requires_grad_(False)
    reader_signature = ur.bb.state_signature(reader)
    model.requires_grad_(False)
    model.core.encoder.requires_grad_(True)
    model.local_head.requires_grad_(True)
    query.requires_grad_(True)
    fixed = parent.cpu(model.core.decoder)
    opts = [
        torch.optim.AdamW(g, lr=meta[k], weight_decay=wd)
        for g, k, wd in zip(
            core.groups(model, query),
            ("encoder_lr", "head_lr", "query_lr"),
            (0.01, 1e-4, 1e-4),
            strict=True,
        )
    ]
    builder = xr.xd.Builder(tm(meta), root(meta))
    fork = meta["task_source"]["chosen"][str(job["seed"])]["sha256"]
    if (path / "last.pt").exists():
        state = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        if state["metadata"] != metadata or state["fork_sha256"] != fork:
            raise ValueError("Resume identity changed")
        verify_history(meta, job, state, len(builder.plan))
        model.load_state_dict(state["model"], strict=True)
        query.load_state_dict(state["query"], strict=True)
        for o, v in zip(opts, state["optimizers"], strict=True):
            o.load_state_dict(v)
        restore_rng(state["rng"])
        expected = (
            state["history"][-1]["validation"] if state["history"] else state["initial_validation"]
        )
        xr.ce.require_nested(
            validation(meta, model, query, device), expected, "shared-history resume"
        )
    else:
        torch.manual_seed(job["seed"] + 20260930)
        initial = validation(meta, model, query, device)
        state = {
            "metadata": metadata,
            "fork_sha256": fork,
            "epoch": 0,
            "initial_validation": initial,
            "best_epoch": 0,
            "best_validation": initial,
            "best_model": parent.cpu(model),
            "best_query": parent.cpu(query),
            "history": [],
        }
        atomic_json(
            gradient_diagnostic(meta, job, model, query, reader, builder, device),
            path / "initial_gradient.json",
        )

    def save():
        state.update(
            model=parent.cpu(model),
            query=parent.cpu(query),
            optimizers=[o.state_dict() for o in opts],
            rng=rng_state(),
        )
        parent.publish(state, path)

    save()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(state["epoch"] + 1, meta["epochs"] + 1):
        started = time.monotonic()
        schedule = plan(meta, job["seed"], epoch, len(builder.plan))
        lr = rates(meta, epoch)
        for o, k in zip(opts, lr, strict=True):
            for g in o.param_groups:
                g["lr"] = lr[k]
        tr, steps = train_epoch(meta, job, model, query, reader, builder, schedule, opts, device)
        val = validation(meta, model, query, device)
        if selected_value(val, state["initial_validation"]) < selected_value(
            state["best_validation"], state["initial_validation"]
        ):
            state.update(
                best_epoch=epoch,
                best_validation=val,
                best_model=parent.cpu(model),
                best_query=parent.cpu(query),
            )
        seconds = time.monotonic() - started
        state["history"].append(
            dict(
                epoch=epoch,
                sampling=schedule[4],
                views=3 * len(schedule[0]),
                steps=steps,
                train=tr,
                validation=val,
                seconds=seconds,
                **lr,
            )
        )
        state["epoch"] = epoch
        save()
        progress(
            f"{name}: {epoch}/{meta['epochs']}, price_val={val['query_price']:.6f}, activity_val={val['activity']:.6f}, best={state['best_epoch']}, {seconds:.1f}s/epoch, this_worker_eta~{seconds * (meta['epochs'] - epoch) / 60:.0f}min"
        )
    verify_history(meta, job, state, len(builder.plan))
    if any(not torch.equal(v, model.core.decoder.state_dict()[k].cpu()) for k, v in fixed.items()):
        raise ValueError("Frozen PCA decoder changed")
    atomic_json(
        gradient_diagnostic(meta, job, model, query, reader, builder, device),
        path / "last_gradient.json",
    )
    if ur.bb.state_signature(reader) != reader_signature:
        raise ValueError("Immutable auxiliary reader changed")
    parent.snapshot(state, path, meta["epochs"])
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
    summary = {
        "selected_epoch": state["best_epoch"],
        "epochs": meta["epochs"],
        "validation": state["best_validation"],
        "initial_validation": state["initial_validation"],
        "fork_sha256": fork,
        "seconds": sum(h["seconds"] for h in state["history"]),
        "views": sum(h["views"] for h in state["history"]),
        "steps": sum(h["steps"] for h in state["history"]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
    }
    atomic_json(summary, path / "training_summary.json")
    files = [
        "history.json",
        "training_state.json",
        "training_summary.json",
        "initial_gradient.json",
        "last_gradient.json",
    ] + [f"e{meta['epochs']}_{k}.pt" for k in ("best", "last")]
    atomic_json(
        {"status": "complete", "metadata": metadata, "files": {n: sha256(path / n) for n in files}},
        path / "completion.json",
    )
    parent.cleanup_completed(path)


def verify_summary(meta, state, summary):
    expected = {
        "selected_epoch": state["best_epoch"],
        "epochs": meta["epochs"],
        "validation": state["best_validation"],
        "initial_validation": state["initial_validation"],
        "fork_sha256": state["fork_sha256"],
        **{k: sum(h[k] for h in state["history"]) for k in ("seconds", "views", "steps")},
    }
    if any(summary[k] != v for k, v in expected.items()):
        raise ValueError("Training summary differs from history")


def lock_models(meta, out):
    trials = {}
    weights = {}
    mh = sha256(out / "manifest.json")
    for job in meta["experiments"]:
        path = out / job["name"]
        done = read_json(path / "completion.json")
        parent.verify_files(path, done["files"])
        expected = {"manifest_sha256": mh, "job": job, "phase": "main"}
        state = read_json(path / "training_state.json")
        if (
            done["metadata"] != expected
            or state["metadata"] != expected
            or done["status"] != "complete"
            or state["epoch"] != meta["epochs"]
        ):
            raise ValueError("Incomplete or foreign worker")
        verify_history(
            meta, job, state, read_json(root(meta) / "train_cross_plan.json")["eligible"]
        )
        if state["history"] != read_json(path / "history.json"):
            raise ValueError("History mismatch")
        if state["fork_sha256"] != meta["task_source"]["chosen"][str(job["seed"])]["sha256"]:
            raise ValueError("Wrong fork")
        summary = read_json(path / "training_summary.json")
        verify_summary(meta, state, summary)
        trials[job["name"]] = summary
        weights[job["name"]] = {}
        for kind in ("best", "last"):
            rel = f"{job['name']}/e{meta['epochs']}_{kind}.pt"
            ck = torch.load(out / rel, map_location="cpu", weights_only=True)
            epoch = state["best_epoch"] if kind == "best" else meta["epochs"]
            val = state["best_validation"] if kind == "best" else state["history"][-1]["validation"]
            if (
                ck["metadata"] != expected
                or ck["budget_epoch"] != meta["epochs"]
                or ck["epoch"] != epoch
                or ck["validation"] != val
            ):
                raise ValueError("Snapshot selection mismatch")
            weights[job["name"]][kind] = {"path": rel, "sha256": sha256(out / rel), "epoch": epoch}
    for seed in (42, 43):
        names = [j["name"] for j in meta["experiments"] if j["seed"] == seed]
        if any(
            trials[n]["initial_validation"] != trials[names[0]]["initial_validation"] for n in names
        ):
            raise ValueError("Arms have different starting function")
    atomic_json(
        {"manifest_sha256": mh, "trials": trials, "weights": weights},
        out / "model_selection_lock.json",
    )


def dispatch(meta, out, jobs):
    pending = list(meta["experiments"])
    running = []
    try:
        while pending or running:
            while pending and len(running) < jobs:
                j = pending.pop(0)
                if (out / j["name"] / "completion.json").exists():
                    worker(out, j["name"])
                    continue
                progress(f"Starting {j['name']}; {len(pending)} queued, {len(running) + 1} active")
                running.append(
                    (
                        j["name"],
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "obson.babel.macro_history_run",
                                "--out",
                                str(out),
                                "--worker",
                                j["name"],
                            ]
                        ),
                    )
                )
            for name, p in list(running):
                rc = p.poll()
                if rc is not None:
                    running.remove((name, p))
                    if rc:
                        raise RuntimeError(
                            f"{name} failed with exit {rc}; original traceback is in log"
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


def preflight(meta, out, device, jobs):
    source = Path(meta["task_source"]["source"])
    source_lock = read_json(source / "model_selection_lock.json")
    results = []
    for seed in (42, 43):
        model, query = construct(meta, seed, device)
        with torch.inference_mode():
            causal = ur.pf.er.trained_causality(
                model, torch.tensor(data(meta, "val")["x"][:4], device=device)
            )
            if causal["status"] == "failed":
                raise ValueError("Source causality failed")
            val = validation(meta, model, query, device)
            xr.ce.require_nested(
                val,
                source_lock["trials"][f"control_s{seed}"]["validation"],
                "source validation replay",
            )
        results.append(
            {"seed": seed, "causality": causal, "validation": val, "source_replayed": True}
        )
        del model, query
    atomic_json(results, out / "preflight.json")
    gc.collect()
    torch.cuda.empty_cache()
    checks = []
    builder = xr.xd.Builder(tm(meta), root(meta))
    for mode in core.MODES:
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model, query = construct(meta, 42, device)
        reader = copy.deepcopy(query).eval().requires_grad_(False)
        model.requires_grad_(False)
        model.core.encoder.requires_grad_(True)
        model.local_head.requires_grad_(True)
        query.requires_grad_(True)
        diagnostic = gradient_diagnostic(
            meta, {"seed": 42, "mode": mode}, model, query, reader, builder, device
        )
        groups = core.groups(model, query)
        if any(not any(p.grad is not None for p in g) for g in groups):
            raise ValueError("Missing preflight gradient group")
        peak = torch.cuda.max_memory_allocated() - baseline
        moments = 8 * sum(p.numel() for g in groups for p in g)
        checks.append(
            {
                "mode": mode,
                "optimizer_steps": 0,
                "diagnostic": diagnostic,
                "estimated_worker_bytes": int(1.3 * (peak + moments)) + 384 * 1024**2,
            }
        )
        del model, query, reader, groups
        gc.collect()
        torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    per_worker = max(c["estimated_worker_bytes"] for c in checks)
    effective_jobs = min(jobs, int(free // per_worker))
    atomic_json(
        {
            "checks": checks,
            "requested_jobs": jobs,
            "effective_jobs": effective_jobs,
            "available_bytes": free,
            "estimated_worker_bytes": per_worker,
            "passed": effective_jobs >= 1,
        },
        out / "gradient_preflight.json",
    )
    if effective_jobs < 1:
        raise ValueError(
            "Memory preflight failed even for one worker; choose new output with smaller MICRO; no updates performed"
        )
    if effective_jobs < jobs:
        progress(
            f"Memory preflight: automatically using {effective_jobs} worker; effective batch and experiment budget unchanged"
        )
    return effective_jobs


def check_disk(meta, out, jobs):
    # Size actual model/optimizer tensors on CPU; no inference or training.
    model, query = construct(meta, 42, "cpu")
    state = sum(
        v.numel() * v.element_size() for m in (model, query) for v in m.state_dict().values()
    )
    trained = sum(p.numel() for g in core.groups(model, query) for p in g)
    retained = 2 * len(meta["experiments"]) * state
    temp = jobs * (2 * (2 * state + 8 * trained) + state)
    required = int(1.2 * (retained + temp)) + 2 * 1024**3
    free = shutil.disk_usage(out).free
    atomic_json(
        {
            "required_bytes": required,
            "available_bytes": free,
            "retained_bytes": retained,
            "atomic_temporary_bytes": temp,
        },
        out / "disk_preflight.json",
    )
    if free < required:
        raise ValueError(
            f"Need {required / 2**30:.1f}GiB free, have {free / 2**30:.1f}GiB. No source files deleted."
        )
    progress(f"Disk: need {required / 2**30:.1f}GiB, available {free / 2**30:.1f}GiB")


def run(source, out, jobs=2, micro=16, evaluate_only=False):
    from . import macro_history_evaluate as ev

    source = source.resolve()
    out = out.resolve()
    parent.warm.check_output(source, out)
    progress("Verifying completed Shared History768 control checkpoints and original lineage")
    identity = source_identity(source)
    meta = make_manifest(source, identity, micro)
    meta["run_directory"] = str(out)
    for p in (
        meta["source"],
        meta["warmup"]["source"],
        meta["start"]["source"],
        root(meta),
        identity["source"],
        identity["grounded_source"],
    ):
        parent.warm.check_output(Path(p), out)
    rt = read_json(Path(identity["source"]) / "runtime.json")
    if rt["torch"] != str(torch.__version__) or rt["numpy"] != np.__version__:
        raise ValueError("Keep source Torch/NumPy versions")
    if out.exists() and any(out.iterdir()) and read_json(out / "manifest.json") != meta:
        raise ValueError("Different source/config: use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(meta, out / "manifest.json")
    if (out / "completion.json").exists():
        done = read_json(out / "completion.json")
        if done["status"] != "complete" or not done["source_unchanged"]:
            raise ValueError("Invalid completion")
        parent.verify_files(out, done["files"])
        return
    complete = all((out / j["name"] / "completion.json").exists() for j in meta["experiments"])
    if evaluate_only and not complete:
        raise ValueError("All six completed workers required for evaluation-only")
    started = time.monotonic()
    atomic_json(
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(),
            "jobs": jobs,
            "micro": micro,
        },
        out / "runtime.json",
    )
    if not complete:
        check_disk(meta, out, jobs)
    xr.xd.verify_data(tm(meta), root(meta))
    macro_data.prepare(meta, out)
    ev.audit_sources(meta, out, "cuda")
    if not complete:
        reports = {str(seed): ev.baseline_gate(meta, out, seed, "cuda") for seed in (42, 43)}
        passed = all(v["passed"] for v in reports.values())
        atomic_json(
            {
                "status": "eligible" if passed else "blocked",
                "qualified": passed,
                "reason": "Zero-fit original-reader physical macro validation gate",
                "encoder_training_authorized": passed,
            },
            out / "audit_status.json",
        )
        if not passed:
            progress(
                "Frozen semantic qualification failed; joint encoder training NOT started. Exporting diagnostics."
            )
            raise SystemExit(3)
        jobs = preflight(meta, out, "cuda", jobs)
    require_gate(out)
    gc.collect()
    torch.cuda.empty_cache()
    if not complete:
        dispatch(meta, out, jobs)
    lock_models(meta, out)
    ev.fit_readouts(meta, out, "cuda")
    ev.evaluate(meta, out, "cuda")
    if source_identity(source) != identity:
        raise ValueError("Source changed during experiment")
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
                and p.name != "run_status.txt"
                and p.suffix not in (".tmp", ".log")
            },
        },
        out / "completion.json",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="checkpoints/babel_grounded_readout768")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    p.add_argument("--micro", type=int, choices=(8, 16, 32, 64), default=16)
    p.add_argument("--worker")
    p.add_argument("--evaluate-only", action="store_true")
    a = p.parse_args()
    ur.bb.ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError("Real training/evaluation requires AutoDL CUDA")
    out = Path(a.out).resolve()
    if a.worker:
        worker(out, a.worker)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ("." + out.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Macro-history experiment already running") from None
        run(Path(a.source), out, a.jobs, a.micro, a.evaluate_only)


if __name__ == "__main__":
    main()
