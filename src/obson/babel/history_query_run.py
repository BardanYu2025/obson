"""One locked three-arm history-query experiment from verified control600 bundles."""

import argparse
import fcntl
import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import history_query as hq
from . import research_state as rs
from . import research_state_delivery as delivery
from .ae_extend import atomic_json, atomic_save, restore_rng, rng_state
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-history-query768-v1"
warm, xr = delivery.warm, delivery.xr
ur = warm.ur


def code_identity():
    from . import history_query_evaluate as ev

    return (
        delivery.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (hq, ev)}
        | {Path(__file__).name: sha256(__file__)}
    )


def source_identity(source):
    m, done = read_json(source / "manifest.json"), read_json(source / "completion.json")
    if (
        m["schema"] != delivery.SCHEMA
        or m["code_sha256"] != delivery.code_identity()
        or done["status"] != "complete"
        or not done["source_unchanged"]
        or any(
            done[k] != 0
            for k in ("encoder_updates", "head_updates", "reader_fits", "statistics_fits")
        )
    ):
        raise ValueError("Completed immutable control600 delivery required")
    verify_files(source, done["files"])
    if warm.source_identity(Path(m["source"])) != m["identity"]:
        raise ValueError("Upstream control600 lineage changed")
    index, cert = (
        read_json(source / "bundle/index.json"),
        read_json(source / "bundle/validation.json"),
    )
    if (
        cert["status"] != "passed"
        or cert["index_sha256"] != sha256(source / "bundle/index.json")
        or index["source_manifest"] != sha256(source / "manifest.json")
    ):
        raise ValueError("Validated matching research bundle required")
    return {
        "manifest": m,
        "files": done["files"] | {"completion.json": sha256(source / "completion.json")},
    }


def tm(meta):
    return meta["identity"]["manifest"]["identity"]["training_manifest"]


def root(meta):
    return Path(meta["identity"]["manifest"]["identity"]["training_source"])


def stats(meta):
    return xr.statistics(tm(meta))


def data(meta, split):
    return delivery.data_for(meta["identity"]["manifest"], split)


def make_manifest(source, identity, micro=16):
    if micro not in (8, 16, 32, 64):
        raise ValueError("Choose micro8/16/32/64 within effective batch128")
    return {
        "schema": SCHEMA,
        "source": str(source),
        "identity": identity,
        "code_sha256": code_identity(),
        "epochs": 100,
        "warm_epochs": 50,
        "budget": 4789,
        "batch": 128,
        "micro": micro,
        "evaluation_batch": 64,
        "encoder_lr": 1e-5,
        "head_lr": 3e-5,
        "query_lr": 3e-4,
        "query_config": hq.CONFIG,
        "query_weight": 0.25,
        "experiments": [
            {"name": f"{mode}_s{s}", "mode": mode, "seed": s}
            for s in (42, 43)
            for mode in ("frozen", "control", "joint")
        ],
        "training": "Same original9909 eligible triplet pool.4789 triplets/epoch,14367 views,38 updates. Original supervision and0.10 same-period consistency retained for control/joint. No cross-period alignment penalty.",
        "query": "One768 state -> four192 memory tokens -> two192-wide cross-attention/FFN blocks,7 outputs per relative historical age. No query self-attention, raw-bar skip, other states, current or future targets. Ages1..127.",
        "targets": "Close log-percent relative to observed CURRENT close; current is label/display anchor only, not decoder input. Source y last row masked: derive its anchor from prior close plus observed last gap/body. Price scale=sqrt(age)*original train delta_scale1, other scales unchanged. Equal available age bands1..16/17..64/65..127, original .4/.2/.1/.3 family weights.",
        "prefixes": "Four distinct nonheld32..127 positions plus128 per view;48/80/112 excluded. Val32/64/96/128; held positions only research, not fresh data.",
        "initialization": "Per-seed shared50 query-only warmup from original control600, all arms fork FINAL warm query exactly. All three reset query AdamW at fork; control/joint also reset encoder/local AdamW. Frozen arm keeps original encoder/local. Input PathInput and fixed PCA decoder restored from validated bundle.",
        "objective": "Query-head gradient coefficient1 in EVERY arm. Encoder receives query gradient0/control or0.25/joint via explicit stop-gradient interpolation; query loss has identical numerical value and decoder optimization opportunities. Original branch unchanged.",
        "selection": "Warmup fixed final50 without research. Main epoch0 candidate; minimize0.5*(old_validation/initial_old + query_validation/initial_query) on same978 validation windows, both factors pinned at fork. All6 full budgets locked before fitting12 best/last x13 x5 Ridge candidates; all readouts locked before research.",
        "decision": "Joint requires >=5% held-query and held-own-anchor-recent gain vs SAME-budget control and frozen query arms; both seeds, both reused research sets,best AND last,weekly CI. Protect original global/local/recent, query endpoint, old/new overlap and calibrated utility/current fields. Original absolute raw/PCA/current protocol separately reported, no automatic promotion.",
        "stop": "One fixed matrix. A new decoder failing to optimize cannot prove absence of information; report tails and epoch0. No automatic LR/width/head/epoch sweep. Equal examples/updates are not equal FLOPs. New price coordinates alone are not credited as encoder gains.",
    }


def construct(meta, seed, device):
    engine = rs.load_bundle(Path(meta["source"]) / "bundle", seed, "MA/15/CZCE.MA601", 15, device)
    model = engine.aligned
    scaler = engine.utility["heads"][0]["statistics"]
    query = hq.HistoryQuery(
        model.core.encoder.coordinates.out_features, scaler, seed, meta["query_config"]
    ).to(device)
    return model, query


def plan(meta, seed, epoch, n, phase):
    stream = 100 + epoch if phase == "main" else 2000 + epoch
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
    weight,
):
    ids, shifts, prefixes, qps, _ = schedule
    enc, head, dec = optimizers
    model.eval()
    query.train()  # All inherited and new dropout are exactly zero.
    total = np.zeros(4)
    steps = 0
    for left in range(0, len(ids), batch):
        size = min(batch, len(ids) - left)
        for opt in optimizers:
            if opt is not None:
                opt.zero_grad(set_to_none=True)
        for start in range(left, left + size, micro):
            stop = min(start + micro, left + size)
            a, b, c, _, _ = builder(ids[start:stop], shifts[start:stop])
            n = stop - start
            joined = batch_tensors({k: np.concatenate([a[k], b[k], c[k]]) for k in a}, device)
            ps = torch.tensor(np.tile(prefixes[start:stop], (3, 1)), device=device)
            qp = torch.tensor(np.tile(qps[start:stop], (3, 1)), device=device)
            z = model.core.encoder(joined["x"])
            g = model.core.decoder(z[:, -1])
            chosen = z[torch.arange(len(z), device=device)[:, None], ps - 1]
            detail = model.local_head(chosen).reshape(len(z), ps.shape[1], 16, 7)
            target, mask = hq.ba.local_targets(joined["y"], joined["mask"], ps, statistics, local)
            supervised = (
                hq.ba.ar.error_rows(g, joined["y"], joined["mask"], statistics, True)["primary"]
                + 0.25 * hq.ba.local_rows(detail, target, mask, True)["primary"]
            )
            base = supervised.reshape(3, n).mean(0)
            within = xr.xp.oc.consistency_rows(
                g[:n],
                g[n : 2 * n],
                joined["mask"][:n],
                joined["mask"][n : 2 * n],
                torch.tensor(shifts[start:stop], device=device),
                statistics,
            )
            picked = z[torch.arange(len(z), device=device)[:, None], qp - 1]
            coefficient = weight if mode == "joint" else 0.0
            picked = picked.detach() + coefficient * (picked - picked.detach())
            pred = query(picked.flatten(0, 1)).reshape(len(z), qp.shape[1], 127, 7)
            y, valid = hq.targets(joined, qp, statistics)
            qloss = (
                hq.metrics(pred, y, valid, statistics, True)["primary"]
                .mean(1)
                .reshape(3, n)
                .mean(0)
            )
            old = base + 0.1 * within
            loss = qloss + (old if enc is not None else old.detach() * 0)
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite query training loss")
            (loss.sum() / size).backward()
            total += [float(v.detach().sum()) for v in (base, within, qloss, loss)]
        for module, opt in zip(
            (model.core.encoder, model.local_head, query), optimizers, strict=True
        ):
            if opt is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
        steps += 1
    return dict(
        zip(
            ("original", "consistency", "query", "optimization_value"),
            (total / len(ids)).tolist(),
            strict=True,
        )
    ), steps


@torch.inference_mode()
def validation(meta, model, query, device, seed=42):
    statistics, local = stats(meta)
    d = data(meta, "val")
    total = 0.0
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
    result = {"original": old, "query": total / len(d["x"])}
    if not np.isfinite([old["selection"], result["query"]]).all():
        raise ValueError("Nonfinite validation cannot select a checkpoint")
    return result


def selected_value(result, initial):
    return 0.5 * (
        result["original"]["selection"] / max(initial["original"]["selection"], 1e-12)
        + result["query"] / max(initial["query"], 1e-12)
    )


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
    epochs = meta["warm_epochs"] if phase == "warm" else meta["epochs"]
    if state["epoch"] > epochs or [h["epoch"] for h in state["history"]] != list(
        range(1, state["epoch"] + 1)
    ):
        raise ValueError("Incomplete or excessive training budget")
    updates = (meta["budget"] + meta["batch"] - 1) // meta["batch"]
    active = phase == "main" and job["mode"] != "frozen"
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
            expected = xr.growth.learning_rate(h["epoch"], epochs, meta[key])
            if not np.isclose(h[key], expected, rtol=1e-14, atol=0):
                raise ValueError("Learning rate schedule changed")
    candidates = [(0, state["initial_validation"])] + [
        (h["epoch"], h["validation"]) for h in state["history"]
    ]
    chosen = min(candidates, key=lambda v: selected_value(v[1], state["initial_validation"]))
    if (state["best_epoch"], state["best_validation"]) != chosen:
        raise ValueError("Validation-only selection changed")


def worker(out, name, phase, device="cuda"):
    meta = read_json(out / "manifest.json")
    if meta["code_sha256"] != code_identity():
        raise ValueError("Worker source code changed")
    job = next(j for j in meta["experiments"] if j["name"] == name)
    path = out / (f"warm_s{job['seed']}" if phase == "warm" else name)
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
        return
    model, query = construct(meta, job["seed"], device)
    frozen_signature = ur.bb.state_signature(model)
    fork = None
    if phase == "main":
        wp = out / f"warm_s{job['seed']}"
        done = read_json(wp / "completion.json")
        verify_files(wp, done["files"])
        ck = torch.load(wp / "last.pt", map_location="cpu", weights_only=True)
        warm_metadata = {
            "manifest_sha256": metadata["manifest_sha256"],
            "job": {"seed": job["seed"]},
            "phase": "warm",
        }
        if (
            ck["epoch"] != meta["warm_epochs"]
            or done["status"] != "complete"
            or ck["metadata"] != warm_metadata
            or done["metadata"] != warm_metadata
        ):
            raise ValueError("Full common warmup required")
        model.load_state_dict(ck["model"])
        query.load_state_dict(ck["query"])
        fork = sha256(wp / "last.pt")
        if ur.bb.state_signature(model) != frozen_signature:
            raise ValueError("Warmup modified parent encoder/head")
    active = phase == "main" and job["mode"] != "frozen"
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
    epochs = meta["warm_epochs"] if phase == "warm" else meta["epochs"]
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
    for epoch in range(state["epoch"] + 1, epochs + 1):
        started = time.monotonic()
        schedule = plan(meta, job["seed"], epoch, len(builder.plan), phase)
        rates = {
            key: xr.growth.learning_rate(epoch, epochs, meta[key])
            for key in ("encoder_lr", "head_lr", "query_lr")
        }
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
            meta["query_weight"],
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
    if not active and ur.bb.state_signature(model) != frozen_signature:
        raise ValueError("Frozen encoder/local head changed")
    summary = dict(
        selected_epoch=state["best_epoch"],
        epochs=epochs,
        phase=phase,
        fork_sha256=fork,
        query_parameters=sum(p.numel() for p in query.parameters()),
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
            "status": "complete",
            "metadata": metadata,
            "files": {
                n: sha256(path / n)
                for n in ("last.pt", "best.pt", "history.json", "training_summary.json")
            },
        },
        path / "completion.json",
    )


def lock_models(meta, out):
    trials = {}
    weights = {}
    for job in meta["experiments"]:
        path = out / job["name"]
        done = read_json(path / "completion.json")
        verify_files(path, done["files"])
        expected = {"manifest_sha256": sha256(out / "manifest.json"), "job": job, "phase": "main"}
        if done["status"] != "complete" or done["metadata"] != expected:
            raise ValueError("All six full budgets required")
        last = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
        verify_history(
            meta, job, last, read_json(root(meta) / "train_cross_plan.json")["eligible"], "main"
        )
        if (
            last["epoch"] != meta["epochs"]
            or last["metadata"] != expected
            or best["metadata"] != expected
            or last["best_epoch"] != best["epoch"]
            or last["best_validation"] != best["validation"]
            or read_json(path / "history.json") != last["history"]
        ):
            raise ValueError("Checkpoint selection or history mismatch")
        for field in ("model", "query"):
            if any(not torch.equal(v, best[field][k]) for k, v in last["best_" + field].items()):
                raise ValueError("Best state differs")
        summary = read_json(path / "training_summary.json")
        expected_summary = {
            "selected_epoch": last["best_epoch"],
            "epochs": meta["epochs"],
            "phase": "main",
            "fork_sha256": last["fork_sha256"],
            "validation": last["best_validation"],
            "initial_validation": last["initial_validation"],
            "decoder_unchanged": True,
            "frozen_parent_unchanged": job["mode"] == "frozen",
            "tail": last["history"][-20:],
            **{
                k: sum(h[k] for h in last["history"])
                for k in ("views", "encoder_steps", "local_steps", "query_steps", "seconds")
            },
        }
        if any(summary[k] != v for k, v in expected_summary.items()):
            raise ValueError("Training summary differs from authoritative history")
        trials[job["name"]] = summary
        weights[job["name"]] = {k: sha256(path / f"{k}.pt") for k in ("best", "last")}
    for seed in (42, 43):
        wp = out / f"warm_s{seed}"
        done = read_json(wp / "completion.json")
        verify_files(wp, done["files"])
        ck = torch.load(wp / "last.pt", map_location="cpu", weights_only=True)
        expected_metadata = {
            "manifest_sha256": sha256(out / "manifest.json"),
            "job": {"seed": seed},
            "phase": "warm",
        }
        if (
            done["status"] != "complete"
            or done["metadata"] != expected_metadata
            or ck["metadata"] != expected_metadata
            or ck["epoch"] != meta["warm_epochs"]
        ):
            raise ValueError("Shared full warmup identity differs")
        verify_history(
            meta,
            {"seed": seed, "mode": "frozen"},
            ck,
            read_json(root(meta) / "train_cross_plan.json")["eligible"],
            "warm",
        )
        expected = sha256(wp / "last.pt")
        if any(
            v["fork_sha256"] != expected for name, v in trials.items() if name.endswith(f"s{seed}")
        ):
            raise ValueError("Arms did not share exact warmup")
    lock = {"manifest_sha256": sha256(out / "manifest.json"), "trials": trials, "weights": weights}
    if (out / "model_selection_lock.json").exists() and read_json(
        out / "model_selection_lock.json"
    ) != lock:
        raise ValueError("Locked model selection changed")
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
                        "obson.babel.history_query_run",
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


@torch.inference_mode()
def preflight(meta, out, device):
    d = data(meta, "val")
    results = []
    for seed in (42, 43):
        model, query = construct(meta, seed, device)
        expected = read_json(root(meta) / "model_selection_lock.json")["trials"][
            f"control_s{seed}"
        ]["validation"]
        actual = xr.validation(tm(meta), {"seed": seed}, model, device)
        xr.ce.require_nested(actual, expected, "original control600 validation")
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


def run(source, out, jobs=1, micro=16, device="cuda"):
    from . import history_query_evaluate as ev

    source, out = source.resolve(), out.resolve()
    progress("Verifying control600 delivery, original checkpoints and immutable source lineage")
    identity = source_identity(source)
    warm.check_output(source, out, identity["manifest"]["identity"])
    meta = make_manifest(source, identity, micro)
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
    preflight(meta, out, device)
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    dispatch(out, [f"frozen_s{s}" for s in (42, 43)], "warm", jobs)
    dispatch(out, [j["name"] for j in meta["experiments"]], "main", jobs)
    lock_models(meta, out)
    ev.fit_readouts(meta, out, device)
    ev.evaluate(meta, out, device)
    if source_identity(source) != identity:
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="checkpoints/babel_control600_v2")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    p.add_argument("--micro", type=int, choices=(8, 16, 32, 64), default=16)
    p.add_argument("--worker")
    p.add_argument("--phase", choices=("warm", "main"), default="main")
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
        run(Path(a.source), out, a.jobs, a.micro)


if __name__ == "__main__":
    main()
