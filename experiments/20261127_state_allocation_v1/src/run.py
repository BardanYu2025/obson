"""AutoDL preflight, immutable matrix card and restorable joint optimization."""

from __future__ import annotations

import json
import math
import random
import shutil
import signal
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .assets import (
    environment,
    export_run,
    load_checkpoint,
    save_checkpoint,
    snapshot_context,
    verify_package,
)
from .common import ROOT, atomic, digest, exclusive, identity, json_file, read_json
from .dataset import Corpus, build, snapshot, verify_cache
from .metrics import paired_evaluation
from .model import StateModel
from .objective import optimizer, update
from .scales import fit
from .scope import select_sources
from .views import sample

PROTOCOL = read_json(ROOT / "configs/protocol.json")
USE_KEYS = {
    "ohlc_native_or_bp",
    "shape_bp",
    "volume_lots",
    "oi_lots",
    "net_bp",
    "efficiency",
    "volatility_bp",
    "drawdown_bp",
    "description_volume_lots",
    "description_delta_oi_lots",
    "p95_native_or_bp",
    "min_market_weeks",
    "refresh_ms",
}


def prepare(raw_root, cache, scope=None):
    manifest = select_sources(snapshot(), scope)
    build(raw_root, cache, manifest)
    cache = Path(cache)
    with exclusive(cache / ".scale.lock"):
        catalogue_id = digest(cache / "catalogue.json")
        path = cache / "scales.json"
        if path.exists():
            fields = read_json(path)
            if fields["catalogue"] != catalogue_id or fields["snapshot"] != identity(
                manifest
            ):
                raise ValueError("Cached scales belong to a different dataset")
        else:
            corpus = Corpus(cache)
            stats = fit(corpus, PROTOCOL["stats_endpoints"])
            fields = {
                "catalogue": catalogue_id,
                "snapshot": identity(manifest),
                "scope": scope,
                "scales": {k: v.tolist() for k, v in stats.items()},
            }
            json_file(path, fields)
        counts = {}
        for g in read_json(cache / "catalogue.json")["groups"]:
            key = f"{g[4]}/{g[1]}"
            counts[key] = (
                np.asarray(counts.get(key, [0] * len(g[7:]))) + np.asarray(g[7:])
            ).tolist()
        json_file(
            cache / "coverage.json",
            {"split_product_clean_mask_description_natural_order": counts},
        )
    return ready_data(cache, verify=False)


def ready_data(cache, verify=True):
    cache = Path(cache)
    rec = read_json(cache / "scales.json")
    source = select_sources(snapshot(), rec["scope"])
    if rec["snapshot"] != identity(source) or rec["catalogue"] != digest(
        cache / "catalogue.json"
    ):
        raise ValueError("Cache or scale input identity changed")
    if verify:
        verify_cache(cache, rec["snapshot"])
    return Corpus(cache), {
        k: np.array(v, dtype=float) for k, v in rec["scales"].items()
    }


def device_setup():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "For real data, run the neural stage on the AutoDL CUDA instance"
        )
    if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda is None:
        raise RuntimeError("Use the experiment's CUDA PyTorch 2.8.0 environment")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device("cuda")


def new_model(arm, stats, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return StateModel(arm, stats, **PROTOCOL["encoder"]).to(device)


def train_limits(card):
    for key in (
        "updates",
        "validation_every",
        "validation_endpoints",
        "checkpoint_every",
        "description_updates",
        "frozen_train_endpoints",
    ):
        if type(card.get(key)) is not int or card[key] < 1:
            raise ValueError(f"Set a positive integer for {key}")
    for key in (
        "gpu_hours_per_run",
        "run_disk_gib",
        "frozen_gpu_hours",
        "frozen_disk_gib",
    ):
        if not math.isfinite(card[key]) or card[key] <= 0:
            raise ValueError(f"Set a positive {key}")
    if card["frozen_train_endpoints"] <= 768:
        raise ValueError("Centered PCA768 needs at least 769 training endpoints")
    if card["checkpoint_every"] > card["updates"]:
        raise ValueError("Checkpoint interval exceeds run length")
    thresholds = card.get("physical_use_limits", {})
    if set(thresholds) != USE_KEYS or any(
        not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0
        for x in thresholds.values()
    ):
        raise ValueError(
            "Enter all pre-score physical use limits listed on study.proposed.json"
        )
    if not isinstance(thresholds["min_market_weeks"], int):
        raise ValueError("min_market_weeks must be an integer")
    if card.get("ohlc_unit") not in ("basis_points", "native_per_contract"):
        raise ValueError("Specify ohlc_unit")
    if not card.get("intended_use") or not card.get("support_and_near_zero_rule"):
        raise ValueError(
            "Name the intended use and the pre-score support/near-zero rule"
        )


def preflight(cache, dest, release):
    device = device_setup()
    corpus, stats = ready_data(cache)
    cache = Path(cache)
    with exclusive(cache / ".preflight.lock"):
        key = digest(cache / "scales.json")
        folder = cache / "preflights" / ("preflight-" + uuid.uuid4().hex[:12])
        snapshot_context(
            folder, cache, release, stats, {"stage": "preflight", "scales_sha256": key}
        )
        reports = {}
        try:
            for arm in ("P2", "P3"):
                sub = folder / arm
                model = new_model(arm, stats, 42, device)
                opt = optimizer(model)
                counts = [
                    sum(p.numel() for p in mod.parameters())
                    for mod in (model.encoder, model.history, model.describe)
                ]
                if counts != [42610308, 2107911, 100754]:
                    raise ValueError(f"Unexpected E/D/A parameter counts: {counts}")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                when = time.monotonic()
                info = update(
                    model,
                    opt,
                    corpus,
                    stats,
                    42,
                    1,
                    PROTOCOL["study_defaults"]["updates"],
                    measure_gradients=True,
                )
                torch.cuda.synchronize()
                seconds = time.monotonic() - when
                sid, row = corpus.draw(0, 0, np.random.default_rng(527))
                s = sample(corpus.shard(sid), row, stats)
                with torch.no_grad():
                    z = model.encode(s).cpu()
                    idx = np.arange(min(8, len(s.period)))
                    pred = model.read(
                        z.to(device), s.query, s.period, s.current, s.block, s.slot, idx
                    )[0].cpu()
                rid = identity([release, key, arm, "preflight"])
                save_checkpoint(sub, model, opt, 1, [info], rid)
                peak = torch.cuda.max_memory_allocated()
                del model, opt
                model = new_model(arm, stats, 42, device)
                opt = optimizer(model)
                step, trace = load_checkpoint(sub, "last", model, opt, rid)
                with torch.no_grad():
                    z2 = model.encode(s)
                    y2 = model.read(
                        z2, s.query, s.period, s.current, s.block, s.slot, idx
                    )[0].cpu()
                if (
                    step != 1
                    or len(trace) != 1
                    or not torch.equal(z, z2.cpu())
                    or not torch.equal(pred, y2)
                ):
                    raise ValueError("CUDA save/restore replay mismatch")
                timings = []
                with torch.no_grad():
                    for _ in range(7):
                        torch.cuda.synchronize()
                        begin = time.monotonic()
                        fresh = sample(corpus.shard(sid), row, stats)
                        model.encode(fresh)
                        torch.cuda.synchronize()
                        timings.append((time.monotonic() - begin) * 1000)
                reports[arm] = {
                    "update_wall_seconds_including_gradient_audit": seconds,
                    "refresh_p50_ms": float(np.median(timings)),
                    "refresh_p95_ms": float(np.quantile(timings, 0.95)),
                    "peak_bytes": peak,
                    "counts": counts,
                    "state_bytes": z.numel() * z.element_size(),
                    "gradient_norms": info["grad_norms_E_D_A"],
                    "tasks": [32, 16, 16],
                    "restored_state_and_prices": True,
                    "weighted_gradient_gram": info["weighted_gradient_gram"],
                    "weighted_gradient_dot_actual_delta": info[
                        "weighted_gradient_dot_actual_delta"
                    ],
                }
                del model, opt
            receipt = {
                "release": release,
                "scales": key,
                "hardware": environment(),
                "arms": reports,
            }
            json_file(folder / "preflight.json", receipt)
            package = export_run(folder, dest)
            receipt["bundle"] = verify_package(package)
            json_file(cache / "preflight.json", receipt)
            card = {
                **PROTOCOL["study_defaults"],
                "release": release,
                "preflight": identity(receipt),
                "scales": key,
                "intended_use": "",
                "ohlc_unit": "basis_points",
                "support_and_near_zero_rule": "",
                "physical_use_limits": {k: None for k in sorted(USE_KEYS)},
            }
            json_file(cache / "study.proposed.json", card)
            print(json.dumps(reports), flush=True)
        except BaseException as err:
            json_file(
                folder / "failure.json", {"stage": "preflight", "error": repr(err)}
            )
            export_run(folder, dest)
            raise


def lock_study(cache, card_path, release):
    cache = Path(cache)
    card = read_json(card_path)
    train_limits(card)
    pre = read_json(cache / "preflight.json")
    scales = read_json(cache / "scales.json")
    if not scales["scope"]:
        raise ValueError(
            "Formal comparison needs the dated physical scope; prepare a separate cache with --scope"
        )
    if (
        card["release"] != release
        or pre["release"] != release
        or card["preflight"] != identity(pre)
        or card["scales"] != digest(cache / "scales.json")
        or pre["scales"] != card["scales"]
    ):
        raise ValueError("Study card, data or preflight identity mismatch")
    numerical_keys = (
        "torch",
        "numpy",
        "pyarrow",
        "cuda",
        "device_capability",
        "cudnn",
        "cublas_workspace",
    )
    current = environment()
    if any(pre["hardware"][k] != current[k] for k in numerical_keys):
        raise ValueError(
            "Run the preflight on this numerical backend before locking the comparison"
        )
    if set(pre["arms"]) != {"P2", "P3"} or not pre.get("bundle"):
        raise ValueError("Incomplete prospective engineering receipt")
    with exclusive(cache / ".study.lock"):
        path = cache / "study.lock.json"
        if path.exists() and read_json(path) != card:
            raise ValueError(
                "Already locked to different budgets/use limits; use a new experiment version"
            )
        if not path.exists():
            json_file(path, card)
    return card


def validation(model, corpus, stats, folder, step, n, split=1):
    path = folder / "evaluations" / f"split-{split}-step-{step:07d}.jsonl"
    with atomic(path) as tmp, tmp.open("w") as out:

        def emit(obj):
            out.write(json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n")

        summary = paired_evaluation(model, corpus, stats, split, n, emit)
    return summary


def train(cache, dest, arm, seed, release, resume=False):
    device = device_setup()
    cache = Path(cache)
    corpus, stats = ready_data(cache)
    card = lock_study(cache, cache / "study.lock.json", release)
    if arm not in PROTOCOL["arms"] or seed not in PROTOCOL["seeds"]:
        raise ValueError("Unknown arm or unpaired seed")
    design = identity(card)
    rid = identity([design, arm, seed])
    folder = cache / "runs" / f"{arm}-{seed}-{rid[:12]}"
    with exclusive(cache / (".run-" + rid)):
        checkpoint = folder / "checkpoints.json"
        if folder.exists() and not resume:
            raise ValueError(f"Existing run {folder}; use resume")
        if resume and not checkpoint.exists():
            raise ValueError("No complete checkpoint for this run")
        if (folder / "completed.json").exists():
            raise ValueError(
                "Training already completed; use export for another backup"
            )
        model = new_model(arm, stats, seed, device)
        opt = optimizer(model)
        if not resume:
            snapshot_context(
                folder,
                cache,
                release,
                stats,
                {"stage": "train", "run": rid, "arm": arm, "seed": seed, "study": card},
            )
            shutil.copyfile(cache / "preflight.json", folder / "preflight.json")
            shutil.copyfile(cache / "scales.json", folder / "data_receipt.json")
        meta = read_json(folder / "run.json")
        numerical_keys = (
            "torch",
            "numpy",
            "pyarrow",
            "cuda",
            "device_capability",
            "cudnn",
            "cublas_workspace",
        )
        current_env = environment()
        if meta["run"] != rid or any(
            meta["environment"][k] != current_env[k] for k in numerical_keys
        ):
            raise ValueError("Run or backend changed on resume")
        complete = False
        budget_path = folder / "runtime.json"
        start = time.monotonic()
        past_wall = read_json(budget_path)["seconds"] if budget_path.exists() else 0.0
        interrupted = []
        handlers = {
            sig: signal.signal(sig, lambda code, frame: interrupted.append(code))
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        try:
            if resume:
                step, trace = load_checkpoint(folder, "last", model, opt, rid)
            else:
                step, trace = 0, []
                save_checkpoint(folder, model, opt, step, trace, rid)
            past_wall = max(past_wall, sum(t.get("seconds", 0.0) for t in trace))
            while step < card["updates"]:
                if (
                    past_wall + time.monotonic() - start
                    >= card["gpu_hours_per_run"] * 3600
                    or interrupted
                ):
                    raise InterruptedError(
                        "Wall-time cap or stop signal; use the saved completed step"
                    )
                used = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
                if (
                    used > card["run_disk_gib"] * 2**30
                    or shutil.disk_usage(folder).free < 2 * 2**30
                ):
                    raise InterruptedError(
                        "Run disk cap or insufficient checkpoint space"
                    )
                since = time.monotonic()
                info = update(
                    model, opt, corpus, stats, seed, step + 1, card["updates"]
                )
                step += 1
                best = False
                if step % card["validation_every"] == 0 or step == card["updates"]:
                    result = validation(
                        model, corpus, stats, folder, step, card["validation_endpoints"]
                    )
                    info["validation"] = result
                    old = [
                        x["validation"]["R"]
                        for x in trace
                        if x.get("validation", {}).get("R") is not None
                    ]
                    best = result["R"] is not None and (
                        not old or result["R"] < min(old)
                    )
                info["seconds"] = time.monotonic() - since
                trace.append(info)
                if (
                    step % card["checkpoint_every"] == 0
                    or "validation" in info
                    or interrupted
                ):
                    save_checkpoint(folder, model, opt, step, trace, rid, best)
                    json_file(
                        budget_path, {"seconds": past_wall + time.monotonic() - start}
                    )
                    print(
                        f"{arm} {seed} step {step}/{card['updates']} losses {info['loss']}",
                        flush=True,
                    )
            json_file(
                folder / "completed.json",
                {"step": step, "primary": "last", "trace": trace},
            )
            complete = True
        except BaseException as err:
            json_file(
                folder / ("failure-" + uuid.uuid4().hex[:10] + ".json"),
                {"error": repr(err), "run": rid},
            )
            raise
        finally:
            for sig, h in handlers.items():
                signal.signal(sig, h)
            json_file(budget_path, {"seconds": past_wall + time.monotonic() - start})
            bundle = export_run(folder, dest)
            proof = verify_package(bundle)
            json_file(folder / "export-receipt.json", {"file": str(bundle), **proof})
            print("Completed" if complete else "Interrupted", folder, proof, flush=True)


def evaluate(cache, folder, release, split):
    if split not in (1, 2):
        raise ValueError(
            "This training-stage entry uses validation or development; final confirmation is separate"
        )
    device = device_setup()
    corpus, stats = ready_data(cache)
    folder = Path(folder)
    run = read_json(folder / "run.json")
    if run["release"] != release or run["study"]["scales"] != digest(
        Path(cache) / "scales.json"
    ):
        raise ValueError("Evaluation release/cache mismatch")
    with exclusive(folder / ".evaluate.lock"):
        if not (folder / "completed.json").exists():
            raise ValueError("Last-N primary evaluation requires completed training")
        model = new_model(run["arm"], stats, run["seed"], device)
        step, _ = load_checkpoint(folder, "last", model, None, run["run"])
        out = folder / "reviews" / ("evaluation-" + uuid.uuid4().hex[:10])
        result = validation(
            model, corpus, stats, out, step, run["study"]["validation_endpoints"], split
        )
        json_file(out / "summary.json", result)
        return out
