"""Zero-fit same-state, same-interval decoder comparison after grounded qualification."""

import argparse
import fcntl
from pathlib import Path

import numpy as np
import torch

from . import grounded_history as core
from . import grounded_history_run as gr
from . import grounded_history_warmup as qw
from . import shared_history_data as sd
from .ae_extend import atomic_json
from .dual_state import sha256, verify_files
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-grounded-readout-audit-v1"
VIEWS = ("earlier_fine", "fine", "coarse")


def source_identity(source):
    meta = read_json(source / "manifest.json")
    if meta["schema"] != gr.SCHEMA or meta["code_sha256"] != gr.code_identity():
        raise ValueError("Original grounded source/code identity differs")
    if gr.source_identity(Path(meta["task_source"]["source"])) != meta["task_source"]:
        raise ValueError("Original encoder source changed")
    files = {
        n: sha256(source / n)
        for n in (
            "manifest.json",
            "runtime.json",
            "grounded_data_lock.json",
            "val_grounded_plan.json",
            "audit_status.json",
        )
    }
    gr.grounded_data.verify(meta, source)
    for seed in (42, 43):
        path = source / f"qualification_s{seed}"
        done = read_json(path / "completion.json")
        cache_lock = read_json(path / "cache_lock.json")
        identity = {
            "manifest": sha256(source / "manifest.json"),
            "cache": sha256(path / "cache_lock.json"),
        }
        if done["identity"] != identity or cache_lock["identity"] != {
            "manifest": identity["manifest"],
            "seed": seed,
            "data_lock": sha256(source / "grounded_data_lock.json"),
        }:
            raise ValueError("Qualification/cache binding differs")
        verify_files(path, done["files"])
        verify_files(path, cache_lock["files"])
        names = {
            *done["files"],
            *cache_lock["files"],
            "completion.json",
            "state_only_resume.pt",
            "query_only_resume.pt",
        }
        for name in names:
            files[f"qualification_s{seed}/{name}"] = sha256(path / name)
    return {"source": str(source), "files": files, "task_source": meta["task_source"]}


def validate_cache(meta, source, seed, cache):
    rows = read_json(source / "val_grounded_plan.json")["rows"]
    d = cache["val"]
    if d["rows"] != rows or len(cache["train"]["rows"]) != meta["budget"]:
        raise ValueError("Cache row inventory differs")
    if cache["target_stats"] != read_json(source / f"qualification_s{seed}/target_stats.json"):
        raise ValueError("Target scaling differs")
    for split in ("train", "val"):
        v = cache[split]
        n = len(v["rows"])
        if (
            v["z"].ndim != 3
            or v["z"].shape[:2] != (3, n)
            or v["q"].shape != (3, n, 2, core.SLOTS, 2)
            or v["y"].shape != (3, n, 2, core.SLOTS)
        ):
            raise ValueError("Cache dimensions differ")
        if not all(torch.isfinite(v[k]).all() for k in ("z", "q", "y")):
            raise ValueError("Nonfinite cache")
    a, m, _ = sd.tensors(rows, np.full(len(rows), 16), "cpu")
    if not torch.equal(core.queries(a, m), d["q"]):
        raise ValueError("Validation query coordinates differ from raw-linked plan")
    y = cache["train"]["y"].double()
    if abs(float(y.mean())) > 2e-6 or abs(float(y.std(unbiased=False)) - 1) > 2e-6:
        raise ValueError("Cached training targets are not standardized as recorded")
    scale = cache["target_stats"]
    if not np.isfinite([scale["mean"], scale["scale"]]).all() or scale["scale"] <= 0:
        raise ValueError("Invalid target scale")
    raw = d["y"].double() * scale["scale"] + scale["mean"]
    if not torch.allclose(raw, raw[1:2].expand_as(raw), atol=2e-5, rtol=2e-5):
        raise ValueError("Cached common targets disagree")


def from_standardized(query, state):
    """Same original HistoryQuery layers; cache already applied its frozen scaler.

    Avoid a lossy inverse-normalize/normalize round trip. No encoder or raw bars.
    Unit tests compare this path exactly with HistoryQuery.forward on raw states.
    """
    c = query.config
    memory = query.memory(state).reshape(len(state), c["slots"], c["width"])
    ages = torch.arange(1, 128, device=state.device)
    q = query.query(query.age_encoding[ages]).expand(len(state), -1, -1)
    for block in query.blocks:
        q = block(q, memory)
    return query.output(q)


def interval_changes(pred, coordinates, statistics):
    if (
        pred.shape[-2:] != (127, 7)
        or coordinates.ndim != 4
        or coordinates.shape[1:] != (2, core.SLOTS, 2)
    ):
        raise ValueError("Full historical predictions and ordered two-query geometry required")
    if ((coordinates < 1) | (coordinates > 127)).any() or not (
        coordinates[..., 0] > coordinates[..., 1]
    ).all():
        raise ValueError("Current/future or inverted query")
    price = core.hq.physical(pred, statistics)[..., 0]
    row = torch.arange(len(price), device=price.device)[:, None, None, None]
    endpoints = price[row, coordinates - 1]
    # Later minus earlier; the unknown display/current anchor cancels exactly.
    return endpoints[..., 1] - endpoints[..., 0]


@torch.no_grad()
def original_predictions(query, dataset, statistics, target_stats, device, batch=128):
    results = []
    for start in range(0, dataset["z"].shape[1], batch):
        z = dataset["z"][:, start : start + batch].to(device)
        q = dataset["q"][:, start : start + batch].to(device)
        pred = from_standardized(query, z.reshape(-1, z.shape[-1]))
        raw = interval_changes(pred, q.reshape(-1, 2, core.SLOTS, 2), statistics)
        results.append(
            ((raw - target_stats["mean"]) / target_stats["scale"])
            .reshape(3, z.shape[1], 2, core.SLOTS)
            .cpu()
            .numpy()
        )
    return np.concatenate(results, 1)


def history_check(meta, path, ablation):
    h = read_json(path / f"{ablation}_history.json")
    expected_steps = (meta["budget"] + meta["batch"] - 1) // meta["batch"]
    if h["epoch"] != meta["qualification_epochs"] or len(h["history"]) != h["epoch"]:
        raise ValueError("Qualification budget differs")
    for i, row in enumerate(h["history"], 1):
        if row["epoch"] != i or row["steps"] != expected_steps or not np.isfinite(row["val_mse"]):
            raise ValueError("Qualification history differs")
    best = min(h["history"], key=lambda row: row["val_mse"])
    if (h["best_epoch"], h["best_loss"]) != (best["epoch"], best["val_mse"]):
        raise ValueError("Qualification selection differs")
    return h


@torch.no_grad()
def replay_auxiliary(meta, source, seed, cache, device):
    path = source / f"qualification_s{seed}"
    identity = read_json(path / "completion.json")["identity"]
    predictions = {}
    losses = {}
    for ablation in qw.ABLATIONS:
        h = history_check(meta, path, ablation)
        if ablation == "none":
            ck = torch.load(path / "selected.pt", map_location="cpu", weights_only=True)
            if ck["identity"] != identity or ck["target_stats"] != cache["target_stats"]:
                raise ValueError("Selected head binding differs")
            state = ck["head"]
        else:
            ck = torch.load(path / f"{ablation}_resume.pt", map_location="cpu", weights_only=True)
            if ck["identity"] != identity or any(
                ck[k] != h[k] for k in ("epoch", "history", "best_epoch", "best_loss")
            ):
                raise ValueError("Ablation resume/history selection differs")
            state = ck["best_head"]
        head = core.GroundedReader(cache["val"]["z"].shape[-1], seed, meta["shared"]["width"]).to(
            device
        )
        head.load_state_dict(state, strict=True)
        head.eval().requires_grad_(False)
        pred = qw.predict(head, cache["val"], device, ablation)
        loss = float((pred - cache["val"]["y"]).square().mean())
        if not np.isclose(loss, h["best_loss"], atol=1e-6, rtol=2e-5):
            raise ValueError(
                f"{ablation} selected validation MSE replay differs: actual={loss}, expected={h['best_loss']}"
            )
        predictions[ablation] = pred.numpy()
        losses[ablation] = loss
    return predictions, losses


def compare(predictions, target, rows, scale):
    original_target = np.asarray(target)
    target = np.asarray(target, dtype=np.float64)
    n = len(rows)
    if target.shape != (3, n, 2, core.SLOTS) or not np.isfinite(target).all():
        raise ValueError("Invalid targets")
    for p in predictions.values():
        if np.shape(p) != target.shape or not np.isfinite(p).all():
            raise ValueError("Invalid prediction")
    informative = np.square(original_target[1, :, 0] - original_target[1, :, 1]).mean(-1) >= 0.05
    result = {}
    for cohort, keep in (("all", np.ones(n, bool)), ("informative", informative)):
        rs = [r for r, k in zip(rows, keep, strict=True) if k]
        result[cohort] = {}
        for v, name in enumerate(VIEWS):
            y = target[v, keep]
            errors = {}
            scores = {}
            for key, p in predictions.items():
                e = np.asarray(p, dtype=np.float64)[v, keep] - y
                errors[key] = np.square(e).mean((1, 2))
                scores[key] = {
                    "mse": float(errors[key].mean()) if len(rs) else None,
                    "mae_bps": float(np.abs(e).mean() * scale * 100) if len(rs) else None,
                }
            cis = {}
            for a, b in (
                ("original", "none"),
                ("original", "state_only"),
                ("original", "query_only"),
                ("original", "train_mean"),
                ("none", "state_only"),
            ):
                cis[f"{a}/{b}"] = qw.interval(errors[a], errors[b], rs, factor=1.0)
            for key in ("original", "none"):
                wrong = np.asarray(predictions[key], dtype=np.float64)[v, keep][:, ::-1]
                cis[f"{key}/wrong_query"] = qw.interval(
                    errors[key], np.square(wrong - y).mean((1, 2)), rs, factor=1.0
                )
            result[cohort][name] = {"rows": len(rs), "scores": scores, "paired_mse_intervals": cis}
    return result


def run(source, out, device="cuda"):
    source = source.resolve()
    out = out.resolve()
    gr.parent.warm.check_output(source, out)
    progress(
        "Checking pinned grounded cache and original control lineage; zero training/fitting/encoder forwards"
    )
    identity = source_identity(source)
    meta = read_json(source / "manifest.json")
    rt = read_json(source / "runtime.json")
    if rt["torch"] != str(torch.__version__) or rt["numpy"] != np.__version__:
        raise ValueError("Keep original Torch/NumPy for numeric replay")
    manifest = {
        "schema": SCHEMA,
        "source": identity,
        "code_sha256": gr.code_identity() | {Path(__file__).name: sha256(__file__)},
        "device": device,
        "optimizer_updates": 0,
        "fits": 0,
        "encoder_forwards": 0,
        "batch": 128,
        "scope": "Fixed qualification validation cohort only; prior validation-selected heads; diagnostic, no new model selection/promotion.",
        "units": "Predicted log-percent interval close change; inherited per-seed train scalar. MAE times100 gives log basis points.",
        "counterfactual": "Both queries on the identical cached normalized state; swap queries without changing truth.",
        "statistics": "Report all rows and original informative subset, paired whole-contract bootstrap2000, factor1; uncorrected and repeatedly used validation.",
    }
    if out.exists() and any(out.iterdir()) and read_json(out / "manifest.json") != manifest:
        raise ValueError("Different audit identity: use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, out / "manifest.json")
    if (out / "completion.json").exists():
        done = read_json(out / "completion.json")
        if done["status"] != "complete":
            raise ValueError("Invalid completion")
        gr.parent.verify_files(out, done["files"])
        return
    summaries = {}
    for seed in (42, 43):
        progress(f"s{seed}: checking frozen cache and replaying three selected heads; zero fitting")
        path = source / f"qualification_s{seed}"
        cache = torch.load(path / "cache.pt", map_location="cpu", weights_only=True)
        validate_cache(meta, source, seed, cache)
        aux, mse = replay_auxiliary(meta, source, seed, cache, device)
        d = cache["val"]
        y = d["y"].numpy()
        rows = d["rows"]
        replay = qw.qualify(aux, y, rows)
        expected = read_json(path / "qualification.json")
        atomic_json(
            {"actual": replay, "expected": expected, "selected_mse": mse},
            out / f"s{seed}_qualification_replay.json",
        )
        gr.xr.ce.require_nested(replay, expected, "original qualification full replay")
        # CPU construction verifies original full source restoration; encoder is discarded unused.
        model, query = gr.construct(meta, seed, "cpu")
        del model
        query.to(device).eval().requires_grad_(False)
        pred = original_predictions(query, d, gr.stats(meta)[0], cache["target_stats"], device)
        predictions = dict(aux, original=pred, train_mean=np.zeros_like(y))
        informative = np.square(y[1, :, 0] - y[1, :, 1]).mean(-1) >= 0.05
        payload = {
            "seed": seed,
            "rows": rows,
            "row_ids": list(range(len(rows))),
            "informative_ids": np.flatnonzero(informative).tolist(),
            "queries_ages": d["q"].tolist(),
            "target_stats": cache["target_stats"],
            "targets": y.tolist(),
            "predictions": {k: v.tolist() for k, v in predictions.items()},
            "cache_sha256": sha256(path / "cache.pt"),
            "cache_dtype": "float32",
            "auxiliary_prediction_dtype": "float32",
            "original_prediction_dtype": "float64",
            "comparison_arithmetic": "float64 errors; original informative filter in float32",
            "note": "All values use the recorded training-only target normalization; raw log-percent=value*scale+mean.",
        }
        atomic_json(payload, out / f"s{seed}_predictions.json")
        summaries[str(seed)] = compare(predictions, y, rows, cache["target_stats"]["scale"])
        progress(
            f"s{seed}: original/new head comparison exported for all{len(rows)} rows and{informative.sum()} informative rows"
        )
        del query, cache
    atomic_json(summaries, out / "metrics.json")
    atomic_json(
        {
            "status": "diagnostic_complete_no_promotion",
            "optimizer_updates": 0,
            "fits": 0,
            "encoder_forwards": 0,
            "interpretation": "Original head outperforming new head establishes a reader limitation for these intervals. Both weak does not prove information absent from the embedding. This is short-interval price content, not a full macro-shape test.",
        },
        out / "decision.json",
    )
    if source_identity(source) != identity:
        raise ValueError("Source changed during audit")
    atomic_json(
        {
            "status": "complete",
            "source_unchanged": True,
            "files": {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file()
                and p.name not in ("completion.json", "run_status.txt")
                and p.suffix not in (".log", ".tmp")
            },
        },
        out / "completion.json",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="checkpoints/babel_grounded_history768")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    gr.ur.bb.ab.configure_runtime()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    source = Path(a.source).resolve()
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        (source.parent / ("." + source.name + ".lock")).open("a") as source_lock,
        (out.parent / ("." + out.name + ".lock")).open("a") as lock,
    ):
        try:
            fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Source experiment or audit already running") from None
        run(source, out, a.device)


if __name__ == "__main__":
    main()
