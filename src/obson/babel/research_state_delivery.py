"""Package control600 and replay its existing evidence without any fitting."""

import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import local_warmstart_run as warm
from . import research_state as rs
from . import research_state_view as view
from . import window_state as ws
from .ae_extend import atomic_json, atomic_save
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-control600-delivery-v1"
xe, xr = warm.xe, warm.xr


def code_identity():
    return (
        rs.code_identity()
        | {Path(m.__file__).name: sha256(m.__file__) for m in (view,)}
        | {
            Path(__file__).name: sha256(__file__),
            "research_state_view.html": sha256(Path(view.__file__).with_suffix(".html")),
        }
    )


def build_bundle(meta, out):
    bundle = out / "bundle"
    bundle.mkdir(exist_ok=True)
    tm = meta["identity"]["training_manifest"]
    root = Path(meta["identity"]["training_source"])
    stats, local = xr.statistics(tm)
    fit = read_json(root / "fit.json")
    names = {}
    periods = sorted(
        {
            int(r["period"])
            for s in ("train", "val", "test", "cross_research")
            for r in read_json(root / f"{s}_inventory.json")
        }
    )
    with np.load(root / "heads.npz", allow_pickle=False) as heads:
        for seed in (42, 43):
            job = next(j for j in tm["experiments"] if j["name"] == f"control_s{seed}")
            model, _ = xe.load(tm, root, job, "best", "cpu")
            if model.core.decoder.residual_enabled:
                raise ValueError("Expected fixed original PCA decoder")
            input_layer = model.core.encoder.backbone.input
            if type(input_layer) is rs.pf.PathInput:
                input_adapter = "causal_path_v1"
            elif type(input_layer) is torch.nn.Linear:
                input_adapter = "linear28"
            else:
                raise ValueError(
                    f"Unsupported source input architecture: {type(input_layer).__name__}"
                )
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            config = dict(
                rs.ba.pt.CONFIG,
                latent=model.core.encoder.coordinates.out_features,
                attention_layers=len(model.core.encoder.backbone.layers),
                attention_ff=model.core.encoder.backbone.layers[0].linear1.out_features,
                heads=8,
                residual_width=state["core.decoder.residual.0.weight"].shape[0],
            )
            name = f"control_s{seed}_best"
            checkpoint = {
                "schema": rs.SCHEMA,
                "seed": seed,
                "config": config,
                "input_adapter": input_adapter,
                "model": state,
                "statistics": stats,
                "local": local,
                "utility": {
                    "heads": fit["heads"][name]["targets"],
                    "weights": heads[name + "_weights"].tolist(),
                    "intercepts": heads[name + "_intercepts"].tolist(),
                    "target_stats": fit["target_stats"],
                },
                "identity": {
                    "job": job["name"],
                    "seed": seed,
                    "checkpoint": "best",
                    "total_epochs": 600,
                    "selected_epoch": 100,
                    "source_weight_sha256": sha256(root / job["name"] / "best.pt"),
                    "supported_periods": periods,
                    "role": "research_candidate",
                    "automatic_promotion": False,
                    "local_head": "original600",
                },
            }
            restored = rs.restore_model(checkpoint)
            if warm.ur.bb.state_signature(restored) != warm.ur.bb.state_signature(model):
                raise ValueError("Portable architecture did not preserve all original weights")
            atomic_save(checkpoint, bundle / f"{name}.pt")
            names[str(seed)] = f"{name}.pt"
    atomic_json(
        {
            "schema": rs.SCHEMA,
            "models": names,
            "code_sha256": rs.code_identity(),
            "files": {n: sha256(bundle / n) for n in names.values()},
            "source_manifest": sha256(out / "manifest.json"),
            "minimum_history": 512,
            "feature_names": list(ws.FEATURES),
            "selection": "Both original control600 validation best; no warm-adapted heads or seed selection",
        },
        bundle / "index.json",
    )
    return bundle


def data_for(meta, split):
    tm = meta["identity"]["training_manifest"]
    return warm.ur.bb.load_data(xr.cr.alignment(xr.parent_meta(tm)), split)


def require_replay(actual, expected, label, out):
    differences = warm.recheck.mismatches(actual, expected)
    atomic_json(
        {"passed": not differences, "differences": differences}, out / f"{label}_replay.json"
    )
    if differences:
        raise ValueError(f"Pinned source replay failed: {label}; see saved differences")


@torch.inference_mode()
def cached_replay(meta, bundle, out, device):
    report_source = Path(meta["source"])
    root = Path(meta["identity"]["training_source"])
    summary = {}
    for split in ("test", "cross_research"):
        data = data_for(meta, split)
        rows = read_json(root / f"{split}_inventory.json")
        examples = []
        for seed in (42, 43):
            name = f"control_s{seed}_best"
            engine = rs.load_bundle(
                bundle, seed, rows[0]["key"], int(rows[0]["period"]), device, _audit=True
            )
            before = warm.ur.bb.state_signature(engine.aligned)
            scores, errors, _, pred = warm.ur.pf.score(
                engine.aligned, data, engine.statistics, engine.local, 64, device
            )
            reference = read_json(report_source / f"{split}_{name}.json")
            nested = {k: {m: v.tolist() for m, v in e.items()} for k, e in errors.items()}
            require_replay(
                nested,
                reference["reconstruction"]["errors"],
                f"{split}_s{seed}_reconstruction",
                out,
            )
            zs = []
            for start in range(0, len(rows), 64):
                x = torch.tensor(np.asarray(data["x"][start : start + 64]), device=device)
                zs.append(engine.model.encoder(x)[:, -1].cpu().numpy())
            z = np.concatenate(zs)
            u = engine.utility
            predicted = rs.up.predict(
                u["heads"],
                np.asarray(u["weights"]),
                np.asarray(u["intercepts"]),
                z,
                u["target_stats"],
            )
            require_replay(
                predicted.tolist(),
                reference["utility"]["predictions"],
                f"{split}_s{seed}_utility",
                out,
            )
            if before != warm.ur.bb.state_signature(engine.aligned):
                raise ValueError("Weights changed during packaging replay")
            summary[f"{split}/s{seed}"] = {
                "windows": len(rows),
                "scores": scores,
                "reconstruction_replay": True,
                "utility_replay": True,
            }
            # Fixed evenly spaced examples, never cherry-picked by errors.
            for i in np.linspace(0, len(rows) - 1, 4, dtype=int):
                examples.append(
                    {
                        "seed": seed,
                        "index": int(i),
                        "row": rows[i],
                        "prediction": pred[i].tolist(),
                        "target": np.asarray(data["y"][i]).tolist(),
                        "mask": np.asarray(data["mask"][i]).tolist(),
                    }
                )
            progress(f"Portable control600 replay: {split}, seed{seed}, {len(rows)} windows")
        atomic_json({"examples": examples}, out / f"{split}_examples.json")
    atomic_json(summary, out / "cached_replay.json")
    return summary


def raw_plan(meta):
    root = Path(meta["identity"]["training_source"])
    raw = Path(meta["identity"]["training_manifest"]["raw_root"])
    plan = []
    for split in ("test", "cross_research"):
        seen = set()
        eligible = []
        for i, row in enumerate(read_json(root / f"{split}_inventory.json")):
            if row["row"] >= ws.WARMUP + 8 and row["key"] not in seen:
                seen.add(row["key"])
                eligible.append((i, row))
        if len(eligible) < 2:
            raise ValueError("Two raw contracts per research set required")
        for i, row in (eligible[0], eligible[-1]):
            symbol, period, contract = row["key"].split("/")
            path = raw / symbol / f"{contract}_{period}m.csv"
            plan.append(
                {"split": split, "index": i, "row": row, "path": str(path), "sha256": sha256(path)}
            )
    return plan


@torch.inference_mode()
def raw_replay(meta, bundle, out, plan, device):
    result = []
    for n, item in enumerate(plan):
        row = item["row"]
        path, end, period = Path(item["path"]), row["row"], int(row["period"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Raw replay file changed")
        df = ws.data.validate_frame(pd.read_csv(path), str(path))
        if str(df.datetime.iloc[end]) != row["end"]:
            raise ValueError("Raw endpoint identity changed")

        def features(frame, period=period):
            return np.column_stack(
                (
                    ws.ae_context.encode_context(frame, period, "ema8_32")["x"],
                    ws.aa.activity_features(frame, period)[0],
                )
            )

        full = features(df)
        prefix = features(df.iloc[: end + 1])
        if not np.array_equal(full[: end + 1], prefix):
            raise ValueError("Future suffix changed prefix features")
        bars = list(ws.frame_bars(df.iloc[: end + 1], row["key"], period))
        cached = np.asarray(data_for(meta, item["split"])["x"][item["index"]])
        for seed in (42, 43):
            engine = rs.load_bundle(bundle, seed, row["key"], period, device, _audit=True)
            restored = rs.load_bundle(bundle, seed, row["key"], period, device, _audit=True)
            for bar in bars[: end - 8]:
                engine.warm(
                    bar, as_of=ws.streaming.timestamp(bar.datetime) + pd.Timedelta(minutes=period)
                )
            restored.restore(engine.snapshot())
            checks = []
            for i in range(end - 8, end + 1):
                bar = bars[i]
                asof = ws.streaming.timestamp(bar.datetime) + pd.Timedelta(minutes=period)
                output = engine.push(bar, as_of=asof)
                if output != restored.push(bar, as_of=asof):
                    raise ValueError("Snapshot continuation differs")
                x = np.stack([v["x"] for v in engine.rows])
                if not np.allclose(x, full[i - 127 : i + 1], atol=1e-6, rtol=2e-5):
                    raise ValueError("Incremental28-feature mismatch")
                normalized = (
                    (full[i - 127 : i + 1] - engine.statistics["x_mean"])
                    / engine.statistics["x_scale"]
                ).astype(np.float32)
                expected = (
                    engine.model.encoder(torch.tensor(normalized[None], device=device))[:, -1]
                    .cpu()
                    .numpy()
                )
                check = warm.recheck.array_replay(np.asarray([output["embedding"]]), expected)
                checks.append(check)
                if not check["passed"]:
                    atomic_json(checks, out / f"raw_{n}_s{seed}_failure.json")
                    raise ValueError("Rolling endpoint/batch mismatch")
                if i == end and not np.allclose(normalized, cached, atol=1e-6, rtol=2e-5):
                    raise ValueError("Raw endpoint differs from original cache")
                if i == end:
                    atomic_json(output, out / f"example_state_{n}_s{seed}.json")
            result.append(
                {
                    "key": row["key"],
                    "seed": seed,
                    "split": item["split"],
                    "steps": 9,
                    "checks": checks,
                    "feature_prefix_equal": True,
                    "snapshot_exact": True,
                    "original_cache_match": True,
                }
            )
        if sha256(path) != item["sha256"]:
            raise ValueError("Raw file mutated during replay")
        progress(f"Rolling control600 replay: {row['key']}, both seeds")
    atomic_json(result, out / "raw_replay.json")


def run(source, out, device="cuda"):
    started = time.time()
    source, out = source.resolve(), out.resolve()
    progress("Verifying completed control600 lineage; zero training and fitting")
    identity = warm.source_identity(source)
    warm.check_output(source, out, identity)
    runtime = read_json(source / "runtime.json")
    if runtime["torch"] != str(torch.__version__) or runtime["numpy"] != np.__version__:
        raise ValueError("Use original Torch/NumPy environment for formal replay")
    meta = {
        "schema": SCHEMA,
        "source": str(source),
        "identity": identity,
        "code_sha256": code_identity(),
        "encoder_updates": 0,
        "head_updates": 0,
        "reader_fits": 0,
        "statistics_fits": 0,
    }
    if out.exists() and any(out.iterdir()) and read_json(out / "manifest.json") != meta:
        raise ValueError("Use a fresh output for a different source/code")
    out.mkdir(parents=True, exist_ok=True)
    (out / "completion.json").unlink(missing_ok=True)
    (out / "bundle/validation.json").unlink(missing_ok=True)
    atomic_json(meta, out / "manifest.json")
    bundle = build_bundle(meta, out)
    plan = raw_plan(meta)
    atomic_json(plan, out / "raw_plan.json")
    cached_replay(meta, bundle, out, device)
    raw_replay(meta, bundle, out, plan, device)
    view.render_source(source, out / "model_review.html")
    if warm.source_identity(source) != identity:
        raise ValueError("Source changed during delivery")
    rs.certify(bundle)
    atomic_json(
        {
            "status": "complete",
            "seconds": time.time() - started,
            "encoder_updates": 0,
            "head_updates": 0,
            "reader_fits": 0,
            "statistics_fits": 0,
            "source_unchanged": True,
            "files": {
                str(p.relative_to(out)): sha256(p)
                for p in out.rglob("*")
                if p.is_file() and p.name not in ("completion.json", "run_status.txt")
            },
        },
        out / "completion.json",
    )
    progress("Validated research bundle and offline model_review.html ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    ws.ea.bb.ab.configure_runtime()
    if not torch.cuda.is_available():
        raise ValueError("Formal real-weight replay runs on AutoDL CUDA")
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / ("." + out.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Research delivery already running") from None
        run(Path(a.source), out)


if __name__ == "__main__":
    main()
