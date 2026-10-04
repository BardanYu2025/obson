"""V07 frozen encoder/query crosses; no fitting or automatic model selection."""

import argparse
import fcntl
import gc
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import history_structure as hs
from . import recovery_lr as lr
from .ae_extend import atomic_json
from .dual_state import sha256
from .holdout_audit import read_json
from .progress import progress

SCHEMA = "babel-recovery-interface-v1"
R1_COMPLETION = "d6ef283d86910327686b1dd5a3dce8bc4e47df8da3f31af154ccc9389ea24456"
SESSION_CAP = 1800
COMBINATIONS = {"source": (0, 0), "joint": (5, 5), "encoder_swap": (5, 0), "query_swap": (0, 5)}
old, r0 = lr.old, lr.r0


def implementation():
    return {
        "r1": lr.implementation(),
        "module": sha256(__file__),
        "script": sha256(lr.REPO / "scripts/babel_recovery_interface_autodl.sh"),
        "protocol": sha256(lr.REPO / "docs/BABEL_RECOVERY_INTERFACE_PLAN.md"),
    }


def separate(out, *sources):
    out = Path(out).resolve()
    for value in sources:
        p = Path(value).resolve()
        if out == p or out in p.parents or p in out.parents:
            raise ValueError("Output must be separate from every source")
    return out


def verify_source(source, out):
    if sha256(source / "completion.json") != R1_COMPLETION:
        raise ValueError("Expected reviewed R1 completion, not another run")
    done = read_json(source / "completion.json")
    for name, digest in done["files"].items():
        r0.required_file(source, name, digest)
    if not (done["source_unchanged"] and done["optimizer_updates"] == 380):
        raise ValueError("R1 source qualification differs")
    rm = read_json(source / "manifest.json")
    if rm["implementation"] != lr.implementation():
        raise ValueError("R1 bound implementation changed")
    context, qual = Path(rm["source"]).resolve(), Path(rm["qualification"]).resolve()
    separate(out, source, context, qual)
    meta, _ = lr.verify_qualification(context, qual, out)
    if sha256(out / "identity.json") != rm["identity_sha256"]:
        raise ValueError("R1 source/cache identity changed")
    return meta, rm, done


def structure_parts(pred, target, mask, stats):
    """Squared physical coarse errors, including explicit support for each width."""
    if pred.shape != target.shape or pred.shape != mask.shape or pred.shape[-2:] != (127, 7):
        raise ValueError("Matched127 historical query tensors required")
    p, y = hs.hq.physical(pred, stats)[..., 0], hs.hq.physical(target, stats)[..., 0]
    values, support = {}, {}
    for width in hs.WIDTHS:
        a, valid, centers = hs.pools(p, mask[..., 0], width)
        b, _, _ = hs.pools(y, mask[..., 0], width)
        level = (a - b) / (stats["delta_scale"][0] * centers.sqrt())
        trend = (a[..., 1:] - a[..., :-1] - (b[..., 1:] - b[..., :-1])) / (
            stats["delta_scale"][0] * (centers[1:] - centers[:-1]).sqrt()
        )
        for name, error, ok in (
            ("level", level, valid),
            ("trend", trend, valid[..., 1:] & valid[..., :-1]),
        ):
            key = f"structure_{name}{width}"
            values[key] = hs.masked_mean(error.square(), ok)
            support[key] = ok.any(-1)
    for name in ("level", "trend"):
        keys = [f"structure_{name}{w}" for w in hs.WIDTHS]
        valid = torch.stack([support[k] for k in keys], -1)
        values[f"structure_{name}"] = hs.masked_mean(
            torch.stack([values[k] for k in keys], -1), valid
        )
        support[f"structure_{name}"] = valid.any(-1)
    primary = 0.5 * (values["structure_level"] + values["structure_trend"])
    expected = hs.metrics(pred, target, mask, stats)["primary"]
    if not torch.allclose(primary, expected, atol=r0.ATOL, rtol=r0.RTOL):
        raise ValueError("Structure decomposition differs from original primary")
    support["structure"] = support["structure_level"] | support["structure_trend"]
    return values, support


def save_arrays(path, arrays):
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix(".tmp").replace(path)


def aggregates(arrays):
    return {
        f"{view}/{context}/{metric}": float(arrays[f"{view}/{metric}"][:, sl].mean(1).mean())
        for view in ("full", "native")
        for context, sl in (("endpoint", slice(-1, None)), ("interior", slice(None, -1)))
        for metric in old.core.METRICS
    }


def describe(arrays, supports):
    result = {}
    for key, a in arrays.items():
        if not np.isfinite(a).all() or (a < 0).any() or a.shape != supports[key].shape:
            raise ValueError("Invalid error/support array")
        rows = []
        for j, prefix in enumerate(old.core.PREFIXES):
            v = a[:, j][supports[key][:, j]]
            rows.append(
                {
                    "prefix": prefix,
                    "n": len(v),
                    "mean": float(v.mean()) if len(v) else None,
                    "p50": float(np.quantile(v, 0.5)) if len(v) else None,
                    "p95": float(np.quantile(v, 0.95)) if len(v) else None,
                }
            )
        result[key] = rows
    return result


@torch.no_grad()
def evaluate(meta, encoder, query, x, device):
    encoder.eval().requires_grad_(False)
    query.eval().requires_grad_(False)
    stats = old.parent.stats(meta)[0]
    arrays, supports = {}, {}
    for start in range(0, len(x), meta["micro"]):
        b = torch.as_tensor(x[start : start + meta["micro"]], device=device)
        ps = torch.tensor(old.core.PREFIXES, device=device).expand(len(b), -1)
        for view, native in (("full", False), ("native", True)):
            _, pred = old.core.predict(encoder, query, b, ps, native)
            y, mask = old.core.original.targets(b, ps, stats, native)
            raw = old.core.original.band_rows(pred, y, mask, stats)
            values = {k: raw[k] for k in old.core.METRICS}
            extra, valid = structure_parts(pred, y, mask, stats)
            values.update(extra)
            valid.update(
                price=mask[..., :2].any(-1).any(-1), activity=mask[..., 2:].any(-1).any(-1)
            )
            for name, (lo, hi) in zip(("near", "mid", "far"), hs.hq.BANDS, strict=True):
                valid[name] = mask[..., lo - 1 : hi, :2].any(-1).any(-1)
            for k, v in values.items():
                key = view + "/" + k
                arrays.setdefault(key, []).append(v.cpu().numpy())
                supports.setdefault(key, []).append(valid[k].cpu().numpy())
        if start and start % (meta["micro"] * 32) == 0:
            progress(f"V07 readout: {min(start + meta['micro'], len(x))}/{len(x)} endpoints")
    return (
        {k: np.concatenate(v) for k, v in arrays.items()},
        {k: np.concatenate(v) for k, v in supports.items()},
    )


def load_pair(encoder, query, states, epochs):
    for name, model, epoch in zip(("encoder", "query"), (encoder, query), epochs, strict=True):
        model.load_state_dict(states[epoch][name], strict=True)
        model.eval().requires_grad_(False)
    return lr.model_signature(encoder, query)


def contrasts(bank, support):
    result = {}
    for key in bank["source"]:
        a, b, c, d = [
            bank[label][key] for label in ("source", "encoder_swap", "query_swap", "joint")
        ]
        de, dd, interaction = b - a, c - a, d - b - c + a
        if not np.allclose(de + dd + interaction, d - a, atol=r0.ATOL, rtol=r0.RTOL):
            raise ValueError("Crossed contrast accounting differs")
        rows = []
        for j, prefix in enumerate(old.core.PREFIXES):
            keep = support[key][:, j]
            row = {"prefix": prefix, "n": int(keep.sum())}
            source = float(a[keep, j].mean()) if keep.any() else None
            row["source_mean"] = source
            for name, values in (
                ("encoder_delta", de),
                ("query_delta", dd),
                ("interaction", interaction),
                ("joint_delta", d - a),
            ):
                v = values[keep, j]
                row[name] = {
                    "mean": float(v.mean()) if len(v) else None,
                    "p50": float(np.quantile(v, 0.5)) if len(v) else None,
                    "p95": float(np.quantile(v, 0.95)) if len(v) else None,
                }
            row["source_ratios"] = {
                label: float(values[key][keep, j].mean()) / source
                if source is not None and source > 0
                else None
                for label, values in bank.items()
            }
            rows.append(row)
        result[key] = rows
    return result


@torch.no_grad()
def run_seed(meta, source, out, seed, val, device="cuda"):
    folder = out / f"s{seed}"
    folder.mkdir(exist_ok=True)
    encoder, query = old.construct(meta, seed, device)
    states = {0: {"encoder": old.cpu_state(encoder), "query": old.cpu_state(query)}}
    done = read_json(source / f"s{seed}/completion.json")
    if lr.model_signature(encoder, query) != done["initial_signature"]:
        raise ValueError("Original module signatures differ")
    ck = torch.load(source / f"s{seed}/last.pt", map_location="cpu", weights_only=True)
    if (
        ck["epoch"] != 5
        or ck["binding"] != done["binding"]
        or ck["current_signature"] != done["final_signature"]
    ):
        raise ValueError("R1 checkpoint identity differs")
    states[5] = {k: ck[k] for k in ("encoder", "query")}
    del ck  # No optimizer is constructed or restored; release serialized Adam tensors.
    expected = read_json(source / f"s{seed}/comparison.json")
    bank, common_support, signatures = {}, None, {}
    for label, epochs in COMBINATIONS.items():
        progress(f"V07 s{seed}/{label}: frozen E{epochs[0]} + D{epochs[1]}")
        signature = load_pair(encoder, query, states, epochs)
        for name, epoch in zip(("encoder", "query"), epochs, strict=True):
            reference = done["initial_signature" if epoch == 0 else "final_signature"][name]
            if signature[name] != reference:
                raise ValueError("Loaded cross differs from bound module signature")
        arrays, supports = evaluate(meta, encoder, query, val, device)
        save_arrays(folder / f"{label}_errors.npz", arrays)
        save_arrays(folder / f"{label}_support.npz", supports)
        means = aggregates(arrays)
        atomic_json(
            {"aggregate24": means, "positions": describe(arrays, supports)},
            folder / f"{label}_metrics.json",
        )
        if label in ("source", "joint"):
            lr.replay_check(
                means,
                expected["initial" if label == "source" else "low_lr_epoch5"],
                folder / f"{label}_replay.json",
            )
        if common_support is not None:
            if any(not np.array_equal(supports[k], common_support[k]) for k in supports):
                raise ValueError("Crossed readers have different target support")
        else:
            common_support = supports
        r0.causality(encoder, query, torch.as_tensor(val[:2], device=device), folder, label)
        if lr.model_signature(encoder, query) != signature:
            raise ValueError("Frozen evaluation mutated parameters/buffers")
        signatures[label] = signature
        bank[label] = {k: a.astype(np.float64) for k, a in arrays.items()}
    atomic_json(contrasts(bank, common_support), folder / "contrasts.json")
    atomic_json(
        {"loaded_modules": signatures, "optimizer_updates": 0, "selected_model": None},
        folder / "receipt.json",
    )
    del encoder, query, states
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def execute(source, out):
    state = {
        "schema": SCHEMA,
        "task_ids": ["V07"],
        "status": "running",
        "optimizer_updates": 0,
        "rs_authorized": False,
    }
    atomic_json(state, out / "status.json")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Real-weight evaluation requires AutoDL CUDA")
        progress("V07: verifying reviewed R1, source lineage and immutable code")
        meta, rm, _ = verify_source(source, out)
        if torch.__version__ != rm["torch"] or torch.cuda.get_device_name() != rm["gpu"]:
            raise ValueError("Use reviewed R1 CUDA environment for numerical replay")
        manifest = {
            "schema": SCHEMA,
            "source": str(source),
            "r1_completion": R1_COMPLETION,
            "implementation": implementation(),
            "combinations": COMBINATIONS,
            "runtime": r0.RUNTIME_PROFILES["context"],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "micro": meta["micro"],
            "session_cap": SESSION_CAP,
        }
        # JSON roundtrip normalizes tuples for exact restart comparison.
        import json

        manifest = json.loads(json.dumps(manifest))
        if (out / "manifest.json").exists() and read_json(out / "manifest.json") != manifest:
            raise ValueError("Run manifest changed")
        atomic_json(manifest, out / "manifest.json")
        if (out / "completion.json").exists():
            done = read_json(out / "completion.json")
            for name, digest in done["files"].items():
                r0.required_file(out, name, digest)
            atomic_json({k: v for k, v in done.items() if k != "files"}, out / "status.json")
            return 0
        if shutil.disk_usage(out).free < 512 * 2**20:
            raise ValueError("Need512MiB for diagnostics/export, no weights are copied")
        val, rows = old.data.cache(Path(rm["source"]), "val")
        if len(val) != 978 or rows != read_json(source / "validation_rows.json"):
            raise ValueError("Validation row identities changed")
        atomic_json(rows, out / "validation_rows.json")
        shutil.copyfile(lr.REPO / "docs/BABEL_RECOVERY_INTERFACE_PLAN.md", out / "protocol.md")
        with r0.replay_runtime("context"):
            for seed in (42, 43):
                run_seed(meta, source, out, seed, val)
        verify_source(source, out)
        if implementation() != manifest["implementation"]:
            raise ValueError("Implementation changed during evaluation")
        state.update(
            status="interface_complete_requires_review",
            source_unchanged=True,
            combinations=8,
            selected_model=None,
        )
        files = {
            str(p.relative_to(out)): sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
            and p.suffix in (".json", ".npz", ".md")
            and p.name not in ("status.json", "completion.json", "request.json", "budget.json")
        }
        atomic_json({**state, "files": files}, out / "completion.json")
        atomic_json(state, out / "status.json")
        return 0
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        atomic_json(state, out / "status.json")
        raise


def supervise(source, out):
    source = Path(source).resolve()
    out = separate(out, source)
    request = {"schema": SCHEMA, "source": str(source), "output": str(out)}
    if (
        out.exists()
        and any(out.iterdir())
        and (not (out / "request.json").exists() or read_json(out / "request.json") != request)
    ):
        raise ValueError("Refuse existing unbound output")
    # Reject ancestor/source collisions before creating any files.
    if (source / "manifest.json").exists():
        rm = read_json(source / "manifest.json")
        separate(out, rm["source"], rm["qualification"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(request, out / "request.json")
        budget = (
            read_json(out / "budget.json")
            if (out / "budget.json").exists()
            else {"used_seconds": 0.0, "attempts": []}
        )
        used = float(budget["used_seconds"])
        if not math.isfinite(used) or not 0 <= used < SESSION_CAP:
            raise ValueError("V07 cumulative budget exhausted")
        remaining = SESSION_CAP - used
        attempt = {"started_unix": time.time(), "reserved_seconds": remaining}
        budget["attempts"].append(attempt)
        budget["used_seconds"] = SESSION_CAP
        atomic_json(budget, out / "budget.json")
        started = time.monotonic()
        try:
            code = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "obson.babel.recovery_interface",
                    "--worker",
                    "--source",
                    str(source),
                    "--out",
                    str(out),
                ],
                timeout=remaining,
            ).returncode
        except subprocess.TimeoutExpired:
            code = 124
            atomic_json(
                {
                    "schema": SCHEMA,
                    "task_ids": ["V07"],
                    "status": "timeout",
                    "optimizer_updates": 0,
                    "rs_authorized": False,
                },
                out / "status.json",
            )
        elapsed = min(remaining, time.monotonic() - started)
        budget["used_seconds"] = used + elapsed
        attempt.update(elapsed_seconds=elapsed, exit_code=code)
        atomic_json(budget, out / "budget.json")
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    source, out = a.source.resolve(), separate(a.out, a.source)
    if a.worker:
        with (out / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raise SystemExit(execute(source, out))
    raise SystemExit(supervise(source, out))


if __name__ == "__main__":
    main()
