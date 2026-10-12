"""F01/F03 locked-last evaluation, with raw-price truth and explicit support.

No fitting or checkpoint selection occurs here. The runner must supply a lock
verifier before research extraction; an interrupted extraction cannot be replayed
silently. Full query predictions remain as hashed remote shards. Downloadable
rows support every gate; eight fixed positions additionally retain raw evidence.
"""

import fcntl
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from . import multiscale_content as content
from . import multiscale_metrics as price
from . import multiscale_training as training
from . import overlap_content_audit as overlap
from . import state_coverage as coverage
from .ae_extend import atomic_json
from .dual_state import sha256

SCHEMA = "babel-multiscale-evaluation-v1"
POSITIONS = tuple(range(1, 129))
SAMPLE_POSITIONS = (1, 32, 64, 80, 96, 112, 127, 128)
FIELDS = (
    "body",
    "volume_vs_prior_ema32",
    "delta_log_volume",
    "oi_change_percent_asinh",
    "oi_per_volume_asinh",
    "turnover_asinh",
)
OVERLAP_GUARDS = (
    "shape_a_mse",
    "shape_b_mse",
    "common_anchored_a_mse",
    "common_anchored_b_mse",
    "disagreement_mse",
)
DRAWS = 2000
BOOTSTRAP_SEED = 20261012


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _positions(value):
    p = np.asarray(value)
    if (
        p.ndim != 1
        or not len(p)
        or p.dtype.kind not in "iu"
        or (p < 1).any()
        or (p > 128).any()
        or not (np.diff(p) > 0).all()
    ):
        raise ValueError("Unique increasing integer positions1..128 required")
    return p


def _paths(prediction, statistics):
    delta = content._statistics(statistics)
    scale = np.sqrt(np.arange(1, 128, dtype=np.float64)) * delta
    historical = np.asarray(prediction, dtype=np.float64)[..., 0] * scale
    return np.concatenate((historical[..., ::-1], np.zeros((*historical.shape[:-1], 1))), -1)


def score_rows(prediction, target, mask, raw_closes, statistics, positions):
    """CPU gateway: per-parent/per-position physical and six-field errors.

    Standardized nonprice MSE averages valid ages within each band. No valid
    age produces NaN, not zero. Price labels must match original float64 closes;
    the current zero anchor is excluded from every price score.
    """
    p, y, m = map(np.asarray, (prediction, target, mask))
    positions = _positions(positions)
    shape = (len(raw_closes), len(positions), 127, 7)
    if (
        not shape[0]
        or p.shape != shape
        or y.shape != shape
        or m.shape != shape
        or p.dtype.kind != "f"
        or y.dtype != np.float64
        or m.dtype != bool
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or not m[..., 0].all()
    ):
        raise ValueError("Matched finite queries, float64 labels, masks and all prices required")
    truth = price.raw_paths(raw_closes, positions)
    target_paths = _paths(y, statistics)
    if not np.allclose(target_paths, truth, rtol=2e-12, atol=2e-12):
        raise ValueError("Price target differs from original float64 closes")
    values = price.price_rows(_paths(p, statistics), truth)
    errors = {f"price/{key}": value for key, value in values.items()}
    support = {
        f"price/{key}": np.full(p.shape[:2], hi - lo + 1, dtype=np.int64)
        for key, (lo, hi) in price.BANDS.items()
    }
    support["price/composite"] = np.full(p.shape[:2], 127, dtype=np.int64)
    with np.errstate(over="ignore"):
        squared = (p[..., 1:].astype(np.float64) - y[..., 1:]) ** 2
    if not np.isfinite(squared).all():
        raise ValueError("Nonprice squared error overflow")
    for band, (lo, hi) in price.BANDS.items():
        valid = m[..., lo - 1 : hi, 1:]
        counts = valid.sum(-2)
        means = np.divide(
            np.where(valid, squared[..., lo - 1 : hi, :], 0).sum(-2),
            counts,
            out=np.full(counts.shape, np.nan),
            where=counts > 0,
        )
        for j, field in enumerate(FIELDS):
            key = f"nonprice/{field}/{band}"
            errors[key], support[key] = means[..., j], counts[..., j]
    return {
        "errors": errors,
        "support": support,
        "mask_digest": hashlib.sha256(m.tobytes()).hexdigest(),
    }


def aggregate_rows(errors, support, positions=POSITIONS):
    """Equal supported states within each parent, never treating missing as zero."""
    if not np.array_equal(_positions(positions), POSITIONS) or set(errors) != set(support):
        raise ValueError("Complete128 positions and matched support required")
    result = {"errors": {}, "support": {}, "contexts": {}}
    n = None
    for key, value in errors.items():
        a, count = np.asarray(value), np.asarray(support[key])
        if (
            a.ndim != 2
            or a.shape[1] != 128
            or not len(a)
            or (n is not None and len(a) != n)
            or count.shape != a.shape
            or count.dtype.kind not in "iu"
            or (count < 0).any()
            or np.isinf(a).any()
            or (a[np.isfinite(a)] < 0).any()
            or not np.array_equal(np.isfinite(a), count > 0)
        ):
            raise ValueError("Finite-or-missing nonnegative rows with exact support required")
        n = len(a)
        for population, part in (
            ("full", slice(None)),
            ("interior", slice(0, 127)),
            ("endpoint", slice(127, 128)),
        ):
            v, c = a[:, part], count[:, part]
            valid = c > 0
            contexts = valid.sum(1)
            name = f"{population}/{key}"
            result["errors"][name] = np.divide(
                np.where(valid, v, 0).sum(1),
                contexts,
                out=np.full(n, np.nan),
                where=contexts > 0,
            )
            result["support"][name] = c.sum(1)
            result["contexts"][name] = contexts
    return result


def overlap_rows(prediction, raw_closes, statistics, positions=SAMPLE_POSITIONS):
    """Original F03 physical-origin removal, retaining truth error and disagreement."""
    positions = _positions(positions)
    p = np.asarray(prediction)
    if p.shape != (len(raw_closes), len(positions), 127, 7) or not np.isfinite(p).all():
        raise ValueError("Matched finite fixed-position overlap queries required")
    if not {64, 112, 127, 128}.issubset(positions):
        raise ValueError("All declared overlap endpoints required")
    paths = _paths(p, statistics) / 100.0
    truth = price.raw_paths(raw_closes, positions) / 100.0
    errors, support = {}, {}
    for shift in (1, 16, 64):
        a, b = list(positions).index(128 - shift), list(positions).index(128)
        measured = overlap.compare(paths[:, a], paths[:, b], truth[:, a], truth[:, b], shift)
        for key, value in measured.items():
            if not isinstance(value, np.ndarray):
                continue
            name = f"overlap/shift{shift}/{key}"
            errors[name] = value
            width = (
                shift
                if key.startswith(("entered", "leaving", "flat_entered", "flat_leaving"))
                else 127 - shift
            )
            support[name] = np.full(len(p), width, dtype=np.int64)
    return {"errors": errors, "support": support}


def _row_labels(rows):
    if not rows:
        raise ValueError("Original nonempty parent identities required")
    months, contracts = [], []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or not row["key"]:
            raise ValueError("Parent contract/period key required")
        month = row.get("month")
        if not isinstance(month, str) or re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month) is None:
            raise ValueError("Original calendar month required")
        months.append(month)
        contracts.append(row["key"] + "/" + month)
    return {"month": months, "contract_period_month": contracts}


def paired_intervals(matrix, rows):
    """Fixed2000 draws of paired parent values, clustered two separate ways."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(rows) or not x.size or np.isinf(x).any():
        raise ValueError("Finite-or-missing paired parent matrix required")
    result = {}
    for kind, labels in _row_labels(rows).items():
        _, codes = np.unique(labels, return_inverse=True)
        groups = int(codes.max()) + 1
        valid = np.isfinite(x)
        sums, counts = np.zeros((groups, x.shape[1])), np.zeros((groups, x.shape[1]))
        np.add.at(sums, codes, np.where(valid, x, 0))
        np.add.at(counts, codes, valid)
        samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, groups, (DRAWS, groups))
        weights = np.asarray([np.bincount(sample, minlength=groups) for sample in samples], float)
        denominator = weights @ counts
        boot = np.divide(
            weights @ sums,
            denominator,
            out=np.full(denominator.shape, np.nan),
            where=denominator > 0,
        )
        result[kind] = []
        for j in range(x.shape[1]):
            points = x[valid[:, j], j]
            draws = boot[np.isfinite(boot[:, j]), j]
            ci = np.quantile(draws, (0.025, 0.975)) if len(draws) else (None, None)
            supported_groups = int((counts[:, j] > 0).sum())
            result[kind].append(
                {
                    "mean": float(points.mean()) if len(points) else None,
                    "low": None if ci[0] is None else float(ci[0]),
                    "high": None if ci[1] is None else float(ci[1]),
                    "parents": len(points),
                    "groups": supported_groups,
                    "valid_draws": len(draws),
                    "supported": bool(
                        len(points) >= 100 and supported_groups >= 6 and len(draws) == DRAWS
                    ),
                }
            )
    return result


def protection_keys():
    keys = [
        f"{population}/price/{band}"
        for population in ("full", "interior", "endpoint")
        for band in price.BANDS
    ]
    keys += [
        f"{population}/nonprice/{field}/{band}"
        for population in ("full", "interior", "endpoint")
        for field in FIELDS
        for band in price.BANDS
    ]
    keys += [f"overlap/shift{shift}/{name}" for shift in (1, 16, 64) for name in OVERLAP_GUARDS]
    return keys


def comparison(candidate, control, rows):
    """All fixed gates; missing support is evidence-insufficient, never a pass."""
    required = ["full/price/composite", *protection_keys()]
    for key in ("mask_digest", "rows_digest", "data_digest"):
        if not candidate.get(key) or candidate[key] != control.get(key):
            raise ValueError("Paired exact mask/parent identity mismatch: " + key)
    if candidate["rows_digest"] != _digest(rows):
        raise ValueError("Comparison rows differ from extraction")
    for family in ("position_support", "support", "contexts"):
        if set(candidate[family]) != set(control[family]):
            raise ValueError("Paired support keys differ")
        if any(
            not np.array_equal(candidate[family][k], control[family][k]) for k in candidate[family]
        ):
            raise ValueError("Paired exact support differs")
        for value in candidate[family].values():
            value = np.asarray(value)
            expected = (len(rows), 128) if family == "position_support" else (len(rows),)
            if value.shape != expected or value.dtype.kind not in "iu" or (value < 0).any():
                raise ValueError("Invalid paired support shape or counts")
    specs = [("gain", required[0], 0.95)] + [("retain/" + k, k, 1.1) for k in required[1:]]
    columns = []
    for _, key, threshold in specs:
        a, b = np.asarray(candidate["errors"][key]), np.asarray(control["errors"][key])
        support = np.asarray(candidate["support"][key])
        if (
            a.shape != (len(rows),)
            or b.shape != a.shape
            or support.shape != a.shape
            or not np.array_equal(np.isfinite(a), support > 0)
            or not np.array_equal(np.isfinite(a), np.isfinite(b))
            or np.isinf(a).any()
            or np.isinf(b).any()
            or (a[np.isfinite(a)] < 0).any()
            or (b[np.isfinite(b)] < 0).any()
        ):
            raise ValueError("Paired metric/support mismatch: " + key)
        if "/price/" in key and not np.isfinite(a).all():
            raise ValueError("All physical price parents must be observed")
        columns.append(a - threshold * b)
    intervals = paired_intervals(np.column_stack(columns), rows)
    checks = {}
    for j, (label, key, threshold) in enumerate(specs):
        a, b = candidate["errors"][key], control["errors"][key]
        valid = np.isfinite(a)
        ca, co = (float(np.mean(v[valid])) if valid.any() else None for v in (a, b))
        ci = {kind: values[j] for kind, values in intervals.items()}
        supported = all(v["supported"] for v in ci.values())
        defined_gain = label != "gain" or (co is not None and co > 0)
        supported = supported and defined_gain
        passed = supported and all(v["high"] <= 0 for v in ci.values())
        checks[label] = {
            "metric": key,
            "candidate_mean": ca,
            "control_mean": co,
            "ratio_of_means": ca / co if co is not None and co > 0 else None,
            "factor": threshold,
            "intervals": ci,
            "status": "pass" if passed else ("fail" if supported else "insufficient"),
            "passed": passed,
            "gain_denominator_defined": defined_gain,
        }
    a = candidate["errors"][required[0]]
    b = control["errors"][required[0]]
    p95a, p95b = float(np.quantile(a, 0.95)), float(np.quantile(b, 0.95))
    tail = {
        "candidate_p95": p95a,
        "control_p95": p95b,
        "factor": 1.1,
        "ratio": p95a / p95b if p95b > 0 else None,
        "passed": p95a <= 1.1 * p95b,
        "scope": "descriptive parent-composite-MSE point guard, not a confidence interval",
    }
    gain = checks["gain"]["passed"]
    protected = all(v["passed"] for k, v in checks.items() if k != "gain") and tail["passed"]
    return {
        "checks": checks,
        "tail": tail,
        "gain_passed": gain,
        "protection_passed": protected,
        "passed": gain and protected,
        "bootstrap": {
            "draws": DRAWS,
            "seed": BOOTSTRAP_SEED,
            "minimum_parents": 100,
            "minimum_groups": 6,
        },
    }


def decisions(banks, row_bank):
    if set(banks) != {"test", "cross_research"} or set(row_bank) != set(banks):
        raise ValueError("Both locked research cohorts required")
    variants = {f"{a}_s{s}" for a in content.ARMS for s in (42, 43)}
    comparisons = {}
    for split, models in banks.items():
        if set(models) != variants:
            raise ValueError("Both arms and both fixed seeds required")
        for seed in (42, 43):
            comparisons[f"{split}/s{seed}"] = comparison(
                models[f"multiscale_s{seed}"], models[f"control_s{seed}"], row_bank[split]
            )
    accepted = all(v["passed"] for v in comparisons.values())
    return {
        "schema": SCHEMA,
        "comparisons": comparisons,
        "limited_hypothesis_supported": accepted,
        "conclusion": "limited_balanced_price_content_gain"
        if accepted
        else "no_accepted_balanced_gain",
        "promoted": False,
        "fit_count": 0,
        "scope": "Fixed scratch recipe on reused research cohorts; no universal embedding, market-regime or prediction acceptance; Macro unchanged",
    }


def summarize(bank):
    result = {}
    for key, values in bank["errors"].items():
        a = np.asarray(values)
        v = a[np.isfinite(a)]
        row = {
            "parents": len(v),
            "missing_parents": len(a) - len(v),
            "mean": float(v.mean()) if len(v) else None,
            "p95": float(np.quantile(v, 0.95)) if len(v) else None,
        }
        if "/price/" in key:
            row["unit"] = "log-percent squared"
            row["aggregate_rmse_bps"] = float(np.sqrt(v.mean()) * 100) if len(v) else None
        elif key.startswith("overlap/") and key.endswith("_mse"):
            row["unit"] = "log-ratio squared"
            row["aggregate_rmse_bps"] = float(np.sqrt(v.mean()) * 10000) if len(v) else None
        else:
            row["unit"] = (
                "standardized training coordinate squared"
                if "/nonprice/" in key
                else "log-ratio squared"
            )
        result[key] = row
    return {
        "schema": SCHEMA,
        "distributions": result,
        "aggregation": "Each parent averages supported contexts equally; missing contexts are excluded, not imputed zero",
        "scope": "All128 metrics come from saved per-position rows; eight fixed positions include predictions and labels; full query shards remain remote",
    }


def _safe_path(out, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe evaluation relative path")
    root = Path(out).resolve()
    path = root / relative
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError("Evaluation paths may not traverse symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Evaluation output escapes run directory")
    return path


def _save_npz(path, arrays):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def _save_npy(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temp.replace(path)


def _pack(bank, families):
    return {f"{family}::{key}": value for family in families for key, value in bank[family].items()}


def _unpack(path):
    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            family, name = key.split("::", 1)
            result.setdefault(family, {})[name] = arrays[key]
    return result


def _read_json(path):
    return json.loads(path.read_text())


def _load_complete(out, receipt, binding):
    saved = _read_json(receipt)
    if (
        saved.get("schema") != SCHEMA
        or saved.get("binding") != binding
        or saved.get("status") != "complete"
    ):
        raise ValueError("Completed evaluation binding changed")
    for relative, expected in saved["files"].items():
        path = _safe_path(out, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("Completed evaluation fingerprint changed: " + relative)
    bank = _unpack(_safe_path(out, saved["row_file"]))
    positions = _unpack(_safe_path(out, saved["position_file"]))
    bank["position_support"] = positions["support"]
    bank["mask_digest"] = saved["mask_digest"]
    bank["rows_digest"] = saved["rows_digest"]
    bank["data_digest"] = saved["data_digest"]
    bank["receipt"] = str(receipt.relative_to(out))
    return bank


@torch.no_grad()
def extract(meta, out, name, encoder, query, x, raw_closes, rows, split, device, *, verify_lock):
    """One bounded extraction per model/cohort; complete receipts are resumable.

    ``verify_lock(meta,out,name,split)`` must certify the COMPLETE four-model
    final120 matrix, source/data lineage and return a JSON binding with
    ``model_lock`` and ``source``. It is called before any model forward and
    again before committing. The root runner owns deadlines and failure export.
    """
    if not callable(verify_lock):
        raise ValueError("Research extraction requires the runner lock verifier")
    if name not in {f"{a}_s{s}" for a in content.ARMS for s in (42, 43)} or split not in (
        "test",
        "cross_research",
    ):
        raise ValueError("Fixed four-model/two-cohort extraction required")
    out = Path(out)
    if out.is_symlink():
        raise ValueError("Run output cannot be a symlink")
    out = out.resolve()
    if not out.is_dir():
        raise ValueError("Runner must create a safe output directory first")
    for key in ("source", "raw_root"):
        if key in meta:
            original = Path(meta[key]).resolve()
            if out == original or out.is_relative_to(original) or original.is_relative_to(out):
                raise ValueError("Evaluation output must be separate from source/raw trees")
    x, raw_closes = np.asarray(x), np.asarray(raw_closes)
    if (
        x.shape != (len(rows), 255, 28)
        or x.dtype != np.float32
        or raw_closes.shape != (len(rows), 255)
        or raw_closes.dtype != np.float64
        or not np.isfinite(x).all()
        or not np.isfinite(raw_closes).all()
        or not (raw_closes > 0).all()
    ):
        raise ValueError("Matched fixed feature/raw-price/parent inventory required")
    _row_labels(rows)
    if meta.get("micro") != 8 or meta.get("batch") != 128 or meta.get("epochs") != 120:
        raise ValueError("Fixed micro8/batch128/last120 extraction required")
    content.huber_coordinate_weights(meta["scales"])
    lock = verify_lock(meta, out, name, split)
    if not isinstance(lock, dict) or not lock.get("model_lock") or not lock.get("source"):
        raise ValueError("Verifier must bind the complete model matrix and source")
    before = training.parameter_signature(encoder, query)
    binding = {
        "lock": lock,
        "name": name,
        "split": split,
        "model_signature": before,
        "rows_digest": _digest(rows),
        "x_sha256": training.array_hash(x),
        "raw_sha256": training.array_hash(raw_closes),
        "statistics_digest": _digest(meta["statistics"]),
        "scales": list(meta["scales"]),
        "positions": list(POSITIONS),
        "sample_positions": list(SAMPLE_POSITIONS),
    }
    data_digest = _digest(
        {key: value for key, value in binding.items() if key not in ("name", "model_signature")}
    )
    prefix = f"evaluation/{name}/{split}"
    directory = _safe_path(out, prefix)
    shards = _safe_path(out, f"predictions/{name}/{split}")
    receipt = _safe_path(out, prefix + "/complete.json")
    ledger_path = _safe_path(out, "evaluation_attempts.json")
    lock_path = _safe_path(out, "evaluation.lock")
    with lock_path.open("a+") as mutex:
        try:
            fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another evaluation owns the run lock") from exc
        if receipt.exists():
            bank = _load_complete(out, receipt, binding)
            ledger = _read_json(ledger_path) if ledger_path.exists() else {}
            attempt = f"{name}/{split}"
            prior = ledger.get(attempt)
            if prior is not None and (
                prior.get("binding_digest") != _digest(binding)
                or prior.get("reserved_states") != len(rows) * 128
            ):
                raise ValueError("Completed receipt and attempt reservation disagree")
            if prior is None or prior.get("status") != "complete":
                ledger[attempt] = {
                    "binding_digest": _digest(binding),
                    "reserved_states": len(rows) * 128,
                    "status": "complete",
                    "reconciled_from_verified_receipt": True,
                    "previous_attempt": prior,
                }
                atomic_json(ledger, ledger_path)
            return bank
        ledger = _read_json(ledger_path) if ledger_path.exists() else {}
        attempt = f"{name}/{split}"
        if attempt in ledger or directory.exists() or shards.exists():
            raise ValueError("Uncommitted extraction is export-only; no duplicate forwards")
        ledger[attempt] = {
            "binding_digest": _digest(binding),
            "reserved_states": len(rows) * 128,
            "status": "started",
        }
        atomic_json(ledger, ledger_path)
        directory.mkdir(parents=True)
        shards.mkdir(parents=True)
        try:
            sample_path = _safe_path(out, f"results/{split}_{name}_predictions.npz")
            if sample_path.exists():
                raise ValueError("Uncommitted fixed-position output already exists")
            sample_path.parent.mkdir(exist_ok=True)
            bank = _extract_reserved(
                meta, out, prefix, shards, sample_path, encoder, query, x, raw_closes, rows, device
            )
            bank["data_digest"] = data_digest
            if (
                training.parameter_signature(encoder, query) != before
                or verify_lock(meta, out, name, split) != lock
            ):
                raise ValueError("Frozen models or source/model lock changed during extraction")
            files = {
                str(p.relative_to(out)): sha256(p)
                for folder in (directory, shards)
                for p in sorted(folder.iterdir())
                if p.is_file()
            }
            files[str(sample_path.relative_to(out))] = sha256(sample_path)
            omitted = {
                key: {
                    "sha256": value,
                    "reason": "full128 query predictions retained remotely; eight fixed positions exported",
                }
                for key, value in files.items()
                if key.startswith("predictions/")
            }
            atomic_json(
                {
                    "schema": SCHEMA,
                    "status": "complete",
                    "binding": binding,
                    "files": files,
                    "omitted_from_download": omitted,
                    "rows_digest": bank["rows_digest"],
                    "mask_digest": bank["mask_digest"],
                    "data_digest": bank["data_digest"],
                    "row_file": prefix + "/parent_rows.npz",
                    "position_file": prefix + "/position_rows.npz",
                    "states": len(rows) * 128,
                    "updates": 0,
                    "fits": 0,
                    "embeddings_saved": False,
                },
                receipt,
            )
            ledger[attempt]["status"] = "complete"
            atomic_json(ledger, ledger_path)
            bank["receipt"] = str(receipt.relative_to(out))
            return bank
        except BaseException as exc:
            ledger[attempt]["status"] = "failed_or_stopped"
            ledger[attempt]["error"] = type(exc).__name__ + ": " + str(exc)
            atomic_json(ledger, ledger_path)
            raise


def _extract_reserved(
    meta, out, prefix, shards, sample_path, encoder, query, x, raw_closes, rows, device
):
    encoder.eval()
    query.eval()
    n = len(rows)
    pred8 = np.empty((n, 8, 127, 7), np.float32)
    y8 = np.empty((n, 8, 127, 7), np.float64)
    m8 = np.empty((n, 8, 127, 7), bool)
    errors, support = {}, {}
    mask_digest = hashlib.sha256()
    sample_index = {p: j for j, p in enumerate(SAMPLE_POSITIONS)}
    for left in range(0, n, 8):
        right = min(left + 8, n)
        block = torch.as_tensor(x[left:right], device=device)
        closes = torch.as_tensor(raw_closes[left:right], device=device)
        allpred = np.empty((right - left, 128, 127, 7), np.float32)
        allmask = np.empty(allpred.shape, bool)
        for start in range(0, 128, 5):
            positions = np.arange(start + 1, min(start + 6, 129))
            ps = torch.as_tensor(positions, device=device).expand(len(block), -1)
            _, pred = coverage.predict(encoder, query, block, ps)
            target, mask = coverage.targets(block, ps, meta["statistics"])
            target = content.replace_price_targets(target, closes, ps, meta["statistics"])
            if pred.dtype != torch.float32 or not torch.isfinite(pred).all():
                raise ValueError("Finite float32 query predictions required")
            p, y, m = pred.cpu().numpy(), target.cpu().numpy(), mask.cpu().numpy()
            scored = score_rows(p, y, m, raw_closes[left:right], meta["statistics"], positions)
            stop = start + len(positions)
            allpred[:, start:stop], allmask[:, start:stop] = p, m
            for family, destination in (("errors", errors), ("support", support)):
                for key, value in scored[family].items():
                    if key not in destination:
                        destination[key] = np.empty((n, 128), dtype=value.dtype)
                    destination[key][left:right, start:stop] = value
            for j, position in enumerate(positions):
                if position in sample_index:
                    k = sample_index[position]
                    pred8[left:right, k], y8[left:right, k], m8[left:right, k] = (
                        p[:, j],
                        y[:, j],
                        m[:, j],
                    )
        mask_digest.update(allmask.tobytes())
        _save_npy(shards / f"parents_{left:05d}_{right:05d}.npy", allpred)
    bank = aggregate_rows(errors, support)
    shared = overlap_rows(pred8, raw_closes, meta["statistics"])
    bank["errors"].update(shared["errors"])
    bank["support"].update(shared["support"])
    bank["position_support"] = support
    bank["rows_digest"], bank["mask_digest"] = _digest(rows), mask_digest.hexdigest()
    directory = _safe_path(out, prefix)
    _save_npz(
        sample_path,
        {
            "predictions": pred8,
            "targets": y8,
            "masks": m8,
            "raw_closes": raw_closes,
            "positions": np.asarray(SAMPLE_POSITIONS),
        },
    )
    _save_npz(
        directory / "position_rows.npz",
        _pack({"errors": errors, "support": support}, ("errors", "support")),
    )
    _save_npz(directory / "parent_rows.npz", _pack(bank, ("errors", "support", "contexts")))
    atomic_json(rows, directory / "rows.json")
    atomic_json(summarize(bank), directory / "metrics.json")
    return bank
