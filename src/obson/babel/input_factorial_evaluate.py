"""Locked structure-by-initialization evaluation, with a historical bar anchor.

Eight new models identify conditional structure/init effects. Two historical bar
references retain the old price protection baseline but are not concurrent
factorial controls. Every public state has its own observed trailing128 context.
"""

import fcntl
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from . import input_factorial as model
from . import input_patch as historical_model
from . import multiscale_content as content
from . import multiscale_evaluate as inherited
from . import observed_history_training as training
from . import state_coverage as coverage
from .ae_extend import atomic_json
from .dual_state import sha256
from .multiscale_evaluate import (
    BOOTSTRAP_SEED as BOOTSTRAP_SEED,
)
from .multiscale_evaluate import (
    DRAWS as DRAWS,
)
from .multiscale_evaluate import (
    FIELDS as FIELDS,
)
from .multiscale_evaluate import (
    OVERLAP_GUARDS as OVERLAP_GUARDS,
)
from .multiscale_evaluate import (
    POSITIONS as POSITIONS,
)
from .multiscale_evaluate import (
    SAMPLE_POSITIONS as SAMPLE_POSITIONS,
)
from .multiscale_evaluate import (
    _digest,
    _pack,
    _read_json,
    _row_labels,
    _safe_path,
    _save_npy,
    _save_npz,
    _unpack,
)
from .multiscale_evaluate import (
    aggregate_rows as aggregate_rows,
)
from .multiscale_evaluate import (
    comparison as comparison,
)
from .multiscale_evaluate import (
    overlap_rows as overlap_rows,
)
from .multiscale_evaluate import (
    paired_intervals as paired_intervals,
)
from .multiscale_evaluate import (
    protection_keys as protection_keys,
)
from .multiscale_evaluate import (
    score_rows as score_rows,
)
from .progress import progress

SCHEMA = "babel-input-factorial-evaluation-v1"
ARMS = ("dense_random", "hybrid_random", "dense_endpoint", "hybrid_endpoint")
TRAIN_RUNS = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in (42, 43))
REFERENCE_RUNS = tuple(f"reference_bar_s{seed}" for seed in (42, 43))
RUNS = TRAIN_RUNS + REFERENCE_RUNS
PAIRS = {
    "structure_random": ("hybrid_random", "dense_random"),
    "structure_endpoint": ("hybrid_endpoint", "dense_endpoint"),
    "initialization_dense": ("dense_endpoint", "dense_random"),
    "initialization_hybrid": ("hybrid_endpoint", "hybrid_random"),
    "historical_bar_anchor": ("hybrid_endpoint", "reference_bar"),
}
COHORTS = ("test", "cross_research")


def metric_contract():
    return {
        "source_module": "obson.babel.multiscale_evaluate",
        "source_sha256": sha256(inherited.__file__),
        "reuse": "Unchanged physical price, six nonprice fields, exact support, F03, dual2000 bootstrap and five-percent gain/ten-percent protection/P95 gates",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "haar_fit": False,
        "scope": "Same observed physical history, not forecasting or universal representation acceptance",
    }


def _output_directory(out):
    path = Path(out).expanduser().absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Evaluation output must not traverse parents or symlinks")
    root = Path(__file__).resolve().parents[3]
    if (
        root == path
        or root.is_relative_to(path)
        or any(path.is_relative_to(root / name) for name in ("src", "tests", "docs", "scripts"))
    ):
        raise ValueError("Evaluation cannot write into source code or its ancestors")
    return path.resolve()


def validate_contract(contract, arm):
    """Validate exact semantics, including distinct historical reference scope."""
    if not isinstance(contract, dict) or arm not in (*ARMS, "reference_bar"):
        raise ValueError("Declared factorial or historical bar contract required")
    expected = (
        historical_model.contract("bar", contract.get("encoder"))
        if arm == "reference_bar"
        else model.contract(arm, contract.get("encoder"))
    )
    if contract != expected:
        raise ValueError("Input grouping, initialization, position or state contract differs")
    return contract


def _model_contract(name, encoder):
    arm = name.rsplit("_s", 1)[0]
    module = historical_model if arm == "reference_bar" else model
    return validate_contract(module.model_contract(encoder), arm)


def interaction(models, rows, seed):
    """Paired four-model difference-in-differences; no checkpoint selection.

    Columns are formed per original parent BEFORE the same cluster resampling.
    Endpoints within a parent are never treated as independent observations.
    Negative means the hybrid-minus-dense difference is lower under endpoint
    initialization. This is a diagnostic, not a replacement acceptance gate.
    """
    keys = tuple(f"full/price/{b}" for b in ("composite", "near", "mid", "far"))
    columns = []
    for key in keys:
        values = {
            arm: np.asarray(models[f"{arm}_s{seed}"]["errors"][key], np.float64) for arm in ARMS
        }
        if any(v.shape != (len(rows),) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Four finite matched parent physical-price errors required")
        columns.append(
            (values["hybrid_endpoint"] - values["dense_endpoint"])
            - (values["hybrid_random"] - values["dense_random"])
        )
    intervals = paired_intervals(np.column_stack(columns), rows)
    metrics = {}
    for j, key in enumerate(keys):
        ci = {kind: values[j] for kind, values in intervals.items()}
        supported = all(v["supported"] for v in ci.values())
        direction = "insufficient" if not supported else "unconfirmed"
        if supported and all(v["high"] < 0 for v in ci.values()):
            direction = "hybrid_relative_benefit_larger_with_endpoint_initialization"
        elif supported and all(v["low"] > 0 for v in ci.values()):
            direction = "hybrid_relative_benefit_larger_with_random_initialization"
        metrics[key] = {"mean": float(columns[j].mean()), "intervals": ci, "direction": direction}
    return {
        "formula": "(hybrid_endpoint-dense_endpoint)-(hybrid_random-dense_random)",
        "metrics": metrics,
        "unit": "physical_price_percent_squared",
        "bootstrap": {"draws": DRAWS, "seed": BOOTSTRAP_SEED, "paired_parent_values": True},
        "acceptance_gate": False,
        "scope": "Conditional interaction diagnostic; no independent-CI subtraction, no multiple-comparison selection, no complete training-adaptation attribution",
    }


def decisions(banks, row_bank):
    """Require the entire prespecified matrix; never select an available subset."""
    if set(banks) != set(COHORTS) or set(row_bank) != set(COHORTS):
        raise ValueError("Both locked research cohorts required")
    groups, interactions = {key: {} for key in PAIRS}, {}
    locks, signatures, configs = set(), {}, set()
    for split, models in banks.items():
        if set(models) != set(RUNS):
            raise ValueError(
                "All eight factorial models and two historical bar references required"
            )
        for name, bank in models.items():
            arm = name.rsplit("_s", 1)[0]
            validate_contract(bank.get("model_contract"), arm)
            configs.add(_digest(bank["model_contract"]["encoder"]))
            if (
                bank.get("model_name") != name
                or bank.get("split") != split
                or not bank.get("model_lock_digest")
                or not bank.get("model_signature")
            ):
                raise ValueError("Comparison model identity or signature differs")
            locks.add(bank["model_lock_digest"])
            if name in signatures and signatures[name] != bank["model_signature"]:
                raise ValueError("Same model must have the same signature in both cohorts")
            signatures[name] = bank["model_signature"]
        for label, (candidate, reference) in PAIRS.items():
            for seed in (42, 43):
                groups[label][f"{split}/s{seed}"] = comparison(
                    models[f"{candidate}_s{seed}"], models[f"{reference}_s{seed}"], row_bank[split]
                )
        for seed in (42, 43):
            interactions[f"{split}/s{seed}"] = interaction(models, row_bank[split], seed)
    if len(locks) != 1 or len(configs) != 1:
        raise ValueError("Cohorts must share one complete ten-model lock and backbone config")
    results = {
        name: {
            "comparisons": values,
            "limited_hypothesis_supported": all(v["passed"] for v in values.values()),
            "role": "historical_frozen_anchor_not_concurrent_control"
            if name == "historical_bar_anchor"
            else "factorial_matched_control",
        }
        for name, values in groups.items()
    }
    structure = {
        k: results[f"structure_{k}"]["limited_hypothesis_supported"] for k in ("random", "endpoint")
    }
    init = {
        k: results[f"initialization_{k}"]["limited_hypothesis_supported"]
        for k in ("dense", "hybrid")
    }
    if all(structure.values()):
        conclusion = "conditional_structure_gain_supported_under_both_initializations"
    elif any(structure.values()):
        conclusion = "initialization_dependent_conditional_structure_gain"
    elif any(init.values()):
        conclusion = "initialization_gain_without_accepted_structure_gain"
    else:
        conclusion = "fixed_32_4_120_recipe_closed_without_balanced_gain"
    return {
        "schema": SCHEMA,
        "primary_candidate": "hybrid_endpoint",
        "structure_comparisons": ["structure_random", "structure_endpoint"],
        "initialization_comparisons": ["initialization_dense", "initialization_hybrid"],
        "comparison_groups": results,
        "interaction": interactions,
        "limited_hypothesis_supported": all(structure.values()),
        "structure_supported_under_both_initializations": all(structure.values()),
        "historical_anchor_passed": results["historical_bar_anchor"][
            "limited_hypothesis_supported"
        ],
        "conclusion": conclusion,
        "promoted": False,
        "fit_count": 0,
        "metric_contract": metric_contract(),
        "scope": "Eight scratch fixed-last120 models identify old96 compression and context-projection initialization conditional on native recent32. Historical bar120 is a frozen anchor, not a concurrent fifth arm. This does not isolate the benefit of native recent32, all optimizer/decoder adaptation, forecasting or general embedding quality; Macro is not promoted.",
        "stopping": "No partial-matrix decisions, seed selection, extra epochs or near-length grid. Inspect physical guards even when original Q decreases.",
        "f11": {
            "new_readout_fits": 0,
            "general_reuse": "unknown_not_tested",
            "scope": "Content retention and measured cost only",
        },
    }


def summarize(bank):
    return inherited.summarize(bank) | {"schema": SCHEMA, "metric_contract": metric_contract()}


def _load_complete(out, receipt, binding):
    saved = _read_json(receipt)
    if (
        saved.get("schema") != SCHEMA
        or saved.get("binding") != binding
        or saved.get("status") != "complete"
    ):
        raise ValueError("Completed evaluation binding changed")
    prefix = f"evaluation/{binding['name']}/{binding['split']}"
    if (
        saved.get("states") != binding["parents"] * 128
        or saved.get("updates") != 0
        or saved.get("fits") != 0
        or saved.get("embeddings_saved") is not False
        or saved.get("row_file") != prefix + "/parent_rows.npz"
        or saved.get("position_file") != prefix + "/position_rows.npz"
        or saved.get("rows_digest") != binding["rows_digest"]
    ):
        raise ValueError("Completed evaluation counts or evidence paths changed")
    required = {
        prefix + "/" + filename
        for filename in ("parent_rows.npz", "position_rows.npz", "rows.json", "metrics.json")
    }
    required.add(f"results/{binding['split']}_{binding['name']}_predictions.npz")
    required.update(
        f"predictions/{binding['name']}/{binding['split']}/parents_{start:05d}_{min(start + 8, binding['parents']):05d}.npy"
        for start in range(0, binding["parents"], 8)
    )
    if set(saved.get("files", {})) != required:
        raise ValueError("Completed evaluation file inventory changed")
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
    bank["model_contract"] = binding["model_contract"]
    bank["model_name"] = binding["name"]
    bank["model_signature"] = binding["model_signature"]
    bank["split"] = binding["split"]
    bank["model_lock_digest"] = _digest(binding["lock"]["model_lock"])
    return bank


@torch.no_grad()
def extract(meta, out, name, encoder, query, x, raw_closes, rows, split, device, *, verify_lock):
    """One bounded extraction per model/cohort; complete receipts are resumable.

    ``verify_lock(meta,out,name,split)`` must certify the COMPLETE eight-model scratch final120 matrix plus two historical references, source/data lineage and return a JSON binding with
    ``model_lock``, ``source`` and ``raw_receipt``. It is called before any model forward and
    again before committing. The root runner owns deadlines and failure export.
    """
    if not callable(verify_lock):
        raise ValueError("Research extraction requires the runner lock verifier")
    if name not in RUNS or split not in (
        "test",
        "cross_research",
    ):
        raise ValueError("Fixed ten-model/two-cohort extraction required")
    out = _output_directory(out)
    if not out.is_dir():
        raise ValueError("Runner must create a safe output directory first")
    for key in ("source", "raw_root", "model_source"):
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
    lock = verify_lock(meta, out, name, split)
    if not isinstance(lock, dict) or any(
        not lock.get(k) for k in ("model_lock", "source", "raw_receipt")
    ):
        raise ValueError("Verifier must bind the complete model matrix, source and raw receipt")
    contract = _model_contract(name, encoder)
    before = training.parameter_signature(encoder, query)
    binding = {
        "lock": lock,
        "name": name,
        "split": split,
        "parents": len(rows),
        "model_signature": before,
        "rows_digest": _digest(rows),
        "x_sha256": training.array_hash(x),
        "raw_sha256": training.array_hash(raw_closes),
        "statistics_digest": _digest(meta["statistics"]),
        "model_contract": contract,
        "metric_contract": metric_contract(),
        "positions": list(POSITIONS),
        "sample_positions": list(SAMPLE_POSITIONS),
    }
    data_digest = _digest(
        {
            key: value
            for key, value in binding.items()
            if key not in ("name", "model_signature", "model_contract")
        }
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
            progress(f"{name}/{split}: verified completed all128 evidence; zero repeated forwards")
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
            started = time.monotonic()
            progress(
                f"{name}/{split}: extracting {len(rows):,} parents ×128 own-context states, zero fitting"
            )
            sample_path = _safe_path(out, f"results/{split}_{name}_predictions.npz")
            if sample_path.exists():
                raise ValueError("Uncommitted fixed-position output already exists")
            sample_path.parent.mkdir(exist_ok=True)
            bank = _extract_reserved(
                meta,
                out,
                prefix,
                shards,
                sample_path,
                encoder,
                query,
                x,
                raw_closes,
                rows,
                device,
                predict_fn=historical_model.predict if name in REFERENCE_RUNS else model.predict,
            )
            bank["data_digest"] = data_digest
            bank["model_contract"] = contract
            bank["model_name"] = name
            bank["model_signature"] = before
            bank["split"] = split
            bank["model_lock_digest"] = _digest(lock["model_lock"])
            if (
                _model_contract(name, encoder) != contract
                or training.parameter_signature(encoder, query) != before
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
            progress(
                f"{name}/{split}: complete; {len(rows) * 128:,} states, elapsed={time.monotonic() - started:.1f}s"
            )
            return bank
        except BaseException as exc:
            ledger[attempt]["status"] = "failed_or_stopped"
            ledger[attempt]["error"] = type(exc).__name__ + ": " + str(exc)
            atomic_json(ledger, ledger_path)
            raise


def _extract_reserved(
    meta,
    out,
    prefix,
    shards,
    sample_path,
    encoder,
    query,
    x,
    raw_closes,
    rows,
    device,
    *,
    predict_fn,
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
    started = last_progress = time.monotonic()
    for left in range(0, n, 8):
        right = min(left + 8, n)
        block = torch.as_tensor(x[left:right], device=device)
        closes = torch.as_tensor(raw_closes[left:right], device=device)
        allpred = np.empty((right - left, 128, 127, 7), np.float32)
        allmask = np.empty(allpred.shape, bool)
        for start in range(0, 128, 5):
            positions = np.arange(start + 1, min(start + 6, 129))
            ps = torch.as_tensor(positions, device=device).expand(len(block), -1)
            _, pred = predict_fn(encoder, query, block, ps)
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
        now = time.monotonic()
        if now - last_progress >= 30 or right == n:
            progress(
                f"{prefix}: {right:,}/{n:,} parents, {right * 128:,} states; elapsed={now - started:.1f}s"
            )
            last_progress = now
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
