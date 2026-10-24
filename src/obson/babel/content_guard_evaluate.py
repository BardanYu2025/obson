"""F01/F05/F11 fixed-state evaluation of soft content-risk feedback.

The eight-model lock includes two untouched sources and three trained arms per
seed. All objective arms use the same plain endpoint encoder and query.
Physical held-out scores remain independent of train-panel feasibility. No fitting
or model promotion occurs during evaluation.
"""

import fcntl
import hashlib
from pathlib import Path

import numpy as np
import torch

from . import content_guard_training as guard_training
from . import multiscale_content as content
from . import multiscale_evaluate as inherited
from . import observed_history_training as training
from . import state_coverage as coverage
from . import state_organization as model
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

SCHEMA = "babel-content-guard-evaluation-v1"
ARMS = ("initial", "control", "fixed", "feedback")
ATTENTION_MODES = dict.fromkeys(ARMS, "causal")
MODEL_ARMS = dict.fromkeys(ARMS, "plain")
RUNS = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in (42, 43))
WRITE_SOURCES = {"plain": []}


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


def _model_contract(name, encoder):
    arm = name.rsplit("_s", 1)[0]
    if (
        arm not in ATTENTION_MODES
        or getattr(encoder, "arm", None) != MODEL_ARMS[arm]
        or getattr(encoder, "attention_mode", None) != ATTENTION_MODES[arm]
    ):
        raise ValueError("Named arm and actual encoder attention mode differ")
    contract = model.model_contract(encoder)
    if (
        contract.get("arm") != MODEL_ARMS[arm]
        or contract.get("attention_mode") != ATTENTION_MODES[arm]
        or contract.get("public_output") != "organized_final_state_only"
        or contract.get("private_memory_exposed") is not False
        or contract.get("memory_write_sources") != WRITE_SOURCES[MODEL_ARMS[arm]]
    ):
        raise ValueError("Encoder endpoint model contract differs")
    return contract


def source_retention(candidate, source, rows):
    """Retain source content within10%, without demanding5% source improvement."""
    result = comparison(candidate, source, rows)
    checks = {k: v for k, v in result["checks"].items() if k != "gain"}
    key = "full/price/composite"
    a, b = candidate["errors"][key], source["errors"][key]
    ci = {k: values[0] for k, values in paired_intervals((a - 1.1 * b)[:, None], rows).items()}
    supported = all(v["supported"] for v in ci.values())
    passed = supported and all(v["high"] <= 0 for v in ci.values())
    ca, co = float(np.mean(a)), float(np.mean(b))
    checks["retain/" + key] = {
        "metric": key,
        "candidate_mean": ca,
        "control_mean": co,
        "ratio_of_means": ca / co if co > 0 else None,
        "factor": 1.1,
        "intervals": ci,
        "status": "pass" if passed else "fail" if supported else "insufficient",
        "passed": passed,
    }
    protected = all(v["passed"] for v in checks.values()) and result["tail"]["passed"]
    return {
        "checks": checks,
        "tail": result["tail"],
        "protection_passed": protected,
        "passed": protected,
        "bootstrap": result["bootstrap"],
        "scope": "Source noninferiority only; source is not an equal-budget training control",
    }


def training_evidence(out):
    """Read the exact final-panel and training-reference files locked by runner.

    This is file/numeric validation only, not a model forward. Model/optimizer
    checkpoints may be omitted from downloads; their identities stay in receipts.
    """
    out = _output_directory(out)
    lock_path = _safe_path(out, "model_selection_lock.json")
    lock = _read_json(lock_path)
    if (
        lock.get("schema") != "babel-content-guard768-v1"
        or lock.get("selection") != "fixed_last60"
        or set(lock.get("models", {})) != set(RUNS)
        or set(lock.get("references", {})) != {"42", "43"}
    ):
        raise ValueError("Complete content-guard source/final/reference lock required")
    reference_lock_path = _safe_path(out, "training_reference_lock.json")
    reference_lock = _read_json(reference_lock_path)
    expected_references = {
        f"training_reference_s{seed}.json": lock["references"][str(seed)] for seed in (42, 43)
    }
    if (
        sha256(reference_lock_path) != lock.get("reference_lock")
        or reference_lock.get("files") != expected_references
        or reference_lock.get("source_identity") != lock.get("source")
    ):
        raise ValueError("Pretraining reference lock differs from selected model lock")
    references, finals = {}, {}
    for seed in (42, 43):
        path = _safe_path(out, f"training_reference_s{seed}.json")
        if sha256(path) != lock["references"][str(seed)]:
            raise ValueError("Frozen training reference fingerprint differs")
        references[str(seed)] = _read_json(path)
    for name in RUNS:
        if name.startswith("initial_"):
            continue
        path = _safe_path(out, f"{name}/final_train_risk.json")
        if sha256(path) != lock["models"][name].get("files", {}).get("final_train_risk.json"):
            raise ValueError("Final training risk fingerprint differs from selected checkpoint")
        finals[name] = _read_json(path)
    return {
        "model_lock": lock,
        "model_lock_digest": _digest(sha256(lock_path)),
        "references": references,
        "finals": finals,
    }


def validate_train_evidence(evidence, banks, lock_digest):
    """Recompute final feasibility; never substitute the online epoch averages."""
    expected = {name for name in RUNS if not name.startswith("initial_")}
    if (
        not isinstance(evidence, dict)
        or evidence.get("model_lock_digest") != lock_digest
        or set(evidence.get("references", {})) != {"42", "43"}
        or set(evidence.get("finals", {})) != expected
    ):
        raise ValueError("All six final training panels must share the research model lock")
    lock = evidence.get("model_lock", {})
    if (
        lock.get("schema") != "babel-content-guard768-v1"
        or lock.get("selection") != "fixed_last60"
        or set(lock.get("models", {})) != set(RUNS)
        or set(lock.get("references", {})) != {"42", "43"}
    ):
        raise ValueError("Training evidence lacks complete model/reference lock")
    if any(
        banks[split][name].get("model_signature") != lock["models"][name].get("signature")
        for split in banks
        for name in RUNS
    ):
        raise ValueError("Training panel lock and all eight research signatures differ")
    results = {}
    for seed in (42, 43):
        reference = evidence["references"][str(seed)]
        parents = reference.get("parents")
        if type(parents) is not int or parents <= 0:
            raise ValueError("Positive training panel parent count required")
        guard_training.validate_reference(reference, seed, parents)
        if reference["source_signature"] != lock["models"][f"initial_s{seed}"].get("signature"):
            raise ValueError("Training reference is not calibrated at the locked source")
    for name, report in evidence["finals"].items():
        seed = name[-2:]
        reference = evidence["references"][seed]
        selected = lock["models"][name]
        if (
            report.get("checkpoint_sha256") != selected.get("files", {}).get("last.pt")
            or report.get("reference_sha256") != lock["references"][seed]
            or report.get("model_signature") != selected.get("signature")
            or report.get("binding") != selected.get("binding")
            or not selected.get("files", {}).get("last.pt")
            or any(
                banks[split][name].get("model_signature") != selected.get("signature")
                for split in banks
            )
            or any(
                report.get(k) != reference[k]
                for k in ("parents", "contexts", "order_hash", "prefix_hash", "support")
            )
        ):
            raise ValueError("Final training risk model/reference/panel identity differs")
        risks = np.asarray(report.get("risks"), dtype=np.float64)
        supplied = np.asarray(report.get("group_ratios"), dtype=np.float64)
        if (
            risks.shape != (21,)
            or supplied.shape != (21,)
            or not np.isfinite(risks).all()
            or (risks < 0).any()
        ):
            raise ValueError("Twenty-one nonnegative final physical risks required")
        ratios = risks / np.asarray(reference["baseline"], dtype=np.float64)
        if not np.isfinite(supplied).all() or not np.allclose(
            supplied, ratios, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Final risk ratios do not match the locked source baseline")
        exact = (ratios <= 1).tolist()
        if report.get("within_source_mean") != exact or report.get(
            "all_within_source_mean"
        ) is not all(exact):
            raise ValueError("Recorded final train-panel flags differ from physical risks")
        query = report.get("query")
        if (
            isinstance(query, bool)
            or not isinstance(query, (float, int))
            or not np.isfinite(query)
            or query < 0
        ):
            raise ValueError("Finite nonnegative final training Q required")
        passed = ratios <= 1 + 1e-6
        results[name] = {
            "ratios": ratios.tolist(),
            "checks_passed": passed.tolist(),
            "passed": bool(passed.all()),
            "model_signature": report["model_signature"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "reference_sha256": report["reference_sha256"],
        }
    return {
        "models": results,
        "feedback_both_seeds_passed": all(
            results[f"feedback_s{seed}"]["passed"] for seed in (42, 43)
        ),
        "factor": 1.0,
        "numeric_tolerance": 1e-6,
        "source": "frozen train-only calibration panel; fixed5 positions per parent at absolute epoch180",
        "scope": "Final frozen-model mean feasibility on this training panel only; no pointwise or held-out guarantee. Epoch controller means at changing weights are not used.",
    }


def decisions(banks, row_bank, train_evidence):
    """Feedback must beat BOTH matched objectives and retain source content.

    Feasibility is measured at fixed final weights on the frozen training panel,
    independently from the online controller measurements. Neither it nor a
    held-out score is a guarantee for every bar or a new independent dataset.
    """
    if set(banks) != {"test", "cross_research"} or set(row_bank) != set(banks):
        raise ValueError("Both locked research cohorts required")
    pairs = {
        "feedback_vs_fixed": ("feedback", "fixed"),
        "feedback_vs_control": ("feedback", "control"),
        "fixed_vs_control": ("fixed", "control"),
        "feedback_vs_initial": ("feedback", "initial"),
    }
    groups = {name: {} for name in pairs}
    locks = set()
    for split, models in banks.items():
        if set(models) != set(RUNS):
            raise ValueError("All eight models and both fixed seeds required")
        for name, bank in models.items():
            arm = name.rsplit("_s", 1)[0]
            contract = bank.get("model_contract", {})
            if (
                bank.get("model_name") != name
                or bank.get("objective_arm") != arm
                or bank.get("split") != split
                or contract.get("arm") != "plain"
                or contract.get("attention_mode") != "causal"
                or contract.get("public_output") != "organized_final_state_only"
                or contract.get("private_memory_exposed") is not False
                or contract.get("memory_write_sources") != []
                or not bank.get("model_lock_digest")
                or not bank.get("model_signature")
            ):
                raise ValueError("Comparison model identity or encoding contract differs")
            locks.add(bank["model_lock_digest"])
        for label, (candidate, reference) in pairs.items():
            compare = source_retention if reference == "initial" else comparison
            for seed in (42, 43):
                groups[label][f"{split}/s{seed}"] = compare(
                    models[f"{candidate}_s{seed}"], models[f"{reference}_s{seed}"], row_bank[split]
                )
    if len(locks) != 1:
        raise ValueError("Research cohorts must share one complete eight-model lock")
    feasibility = validate_train_evidence(train_evidence, banks, next(iter(locks)))
    results = {
        name: {
            "comparisons": values,
            "limited_hypothesis_supported": all(v["passed"] for v in values.values()),
            "conclusion": "required_content_conditions_passed"
            if all(v["passed"] for v in values.values())
            else "content_conditions_not_confirmed",
        }
        for name, values in groups.items()
    }
    required = ("feedback_vs_fixed", "feedback_vs_control", "feedback_vs_initial")
    research_passed = all(results[name]["limited_hypothesis_supported"] for name in required)
    supported = research_passed and feasibility["feedback_both_seeds_passed"]
    return {
        "schema": SCHEMA,
        "primary_comparison": "feedback_vs_fixed",
        "required_comparisons": list(required),
        "secondary_comparison": "fixed_vs_control",
        "comparison_groups": results,
        "train_panel_feasibility": feasibility,
        "research_conditions_passed": research_passed,
        "limited_hypothesis_supported": supported,
        "conclusion": "limited_content_feedback_recipe_gain"
        if supported
        else "no_accepted_content_feedback_gain",
        "promoted": False,
        "fit_count": 0,
        "metric_contract": metric_contract(),
        "f11": {
            "scope": "Content retention and measured training/readout cost only",
            "new_readout_fits": 0,
            "general_reuse": "unknown_not_tested",
        },
        "scope": "Only final768 of independent closed128 endpoint windows. Feedback compared with fixed normalized physical risks isolates controller recipe; both are compared with originalQ. Soft train-mean constraints do not guarantee per-example or research protection. Reused research cohorts; no generic reuse, forecasting or Macro promotion.",
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
    bank["objective_arm"] = binding["objective_arm"]
    bank["split"] = binding["split"]
    bank["model_lock_digest"] = _digest(binding["lock"]["model_lock"])
    return bank


@torch.no_grad()
def extract(meta, out, name, encoder, query, x, raw_closes, rows, split, device, *, verify_lock):
    """One bounded extraction per model/cohort; complete receipts are resumable.

    ``verify_lock(meta,out,name,split)`` must certify the COMPLETE eight-model
    initial180 plus fixed final60-continuation matrix, source/data lineage and return a JSON binding with
    ``model_lock``, ``source`` and ``raw_receipt``. It is called before any model forward and
    again before committing. The root runner owns deadlines and failure export.
    """
    if not callable(verify_lock):
        raise ValueError("Research extraction requires the runner lock verifier")
    if name not in {f"{a}_s{s}" for a in ARMS for s in (42, 43)} or split not in (
        "test",
        "cross_research",
    ):
        raise ValueError("Fixed eight-model/two-cohort extraction required")
    out = _output_directory(out)
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
    if meta.get("micro") != 8 or meta.get("batch") != 128 or meta.get("epochs") != 60:
        raise ValueError("Fixed micro8/batch128/60-additional-epoch extraction required")
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
        "objective_arm": name.rsplit("_s", 1)[0],
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
            if key not in ("name", "objective_arm", "model_signature", "model_contract")
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
            bank["model_contract"] = contract
            bank["model_name"] = name
            bank["model_signature"] = before
            bank["objective_arm"] = name.rsplit("_s", 1)[0]
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
            _, pred = model.predict(encoder, query, block, ps)
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
