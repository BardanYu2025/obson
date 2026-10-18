"""F05/F01/F11 final-state evaluation of bounded layer supervision.

The six-model lock includes two untouched initial states and two continuation
arms per seed. Physical scores and decision gates are unchanged; the layer2
training auxiliary has no evaluation role. No fitting or model promotion.
"""

import fcntl
import hashlib
from pathlib import Path

import numpy as np
import torch

from . import layer_supervision as model
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

SCHEMA = "babel-layer-supervision-evaluation-v1"
ARMS = ("initial", "control", "layer2")
ATTENTION_MODES = dict.fromkeys(ARMS, "causal")
MODEL_ARMS = {"initial": "control", "control": "control", "layer2": "layer2"}
RUNS = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in (42, 43))


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
    ):
        raise ValueError("Encoder endpoint model contract differs")
    return contract


def decisions(banks, row_bank):
    """Separate the matched mechanism contrast from the longitudinal budget contrast."""
    if set(banks) != {"test", "cross_research"} or set(row_bank) != set(banks):
        raise ValueError("Both locked research cohorts required")
    variants = set(RUNS)
    groups = {"layer2_vs_control": {}, "control_vs_initial": {}}
    locks = set()
    for split, models in banks.items():
        if set(models) != variants:
            raise ValueError(
                "All six models including initial states and both fixed seeds required"
            )
        for name, bank in models.items():
            arm = name.rsplit("_s", 1)[0]
            contract = bank.get("model_contract", {})
            if (
                bank.get("model_name") != name
                or bank.get("split") != split
                or contract.get("arm") != MODEL_ARMS[arm]
                or contract.get("attention_mode") != "causal"
                or not bank.get("model_lock_digest")
            ):
                raise ValueError("Comparison model identity or attention contract differs")
            locks.add(bank["model_lock_digest"])
        for seed in (42, 43):
            key = f"{split}/s{seed}"
            groups["layer2_vs_control"][key] = comparison(
                models[f"layer2_s{seed}"], models[f"control_s{seed}"], row_bank[split]
            )
            groups["control_vs_initial"][key] = comparison(
                models[f"control_s{seed}"], models[f"initial_s{seed}"], row_bank[split]
            )
    if len(locks) != 1:
        raise ValueError("Research cohorts must share one complete six-model lock")
    group_results = {
        name: {
            "comparisons": comparisons,
            "limited_hypothesis_supported": all(v["passed"] for v in comparisons.values()),
            "conclusion": "limited_balanced_price_content_gain"
            if all(v["passed"] for v in comparisons.values())
            else "no_accepted_balanced_gain",
        }
        for name, comparisons in groups.items()
    }
    return {
        "schema": SCHEMA,
        "primary_comparison": "layer2_vs_control",
        "comparison_groups": group_results,
        "limited_hypothesis_supported": group_results["layer2_vs_control"][
            "limited_hypothesis_supported"
        ],
        "budget_hypothesis_supported": group_results["control_vs_initial"][
            "limited_hypothesis_supported"
        ],
        "conclusion": group_results["layer2_vs_control"]["conclusion"],
        "promoted": False,
        "fit_count": 0,
        "metric_contract": metric_contract(),
        "f11": {
            "scope": "Content retention and measured training/readout cost only",
            "new_readout_fits": 0,
            "general_reuse": "unknown_not_tested",
        },
        "scope": "Final768 independently encoded closed128 endpoint states only. Auxiliary layer predictions are excluded. Matched mechanism and longitudinal optimization-budget contrasts are separate; neither substitutes for the other. Reused research cohorts, no universal embedding or forecasting acceptance; Macro unchanged",
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
    bank["split"] = binding["split"]
    bank["model_lock_digest"] = _digest(binding["lock"]["model_lock"])
    return bank


@torch.no_grad()
def extract(meta, out, name, encoder, query, x, raw_closes, rows, split, device, *, verify_lock):
    """One bounded extraction per model/cohort; complete receipts are resumable.

    ``verify_lock(meta,out,name,split)`` must certify the COMPLETE six-model
    initial120 plus fixed final60-continuation matrix, source/data lineage and return a JSON binding with
    ``model_lock``, ``source`` and ``raw_receipt``. It is called before any model forward and
    again before committing. The root runner owns deadlines and failure export.
    """
    if not callable(verify_lock):
        raise ValueError("Research extraction requires the runner lock verifier")
    if name not in {f"{a}_s{s}" for a in ARMS for s in (42, 43)} or split not in (
        "test",
        "cross_research",
    ):
        raise ValueError("Fixed six-model/two-cohort extraction required")
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
