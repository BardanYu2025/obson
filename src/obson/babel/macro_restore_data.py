"""Recover original Macro root inputs/PCA, without any learned model execution.

Original labels/masks AND normalized inputs must match retained NPY hashes before
CUDA training is allowed. Refit PCA has a new identity; no old basis is invented.
"""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from . import architecture as ar
from . import architecture_benchmark as ab
from . import coverage_audit as cov
from . import data, pca_teacher, prefix_readout, state_holdout, state_rollout_data
from .ae_extend import atomic_json
from .progress import progress
from .raw_snapshot import digest, no_links

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "artifacts/babel_macro_recovery_20261115"
SNAPSHOT = REPO / "artifacts/babel_fresh_20261115"
SCHEMA = "babel-macro-root-data-v1"
SPLITS = ("train", "val", "test", "cross_research")
ASOF = "2026-09-12 03:00:00"
COUNTS = {"train": 4789, "val": 978, "test": 984, "cross_research": 453}


def read(path):
    return json.loads(Path(path).read_text())


def schema_node(value, schema):
    if isinstance(value, dict):
        if value.get("schema") == schema:
            return value
        for child in value.values():
            found = schema_node(child, schema)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = schema_node(child, schema)
            if found is not None:
                return found
    return None


def source_contract():
    evidence = read(ASSETS / "manifest.json")
    for name, item in evidence["files"].items():
        expected = item if isinstance(item, str) else item["sha256"]
        if digest(ASSETS / name) != expected:
            raise ValueError(f"Recovered source evidence changed: {name}")
    original = read(ASSETS / "retained_source_manifest.json")
    alignment = schema_node(original, "babel-bar-alignment-v1")
    sampling = schema_node(original, "babel-window-sampling-v1")
    return {
        "retained_manifest_sha256": digest(ASSETS / "retained_source_manifest.json"),
        "expected_cache": sampling["source_identity"]["cache_files"],
        "bounds": sampling["coverage_identity"]["boundaries"],
        "config": dict(alignment["config"], latent=768),
        "local_hidden": alignment["local_hidden"],
        "dense_count": sampling["coverage_identity"]["candidate"]["coverage"]["windows"],
        "counts": COUNTS,
    }


def raw_identity(root):
    root = no_links(root)
    expected = {
        name.removeprefix("contracts/"): info
        for name, info in read(SNAPSHOT / "manifest.json")["files"].items()
        if name.startswith("contracts/")
    }
    actual = {str(p.relative_to(root)) for p in root.glob("*/*m.csv")}
    if set(expected) != actual or len(expected) != 1251:
        raise ValueError("Original1251 single-contract inventory differs")
    for i, (name, info) in enumerate(sorted(expected.items()), 1):
        path = no_links(root / name)
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Raw file fingerprint mismatch: {name}")
        if i % 250 == 0:
            progress(f"Original raw hashes: {i}/1251")
    return hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()


def original_rows(series, bounds, split):
    known = read(ASSETS / f"{split}_rows.json")
    if split != "cross_research":
        if [r["original_index"] for r in known] != list(range(COUNTS[split])):
            raise ValueError("Original training/validation/test rows were filtered")
        return [
            {k: v for k, v in r.items() if k not in ("original_index", "input_start")}
            for r in known
        ]
    # Full128 original held-symbol cohort has453 rows; full255 later retained439.
    keys = state_holdout.endpoint_keys(series, bounds, cap=100000000)
    rows = []
    for i, end in keys:
        s = series[i]
        stamp = s.frame.datetime.iloc[end]
        rows.append(
            {
                "key": s.key,
                "symbol": s.code,
                "period": s.period,
                "row": int(end),
                "end": str(stamp),
                "month": str(stamp.to_period("M")),
                "week": str(pd.Timestamp(s.sessions[end]).to_period("W-SUN")),
            }
        )
    if len(rows) != COUNTS[split] or any(
        any(rows[r["original_index"]][k] != r[k] for k in rows[0]) for r in known
    ):
        raise ValueError("Full453 cross inventory does not preserve known439 original identities")
    return rows


def windows(rows, lookup, encoded, bounds, split):
    blocks = []
    for row in rows:
        s, end = lookup[row["key"]], row["row"]
        if end < 127 or str(s.frame.datetime.iloc[end]) != row["end"]:
            raise ValueError("Raw window identity changed")
        same = cov.rp.time_mask(s, bounds, "test" if split == "cross_research" else split)
        if not same[end - 127 : end + 1].all() or not s.main[end]:
            raise ValueError("Window crosses original contract/partition or selection")
        blocks.append(encoded[s.key][end - 127 : end + 1])
    return np.asarray(blocks, dtype=np.float32)


def write_array(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.save(f, value, allow_pickle=False)
    temp.replace(path)
    return digest(path)


def cache_check(expected, out, split, arrays):
    result = {}
    for key in ("x", "y", "mask"):
        name = f"{split}_{key}.npy"
        value = arrays[key]
        checksum = write_array(out / "cache" / name, value)
        result[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": checksum,
            "historical_sha256": expected[name],
            "exact": checksum == expected[name],
        }
    return result


def fit_assets(train, stats, out):
    progress("CPU PCA: original4789 training targets, full SVD896 then fixed prefix768")
    full = ab.fit_pca(train, 896)
    pca = {k: (v if k == "mean" else v[:768]) for k, v in full.items()}
    temp = out / "cache/pca768.tmp.npz"
    np.savez(temp, **pca)
    temp.replace(out / "cache/pca768.npz")
    coordinates = pca_teacher.fit_coordinate_scales(pca_teacher.coefficients(pca, train))
    local = prefix_readout.fit_target_scales(
        [prefix_readout.targets(train, stats, p) for p in prefix_readout.PREFIXES]
    )
    atomic_json(coordinates, out / "cache/coordinates768.json")
    atomic_json(local, out / "cache/local_scales.json")
    basis = pca["components"].astype(np.float64)
    flat = train["y"].reshape(len(train["y"]), -1).astype(np.float64)
    centered = flat - pca["mean"]
    coefficients = centered @ basis.T
    residual = centered - coefficients @ basis
    orthogonal = float(np.abs(basis @ basis.T - np.eye(768)).max())
    mse = float(np.square(residual).mean())
    spectral = float((full["singular"][768:].astype(float) ** 2).sum() / flat.size)
    # Float32 basis storage adds rounding; this is a numerical algebra gate, not science.
    if orthogonal > 1e-5 or abs(mse - spectral) > 1e-8 * max(
        1.0, float(np.square(centered).mean())
    ):
        raise ValueError("PCA orthogonality/reconstruction spectrum check failed")
    return {
        "fit_split": "train",
        "rows": len(flat),
        "full_width": 896,
        "retained_width": 768,
        "orthogonality_max_abs": orthogonal,
        "mse": mse,
        "svd_tail_mse": spectral,
        "historical_basis_recovered": False,
        "identity": "rederived_same_train_only_SVD_recipe_new_basis_fingerprint",
        "no_research_selection": True,
    }


def arrays(out, split):
    folder = "candidates" if split == "pool" else "cache"
    stem = "train" if split == "pool" else split
    return {
        k: np.load(out / folder / f"{stem}_{k}.npy", mmap_mode="r", allow_pickle=False)
        for k in ("x", "y", "mask")
    }


def verify_prepared(out, *, require_inputs=True):
    done = read(out / "data_receipt.json")
    for name, checksum in done["files"].items():
        if digest(no_links(out / name)) != checksum:
            raise ValueError(f"Prepared asset changed: {name}")
    if require_inputs and not done["training_allowed"]:
        raise ValueError(
            "Original input hashes not reproduced; zero neural updates allowed; see data_audit.json"
        )
    return done


def prepare(raw, out, binding):
    raw, out = no_links(raw), no_links(out)
    contract = source_contract()
    raw_hash = raw_identity(raw)
    metadata = {
        "schema": SCHEMA,
        "source": contract,
        "raw_fingerprint": raw_hash,
        "binding": binding,
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() and read(out / "manifest.json") != metadata:
        raise ValueError("Preparation source/code changed; use a new recovery run")
    atomic_json(metadata, out / "manifest.json")
    if (out / "data_receipt.json").exists():
        return verify_prepared(out, require_inputs=False)
    for folder in ("cache", "candidates"):
        (out / folder).mkdir(exist_ok=True)
    stats = read(ASSETS / "statistics.json")
    atomic_json(stats, out / "cache/statistics.json")
    audit = {
        "schema": SCHEMA,
        "source": contract,
        "splits": {},
        "pca": None,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "full255_note": "Not the original Macro input. Original cross453 retained;439 belongs to later full255 eligibility.",
        "neural_updates": 0,
    }
    for cross, splits in ((False, SPLITS[:3]), (True, SPLITS[3:])):
        symbols = (
            ["RM", "au", "c", "cs", "ru"]
            if cross
            else ["MA", "TA", "ag", "cu", "hc", "i", "j", "jm", "m", "p", "rb", "sr", "y"]
        )
        series, _ = data.load_series(raw, symbols, (15, 30, 60), asof=ASOF)
        lookup = {s.key: s for s in series}
        encoded = {}
        for i, s in enumerate(series, 1):
            encoded[s.key] = state_rollout_data.encode(s.frame, s.period)
            if i % 150 == 0:
                progress(f"Original features {i}/{len(series)}; CPU only")
        for split in splits:
            rows = original_rows(series, contract["bounds"], split)
            x = windows(rows, lookup, encoded, contract["bounds"], split)
            y, mask = ar.ordered_targets(x)
            if split == "train":
                refit = ar.fit_scales(x, y, mask)
                differences = {
                    k: float(np.abs(np.asarray(v) - stats[k]).max())
                    for k, v in refit.items()
                    if isinstance(v, list)
                }
                audit["statistics_refit_max_abs"] = differences
                if max(differences.values()) > 1e-8:
                    raise ValueError("Original training statistics do not reproduce")
            xx, yy, mask = ar.normalize(x, y, mask, stats)
            audit["splits"][split] = cache_check(
                contract["expected_cache"], out, split, {"x": xx, "y": yy, "mask": mask}
            )
            atomic_json(rows, out / "cache" / f"{split}_inventory.json")
            # Prefix truncation checks actual feature-generation causality, no neural execution.
            checked = set()
            for row in rows:
                if row["key"] in checked:
                    continue
                s, end = lookup[row["key"]], row["row"]
                prefix = state_rollout_data.encode(s.frame.iloc[: end + 1], s.period)
                if not np.array_equal(prefix, encoded[s.key][: end + 1]):
                    raise ValueError("Feature generation read future rows")
                checked.add(s.key)
            audit["splits"][split]["causal_prefix_contracts"] = len(checked)
            atomic_json(audit, out / "data_audit.json")
            progress(
                f"{split}: {len(rows)} original128 rows; hashes="
                + str({k: audit["splits"][split][k]["exact"] for k in ("x", "y", "mask")})
            )
        if not cross:
            candidates = [
                (s.key, int(end))
                for s in series
                for end in cov.endpoints(s, contract["bounds"], "train", stride=16, minimum=512)
            ]
            if len(candidates) != contract["dense_count"]:
                raise ValueError("Original dense candidate pool count changed")
            inventory = []
            banks = {}
            for start in range(0, len(candidates), 256):
                block = candidates[start : start + 256]
                x = np.stack([encoded[key][end - 127 : end + 1] for key, end in block])
                y, mask = ar.ordered_targets(x)
                xx, yy, mask = ar.normalize(x, y, mask, stats)
                for key, value in {"x": xx, "y": yy, "mask": mask}.items():
                    if key not in banks:
                        banks[key] = np.lib.format.open_memmap(
                            out / "candidates" / f"train_{key}.npy",
                            mode="w+",
                            dtype=value.dtype,
                            shape=(len(candidates), *value.shape[1:]),
                        )
                    banks[key][start : start + len(block)] = value
                for key, end in block:
                    s = lookup[key]
                    stamp = s.frame.datetime.iloc[end]
                    inventory.append(
                        {
                            "key": key,
                            "symbol": s.code,
                            "period": s.period,
                            "row": end,
                            "end": str(stamp),
                            "month": str(stamp.to_period("M")),
                        }
                    )
            for bank in banks.values():
                bank.flush()
            del banks
            atomic_json(inventory, out / "candidates/inventory.json")
            audit["dense_candidates"] = len(inventory)
            pool = arrays(out, "pool")
            original = arrays(out, "train")
            index = {(r["key"], r["row"]): i for i, r in enumerate(inventory)}
            train_rows = read(out / "cache/train_inventory.json")
            if len(index) != len(inventory):
                raise ValueError("Duplicate dense candidate identity")
            for i, row in enumerate(train_rows):
                j = index.get((row["key"], row["row"]))
                if j is None or any(not np.array_equal(pool[k][j], original[k][i]) for k in pool):
                    raise ValueError("Dense pool lost or changed an original training window")
            audit["dense_original_rows_replayed"] = len(train_rows)
            del pool, original
        del encoded, series, lookup
    labels_exact = all(audit["splits"][s][k]["exact"] for s in SPLITS for k in ("y", "mask"))
    if not labels_exact:
        atomic_json(audit, out / "data_audit.json")
        raise ValueError("Original target/mask hashes differ; PCA/training forbidden")
    audit["pca"] = fit_assets(arrays(out, "train"), stats, out)
    training_allowed = all(audit["splits"][s]["x"]["exact"] for s in SPLITS)
    audit["training_allowed"] = training_allowed
    audit["qualification"] = (
        "exact_original128_inputs_targets_masks_and_rederived_PCA"
        if training_allowed
        else "original_targets_masks_exact_input_fingerprints_unresolved_no_training"
    )
    atomic_json(audit, out / "data_audit.json")
    files = {
        str(p.relative_to(out)): digest(p)
        for p in out.rglob("*")
        if p.is_file() and p.name not in ("data_receipt.json",) and ".tmp" not in p.name
    }
    receipt = {
        "schema": SCHEMA,
        "training_allowed": training_allowed,
        "files": files,
        "neural_updates": 0,
    }
    atomic_json(receipt, out / "data_receipt.json")
    return receipt
