"""Synthetic source/raw provenance tests; never restore actual weights."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from obson.babel import multiscale_data as data
from obson.babel.ae_extend import atomic_json
from obson.babel.dual_state import sha256

STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


def source_directory(tmp_path, manifest=None):
    source = tmp_path / "source"
    source.mkdir()
    atomic_json({} if manifest is None else manifest, source / "manifest.json")
    atomic_json({}, source / "completion.json")
    return source


def legacy_verifier(meta, scratch_paths, *, fail=None, write_identity=True):
    """Model the inherited write -> read -> exact-byte-hash verification chain."""

    def verify(source, scratch):
        assert scratch.is_dir()
        assert scratch == scratch.resolve()
        scratch_paths.append(scratch)
        if write_identity:
            identity = {
                "source": str(source),
                "manifest_sha256": sha256(source / "manifest.json"),
                "selected_weight_edges": [],
            }
            atomic_json(identity, scratch / "identity.json")
            expected_hash = sha256(scratch / "identity.json")
            # These two consumers represent qualification JSON replay and the
            # downstream immutable source check. Both require the real file.
            assert json.loads((scratch / "identity.json").read_text()) == identity
            assert sha256(scratch / "identity.json") == expected_hash
        if fail is not None:
            raise ValueError(fail)
        return copy.deepcopy(meta)

    return verify


def tree_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def rows(split, n):
    year = {"train": 2023, "val": 2024, "test": 2025, "cross_research": 2025}[split]
    return [
        {
            "key": f"A/15/A{i}",
            "symbol": "A",
            "period": 15,
            "row": 254,
            "input_start": f"{year}-01-01 00:00:00",
            "end": f"{year}-01-04 00:00:00",
        }
        for i in range(n)
    ]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    counts = {"train": 3, "val": 2, "test": 2, "cross_research": 2}
    monkeypatch.setattr(data, "COUNTS", counts)
    out = tmp_path / "out"
    out.mkdir()
    meta = {
        "coverage": str(tmp_path / "coverage"),
        "coverage_meta": {
            "context": str(tmp_path / "context"),
            "source_meta": {
                "identity": {
                    "manifest": {
                        "identity": {
                            "training_manifest": {"raw_root": str(tmp_path / "raw_source")}
                        }
                    }
                }
            },
            "statistics": STATS,
            "source_indexes": {s: str(i) * 64 for i, s in enumerate(counts)},
        },
        "statistics": STATS,
    }
    atomic_json(meta, out / "manifest.json")
    bank = {s: (np.zeros((n, 255, 28), np.float32), rows(s, n)) for s, n in counts.items()}
    monkeypatch.setattr(data.original, "cache", lambda meta, split: bank[split])

    def prepare(meta, out, splits):
        (out / "raw").mkdir(exist_ok=True)
        for split in splits:
            path = out / "raw" / f"{split}.npz"
            if path.exists():
                continue
            raw = np.tile(np.linspace(100, 102, 255), (counts[split], 1))
            np.savez(path, close=raw)
            atomic_json(
                {
                    "binding": {
                        "manifest": sha256(out / "manifest.json"),
                        "source_index": meta["coverage_meta"]["source_indexes"][split],
                    },
                    "sha256": sha256(path),
                },
                out / "raw" / f"{split}.json",
            )
            atomic_json(bank[split][1], out / f"{split}_rows.json")
        return {}

    monkeypatch.setattr(data.original, "prepare", prepare)
    return meta, out, bank


def test_manifest_first_and_source_destination_edges(tmp_path, prepared, monkeypatch):
    meta, out, _ = prepared
    source = source_directory(tmp_path)
    scratch_paths = []
    monkeypatch.setattr(data.original, "verify", legacy_verifier(meta, scratch_paths))
    result = data.verify(source, out)
    assert {key: value for key, value in result.items() if key != "source_verification"} == meta
    assert result["source_verification"]["source_manifests"] == {
        str(source / "manifest.json"): sha256(source / "manifest.json")
    }
    for wrong in (data.raw_root(meta), data.raw_root(meta) / "nested"):
        with pytest.raises(ValueError, match="raw data root"):
            data.verify(source, wrong)
    for wrong in (source, source / "nested", tmp_path):
        with pytest.raises(ValueError, match="Separate"):
            data.verify(source, wrong)
    link = tmp_path / "linked"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        data.verify(link, out)
    with pytest.raises(ValueError, match="traversal"):
        data.verify(source, out / ".." / "escape")
    (out / "manifest.json").unlink()
    with (
        patch.object(data.original, "prepare") as delegated,
        pytest.raises(ValueError, match="manifest"),
    ):
        data.prepare(meta, out, ("train", "val"))
    delegated.assert_not_called()
    assert all(not path.exists() for path in scratch_paths)


def test_first_verify_creates_only_temporary_audit_and_retains_its_evidence(
    tmp_path, prepared, monkeypatch
):
    meta, _, _ = prepared
    source = source_directory(tmp_path)
    out = tmp_path / "not_created" / "run"
    scratch_paths = []
    monkeypatch.setattr(data.original, "verify", legacy_verifier(meta, scratch_paths))

    result = data.verify(source, out)

    assert not out.parent.exists()
    assert len(scratch_paths) == 1 and not scratch_paths[0].exists()
    evidence = result["source_verification"]
    assert evidence["identity"] == {
        "source": str(source),
        "manifest_sha256": sha256(source / "manifest.json"),
        "selected_weight_edges": [],
    }
    expected = tmp_path / "expected_identity.json"
    atomic_json(evidence["identity"], expected)
    assert evidence["identity_sha256"] == sha256(expected)
    assert str(scratch_paths[0]) not in json.dumps(result)


def test_repeated_verify_preserves_existing_output_and_has_stable_evidence(
    tmp_path, prepared, monkeypatch
):
    meta, out, _ = prepared
    source = source_directory(tmp_path)
    (out / "unrelated.txt").write_text("preserve existing evidence")
    before = tree_bytes(out)
    scratch_paths = []
    monkeypatch.setattr(data.original, "verify", legacy_verifier(meta, scratch_paths))

    first = data.verify(source, out)
    second = data.verify(source, out)

    assert first == second
    assert tree_bytes(out) == before
    assert len(set(scratch_paths)) == 2
    assert all(not path.exists() for path in scratch_paths)
    assert all(str(path) not in json.dumps(first) for path in scratch_paths)


@pytest.mark.parametrize("fault", ("source", "statistics", "missing_identity"))
def test_invalid_source_audit_cleans_scratch_and_never_creates_output(
    fault, tmp_path, prepared, monkeypatch
):
    meta, _, _ = prepared
    meta = copy.deepcopy(meta)
    if fault == "statistics":
        meta["statistics"] = copy.deepcopy(meta["statistics"])
        meta["statistics"]["x_scale"][0] *= 2
    source = source_directory(tmp_path)
    out = tmp_path / "not_created" / "run"
    scratch_paths = []
    monkeypatch.setattr(
        data.original,
        "verify",
        legacy_verifier(
            meta,
            scratch_paths,
            fail="Forged source identity" if fault == "source" else None,
            write_identity=fault != "missing_identity",
        ),
    )
    message = {
        "source": "Forged source",
        "statistics": "statistics differ",
        "missing_identity": "identity evidence",
    }[fault]
    with pytest.raises(ValueError, match=message):
        data.verify(source, out)
    assert not out.parent.exists()
    assert len(scratch_paths) == 1 and not scratch_paths[0].exists()


@pytest.mark.parametrize(
    "field", ("source", "qualification", "context", "coverage", "training_source", "raw_root")
)
def test_all_ancestral_manifest_branches_remain_protected(field, tmp_path, prepared, monkeypatch):
    meta, _, _ = prepared
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    source = source_directory(tmp_path, {"nested": [{field: str(ancestor)}]})
    # Exercise a cycle too: source roots may recur in nested identity manifests.
    atomic_json({"source": str(source)}, ancestor / "manifest.json")
    before = tree_bytes(ancestor)
    scratch_paths = []
    monkeypatch.setattr(data.original, "verify", legacy_verifier(meta, scratch_paths))
    for out in (ancestor, ancestor / "new_run"):
        with pytest.raises(ValueError, match="ancestral source"):
            data.verify(source, out)
        assert tree_bytes(ancestor) == before
        assert not (ancestor / "new_run").exists()
    assert all(not path.exists() for path in scratch_paths)


def test_ancestor_graph_collects_manifest_hashes_once_despite_cycles(
    tmp_path, prepared, monkeypatch
):
    meta, out, _ = prepared
    ancestor = tmp_path / "ancestor"
    qualification = tmp_path / "qualification"
    ancestor.mkdir()
    qualification.mkdir()
    source = source_directory(tmp_path, {"source": str(ancestor)})
    atomic_json({"qualification": str(qualification)}, ancestor / "manifest.json")
    atomic_json({"identity": {"source": str(source)}}, qualification / "manifest.json")
    scratch_paths = []
    monkeypatch.setattr(data.original, "verify", legacy_verifier(meta, scratch_paths))

    result = data.verify(source, out)

    assert result["source_verification"]["source_manifests"] == {
        str(root / "manifest.json"): sha256(root / "manifest.json")
        for root in (source, ancestor, qualification)
    }
    assert all(
        Path(name).is_absolute() for name in result["source_verification"]["source_manifests"]
    )
    assert all(not path.exists() for path in scratch_paths)


def test_source_constant_and_verified_completion_required(tmp_path):
    assert data.COMPLETION == "835cce0452c27173f808b530871624b59041dceb71552074bf32e4c722e4139d"
    root = tmp_path / "source"
    root.mkdir()
    atomic_json({}, root / "completion.json")
    atomic_json({}, root / "manifest.json")
    with pytest.raises(ValueError, match="reviewed overlap"):
        data.verify(root, tmp_path / "new_run")
    assert not (tmp_path / "new_run").exists()


def test_full255_train_val_overlap_fails_before_raw_prepare(prepared):
    meta, out, bank = prepared
    bank["val"] = (bank["val"][0], copy.deepcopy(bank["train"][1][:2]))
    with (
        patch.object(data.original, "prepare") as delegated,
        pytest.raises(ValueError, match="overlap"),
    ):
        data.prepare(meta, out, ("train", "val"))
    delegated.assert_not_called()


def test_raw_provenance_dtype_endpoints_and_no_source_model_loading(prepared):
    meta, out, bank = prepared
    with patch.object(data.original, "load_source", side_effect=AssertionError("No real weights")):
        receipts = data.prepare(meta, out, ("train", "val"))
    assert set(receipts) == {"train", "val"}
    assert data.closes(out, "train").dtype == np.float64
    assert json.loads((out / "data_partition_audit.json").read_text())["overlapping_intervals"] == 0
    assert data.prepare(meta, out, ("train", "val")) == receipts
    bank["test"][1][0]["end"] = "2026-09-12 03:00:00"
    with pytest.raises(ValueError, match="not closed"):
        data.cache(meta, "test")
    bank["test"][1][0]["end"] = "2025-01-04 00:00:00+08:00"
    with pytest.raises(ValueError, match="Naive"):
        data.cache(meta, "test")


def test_float32_raw_cannot_silently_replace_original_prices(prepared):
    meta, out, _ = prepared
    data.prepare(meta, out, ("train", "val"))
    path = out / "raw" / "train.npz"
    raw = data.closes(out, "train")
    np.savez(path, close=raw.astype(np.float32))
    receipt = json.loads((out / "raw" / "train.json").read_text())
    receipt["sha256"] = sha256(path)
    atomic_json(receipt, out / "raw" / "train.json")
    with pytest.raises(ValueError, match="float64"):
        data.closes(out, "train")


def test_fit_once_all_endpoints_and_reentry_never_refits(prepared):
    meta, out, bank = prepared
    data.prepare(meta, out, ("train", "val"))
    raw = data.closes(out, "train")
    result = data.fit_scales(meta, out, raw, bank["train"][1])
    assert result["contexts"] == 3 * 128 and result["training_rows"] == 3
    receipt = json.loads((out / "multiscale_scales_receipt.json").read_text())
    assert receipt["binding"]["raw_array_sha256"] == data.training.array_hash(raw)
    assert receipt["sha256"] == sha256(out / "multiscale_scales.json")
    with patch.object(data.training, "fit_train_scales", side_effect=AssertionError("No refit")):
        assert data.fit_scales(meta, out, raw, bank["train"][1]) == result
    with pytest.raises(ValueError, match="float64"):
        data.fit_scales(meta, out, raw.astype(np.float32), bank["train"][1])
    with pytest.raises(ValueError, match="rows differ"):
        data.fit_scales(meta, out, raw, list(reversed(bank["train"][1])))
    result["scales"][0] *= 2
    atomic_json(result, out / "multiscale_scales.json")
    with pytest.raises(ValueError, match="Frozen"):
        data.fit_scales(meta, out, raw, bank["train"][1])


def test_incomplete_scale_commit_refused_without_automatic_refit(prepared):
    meta, out, bank = prepared
    data.prepare(meta, out, ("train", "val"))
    atomic_json({}, out / "multiscale_scales.json")
    with (
        patch.object(data.training, "fit_train_scales") as fitter,
        pytest.raises(ValueError, match="Incomplete"),
    ):
        data.fit_scales(meta, out, data.closes(out, "train"), bank["train"][1])
    fitter.assert_not_called()


def test_symlink_temporary_cannot_redirect_inherited_atomic_writes(prepared, tmp_path):
    meta, out, _ = prepared
    victim = tmp_path / "victim.json"
    victim.write_text("untouched")
    (out / "data_partition_audit.json.tmp").symlink_to(victim)
    with (
        patch.object(data.original, "prepare") as delegated,
        pytest.raises(ValueError, match="Symlink"),
    ):
        data.prepare(meta, out, ("train", "val"))
    delegated.assert_not_called()
    assert victim.read_text() == "untouched"


def test_cache_rejects_float64_features_and_duplicate_rows(prepared):
    meta, _, bank = prepared
    x, rows_ = bank["train"]
    bank["train"] = (x.astype(np.float64), rows_)
    with pytest.raises(ValueError, match="float32"):
        data.cache(meta, "train")
    bank["train"] = (x, [rows_[0], rows_[0], rows_[2]])
    with pytest.raises(ValueError, match="Duplicate"):
        data.cache(meta, "train")
