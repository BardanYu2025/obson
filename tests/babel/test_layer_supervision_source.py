"""Synthetic receipt/loader tests; no real model weights or inference."""

import copy
import json
import math

import numpy as np
import pytest

from obson.babel import layer_supervision_source as subject


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    """Use the REAL old receipt checker with fake byte files, never torch.load."""
    source, data_source, out = (tmp_path / name for name in ("source", "data", "out"))
    source.mkdir()
    data_source.mkdir()
    write_json(data_source / "manifest.json", {"synthetic": "data"})
    write_json(data_source / "completion.json", {"synthetic": "completed data"})
    counts = {"train": 3, "val": 2, "test": 2, "cross_research": 2}
    monkeypatch.setattr(subject.data, "COUNTS", counts)
    monkeypatch.setattr(subject.old, "implementation", lambda: {"fixture": "synthetic code"})
    verified = {
        "coverage": str(tmp_path / "coverage"),
        "coverage_meta": {"statistics": {"x_mean": [0.0]}},
        "statistics": {"x_mean": [0.0]},
        "source_verification": {"identity_sha256": "synthetic lineage"},
    }
    meta = verified | {
        "schema": subject.old.SCHEMA,
        "source": str(data_source),
        "source_identity": subject.old.source_identity(data_source),
        "encoder": subject.old.model.CONFIG,
        "counts": counts,
        "epochs": subject.old.EPOCHS,
        "batch": subject.old.BATCH,
        "micro": subject.old.MICRO,
        "protocol": subject.old.protocol(),
        "implementation": subject.old.implementation(),
    }
    write_json(source / "manifest.json", meta)
    manifest_hash = subject.sha256(source / "manifest.json")
    monkeypatch.setattr(subject, "MANIFEST_SHA256", manifest_hash)
    write_json(
        source / "request.json",
        {
            "schema": subject.old.SCHEMA,
            "source": str(data_source),
            "output": str(source),
            "protocol": meta["protocol"],
            "implementation": meta["implementation"],
        },
    )
    root_files = {name: subject.sha256(source / name) for name in ("manifest.json", "request.json")}
    weights, payloads = {}, {}
    for seed in (42, 43):
        name = f"control_s{seed}"
        folder = source / name
        folder.mkdir()
        job = {"name": name, "arm": "control", "seed": seed}
        binding = subject.old.binding(source, job)
        history = []
        for epoch in range(1, subject.old.EPOCHS + 1):
            order, ps = subject.old.training.schedule(counts["train"], seed, epoch)
            history.append(
                {
                    "arm": "control",
                    "epoch": epoch,
                    "steps": math.ceil(counts["train"] / subject.old.BATCH),
                    "examples": counts["train"],
                    "supervised_states": counts["train"] * 5,
                    "lr": subject.old.training.learning_rate(epoch),
                    "batch_size": subject.old.BATCH,
                    "micro_size": subject.old.MICRO,
                    "order_hash": subject.old.training.array_hash(order),
                    "prefix_hash": subject.old.training.array_hash(ps),
                    "position_counts": np.bincount(ps.ravel() - 1, minlength=128).tolist(),
                    "model_contract": {
                        "arm": "control",
                        "attention_mode": "causal",
                        "endpoint_only": True,
                        "input_length": 128,
                    },
                    "validation": {"synthetic": 1.0} if epoch % 5 == 0 else None,
                }
            )
        summary = {
            "binding": binding,
            "epoch": subject.old.EPOCHS,
            "initial_signature": f"synthetic initial{seed}",
            "signature": f"synthetic last{seed}",
            "history": history,
            "initial_validation": {"synthetic": 2.0},
            "updates": subject.old.EPOCHS,
            "examples": counts["train"] * subject.old.EPOCHS,
            "supervised_states": counts["train"] * subject.old.EPOCHS * 5,
        }
        (folder / "last.pt").write_bytes(f"synthetic-not-torch-{seed}".encode())
        weights[seed] = subject.sha256(folder / "last.pt")
        for name_audit in ("causal_audit.json", "final_causal_audit.json"):
            write_json(folder / name_audit, {"passed": True, "checks": {"synthetic": True}})
        write_json(
            folder / "checkpoint.json",
            {
                "epoch": subject.old.EPOCHS,
                "sha256": weights[seed],
                "binding": binding,
            },
        )
        write_json(folder / "training_summary.json", summary)
        (folder / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        files = {p.name: subject.sha256(p) for p in folder.iterdir()}
        write_json(
            folder / "completion.json",
            {
                "binding": binding,
                "epoch": subject.old.EPOCHS,
                "initial_signature": summary["initial_signature"],
                "signature": summary["signature"],
                "updates": summary["updates"],
                "files": files,
            },
        )
        root_files.update({f"{name}/{p.name}": subject.sha256(p) for p in folder.iterdir()})
        payloads[seed] = {
            key: copy.deepcopy(summary[key])
            for key in (
                "binding",
                "epoch",
                "history",
                "initial_signature",
                "initial_validation",
                "signature",
            )
        } | {
            "encoder": {"synthetic": 1.0},
            "query": {"synthetic": 2.0},
            "optimizer": {
                "state": {0: {"synthetic moment": 1.0}},
                "param_groups": [{"params": [0]}],
            },
            "rng": {"synthetic": 3.0},
        }
    # Retired/unused source artifacts are bound historically but need not remain.
    root_files["full_history_s42/last.pt"] = "e" * 64
    root_files["predictions/deleted_unused.npy"] = "f" * 64
    write_json(
        source / "completion.json",
        {
            "schema": subject.old.SCHEMA,
            "status": "complete_requires_review",
            "promoted": False,
            "limited_hypothesis_supported": False,
            "files": root_files,
        },
    )
    monkeypatch.setattr(subject, "CHECKPOINT_SHA256", weights)
    monkeypatch.setattr(subject, "COMPLETION_SHA256", subject.sha256(source / "completion.json"))
    calls = []

    def verify_data(actual_source, actual_out):
        calls.append((actual_source, actual_out))
        assert actual_source == data_source
        return copy.deepcopy(verified)

    monkeypatch.setattr(subject.data, "verify", verify_data)
    return {
        "source": source,
        "data_source": data_source,
        "out": out,
        "verified": verified,
        "payloads": payloads,
        "calls": calls,
    }


def test_read_only_verification_consumes_only_controls(source_tree, monkeypatch):
    source, out = source_tree["source"], source_tree["out"]
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        subject.torch, "load", lambda *a, **kw: pytest.fail("verify must not load weights")
    )
    result = subject.verify(source, out)
    after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert before == after
    assert not out.exists()
    assert result["verified_data"] == source_tree["verified"]
    assert source_tree["calls"] == [(source_tree["data_source"], out)]
    assert set(result["identity"]["controls"]) == {"42", "43"}
    assert not any(
        "full_history" in key or "predictions/" in key for key in result["identity"]["files"]
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "relative",
    ["manifest.json", "completion.json", "control_s42/last.pt", "control_s43/checkpoint.json"],
)
def test_mutated_consumed_source_fails_before_data_access(source_tree, relative):
    with (source_tree["source"] / relative).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="fingerprint|Committed artifact"):
        subject.verify(source_tree["source"], source_tree["out"])
    assert not source_tree["calls"]


def test_root_completion_must_cover_consumed_worker(source_tree, monkeypatch):
    path = source_tree["source"] / "completion.json"
    value = json.loads(path.read_text())
    del value["files"]["control_s42/history.jsonl"]
    write_json(path, value)
    monkeypatch.setattr(subject, "COMPLETION_SHA256", subject.sha256(path))
    with pytest.raises(ValueError, match="omits consumed dependency"):
        subject.verify(source_tree["source"], source_tree["out"])


def test_bound_implementation_must_match(source_tree, monkeypatch):
    monkeypatch.setattr(subject.old, "implementation", lambda: {"fixture": "changed code"})
    with pytest.raises(ValueError, match="bound implementation"):
        subject.verify(source_tree["source"], source_tree["out"])


@pytest.mark.parametrize("field", subject.DATA_KEYS)
def test_independent_data_must_match_original_fields(source_tree, monkeypatch, field):
    modified = copy.deepcopy(source_tree["verified"])
    modified[field] = {"different": True}
    monkeypatch.setattr(subject.data, "verify", lambda *args: modified)
    with pytest.raises(ValueError, match="verified data lineage"):
        subject.verify(source_tree["source"], source_tree["out"])


def test_uncommitted_source_stops_without_repair(source_tree):
    pending = source_tree["source"] / "control_s42/pending_epoch.json"
    write_json(pending, {"epoch": 121})
    with pytest.raises(ValueError, match="unresolved transaction"):
        subject.verify(source_tree["source"], source_tree["out"])
    assert pending.exists()


def test_output_isolation_and_symlinks(source_tree, tmp_path):
    source = source_tree["source"]
    for out in (source, source / "nested", source.parent):
        with pytest.raises(ValueError, match="separate"):
            subject.verify(source, out)
    link = tmp_path / "link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        subject.verify(link, source_tree["out"])
    weight = source / "control_s42/last.pt"
    target = tmp_path / "weight"
    weight.replace(target)
    weight.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        subject.verify(source, source_tree["out"])


def test_checkpoint_load_is_weights_only_and_identity_bound(source_tree, monkeypatch):
    result = subject.verify(source_tree["source"], source_tree["out"])
    meta = {"model_source": str(source_tree["source"]), "source_identity": result["identity"]}
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return source_tree["payloads"][42]

    monkeypatch.setattr(subject.torch, "load", load)
    checkpoint = subject.checkpoint(meta, 42)
    assert checkpoint is source_tree["payloads"][42]
    assert calls == [
        (
            source_tree["source"] / "control_s42/last.pt",
            {"map_location": "cpu", "weights_only": True},
        )
    ]
    invalid = copy.deepcopy(meta)
    invalid["source_identity"]["controls"]["42"]["signature"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        subject.checkpoint(invalid, 42)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change", ["optimizer_missing", "optimizer_empty", "epoch", "history", "signature", "binding"]
)
def test_checkpoint_payload_cannot_silently_reset_or_change_source(
    source_tree, monkeypatch, change
):
    result = subject.verify(source_tree["source"], source_tree["out"])
    meta = {"model_source": str(source_tree["source"]), "source_identity": result["identity"]}
    value = copy.deepcopy(source_tree["payloads"][43])
    if change == "optimizer_missing":
        del value["optimizer"]
    elif change == "optimizer_empty":
        value["optimizer"]["state"] = {}
    elif change == "history":
        value["history"] = value["history"][:-1]
    elif change == "epoch":
        value["epoch"] = 119
    else:
        value[change] = "changed"
    monkeypatch.setattr(subject.torch, "load", lambda *a, **kw: value)
    with pytest.raises(ValueError, match="Complete|metadata differs"):
        subject.checkpoint(meta, 43)


@pytest.mark.parametrize("seed", [41, 44, True, 42.0, "42"])
def test_only_two_reviewed_source_seeds(seed):
    with pytest.raises(ValueError, match="seeds42/43"):
        subject.checkpoint({}, seed)
