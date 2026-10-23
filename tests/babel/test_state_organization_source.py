"""Synthetic source receipts; no real checkpoint deserialization or model execution."""

import copy
import json
import math
from functools import lru_cache

import numpy as np
import pytest

from obson.babel import state_organization_source as subject

_source_schedule = subject.old.training.schedule


@lru_cache(maxsize=120)
def cached_original_schedule(n, seed, epoch):
    # Cache pure original arithmetic; the real receipt checker still compares
    # every saved hash/count and never mutates the returned arrays.
    return _source_schedule(n, seed, epoch)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    source, out, data_source, prior = [tmp_path / n for n in ("source", "out", "data", "prior")]
    monkeypatch.setattr(subject.old.training, "schedule", cached_original_schedule)
    source.mkdir()
    data_source.mkdir()
    prior.mkdir()
    monkeypatch.setattr(subject.old, "implementation", lambda: {"synthetic": "code"})
    counts = dict(subject.data.COUNTS)
    verified = {
        "coverage": str(tmp_path / "coverage"),
        "coverage_meta": {"statistics": {"x_mean": [0.0]}},
        "statistics": {"x_mean": [0.0]},
        "source_verification": {"identity": "synthetic"},
    }
    prior_identity = {"synthetic": "reviewed120"}
    meta = verified | {
        "schema": subject.old.SCHEMA,
        "source": str(data_source),
        "model_source": str(prior),
        "source_identity": prior_identity,
        "encoder": subject.old.model.CONFIG,
        "counts": counts,
        "epochs": 60,
        "batch": 128,
        "micro": 8,
        "protocol": subject.old.protocol(),
        "implementation": subject.old.implementation(),
    }
    write(source / "manifest.json", meta)
    monkeypatch.setattr(subject, "MANIFEST_SHA256", subject.sha256(source / "manifest.json"))
    write(
        source / "request.json",
        {
            "schema": subject.old.SCHEMA,
            "source": str(prior),
            "output": str(source),
            "protocol": meta["protocol"],
            "implementation": meta["implementation"],
        },
    )
    files = {n: subject.sha256(source / n) for n in ("manifest.json", "request.json")}
    weights, payloads = {}, {}
    for seed in (42, 43):
        name = f"control_s{seed}"
        folder = source / name
        folder.mkdir()
        job = {"name": name, "arm": "control", "seed": seed}
        binding = subject.old.binding(source, job)
        history = []
        for epoch in range(1, 61):
            order, ps = subject.old.training.schedule(counts["train"], seed, epoch)
            history.append(
                {
                    "arm": "control",
                    "epoch": epoch,
                    "global_epoch": 120 + epoch,
                    "alpha": subject.old.training.alpha(epoch),
                    "head_passes": 2,
                    "steps": math.ceil(counts["train"] / 128),
                    "examples": counts["train"],
                    "supervised_states": counts["train"] * 5,
                    "query_state_reads": counts["train"] * 10,
                    "lr": subject.old.training.learning_rate(epoch),
                    "batch_size": 128,
                    "micro_size": 8,
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
            "epoch": 60,
            "initial_signature": f"initial{seed}",
            "signature": f"last{seed}",
            "history": history,
            "initial_validation": {"synthetic": 2.0},
            "updates": 2280,
            "cumulative_updates": 6840,
            "examples": counts["train"] * 60,
            "supervised_states": counts["train"] * 60 * 5,
            "query_state_reads": counts["train"] * 60 * 10,
        }
        (folder / "last.pt").write_bytes(f"fake-not-a-checkpoint-{seed}".encode())
        weights[seed] = subject.sha256(folder / "last.pt")
        write(
            folder / "checkpoint.json", {"epoch": 60, "sha256": weights[seed], "binding": binding}
        )
        for name_audit in ("causal_audit.json", "final_causal_audit.json"):
            write(folder / name_audit, {"passed": True, "checks": {"synthetic": True}})
        write(
            folder / "source_restore.json",
            {
                "source_identity": prior_identity,
                "source_seed": seed,
                "source_updates": 4560,
                "signature": summary["initial_signature"],
                "complete_adam_restored": True,
                "optimizer_reset": False,
            },
        )
        write(folder / "training_summary.json", summary)
        (folder / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        done_files = {p.name: subject.sha256(p) for p in folder.iterdir()}
        write(
            folder / "completion.json",
            {
                k: summary[k]
                for k in (
                    "binding",
                    "epoch",
                    "initial_signature",
                    "signature",
                    "updates",
                    "cumulative_updates",
                )
            }
            | {"files": done_files},
        )
        files.update({f"{name}/{p.name}": subject.sha256(p) for p in folder.iterdir()})
        payloads[seed] = {
            k: copy.deepcopy(summary[k])
            for k in (
                "binding",
                "epoch",
                "history",
                "initial_signature",
                "initial_validation",
                "signature",
            )
        } | {
            "encoder": {"synthetic": 1},
            "query": {"synthetic": 2},
            "rng": {"synthetic": 3},
            "optimizer": {"state": {0: {"synthetic": 4}}, "param_groups": [{"params": [0]}]},
        }
    files["layer2_s42/last.pt"] = "e" * 64
    files["predictions/deleted.npy"] = "f" * 64
    write(
        source / "completion.json",
        {
            "schema": subject.old.SCHEMA,
            "status": "complete_requires_review",
            "promoted": False,
            "limited_hypothesis_supported": False,
            "files": files,
        },
    )
    monkeypatch.setattr(subject, "COMPLETION_SHA256", subject.sha256(source / "completion.json"))
    monkeypatch.setattr(subject, "CHECKPOINT_SHA256", weights)
    calls = []

    def verify(actual_source, actual_out):
        calls.append((actual_source, actual_out))
        assert actual_source == data_source
        return copy.deepcopy(verified)

    monkeypatch.setattr(subject.data, "verify", verify)
    return {
        "source": source,
        "out": out,
        "verified": verified,
        "payloads": payloads,
        "calls": calls,
        "prior": prior,
    }


def test_consumes_only_controls_and_data_read_only(tree, monkeypatch):
    source = tree["source"]
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        subject.torch, "load", lambda *a, **k: pytest.fail("No weights during verification")
    )
    result = subject.verify(source, tree["out"])
    after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    assert before == after and result["verified_data"] == tree["verified"]
    assert not tree["out"].exists()
    assert all(not p.startswith(("layer2_", "predictions/")) for p in result["identity"]["files"])
    assert result["identity"]["controls"]["42"]["epoch"] == 60
    assert result["identity"]["controls"]["42"]["cumulative_epoch"] == 180
    assert result["identity"]["controls"]["42"]["cumulative_updates"] == 6840


@pytest.mark.parametrize(
    "relative",
    ["manifest.json", "request.json", "control_s42/last.pt", "control_s43/history.jsonl"],
)
def test_changed_consumed_byte_stops_before_data(tree, relative):
    p = tree["source"] / relative
    p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        subject.verify(tree["source"], tree["out"])
    assert not tree["calls"]


def test_uncommitted_source_not_repaired(tree):
    p = tree["source"] / "control_s42/pending_epoch.json"
    write(p, {"epoch": 61})
    with pytest.raises(ValueError, match="unresolved"):
        subject.verify(tree["source"], tree["out"])
    assert p.exists()


@pytest.mark.parametrize("relation", ["inside", "ancestor", "prior", "symlink"])
def test_output_isolation(tree, relation):
    out = {
        "inside": tree["source"] / "child",
        "ancestor": tree["source"].parent,
        "prior": tree["prior"] / "child",
    }.get(relation)
    if relation == "symlink":
        out = tree["out"]
        out.symlink_to(tree["source"], target_is_directory=True)
    with pytest.raises(ValueError):
        subject.verify(tree["source"], out)


def test_checkpoint_local60_global180_and_weights_only(tree, monkeypatch):
    meta = {
        "model_source": str(tree["source"]),
        "source_identity": subject._identity(tree["source"])[1],
    }

    def load(path, **kwargs):
        assert path == tree["source"] / "control_s42/last.pt"
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return copy.deepcopy(tree["payloads"][42])

    monkeypatch.setattr(subject.torch, "load", load)
    ck = subject.checkpoint(meta, 42)
    assert ck["epoch"] == 60 and ck["history"][-1]["global_epoch"] == 180


@pytest.mark.parametrize(
    "mutation", ["epoch180", "missing_optimizer", "missing_rng", "bad_global_epoch", "extra_key"]
)
def test_bad_payload_rejected(tree, monkeypatch, mutation):
    value = copy.deepcopy(tree["payloads"][42])
    if mutation == "epoch180":
        value["epoch"] = 180
    elif mutation == "missing_optimizer":
        value["optimizer"] = {}
    elif mutation == "missing_rng":
        value["rng"] = {}
    elif mutation == "bad_global_epoch":
        value["history"][-1]["global_epoch"] = 60
    else:
        value["unused"] = "extra"
    monkeypatch.setattr(subject.torch, "load", lambda *a, **k: value)
    meta = {
        "model_source": str(tree["source"]),
        "source_identity": subject._identity(tree["source"])[1],
    }
    with pytest.raises(ValueError):
        subject.checkpoint(meta, 42)


def test_changed_independent_data_rejected(tree, monkeypatch):
    wrong = copy.deepcopy(tree["verified"])
    wrong["statistics"] = {"different": True}
    monkeypatch.setattr(subject.data, "verify", lambda *a: wrong)
    with pytest.raises(ValueError, match="statistics"):
        subject.verify(tree["source"], tree["out"])
