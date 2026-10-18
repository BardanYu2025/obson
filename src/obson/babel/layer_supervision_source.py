"""Read-only, fixed-source restoration for F05/F01 layer supervision.

Only the two reviewed observed-history control workers are consumed. The old
root completion also names unused full-attention weights and prediction shards;
those are deliberately not prerequisites for this new, separately bound run.
Real checkpoints are loaded only by the AutoDL caller. This module never runs a
model or repairs an old checkpoint transaction.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from . import multiscale_data as data
from . import observed_history_run as old
from .dual_state import sha256
from .holdout_audit import read_json

SCHEMA = "babel-layer-supervision-source-v1"
MANIFEST_SHA256 = "d8246106b4ea5b59aabf35603c90e98569417b1cf1ec84bb1156e46411f33f6c"
COMPLETION_SHA256 = "f3fc47547482a6d212738e8070d5adefca6d7099ef713bc49b217c2b19c99ca8"
CHECKPOINT_SHA256 = {
    42: "b0ed63644211d863b1a9af254165ab0bc5184d85b3772790f64beb76fa6ebb9d",
    43: "c4acde8cb6c8599ddd7959ad9db166e467e6b8f279be5c562ed377aeac3f1e36",
}
DATA_KEYS = ("coverage", "coverage_meta", "statistics", "source_verification")
CHECKPOINT_KEYS = {
    "binding",
    "epoch",
    "history",
    "initial_signature",
    "initial_validation",
    "encoder",
    "query",
    "optimizer",
    "rng",
    "signature",
}


def _json(value):
    """Compare serialized historical recipes, not Python tuple/list accidents."""
    return json.loads(json.dumps(value, allow_nan=False))


def _seed(seed):
    if type(seed) is not int or seed not in CHECKPOINT_SHA256:
        raise ValueError("Only reviewed control seeds42/43 are source models")
    return seed


def _file(root, relative, expected):
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("Unsafe source receipt member")
    path = data.safe_path(root / member)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Fixed source fingerprint differs: {relative}")
    return path


def _identity(source):
    """Verify only consumed files, never mutate the fixed source tree."""
    source = data.safe_path(source)
    if not source.is_dir():
        raise ValueError("Reviewed observed-history source directory is missing")
    manifest_path = _file(source, "manifest.json", MANIFEST_SHA256)
    completion_path = _file(source, "completion.json", COMPLETION_SHA256)
    meta, completion = read_json(manifest_path), read_json(completion_path)
    if (
        meta.get("schema") != old.SCHEMA
        or completion.get("schema") != old.SCHEMA
        or completion.get("status") != "complete_requires_review"
        or completion.get("promoted") is not False
        or completion.get("limited_hypothesis_supported") is not False
        or meta.get("protocol") != _json(old.protocol())
        or meta.get("implementation") != _json(old.implementation())
        or meta.get("encoder") != old.model.CONFIG
        or meta.get("counts") != data.COUNTS
        or meta.get("epochs") != old.EPOCHS
        or meta.get("batch") != old.BATCH
        or meta.get("micro") != old.MICRO
    ):
        raise ValueError("Reviewed source schema, recipe or bound implementation differs")
    root_files = completion.get("files", {})
    if root_files.get("manifest.json") != MANIFEST_SHA256:
        raise ValueError("Source root completion does not bind its manifest")
    files = {"manifest.json": MANIFEST_SHA256, "completion.json": COMPLETION_SHA256}
    request_path = _file(source, "request.json", root_files.get("request.json"))
    request = read_json(request_path)
    if request != {
        "schema": old.SCHEMA,
        "source": meta["source"],
        "output": str(source),
        "protocol": meta["protocol"],
        "implementation": meta["implementation"],
    }:
        raise ValueError("Original source request differs from its bound manifest")
    files["request.json"] = root_files["request.json"]
    controls = {}
    for seed in CHECKPOINT_SHA256:
        name = f"control_s{seed}"
        job = {"name": name, "arm": "control", "seed": seed}
        completion_member = f"{name}/completion.json"
        _file(source, completion_member, root_files.get(completion_member))
        # This function is read-only. In contrast, recover_checkpoint may repair
        # receipts or remove pending files and must not be called on a source.
        done = old.check_job(meta, source, job)
        if done["files"]["last.pt"] != CHECKPOINT_SHA256[seed]:
            raise ValueError(f"Reviewed control checkpoint differs: {name}")
        files[completion_member] = root_files[completion_member]
        for relative, digest in done["files"].items():
            member = f"{name}/{relative}"
            if root_files.get(member) != digest:
                raise ValueError(f"Source root completion omits consumed dependency: {member}")
            # check_job has checked the bytes and rejected symlink components.
            files[member] = digest
        controls[str(seed)] = {
            "job": job,
            "checkpoint": f"{name}/last.pt",
            "checkpoint_sha256": CHECKPOINT_SHA256[seed],
            "completion_sha256": root_files[completion_member],
            "binding": done["binding"],
            "epoch": done["epoch"],
            "updates": done["updates"],
            "signature": done["signature"],
            "initial_signature": done["initial_signature"],
        }
    return meta, {
        "schema": SCHEMA,
        "source": str(source),
        "manifest": MANIFEST_SHA256,
        "completion": COMPLETION_SHA256,
        "files": dict(sorted(files.items())),
        "controls": controls,
        "consumed_scope": (
            "Reviewed manifest/request/completion and both complete control workers; "
            "unused full_history weights and old prediction shards are not consumed"
        ),
    }


def verify(source, out):
    """Return original source metadata, independently verified data and identity.

    Source verification remains read-only. The inherited data verifier owns a
    separate temporary directory for its legacy diagnostic writes. No new-run
    metadata is passed as though it were an original data manifest.
    """
    source, out = data.safe_path(source), data.safe_path(out)
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Output must be separate from the immutable model source")
    original_meta, identity = _identity(source)
    data_source = data.safe_path(original_meta["source"])
    if old.source_identity(data_source) != original_meta["source_identity"]:
        raise ValueError("Original data-source completion or manifest differs")
    verified_data = data.verify(data_source, out)
    if any(original_meta.get(key) != verified_data.get(key) for key in DATA_KEYS):
        raise ValueError("Original source statistics, coverage or verified data lineage differs")
    return {
        "original_meta": original_meta,
        "verified_data": verified_data,
        "identity": identity,
    }


def checkpoint(meta, seed):
    """Load a bound control checkpoint without executing it or changing source.

    The real CUDA runner must subsequently strict-load the models and run the
    original optimizer validator, including every Adam moment and4560 steps.
    CPU tests may exercise this boundary only with synthetic/mocked payloads.
    """
    seed = _seed(seed)
    source = data.safe_path(meta["model_source"])
    _, identity = _identity(source)
    if identity != meta["source_identity"]:
        raise ValueError("Selected source identity changed before checkpoint loading")
    selected = identity["controls"][str(seed)]
    path = _file(source, selected["checkpoint"], selected["checkpoint_sha256"])
    value = torch.load(path, map_location="cpu", weights_only=True)
    # Detect a replacement during reading before the caller can use the state.
    _file(source, selected["checkpoint"], selected["checkpoint_sha256"])
    summary = read_json(source / f"control_s{seed}" / "training_summary.json")
    if not isinstance(value, Mapping) or set(value) != CHECKPOINT_KEYS:
        raise ValueError("Complete original model, optimizer and RNG checkpoint required")
    expected = {
        "binding": selected["binding"],
        "epoch": old.EPOCHS,
        "history": summary["history"],
        "signature": selected["signature"],
        "initial_signature": selected["initial_signature"],
        "initial_validation": summary["initial_validation"],
    }
    if any(value[key] != item for key, item in expected.items()):
        raise ValueError("Source checkpoint metadata differs from reviewed training summary")
    if len(value["history"]) != old.EPOCHS:
        raise ValueError("Source checkpoint is not the reviewed last120")
    if any(
        not isinstance(value[key], Mapping) or not value[key] for key in ("encoder", "query", "rng")
    ):
        raise ValueError("Original model or RNG state is missing")
    optimizer = value["optimizer"]
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer) != {"state", "param_groups"}
        or not isinstance(optimizer["state"], Mapping)
        or not optimizer["state"]
        or not isinstance(optimizer["param_groups"], list)
        or len(optimizer["param_groups"]) != 1
    ):
        raise ValueError("Complete single-group original Adam state is required")
    return value
