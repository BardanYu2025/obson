"""Read-only, fixed-source restoration for F05/F01 input organization and initialization.

Only the two reviewed input-patch bar workers are consumed. The old
root completion also names unused local/patch weights and prediction shards;
those are deliberately not prerequisites for this new, separately bound run.
Real checkpoints are loaded only by the AutoDL caller. This module never runs a
model or repairs an old checkpoint transaction.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from . import input_patch_run as old
from . import multiscale_data as data
from .dual_state import sha256
from .holdout_audit import read_json

SCHEMA = "babel-input-factorial-source-v1"
MANIFEST_SHA256 = "02828075b0aa2484bc4de4f6648cb1c5873f40327a8408c6dd4dedaf01af3243"
COMPLETION_SHA256 = "f6c4654241af1d103db49ac29c1e652158ef3a19a2af53f24096dee926df6589"
CHECKPOINT_SHA256 = {
    42: "f75e479ba8593c3d01b7bf17366e9e2a1a7bd9d14a2ae972d8264ea5c24b1a73",
    43: "bd32397899222d1c6ab3edb3b6818842e2d53622e4a4e8b6049ff9b2dbaea8d0",
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
        raise ValueError("Only reviewed bar seeds42/43 are source models")
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
        raise ValueError("Reviewed input-patch bar source directory is missing")
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
        name = f"initialization_s{seed}.json"
        _file(source, name, root_files.get(name))
        files[name] = root_files[name]
    for seed in CHECKPOINT_SHA256:
        name = f"bar_s{seed}"
        job = {"name": name, "arm": "bar", "seed": seed}
        completion_member = f"{name}/completion.json"
        _file(source, completion_member, root_files.get(completion_member))
        # This function is read-only. In contrast, recover_checkpoint may repair
        # receipts or remove pending files and must not be called on a source.
        done = old.check_job(meta, source, job)
        if done["files"]["last.pt"] != CHECKPOINT_SHA256[seed]:
            raise ValueError(f"Reviewed bar checkpoint differs: {name}")
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
            "Reviewed manifest/request/completion and both complete bar workers; "
            "unused local4/patch4 weights and old prediction shards are not consumed"
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
    """Load a bound bar checkpoint without executing it or changing source.

    The full payload includes Adam tensors for integrity checks only; these
    are never restored into the new scratch training optimizer.
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
    summary = read_json(source / f"bar_s{seed}" / "training_summary.json")
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


def validate_reference_state(ck, encoder, query, epoch, steps):
    """Validate the complete historical payload without requiring its GPU topology.

    This is the original strict tensor/Adam/RNG schema validation, specialized
    for a frozen inference reference. The caller loads encoder/query only and
    never restores its optimizer or RNG. The original CUDA RNG entries remain
    unmodified and individually validated; their number need not equal the
    current visible GPU count. New-worker resume keeps the old strict validator.
    Source metadata/history/hashes are independently checked by checkpoint().
    """
    import random
    from collections.abc import Mapping

    required = {
        "binding",
        "initial_signature",
        "initial_validation",
        "epoch",
        "history",
        "encoder",
        "query",
        "optimizer",
        "rng",
        "signature",
    }
    if (
        not isinstance(ck, Mapping)
        or set(ck) != required
        or type(ck["epoch"]) is not int
        or ck["epoch"] != epoch
    ):
        raise ValueError("Committed scratch checkpoint schema differs")
    for family, module in (("encoder", encoder), ("query", query)):
        current, saved = module.state_dict(), ck[family]
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            raise ValueError("Saved model tensor keys differ")
        for name, ref in current.items():
            value = saved[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != ref.shape
                or value.dtype != ref.dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Saved {family}/{name} shape/dtype/value differs")
    opt = ck["optimizer"]
    parameters = list(encoder.parameters()) + list(query.parameters())
    ids = list(range(len(parameters)))
    if (
        not isinstance(opt, Mapping)
        or set(opt) != {"state", "param_groups"}
        or len(opt["param_groups"]) != 1
    ):
        raise ValueError("Saved optimizer schema differs")
    group = opt["param_groups"][0]
    expected = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "lr": old.training.learning_rate(epoch) if epoch else old.training.PEAK_LR,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    if (
        group.get("params") != ids
        or not set(expected).issubset(group)
        or any(group.get(k) != v for k, v in expected.items())
        or set(opt["state"]) != (set(ids) if steps else set())
    ):
        raise ValueError("Saved optimizer recipe/order/support differs")
    for index, parameter in enumerate(parameters):
        if not steps:
            break
        moment = opt["state"][index]
        if not isinstance(moment, Mapping) or set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Saved Adam moment schema differs")
        step = moment["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.ndim != 0
            or step.dtype != torch.float32
            or not torch.isfinite(step).all()
            or float(step) != steps
        ):
            raise ValueError("Saved Adam steps differ")
        for key in ("exp_avg", "exp_avg_sq"):
            value = moment[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or not torch.isfinite(value).all()
                or (key == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError("Saved Adam moment shape/dtype/value differs")
    rng = ck["rng"]
    if not isinstance(rng, Mapping) or set(rng) != {"torch", "cuda", "python", "numpy"}:
        raise ValueError("Saved RNG schema differs")
    cpu = rng["torch"]
    if (
        not isinstance(cpu, torch.Tensor)
        or cpu.dtype != torch.uint8
        or cpu.shape != torch.get_rng_state().shape
    ):
        raise ValueError("Saved CPU RNG dtype/shape differs")
    local = torch.Generator()
    local.set_state(cpu)
    if not isinstance(rng["cuda"], list):
        raise ValueError("Historical CUDA RNG must be a list; it is never restored")
    for value in rng["cuda"]:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or not value.numel()
        ):
            raise ValueError("Saved CUDA RNG dtype/shape differs")
    random.Random().setstate(rng["python"])
    n = rng["numpy"]
    if not isinstance(n, (tuple, list)) or len(n) != 5 or n[0] != "MT19937":
        raise ValueError("Saved NumPy RNG schema differs")
    nums = np.asarray(n[1])
    if (
        nums.shape != (624,)
        or nums.dtype.kind not in "iu"
        or (nums < 0).any()
        or (nums > 2**32 - 1).any()
    ):
        raise ValueError("Saved NumPy RNG words differ")
    np.random.RandomState().set_state((n[0], nums.astype(np.uint32), n[2], n[3], n[4]))
