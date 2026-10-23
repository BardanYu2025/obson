"""Matched Q-only organization adaptation with explicit inherited/new Adam groups.

Real-weight restoration and updates belong to the AutoDL runner. Local tests use
synthetic tensors and never execute a real checkpoint. The inherited encoder and
query keep their original Adam moments; new memory parameters begin with no state.
"""

import copy
import hashlib
import math
import time
from collections.abc import Mapping

import numpy as np
import torch

from . import history_query as hq
from . import multiscale_content as content
from . import multiscale_training as original
from . import state_coverage as coverage
from . import state_organization as model
from .ae_extend import restore_rng

SCHEMA = "babel-state-organization-training-v1"
ARMS = model.ARMS
SOURCE_EPOCH = 180
SOURCE_LOCAL_EPOCH = 60
SOURCE_UPDATES = 6840
EPOCHS = 60
BATCH_SIZE = 128
MICRO_SIZE = 8
INHERITED_LR = original.learning_rate(original.EPOCHS)
MEMORY_LR = 3e-4
PEAK_LR = INHERITED_LR
CLIP_NORM = 1.0
WEIGHT_DECAY = 0.01
array_hash = original.array_hash
parameter_signature = original.parameter_signature


def _epoch(value, *, allow_zero=False):
    if type(value) is not int or not (0 if allow_zero else 1) <= value <= EPOCHS:
        raise ValueError("Additional epoch must be an integer within the fixed60 budget")
    return value


def learning_rate(epoch):
    _epoch(epoch, allow_zero=True)
    return INHERITED_LR


def schedule(n, seed, epoch):
    n = original._positive_integer(n, "Row count")
    _epoch(epoch)
    return coverage.schedule(n, original._seed(seed), SOURCE_EPOCH + epoch, "D")


def parameter_groups(encoder, query):
    """Deterministic inherited order: encoder excluding memory, then old query."""
    inherited, memory = [], []
    for name, value in encoder.named_parameters():
        (memory if name.startswith("memory.") else inherited).append(("encoder/" + name, value))
    inherited.extend(("query/" + name, value) for name, value in query.named_parameters())
    if not inherited or len({id(v) for _, v in inherited + memory}) != len(inherited + memory):
        raise ValueError("Unique inherited/query/memory parameter inventory required")
    if bool(memory) != (encoder.arm != "plain"):
        raise ValueError("Memory parameter presence differs from the declared arm")
    return {"inherited": inherited, "memory": memory}


def inherited_signature(encoder, query):
    """Exact historical signature, excluding only explicitly new memory tensors."""
    digest = hashlib.sha256()
    for family, net in (("encoder", encoder), ("query", query)):
        for name, tensor in net.state_dict().items():
            if family == "encoder" and name.startswith("memory."):
                continue
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{family}/{name}:{value.dtype}:{value.shape}\n".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def new_optimizer(encoder, query):
    groups = parameter_groups(encoder, query)
    descriptions = [{"params": [p for _, p in groups["inherited"]], "lr": INHERITED_LR}]
    if groups["memory"]:
        descriptions.append({"params": [p for _, p in groups["memory"]], "lr": MEMORY_LR})
    return torch.optim.AdamW(
        descriptions,
        lr=INHERITED_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


def _layout(optimizer, encoder, query):
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("AdamW required")
    groups = parameter_groups(encoder, query)
    active = [name for name in ("inherited", "memory") if groups[name]]
    if len(optimizer.param_groups) != len(active):
        raise ValueError("Exact inherited and optional memory Adam groups required")
    for name, group in zip(active, optimizer.param_groups, strict=True):
        if list(map(id, group["params"])) != [id(p) for _, p in groups[name]]:
            raise ValueError("Optimizer parameter identity/order differs")
        expected = {
            "lr": INHERITED_LR if name == "inherited" else MEMORY_LR,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": False,
            "maximize": False,
        }
        if any(group.get(k) != v for k, v in expected.items()):
            raise ValueError("Fixed inherited/memory optimizer recipe differs")
    return groups


def _moment(state, parameter, steps):
    if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("Complete Adam step and both moments required")
    step = state["step"]
    if (
        not isinstance(step, torch.Tensor)
        or step.numel() != 1
        or not torch.isfinite(step).all()
        or float(step) != steps
    ):
        raise ValueError("Adam step differs from inherited/new exposure")
    for name in ("exp_avg", "exp_avg_sq"):
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != parameter.shape
            or value.dtype != parameter.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid Adam moment shape/dtype/value")
        if name == "exp_avg_sq" and (value < 0).any():
            raise ValueError("Negative Adam second moment")


def validate_optimizer(optimizer, encoder, query, epoch, steps):
    _epoch(epoch, allow_zero=True)
    if type(steps) is not int or steps < 0 or (epoch == 0) != (steps == 0):
        raise ValueError("Additional committed epoch/steps differ")
    groups = _layout(optimizer, encoder, query)
    expected = {p for _, p in groups["inherited"]}
    if steps:
        expected.update(p for _, p in groups["memory"])
    if set(optimizer.state) != expected:
        raise ValueError("Inherited/new Adam state support differs")
    for family, members in groups.items():
        for _, parameter in members:
            if not torch.isfinite(parameter).all():
                raise ValueError("Nonfinite model parameter")
            if family == "memory" and not steps:
                continue
            _moment(
                optimizer.state[parameter],
                parameter,
                SOURCE_UPDATES + steps if family == "inherited" else steps,
            )


def _strict_tensors(current, saved, label):
    if not isinstance(saved, Mapping) or list(saved) != list(current):
        raise ValueError(f"Exact inherited {label} tensor keys/order required")
    for name, ref in current.items():
        value = saved[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != ref.shape
            or value.dtype != ref.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Inherited {label}/{name} shape/dtype/value differs")


def restore_source(encoder, query, optimizer, checkpoint):
    """Strictly restore the reviewed control180; append empty memory Adam state.

    Source optimizer numeric IDs are accepted only in the historical contiguous
    parameter order. That order is checked against exact inherited state tensor
    keys and recorded explicitly by name in the returned migration receipt.
    """
    groups = _layout(optimizer, encoder, query)
    if optimizer.state:
        raise ValueError("Source restoration requires an empty destination optimizer")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("epoch") != SOURCE_LOCAL_EPOCH
        or not isinstance(checkpoint.get("history"), list)
        or len(checkpoint["history"]) != SOURCE_LOCAL_EPOCH
        or checkpoint["history"][-1].get("global_epoch") != SOURCE_EPOCH
    ):
        raise ValueError("Reviewed local60 / cumulative180 source checkpoint required")
    inherited = {k: v for k, v in encoder.state_dict().items() if not k.startswith("memory.")}
    _strict_tensors(inherited, checkpoint.get("encoder"), "encoder")
    _strict_tensors(query.state_dict(), checkpoint.get("query"), "query")
    source_opt = checkpoint.get("optimizer")
    if (
        not isinstance(source_opt, Mapping)
        or set(source_opt) != {"state", "param_groups"}
        or not isinstance(source_opt["state"], Mapping)
        or not isinstance(source_opt["param_groups"], list)
        or len(source_opt["param_groups"]) != 1
    ):
        raise ValueError("Complete original single-group Adam source required")
    original_group = source_opt["param_groups"][0]
    indices = list(range(len(groups["inherited"])))
    if original_group.get("params") != indices or set(source_opt["state"]) != set(indices):
        raise ValueError("Original Adam numeric IDs/order/support differ")
    expected = {
        "lr": INHERITED_LR,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": WEIGHT_DECAY,
        "amsgrad": False,
        "maximize": False,
    }
    if any(original_group.get(k) != v for k, v in expected.items()):
        raise ValueError("Original Adam hyperparameters differ")
    for index, (_, parameter) in enumerate(groups["inherited"]):
        _moment(source_opt["state"][index], parameter, SOURCE_UPDATES)
    if not isinstance(checkpoint.get("rng"), Mapping) or not checkpoint["rng"]:
        raise ValueError("Original RNG state required")
    memory_before = {
        k: v.detach().clone() for k, v in encoder.state_dict().items() if k.startswith("memory.")
    }
    encoder.load_state_dict(dict(checkpoint["encoder"]) | memory_before, strict=True)
    query.load_state_dict(checkpoint["query"], strict=True)
    if inherited_signature(encoder, query) != checkpoint.get("signature"):
        raise ValueError("Inherited source signature differs after strict load")
    # Destination state IDs use exactly the verified inherited order first.
    merged = optimizer.state_dict()
    merged["state"] = copy.deepcopy(source_opt["state"])
    merged["param_groups"][0] = copy.deepcopy(original_group)
    optimizer.load_state_dict(merged)
    validate_optimizer(optimizer, encoder, query, 0, 0)
    restore_rng(checkpoint["rng"])
    if any(not torch.equal(encoder.state_dict()[k], v) for k, v in memory_before.items()):
        raise ValueError("Source migration modified new memory initialization")
    return {
        "source_local_epoch": SOURCE_LOCAL_EPOCH,
        "source_epoch": SOURCE_EPOCH,
        "source_updates": SOURCE_UPDATES,
        "signature": checkpoint["signature"],
        "inherited_signature": inherited_signature(encoder, query),
        "model_signature": parameter_signature(encoder, query),
        "complete_inherited_adam_restored": True,
        "optimizer_reset": False,
        "memory_adam_initial_state": "empty",
        "rng_restored": True,
        "parameter_names": {k: [name for name, _ in values] for k, values in groups.items()},
        "learning_rates": {
            k: INHERITED_LR if k == "inherited" else MEMORY_LR
            for k, values in groups.items()
            if values
        },
        "new_parameters": sum(p.numel() for _, p in groups["memory"]),
    }


def make_matched_models(seed, device, config=None):
    seed = original._seed(seed)
    return {arm: model.make_models(arm, seed, device, config) for arm in ARMS}


def training_terms(encoder, query, x, raw_closes, positions, statistics, arm):
    contract = model.model_contract(encoder)
    if arm not in ARMS or contract["arm"] != arm:
        raise ValueError("Declared training arm differs from encoder organization")
    _, prediction = model.predict(encoder, query, x, positions)
    target, mask = model.targets(x, positions, statistics)
    target = content.replace_price_targets(target, raw_closes, positions, statistics)
    result = hq.metrics(prediction, target, mask, statistics, smooth=True)
    return result | {"objective": result["primary"], "query": result["primary"]}


def _grad_norm(parameters):
    parts = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not parts:
        return 0.0
    return float(torch.stack(parts).norm(2).cpu())


def train_epoch(
    encoder,
    query,
    x,
    raw_closes,
    statistics,
    optimizer,
    *,
    arm,
    seed,
    epoch,
    device=None,
    batch_size=BATCH_SIZE,
    micro_size=MICRO_SIZE,
    on_step=None,
):
    contract = model.model_contract(encoder)
    if contract["arm"] != arm or arm not in ARMS:
        raise ValueError("Training arm differs from model")
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable")
    batch_size = original._positive_integer(batch_size, "Effective batch size")
    micro_size = original._positive_integer(micro_size, "Microbatch size")
    if micro_size > batch_size:
        raise ValueError("Microbatch cannot exceed effective batch")
    x, raw_closes = np.asarray(x), original._closes(raw_closes)
    original._statistics(statistics)
    order, positions = schedule(len(x), seed, epoch)
    requested = next(encoder.parameters()).device if device is None else torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    parameters = original._validate_training(encoder, query, x, raw_closes, optimizer, requested)
    validate_optimizer(
        optimizer, encoder, query, epoch - 1, (epoch - 1) * math.ceil(len(x) / batch_size)
    )
    groups = parameter_groups(encoder, query)
    encoder.train()
    query.train()
    totals, norms = {}, []
    group_norms = {name: [] for name in groups}
    original._sync(requested)
    started = time.perf_counter()
    for left in range(0, len(x), batch_size):
        size = min(batch_size, len(x) - left)
        optimizer.zero_grad(set_to_none=True)
        batch_totals = {}
        for start in range(left, left + size, micro_size):
            stop = min(start + micro_size, left + size)
            indices = order[start:stop]
            features = torch.as_tensor(x[indices], device=requested)
            closes = torch.as_tensor(raw_closes[indices], device=requested)
            ps = torch.as_tensor(positions[start:stop], device=requested)
            terms = training_terms(encoder, query, features, closes, ps, statistics, arm)
            values = terms["objective"].mean(1)
            if values.shape != (len(indices),) or not torch.isfinite(values).all():
                raise ValueError("Nonfinite or invalid training objective")
            for name, value in content.detached_summary(terms).items():
                addition = np.asarray(value, dtype=np.float64) * len(indices)
                batch_totals[name] = batch_totals.get(name, 0.0) + addition
            (values.sum() / size).backward()
        measured = {name: _grad_norm([p for _, p in values]) for name, values in groups.items()}
        norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM, error_if_nonfinite=True)
            .detach()
            .cpu()
        )
        optimizer.step()
        norms.append(norm)
        for name, value in measured.items():
            group_norms[name].append(value)
        for name, value in batch_totals.items():
            totals[name] = totals.get(name, 0.0) + value
        if on_step is not None:
            on_step(
                {
                    "event": "after_update",
                    "arm": arm,
                    "epoch": epoch,
                    "global_epoch": SOURCE_EPOCH + epoch,
                    "step": len(norms),
                    "examples": size,
                    "supervised_states": size * 5,
                    "query_state_reads": size * 5,
                    "head_passes": 1,
                    "lr": INHERITED_LR,
                    "memory_lr": MEMORY_LR if groups["memory"] else None,
                    "preclip_norm": norm,
                    "preclip_group_norms": measured,
                    "order_hash": array_hash(order[left : left + size]),
                    "prefix_hash": array_hash(positions[left : left + size]),
                }
            )
    original._sync(requested)
    components = {name: (value / len(x)).tolist() for name, value in totals.items()}
    return {
        "schema": SCHEMA,
        "arm": arm,
        "model_contract": contract,
        "epoch": epoch,
        "global_epoch": SOURCE_EPOCH + epoch,
        "head_passes": 1,
        "query_state_reads": len(x) * 5,
        "lr": INHERITED_LR,
        "memory_lr": MEMORY_LR if groups["memory"] else None,
        "steps": len(norms),
        "examples": len(x),
        "supervised_states": len(x) * 5,
        "batch_size": batch_size,
        "micro_size": micro_size,
        "seconds": time.perf_counter() - started,
        "loss": components["objective"],
        "components": components,
        "order_hash": array_hash(order),
        "prefix_hash": array_hash(positions),
        "position_counts": np.bincount(positions.ravel() - 1, minlength=128).tolist(),
        "preclip_mean": float(np.mean(norms)),
        "preclip_max": max(norms),
        "steps_clipped": sum(v > CLIP_NORM for v in norms),
        "preclip_groups": {
            name: {
                "mean": float(np.mean(values)),
                "max": max(values),
                "steps_above_one": sum(v > 1.0 for v in values),
            }
            for name, values in group_norms.items()
        },
        "clip_policy": "one global norm1 across inherited and memory; groups diagnostic only",
    }


@torch.no_grad()
def validation(meta, encoder, query, x, raw, arm, device, *, micro_size=MICRO_SIZE):
    encoder.eval()
    query.eval()
    totals, contexts = {}, 0
    for start in range(0, len(x), micro_size):
        features = torch.as_tensor(x[start : start + micro_size], device=device)
        closes = torch.as_tensor(raw[start : start + micro_size], device=device)
        for left in range(0, len(model.VAL_POSITIONS), 5):
            ps = torch.tensor(model.VAL_POSITIONS[left : left + 5], device=device).expand(
                len(features), -1
            )
            terms = training_terms(encoder, query, features, closes, ps, meta["statistics"], arm)
            weight = len(features) * ps.shape[1]
            for key, value in content.detached_summary(terms).items():
                totals[key] = totals.get(key, 0.0) + np.asarray(value) * weight
            contexts += weight
    return {
        "positions": list(model.VAL_POSITIONS),
        "parents": len(x),
        "contexts": contexts,
        "components": {key: (value / contexts).tolist() for key, value in totals.items()},
    }
