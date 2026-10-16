"""Matched Q-only training for observed-window attention; no runner or weights.

The sole arm variable is the encoder attention graph. Labels, query decoder,
uniform five-endpoint sampling, AdamW and update accounting remain paired.
"""

import time

import numpy as np
import torch

from . import history_query as hq
from . import multiscale_content as content
from . import multiscale_training as original
from . import observed_history as model

SCHEMA = "babel-observed-history-training-v1"
ARMS = model.ARMS
EPOCHS = original.EPOCHS
BATCH_SIZE = original.BATCH_SIZE
MICRO_SIZE = original.MICRO_SIZE
PEAK_LR = original.PEAK_LR
WEIGHT_DECAY = original.WEIGHT_DECAY
CLIP_NORM = original.CLIP_NORM
array_hash = original.array_hash
parameter_signature = original.parameter_signature
new_optimizer = original.new_optimizer
learning_rate = original.learning_rate
schedule = original.schedule


def make_matched_models(seed, device, config=None):
    """Independent, exactly equal starting tensors with different graph metadata."""
    seed = original._seed(seed)
    return {arm: model.make_models(arm, seed, device, config) for arm in ARMS}


def _require_arm(encoder, arm):
    contract = model.model_contract(encoder)
    if arm not in ARMS or contract["arm"] != arm:
        raise ValueError("Declared training arm differs from encoder attention semantics")
    return contract


def training_terms(encoder, query, x, raw_closes, positions, statistics, arm):
    """Identical original Q for both arms, using raw float64 price supervision.

    No Haar scales, extra structure loss or target/feature decoder bypass. The
    unchanged body/activity targets and masks come from observed features.
    Returns one scalar per parent and sampled endpoint, before parent averaging.
    """
    _require_arm(encoder, arm)
    _, pred = model.predict(encoder, query, x, positions)
    target, mask = model.targets(x, positions, statistics)
    target = content.replace_price_targets(target, raw_closes, positions, statistics)
    result = hq.metrics(pred, target, mask, statistics, smooth=True)
    return result | {"objective": result["primary"], "query": result["primary"]}


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
    """Exact effective-batch accumulation, including the actual-sized last batch.

    on_step runs AFTER the optimizer update. Checkpoint transactions, restart
    policy, source identities, CUDA budgets and journal recovery belong to the
    caller; this component never loads historical weights or writes a file.
    """
    contract = _require_arm(encoder, arm)
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable")
    batch_size = original._positive_integer(batch_size, "Effective batch size")
    micro_size = original._positive_integer(micro_size, "Microbatch size")
    if micro_size > batch_size:
        raise ValueError("Microbatch cannot exceed effective batch")
    x, raw_closes = np.asarray(x), original._closes(raw_closes)
    original._statistics(statistics)
    order, positions = schedule(len(x), seed, epoch)
    lr = learning_rate(epoch)
    actual_device = next(encoder.parameters()).device
    requested = actual_device if device is None else torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    parameters = original._validate_training(encoder, query, x, raw_closes, optimizer, requested)
    for group in optimizer.param_groups:
        group["lr"] = lr
    encoder.train()
    query.train()
    totals, norms = {}, []
    order_hash, prefix_hash = array_hash(order), array_hash(positions)
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
        norm = torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM, error_if_nonfinite=True)
        norm = float(norm.detach().cpu())
        optimizer.step()
        norms.append(norm)
        for name, value in batch_totals.items():
            totals[name] = totals.get(name, 0.0) + value
        if on_step is not None:
            on_step(
                {
                    "event": "after_update",
                    "arm": arm,
                    "attention_mode": encoder.attention_mode,
                    "epoch": int(epoch),
                    "step": len(norms),
                    "examples": size,
                    "supervised_states": size * 5,
                    "lr": lr,
                    "preclip_norm": norm,
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
        "epoch": int(epoch),
        "lr": lr,
        "steps": len(norms),
        "examples": len(x),
        "supervised_states": len(x) * 5,
        "batch_size": batch_size,
        "micro_size": micro_size,
        "seconds": time.perf_counter() - started,
        "loss": components["objective"],
        "components": components,
        "order_hash": order_hash,
        "prefix_hash": prefix_hash,
        "position_counts": np.bincount(positions.ravel() - 1, minlength=128).tolist(),
        "preclip_mean": float(np.mean(norms)),
        "preclip_max": max(norms),
        "steps_clipped": sum(norm > CLIP_NORM for norm in norms),
    }
