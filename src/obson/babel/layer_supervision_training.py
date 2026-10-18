"""Matched Q allocation across encoder depths, with restored Adam continuation.

The final Q coefficient is redistributed, not supplemented: candidate uses
(1-alpha)*final_Q + alpha*block2_Q. Control reads final twice through the same
head and uses the same combination. Both retain a single final public state. Alpha is an objective coefficient,
not a gradient-contribution percentage; the two readouts can have different
loss magnitudes and gradient scales.
Source checkpoint/optimizer verification and remote-only execution are runner
responsibilities; this module never loads real weights or writes files.
"""

import time

import numpy as np
import torch

from . import history_query as hq
from . import layer_supervision as model
from . import multiscale_content as content
from . import multiscale_training as original
from . import state_coverage as coverage

SCHEMA = "babel-layer-supervision-training-v1"
ARMS = model.ARMS
SOURCE_EPOCH = 120
EPOCHS = 60
BATCH_SIZE = original.BATCH_SIZE
MICRO_SIZE = original.MICRO_SIZE
CONTINUE_LR = original.learning_rate(SOURCE_EPOCH)
PEAK_LR = CONTINUE_LR
MAX_ALPHA = 0.2
RAMP_EPOCHS = 10
HEAD_PASSES = 2
WEIGHT_DECAY = original.WEIGHT_DECAY
CLIP_NORM = original.CLIP_NORM
array_hash = original.array_hash
parameter_signature = original.parameter_signature


def _epoch(value, *, allow_zero=False):
    if (
        allow_zero
        and not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        and value == 0
    ):
        return 0
    value = original._positive_integer(value, "Additional epoch")
    if value > EPOCHS:
        raise ValueError("Additional epoch exceeds the fixed60-epoch budget")
    return value


def alpha(epoch):
    return MAX_ALPHA * min(_epoch(epoch, allow_zero=True) / RAMP_EPOCHS, 1.0)


def learning_rate(epoch):
    _epoch(epoch, allow_zero=True)
    return PEAK_LR


def schedule(n, seed, epoch):
    """Continue sampling at absolute epoch121..180, never replay epoch1..60."""
    n = original._positive_integer(n, "Row count")
    return coverage.schedule(n, original._seed(seed), SOURCE_EPOCH + _epoch(epoch), "D")


def new_optimizer(encoder, query):
    """Construct AdamW for strict restoration; not permission to start fresh.

    The runner must load and verify the original last120 Adam state before any
    real update. Parameter order and every setting except the fixed continuation
    LR are identical to the source optimizer.
    """
    optimizer = original.new_optimizer(encoder, query)
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR
    optimizer.defaults["lr"] = PEAK_LR
    return optimizer


def make_matched_models(seed, device, config=None):
    """Independent equal layouts; the runner restores both from the same source."""
    seed = original._seed(seed)
    return {arm: model.make_models(arm, seed, device, config) for arm in ARMS}


def _require_arm(encoder, arm):
    contract = model.model_contract(encoder)
    if arm not in ARMS or contract["arm"] != arm:
        raise ValueError("Declared training arm differs from encoder supervision semantics")
    return contract


def training_terms(encoder, query, x, raw_closes, positions, statistics, arm, *, epoch=0):
    """Two shared-head reads, identical targets and Q coefficients summing to one.

    Control's second read uses final again; candidate's uses block2. There is no
    additional head, scaler, raw-input bypass, EMA teacher or detached auxiliary
    route. The original component names and query/primary always describe FINAL
    readout; objective alone is the weighted training combination.
    """
    _require_arm(encoder, arm)
    weight = alpha(epoch)
    final, middle = model.read_state_levels(encoder, x, positions)
    auxiliary = final if arm == "control" else middle
    shape = (len(x), positions.shape[1], 127, 7)
    prediction = query(final.flatten(0, 1)).reshape(shape)
    auxiliary_prediction = query(auxiliary.flatten(0, 1)).reshape(shape)
    target, mask = model.targets(x, positions, statistics)
    target = content.replace_price_targets(target, raw_closes, positions, statistics)
    result = hq.metrics(prediction, target, mask, statistics, smooth=True)
    extra = hq.metrics(auxiliary_prediction, target, mask, statistics, smooth=True)
    objective = (1.0 - weight) * result["primary"] + weight * extra["primary"]
    return (
        result
        | {f"auxiliary_{name}": value for name, value in extra.items()}
        | {
            "objective": objective,
            "query": result["primary"],
            "final_query": result["primary"],
            "auxiliary_query": extra["primary"],
        }
    )


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
            terms = training_terms(
                encoder, query, features, closes, ps, statistics, arm, epoch=epoch
            )
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
                    "global_epoch": SOURCE_EPOCH + int(epoch),
                    "alpha": alpha(epoch),
                    "head_passes": HEAD_PASSES,
                    "epoch": int(epoch),
                    "step": len(norms),
                    "examples": size,
                    "supervised_states": size * 5,
                    "query_state_reads": HEAD_PASSES * size * 5,
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
        "global_epoch": SOURCE_EPOCH + int(epoch),
        "alpha": alpha(epoch),
        "head_passes": HEAD_PASSES,
        "query_state_reads": HEAD_PASSES * len(x) * 5,
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
