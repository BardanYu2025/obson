"""F01/F03 matched scratch-training components; no experiment runner or weight loading.

The caller owns immutable data identities, CUDA execution, budgets, checkpoint
transactions and research selection. This module supplies one deterministic
epoch and CPU-only scale fitting. Synthetic callers may use smaller networks
and batches; the planned experiment uses effective batch128 and microbatch8.
"""

import copy
import hashlib
import io
import time
from collections.abc import Mapping

import numpy as np
import torch

from . import multiscale_content as content
from . import state_coverage as coverage

SCHEMA = "babel-multiscale-training-v1"
ARMS = content.ARMS
EPOCHS = 120
BATCH_SIZE = 128
MICRO_SIZE = 8
PEAK_LR = 3e-4
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("Seed must be a nonnegative integer")
    return int(value)


def array_hash(value):
    """Same NPY byte identity as the inherited sampling receipts."""
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def parameter_signature(encoder, query):
    """Bind every parameter AND frozen buffer without serializing a checkpoint."""
    digest = hashlib.sha256()
    for family, model in (("encoder", encoder), ("query", query)):
        for name, tensor in model.state_dict().items():
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{family}/{name}:{value.dtype}:{value.shape}\n".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def make_matched_models(seed, device, config=None):
    """Independent scratch model pairs with bit-identical starts for both arms.

    No source checkpoint, PCA teacher or fitted state scaler is loaded. The
    inherited scratch factory fixes the query's state mean/scale to zero/one.
    """
    pair = coverage.make_models("rolling", _seed(seed), device, config)
    return {ARMS[0]: pair, ARMS[1]: copy.deepcopy(pair)}


def new_optimizer(encoder, query):
    """Fresh AdamW; a future resume runner must restore its own bound state."""
    return torch.optim.AdamW(
        list(encoder.parameters()) + list(query.parameters()),
        lr=PEAK_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


def learning_rate(epoch):
    epoch = _positive_integer(epoch, "Epoch")
    if epoch > EPOCHS:
        raise ValueError("Epoch exceeds the fixed120-epoch schedule")
    return coverage.learning_rate(epoch, EPOCHS, PEAK_LR)


def schedule(n, seed, epoch):
    """Original D uniform five-endpoint schedule, paired across both objectives.

    ``positions[k]`` belongs to sampling slot k, whose parent row is order[k].
    It must not be indexed by order a second time.
    """
    n = _positive_integer(n, "Row count")
    learning_rate(epoch)
    return coverage.schedule(n, _seed(seed), int(epoch), "D")


def _closes(value):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != 255 or not len(value) or value.dtype != np.float64:
        raise ValueError("Original float64 nonempty N x255 close prices required")
    if not np.isfinite(value).all() or not (value > 0).all():
        raise ValueError("Close prices must be finite and positive")
    return value


def _statistics(statistics):
    if not isinstance(statistics, Mapping):
        raise ValueError("Fixed training statistics mapping required")
    for name, length in (("x_mean", 28), ("x_scale", 28), ("y_mean", 7), ("y_scale", 7)):
        values = np.asarray(statistics.get(name, []))
        if (
            values.shape != (length,)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Finite fixed training {name} required")
        if name.endswith("scale") and not (values > 0).all():
            raise ValueError(f"Positive fixed training {name} required")
    delta = np.asarray(statistics.get("delta_scale", []))
    if (
        delta.ndim != 1
        or not len(delta)
        or delta.dtype.kind not in "fiu"
        or not np.isfinite(delta).all()
        or not (delta > 0).all()
    ):
        raise ValueError("Positive fixed training delta_scale required")


def fit_train_scales(raw_closes, statistics, *, split, chunk_size=8, positions_per_chunk=16):
    """Streaming NumPy RMS over all128 endpoints of every supplied training row.

    Input validation rejects an explicit validation/research split. Only the
    future runner can certify that the supplied rows really match the locked
    train inventory (4789 in the planned experiment). No feature/scaler refit
    occurs. Peak path storage is bounded by chunk_size*positions_per_chunk*128.
    """
    if split != "train":
        raise ValueError("Band scales may only be fitted on locked training rows")
    raw_closes = _closes(raw_closes)
    _statistics(statistics)
    chunk_size = _positive_integer(chunk_size, "Chunk size")
    positions_per_chunk = _positive_integer(positions_per_chunk, "Position chunk")
    if positions_per_chunk > 128:
        raise ValueError("Position chunk cannot exceed128")
    accumulator = content.BandScaleAccumulator(statistics["delta_scale"][0])
    for left in range(0, len(raw_closes), chunk_size):
        block = raw_closes[left : left + chunk_size]
        # A view: no allocation of N*128 complete windows.
        windows = np.lib.stride_tricks.sliding_window_view(block, 128, axis=1)
        for start in range(0, 128, positions_per_chunk):
            observed = windows[:, start : start + positions_per_chunk]
            anchor = observed[..., -1:]
            difference = observed - anchor
            near = np.abs(difference) <= anchor * 0.5
            ratio = np.where(near, difference, 0.0) / anchor
            path = np.where(near, np.log1p(ratio), np.log(observed) - np.log(anchor)) * 100
            accumulator.update(path, split=split)
    result = accumulator.finalize()
    return result | {
        "training_rows": len(raw_closes),
        "positions_per_row": 128,
        "raw_closes_sha256": array_hash(raw_closes),
        "chunk_size": chunk_size,
        "positions_per_chunk": positions_per_chunk,
        "row_provenance": "caller must bind the actual training inventory",
    }


def training_terms(encoder, query, x, raw_closes, positions, statistics, scales, arm):
    """Both arms receive identical observed raw-price labels and old other fields."""
    _, pred = coverage.predict(encoder, query, x, positions)
    target, mask = coverage.targets(x, positions, statistics)
    target = content.replace_price_targets(target, raw_closes, positions, statistics)
    return content.loss_rows(pred, target, mask, statistics, scales, arm)


def _validate_training(encoder, query, x, closes, optimizer, device):
    if x.shape != (len(closes), 255, 28) or x.dtype != np.float32:
        raise ValueError("Matched normalized float32 N x255 x28 features required")
    # Validate every row before ANY update, without making one large boolean array.
    for start in range(0, len(x), BATCH_SIZE):
        if not np.isfinite(x[start : start + BATCH_SIZE]).all():
            raise ValueError("Nonfinite normalized training features")
    parameters = list(encoder.parameters()) + list(query.parameters())
    if not parameters or any(not p.requires_grad or p.dtype != torch.float32 for p in parameters):
        raise ValueError("All encoder/query parameters must be trainable float32")
    if any(p.device != device for p in parameters):
        raise ValueError("Parameters and requested training device differ")
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {
        id(p) for p in parameters
    }:
        raise ValueError("Optimizer must update exactly the encoder/query parameters")
    return parameters


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(
    encoder,
    query,
    x,
    raw_closes,
    statistics,
    scales,
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
    """Train one epoch with exact parent-row gradient accumulation, including tail.

    Every parent contributes the mean of five uniformly sampled full-context
    states. Microbatch sums divide by the ACTUAL effective batch size; a short
    tail remains one ordinary update. LR follows the inherited120-epoch recipe.
    Metrics detach before logging. ``on_step(receipt)`` runs AFTER each update;
    an exception there does not roll back parameters. This is not a transactional
    runner and does not provide resume, checkpoint or source-binding guarantees.
    """
    if arm not in ARMS:
        raise ValueError(f"Expected a declared arm from {ARMS}")
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable")
    batch_size = _positive_integer(batch_size, "Effective batch size")
    micro_size = _positive_integer(micro_size, "Microbatch size")
    if micro_size > batch_size:
        raise ValueError("Microbatch cannot exceed effective batch")
    x, raw_closes = np.asarray(x), _closes(raw_closes)
    _statistics(statistics)
    content.huber_coordinate_weights(scales)
    order, positions = schedule(len(x), seed, epoch)
    lr = learning_rate(epoch)
    actual_device = next(encoder.parameters()).device
    requested = actual_device if device is None else torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    parameters = _validate_training(encoder, query, x, raw_closes, optimizer, requested)
    for group in optimizer.param_groups:
        group["lr"] = lr
    encoder.train()
    query.train()
    totals, norms = {}, []
    order_hash, prefix_hash = array_hash(order), array_hash(positions)
    _sync(requested)
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
            terms = training_terms(encoder, query, features, closes, ps, statistics, scales, arm)
            values = terms["objective"].mean(1)
            if values.shape != (len(indices),) or not torch.isfinite(values).all():
                raise ValueError("Nonfinite or invalid training objective")
            summary = content.detached_summary(terms)
            for name, value in summary.items():
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
    _sync(requested)
    components = {name: (value / len(x)).tolist() for name, value in totals.items()}
    return {
        "schema": SCHEMA,
        "arm": arm,
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
