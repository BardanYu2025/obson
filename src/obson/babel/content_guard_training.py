"""F01/F05: train-only source-relative content risks and soft feedback.

The feedback controller is an empirical, bounded Lagrangian recipe. It is NOT
an enforceable per-sample, validation or research non-degradation guarantee.
Only the original plain encoder and query are used. Physical MSE fields are
kept separate across ages; source scales are frozen before any adaptation.
"""

import math
import time

import numpy as np
import torch

from . import history_query as hq
from . import multiscale_content as content
from . import multiscale_training as original
from . import state_coverage as coverage
from . import state_organization as model
from . import state_organization_training as prior

SCHEMA = "babel-content-guard-training-v1"
REFERENCE_SCHEMA = "babel-content-guard-reference-v1"
ARMS = ("control", "fixed", "feedback")
SOURCE_EPOCH = prior.SOURCE_EPOCH
SOURCE_LOCAL_EPOCH = prior.SOURCE_LOCAL_EPOCH
SOURCE_UPDATES = prior.SOURCE_UPDATES
EPOCHS = prior.EPOCHS
BATCH_SIZE = prior.BATCH_SIZE
MICRO_SIZE = prior.MICRO_SIZE
INHERITED_LR = prior.INHERITED_LR
PEAK_LR = INHERITED_LR
CLIP_NORM = prior.CLIP_NORM
WEIGHT_DECAY = prior.WEIGHT_DECAY
DUAL_RATE = 0.1
DUAL_CAP = 10.0
FIELDS = (
    "price",
    "body",
    "volume_vs_prior_ema32",
    "delta_log_volume",
    "oi_change_percent_asinh",
    "oi_per_volume_asinh",
    "turnover_asinh",
)
GROUP_NAMES = tuple(f"{band}/{field}" for band in ("near", "mid", "far") for field in FIELDS)
GROUP_COUNT = len(GROUP_NAMES)
array_hash = prior.array_hash
parameter_signature = prior.parameter_signature
inherited_signature = prior.inherited_signature
new_optimizer = prior.new_optimizer
restore_source = prior.restore_source
validate_optimizer = prior.validate_optimizer
parameter_groups = prior.parameter_groups
schedule = prior.schedule
learning_rate = prior.learning_rate


def _arm(arm):
    if arm not in ARMS:
        raise ValueError("Declared control/fixed/feedback objective required")
    return arm


def make_models(arm, seed, device, config=None):
    _arm(arm)
    return model.make_models("plain", seed, device, config)


def _plain(encoder):
    contract = model.model_contract(encoder)
    if contract["arm"] != "plain":
        raise ValueError("All content-guard arms require the identical plain architecture")
    return contract


def _vector(value, label, *, positive=False):
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (GROUP_COUNT,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must have 21 finite values")
    if positive and not (values > 0).all():
        raise ValueError(f"{label} must be strictly positive; no invented scale floor")
    return values


def initial_multipliers(arm):
    return [0.0 if _arm(arm) == "control" else 1.0] * GROUP_COUNT


def validate_multipliers(arm, values):
    values = _vector(values, "Multipliers")
    _arm(arm)
    if (values < 0).any() or (values > DUAL_CAP).any():
        raise ValueError("Multipliers outside fixed [0,10] bound")
    if arm != "feedback" and not np.array_equal(values, initial_multipliers(arm)):
        raise ValueError("Control/fixed multipliers cannot change")
    return values


def update_multipliers(arm, used, ratios):
    values = validate_multipliers(arm, used)
    ratios = _vector(ratios, "Online risk ratios")
    if (ratios < 0).any():
        raise ValueError("Risk ratios cannot be negative")
    if arm == "feedback":
        values = np.clip(values + DUAL_RATE * (ratios - 1.0), 0.0, DUAL_CAP)
    return values.tolist()


def calibration_schedule(n, seed):
    n = original._positive_integer(n, "Calibration parent count")
    return coverage.schedule(n, original._seed(seed), SOURCE_EPOCH, "D")


def group_support(mask):
    if mask.ndim != 4 or mask.shape[-2:] != (127, 7) or mask.dtype != torch.bool:
        raise ValueError("Boolean parent x context x127 x7 masks required")
    return torch.cat([mask[..., lo - 1 : hi, :].any(-2) for lo, hi in hq.BANDS], -1)


def physical_rows(prediction, target, mask, statistics):
    """21 per-context risks, equal eligible ages within each field/age band.

    An unavailable context has zero storage and FALSE support. It is never
    counted as a perfectly reconstructed observation in a mean or gradient.
    Price is in hq.physical units; each ratio is invariant to any fixed unit
    conversion applied to both risk and its frozen source baseline.
    """
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Matched predictions, targets and masks required")
    support = group_support(mask)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Finite predictions and targets required")
    error = (hq.physical(prediction, statistics) - hq.physical(target, statistics)).square()
    if not torch.isfinite(error).all():
        raise ValueError("Nonfinite physical squared risk before any optimizer update")
    rows = []
    for lo, hi in hq.BANDS:
        valid = mask[..., lo - 1 : hi, :]
        rows.append(
            torch.where(valid, error[..., lo - 1 : hi, :], 0).sum(-2) / valid.sum(-2).clamp_min(1)
        )
    result = torch.cat(rows, -1)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite physical risk reduction before any optimizer update")
    return result, support


def training_terms(
    encoder, query, x, raw_closes, positions, statistics, arm, reference=None, multipliers=None
):
    _arm(arm)
    _plain(encoder)
    _, prediction = model.predict(encoder, query, x, positions)
    target, mask = model.targets(x, positions, statistics)
    target = content.replace_price_targets(target, raw_closes, positions, statistics)
    terms = hq.metrics(prediction, target, mask, statistics, smooth=True)
    risks, supported = physical_rows(prediction, target, mask, statistics)
    # "objective" is deliberately absent: group means require complete effective
    # batch supports, so averaging microbatch objectives would be wrong.
    return terms | {"query": terms["primary"], "group_rows": risks, "group_support": supported}


def batch_objective(terms, reference, multipliers, arm, *, counts=None, contexts=None):
    """A complete batch scalar, or its additive microbatch contribution.

    Callers accumulating microbatches supply the SAME full effective-batch
    eligible-context counts and total context count to every contribution.
    Missing groups are excluded, without redistributing the fixed /21 factor.
    Negative Lagrangian values are allowed and never signal loss corruption.
    """
    weights = validate_multipliers(arm, multipliers)
    risks, support, q = terms["group_rows"], terms["group_support"], terms["query"]
    if risks.shape != support.shape or risks.shape[:-1] != q.shape:
        raise ValueError("Per-context Q and group risk shapes differ")
    contexts = q.numel() if contexts is None else int(contexts)
    if contexts < q.numel() or contexts <= 0:
        raise ValueError("Full context count smaller than this microbatch")
    value = q.sum() / contexts
    if arm == "control":
        return value
    baseline = risks.new_tensor(_vector(reference["baseline"], "Frozen baseline", positive=True))
    q0 = float(reference["q0"])
    if not math.isfinite(q0) or q0 <= 0:
        raise ValueError("Positive finite source Q0 required")
    counts = support.sum((0, 1)) if counts is None else torch.as_tensor(counts, device=risks.device)
    if (
        counts.shape != (GROUP_COUNT,)
        or not torch.isfinite(counts).all()
        or (counts < support.sum((0, 1))).any()
    ):
        raise ValueError("Invalid full effective-batch eligible counts")
    numerator = torch.where(support, risks / baseline - 1.0, 0).sum((0, 1))
    # All microbatch constants sum to -1 for each supported full-batch group.
    return (
        value
        + q0 / GROUP_COUNT * (risks.new_tensor(weights) * numerator / counts.clamp_min(1)).sum()
    )


def validate_reference(reference, seed, parents):
    order, positions = calibration_schedule(parents, seed)
    expected = {
        "schema": REFERENCE_SCHEMA,
        "seed": seed,
        "parents": parents,
        "contexts": parents * 5,
        "order_hash": array_hash(order),
        "prefix_hash": array_hash(positions),
        "group_names": list(GROUP_NAMES),
        "calibration_epoch": SOURCE_EPOCH,
    }
    if any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError("Frozen training reference identity differs")
    _vector(reference["baseline"], "Frozen baseline", positive=True)
    support = _vector(reference["support"], "Frozen support", positive=True)
    if (support > parents * 5).any() or not np.array_equal(support, support.astype(np.int64)):
        raise ValueError("Calibration support counts invalid")
    if not math.isfinite(float(reference["q0"])) or reference["q0"] <= 0:
        raise ValueError("Frozen Q0 must be positive finite")
    signature = reference.get("source_signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError("Frozen reference must bind source tensor signature")
    return reference


@torch.no_grad()
def _panel(encoder, query, x, raw, statistics, *, seed, device, micro_size):
    _plain(encoder)
    micro_size = original._positive_integer(micro_size, "Panel micro size")
    order, positions = calibration_schedule(len(x), seed)
    totals = np.zeros(GROUP_COUNT, dtype=np.float64)
    counts = np.zeros(GROUP_COUNT, dtype=np.int64)
    qsum = 0.0
    modes = encoder.training, query.training
    encoder.eval()
    query.eval()
    try:
        for left in range(0, len(x), micro_size):
            ids = order[left : left + micro_size]
            terms = training_terms(
                encoder,
                query,
                torch.as_tensor(x[ids], device=device),
                torch.as_tensor(raw[ids], device=device),
                torch.as_tensor(positions[left : left + micro_size], device=device),
                statistics,
                "control",
            )
            totals += terms["group_rows"].sum((0, 1)).cpu().numpy()
            counts += terms["group_support"].sum((0, 1)).cpu().numpy()
            qsum += float(terms["query"].sum().cpu())
    finally:
        encoder.train(modes[0])
        query.train(modes[1])
    if (counts <= 0).any():
        raise ValueError("Calibration panel lacks a required content group")
    risks = totals / counts
    if not np.isfinite(risks).all() or (risks < 0).any():
        raise ValueError("Nonfinite/negative panel risk")
    return {
        "risks": risks.tolist(),
        "support": counts.tolist(),
        "query": qsum / (len(x) * 5),
        "order_hash": array_hash(order),
        "prefix_hash": array_hash(positions),
        "parents": len(x),
        "contexts": len(x) * 5,
    }


def fit_reference(encoder, query, x, raw, statistics, *, seed, device, micro_size=MICRO_SIZE):
    panel = _panel(
        encoder, query, x, raw, statistics, seed=seed, device=device, micro_size=micro_size
    )
    value = {
        "schema": REFERENCE_SCHEMA,
        "seed": seed,
        "group_names": list(GROUP_NAMES),
        "calibration_epoch": SOURCE_EPOCH,
        "baseline": panel.pop("risks"),
        "q0": panel.pop("query"),
        "source_signature": parameter_signature(encoder, query),
    } | panel
    return validate_reference(value, seed, len(x))


def evaluate_reference(
    encoder, query, x, raw, statistics, reference, *, seed=None, device, micro_size=MICRO_SIZE
):
    seed = reference["seed"] if seed is None else seed
    validate_reference(reference, seed, len(x))
    panel = _panel(
        encoder, query, x, raw, statistics, seed=seed, device=device, micro_size=micro_size
    )
    if panel["support"] != reference["support"]:
        raise ValueError("Final fixed-panel masks differ from source calibration")
    ratios = np.asarray(panel["risks"]) / np.asarray(reference["baseline"])
    return panel | {
        "group_ratios": ratios.tolist(),
        "within_source_mean": (ratios <= 1).tolist(),
        "all_within_source_mean": bool((ratios <= 1).all()),
        "scope": "Final frozen model on fixed TRAIN calibration panel only; exact point ratios, not population or per-example non-degradation",
    }


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
    reference,
    multipliers,
    device=None,
    batch_size=BATCH_SIZE,
    micro_size=MICRO_SIZE,
    on_step=None,
):
    _plain(encoder)
    weights = validate_multipliers(arm, multipliers).tolist()
    validate_reference(reference, seed, len(x))
    batch_size = original._positive_integer(batch_size, "Effective batch")
    micro_size = original._positive_integer(micro_size, "Microbatch")
    if micro_size > batch_size or (on_step is not None and not callable(on_step)):
        raise ValueError("Invalid microbatch or callback")
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
    encoder.train()
    query.train()
    original._sync(requested)
    started = time.perf_counter()
    sums, supports = np.zeros(GROUP_COUNT), np.zeros(GROUP_COUNT, dtype=np.int64)
    totals, norms, objectives = {}, [], []
    for left in range(0, len(x), batch_size):
        size = min(batch_size, len(x) - left)
        ids = order[left : left + size]
        # Label-only pass, never a second encoder forward. The denominator is
        # fixed across all microbatches, including missing masks and tail batch.
        with torch.no_grad():
            _, mask = model.targets(
                torch.as_tensor(x[ids], device=requested),
                torch.as_tensor(positions[left : left + size], device=requested),
                statistics,
            )
            full_counts = group_support(mask).sum((0, 1))
        optimizer.zero_grad(set_to_none=True)
        objective_value = 0.0
        for start in range(left, left + size, micro_size):
            stop = min(start + micro_size, left + size)
            ix = order[start:stop]
            terms = training_terms(
                encoder,
                query,
                torch.as_tensor(x[ix], device=requested),
                torch.as_tensor(raw_closes[ix], device=requested),
                torch.as_tensor(positions[start:stop], device=requested),
                statistics,
                arm,
            )
            loss = batch_objective(
                terms, reference, weights, arm, counts=full_counts, contexts=size * 5
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite content guard objective")
            loss.backward()
            objective_value += float(loss.detach().cpu())
            sums += terms["group_rows"].detach().sum((0, 1)).cpu().numpy()
            supports += terms["group_support"].sum((0, 1)).cpu().numpy()
            for key, value in terms.items():
                if key not in ("group_rows", "group_support"):
                    totals[key] = totals.get(key, 0.0) + float(value.detach().sum().cpu())
        norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM, error_if_nonfinite=True)
            .detach()
            .cpu()
        )
        optimizer.step()
        norms.append(norm)
        objectives.append(objective_value * size)
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
                    "memory_lr": None,
                    "preclip_norm": norm,
                    "order_hash": array_hash(order[left : left + size]),
                    "prefix_hash": array_hash(positions[left : left + size]),
                    "multipliers_used": weights,
                    "group_support": full_counts.cpu().tolist(),
                }
            )
    original._sync(requested)
    if (supports <= 0).any():
        raise ValueError("Training epoch lacks a required content group")
    risks = sums / supports
    ratios = risks / np.asarray(reference["baseline"])
    following = update_multipliers(arm, weights, ratios)
    components = {key: value / (len(x) * 5) for key, value in totals.items()}
    components["objective"] = sum(objectives) / len(x)
    return {
        "schema": SCHEMA,
        "arm": arm,
        "model_contract": _plain(encoder),
        "epoch": epoch,
        "global_epoch": SOURCE_EPOCH + epoch,
        "head_passes": 1,
        "query_state_reads": len(x) * 5,
        "lr": INHERITED_LR,
        "memory_lr": None,
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
        "steps_clipped": sum(value > CLIP_NORM for value in norms),
        "group_risks": risks.tolist(),
        "group_ratios": ratios.tolist(),
        "group_support": supports.tolist(),
        "multipliers_used": weights,
        "multipliers_next": following,
        "saturated_groups": [
            GROUP_NAMES[i] for i, value in enumerate(following) if value == DUAL_CAP
        ],
        "risk_scope": "Online TRAIN epoch mean at changing weights, not final-model risk",
        "clip_policy": "one global norm1, same inherited parameters in every arm",
    }


@torch.no_grad()
def validation(meta, encoder, query, x, raw, arm, device, *, epoch=0, micro_size=MICRO_SIZE):
    _arm(arm)
    _plain(encoder)
    # Original validation Q remains directly replayable. No controller update,
    # baseline fitting or epoch selection consumes validation measurements.
    return prior.validation(meta, encoder, query, x, raw, "plain", device, micro_size=micro_size)
