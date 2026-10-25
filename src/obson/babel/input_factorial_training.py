"""Matched scratch Q-only training for the structure-by-initialization factorial.

The arms change input grouping/stride. Labels, query decoder,
uniform five-endpoint sampling, AdamW and update accounting remain paired.
"""

import time

import numpy as np
import torch

from . import history_query as hq
from . import input_factorial as model
from . import multiscale_content as content
from . import multiscale_metrics as physical_metrics
from . import multiscale_training as original

SCHEMA = "babel-input-factorial-training-v1"
ARMS = model.ARMS
EPOCHS = original.EPOCHS
BATCH_SIZE = original.BATCH_SIZE
MICRO_SIZE = original.MICRO_SIZE
PEAK_LR = original.PEAK_LR
WEIGHT_DECAY = original.WEIGHT_DECAY
CLIP_NORM = original.CLIP_NORM
array_hash = original.array_hash
parameter_signature = original.parameter_signature
initialization_receipt = model.initialization_receipt
new_optimizer = original.new_optimizer
learning_rate = original.learning_rate
schedule = original.schedule


def make_matched_models(seed, device, config=None):
    """Independent scratch pairs with matched common tensors and input controls."""
    seed = original._seed(seed)
    return {arm: model.make_models(arm, seed, device, config) for arm in ARMS}


def _require_arm(encoder, arm):
    contract = model.model_contract(encoder)
    if arm not in ARMS or contract["arm"] != arm:
        raise ValueError("Declared training arm differs from encoder input semantics")
    return contract


def _prediction_and_terms(encoder, query, x, raw_closes, positions, statistics, arm):
    """Identical original Q for all four arms, using raw float64 price supervision.

    No Haar scales, extra structure loss or target/feature decoder bypass. The
    unchanged body/activity targets and masks come from observed features.
    Returns one scalar per parent and sampled endpoint, before parent averaging.
    """
    _require_arm(encoder, arm)
    _, pred = model.predict(encoder, query, x, positions)
    target, mask = model.targets(x, positions, statistics)
    target = content.replace_price_targets(target, raw_closes, positions, statistics)
    result = hq.metrics(pred, target, mask, statistics, smooth=True)
    return pred, result | {"objective": result["primary"], "query": result["primary"]}


def training_terms(encoder, query, x, raw_closes, positions, statistics, arm):
    return _prediction_and_terms(encoder, query, x, raw_closes, positions, statistics, arm)[1]


def physical_price_panel(prediction, raw_closes, positions, statistics):
    """Original near/mid/far MSE and equal-band composite; no new objective.

    Supports different fixed sampled positions for each parent. Conversion and
    scoring match multiscale_evaluate; age0's observed zero anchor is excluded.
    """
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if torch.is_tensor(positions):
        positions = positions.detach().cpu().numpy()
    if torch.is_tensor(raw_closes):
        raw_closes = raw_closes.detach().cpu().numpy()
    prediction, ps, raw_closes = (
        np.asarray(prediction),
        np.asarray(positions),
        original._closes(raw_closes),
    )
    original._statistics(statistics)
    if ps.ndim == 1:
        ps = np.broadcast_to(ps, (len(raw_closes), len(ps)))
    if ps.ndim != 2 or ps.shape[0] != len(raw_closes) or prediction.shape != (*ps.shape, 127, 7):
        raise ValueError("Matched parents, query positions and seven-field predictions required")
    historical = prediction[..., 0].astype(np.float64) * (
        np.sqrt(np.arange(1, 128, dtype=np.float64)) * statistics["delta_scale"][0]
    )
    paths = np.concatenate((historical[..., ::-1], np.zeros((*ps.shape, 1))), -1)
    truth = np.concatenate(
        [physical_metrics.raw_paths(raw_closes[i : i + 1], ps[i]) for i in range(len(raw_closes))],
        0,
    )
    return {
        name: float(value.mean())
        for name, value in physical_metrics.price_rows(paths, truth).items()
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


def gradient_diagnostic(encoder, query, x8, raw8, statistics, seed, device):
    """Four actual weighted-Q gradients on a fixed eight-parent train panel.

    The caller certifies these are the first eight locked training parents.
    We use the original epoch1 sampling slots once, including their permutation.
    autograd.grad never writes parameter .grad or optimizer state. No gradient
    fraction is inferred from scalar losses; component cosines and summed-Q
    gradients expose cancellations. This is a diagnostic, not a selection rule.
    """
    declared = model.model_contract(encoder)
    x8, raw8 = np.asarray(x8), original._closes(raw8)
    if (
        x8.shape != (8, 255, 28)
        or x8.dtype != np.float32
        or len(raw8) != 8
        or not np.isfinite(x8).all()
    ):
        raise ValueError(
            "Diagnostic requires exactly eight finite float32 training parents and float64 closes"
        )
    original._statistics(statistics)
    requested = torch.device(device)
    actual = next(encoder.parameters()).device
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    if requested != actual or next(query.parameters()).device != actual:
        raise ValueError("Diagnostic model and requested device differ")
    weights = {"path": 0.4, "change1": 0.2, "body": 0.1, "activity": 0.3}
    grouped = {
        "input": list(encoder.input.parameters()),
        "context_input": list(encoder.context_input.parameters()),
        "backbone_coordinates": list(encoder.layers.parameters())
        + list(encoder.coordinates.parameters()),
        "query": list(query.parameters()),
    }
    parameters = [p for values in grouped.values() for p in values]
    if (
        len({id(p) for p in parameters}) != len(parameters)
        or {id(p) for p in parameters}
        != {id(p) for network in (encoder, query) for p in network.parameters()}
        or not all(p.requires_grad for p in parameters)
    ):
        raise ValueError("Diagnostic groups must partition every trainable parameter exactly once")
    before_signature = parameter_signature(encoder, query)
    modes = [
        (module, module.training) for network in (encoder, query) for module in network.modules()
    ]
    order, positions = schedule(8, seed, 1)
    devices = [actual.index] if actual.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            encoder.train()
            query.train()
            xx = torch.as_tensor(x8[order], device=requested)
            cc = torch.as_tensor(raw8[order], device=requested)
            ps = torch.as_tensor(positions, device=requested)
            prediction, terms = _prediction_and_terms(
                encoder, query, xx, cc, ps, statistics, declared["arm"]
            )
            price_metrics = physical_price_panel(prediction, raw8[order], positions, statistics)
            _, target_mask = model.targets(xx, ps, statistics)
            support = {
                "channel_order": [
                    "path",
                    "body",
                    "activity0",
                    "activity1",
                    "activity2",
                    "activity3",
                    "activity4",
                ],
                "valid_age_counts": target_mask.sum(dim=(0, 1, 2)).detach().cpu().tolist(),
                "possible_per_channel": int(
                    target_mask.shape[0] * target_mask.shape[1] * target_mask.shape[2]
                ),
                "band_valid_age_counts": {
                    name: target_mask[:, :, lo - 1 : hi].sum(dim=(0, 1, 2)).detach().cpu().tolist()
                    for name, (lo, hi) in physical_metrics.BANDS.items()
                },
            }
            losses = {name: weight * terms[name].mean() for name, weight in weights.items()}
            if not all(torch.isfinite(value) for value in losses.values()):
                raise ValueError("Nonfinite diagnostic loss")
            gradients = {
                name: tuple(
                    g.detach() for g in torch.autograd.grad(value, parameters, retain_graph=True)
                )
                for name, value in losses.items()
            }
            objective = tuple(
                g.detach() for g in torch.autograd.grad(terms["objective"].mean(), parameters)
            )
            summed = tuple(
                sum(gradients[name][i] for name in weights) for i in range(len(parameters))
            )
            for name, values in {
                **gradients,
                "objective": objective,
                "component_sum": summed,
            }.items():
                if not all(bool(torch.isfinite(value).all()) for value in values):
                    raise ValueError(f"Nonfinite actual diagnostic gradient: {name}")

            def dot(a, b):
                # FP64 reductions of actual raw gradients, not clipped gradients.
                return sum(
                    float((p.double() * q.double()).sum().cpu()) for p, q in zip(a, b, strict=True)
                )

            def norm(values):
                return float(np.sqrt(max(0.0, dot(values, values))))

            def summary(indices):
                local = {
                    name: tuple(values[i] for i in indices) for name, values in gradients.items()
                }
                q_gradient = tuple(objective[i] for i in indices)
                summed_gradient = tuple(summed[i] for i in indices)
                norms = {name: norm(values) for name, values in local.items()}
                objective_norm = norm(q_gradient)
                cosines = {}
                names = list(weights)
                for i, left in enumerate(names):
                    for right in names[i + 1 :]:
                        denominator = norms[left] * norms[right]
                        cosines[f"{left}__{right}"] = (
                            float(np.clip(dot(local[left], local[right]) / denominator, -1.0, 1.0))
                            if denominator > 0
                            else None
                        )
                difference = tuple(a - b for a, b in zip(summed_gradient, q_gradient, strict=True))
                max_abs = max(float(value.abs().max().cpu()) for value in difference)
                agrees = all(
                    torch.allclose(a, b, atol=2e-6, rtol=2e-4)
                    for a, b in zip(summed_gradient, q_gradient, strict=True)
                )
                return {
                    "component_gradient_norms": norms,
                    "component_pair_cosines": cosines,
                    "objective_gradient_norm": objective_norm,
                    "sum_check": {
                        "passed": agrees,
                        "max_abs": max_abs,
                        "relative_l2": norm(difference) / max(objective_norm, 1e-30),
                        "atol": 2e-6,
                        "rtol": 2e-4,
                    },
                }

            start, group_reports = 0, {}
            for name, values in grouped.items():
                group_reports[name] = summary(range(start, start + len(values)))
                start += len(values)
            global_report = summary(range(len(parameters)))
            qnorm = global_report["objective_gradient_norm"]
            clip_coefficient = min(1.0, CLIP_NORM / (qnorm + 1e-6))
            report = {
                "schema": "babel-input-factorial-gradient-diagnostic-v1",
                "arm": declared["arm"],
                "seed": int(seed),
                "updates": 0,
                "panel": "first8_locked_train_parents_epoch1_uniform5_sampling_slots",
                "parents": 8,
                "supervised_states": 40,
                "order_hash": array_hash(order),
                "prefix_hash": array_hash(positions),
                "weights": weights,
                "weighted_component_losses": {
                    name: float(value.detach().cpu()) for name, value in losses.items()
                },
                "objective": float(terms["objective"].mean().detach().cpu()),
                "physical_price_mse": price_metrics,
                "physical_price_units": "squared_100_log_price_ratio; near1_16_mid17_64_far65_127; age0_excluded",
                "target_support": support,
                "groups": group_reports,
                "global": global_report,
                "global_clip": {
                    "max_norm": CLIP_NORM,
                    "coefficient": clip_coefficient,
                    "clipped": clip_coefficient < 1.0,
                    "applied": False,
                },
                "all_component_sums_match_objective": all(
                    v["sum_check"]["passed"] for v in (*group_reports.values(), global_report)
                ),
                "limits": "Fixed train panel only; raw gradients and cosines are not scalar loss shares, causal attribution, test selection or an added training update",
            }
    finally:
        for module, mode in modes:
            module.training = mode
    if parameter_signature(encoder, query) != before_signature:
        raise ValueError("Diagnostic modified model parameters or buffers")
    report["parameters_and_buffers_unchanged"] = True
    # The runner must save this numerical evidence before rejecting failed
    # checks. Returning a failed receipt does not authorize a training update.
    report["passed"] = report["all_component_sums_match_objective"]
    return report
