"""Synthetic paired updates and actual-Q gradient decomposition, no real weights."""

import copy
import json

import numpy as np
import pytest
import torch

from obson.babel import input_factorial as model
from obson.babel import input_factorial_training as training
from obson.babel import multiscale_content as content
from obson.babel import multiscale_evaluate as evaluation
from obson.babel import multiscale_training as original
from obson.babel import state_coverage as coverage

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def cpu_runtime():
    threads, fastpath = torch.get_num_threads(), torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def synthetic(n=3):
    rng = np.random.default_rng(64)
    x = rng.normal(0, 0.01, (n, 255, 28)).astype(np.float32)
    x[..., 23:28], x[..., 22] = 1, 0.1
    close = 100 * np.exp(rng.normal(0, 0.002, (n, 255)).cumsum(1))
    return x, close.astype(np.float64)


def run_epoch(pair, x, raw, *, arm, micro=1, optimizer=None, on_step=None):
    return training.train_epoch(
        *pair,
        x,
        raw,
        STATS,
        optimizer or training.new_optimizer(*pair),
        arm=arm,
        seed=42,
        epoch=1,
        batch_size=2,
        micro_size=micro,
        on_step=on_step,
    )


def test_fixed_schedule_and_single_original_adam_recipe():
    for epoch in (1, 10, 120):
        order, positions = training.schedule(129, 43, epoch)
        oo, pp = coverage.schedule(129, 43, epoch, "D")
        np.testing.assert_array_equal(order, oo)
        np.testing.assert_array_equal(positions, pp)
        assert training.learning_rate(epoch) == original.learning_rate(epoch)
    pair = model.make_models("dense_endpoint", 42, "cpu", CONFIG)
    optimizer = training.new_optimizer(*pair)
    assert len(optimizer.param_groups) == 1 and not optimizer.state
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8 and optimizer.defaults["weight_decay"] == 0.01
    assert (training.EPOCHS, training.BATCH_SIZE, training.MICRO_SIZE) == (120, 128, 8)


@pytest.mark.parametrize("arm", model.ARMS)
def test_original_weighted_q_and_every_parameter_receive_gradient(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    x, raw = synthetic(2)
    positions = torch.tensor([[1, 31, 128], [2, 80, 127]])
    terms = training.training_terms(
        *pair, torch.tensor(x), torch.tensor(raw), positions, STATS, arm
    )
    torch.testing.assert_close(
        terms["objective"],
        0.4 * terms["path"]
        + 0.2 * terms["change1"]
        + 0.1 * terms["body"]
        + 0.3 * terms["activity"],
        atol=0,
        rtol=0,
    )
    assert terms["objective"].shape == (2, 3)
    terms["objective"].mean().backward()
    for net in pair:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert pair[0].context_input.weight.grad[:, :84].abs().sum() > 0


@pytest.mark.parametrize("arm", model.ARMS)
def test_microbatch_short_tail_matches_whole_effective_batch(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    reference = copy.deepcopy(pair)
    x, raw = synthetic(3)

    def optimizer(models):
        return torch.optim.SGD([p for net in models for p in net.parameters()], lr=0.1)

    callbacks = []
    report = run_epoch(
        pair, x, raw, arm=arm, optimizer=optimizer(pair), micro=1, on_step=callbacks.append
    )
    expected = run_epoch(reference, x, raw, arm=arm, optimizer=optimizer(reference), micro=2)
    for left, right in zip(pair, reference, strict=True):
        for p, q in zip(left.parameters(), right.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    assert report["steps"] == expected["steps"] == 2
    assert [c["examples"] for c in callbacks] == [2, 1]
    assert report["supervised_states"] == 15
    assert (
        report["order_hash"] == expected["order_hash"]
        and report["prefix_hash"] == expected["prefix_hash"]
    )
    for key in report["components"]:
        np.testing.assert_allclose(
            report["components"][key], expected["components"][key], atol=2e-6, rtol=2e-5
        )


def test_all_four_arms_use_same_positions_and_parent_exposure():
    pairs, reports = training.make_matched_models(42, "cpu", CONFIG), []
    x, raw = synthetic(3)
    for arm, pair in pairs.items():
        reports.append(run_epoch(pair, x, raw, arm=arm))
    for name in ("order_hash", "prefix_hash", "position_counts", "steps", "examples", "lr"):
        assert all(r[name] == reports[0][name] for r in reports)


@pytest.mark.parametrize("arm", model.ARMS)
def test_future_feature_and_raw_prices_never_change_the_loss(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    x, raw = synthetic(1)
    xx, cc, ps = torch.tensor(x), torch.tensor(raw), torch.tensor([[48]])
    changed_x, changed_c = xx.clone(), cc.clone()
    changed_x[:, 175:] += 100
    changed_c[:, 175:] *= 3
    before = training.training_terms(*pair, xx, cc, ps, STATS, arm)
    after = training.training_terms(*pair, changed_x, changed_c, ps, STATS, arm)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)


@pytest.mark.parametrize("arm", ["dense_random", "hybrid_endpoint"])
def test_gradient_diagnostic_sums_actual_gradients_and_preserves_rng_grads_optimizer_and_modes(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    optimizer = training.new_optimizer(*pair)
    x, raw = synthetic(8)
    # Use a real synthetic update first so optimizer-state preservation is meaningful.
    run_epoch(pair, x[:2], raw[:2], arm=arm, optimizer=optimizer)
    pair[0].eval()
    pair[0].layers[0].train()
    before_modes = [m.training for net in pair for m in net.modules()]
    before_signature = training.parameter_signature(*pair)
    before_grad = [
        p.grad.clone() if p.grad is not None else None for net in pair for p in net.parameters()
    ]
    before_opt = copy.deepcopy(optimizer.state_dict())
    rng = torch.random.get_rng_state().clone()
    report = training.gradient_diagnostic(*pair, x, raw, STATS, 42, "cpu")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert training.parameter_signature(*pair) == before_signature
    assert [m.training for net in pair for m in net.modules()] == before_modes
    for p, grad in zip((p for net in pair for p in net.parameters()), before_grad, strict=True):
        assert (p.grad is None) == (grad is None)
        if grad is not None:
            torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)
    for key, state in before_opt["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                optimizer.state_dict()["state"][key][name], value, atol=0, rtol=0
            )
    assert report["updates"] == 0 and report["supervised_states"] == 40
    assert report["all_component_sums_match_objective"]
    assert set(report["groups"]) == {"input", "context_input", "backbone_coordinates", "query"}
    norm = report["global"]["objective_gradient_norm"]
    assert report["global_clip"]["coefficient"] == min(1.0, 1.0 / (norm + 1e-6))
    assert not report["global_clip"]["applied"]
    np.testing.assert_allclose(
        sum(report["weighted_component_losses"].values()), report["objective"], atol=1e-12
    )
    json.dumps(report, allow_nan=False)


def test_diagnostic_fixed_panel_is_identical_and_rejects_bad_inputs():
    pair = model.make_models("hybrid_endpoint", 43, "cpu", CONFIG)
    x, raw = synthetic(8)
    a = training.gradient_diagnostic(*pair, x, raw, STATS, 43, "cpu")
    b = training.gradient_diagnostic(*pair, x, raw, STATS, 43, "cpu")
    assert a == b
    with pytest.raises(ValueError, match="eight"):
        training.gradient_diagnostic(*pair, x[:7], raw[:7], STATS, 43, "cpu")
    with pytest.raises(ValueError, match="float64"):
        training.gradient_diagnostic(*pair, x, raw.astype(np.float32), STATS, 43, "cpu")


def test_physical_diagnostic_matches_existing_evaluator_for_per_parent_positions():
    pair = model.make_models("hybrid_endpoint", 42, "cpu", CONFIG)
    x, raw = synthetic(3)
    ps = np.asarray([[1, 50, 128], [2, 49, 127], [3, 51, 126]], dtype=np.int64)
    with torch.no_grad():
        _, predicted = model.predict(*pair, torch.tensor(x), torch.tensor(ps))
    report = training.physical_price_panel(predicted, raw, ps, STATS)
    expected = {name: [] for name in report}
    for i in range(3):
        targets, masks = model.targets(
            torch.tensor(x[i : i + 1]), torch.tensor(ps[i : i + 1]), STATS
        )
        targets = content.replace_price_targets(
            targets, torch.tensor(raw[i : i + 1]), torch.tensor(ps[i : i + 1]), STATS
        )
        scored = evaluation.score_rows(
            predicted[i : i + 1].numpy(),
            targets.numpy(),
            masks.numpy(),
            raw[i : i + 1],
            STATS,
            ps[i],
        )
        for name in report:
            expected[name].append(scored["errors"][f"price/{name}"].mean())
    for name in report:
        np.testing.assert_allclose(report[name], np.mean(expected[name]), atol=0, rtol=1e-15)
    shared = np.tile(ps[0], (3, 1))
    assert training.physical_price_panel(
        predicted, raw, ps[0], STATS
    ) == training.physical_price_panel(predicted, raw, shared, STATS)


def test_diagnostic_global_objective_norm_matches_independent_backward():
    pair = model.make_models("dense_endpoint", 42, "cpu", CONFIG)
    x, raw = synthetic(8)
    report = training.gradient_diagnostic(*pair, x, raw, STATS, 42, "cpu")
    order, ps = training.schedule(8, 42, 1)
    terms = training.training_terms(
        *pair,
        torch.tensor(x[order]),
        torch.tensor(raw[order]),
        torch.tensor(ps),
        STATS,
        "dense_endpoint",
    )
    terms["objective"].mean().backward()
    norm = np.sqrt(
        sum(float(p.grad.double().square().sum()) for net in pair for p in net.parameters())
    )
    np.testing.assert_allclose(report["global"]["objective_gradient_norm"], norm, rtol=1e-12)
    with torch.no_grad():
        _, prediction = model.predict(*pair, torch.tensor(x[order]), torch.tensor(ps))
    assert report["physical_price_mse"] == training.physical_price_panel(
        prediction, raw[order], ps, STATS
    )


def test_zero_gradient_norms_have_undefined_cosines_not_nan_or_fabricated_shares():
    pair = model.make_models("hybrid_endpoint", 42, "cpu", CONFIG)
    x, raw = np.zeros((8, 255, 28), dtype=np.float32), np.full((8, 255), 100.0, dtype=np.float64)
    x[..., 23:28] = 1
    with torch.no_grad():
        pair[1].output.weight.zero_()
        pair[1].output.bias.zero_()
    report = training.gradient_diagnostic(*pair, x, raw, STATS, 42, "cpu")
    assert report["objective"] == 0 and report["passed"]
    assert report["global"]["objective_gradient_norm"] == 0
    assert all(value is None for value in report["global"]["component_pair_cosines"].values())
    assert report["global_clip"]["coefficient"] == 1
    assert report["target_support"]["valid_age_counts"] == [40 * 127] * 7
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_finite_forward_with_corrupted_gradient_fails_closed(nonfinite):
    pair = model.make_models("dense_endpoint", 42, "cpu", CONFIG)
    x, raw = synthetic(8)
    pair[0].eval()
    before = training.parameter_signature(*pair)
    hook = pair[0].context_input.weight.register_hook(
        lambda value: torch.full_like(value, nonfinite)
    )
    try:
        with pytest.raises(ValueError, match="Nonfinite actual diagnostic gradient"):
            training.gradient_diagnostic(*pair, x, raw, STATS, 42, "cpu")
    finally:
        hook.remove()
    assert training.parameter_signature(*pair) == before
    assert not pair[0].training
