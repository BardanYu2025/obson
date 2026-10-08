"""Synthetic causal/gradient contracts; no real weights or optimizer updates."""

from importlib import import_module
from unittest.mock import patch

import pytest
import torch
from test_history_query import fixture

from obson.babel import history_query as hq
from obson.babel import history_query_run as hqr
from obson.babel import history_structure as hs
from obson.babel import recovery_objectives as ro
from obson.babel import uniform_context as uc


@pytest.fixture
def case():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model, query, data, stats, local = fixture()
    model.local_head.requires_grad_(False)
    batch = hqr.batch_tensors(data, "cpu")
    prefixes = torch.tensor([[32, 64, 96, 128]] * len(batch["x"]))
    with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No updates")):
        yield model, query, batch, prefixes, stats, local
    torch.set_num_threads(old_threads)


def objective(case, arm):
    return ro.prefix_loss_rows(*case, arm)


def test_baseline_matches_original_objective_and_factor_values(case):
    model, query, batch, prefixes, stats, _ = case
    _, pred = hq.predict_states(model, query, batch["x"], prefixes)
    y, mask = hq.targets(batch, prefixes, stats)
    original = uc.loss_rows(pred, y, mask, stats, True)["objective"].mean(1)
    results = {arm: objective(case, arm) for arm in ro.ARMS}
    base = results["baseline"]
    torch.testing.assert_close(base["optimization_value"], original, rtol=1e-12, atol=1e-12)
    for arm, config in ro.ARMS.items():
        for key in ("query", "remote_structure", "local_diagnostic"):
            torch.testing.assert_close(results[arm][key], base[key], rtol=0, atol=0)
        expected = base["query"] + config.remote_structure * base["remote_structure"]
        expected = expected + config.local_to_encoder * 0.25 * base["local_diagnostic"]
        torch.testing.assert_close(results[arm]["optimization_value"], expected)


def test_local_route_reaches_encoder_but_not_query_or_frozen_reader(case):
    model, query, *_ = case
    assert not objective(case, "baseline")["local_diagnostic"].requires_grad
    loss = objective(case, "with_local")["local_diagnostic"].mean()
    parameters = [p for p in model.core.encoder.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, parameters + list(query.parameters()), allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients[: len(parameters)])
    assert all(g is None for g in gradients[len(parameters) :])
    assert all(p.grad is None and not p.requires_grad for p in model.local_head.parameters())


def test_four_arm_encoder_gradient_differences_are_the_declared_terms(case):
    model = case[0]
    parameters = [p for p in model.core.encoder.parameters() if p.requires_grad]

    def gradient(arm, key):
        loss = objective(case, arm)[key].mean()
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        return torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).flatten()
                for p, g in zip(parameters, grads, strict=True)
            ]
        )

    g = {arm: gradient(arm, "optimization_value") for arm in ro.ARMS}
    remote = gradient("baseline", "remote_structure")
    local = 0.25 * gradient("with_local", "local_diagnostic")
    assert remote.norm() > 0 and local.norm() > 0
    for a, b, expected in (
        ("baseline", "without_remote", remote),
        ("with_local", "baseline", local),
        ("without_remote_with_local", "without_remote", local),
        ("with_local", "without_remote_with_local", remote),
    ):
        torch.testing.assert_close(g[a] - g[b], expected, rtol=2e-4, atol=2e-6)


def test_remote_term_excludes_near_but_primary_query_does_not(case):
    stats = case[-2]
    pred = torch.full((2, 3, 127, 7), 0.4, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.ones_like(pred, dtype=torch.bool)
    remote = hs.metrics(pred, target, mask, stats, True)["primary"].sum()
    g = torch.autograd.grad(remote, pred)[0]
    assert torch.count_nonzero(g[..., :16, :]) == 0
    assert g[..., 16:, 0].abs().sum() > 0
    primary = hq.metrics(pred, target, mask, stats, True)["primary"].sum()
    assert torch.autograd.grad(primary, pred)[0][..., :16, 0].abs().sum() > 0


@pytest.mark.parametrize("arm", ro.ARMS)
def test_future_after_prefix_cannot_change_losses(case, arm):
    model, query, batch, _, stats, local = case
    ps = torch.full((len(batch["x"]), 1), 80)
    original = ro.prefix_loss_rows(model, query, batch, ps, stats, local, arm)
    changed = {k: v.clone() for k, v in batch.items()}
    changed["x"][:, 80:] += 100
    changed["y"][:, 80:] += 100
    after = ro.prefix_loss_rows(model, query, changed, ps, stats, local, arm)
    for key in original:
        torch.testing.assert_close(original[key], after[key], rtol=0, atol=0)


def test_unqualified_reader_and_unknown_arm_fail(case):
    with pytest.raises(ValueError, match="Unknown"):
        objective(case, "made_up")
    case[0].local_head.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        objective(case, "baseline")
    case[0].local_head.requires_grad_(False).train()
    with pytest.raises(ValueError, match="evaluation"):
        objective(case, "baseline")


@pytest.mark.parametrize(
    "module,mode,local_reaches_encoder",
    [
        ("bar_semantics", "control", True),
        ("bar_semantics", "additive", True),
        ("bar_semantics", "uniform", False),
        ("history_structure", "control", False),
        ("history_structure", "structure", False),
        ("history_structure", "consistent", False),
        ("causal_pool", "plain", False),
        ("depth_scaling", "d4", False),
        ("activity_weight", 0.3, False),
        ("shared_history", None, False),
        ("grounded_history", None, False),
        ("macro_history", None, False),
        ("linear_history", None, False),
    ],
)
def test_historical_local_loss_route_not_inferred_from_head_training(
    case, module, mode, local_reaches_encoder
):
    """V05: exercise the bound objective, including wrappers and reused aliases.

    A nonzero loss gradient at a trainable local reader does not imply that
    this loss reaches the encoder. The two early attached arms are positive
    controls: a blanket assertion that all historical local heads detach fails.
    Only synthetic forward/backward; no optimizer updates or real weights.
    """
    model, query, batch, prefixes, stats, local = case
    model.local_head.requires_grad_(True)
    # Three views of one synthetic endpoint suffice for these objective routes.
    batch = {k: v[:1].repeat(3, *([1] * (v.ndim - 1))) for k, v in batch.items()}
    prefixes = prefixes[:1].repeat(3, 1)
    seen = {}

    def capture(key):
        def hook(_module, _inputs, output):
            seen[key] = output

        return hook

    encoder_hook = model.core.encoder.register_forward_hook(capture("state"))
    reader_hook = model.local_head.register_forward_hook(capture("local"))
    try:
        args = (model, query, batch, prefixes, prefixes, torch.tensor([1]), stats, local)
        objective_module = import_module("obson.babel." + module)
        result = objective_module.losses(*args, *(() if mode is None else (mode,)))
        parts = result[0] if isinstance(result, tuple) else result
        loss = parts["optimization_value"].sum()
        local_gradient = torch.autograd.grad(loss, seen["local"], retain_graph=True)[0]
        assert torch.isfinite(local_gradient).all() and local_gradient.abs().sum() > 0
        state_gradient = torch.autograd.grad(
            seen["local"],
            seen["state"],
            grad_outputs=local_gradient,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if local_reaches_encoder:
            assert state_gradient is not None and state_gradient.abs().sum() > 0
        else:
            assert state_gradient is None
        # The detached diagnostic must not be confused with all encoder training.
        total_gradient = torch.autograd.grad(loss, seen["state"])[0]
        assert torch.isfinite(total_gradient).all() and total_gradient.abs().sum() > 0
    finally:
        encoder_hook.remove()
        reader_hook.remove()
