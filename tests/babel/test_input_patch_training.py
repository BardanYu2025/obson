"""Synthetic-only paired training; real-source training belongs on AutoDL."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import history_query as hq
from obson.babel import input_patch as model
from obson.babel import input_patch_training as training
from obson.babel import multiscale_content as content
from obson.babel import observed_history as old_model
from obson.babel import observed_history_training as old_training
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
    threads = torch.get_num_threads()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def synthetic(n=3):
    rng = np.random.default_rng(64)
    x = rng.normal(0, 0.01, (n, 255, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    close = 100 * np.exp(rng.normal(0, 0.002, (n, 255)).cumsum(1))
    return x, close.astype(np.float64)


def run_epoch(
    pair, x, closes, *, arm="bar", micro=1, batch=2, optimizer=None, on_step=None, epoch=1
):
    if optimizer is None:
        optimizer = training.new_optimizer(*pair)
    return training.train_epoch(
        *pair,
        x,
        closes,
        STATS,
        optimizer,
        arm=arm,
        seed=42,
        epoch=epoch,
        batch_size=batch,
        micro_size=micro,
        on_step=on_step,
    )


def test_scratch_initializations_match_without_loading_changing_rng_or_fitting_stats():
    rng = torch.random.get_rng_state().clone()
    with patch.object(torch, "load", side_effect=AssertionError("No source weights")):
        pairs = training.make_matched_models(42, "cpu", CONFIG)
        receipt = model.initialization_receipt(42, "cpu", CONFIG)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert receipt["passed"], receipt
    assert training.parameter_signature(*pairs["local4"]) == training.parameter_signature(
        *pairs["patch4"]
    )
    assert receipt["arms"]["patch4"]["signature"] == training.parameter_signature(*pairs["patch4"])
    old = coverage.make_models("rolling", 42, "cpu", CONFIG)
    assert training.parameter_signature(*pairs["bar"]) == training.parameter_signature(*old)
    for arm, pair in pairs.items():
        assert pair[1].state_mean.count_nonzero() == 0 and torch.all(pair[1].state_scale == 1)
        assert (
            receipt["arms"][arm]["parameter_bytes"] == 4 * receipt["arms"][arm]["parameter_count"]
        )
    for left, right in zip(pairs["local4"], pairs["patch4"], strict=True):
        for p, q in zip(left.parameters(), right.parameters(), strict=True):
            assert p.data_ptr() != q.data_ptr()
            torch.testing.assert_close(p, q, atol=0, rtol=0)
    with torch.no_grad():
        pairs["patch4"][1].output.weight[0, 0] += 0.1
    assert not model.compare_initialized_pairs(pairs)["passed"]


def test_schedule_learning_rate_and_optimizer_preserve_original_recipe():
    for epoch in (1, 10, 120):
        order, positions = training.schedule(129, 43, epoch)
        expected = coverage.schedule(129, 43, epoch, "D")
        np.testing.assert_array_equal(order, expected[0])
        np.testing.assert_array_equal(positions, expected[1])
        assert np.all(np.diff(positions, axis=1) > 0)
        assert training.learning_rate(epoch) == coverage.learning_rate(epoch, 120, 3e-4)
    pair = model.make_models("bar", 42, "cpu", CONFIG)
    opt = training.new_optimizer(*pair)
    assert not opt.state
    assert opt.defaults["betas"] == (0.9, 0.999)
    assert opt.defaults["eps"] == 1e-8 and opt.defaults["weight_decay"] == 0.01
    for epoch in (0, 121, True):
        with pytest.raises(ValueError):
            training.learning_rate(epoch)


def test_bar_original_q_and_all_neural_gradients_match_existing_training_entry():
    current = model.make_models("bar", 42, "cpu", CONFIG)
    prior = old_model.make_models("control", 42, "cpu", CONFIG)
    x, close = synthetic(2)
    xx, cc = torch.tensor(x), torch.tensor(close)
    ps = torch.tensor([[1, 31, 128], [2, 80, 127]])
    actual = training.training_terms(*current, xx, cc, ps, STATS, "bar")
    expected = old_training.training_terms(*prior, xx, cc, ps, STATS, "control")
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    actual["objective"].mean().backward()
    expected["objective"].mean().backward()
    for first, second in zip(current, prior, strict=True):
        for p, q in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)


@pytest.mark.parametrize("arm", training.ARMS)
def test_original_q_and_raw_price_labels_exactly_shared_and_both_networks_receive_gradient(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    x, close = synthetic(2)
    xx, cc = torch.tensor(x), torch.tensor(close)
    ps = torch.tensor([[1, 31, 128], [2, 80, 127]])
    before_stats = copy.deepcopy(STATS)
    captured = {}
    original_metrics = hq.metrics

    def observe(pred, target, mask, stats, smooth=False):
        captured.update(target=target.detach().clone(), mask=mask.clone(), smooth=smooth)
        return original_metrics(pred, target, mask, stats, smooth=smooth)

    with patch.object(hq, "metrics", side_effect=observe):
        terms = training.training_terms(*pair, xx, cc, ps, STATS, arm)
    assert terms["objective"].shape == (2, 3)
    assert captured["smooth"]
    torch.testing.assert_close(
        terms["objective"],
        (
            0.4 * terms["path"]
            + 0.2 * terms["change1"]
            + 0.1 * terms["body"]
            + 0.3 * terms["activity"]
        ),
        atol=0,
        rtol=0,
    )
    assert not any("multiscale" in k or "structure" in k for k in terms)
    old, mask = coverage.targets(xx, ps, STATS)
    assert captured["target"].dtype == torch.float64
    torch.testing.assert_close(captured["target"][..., 1:], old[..., 1:].double(), atol=0, rtol=0)
    torch.testing.assert_close(
        captured["target"][..., 0], content.raw_price_targets(cc, ps, STATS), atol=0, rtol=0
    )
    assert torch.equal(captured["mask"], mask)
    assert before_stats == STATS
    terms["objective"].mean().backward()
    for network in pair:
        grads = [p.grad for p in network.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0


@pytest.mark.parametrize("arm", training.ARMS)
def test_micro_accumulation_and_short_tail_match_independent_whole_batch(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    batched, reference = copy.deepcopy(pair), copy.deepcopy(pair)
    x, close = synthetic(3)

    def optimizer(models):
        return torch.optim.SGD([p for network in models for p in network.parameters()], lr=0.1)

    a = run_epoch(pair, x, close, arm=arm, optimizer=optimizer(pair), micro=1)
    b = run_epoch(batched, x, close, arm=arm, optimizer=optimizer(batched), micro=2)
    opt = optimizer(reference)
    for group in opt.param_groups:
        group["lr"] = coverage.learning_rate(1, 120, 3e-4)
    order, positions = coverage.schedule(len(x), 42, 1, "D")
    for left in range(0, len(x), 2):
        ids = order[left : left + 2]
        opt.zero_grad(set_to_none=True)
        terms = training.training_terms(
            *reference,
            torch.tensor(x[ids]),
            torch.tensor(close[ids]),
            torch.tensor(positions[left : left + 2]),
            STATS,
            arm,
        )
        terms["objective"].mean().backward()
        torch.nn.utils.clip_grad_norm_([p for net in reference for p in net.parameters()], 1.0)
        opt.step()
    for other in (batched, reference):
        for first, second in zip(pair, other, strict=True):
            for p, q in zip(first.parameters(), second.parameters(), strict=True):
                torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    assert a["steps"] == b["steps"] == 2
    assert a["supervised_states"] == b["supervised_states"] == 15
    assert a["order_hash"] == b["order_hash"] and a["prefix_hash"] == b["prefix_hash"]
    for name in a["components"]:
        np.testing.assert_allclose(
            a["components"][name], b["components"][name], atol=2e-6, rtol=2e-5
        )


def test_all_arms_have_matched_exposure_callbacks_and_fixed_buffers():
    pairs = training.make_matched_models(42, "cpu", CONFIG)
    x, close = synthetic(3)
    reports = []
    for arm, pair in pairs.items():
        before = [{k: v.clone() for k, v in net.named_buffers()} for net in pair]
        opt = training.new_optimizer(*pair)
        receipts = []
        report = run_epoch(pair, x, close, arm=arm, optimizer=opt, on_step=receipts.append)
        assert [r["examples"] for r in receipts] == [2, 1]
        assert [r["step"] for r in receipts] == [1, 2]
        assert all(r["attention_mode"] == pair[0].attention_mode for r in receipts)
        assert all(s["step"].item() == 2 for s in opt.state.values())
        assert sum(report["position_counts"]) == 15
        for network, saved in zip(pair, before, strict=True):
            for name, value in network.named_buffers():
                torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
        json.dumps({"report": report, "receipts": receipts}, allow_nan=False)
        reports.append((report, receipts))
    for field in ("order_hash", "prefix_hash", "position_counts", "steps", "examples", "lr"):
        assert reports[0][0][field] == reports[1][0][field] == reports[2][0][field]
    for a, b in zip(reports[0][1], reports[1][1], strict=True):
        assert a["order_hash"] == b["order_hash"] and a["prefix_hash"] == b["prefix_hash"]


@pytest.mark.parametrize("arm", training.ARMS)
def test_future_raw_and_features_do_not_change_loss_or_gradient(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    x, close = synthetic(1)
    xx, cc = torch.tensor(x), torch.tensor(close)
    ps = torch.tensor([[48]])
    altered_x, altered_close = xx.clone(), cc.clone()
    altered_x[:, 175:] += 100
    altered_close[:, 175:] *= 3
    terms = training.training_terms(*pair, xx, cc, ps, STATS, arm)
    changed = training.training_terms(*pair, altered_x, altered_close, ps, STATS, arm)
    for name in terms:
        torch.testing.assert_close(terms[name], changed[name], atol=0, rtol=0)


@pytest.mark.parametrize("arm", training.ARMS)
def test_epoch_boundary_restored_adam_matches_uninterrupted_synthetic_training(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    x, close = synthetic(2)
    optimizer = training.new_optimizer(*pair)
    run_epoch(pair, x, close, arm=arm, optimizer=optimizer)
    saved_networks = [copy.deepcopy(net.state_dict()) for net in pair]
    saved_opt = copy.deepcopy(optimizer.state_dict())
    restored = model.make_models(arm, 42, "cpu", CONFIG)
    for net, state in zip(restored, saved_networks, strict=True):
        net.load_state_dict(state, strict=True)
    restored_opt = training.new_optimizer(*restored)
    restored_opt.load_state_dict(saved_opt)
    a = run_epoch(pair, x, close, arm=arm, optimizer=optimizer, epoch=2)
    b = run_epoch(restored, x, close, arm=arm, optimizer=restored_opt, epoch=2)
    assert training.parameter_signature(*pair) == training.parameter_signature(*restored)
    assert a["order_hash"] == b["order_hash"] and a["prefix_hash"] == b["prefix_hash"]
    assert a["components"] == b["components"]


def test_invalid_arm_data_and_gradient_fail_before_update():
    pair = model.make_models("bar", 42, "cpu", CONFIG)
    x, close = synthetic(3)
    opt = training.new_optimizer(*pair)
    signature = training.parameter_signature(*pair)
    with pytest.raises(ValueError, match="semantics"):
        run_epoch(pair, x, close, arm="patch4", optimizer=opt)
    bad = x.copy()
    bad[-1, -1, -1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        run_epoch(pair, bad, close, optimizer=opt)
    with pytest.raises(ValueError, match="float64"):
        run_epoch(pair, x, close.astype(np.float32), optimizer=opt)
    assert not opt.state and training.parameter_signature(*pair) == signature
    handle = next(pair[0].parameters()).register_hook(lambda grad: torch.full_like(grad, torch.inf))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            run_epoch(pair, x, close, optimizer=opt)
    finally:
        handle.remove()
    assert not opt.state and training.parameter_signature(*pair) == signature


def test_callback_after_update_and_default_sizes():
    pair = model.make_models("bar", 42, "cpu", CONFIG)
    x, close = synthetic(1)
    optimizer = training.new_optimizer(*pair)

    def stop(receipt):
        assert receipt["event"] == "after_update"
        assert all(s["step"].item() == 1 for s in optimizer.state.values())
        raise RuntimeError("Caller must journal completed update")

    with pytest.raises(RuntimeError, match="journal"):
        run_epoch(pair, x, close, optimizer=optimizer, on_step=stop)
    report = training.train_epoch(*pair, x, close, STATS, optimizer, arm="bar", seed=42, epoch=2)
    assert (report["batch_size"], report["micro_size"], report["steps"]) == (128, 8, 1)
