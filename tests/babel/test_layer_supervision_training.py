"""Synthetic forward/backward accounting only; optimizer.step is always a no-op."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import history_query as hq
from obson.babel import layer_supervision as model
from obson.babel import layer_supervision_training as training
from obson.babel import multiscale_content as content
from obson.babel import observed_history as source_model
from obson.babel import observed_history_training as source_training
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


def no_update_optimizer(pair):
    opt = training.new_optimizer(*pair)
    snapshots = []

    def capture():
        snapshots.append([p.grad.detach().clone() for net in pair for p in net.parameters()])

    opt.step = capture
    return opt, snapshots


def run_epoch(pair, x, closes, *, arm="control", micro=1, batch=2, epoch=1, on_step=None):
    opt, gradients = no_update_optimizer(pair)
    before = training.parameter_signature(*pair)
    report = training.train_epoch(
        *pair,
        x,
        closes,
        STATS,
        opt,
        arm=arm,
        seed=42,
        epoch=epoch,
        batch_size=batch,
        micro_size=micro,
        on_step=on_step,
    )
    assert training.parameter_signature(*pair) == before
    assert not opt.state
    return report, gradients


def test_matched_initialization_has_no_extra_parameters_and_matches_source_order():
    rng = torch.random.get_rng_state().clone()
    with patch.object(torch, "load", side_effect=AssertionError("No real weights")):
        pairs = training.make_matched_models(42, "cpu", CONFIG)
    assert torch.equal(rng, torch.random.get_rng_state())
    source = source_model.make_models("control", 42, "cpu", CONFIG)
    signatures = [training.parameter_signature(*pair) for pair in pairs.values()]
    assert signatures == [training.parameter_signature(*source)] * 2
    for a, b in zip(*pairs.values(), strict=True):
        assert list(a.state_dict()) == list(b.state_dict())
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            assert p.data_ptr() != q.data_ptr()
    assert all(pair[0].attention_mode == "causal" for pair in pairs.values())


def test_absolute_sampling_epoch_fixed_lr_and_gradual_redistribution():
    for epoch in (1, 10, 60):
        actual = training.schedule(129, 43, epoch)
        expected = coverage.schedule(129, 43, 120 + epoch, "D")
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        assert training.learning_rate(epoch) == source_training.learning_rate(120)
    assert training.alpha(0) == 0
    assert training.alpha(1) == pytest.approx(0.02, rel=1e-15)
    assert training.alpha(10) == training.alpha(60) == 0.2
    for value in (-1, True, 61, 2.5):
        with pytest.raises(ValueError):
            training.alpha(value)
    with pytest.raises(ValueError):
        training.schedule(3, 42, 0)
    pair = model.make_models("control", 42, "cpu", CONFIG)
    opt = training.new_optimizer(*pair)
    assert opt.defaults["lr"] == source_training.learning_rate(120)
    assert opt.defaults["betas"] == (0.9, 0.999)
    assert opt.defaults["eps"] == 1e-8 and opt.defaults["weight_decay"] == 0.01
    assert not opt.state


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("epoch", [0, 1, 10, 60])
def test_control_repeated_shared_read_matches_original_q_and_gradients(checkpointing, epoch):
    pair = model.make_models("control", 42, "cpu", CONFIG)
    old = source_model.make_models("control", 42, "cpu", CONFIG)
    pair[0].checkpointing = old[0].checkpointing = checkpointing
    x, close = synthetic(1)
    xx, cc, ps = torch.tensor(x), torch.tensor(close), torch.tensor([[1, 64, 128]])
    with patch.object(pair[1], "forward", wraps=pair[1].forward) as reads:
        terms = training.training_terms(*pair, xx, cc, ps, STATS, "control", epoch=epoch)
    assert reads.call_count == 2
    original = source_training.training_terms(*old, xx, cc, ps, STATS, "control")
    torch.testing.assert_close(terms["final_query"], original["objective"], atol=0, rtol=0)
    torch.testing.assert_close(terms["auxiliary_query"], terms["final_query"], atol=0, rtol=0)
    torch.testing.assert_close(terms["objective"], original["objective"], atol=1e-14, rtol=1e-14)
    terms["objective"].mean().backward()
    original["objective"].mean().backward()
    for a, b in zip(pair, old, strict=True):
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=2e-5)


def test_candidate_auxiliary_gradients_stop_at_block2_and_final_q_remains_separate():
    pair = model.make_models("layer2", 42, "cpu", CONFIG)
    x, close = synthetic(1)
    xx, cc, ps = torch.tensor(x), torch.tensor(close), torch.tensor([[48]])
    terms = training.training_terms(*pair, xx, cc, ps, STATS, "layer2", epoch=10)
    torch.testing.assert_close(
        terms["objective"],
        0.8 * terms["final_query"] + 0.2 * terms["auxiliary_query"],
        atol=0,
        rtol=0,
    )
    assert not torch.allclose(terms["final_query"], terms["auxiliary_query"])
    parameters = list(pair[0].named_parameters()) + [
        ("query." + name, p) for name, p in pair[1].named_parameters()
    ]
    aux = torch.autograd.grad(
        terms["auxiliary_query"].mean(),
        [p for _, p in parameters],
        allow_unused=True,
        retain_graph=True,
    )
    final = torch.autograd.grad(
        terms["final_query"].mean(),
        [p for _, p in parameters],
        retain_graph=True,
    )
    combined = torch.autograd.grad(terms["objective"].mean(), [p for _, p in parameters])
    for (name, _), a, f, c in zip(parameters, aux, final, combined, strict=True):
        if name.startswith(("layers.2.", "layers.3.")):
            assert a is None
            torch.testing.assert_close(c, 0.8 * f, atol=2e-7, rtol=2e-5)
        else:
            assert a is not None and torch.isfinite(a).all()
            torch.testing.assert_close(c, 0.8 * f + 0.2 * a, atol=2e-7, rtol=2e-5)
    assert sum(a.abs().sum().item() for a in aux if a is not None) > 0


@pytest.mark.parametrize("arm", model.ARMS)
def test_both_reads_receive_identical_raw_price_labels_and_masks(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    x, close = synthetic(1)
    xx, cc, ps = torch.tensor(x), torch.tensor(close), torch.tensor([[1, 64, 128]])
    captured = []
    metrics = hq.metrics

    def observe(pred, target, mask, stats, smooth=False):
        captured.append((target.detach().clone(), mask.clone(), smooth))
        return metrics(pred, target, mask, stats, smooth=smooth)

    with patch.object(hq, "metrics", side_effect=observe):
        training.training_terms(*pair, xx, cc, ps, STATS, arm, epoch=10)
    assert len(captured) == 2 and all(row[2] for row in captured)
    for target, mask, _ in captured:
        old, expected_mask = coverage.targets(xx, ps, STATS)
        torch.testing.assert_close(target[..., 1:], old[..., 1:].double(), atol=0, rtol=0)
        torch.testing.assert_close(
            target[..., 0], content.raw_price_targets(cc, ps, STATS), atol=0, rtol=0
        )
        assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize("arm", model.ARMS)
def test_microaccumulation_short_tail_and_receipts_match_whole_batches_without_updates(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    batched = copy.deepcopy(pair)
    x, close = synthetic(3)
    receipts = []
    a, ga = run_epoch(pair, x, close, arm=arm, micro=1, epoch=12, on_step=receipts.append)
    b, gb = run_epoch(batched, x, close, arm=arm, micro=2, epoch=12)
    assert [r["examples"] for r in receipts] == [2, 1]
    assert a["steps"] == b["steps"] == 2
    assert a["global_epoch"] == 132 and a["alpha"] == 0.2
    assert a["head_passes"] == 2 and a["query_state_reads"] == 30
    assert a["supervised_states"] == 15
    assert a["order_hash"] == b["order_hash"] and a["prefix_hash"] == b["prefix_hash"]
    assert sum(a["position_counts"]) == 15
    assert all(r["head_passes"] == 2 and r["global_epoch"] == 132 for r in receipts)
    for sa, sb in zip(ga, gb, strict=True):
        for pa, pb in zip(sa, sb, strict=True):
            torch.testing.assert_close(pa, pb, atol=2e-6, rtol=3e-5)
    for name in a["components"]:
        np.testing.assert_allclose(
            a["components"][name], b["components"][name], atol=2e-6, rtol=2e-5
        )
    json.dumps({"report": a, "receipts": receipts}, allow_nan=False)


@pytest.mark.parametrize("arm", model.ARMS)
def test_future_features_and_raw_prices_cannot_change_either_loss_or_gradient(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    x, close = synthetic(1)
    xx, cc, ps = torch.tensor(x), torch.tensor(close), torch.tensor([[48]])
    changed_x, changed_close = xx.clone(), cc.clone()
    changed_x[:, 175:] += 100
    changed_close[:, 175:] *= 3
    parameters = [p for net in pair for p in net.parameters()]
    a = training.training_terms(*pair, xx, cc, ps, STATS, arm, epoch=10)
    ga = torch.autograd.grad(a["objective"].mean(), parameters)
    b = training.training_terms(*pair, changed_x, changed_close, ps, STATS, arm, epoch=10)
    gb = torch.autograd.grad(b["objective"].mean(), parameters)
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    for p, q in zip(ga, gb, strict=True):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_invalid_epoch_arm_and_data_rejected_before_noop_optimizer_is_called():
    pair = model.make_models("control", 42, "cpu", CONFIG)
    x, close = synthetic(2)
    with pytest.raises(ValueError, match="semantics"):
        run_epoch(pair, x, close, arm="layer2")
    with pytest.raises(ValueError, match="fixed60"):
        run_epoch(pair, x, close, epoch=61)
    bad = x.copy()
    bad[-1, -1, -1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        run_epoch(pair, bad, close)
    with pytest.raises(ValueError, match="float64"):
        run_epoch(pair, x, close.astype(np.float32))


def test_synthetic_source_adam_mapping_restores_without_updates_or_new_parameters():
    source = source_model.make_models("control", 42, "cpu", CONFIG)
    source_opt = source_training.new_optimizer(*source)
    source_opt.param_groups[0]["lr"] = source_training.learning_rate(120)
    for index, parameter in enumerate(source_opt.param_groups[0]["params"]):
        # These are invented fixtures, not the result of a local training step.
        source_opt.state[parameter] = {
            "step": torch.tensor(4560.0),
            "exp_avg": torch.full_like(parameter, (index + 1) / 1000),
            "exp_avg_sq": torch.full_like(parameter, (index + 1) / 100),
        }
    saved = copy.deepcopy(source_opt.state_dict())
    for arm in model.ARMS:
        pair = model.make_models(arm, 42, "cpu", CONFIG)
        opt = training.new_optimizer(*pair)
        with patch.object(opt, "step", side_effect=AssertionError("No updates")):
            opt.load_state_dict(saved)
        assert opt.param_groups[0]["lr"] == training.learning_rate(1)
        for old, new in zip(
            source_opt.param_groups[0]["params"], opt.param_groups[0]["params"], strict=True
        ):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    source_opt.state[old][key], opt.state[new][key], atol=0, rtol=0
                )


def test_candidate_epoch_zero_has_original_final_q_loss_and_gradients():
    candidate = model.make_models("layer2", 42, "cpu", CONFIG)
    source = source_model.make_models("control", 42, "cpu", CONFIG)
    x, close = synthetic(1)
    xx, cc, ps = torch.tensor(x), torch.tensor(close), torch.tensor([[32]])
    new = training.training_terms(*candidate, xx, cc, ps, STATS, "layer2", epoch=0)
    old = source_training.training_terms(*source, xx, cc, ps, STATS, "control")
    torch.testing.assert_close(new["objective"], old["objective"], atol=0, rtol=0)
    new["objective"].mean().backward()
    old["objective"].mean().backward()
    for a, b in zip(candidate, source, strict=True):
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
