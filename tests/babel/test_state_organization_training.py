"""Synthetic tensors only: forwards/gradients and Adam bookkeeping, no weight updates."""

import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import observed_history as original_model
from obson.babel import observed_history_training as original_training
from obson.babel import state_coverage as coverage
from obson.babel import state_organization as model
from obson.babel import state_organization_training as training
from obson.babel.ae_extend import rng_state

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.0] * 7,
    "y_scale": [1.0] * 7,
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def cpu():
    threads = torch.get_num_threads()
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fast)


def data(n=3):
    rng = np.random.default_rng(632)
    x = rng.normal(0, 0.01, (n, 255, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    closes = 100 * np.exp(rng.normal(0, 0.002, (n, 255)).cumsum(1))
    return x, closes.astype(np.float64)


def synthetic_checkpoint(seed=42):
    encoder, query = original_model.make_models("control", seed, "cpu", CONFIG)
    opt = original_training.new_optimizer(encoder, query)
    opt.param_groups[0]["lr"] = training.INHERITED_LR
    for i, p in enumerate(opt.param_groups[0]["params"]):
        opt.state[p] = {
            "step": torch.tensor(float(training.SOURCE_UPDATES)),
            "exp_avg": torch.full_like(p, (i + 1) / 1000),
            "exp_avg_sq": torch.full_like(p, (i + 1) / 100),
        }
    return {
        "epoch": 60,
        "history": [{"global_epoch": 121 + i} for i in range(60)],
        "encoder": copy.deepcopy(encoder.state_dict()),
        "query": copy.deepcopy(query.state_dict()),
        "optimizer": copy.deepcopy(opt.state_dict()),
        "rng": rng_state(),
        "signature": training.parameter_signature(encoder, query),
    }


def restored(arm="plain"):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    opt = training.new_optimizer(*pair)
    ck = synthetic_checkpoint()
    receipt = training.restore_source(*pair, opt, ck)
    return pair, opt, receipt


@pytest.mark.parametrize("arm", model.ARMS)
def test_strict_adam_migration_keeps_old_values_order_and_new_empty(arm):
    pair = model.make_models(arm, 42, "cpu", CONFIG)
    opt = training.new_optimizer(*pair)
    memory_before = {
        k: v.clone() for k, v in pair[0].state_dict().items() if k.startswith("memory.")
    }
    ck = synthetic_checkpoint()
    snapshot = copy.deepcopy(ck)
    rng_before = ck["rng"]["torch"].clone()
    with patch.object(torch, "load", side_effect=AssertionError("No real weights")):
        receipt = training.restore_source(*pair, opt, ck)
    assert receipt["inherited_signature"] == ck["signature"]
    assert receipt["source_local_epoch"] == 60 and receipt["source_epoch"] == 180
    groups = training.parameter_groups(*pair)
    for i, (_, p) in enumerate(groups["inherited"]):
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opt.state[p][name], snapshot["optimizer"]["state"][i][name], rtol=0, atol=0
            )
            assert opt.state[p][name].data_ptr() != ck["optimizer"]["state"][i][name].data_ptr()
    assert all(p not in opt.state for _, p in groups["memory"])
    for k, v in memory_before.items():
        assert torch.equal(pair[0].state_dict()[k], v)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    training.validate_optimizer(opt, *pair, 0, 0)


def test_schedule_is_absolute181_to240_and_not_reused_source_schedule():
    for epoch in (1, 10, 60):
        actual = training.schedule(23, 43, epoch)
        expected = coverage.schedule(23, 43, 180 + epoch, "D")
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        assert training.learning_rate(epoch) == original_training.learning_rate(120)
    for invalid in (-1, True, 0, 61, 1.5):
        with pytest.raises(ValueError):
            training.schedule(3, 42, invalid)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_tensor",
        "extra_tensor",
        "wrong_dtype",
        "wrong_shape",
        "bad_signature",
        "wrong_step",
        "negative_moment",
        "missing_moment",
        "shuffled_ids",
        "bad_lr",
        "missing_rng",
        "wrong_epoch",
    ],
)
def test_malformed_source_cannot_silently_reset_or_partial_load(mutation):
    pair = model.make_models("staged_memory", 42, "cpu", CONFIG)
    opt = training.new_optimizer(*pair)
    ck = synthetic_checkpoint()
    key = next(iter(ck["encoder"]))
    if mutation == "missing_tensor":
        del ck["encoder"][key]
    elif mutation == "extra_tensor":
        ck["encoder"]["memory.unexpected"] = torch.zeros(1)
    elif mutation == "wrong_dtype":
        ck["encoder"][key] = ck["encoder"][key].double()
    elif mutation == "wrong_shape":
        ck["encoder"][key] = torch.zeros(1)
    elif mutation == "bad_signature":
        ck["signature"] = "changed"
    elif mutation == "wrong_step":
        ck["optimizer"]["state"][0]["step"] = torch.tensor(4560.0)
    elif mutation == "negative_moment":
        ck["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)
    elif mutation == "missing_moment":
        del ck["optimizer"]["state"][0]["exp_avg"]
    elif mutation == "shuffled_ids":
        ck["optimizer"]["param_groups"][0]["params"].reverse()
    elif mutation == "bad_lr":
        ck["optimizer"]["param_groups"][0]["lr"] = 3e-4
    elif mutation == "missing_rng":
        ck["rng"] = {}
    else:
        ck["epoch"] = 180
    with pytest.raises(ValueError):
        training.restore_source(*pair, opt, ck)
    assert not opt.state


@pytest.mark.parametrize("arm", model.ARMS)
def test_one_original_q_with_same_raw_targets_no_auxiliary_objective(arm):
    pair, opt, _ = restored(arm)
    original = original_model.make_models("control", 42, "cpu", CONFIG)
    ck = synthetic_checkpoint()
    original[0].load_state_dict(ck["encoder"])
    original[1].load_state_dict(ck["query"])
    x, close = data(1)
    x = torch.tensor(x)
    close = torch.tensor(close)
    ps = torch.tensor([[1, 64, 128]])
    before = training.training_terms(*pair, x, close, ps, STATS, arm)
    expected = original_training.training_terms(*original, x, close, ps, STATS, "control")
    assert set(before) == set(expected)
    for k in before:
        torch.testing.assert_close(before[k], expected[k], rtol=0, atol=0)
    before["objective"].mean().backward()
    expected["objective"].mean().backward()
    inherited = training.parameter_groups(*pair)["inherited"]
    old = list(original[0].named_parameters()) + list(original[1].named_parameters())
    for (_, p), (_, q) in zip(inherited, old, strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=2e-6)
    assert all("auxiliary" not in k for k in before)


def no_update_epoch(pair, opt, x, close, *, micro):
    gradients = []

    def capture():
        gradients.append(
            [
                p.grad.clone() if p.grad is not None else None
                for net in pair
                for p in net.parameters()
            ]
        )

    opt.step = capture
    before = training.parameter_signature(*pair)
    report = training.train_epoch(
        *pair,
        x,
        close,
        STATS,
        opt,
        arm=pair[0].arm,
        seed=42,
        epoch=1,
        batch_size=2,
        micro_size=micro,
    )
    assert training.parameter_signature(*pair) == before
    return report, gradients


@pytest.mark.parametrize("arm", model.ARMS)
def test_microaccumulation_tail_and_group_norms_without_weight_updates(arm):
    x, close = data(3)
    a, oa, _ = restored(arm)
    b, ob, _ = restored(arm)
    ra, ga = no_update_epoch(a, oa, x, close, micro=1)
    rb, gb = no_update_epoch(b, ob, x, close, micro=2)
    for aa, bb in zip(ga, gb, strict=True):
        for p, q in zip(aa, bb, strict=True):
            assert (p is None) == (q is None)
            if p is not None:
                torch.testing.assert_close(p, q, rtol=2e-4, atol=2e-5)
    assert ra["steps"] == 2 and ra["examples"] == 3 and ra["supervised_states"] == 15
    assert ra["query_state_reads"] == 15 and ra["head_passes"] == 1 and ra["global_epoch"] == 181
    assert ra["order_hash"] == rb["order_hash"] and ra["prefix_hash"] == rb["prefix_hash"]
    assert sum(ra["position_counts"]) == 15
    assert ra["preclip_groups"]["inherited"]["mean"] > 0
    assert (
        ra["preclip_groups"]["memory"]["mean"] > 0
        if arm != "plain"
        else ra["preclip_groups"]["memory"]["mean"] == 0
    )
    assert ra["loss"] == pytest.approx(rb["loss"], rel=2e-6)


@pytest.mark.parametrize("arm", model.ARMS)
def test_future_features_and_raw_close_cannot_change_loss_or_gradient(arm):
    pair, _, _ = restored(arm)
    x, close = data(1)
    x = torch.tensor(x)
    close = torch.tensor(close)
    ps = torch.tensor([[17]])
    a = training.training_terms(*pair, x, close, ps, STATS, arm)["objective"].sum()
    parameters = list(pair[0].parameters()) + list(pair[1].parameters())
    ga = torch.autograd.grad(a, parameters)
    xx = x.clone()
    cc = close.clone()
    xx[:, 144:] += 23
    cc[:, 144:] *= 100
    b = training.training_terms(*pair, xx, cc, ps, STATS, arm)["objective"].sum()
    gb = torch.autograd.grad(b, parameters)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for p, q in zip(ga, gb, strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_optimizer_resume_requires_group_steps_and_complete_support():
    pair, opt, _ = restored("staged_memory")
    groups = training.parameter_groups(*pair)
    for _, p in groups["inherited"]:
        opt.state[p]["step"].add_(38)
    for _, p in groups["memory"]:
        opt.state[p] = {
            "step": torch.tensor(38.0),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.ones_like(p),
        }
    training.validate_optimizer(opt, *pair, 1, 38)
    saved = copy.deepcopy(opt.state_dict())
    new = training.new_optimizer(*pair)
    new.load_state_dict(saved)
    training.validate_optimizer(new, *pair, 1, 38)
    p = groups["memory"][0][1]
    new.state[p]["step"].fill_(6878)
    with pytest.raises(ValueError, match="step"):
        training.validate_optimizer(new, *pair, 1, 38)
    new.state[p]["step"].fill_(38)
    del new.state[p]
    with pytest.raises(ValueError, match="support"):
        training.validate_optimizer(new, *pair, 1, 38)


def test_fresh_adam_rejected_before_first_update():
    pair = model.make_models("plain", 42, "cpu", CONFIG)
    opt = training.new_optimizer(*pair)
    x, close = data(1)
    opt.step = lambda: pytest.fail("Fresh optimizer must not update")
    with pytest.raises(ValueError, match="support"):
        training.train_epoch(*pair, x, close, STATS, opt, arm="plain", seed=42, epoch=1)
