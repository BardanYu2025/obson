"""Independent arithmetic and synthetic neural-path tests; no real weights."""

import copy

import numpy as np
import pytest
import torch

from obson.babel import content_guard_training as t
from obson.babel import state_organization_training as old

CONFIG = {"width": 32, "heads": 4, "ff": 64, "layers": 4}
STATS = {
    "x_mean": [0.0] * 28,
    "x_scale": [1.0] * 28,
    "y_mean": [0.1] * 7,
    "y_scale": [0.8, 1.3, 0.7, 2.0, 1.1, 0.9, 1.7],
    "delta_scale": [0.1, 0.2, 0.4, 0.8],
}


@pytest.fixture(autouse=True)
def runtime():
    threads = torch.get_num_threads()
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fast)


def data(n=3):
    rng = np.random.default_rng(172)
    x = rng.normal(0, 0.01, (n, 255, 28)).astype(np.float32)
    x[..., 23:28] = 1
    x[..., 22] = 0.1
    x[0, :, 23:28] = 0  # some parents have genuinely missing activity groups
    close = 100 * np.exp(rng.normal(0, 0.002, (n, 255)).cumsum(1))
    return x, close


def restored():
    pair = t.make_models("control", 42, "cpu", CONFIG)
    opt = t.new_optimizer(*pair)
    for p in opt.param_groups[0]["params"]:
        opt.state[p] = {
            "step": torch.tensor(float(t.SOURCE_UPDATES)),
            "exp_avg": torch.zeros_like(p),
            "exp_avg_sq": torch.ones_like(p) * 0.01,
        }
    t.validate_optimizer(opt, *pair, 0, 0)
    return pair, opt


def tensors():
    rng = np.random.default_rng(1)
    p = torch.tensor(rng.normal(size=(3, 2, 127, 7)), dtype=torch.float64, requires_grad=True)
    y = torch.tensor(rng.normal(size=p.shape), dtype=torch.float64)
    m = torch.tensor(rng.random(p.shape) > 0.2)
    m[0, :, :, 3] = False
    m[:, 0, :16, 5] = False
    return p, y, m


def test_physical_rows_match_numpy_in_physical_units_with_missing_support():
    p, y, mask = tensors()
    actual, support = t.physical_rows(p, y, mask, STATS)
    scale = np.empty((127, 7))
    scale[:, 0] = np.sqrt(np.arange(1, 128)) * 0.1
    scale[:, 1:] = STATS["y_scale"][1:]
    error = ((p.detach().numpy() - y.numpy()) * scale) ** 2
    oracle, available = np.zeros((3, 2, 21)), np.zeros((3, 2, 21), bool)
    for b in range(3):
        for q in range(2):
            for band, (lo, hi) in enumerate(((1, 16), (17, 64), (65, 127))):
                for field in range(7):
                    valid = mask[b, q, lo - 1 : hi, field].numpy()
                    j = band * 7 + field
                    available[b, q, j] = valid.any()
                    oracle[b, q, j] = (
                        error[b, q, lo - 1 : hi, field][valid].mean() if valid.any() else 0
                    )
    np.testing.assert_allclose(actual.detach(), oracle, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(support, available)
    actual.sum().backward()
    assert torch.equal(p.grad[~mask], torch.zeros_like(p.grad[~mask]))


@pytest.mark.parametrize("arm", t.ARMS)
def test_scalar_and_gradients_equal_independent_full_batch_formula(arm):
    p, y, mask = tensors()
    risk, support = t.physical_rows(p, y, mask, STATS)
    q = p.square().mean((-1, -2))
    ref = {"baseline": np.linspace(0.1, 2, 21).tolist(), "q0": 0.7}
    weights = t.initial_multipliers(arm)
    if arm == "feedback":
        weights = np.linspace(0, 3, 21).tolist()
    actual = t.batch_objective(
        {"query": q, "group_rows": risk, "group_support": support}, ref, weights, arm
    )
    expected = q.mean()
    if arm != "control":
        for j in range(21):
            if support[..., j].any():
                expected = expected + 0.7 / 21 * weights[j] * (
                    risk[..., j][support[..., j]].mean() / ref["baseline"][j] - 1
                )
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, p, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, p)[0]
    torch.testing.assert_close(ga, ge, rtol=1e-13, atol=1e-13)


def test_microbatch_contributions_use_effective_support_not_micro_means():
    p, y, mask = tensors()
    risk, support = t.physical_rows(p, y, mask, STATS)
    q = p.square().mean((-1, -2))
    ref = {"baseline": [1.0] * 21, "q0": 0.7}
    weights = [1.0] * 21
    whole = t.batch_objective(
        {"query": q, "group_rows": risk, "group_support": support}, ref, weights, "fixed"
    )
    pieces = [
        t.batch_objective(
            {
                "query": q[i : i + 1],
                "group_rows": risk[i : i + 1],
                "group_support": support[i : i + 1],
            },
            ref,
            weights,
            "fixed",
            counts=support.sum((0, 1)),
            contexts=6,
        )
        for i in range(3)
    ]
    torch.testing.assert_close(sum(pieces), whole)
    a = torch.autograd.grad(sum(pieces), p, retain_graph=True)[0]
    b = torch.autograd.grad(whole, p)[0]
    torch.testing.assert_close(a, b)


def test_negative_lagrangian_is_valid_and_missing_groups_are_not_rewarded():
    risks = torch.zeros(1, 1, 21, requires_grad=True)
    support = torch.ones_like(risks, dtype=torch.bool)
    support[..., -1] = False
    q = torch.ones(1, 1, requires_grad=True) * 0.1
    loss = t.batch_objective(
        {"query": q, "group_rows": risks, "group_support": support},
        {"baseline": [1.0] * 21, "q0": 1.0},
        [1.0] * 21,
        "fixed",
    )
    assert loss.item() == pytest.approx(0.1 - 20 / 21)
    loss.backward()
    assert risks.grad[0, 0, -1] == 0


def test_feedback_responds_per_group_not_compensated_mean_and_has_bounded_projection():
    ratios = [0.2] * 21
    ratios[0], ratios[1] = 3.0, 1000.0
    result = t.update_multipliers("feedback", [1.0] * 21, ratios)
    assert result[0] == pytest.approx(1.2)
    assert result[1] == 10
    assert result[2] == pytest.approx(0.92)
    assert t.update_multipliers("feedback", [0.0] * 21, [0.2] * 21) == [0.0] * 21
    assert t.update_multipliers("control", [0.0] * 21, ratios) == [0.0] * 21
    assert t.update_multipliers("fixed", [1.0] * 21, ratios) == [1.0] * 21


@pytest.mark.parametrize("bad", [[-1.0] * 21, [11.0] * 21, [float("nan")] * 21, [1.0] * 20])
def test_invalid_feedback_state_rejected(bad):
    with pytest.raises(ValueError):
        t.validate_multipliers("feedback", bad)


def test_reference_is_exact_fixed_training_panel_and_not_fitted_again_at_final():
    pair, opt = restored()
    x, raw = data()
    source = t.parameter_signature(*pair)
    before = copy.deepcopy(opt.state_dict())
    pair[0].train()
    pair[1].eval()
    ref = t.fit_reference(*pair, x, raw, STATS, seed=42, device="cpu", micro_size=2)
    assert ref["source_signature"] == source
    assert ref["parents"] == 3 and ref["contexts"] == 15 and ref["calibration_epoch"] == 180
    assert pair[0].training and not pair[1].training
    assert len(set(ref["support"])) > 1
    final = t.evaluate_reference(*pair, x, raw, STATS, ref, device="cpu", micro_size=1)
    np.testing.assert_allclose(final["group_ratios"], 1, rtol=2e-6, atol=2e-6)
    assert t.parameter_signature(*pair) == source
    for k in before["state"]:
        for key in before["state"][k]:
            torch.testing.assert_close(
                before["state"][k][key], opt.state_dict()["state"][k][key], rtol=0, atol=0
            )
    corrupted = copy.deepcopy(ref)
    corrupted["baseline"][0] = 0
    with pytest.raises(ValueError, match="no invented scale floor"):
        t.validate_reference(corrupted, 42, 3)
    corrupted = copy.deepcopy(ref)
    corrupted["prefix_hash"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        t.validate_reference(corrupted, 42, 3)
    altered = x.copy()
    altered[:, :, 23:28] = 0
    with pytest.raises(ValueError, match="lacks|required|masks"):
        t.evaluate_reference(*pair, altered, raw, STATS, ref, device="cpu")


def test_control_matches_original_q_and_all_original_parameter_gradients():
    a, _ = restored()
    b, _ = restored()
    x, raw = data(2)
    ps = torch.tensor([[1, 28, 64, 99, 128]]).repeat(2, 1)
    args = (torch.as_tensor(x), torch.as_tensor(raw), ps, STATS)
    terms = t.training_terms(*a, *args, "control")
    original = old.training_terms(*b, *args, "plain")
    loss = t.batch_objective(terms, {}, [0.0] * 21, "control")
    torch.testing.assert_close(loss, original["objective"].mean(), rtol=0, atol=0)
    loss.backward()
    original["objective"].mean().backward()
    for na, nb in zip(a, b, strict=True):
        for pa, pb in zip(na.parameters(), nb.parameters(), strict=True):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-5, atol=2e-6)


def no_update_epoch(arm, micro):
    pair, opt = restored()
    x, raw = data(3)
    ref = t.fit_reference(*pair, x, raw, STATS, seed=42, device="cpu")
    gradients = []

    def capture():
        gradients.append([p.grad.clone() for net in pair for p in net.parameters()])

    opt.step = capture
    before = t.parameter_signature(*pair)
    report = t.train_epoch(
        *pair,
        x,
        raw,
        STATS,
        opt,
        arm=arm,
        seed=42,
        epoch=1,
        reference=ref,
        multipliers=t.initial_multipliers(arm),
        batch_size=2,
        micro_size=micro,
    )
    assert t.parameter_signature(*pair) == before
    return report, gradients


@pytest.mark.parametrize("arm", t.ARMS)
def test_real_training_path_micro_tail_missing_masks_and_controller(arm):
    a, ga = no_update_epoch(arm, 1)
    b, gb = no_update_epoch(arm, 2)
    assert a["steps"] == 2 and a["supervised_states"] == 15 and a["global_epoch"] == 181
    assert a["head_passes"] == 1 and a["query_state_reads"] == 15
    assert a["multipliers_next"] == t.update_multipliers(
        arm, a["multipliers_used"], a["group_ratios"]
    )
    assert len(set(a["group_support"])) > 1
    np.testing.assert_allclose(a["group_risks"], b["group_risks"], rtol=2e-5)
    assert a["loss"] == pytest.approx(b["loss"], rel=2e-6)
    for ba, bb in zip(ga, gb, strict=True):
        for pa, pb in zip(ba, bb, strict=True):
            torch.testing.assert_close(pa, pb, rtol=2e-4, atol=2e-5)


def test_reference_panel_differs_from_training_epoch_and_rejects_memory_architecture():
    o, p = t.calibration_schedule(8, 42)
    o2, p2 = t.schedule(8, 42, 1)
    assert not np.array_equal(p, p2)
    assert o.shape == o2.shape == (8,)
    pair = t.model.make_models("late_memory", 42, "cpu", CONFIG)
    x, raw = data(2)
    with pytest.raises(ValueError, match="plain"):
        t.fit_reference(*pair, x, raw, STATS, seed=42, device="cpu")


def test_physical_overflow_is_rejected_before_it_can_hide_behind_control_q():
    p, y, mask = tensors()
    p = p.detach().fill_(1e200).requires_grad_()
    with pytest.raises(ValueError, match="before any optimizer update"):
        t.physical_rows(p, y, mask, STATS)


def test_fixed_and_feedback_first_epoch_use_identical_gradients_and_adam_updates():
    x, raw = data(3)
    pairs = [restored(), restored()]
    reports = []
    for arm, (pair, optimizer) in zip(("fixed", "feedback"), pairs, strict=True):
        ref = t.fit_reference(*pair, x, raw, STATS, seed=42, device="cpu")
        reports.append(
            t.train_epoch(
                *pair,
                x,
                raw,
                STATS,
                optimizer,
                arm=arm,
                seed=42,
                epoch=1,
                reference=ref,
                multipliers=t.initial_multipliers(arm),
                batch_size=2,
                micro_size=1,
            )
        )
    assert reports[0]["loss"] == reports[1]["loss"]
    assert reports[0]["multipliers_used"] == reports[1]["multipliers_used"]
    assert reports[0]["multipliers_next"] != reports[1]["multipliers_next"]
    assert t.parameter_signature(*pairs[0][0]) == t.parameter_signature(*pairs[1][0])
    for ka, kb in zip(pairs[0][1].state.values(), pairs[1][1].state.values(), strict=True):
        for key in ka:
            torch.testing.assert_close(ka[key], kb[key], rtol=0, atol=0)
