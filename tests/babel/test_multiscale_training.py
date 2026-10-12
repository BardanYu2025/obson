"""Synthetic-only training-core tests; no historical checkpoints or market weights."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from obson.babel import multiscale_content as content
from obson.babel import multiscale_training as training
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


def models():
    return training.make_matched_models(42, "cpu", CONFIG)


def scales(closes):
    return training.fit_train_scales(closes, STATS, split="train")["scales"]


def run_epoch(pair, x, closes, *, arm="control", micro=1, batch=2, optimizer=None, on_step=None):
    if optimizer is None:
        optimizer = training.new_optimizer(*pair)
    return training.train_epoch(
        *pair,
        x,
        closes,
        STATS,
        scales(closes),
        optimizer,
        arm=arm,
        seed=42,
        epoch=1,
        batch_size=batch,
        micro_size=micro,
        on_step=on_step,
    )


def test_scratch_initialization_no_load_same_buffers_independent_parameters_and_rng():
    rng = torch.random.get_rng_state().clone()
    with patch.object(torch, "load", side_effect=AssertionError("No source weights allowed")):
        pairs = models()
    assert torch.equal(rng, torch.random.get_rng_state())
    first, second = pairs.values()
    assert training.parameter_signature(*first) == training.parameter_signature(*second)
    for left, right in zip(first, second, strict=True):
        for p, q in zip(left.parameters(), right.parameters(), strict=True):
            assert p.data_ptr() != q.data_ptr()
            torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert first[0].mode == "rolling"
    assert first[1].state_mean.count_nonzero() == 0
    assert torch.all(first[1].state_scale == 1)
    other = training.make_matched_models(43, "cpu", CONFIG)
    assert training.parameter_signature(*first) != training.parameter_signature(*other["control"])


def test_sampler_exactly_inherits_d_and_lr_fixed_budget():
    for seed in (42, 43):
        for epoch in (1, 2, 120):
            actual = training.schedule(129, seed, epoch)
            expected = coverage.schedule(129, seed, epoch, "D")
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)
            assert np.all(np.diff(actual[1], axis=1) > 0)
    assert training.learning_rate(1) == pytest.approx(3e-5)
    assert training.learning_rate(10) == 3e-4
    assert training.learning_rate(120) == pytest.approx(1.5e-5)
    for value in (0, 121, True, 1.5):
        with pytest.raises(ValueError):
            training.learning_rate(value)


def test_streaming_fit_all128_paths_matches_independent_numpy_windows():
    _, closes = synthetic(3)
    result = training.fit_train_scales(
        closes, STATS, split="train", chunk_size=1, positions_per_chunk=7
    )
    other = training.fit_train_scales(
        closes, STATS, split="train", chunk_size=3, positions_per_chunk=128
    )
    # Independently build each physical observed path, no query target inversion.
    paths = np.asarray(
        [100 * np.log(row[p : p + 128] / row[p + 127]) for row in closes for p in range(128)]
    )
    coefficients = content.haar_numpy(paths)
    expected = [np.sqrt(np.square(coefficients[:, a:b]).mean()) for a, b in content.BAND_SLICES]
    np.testing.assert_allclose(result["unfloored_rms"], expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result["scales"], other["scales"], rtol=1e-12, atol=1e-13)
    assert result["contexts"] == 3 * 128
    assert result["counts"] == [3 * 128 * size for size in content.BAND_SIZES]
    assert result["training_rows"] == 3 and result["positions_per_row"] == 128
    assert result["raw_closes_sha256"] == other["raw_closes_sha256"]
    json.dumps(result, allow_nan=False)


def test_fit_rejects_wrong_split_dtype_invalid_raw_before_accumulation():
    _, close = synthetic(1)
    for split in ("val", "test", "cross_research", None):
        with pytest.raises(ValueError, match="training"):
            training.fit_train_scales(close, STATS, split=split)
    with pytest.raises(ValueError, match="float64"):
        training.fit_train_scales(close.astype(np.float32), STATS, split="train")
    for bad in (np.nan, 0.0, -2.0):
        changed = close.copy()
        changed[0, -1] = bad
        with pytest.raises(ValueError, match="finite and positive"):
            training.fit_train_scales(changed, STATS, split="train")
    flat = np.full((1, 255), 50.0, dtype=np.float64)
    result = training.fit_train_scales(flat, STATS, split="train")
    np.testing.assert_array_equal(result["unfloored_rms"], np.zeros(8))
    np.testing.assert_array_equal(result["scales"], np.full(8, 0.0001))


@pytest.mark.parametrize("arm", training.ARMS)
def test_both_objectives_reach_encoder_and_query_and_use_same_raw_prices(arm):
    pair = models()[arm]
    x, close = synthetic(2)
    xx, cc = torch.tensor(x), torch.tensor(close)
    ps = torch.tensor([[1, 31, 128], [2, 80, 127]])
    before_stats = copy.deepcopy(STATS)
    terms = training.training_terms(*pair, xx, cc, ps, STATS, scales(close), arm)
    assert terms["objective"].shape == (2, 3)
    assert terms["multiscale_bands"].shape == (2, 3, 8)
    terms["objective"].mean().backward()
    for model in pair:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    # Observe the real labels passed into the objective, with no model bypass.
    captured = {}
    original_loss = content.loss_rows

    def observe(pred, target, mask, *args):
        captured.update(target=target.clone(), mask=mask.clone())
        return original_loss(pred, target, mask, *args)

    with patch.object(content, "loss_rows", side_effect=observe):
        training.training_terms(*pair, xx, cc, ps, STATS, scales(close), arm)
    old, old_mask = coverage.targets(xx, ps, STATS)
    assert captured["target"].dtype == torch.float64
    torch.testing.assert_close(captured["target"][..., 1:], old[..., 1:].double(), atol=0, rtol=0)
    assert torch.equal(captured["mask"], old_mask)
    torch.testing.assert_close(
        captured["target"][..., 0], content.raw_price_targets(cc, ps, STATS), atol=0, rtol=0
    )
    assert not torch.allclose(captured["target"][..., 0], old[..., 0].double())
    assert before_stats == STATS


@pytest.mark.parametrize("arm", training.ARMS)
def test_micro_and_tail_equivalence_to_whole_effective_batch(arm):
    pair = models()[arm]
    other = copy.deepcopy(pair)
    reference = copy.deepcopy(pair)
    x, closes = synthetic(3)

    # SGD isolates the sum/divide/clip arithmetic; Adam update counts are checked
    # below. Near-zero gradients otherwise amplify irrelevant Adam roundoff.
    def sgd(models):
        return torch.optim.SGD([p for model in models for p in model.parameters()], lr=0.1)

    a = run_epoch(pair, x, closes, arm=arm, micro=1, optimizer=sgd(pair))
    b = run_epoch(other, x, closes, arm=arm, micro=2, optimizer=sgd(other))
    # Independent whole-batch reference explicitly averages the short tail,
    # catching an implementation that incorrectly divides BOTH micro variants
    # by configured batch_size rather than actual tail size.
    reference_opt = sgd(reference)
    for group in reference_opt.param_groups:
        group["lr"] = coverage.learning_rate(1, 120, 3e-4)
    order, positions = coverage.schedule(len(x), 42, 1, "D")
    for left in range(0, len(x), 2):
        ids = order[left : left + 2]
        reference_opt.zero_grad(set_to_none=True)
        terms = training.training_terms(
            *reference,
            torch.tensor(x[ids]),
            torch.tensor(closes[ids]),
            torch.tensor(positions[left : left + 2]),
            STATS,
            scales(closes),
            arm,
        )
        terms["objective"].mean().backward()
        torch.nn.utils.clip_grad_norm_([p for model in reference for p in model.parameters()], 1.0)
        reference_opt.step()
    for left, right in zip(pair, other, strict=True):
        for p, q in zip(left.parameters(), right.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    for left, right in zip(pair, reference, strict=True):
        for p, q in zip(left.parameters(), right.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)
    assert a["steps"] == b["steps"] == 2
    assert a["examples"] == 3 and a["supervised_states"] == 15
    assert a["order_hash"] == b["order_hash"] and a["prefix_hash"] == b["prefix_hash"]
    for name in a["components"]:
        np.testing.assert_allclose(
            a["components"][name], b["components"][name], atol=2e-6, rtol=2e-5
        )


def test_adam_accounting_matched_order_detached_receipts_and_fixed_buffers():
    pairs = models()
    x, closes = synthetic(3)
    reports = []
    for arm, pair in pairs.items():
        opt = training.new_optimizer(*pair)
        assert not opt.state
        assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["weight_decay"] == 0.01
        buffers = [{k: v.clone() for k, v in model.named_buffers()} for model in pair]
        receipts = []
        before = training.parameter_signature(*pair)
        with patch.object(torch, "load", side_effect=AssertionError("No checkpoint loading")):
            report = run_epoch(pair, x, closes, arm=arm, optimizer=opt, on_step=receipts.append)
        assert training.parameter_signature(*pair) != before
        assert [r["step"] for r in receipts] == [1, 2]
        assert [r["examples"] for r in receipts] == [2, 1]
        assert sum(r["supervised_states"] for r in receipts) == report["supervised_states"] == 15
        assert all(r["event"] == "after_update" for r in receipts)
        assert all(s["step"].item() == 2 for s in opt.state.values())
        assert all(g["lr"] == coverage.learning_rate(1, 120, 3e-4) for g in opt.param_groups)
        for model, saved in zip(pair, buffers, strict=True):
            for name, value in model.named_buffers():
                torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
        assert sum(report["position_counts"]) == 15
        assert len(report["components"]["multiscale_bands"]) == 8
        json.dumps({"report": report, "receipts": receipts}, allow_nan=False)
        reports.append((report, receipts))
    for field in ("order_hash", "prefix_hash", "position_counts", "steps", "examples", "lr"):
        assert reports[0][0][field] == reports[1][0][field]
    for left, right in zip(reports[0][1], reports[1][1], strict=True):
        assert left["order_hash"] == right["order_hash"]
        assert left["prefix_hash"] == right["prefix_hash"]


def test_future_values_cannot_change_endpoint_prediction_or_raw_target():
    encoder, query = models()["control"]
    encoder.eval()
    query.eval()
    x, close = synthetic(1)
    x, close = torch.tensor(x), torch.tensor(close)
    for endpoint in (1, 48, 127):
        positions = torch.tensor([[endpoint]])
        changed_x, changed_close = x.clone(), close.clone()
        changed_x[:, endpoint + 127 :] += 1000
        changed_close[:, endpoint + 127 :] *= 2
        with torch.no_grad():
            z, pred = coverage.predict(encoder, query, x, positions)
            zz, altered = coverage.predict(encoder, query, changed_x, positions)
        torch.testing.assert_close(z, zz, atol=0, rtol=0)
        torch.testing.assert_close(pred, altered, atol=0, rtol=0)
        old, mask = coverage.targets(x, positions, STATS)
        other, other_mask = coverage.targets(changed_x, positions, STATS)
        raw = content.replace_price_targets(old, close, positions, STATS)
        raw_other = content.replace_price_targets(other, changed_close, positions, STATS)
        torch.testing.assert_close(raw, raw_other, atol=0, rtol=0)
        assert torch.equal(mask, other_mask)
        assert query(z[:, 0], torch.tensor([20])).shape == (1, 1, 7)
        with pytest.raises(ValueError, match="historical"):
            query(z[:, 0], torch.tensor([0]))


def test_invalid_input_or_nonfinite_gradient_stops_before_optimizer_update():
    pair = models()["control"]
    x, close = synthetic(3)
    opt = training.new_optimizer(*pair)
    before = training.parameter_signature(*pair)
    bad = x.copy()
    bad[-1, -1, -1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        run_epoch(pair, bad, close, optimizer=opt)
    assert not opt.state and training.parameter_signature(*pair) == before
    # Finite forward loss but nonfinite backward must also fail before Adam.
    handle = next(pair[0].parameters()).register_hook(lambda grad: torch.full_like(grad, torch.inf))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            run_epoch(pair, x, close, optimizer=opt)
    finally:
        handle.remove()
    assert not opt.state and training.parameter_signature(*pair) == before


def test_callback_is_after_update_and_not_a_resume_transaction():
    pair = models()["control"]
    x, close = synthetic(1)
    opt = training.new_optimizer(*pair)
    before = training.parameter_signature(*pair)

    def stopped(receipt):
        assert receipt["step"] == 1 and receipt["event"] == "after_update"
        assert all(state["step"].item() == 1 for state in opt.state.values())
        raise RuntimeError("future runner must journal this committed optimizer update")

    with pytest.raises(RuntimeError, match="future runner"):
        run_epoch(pair, x, close, optimizer=opt, on_step=stopped)
    assert training.parameter_signature(*pair) != before


def test_default_batch_is_128_micro8_and_foreign_optimizer_is_rejected():
    pair = models()["control"]
    x, close = synthetic(1)
    bad = torch.optim.AdamW(pair[0].parameters())
    with pytest.raises(ValueError, match="exactly"):
        run_epoch(pair, x, close, optimizer=bad)
    report = training.train_epoch(
        *pair,
        x,
        close,
        STATS,
        scales(close),
        training.new_optimizer(*pair),
        arm="control",
        seed=42,
        epoch=1,
    )
    assert (report["batch_size"], report["micro_size"], report["steps"]) == (128, 8, 1)
